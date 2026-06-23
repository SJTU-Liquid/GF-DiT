# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/utils.py
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM-Omni distributed state.

It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model parallelism,
 you can skip the model parallel initialization and destruction steps.
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.collective_runtime import (
    LogicalGroupHandle,
    is_torch_process_group,
    make_logical_group,
    register_static_gfc_subgroup,
    runtime_v2_uses_gfc,
    shutdown_runtime_v2_collective_runtime,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.platforms import current_omni_platform

from .group_coordinator import (
    GroupCoordinator,
    LogicalGroupCoordinator,
    LogicalSequenceParallelGroupCoordinator,
    PipelineGroupCoordinator,
    SequenceParallelGroupCoordinator,
)

env_info = envs.PACKAGES_CHECKER.get_packages_info()

HAS_FLASH_ATTN = env_info["has_flash_attn"]

logger = init_logger(__name__)


_WORLD: GroupCoordinator | None = None
# get _TP&_EP from vllm.distributed.parallel_state
_SP: SequenceParallelGroupCoordinator | None = None
_PP: PipelineGroupCoordinator | None = None
_CFG: GroupCoordinator | None = None
_DP: GroupCoordinator | None = None
_FS: GroupCoordinator | None = None  # Fully Sharded (HSDP shard dimension)
_DIT: GroupCoordinator | None = None
_RUNTIME_V2_GROUP_SESSIONS: dict[str, dict[str, Any]] = {}


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: list[int], mask: list[bool]
) -> list[list[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (list[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (list[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: list[int], init=1) -> list[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: list[int], b: list[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert sum([x * y for x, y in zip(idx, stride[:-1])]) == index, (
            f"idx {index} with shape {shape} mismatch the return idx {idx}"
        )
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride) + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator:
    def __init__(
        self,
        tp: int,
        sp: int,
        pp: int,
        cfg: int,
        dp: int,
        fs: int = 1,
        order: str = "tp-sp-pp-cfg-dp",
        rank_offset: int = 0,
    ) -> None:
        self.tp = tp
        self.sp = sp
        self.pp = pp
        self.cfg = cfg
        self.dp = dp
        self.fs = fs
        self.rank_offset = rank_offset
        self.world_size = tp * sp * pp * cfg * dp
        self.ep = tp * sp * cfg * dp  # EP level exclude PP

        self.name_to_size = {
            "tp": self.tp,
            "sp": self.sp,
            "pp": self.pp,
            "cfg": self.cfg,
            "dp": self.dp,
            "fs": self.fs,
        }
        order = order.lower()

        for name in self.name_to_size.keys():
            # Skip 'fs' validation - it's handled separately with independent_ranks=True
            # and doesn't participate in the main orthogonal rank generation
            if name == "fs":
                continue
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), "
                    f"but you haven't specified the order ({order})."
                )
            elif name not in order:
                order = order + "-" + name

        self.order = order
        self.ordered_size = []

        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split("-")
        token = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ranks: bool = False):
        """Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
            independent_ranks (bool):
                If True, generate independent rank groups that divide the world
                into groups of the specified size. Used for FS (fully shard) groups
                which operate independently from the main parallelism hierarchy.
        """
        if independent_ranks and token == "fs":
            # FS groups divide world into groups of size fs
            # e.g., world_size=8, fs=4 -> [[0,1,2,3], [4,5,6,7]]
            ranks = []
            num_groups = self.world_size // self.fs
            for i in range(num_groups):
                group = list(range(i * self.fs + self.rank_offset, (i + 1) * self.fs + self.rank_offset))
                ranks.append(group)
            return ranks

        if token == "ep":
            ranks = []
            num_pp_stages = self.pp
            for i in range(num_pp_stages):
                start = i * self.ep + self.rank_offset
                end = start + self.ep
                ranks.append(list(range(start, end)))
            return ranks

        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks


# * QUERY
def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


# SP
def get_sp_group() -> SequenceParallelGroupCoordinator:
    assert _SP is not None, "pipeline model parallel group is not initialized"
    return _SP


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_sp_group().world_size


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_sp_group().rank_in_group


def get_ulysses_parallel_world_size():
    return get_sp_group().ulysses_world_size


def get_ulysses_parallel_rank():
    return get_sp_group().ulysses_rank


def get_ring_parallel_world_size():
    return get_sp_group().ring_world_size


def get_ring_parallel_rank():
    return get_sp_group().ring_rank


# PP
def get_pp_group() -> PipelineGroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def is_pipeline_first_stage():
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)


# CFG
def get_cfg_group() -> GroupCoordinator:
    assert _CFG is not None, "classifier_free_guidance parallel group is not initialized"
    return _CFG


def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group


# DP
def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "pipeline model parallel group is not initialized"
    return _DP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_dp_group().world_size


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_dp_group().rank_in_group


# FS (Fully Shard / HSDP shard dimension)
def get_fs_group() -> GroupCoordinator:
    assert _FS is not None, "fully shard group is not initialized"
    return _FS


def get_fully_shard_world_size():
    """Return world size for the fully shard group."""
    return get_fs_group().world_size


def get_fully_shard_rank():
    """Return my rank for the fully shard group."""
    return get_fs_group().rank_in_group


def is_dp_last_group():
    """Return True if in the last data parallel group, False otherwise."""
    return (
        get_sequence_parallel_rank() == (get_sequence_parallel_world_size() - 1)
        and get_classifier_free_guidance_rank() == (get_classifier_free_guidance_world_size() - 1)
        and get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)
    )


def get_dit_world_size():
    """Return world size for the DiT model."""
    return (
        get_data_parallel_world_size()
        * get_classifier_free_guidance_world_size()
        * get_sequence_parallel_world_size()
        * get_pipeline_parallel_world_size()
        * get_tensor_model_parallel_world_size()
    )


# * SET


def init_world_group(ranks: list[int], local_rank: int, backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str | None = None,
):
    if backend is None:
        backend = current_omni_platform.dist_backend
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing distributed environment"
        )
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        device_id = torch.distributed.get_rank() % current_omni_platform.get_device_count()
        current_omni_platform.set_device(current_omni_platform.get_torch_device(device_id))
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (
        _DP is not None
        and _CFG is not None
        and _SP is not None
        and _PP is not None
        and vllm_parallel_state._TP is not None
    )


def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    parallel_mode: str,
    allow_non_member: bool = False,
    logical_group: bool = False,
    group_id: str = "",
    **kwargs,
) -> GroupCoordinator:
    assert parallel_mode in [
        "data",
        "pipeline",
        "tensor",
        "expert",
        "sequence",
        "classifier_free_guidance",
        "fully_shard",
    ], f"parallel_mode {parallel_mode} is not supported"
    if logical_group:
        if parallel_mode == "sequence":
            return LogicalSequenceParallelGroupCoordinator(
                group_ranks=group_ranks,
                local_rank=local_rank,
                torch_distributed_backend=backend,
                allow_non_member=allow_non_member,
                group_id=group_id or parallel_mode,
                **kwargs,
            )
        return LogicalGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            allow_non_member=allow_non_member,
            group_id=group_id or parallel_mode,
        )
    if parallel_mode == "pipeline":
        return PipelineGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            allow_non_member=allow_non_member,
        )
    elif parallel_mode == "sequence":
        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            allow_non_member=allow_non_member,
            **kwargs,
        )
    else:
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            allow_non_member=allow_non_member,
        )


def init_dit_group(
    dit_parallel_size: int,
    backend: str,
):
    global _DIT
    _DIT = torch.distributed.new_group(ranks=list(range(dit_parallel_size)), backend=backend)


def get_dit_group():
    assert _DIT is not None, "DIT group is not initialized"
    return _DIT


def get_runtime_v2_model_parallel_session(group_id: str) -> dict[str, Any]:
    try:
        return _RUNTIME_V2_GROUP_SESSIONS[str(group_id)]
    except KeyError:
        raise KeyError(f"runtime_v2 model-parallel session not found for group_id={group_id!r}") from None


def activate_runtime_v2_model_parallel_session(group_id: str) -> None:
    session = get_runtime_v2_model_parallel_session(group_id)
    global _WORLD, _DP, _CFG, _PP, _SP, _FS, _DIT
    _WORLD = session["world"]
    _DP = session["dp"]
    _CFG = session["cfg"]
    _PP = session["pp"]
    _SP = session["sp"]
    _FS = session["fs"]
    _DIT = session["dit"]
    vllm_parallel_state._DP = _DP
    vllm_parallel_state._PP = _PP
    vllm_parallel_state._TP = session["tp"]
    if hasattr(vllm_parallel_state, "_EP"):
        vllm_parallel_state._EP = session.get("ep")


def ensure_runtime_v2_model_parallel_session(group_spec: Mapping[str, Any], backend: str | None = None) -> None:
    """Register one runtime_v2 model-parallel session on demand.

    This is intended for the GFC backend, where dynamic groups are logical
    host-side metadata and do not require a world-wide torch ``new_group``
    sequence. The torch backend still requires static session construction.
    """
    spec = _normalize_runtime_v2_execution_group(group_spec, 0)
    group_id = str(spec["group_id"])
    if group_id in _RUNTIME_V2_GROUP_SESSIONS:
        return
    if not runtime_v2_uses_gfc():
        raise RuntimeError(
            f"runtime_v2 group {group_id!r} was not initialized; dynamic on-demand groups require "
            "runtime_v2_collective_backend='gfc'"
        )
    if backend is None:
        backend = current_omni_platform.dist_backend
    rank = torch.distributed.get_rank()
    local_rank = get_world_group().local_rank
    session = _build_runtime_v2_group_session(
        spec=spec,
        rank=rank,
        local_rank=local_rank,
        backend=backend,
        logical_group=True,
    )
    if session is None:
        return
    _RUNTIME_V2_GROUP_SESSIONS[group_id] = session


# adapted from https://github.com/feifeibear/long-context-attention/blob/main/yunchang/globals.py
def set_seq_parallel_pg(
    sp_ulysses_degree: int,
    sp_ring_degree: int,
    rank: int,
    world_size: int,
    use_ulysses_low: bool = True,
    sp_group_ranks: list[list[int]] | None = None,
) -> tuple[torch.distributed.ProcessGroup, torch.distributed.ProcessGroup]:
    """
    Initialize sequence-parallel Ulysses and Ring process groups.

    This builds sequence-parallel (SP) subgroups inside each data-parallel (DP)
    slice. The SP group size is sp_ulysses_degree * sp_ring_degree, and
    world_size must be divisible by that size.

    Args:
        sp_ulysses_degree: Size of each Ulysses subgroup.
        sp_ring_degree: Size of each Ring subgroup.
        rank: Global rank of the current process.
        world_size: Total number of processes.
        use_ulysses_low: If True, Ulysses groups are contiguous chunks and Ring
            groups are strided within each SP group. If False, the opposite.
        sp_group_ranks: Optional explicit SP groups. Each entry must be a list
            of length sp_ulysses_degree * sp_ring_degree. When provided, groups
            are built from these ranks instead of auto-generated contiguous
            ranges.

    Returns:
        ulyssess_pg (torch.distributed.ProcessGroup): The Ulysses process group
            for this rank.
        ring_pg (torch.distributed.ProcessGroup): The Ring process group for
            this rank.

    Raises:
        ValueError: If sp_group_ranks length does not match world_size or any
            entry has the wrong size.
        AssertionError: If world_size is not divisible by sp_size.

    Behavior:
        - If sp_group_ranks is provided, groups are built per entry and each
          entry is further split into Ulysses/Ring groups according to
          use_ulysses_low.
        - If sp_group_ranks is None, groups are auto-generated within each DP
          slice using offsets of size sp_size.
    """
    sp_size = sp_ring_degree * sp_ulysses_degree
    dp_size = world_size // sp_size

    assert world_size % sp_size == 0, f"world_size {world_size} % sp_size {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    if sp_group_ranks is not None:
        if len(sp_group_ranks) * sp_size != world_size:
            raise ValueError(
                f"Invalid sp_group_ranks: expected {world_size // sp_size} groups of size {sp_size}, "
                f"but got {len(sp_group_ranks)} groups."
            )
        logger.info(
            "Building SP subgroups from explicit sp_group_ranks "
            f"(sp_size={sp_size}, ulysses={sp_ulysses_degree}, ring={sp_ring_degree}, "
            f"use_ulysses_low={use_ulysses_low})."
        )
        local_sp_group = None
        local_ulysses = None
        local_ring = None
        for group_ranks in sp_group_ranks:
            if len(group_ranks) != sp_size:
                raise ValueError(f"Invalid sp_group_ranks entry: expected size {sp_size}, got {len(group_ranks)}.")
            if rank in group_ranks:
                local_sp_group = list(group_ranks)
            if use_ulysses_low:
                # Ulysses groups are contiguous chunks; Ring groups are strided.
                for i in range(num_ulysses_pgs):
                    ulysses_ranks = group_ranks[i * sp_ulysses_degree : (i + 1) * sp_ulysses_degree]
                    group = torch.distributed.new_group(ulysses_ranks)
                    if rank in ulysses_ranks:
                        ulyssess_pg = group
                        local_ulysses = list(ulysses_ranks)
                for i in range(num_ring_pgs):
                    ring_ranks = group_ranks[i::num_ring_pgs]
                    group = torch.distributed.new_group(ring_ranks)
                    if rank in ring_ranks:
                        ring_pg = group
                        local_ring = list(ring_ranks)
            else:
                # Ring groups are contiguous chunks; Ulysses groups are strided.
                for i in range(num_ring_pgs):
                    ring_ranks = group_ranks[i * sp_ring_degree : (i + 1) * sp_ring_degree]
                    group = torch.distributed.new_group(ring_ranks)
                    if rank in ring_ranks:
                        ring_pg = group
                        local_ring = list(ring_ranks)
                for i in range(num_ulysses_pgs):
                    ulysses_ranks = group_ranks[i::num_ulysses_pgs]
                    group = torch.distributed.new_group(ulysses_ranks)
                    if rank in ulysses_ranks:
                        ulyssess_pg = group
                        local_ulysses = list(ulysses_ranks)
        if local_sp_group is not None:
            logger.info(
                "SP group details for rank %d: sp_group=%s, ulysses_group=%s, ring_group=%s",
                rank,
                local_sp_group,
                local_ulysses,
                local_ring,
            )
    else:
        if use_ulysses_low:
            for dp_rank in range(dp_size):
                offset = dp_rank * sp_size
                for i in range(num_ulysses_pgs):
                    ulysses_ranks = list(
                        range(
                            i * sp_ulysses_degree + offset,
                            (i + 1) * sp_ulysses_degree + offset,
                        )
                    )
                    group = torch.distributed.new_group(ulysses_ranks)
                    if rank in ulysses_ranks:
                        ulyssess_pg = group

                for i in range(num_ring_pgs):
                    ring_ranks = list(range(i + offset, sp_size + offset, num_ring_pgs))
                    group = torch.distributed.new_group(ring_ranks)
                    if rank in ring_ranks:
                        ring_pg = group

        else:
            for dp_rank in range(dp_size):
                offset = dp_rank * sp_size
                for i in range(num_ring_pgs):
                    ring_ranks = list(range(i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset))
                    group = torch.distributed.new_group(ring_ranks)
                    if rank in ring_ranks:
                        ring_pg = group

                for i in range(num_ulysses_pgs):
                    ulysses_ranks = list(range(i + offset, sp_size + offset, num_ulysses_pgs))
                    group = torch.distributed.new_group(ulysses_ranks)
                    if rank in ulysses_ranks:
                        ulyssess_pg = group

    return ulyssess_pg, ring_pg


def initialize_model_parallel(
    data_parallel_size: int = 1,
    cfg_parallel_size: int = 1,
    sequence_parallel_size: int | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    fully_shard_degree: int = 1,
    hsdp_replicate_size: int = 1,
    enable_expert_parallel: bool = False,
    backend: str | None = None,
) -> None:
    if backend is None:
        backend = current_omni_platform.dist_backend
    """
    Initialize model parallel groups.

    Arguments:
        data_parallel_size: number of data parallelism groups.
        cfg_parallel_size: number of GPUs used for Classifier Free Guidance (CFG) parallelism.
        sequence_parallel_size: number of GPUs used for sequence parallelism.
            sequence_parallel_size = ulysses_degree * ring_degree
        ulysses_degree: number of GPUs used for ulysses sequence parallelism.
        ring_degree: number of GPUs used for ring sequence parallelism.
        tensor_parallel_size: number of GPUs used for tensor parallelism.
        pipeline_parallel_size: number of GPUs used for pipeline parallelism.
        fully_shard_degree: number of GPUs used for fully sharded data parallelism (HSDP shard dimension).
        backend: distributed backend of pytorch collective comm.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 groups to parallelize the batch dim(dp), 2 groups to parallelize
    split batch caused by CFG, and 2 GPUs to parallelize sequence.

    dp_size (2) * cfg_size (2) * sp_size (2) * pp_size (2) = 16.

    The present function will create 8 data-parallel groups,
    8 CFG group, 8 pipeline-parallel group, and
    8 sequence-parallel groups:
        8 data-parallel groups:
            [g0, g8], [g1, g9], [g2, g10], [g3, g11],
            [g4, g12], [g5, g13], [g6, g14], [g7, g15]
        8 CFG-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7],
            [g8, g12], [g9, g13], [g10, g14], [g11, g15]
        8 sequence-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7],
            [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 pipeline-parallel groups:
            [g0, g2], [g4, g6], [g8, g10], [g12, g14],
            [g1, g3], [g5, g7], [g9, g11], [g13, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if sequence_parallel_size is None:
        sequence_parallel_size = ring_degree * ulysses_degree
        logger.info(
            f"sequence_parallel_size is not provided, using ring_degree * ulysses_degree = {sequence_parallel_size}"
        )

    if sequence_parallel_size != ring_degree * ulysses_degree:
        raise ValueError(
            "sequence_parallel_size is not equal to ring_degree * ulysses_degree,"
            f" but got {sequence_parallel_size} != {ring_degree} * {ulysses_degree}"
        )

    # FIXME: Since the async p2p communication operation of NPU is not same as cuda in torch,
    # the pipefusion is not ready for npu yet
    if current_omni_platform.is_npu():
        assert pipeline_parallel_size == 1, "Current pipefusion is not ready for NPU"

    dit_parallel_size = (
        data_parallel_size * cfg_parallel_size * sequence_parallel_size * pipeline_parallel_size * tensor_parallel_size
    )

    # Check for standalone HSDP: all non-HSDP parallelism dimensions are 1
    is_standalone_hsdp = dit_parallel_size == 1 and fully_shard_degree > 1

    # For standalone HSDP: use (fully_shard_degree * hsdp_replicate_size) as dit_parallel_size
    # This ensures orthogonal rank generation works correctly for all HSDP workers
    if is_standalone_hsdp:
        dit_parallel_size = fully_shard_degree * hsdp_replicate_size

    if world_size < dit_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is less than "
            f"tensor_parallel_size ({tensor_parallel_size}) x "
            f"pipeline_parallel_size ({pipeline_parallel_size}) x"
            f"sequence_parallel_size ({sequence_parallel_size}) x"
            f"cfg_parallel_size "
            f"({cfg_parallel_size}) x"
            f"data_parallel_size ({data_parallel_size})"
        )

    # For standalone HSDP, use (fully_shard_degree * hsdp_replicate_size) as data_parallel_size
    # so that RankGenerator.world_size matches the actual number of workers
    effective_dp_size = (fully_shard_degree * hsdp_replicate_size) if is_standalone_hsdp else data_parallel_size

    rank_generator: RankGenerator = RankGenerator(
        tensor_parallel_size,
        sequence_parallel_size,
        pipeline_parallel_size,
        cfg_parallel_size,
        effective_dp_size,
        fs=fully_shard_degree,
        order="tp-sp-pp-cfg-dp",
    )
    sp_group_ranks = rank_generator.get_ranks("sp")
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    _DP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("dp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="data",
    )
    vllm_parallel_state._DP = _DP

    global _CFG
    assert _CFG is None, "classifier_free_guidance group is already initialized"
    _CFG = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("cfg"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
    )
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    _PP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("pp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="pipeline",
    )
    vllm_parallel_state._PP = _PP

    global _SP
    assert _SP is None, "sequence parallel group is already initialized"
    ulysses_pg, ring_pg = set_seq_parallel_pg(
        sp_ulysses_degree=ulysses_degree,
        sp_ring_degree=ring_degree,
        rank=get_world_group().rank_in_group,
        world_size=dit_parallel_size,
        sp_group_ranks=sp_group_ranks,
    )
    _SP = init_model_parallel_group(
        group_ranks=sp_group_ranks,
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="sequence",
        ulysses_group=ulysses_pg,
        ring_group=ring_pg,
    )

    assert vllm_parallel_state._TP is None, "Tensor parallel group is already initialized"
    vllm_parallel_state._TP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("tp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="tensor",
    )

    global _FS
    assert _FS is None, "fully shard group is already initialized"
    _FS = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("fs", independent_ranks=True),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="fully_shard",
    )

    if enable_expert_parallel:
        od_config: OmniDiffusionConfig | None = get_forward_context().omni_diffusion_config
        if od_config and od_config.is_moe:
            vllm_parallel_state._EP = init_model_parallel_group(
                group_ranks=rank_generator.get_ranks("ep"),
                local_rank=get_world_group().local_rank,
                backend=backend,
                parallel_mode="expert",
            )
        else:
            raise RuntimeError("Expert parallelism enabled for a non-MoE model ")

    init_dit_group(dit_parallel_size, backend)


def _normalize_runtime_v2_execution_group(raw_spec: Mapping[str, Any], index: int) -> dict[str, Any]:
    # Normalize one execution-group spec into validated integer dimensions and
    # explicit rank list. This keeps downstream group construction strict and
    # deterministic.
    group_id = str(raw_spec.get("group_id", f"g{index}"))
    ranks_raw = raw_spec.get("ranks")
    if not isinstance(ranks_raw, list) or not ranks_raw:
        raise ValueError(f"runtime_v2 execution group {group_id!r} must provide non-empty 'ranks' list")
    ranks = [int(rank) for rank in ranks_raw]
    tp = int(raw_spec.get("tp", 1))
    sp = int(raw_spec.get("sp", 1))
    cfg = int(raw_spec.get("cfg", 1))
    ulysses_degree = int(raw_spec.get("ulysses_degree", 1))
    ring_degree = int(raw_spec.get("ring_degree", 1))
    if tp < 1 or sp < 1 or cfg < 1:
        raise ValueError(f"runtime_v2 execution group {group_id!r} must have tp/sp/cfg >= 1")
    if ulysses_degree < 1 or ring_degree < 1:
        raise ValueError(f"runtime_v2 execution group {group_id!r} must have ulysses_degree/ring_degree >= 1")
    if sp != ulysses_degree * ring_degree:
        raise ValueError(
            f"runtime_v2 execution group {group_id!r} has invalid sequence parallel config: "
            f"sp={sp} != ulysses_degree({ulysses_degree}) * ring_degree({ring_degree})"
        )
    if len(ranks) != tp * sp * cfg:
        raise ValueError(
            f"runtime_v2 execution group {group_id!r} has invalid size: len(ranks)={len(ranks)} "
            f"!= tp({tp})*sp({sp})*cfg({cfg})={tp * sp * cfg}"
        )
    return {
        "group_id": group_id,
        "ranks": ranks,
        "tp": tp,
        "sp": sp,
        "cfg": cfg,
        "ulysses_degree": ulysses_degree,
        "ring_degree": ring_degree,
    }


def _remap_local_rank_groups(local_groups: list[list[int]], global_ranks: list[int]) -> list[list[int]]:
    # Convert local rank ids (0..N-1 inside one execution group) to global
    # world ranks.
    mapped: list[list[int]] = []
    for local_group in local_groups:
        mapped.append([global_ranks[int(local_rank)] for local_rank in local_group])
    return mapped


def _create_sp_subgroups_for_one_sp_group(
    *,
    sp_group_ranks: list[int],
    ulysses_degree: int,
    ring_degree: int,
    rank: int,
    backend: str,
    use_ulysses_low: bool = True,
) -> tuple[Any | None, Any | None]:
    # Build Ulysses/Ring process groups for one SP slice. Caller may invoke
    # this for all slices; only the slice containing `rank` returns non-None
    # local handles for the current process.
    if len(sp_group_ranks) != ulysses_degree * ring_degree:
        raise ValueError(
            f"invalid SP group size {len(sp_group_ranks)} for "
            f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
        )
    num_ulysses_pgs = ring_degree
    num_ring_pgs = ulysses_degree
    local_ulysses_pg = None
    local_ring_pg = None

    if use_ulysses_low:
        for i in range(num_ulysses_pgs):
            ulysses_ranks = sp_group_ranks[i * ulysses_degree : (i + 1) * ulysses_degree]
            group = torch.distributed.new_group(ulysses_ranks, backend=backend)
            if rank in ulysses_ranks:
                local_ulysses_pg = group
        for i in range(num_ring_pgs):
            ring_ranks = sp_group_ranks[i::num_ring_pgs]
            group = torch.distributed.new_group(ring_ranks, backend=backend)
            if rank in ring_ranks:
                local_ring_pg = group
    else:
        for i in range(num_ring_pgs):
            ring_ranks = sp_group_ranks[i * ring_degree : (i + 1) * ring_degree]
            group = torch.distributed.new_group(ring_ranks, backend=backend)
            if rank in ring_ranks:
                local_ring_pg = group
        for i in range(num_ulysses_pgs):
            ulysses_ranks = sp_group_ranks[i::num_ulysses_pgs]
            group = torch.distributed.new_group(ulysses_ranks, backend=backend)
            if rank in ulysses_ranks:
                local_ulysses_pg = group

    return local_ulysses_pg, local_ring_pg


def _create_logical_sp_subgroups_for_one_sp_group(
    *,
    sp_group_ranks: list[int],
    ulysses_degree: int,
    ring_degree: int,
    rank: int,
    use_ulysses_low: bool = True,
    group_id_prefix: str = "",
) -> tuple[LogicalGroupHandle | None, LogicalGroupHandle | None]:
    if len(sp_group_ranks) != ulysses_degree * ring_degree:
        raise ValueError(
            f"invalid SP group size {len(sp_group_ranks)} for "
            f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
        )
    num_ulysses_pgs = ring_degree
    num_ring_pgs = ulysses_degree
    local_ulysses_pg = None
    local_ring_pg = None

    if use_ulysses_low:
        for i in range(num_ulysses_pgs):
            ulysses_ranks = sp_group_ranks[i * ulysses_degree : (i + 1) * ulysses_degree]
            if rank in ulysses_ranks:
                local_ulysses_pg = make_logical_group(
                    ulysses_ranks,
                    group_id=f"{group_id_prefix}:ulysses:{i}",
                )
        for i in range(num_ring_pgs):
            ring_ranks = sp_group_ranks[i::num_ring_pgs]
            if rank in ring_ranks:
                local_ring_pg = make_logical_group(
                    ring_ranks,
                    group_id=f"{group_id_prefix}:ring:{i}",
                )
    else:
        for i in range(num_ring_pgs):
            ring_ranks = sp_group_ranks[i * ring_degree : (i + 1) * ring_degree]
            if rank in ring_ranks:
                local_ring_pg = make_logical_group(
                    ring_ranks,
                    group_id=f"{group_id_prefix}:ring:{i}",
                )
        for i in range(num_ulysses_pgs):
            ulysses_ranks = sp_group_ranks[i::num_ulysses_pgs]
            if rank in ulysses_ranks:
                local_ulysses_pg = make_logical_group(
                    ulysses_ranks,
                    group_id=f"{group_id_prefix}:ulysses:{i}",
                )

    return local_ulysses_pg, local_ring_pg


def _validate_gfc_runtime_v2_group(spec: Mapping[str, Any]) -> None:
    if int(spec["tp"]) != 1:
        raise ValueError(
            "runtime_v2_collective_backend='gfc' does not replace tensor-parallel all_reduce in v1; "
            f"group {spec['group_id']!r} must use tp=1"
        )
    if int(spec["cfg"]) != 1:
        raise ValueError(
            "runtime_v2_collective_backend='gfc' does not replace CFG collectives in v1; "
            f"group {spec['group_id']!r} must use cfg=1"
        )
    if len(spec["ranks"]) > 16:
        raise ValueError(
            "runtime_v2_collective_backend='gfc' supports execution groups of at most 16 ranks in v1; "
            f"group {spec['group_id']!r} has {len(spec['ranks'])} ranks"
        )


def _build_runtime_v2_group_session(
    *,
    spec: Mapping[str, Any],
    rank: int,
    local_rank: int,
    backend: str,
    logical_group: bool,
) -> dict[str, Any] | None:
    group_ranks = list(spec["ranks"])
    is_member = rank in group_ranks
    if logical_group and not is_member:
        return None
    if logical_group:
        _validate_gfc_runtime_v2_group(spec)

    rank_generator = RankGenerator(
        tp=int(spec["tp"]),
        sp=int(spec["sp"]),
        pp=1,
        cfg=int(spec["cfg"]),
        dp=1,
        fs=1,
        order="tp-sp-pp-cfg-dp",
        rank_offset=0,
    )
    dp_group_ranks = _remap_local_rank_groups(rank_generator.get_ranks("dp"), group_ranks)
    cfg_group_ranks = _remap_local_rank_groups(rank_generator.get_ranks("cfg"), group_ranks)
    pp_group_ranks = _remap_local_rank_groups(rank_generator.get_ranks("pp"), group_ranks)
    mapped_sp_group_ranks = _remap_local_rank_groups(rank_generator.get_ranks("sp"), group_ranks)
    sp_group_ranks = mapped_sp_group_ranks
    tp_group_ranks = _remap_local_rank_groups(rank_generator.get_ranks("tp"), group_ranks)
    fs_group_ranks = _remap_local_rank_groups(rank_generator.get_ranks("fs", independent_ranks=True), group_ranks)

    local_ulysses_pg = None
    local_ring_pg = None
    for idx, one_sp_group_ranks in enumerate(mapped_sp_group_ranks):
        if logical_group:
            ulysses_pg, ring_pg = _create_logical_sp_subgroups_for_one_sp_group(
                sp_group_ranks=one_sp_group_ranks,
                ulysses_degree=int(spec["ulysses_degree"]),
                ring_degree=int(spec["ring_degree"]),
                rank=rank,
                group_id_prefix=f"{spec['group_id']}:sp:{idx}",
            )
        else:
            ulysses_pg, ring_pg = _create_sp_subgroups_for_one_sp_group(
                sp_group_ranks=one_sp_group_ranks,
                ulysses_degree=int(spec["ulysses_degree"]),
                ring_degree=int(spec["ring_degree"]),
                rank=rank,
                backend=backend,
            )
        if ulysses_pg is not None:
            local_ulysses_pg = ulysses_pg
        if ring_pg is not None:
            local_ring_pg = ring_pg

    group_prefix = str(spec["group_id"])
    dp = init_model_parallel_group(
        group_ranks=dp_group_ranks,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="data",
        allow_non_member=True,
        logical_group=logical_group,
        group_id=f"{group_prefix}:dp",
    )
    cfg = init_model_parallel_group(
        group_ranks=cfg_group_ranks,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
        allow_non_member=True,
        logical_group=logical_group,
        group_id=f"{group_prefix}:cfg",
    )
    pp = init_model_parallel_group(
        group_ranks=pp_group_ranks,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="pipeline",
        allow_non_member=True,
        logical_group=logical_group,
        group_id=f"{group_prefix}:pp",
    )
    sp = init_model_parallel_group(
        group_ranks=sp_group_ranks,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="sequence",
        allow_non_member=True,
        logical_group=logical_group,
        group_id=f"{group_prefix}:sp",
        ulysses_group=local_ulysses_pg,
        ring_group=local_ring_pg,
    )
    tp = init_model_parallel_group(
        group_ranks=tp_group_ranks,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="tensor",
        allow_non_member=True,
        logical_group=logical_group,
        group_id=f"{group_prefix}:tp",
    )
    fs = init_model_parallel_group(
        group_ranks=fs_group_ranks,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="fully_shard",
        allow_non_member=True,
        logical_group=logical_group,
        group_id=f"{group_prefix}:fs",
    )
    if logical_group:
        dit_group = make_logical_group(group_ranks, group_id=f"{group_prefix}:dit")
    else:
        dit_group = torch.distributed.new_group(ranks=group_ranks, backend=backend)

    if not is_member:
        return None
    if local_ulysses_pg is None or local_ring_pg is None:
        raise RuntimeError(
            f"failed to resolve ulysses/ring subgroup for rank {rank} "
            f"in runtime_v2 group {spec['group_id']!r}"
        )
    return {
        "world": get_world_group(),
        "dp": dp,
        "cfg": cfg,
        "pp": pp,
        "sp": sp,
        "tp": tp,
        "fs": fs,
        "dit": dit_group,
        "ep": None,
    }


def initialize_model_parallel_from_execution_groups(
    *,
    execution_groups: list[dict[str, Any]],
    current_group_id: str | None = None,
    enable_expert_parallel: bool = False,
    backend: str | None = None,
) -> None:
    """Initialize model-parallel groups from explicit runtime_v2 execution groups.

    All workers share one global torch.distributed world. Runtime v2 may place a
    worker in multiple execution groups, so this path builds one complete
    TP/SP/CFG/PP/FS/DIT session per group and activates `current_group_id` as
    the legacy global view used by existing model code.
    """
    if backend is None:
        backend = current_omni_platform.dist_backend
    assert torch.distributed.is_initialized()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if not execution_groups:
        raise ValueError("runtime_v2 shared-world mode requires non-empty execution_groups")

    normalized_groups: list[dict[str, Any]] = []
    all_group_ranks: list[int] = []
    current_group_found = current_group_id is None
    rank_in_current_group = current_group_id is None
    rank_covered = False
    for idx, raw_group in enumerate(execution_groups):
        if not isinstance(raw_group, Mapping):
            raise ValueError(
                f"runtime_v2 execution_groups entry #{idx} must be a mapping, got {type(raw_group)!r}"
            )
        spec = _normalize_runtime_v2_execution_group(raw_group, idx)
        normalized_groups.append(spec)
        all_group_ranks.extend(spec["ranks"])
        if current_group_id is not None and spec["group_id"] == current_group_id:
            current_group_found = True
            if rank in spec["ranks"]:
                rank_in_current_group = True
        if rank in spec["ranks"]:
            rank_covered = True

    if not rank_covered:
        raise ValueError(f"rank {rank} does not belong to any runtime_v2 execution group")
    if not current_group_found:
        raise ValueError(f"current_group_id={current_group_id!r} does not exist in runtime_v2 execution_groups")
    if not rank_in_current_group:
        raise ValueError(
            f"rank {rank} does not belong to current_group_id={current_group_id!r} "
            "in runtime_v2 execution_groups"
        )
    expected_ranks = set(range(world_size))
    actual_ranks = set(all_group_ranks)
    missing_ranks = sorted(expected_ranks - actual_ranks)
    extra_ranks = sorted(actual_ranks - expected_ranks)
    if missing_ranks:
        raise ValueError(f"runtime_v2 execution_groups missing ranks from global world: {missing_ranks!r}")
    if extra_ranks:
        raise ValueError(f"runtime_v2 execution_groups contain out-of-range ranks: {extra_ranks!r}")

    global _RUNTIME_V2_GROUP_SESSIONS
    if _RUNTIME_V2_GROUP_SESSIONS:
        raise RuntimeError("runtime_v2 model-parallel sessions are already initialized")
    if _DP is not None or _CFG is not None or _PP is not None or _SP is not None or vllm_parallel_state._TP is not None:
        raise RuntimeError("model-parallel groups are already initialized")

    local_rank = get_world_group().local_rank
    # Static initialization: every rank participates, so torch.distributed.new_group
    # works correctly. Only the on-demand ensure_runtime_v2_model_parallel_session
    # path goes through the pure-logical branch (where non-members must skip).
    sessions: dict[str, dict[str, Any]] = {}
    for spec in normalized_groups:
        session = _build_runtime_v2_group_session(
            spec=spec,
            rank=rank,
            local_rank=local_rank,
            backend=backend,
            logical_group=False,
        )
        if session is not None:
            sessions[spec["group_id"]] = session

    # When GFC is the active backend, side-attach GFC group descriptors to the
    # SP-side torch subgroups so comm.py can route all_gather / all_to_all
    # through the symmetric-memory path while everything else (TP all_reduce,
    # VAE gather/broadcast, ring P2P, PP send/recv) keeps using torch.
    if runtime_v2_uses_gfc():
        for session in sessions.values():
            sp_coord = session.get("sp")
            if sp_coord is None:
                continue
            for pg in (sp_coord.device_group, sp_coord.ulysses_group, sp_coord.ring_group):
                if pg is not None and is_torch_process_group(pg):
                    register_static_gfc_subgroup(pg)

    if enable_expert_parallel:
        raise NotImplementedError(
            "runtime_v2 shared-world initialization currently does not support expert parallel groups"
        )

    if not sessions:
        raise RuntimeError(f"rank {rank} did not initialize any runtime_v2 group session")
    active_group_id = str(current_group_id) if current_group_id is not None else next(iter(sessions))
    if active_group_id not in sessions:
        raise RuntimeError(
            f"rank {rank} does not have a runtime_v2 session for active group_id={active_group_id!r}"
        )
    _RUNTIME_V2_GROUP_SESSIONS = sessions
    activate_runtime_v2_model_parallel_session(active_group_id)


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _RUNTIME_V2_GROUP_SESSIONS
    if _RUNTIME_V2_GROUP_SESSIONS:
        destroyed: set[int] = set()
        for session in _RUNTIME_V2_GROUP_SESSIONS.values():
            for key in ("dp", "cfg", "pp", "sp", "tp", "fs"):
                group = session.get(key)
                if group is None or id(group) in destroyed:
                    continue
                destroyed.add(id(group))
                group.destroy()
            dit_group = session.get("dit")
            if dit_group is not None and id(dit_group) not in destroyed:
                destroyed.add(id(dit_group))
                if is_torch_process_group(dit_group):
                    torch.distributed.destroy_process_group(dit_group)
        _RUNTIME_V2_GROUP_SESSIONS.clear()

    global _DP
    if _DP:
        _DP.destroy()
    _DP = None

    global _CFG
    if _CFG:
        _CFG.destroy()
    _CFG = None

    global _SP
    if _SP:
        _SP.destroy()
    _SP = None

    if vllm_parallel_state._TP:
        vllm_parallel_state._TP.destroy()
    vllm_parallel_state._TP = None

    if vllm_parallel_state._EP:
        vllm_parallel_state._EP.destroy()
    vllm_parallel_state._EP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _FS
    if _FS:
        _FS.destroy()
    _FS = None

    global _DIT
    _DIT = None


def destroy_distributed_environment():
    global _WORLD, _RUNTIME_V2_GROUP_SESSIONS
    _RUNTIME_V2_GROUP_SESSIONS.clear()
    shutdown_runtime_v2_collective_runtime()
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def destroy_distributed_env():
    if model_parallel_is_initialized():
        destroy_model_parallel()
    destroy_distributed_environment()
