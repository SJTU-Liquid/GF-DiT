# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2._env import env_flag

logger = init_logger(__name__)

_VALID_BACKENDS = {"torch", "gfc"}
_COLLECTIVE_BACKEND = "torch"
_GFC_RUNTIME: Any | None = None
# Optional second GFC runtime dedicated to RESHARD/migration P2P. Kept separate
# from _GFC_RUNTIME so migration barriers never interleave with DiT collective
# barriers (GFC requires a consistent per-rank-pair barrier order). Created only
# when migration GFC P2P is opted into via VLLM_RUNTIME_V2_MIGRATE_GFC_P2P.
_GFC_MIGRATE_RUNTIME: Any | None = None
# Side-table mapping id(torch.distributed.ProcessGroup) -> GFC GroupDescriptor.
# Torch ProcessGroup is a pybind11 type and does not accept setattr, so we keep
# the descriptor here. Cleared on shutdown_runtime_v2_collective_runtime.
_GFC_GROUP_BY_PG_ID: dict[int, Any] = {}


def normalize_runtime_v2_collective_backend(value: Any) -> str:
    backend = str(value or "torch").lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            "runtime_v2_collective_backend must be one of ('torch', 'gfc'), "
            f"got {value!r}"
        )
    return backend


def set_runtime_v2_collective_backend(backend: str) -> None:
    global _COLLECTIVE_BACKEND
    _COLLECTIVE_BACKEND = normalize_runtime_v2_collective_backend(backend)


def get_runtime_v2_collective_backend() -> str:
    return _COLLECTIVE_BACKEND


def runtime_v2_uses_gfc() -> bool:
    return _COLLECTIVE_BACKEND == "gfc"


def init_runtime_v2_collective_runtime(
    *,
    backend: str,
    device: torch.device,
    max_group_size: int | None = None,
    max_groups: int | None = None,
    max_collective_bytes: int = 128 * 1024 * 1024,
) -> Any | None:
    """Initialize the runtime_v2 collective backend for this process.

    The torch backend needs no process-local runtime. The GFC backend performs
    a one-time symmetric-memory rendezvous across the already initialized
    world process group; it is intentionally imported only when selected.

    ``max_groups`` is accepted for backward compatibility but ignored: GFC no
    longer pre-allocates a fixed group pool, so the number of registered groups
    is unbounded (each group allocates its own device rank list on demand).
    """

    backend = normalize_runtime_v2_collective_backend(backend)
    set_runtime_v2_collective_backend(backend)
    if backend == "torch":
        return None

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("runtime_v2_collective_backend='gfc' requires torch.distributed to be initialized")
    if device.type != "cuda":
        raise RuntimeError(f"runtime_v2_collective_backend='gfc' requires a CUDA device, got {device!r}")

    global _GFC_RUNTIME
    if _GFC_RUNTIME is not None:
        return _GFC_RUNTIME

    try:
        from gfc import SymmetricCollectiveConfig, SymmetricCollectiveRuntime
    except ImportError as exc:
        raise RuntimeError(
            "runtime_v2_collective_backend='gfc' requires the 'gfc' package to be installed; "
            "install it before starting the runtime"
        ) from exc

    world_size = int(dist.get_world_size())
    max_group = int(max_group_size or world_size)
    if max_groups is not None:
        logger.warning(
            "runtime_v2 GFC: max_groups=%s is deprecated and ignored; GFC group "
            "registration is now unbounded.",
            max_groups,
        )
    config_kwargs: dict[str, Any] = dict(
        max_group_size=max_group,
        max_collective_bytes=int(max_collective_bytes),
        enable_debug_checks=False,
    )
    # Older gfc builds pre-allocated a fixed group pool and require max_groups;
    # newer builds dropped it for unbounded per-group registration. Pass it only
    # when the installed SymmetricCollectiveConfig actually declares the param, so
    # we neither TypeError on an old build (missing required arg) nor on a new one
    # (unexpected kwarg). The deprecated caller value is ignored either way.
    try:
        _accepts_max_groups = "max_groups" in inspect.signature(SymmetricCollectiveConfig).parameters
    except (TypeError, ValueError):
        _accepts_max_groups = False
    if _accepts_max_groups:
        config_kwargs["max_groups"] = 64
    config = SymmetricCollectiveConfig(**config_kwargs)
    _GFC_RUNTIME = SymmetricCollectiveRuntime(config, device=device)
    logger.info(
        "runtime_v2 GFC collective runtime initialized: world_size=%s max_group_size=%s "
        "(group count unbounded; per-group rank lists)",
        world_size,
        max_group,
    )
    # A dedicated runtime for RESHARD/migration P2P. Its stream, signal
    # buffers, and epoch space are independent of the DiT collective runtime,
    # so migration p2p (driven by the command thread) cannot desync DiT's
    # per-rank-pair barrier ordering. Built here -- in lockstep on every rank
    # -- because the SymmetricCollectiveRuntime constructor is itself
    # collective (symm-mem enable + rendezvous + nonce broadcast). Built by
    # default whenever GFC is the backend (migration uses it unless
    # VLLM_RUNTIME_V2_MIGRATE_GFC_P2P=0 forces NCCL).
    if env_flag("VLLM_RUNTIME_V2_MIGRATE_GFC_P2P", True):
        global _GFC_MIGRATE_RUNTIME
        _GFC_MIGRATE_RUNTIME = SymmetricCollectiveRuntime(config, device=device)
        logger.info(
            "runtime_v2 GFC migration runtime initialized (dedicated RESHARD P2P)"
        )
    return _GFC_RUNTIME


def get_gfc_runtime() -> Any:
    if _GFC_RUNTIME is None:
        raise RuntimeError("GFC collective runtime is not initialized")
    return _GFC_RUNTIME


def get_gfc_migrate_runtime() -> Any:
    if _GFC_MIGRATE_RUNTIME is None:
        raise RuntimeError(
            "GFC migration runtime is not initialized; set "
            "VLLM_RUNTIME_V2_MIGRATE_GFC_P2P=1 before starting the runtime"
        )
    return _GFC_MIGRATE_RUNTIME


@dataclass
class LogicalGroupHandle:
    """Ordered rank metadata plus optional concrete backend handles."""

    ranks: tuple[int, ...]
    rank: int
    group_id: str = ""
    torch_group: Any | None = None
    gfc_group: Any | None = None

    def __post_init__(self) -> None:
        self.ranks = tuple(int(rank) for rank in self.ranks)
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError(f"logical group ranks must be unique, got {self.ranks!r}")
        self.rank = int(self.rank)

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def rank_in_group(self) -> int:
        try:
            return self.ranks.index(self.rank)
        except ValueError:
            return -1

    @property
    def device_group(self) -> Any | None:
        return self.torch_group

    @property
    def local_index(self) -> int:
        return self.rank_in_group

    def size(self) -> int:
        return self.world_size


def make_logical_group(
    ranks: tuple[int, ...] | list[int],
    *,
    group_id: str = "",
    torch_group: Any | None = None,
    register_gfc: bool | None = None,
) -> LogicalGroupHandle:
    ordered_ranks = tuple(int(rank) for rank in ranks)
    rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else -1
    gfc_group = None
    should_register = runtime_v2_uses_gfc() if register_gfc is None else bool(register_gfc)
    if should_register and rank in ordered_ranks and len(ordered_ranks) > 1:
        gfc_group = get_gfc_runtime().register_group(ordered_ranks)
    return LogicalGroupHandle(
        ranks=ordered_ranks,
        rank=rank,
        group_id=group_id,
        torch_group=torch_group,
        gfc_group=gfc_group,
    )


def is_torch_process_group(group: Any) -> bool:
    process_group_type = getattr(dist, "ProcessGroup", None)
    return process_group_type is not None and isinstance(group, process_group_type)


def get_group_ranks(group: Any | None) -> tuple[int, ...]:
    if group is None:
        return tuple(range(int(dist.get_world_size())))
    if isinstance(group, LogicalGroupHandle):
        return group.ranks
    ranks = getattr(group, "ranks", None)
    if ranks is not None:
        return tuple(int(rank) for rank in ranks)
    world_size = int(dist.get_world_size(group))
    return tuple(int(dist.get_global_rank(group, idx)) for idx in range(world_size))


def get_group_world_size(group: Any | None) -> int:
    if group is None:
        return int(dist.get_world_size())
    if isinstance(group, LogicalGroupHandle):
        return group.world_size
    world_size = getattr(group, "world_size", None)
    if world_size is not None and not is_torch_process_group(group):
        return int(world_size)
    return int(dist.get_world_size(group))


def get_group_rank(group: Any | None) -> int:
    if group is None:
        return int(dist.get_rank())
    if isinstance(group, LogicalGroupHandle):
        return int(group.rank_in_group)
    rank_in_group = getattr(group, "rank_in_group", None)
    if rank_in_group is not None and not is_torch_process_group(group):
        return int(rank_in_group)
    return int(dist.get_rank(group))


def _torch_group(group: Any | None) -> Any | None:
    if isinstance(group, LogicalGroupHandle):
        return group.torch_group
    return group


def _gfc_group(group: Any | None) -> Any | None:
    if isinstance(group, LogicalGroupHandle):
        return group.gfc_group
    descriptor = _GFC_GROUP_BY_PG_ID.get(id(group))
    if descriptor is not None:
        return descriptor
    return getattr(group, "gfc_group", None)


def attach_gfc_group_to_pg(pg: Any, gfc_group: Any) -> None:
    """Bind a GFC GroupDescriptor to a torch ProcessGroup for collective routing.

    Used when a torch subgroup was created via ``new_group`` at static startup
    and we additionally want GFC's all-gather / all-to-all path to apply to
    it. Stored in a side-table because torch's pybind11 ProcessGroup does not
    accept setattr.
    """
    _GFC_GROUP_BY_PG_ID[id(pg)] = gfc_group


def register_static_gfc_subgroup(pg: Any) -> Any | None:
    """Register a GFC group for an existing torch subgroup and attach it.

    Returns the GFC descriptor, or ``None`` if the world size is 1 or GFC is
    not the active backend. Idempotent: re-registering a pg replaces the
    descriptor.
    """
    if pg is None or not runtime_v2_uses_gfc():
        return None
    ranks = tuple(int(dist.get_global_rank(pg, idx)) for idx in range(int(dist.get_world_size(pg))))
    if len(ranks) <= 1:
        return None
    descriptor = get_gfc_runtime().register_group(ranks)
    attach_gfc_group_to_pg(pg, descriptor)
    return descriptor


def _dynamo_disable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Apply torch._dynamo.disable if torch.compile is available.

    The GFC wrappers bridge to the external GFC runtime (symmetric-memory
    kernels, event handshake, side-table lookup). Dynamo cannot trace any of
    that, and tracing through ``input_.nbytes`` blows up on FakeTensors with
    symbolic shapes anyway. Force a graph break so the wrapper runs eagerly.
    """
    try:
        import torch._dynamo as _dynamo
    except ImportError:  # pragma: no cover - dynamo always present in supported torch
        return fn
    return _dynamo.disable(fn)


@_dynamo_disable
def _run_gfc_collective(
    tensors: Sequence[torch.Tensor],
    fn: Callable[[Any], None],
) -> None:
    if not tensors:
        raise RuntimeError("_run_gfc_collective requires at least one tensor")
    first = tensors[0]
    if not first.is_cuda:
        raise RuntimeError(
            "runtime_v2_collective_backend='gfc' requires CUDA tensors; "
            f"got tensor on device={first.device!r}"
        )
    runtime = get_gfc_runtime()
    stream = torch.cuda.current_stream(first.device)
    start = torch.cuda.Event()
    start.record(stream)
    runtime.wait_for_external(start)
    # Tell the PyTorch caching allocator that these buffers are now in use by
    # the GFC stream. Without this, when the caller's locals go out of scope
    # the allocator only sees the caller stream's events and may reuse the
    # buffer for a fresh allocation BEFORE the GFC kernel finishes reading or
    # writing it (illegal memory access). record_stream marks the buffer as
    # busy until the GFC stream catches up. Correctness for the *caller's*
    # next read of the output tensor is still enforced by stream.wait_event
    # below — record_stream is only about safe freeing.
    for t in tensors:
        t.record_stream(runtime.stream)
    fn(runtime)
    done = runtime.record_event()
    # GPU-side cross-stream sync: caller's compute stream waits for the GFC
    # stream's `done` event before reading the output tensor. Both
    # record_event and wait_event are non-blocking; no host polling.
    stream.wait_event(done)


def _require_gfc_group(group: Any | None) -> Any:
    gfc_group = _gfc_group(group)
    if gfc_group is None:
        raise RuntimeError(
            "runtime_v2_collective_backend='gfc' is active but the collective group has no GFC descriptor; "
            f"got group={group!r}. Build groups via make_logical_group() so register_group() runs"
        )
    return gfc_group


@_dynamo_disable
def all_to_all_single(output: torch.Tensor, input_: torch.Tensor, *, group: Any | None) -> None:
    if runtime_v2_uses_gfc():
        world_size = get_group_world_size(group)
        if world_size == 1:
            output.copy_(input_)
            return
        gfc_group = _require_gfc_group(group)
        if input_.nbytes % world_size != 0:
            raise ValueError(
                f"all_to_all_single input bytes ({input_.nbytes}) must be divisible by group size ({world_size})"
            )
        input_contig = input_.contiguous()
        if output.is_contiguous():
            output_contig = output
            copy_back = False
        else:
            output_contig = torch.empty_like(output, memory_format=torch.contiguous_format)
            copy_back = True

        def submit(runtime: Any) -> None:
            runtime.all2all(
                input_contig,
                output_contig,
                gfc_group,
                slice_bytes=input_contig.nbytes // world_size,
            )

        _run_gfc_collective((input_contig, output_contig), submit)
        if copy_back:
            output.copy_(output_contig)
        return

    dist.all_to_all_single(output, input_, group=_torch_group(group))


@_dynamo_disable
def all_gather_into_tensor(output_tensor: torch.Tensor, input_: torch.Tensor, *, group: Any | None) -> None:
    if runtime_v2_uses_gfc():
        world_size = get_group_world_size(group)
        if world_size == 1:
            output_tensor.reshape(-1).copy_(input_.reshape(-1))
            return
        gfc_group = _require_gfc_group(group)
        input_contig = input_.contiguous()
        if output_tensor.is_contiguous():
            output_contig = output_tensor
            copy_back = False
        else:
            output_contig = torch.empty_like(output_tensor, memory_format=torch.contiguous_format)
            copy_back = True

        def submit(runtime: Any) -> None:
            runtime.all_gather(input_contig, output_contig, gfc_group)

        _run_gfc_collective((input_contig, output_contig), submit)
        if copy_back:
            output_tensor.copy_(output_contig)
        return

    dist.all_gather_into_tensor(output_tensor, input_, group=_torch_group(group))


@_dynamo_disable
def all_gather_tensor_list(tensor_list: list[torch.Tensor], input_: torch.Tensor, *, group: Any | None) -> None:
    if runtime_v2_uses_gfc():
        world_size = get_group_world_size(group)
        if len(tensor_list) != world_size:
            raise ValueError(f"all_gather tensor_list length {len(tensor_list)} != group size {world_size}")
        if world_size == 1:
            tensor_list[0].copy_(input_)
            return
        _require_gfc_group(group)
        output = torch.empty(
            (world_size, *tuple(input_.shape)),
            dtype=input_.dtype,
            device=input_.device,
        )
        all_gather_into_tensor(output, input_, group=group)
        for idx, tensor in enumerate(tensor_list):
            tensor.copy_(output[idx])
        return

    dist.all_gather(tensor_list, input_, group=_torch_group(group))


def shutdown_runtime_v2_collective_runtime() -> None:
    global _GFC_RUNTIME, _GFC_MIGRATE_RUNTIME, _COLLECTIVE_BACKEND
    if _GFC_MIGRATE_RUNTIME is not None:
        _GFC_MIGRATE_RUNTIME.shutdown()
        _GFC_MIGRATE_RUNTIME = None
    if _GFC_RUNTIME is not None:
        _GFC_RUNTIME.shutdown()
        _GFC_RUNTIME = None
    _GFC_GROUP_BY_PG_ID.clear()
    _COLLECTIVE_BACKEND = "torch"
