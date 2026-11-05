"""SymmetricCollectiveRuntime.

Phase 2 scope: bootstrap (init_process_group already done by caller),
``enable_symm_mem_for_group`` on the world PG, allocation of the four
symmetric regions, footprint logging, session-nonce broadcast, group pool
table on device. Collective methods (``barrier``, ``all_gather``, ``all2all``,
``p2p_put``, ``p2p_get``) are filled in by later phases.
"""

from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from gfc.autotune import AutotuneTable, Policy
from gfc.buffer import SymmRegions, allocate_symm_regions, zero_signal_grids
from gfc.config import SymmetricCollectiveConfig
from gfc.env import read_env_overrides
from gfc.group import GroupDescriptor, _EpochManager, compute_token, stable_hash64
from gfc.kernels._common import vec_width_bytes
from gfc.kernels.all2all import launch_all2all_pull
from gfc.kernels.all_gather import launch_all_gather_pull
from gfc.kernels.barrier import launch_barrier
from gfc.kernels.copy_engine import (
    launch_all2all_copy_engine,
    launch_all_gather_copy_engine,
    launch_p2p_copy_engine,
)
from gfc.kernels.copy_pull import launch_pull_copy
from gfc.kernels.fused import launch_all2all_fused, launch_all_gather_fused
from gfc.kernels.p2p_put import launch_p2p_put_copy
from gfc.kernels.tma_paths import (
    launch_all2all_tma,
    launch_all_gather_tma,
    launch_p2p_copy_tma,
)
from gfc.tma_probe import TMAUnsupportedError, probe_tma_supported
from gfc.logging import get_logger, log_footprint


_NONCE_STORE_KEY = "gfc/session_nonce"


class SymmetricCollectiveRuntime:
    """Single-stream runtime over a one-time world-group symmetric rendezvous.

    Owns one CUDA stream (``self.stream``); all collective submissions are
    serialized through ``self._submit_lock`` and enqueued onto it. The
    bootstrap process group is used only for rendezvous, the session-nonce
    broadcast, and debug-mode consistency checks — never on a collective
    data path.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        config: SymmetricCollectiveConfig,
        device: torch.device,
        world_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        env = read_env_overrides()
        if env.debug:
            config = _with_debug(config, True)
        if env.use_tma:
            config = _with_use_tma(config, True)
        if env.use_copy_engine:
            config = _with_use_copy_engine(config, True)
        if env.copy_sms is not None:
            config = _with_copy_sms(config, env.copy_sms)
        if env.timeout_ms is not None:
            config = _with_timeout(config, env.timeout_ms)
        if env.max_pipeline_chunks is not None:
            config = _replace(config, max_pipeline_chunks=env.max_pipeline_chunks)
        if env.pipeline_chunks is not None:
            config = _replace(config, pipeline_chunks=env.pipeline_chunks)
        if env.enable_fused_path is not None:
            config = _replace(config, enable_fused_path=env.enable_fused_path)
        if env.fused_num_channels is not None:
            config = _replace(config, fused_num_channels=env.fused_num_channels)
        if env.fused_chunk_size is not None:
            config = _replace(config, fused_chunk_size=env.fused_chunk_size)
        if env.autotune_config_path is not None:
            config = _replace(config, autotune_config_path=env.autotune_config_path)

        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"gfc requires a CUDA device, got {self.device!r}")

        assert dist.is_initialized(), (
            "torch.distributed must be initialised before SymmetricCollectiveRuntime"
        )
        self.world_pg = world_group if world_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.world_pg)
        self.world_size = dist.get_world_size(self.world_pg)

        # Hang-trace ring is opt-in via GFC_TRACE=1. _trace_record() does an
        # edge_seq.cpu().tolist() which is a device->host sync; leaving it on
        # the hot path defeats async pipelining. Default OFF in production.
        self._trace_enabled = os.environ.get("GFC_TRACE") == "1"

        # The data-plane stream. All collectives are enqueued here.
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        # Auxiliary stream used by the pipelined-staging path to overlap the
        # local input-> comm_buf copy of chunk k+1 with the cross-rank pull
        # of chunk k on ``self.stream``. Data buffer reuse is still ordered by
        # the main stream and the post barrier.
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._submit_lock = threading.Lock()

        # Autotune table: when present, every all_gather / all2all call
        # consults this for path + knob selection. Loading is once-only.
        self.autotune: AutotuneTable | None = None
        if config.autotune_config_path:
            self.autotune = AutotuneTable.from_json(config.autotune_config_path)

        # Persistent-kernel grid size matches the device SM count so each
        # SM owns one program that loops over (peer, tile) work items.
        self.num_sms = int(
            torch.cuda.get_device_properties(self.device).multi_processor_count
        )
        self.copy_sms = min(self.num_sms, int(config.copy_sms))
        self.copy_engine_enabled = bool(config.use_copy_engine)

        # Symmetric-memory enable + region allocation must run on every rank
        # in the world group concurrently. enable_symm_mem_for_group is a
        # collective; rendezvous is too.
        enable_symm_mem_for_group(self.world_pg.group_name)
        self.regions: SymmRegions = allocate_symm_regions(
            config, self.device, self.world_pg, self.world_size
        )

        # Per-rank local edge-sequence counter, indexed by *global* peer rank.
        # Drives the per-edge double-buffer slot inside the barrier kernel:
        # ``slot = edge_seq[peer] & 1``. Local-only; peers do not read it,
        # so it is not in symmetric memory. Lives on the device so that the
        # slot computation is fully GPU-resident and CUDA-graph safe — host
        # never participates in slot selection.
        self.edge_seq = torch.zeros(
            self.world_size, dtype=torch.uint64, device=self.device
        )

        # Session nonce: rank 0 generates, broadcast via TCPStore-backed
        # bootstrap PG using object collectives.
        self.session_nonce = self._broadcast_session_nonce()
        assert self.session_nonce != 0, "session_nonce must be non-zero"

        # Group registry and per-group epoch manager.
        self._registered: dict[int, GroupDescriptor] = {}
        self._epochs = _EpochManager()
        self._free_pool_rows = list(reversed(range(config.max_groups)))

        # Device-resident table of ordered rank lists per group pool row.
        # Row r is the rank list of whatever group occupies pool row r.
        # Sentinel: unused entries are 0xFFFFFFFF (== UINT32_MAX).
        self._group_ranks_pool = torch.full(
            (config.max_groups, config.max_group_size),
            0xFFFFFFFF,
            dtype=torch.uint32,
            device=self.device,
        )

        # Optional TMA path. If use_tma is set, probe must pass — no silent
        # fallback. If it fails we raise; users can flip use_tma=False to
        # keep going on the vec path.
        self.tma_enabled = False
        if config.use_tma and self.copy_engine_enabled:
            raise ValueError("use_tma and use_copy_engine are mutually exclusive")

        if config.use_tma:
            if probe_tma_supported(self):
                self.tma_enabled = True
                get_logger().info(
                    "TMA enabled (peer-pointer load verified)"
                )
            else:
                raise TMAUnsupportedError(
                    "TMA probe failed on this hardware/driver; "
                    "set use_tma=False to use the vec fallback"
                )
        else:
            get_logger().info("TMA disabled (use_tma=False)")
        if self.copy_engine_enabled:
            get_logger().info("copy-engine path enabled (cudaMemcpyAsync)")

        log = get_logger()
        if self.rank == 0:
            log_footprint(
                max_collective_bytes=config.max_collective_bytes,
                num_signal_slots=config.num_signal_slots,
                world_size=self.world_size,
                enable_debug_checks=config.enable_debug_checks,
                fused_num_channels=config.fused_num_channels,
                fused_max_chunks_per_channel=config.fused_max_chunks_per_channel,
            )
        log.info(
            "rank=%d world=%d device=%s symm_mem enabled (group=%s)",
            self.rank,
            self.world_size,
            self.device,
            self.world_pg.group_name,
        )

    # ----------------------------------------------------------------- nonce
    def _broadcast_session_nonce(self) -> int:
        """Rank 0 picks a non-zero u64 nonce; broadcast over bootstrap PG."""
        if self.rank == 0:
            nonce = secrets.randbits(64) | 1
        else:
            nonce = 0
        obj_list = [nonce]
        dist.broadcast_object_list(obj_list, src=0, group=self.world_pg)
        n = int(obj_list[0])
        if self.config.enable_debug_checks:
            # Cross-check that everyone agrees on the broadcast value.
            gathered: list[Optional[int]] = [None] * self.world_size
            dist.all_gather_object(gathered, n, group=self.world_pg)
            assert all(g == n for g in gathered), (
                f"session_nonce mismatch across ranks: {gathered}"
            )
        return n

    # ------------------------------------------------- group registration
    def register_group(
        self,
        ranks: Sequence[int],
        group_id: Optional[int] = None,
    ) -> GroupDescriptor:
        rt = tuple(int(r) for r in ranks)
        assert 0 < len(rt) <= self.config.max_group_size, (
            f"|group|={len(rt)} must be in (0, {self.config.max_group_size}]"
        )
        assert len(set(rt)) == len(rt), f"ranks must be unique: {rt}"
        assert all(0 <= r < self.world_size for r in rt), (
            f"ranks out of [0,{self.world_size}): {rt}"
        )

        gid = (
            group_id
            if group_id is not None
            else stable_hash64(b"gfc-group-v1", rt)
        )

        if gid in self._registered:
            return self._registered[gid]

        assert len(self._registered) < self.config.max_groups, "max_groups exceeded"
        pool_row = self._allocate_pool_row(rt)
        local_index = rt.index(self.rank) if self.rank in rt else -1
        desc = GroupDescriptor(
            group_id=gid,
            ranks=rt,
            local_index=local_index,
            size=len(rt),
            pool_row=pool_row,
        )
        self._registered[gid] = desc

        if self.config.enable_debug_checks:
            self._dbg_verify_group_consistency(gid, rt)

        return desc

    def _allocate_pool_row(self, ranks: tuple[int, ...]) -> int:
        row = self._free_pool_rows.pop()
        # Write the ordered rank list into row `row` and pad the tail with sentinel.
        row_t = torch.full(
            (self.config.max_group_size,),
            0xFFFFFFFF,
            dtype=torch.uint32,
            device=self.device,
        )
        row_t[: len(ranks)] = torch.tensor(ranks, dtype=torch.uint32, device=self.device)
        self._group_ranks_pool[row].copy_(row_t)
        return row

    def _dbg_verify_group_consistency(
        self, gid: int, ranks: tuple[int, ...]
    ) -> None:
        gathered: list[Optional[tuple[int, tuple[int, ...]]]] = [None] * self.world_size
        dist.all_gather_object(gathered, (gid, ranks), group=self.world_pg)
        ref = (gid, ranks)
        for r, g in enumerate(gathered):
            if g != ref:
                raise RuntimeError(
                    f"register_group consistency check failed: "
                    f"rank {r} sees {g}, local sees {ref}"
                )

    # ----------------------------------------------------------- group internals
    def _group_ranks_row_ptr(self, desc: GroupDescriptor) -> int:
        """Device address of row ``desc.pool_row`` of ``group_ranks_pool``."""
        row_tensor = self._group_ranks_pool[desc.pool_row]
        return int(row_tensor.data_ptr())

    # --------------------------------------------------------------- collectives
    def barrier(
        self,
        group: GroupDescriptor,
    ) -> None:
        """Bare release/acquire-sys barrier across ``group``.

        Consumes one epoch for ``group.group_id`` (``+= 1``). Synchronisation
        is per-edge double-buffered: for every peer ``p`` in ``group``, the
        kernel selects slot ``edge_seq[p] & 1`` and writes/reads that slot
        through ``signal_buf``.

        Ordering constraint — read this before driving GFC from your own
        scheduler
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Within any **rank pair** (p, q), the sequence of barriers that
        involve both p and q must be identical on the two endpoints. The
        protocol uses ``edge_seq[peer] & 1`` to pick the double-buffer slot
        and assumes both ends agree on which slot belongs to which barrier;
        if the two ranks issue barriers in different orders, slot indices
        diverge and the pair deadlocks (rank p waits in slot 0 while rank q
        writes slot 1, or two pair-involving barriers race for the same
        slot and one overwrites the other).

        Concretely: if rank 0 issues ``barrier(G_AB)`` then ``barrier(G_AC)``,
        every other rank that is in either group **must** see the same
        relative order of those two barriers when restricted to the pairs
        they participate in. A central single-threaded driver (one
        ``submit_lock`` per rank, broadcasting the same group schedule)
        trivially satisfies this. Per-rank independent decisions do not —
        if rank 1 and rank 2 might disagree on whether ``G_12`` or some
        other pair-involving group runs first, the protocol is unsafe.
        """
        assert group.local_index >= 0, "this rank is not in group"
        assert group.size <= self.config.max_group_size

        with self._submit_lock:
            epoch = self._epochs.next_for_barrier(group.group_id)
            token = compute_token(self.session_nonce, group.group_id, epoch)
            with torch.cuda.stream(self.stream):
                launch_barrier(
                    token=token,
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=self._group_ranks_row_ptr(group),
                    signal_ptrs_dev=self.signal_buf_ptrs_dev,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )

    # -------------------------------------------------------------- all_gather
    def all_gather(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        *,
        pipeline_chunks: Optional[int] = None,
    ) -> None:
        """Pull-model all_gather across ``group``.

        ``output`` must hold ``group.size * input_.nbytes`` bytes. After
        return, ``output[i * input_.nbytes : (i+1) * input_.nbytes]`` contains
        rank ``group.ranks[i]``'s input.

        ``pipeline_chunks`` overrides ``config.pipeline_chunks`` for this
        call. When effectively > 1 (and on the vec data path), the local
        stage of chunk ``k+1`` runs on ``copy_stream`` while the remote pull
        of chunk ``k`` runs on the main stream.
        """
        _validate_collective_args(
            input_, output, group, self.device, self.config,
            kind="all_gather",
        )
        assert input_.nbytes == output.nbytes // group.size, (
            f"output size {output.nbytes} != group.size * input.nbytes "
            f"{group.size * input_.nbytes}"
        )

        slice_bytes = input_.nbytes
        chunks = self._effective_pipeline_chunks(slice_bytes, pipeline_chunks)
        policy = None
        if self.autotune is not None:
            policy = self.autotune.policy_for("allgather", group.size, slice_bytes)

        with self._submit_lock:
            comm_buf_offset = 0
            row_ptr = self._group_ranks_row_ptr(group)
            sig_ptrs = self.signal_buf_ptrs_dev
            cb_ptrs = self.comm_buf_ptrs_dev

            if policy is not None:
                post_e, post_tok = self._dispatch_all_gather(
                    policy=policy,
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                )
            elif self.config.enable_fused_path:
                # Fused single-kernel path. One pre-barrier (start line), then
                # a persistent kernel that stages + publishes step counters +
                # pulls all chunks. No post-barrier kernel — last chunk's
                # step token serves as completion.
                pre_e = self._epochs.next_for_barrier(group.group_id)
                step_base_e = self._epochs.next_for_barrier(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                step_base = compute_token(
                    self.session_nonce, group.group_id, step_base_e
                )
                post_e = step_base_e
                post_tok = step_base

                with torch.cuda.stream(self.stream):
                    launch_barrier(
                        token=pre_tok,
                        self_global_rank=self.rank,
                        group_ranks_row_ptr=row_ptr,
                        signal_ptrs_dev=sig_ptrs,
                        edge_seq_ptr=self.edge_seq_ptr,
                        group_size=group.size,
                        world_size=self.world_size,
                    )
                    launch_all_gather_fused(
                        input_ptr=int(input_.data_ptr()),
                        output_ptr=int(output.data_ptr()),
                        comm_buf_ptrs_dev=cb_ptrs,
                        step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                        group_ranks_row_ptr=row_ptr,
                        comm_buf_offset=comm_buf_offset,
                        self_global_rank=self.rank,
                        self_local_index=group.local_index,
                        slice_bytes=slice_bytes,
                        group_size=group.size,
                        world_size=self.world_size,
                        max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                        num_channels=self.config.fused_num_channels,
                        chunk_bytes=self.config.fused_chunk_size,
                        step_base=step_base,
                        alignment=self.config.alignment,
                    )
            elif chunks <= 1:
                pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

                with torch.cuda.stream(self.stream):
                    # 1. Stage local input into our own comm_buf.
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + slice_bytes
                    ]
                    local_buf.copy_(input_.view(torch.uint8).reshape(-1))

                    # 2. Pre-barrier: every peer must have its slice in place.
                    launch_barrier(
                        token=pre_tok,
                        self_global_rank=self.rank,
                        group_ranks_row_ptr=row_ptr,
                        signal_ptrs_dev=sig_ptrs,
                        edge_seq_ptr=self.edge_seq_ptr,
                        group_size=group.size,
                        world_size=self.world_size,
                    )

                    # 3. Pull copy from each peer into the output.
                    if self.copy_engine_enabled:
                        launch_all_gather_copy_engine(
                            peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks=group.ranks,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled:
                        launch_all_gather_tma(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(
                            self.regions.comm_buf.local_ptr,
                            int(output.data_ptr()),
                            slice_bytes,
                        )
                        launch_all_gather_pull(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )

                    # 4. Post-barrier: subsequent collectives must wait for
                    #    these pulls to complete before reusing comm_buf.
                    launch_barrier(
                        token=post_tok,
                        self_global_rank=self.rank,
                        group_ranks_row_ptr=row_ptr,
                        signal_ptrs_dev=sig_ptrs,
                        edge_seq_ptr=self.edge_seq_ptr,
                        group_size=group.size,
                        world_size=self.world_size,
                    )
            else:
                # Pipelined chunked path on the vec kernels.
                post_e = self._all_gather_pipelined(
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                    chunks=chunks,
                )
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

    # ----------------------------------------------------------------- all2all
    def all2all(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        *,
        slice_bytes: Optional[int] = None,
        pipeline_chunks: Optional[int] = None,
    ) -> None:
        """Pull-model all-to-all across ``group``.

        ``input_`` is laid out as ``group.size`` contiguous slices of equal
        ``slice_bytes`` bytes — slice ``t`` is destined for ``group.ranks[t]``.
        After return, ``output[i * slice_bytes : (i+1) * slice_bytes]`` holds
        slice ``self_local_index`` from peer ``group.ranks[i]``.

        ``pipeline_chunks`` overrides ``config.pipeline_chunks`` for this
        call. When effectively > 1 (and on the vec data path), the local
        stage of chunk ``k+1`` runs on ``copy_stream`` while the remote pull
        of chunk ``k`` runs on the main stream.
        """
        # TRACE: snapshot before submission. Opt-in via GFC_TRACE=1 — the
        # snapshot does an edge_seq.cpu().tolist() which is a device->host
        # sync that would defeat any caller's attempt to overlap host work
        # with the GFC stream. Default OFF in the hot path.
        if self._trace_enabled:
            try:
                self._trace_record("all2all_submit", group,
                                   {"slice_bytes": int(slice_bytes)})
            except Exception:
                pass
        _validate_collective_args(
            input_, output, group, self.device, self.config, kind="all2all"
        )
        assert input_.nbytes == output.nbytes, (
            f"all2all: input.nbytes={input_.nbytes} output.nbytes={output.nbytes}"
        )
        if slice_bytes is None:
            assert input_.nbytes % group.size == 0, (
                f"all2all: input.nbytes={input_.nbytes} not divisible by "
                f"group.size={group.size}"
            )
            slice_bytes = input_.nbytes // group.size
        assert slice_bytes > 0
        assert slice_bytes * group.size <= self.config.max_collective_bytes, (
            f"all2all: slice_bytes*group.size = {slice_bytes*group.size} > "
            f"max_collective_bytes {self.config.max_collective_bytes}"
        )

        chunks = self._effective_pipeline_chunks(slice_bytes, pipeline_chunks)
        policy = None
        if self.autotune is not None:
            policy = self.autotune.policy_for("all2all", group.size, slice_bytes)

        with self._submit_lock:
            comm_buf_offset = 0
            row_ptr = self._group_ranks_row_ptr(group)
            sig_ptrs = self.signal_buf_ptrs_dev
            cb_ptrs = self.comm_buf_ptrs_dev
            staged_bytes = slice_bytes * group.size

            if policy is not None:
                post_e, post_tok = self._dispatch_all2all(
                    policy=policy,
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                    staged_bytes=staged_bytes,
                )
            elif self.config.enable_fused_path:
                # Fused single-kernel all2all. See ``all_gather`` for the
                # protocol overview; the all2all variant partitions along the
                # per-peer-slice axis.
                pre_e = self._epochs.next_for_barrier(group.group_id)
                step_base_e = self._epochs.next_for_barrier(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                step_base = compute_token(
                    self.session_nonce, group.group_id, step_base_e
                )
                post_e = step_base_e
                post_tok = step_base

                with torch.cuda.stream(self.stream):
                    launch_barrier(
                        token=pre_tok,
                        self_global_rank=self.rank,
                        group_ranks_row_ptr=row_ptr,
                        signal_ptrs_dev=sig_ptrs,
                        edge_seq_ptr=self.edge_seq_ptr,
                        group_size=group.size,
                        world_size=self.world_size,
                    )
                    launch_all2all_fused(
                        input_ptr=int(input_.data_ptr()),
                        output_ptr=int(output.data_ptr()),
                        comm_buf_ptrs_dev=cb_ptrs,
                        step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                        group_ranks_row_ptr=row_ptr,
                        comm_buf_offset=comm_buf_offset,
                        self_global_rank=self.rank,
                        self_local_index=group.local_index,
                        slice_bytes=slice_bytes,
                        group_size=group.size,
                        world_size=self.world_size,
                        max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                        num_channels=self.config.fused_num_channels,
                        chunk_bytes=self.config.fused_chunk_size,
                        step_base=step_base,
                        alignment=self.config.alignment,
                    )
            elif chunks <= 1:
                pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

                with torch.cuda.stream(self.stream):
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + staged_bytes
                    ]
                    local_buf.copy_(input_.view(torch.uint8).reshape(-1)[:staged_bytes])

                    launch_barrier(
                        token=pre_tok,
                        self_global_rank=self.rank,
                        group_ranks_row_ptr=row_ptr,
                        signal_ptrs_dev=sig_ptrs,
                        edge_seq_ptr=self.edge_seq_ptr,
                        group_size=group.size,
                        world_size=self.world_size,
                    )

                    if self.copy_engine_enabled:
                        launch_all2all_copy_engine(
                            peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks=group.ranks,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled:
                        launch_all2all_tma(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(
                            self.regions.comm_buf.local_ptr,
                            int(output.data_ptr()),
                            slice_bytes,
                        )
                        launch_all2all_pull(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )

                    launch_barrier(
                        token=post_tok,
                        self_global_rank=self.rank,
                        group_ranks_row_ptr=row_ptr,
                        signal_ptrs_dev=sig_ptrs,
                        edge_seq_ptr=self.edge_seq_ptr,
                        group_size=group.size,
                        world_size=self.world_size,
                    )
            else:
                post_e = self._all2all_pipelined(
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                    chunks=chunks,
                )
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

    # ----------------------------------------------------------------- p2p
    def p2p_put(
        self,
        dst_rank: int,
        src: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Push protocol — sender side.

        Stages ``src`` into local ``comm_buf``, runs the pre-barrier
        on the 2-rank ordered group ``[self.rank, dst_rank]``, issues a remote
        copy from local comm_buf into ``dst_rank``'s comm_buf, then the
        post-barrier. The partner rank must call
        :meth:`p2p_put_recv` with mirrored args on the same submission ordinal.
        """
        assert dst_rank != self.rank, "p2p_put: dst_rank must differ from self.rank"
        self._p2p_protocol(
            src_rank=self.rank,
            dst_rank=dst_rank,
            local_tensor=src,
            role="sender",
            protocol="put",
            nbytes=nbytes,
        )

    def p2p_put_recv(
        self,
        src_rank: int,
        dst: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Push protocol — receiver side. Participates in the 2-rank barriers
        on ``[src_rank, self.rank]`` and copies the staged comm_buf into
        ``dst`` after the post-barrier."""
        assert src_rank != self.rank, "p2p_put_recv: src_rank must differ from self.rank"
        self._p2p_protocol(
            src_rank=src_rank,
            dst_rank=self.rank,
            local_tensor=dst,
            role="receiver",
            protocol="put",
            nbytes=nbytes,
        )

    def p2p_get(
        self,
        src_rank: int,
        dst: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Pull protocol — receiver side. Issues pre-barrier on
        ``[src_rank, self.rank]``, ``pull_copy_kernel`` from the peer's
        ``comm_buf`` into ``dst``, then post-barrier. The partner
        rank must call :meth:`p2p_get_serve` to stage and participate."""
        assert src_rank != self.rank, "p2p_get: src_rank must differ from self.rank"
        self._p2p_protocol(
            src_rank=src_rank,
            dst_rank=self.rank,
            local_tensor=dst,
            role="receiver",
            protocol="get",
            nbytes=nbytes,
        )

    def p2p_get_serve(
        self,
        dst_rank: int,
        src: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Pull protocol — sender side. Stages ``src`` into local
        ``comm_buf`` and participates in the 2-rank barriers so
        ``dst_rank``'s :meth:`p2p_get` can pull from it."""
        assert dst_rank != self.rank, "p2p_get_serve: dst_rank must differ from self.rank"
        self._p2p_protocol(
            src_rank=self.rank,
            dst_rank=dst_rank,
            local_tensor=src,
            role="sender",
            protocol="get",
            nbytes=nbytes,
        )

    # ------------------------------------------------------ p2p shared core
    def _p2p_protocol(
        self,
        *,
        src_rank: int,
        dst_rank: int,
        local_tensor: torch.Tensor,
        role: str,
        protocol: str,
        nbytes: Optional[int],
    ) -> None:
        assert local_tensor.is_cuda and local_tensor.device == self.device
        assert local_tensor.is_contiguous(), "p2p: tensor must be contiguous"
        if nbytes is None:
            nbytes = local_tensor.nbytes
        assert 0 < nbytes <= local_tensor.nbytes
        assert nbytes <= self.config.max_collective_bytes, (
            f"p2p: nbytes={nbytes} > max_collective_bytes={self.config.max_collective_bytes}"
        )

        group = self.register_group((src_rank, dst_rank))
        assert group.local_index >= 0  # this rank is in the 2-rank group

        with self._submit_lock:
            pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
            pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
            post_tok = compute_token(self.session_nonce, group.group_id, post_e)

            comm_buf_offset = 0
            row_ptr = self._group_ranks_row_ptr(group)
            sig_ptrs = self.signal_buf_ptrs_dev
            cb_local_ptr = self.regions.comm_buf.local_ptr + comm_buf_offset

            with torch.cuda.stream(self.stream):
                # Stage on the sender side.
                if role == "sender":
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + nbytes
                    ]
                    local_buf.copy_(local_tensor.view(torch.uint8).reshape(-1)[:nbytes])

                launch_barrier(
                    token=pre_tok,
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=row_ptr,
                    signal_ptrs_dev=sig_ptrs,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )

                if protocol == "put" and role == "sender":
                    # Remote write: src is local comm_buf, dst is peer comm_buf.
                    peer_comm_ptrs = self.regions.comm_buf.handle.buffer_ptrs
                    dst_ptr_remote = int(peer_comm_ptrs[dst_rank]) + comm_buf_offset
                    if self.copy_engine_enabled:
                        launch_p2p_copy_engine(
                            src_ptr=cb_local_ptr,
                            dst_ptr=dst_ptr_remote,
                            nbytes=nbytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled and nbytes % 16 == 0:
                        launch_p2p_copy_tma(
                            src_ptr=cb_local_ptr,
                            dst_ptr=dst_ptr_remote,
                            nbytes=nbytes,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(cb_local_ptr, dst_ptr_remote, nbytes)
                        launch_p2p_put_copy(
                            src_ptr=cb_local_ptr,
                            dst_ptr=dst_ptr_remote,
                            nbytes=nbytes,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )
                elif protocol == "get" and role == "receiver":
                    # Remote read: src is peer comm_buf, dst is local user tensor.
                    peer_comm_ptrs = self.regions.comm_buf.handle.buffer_ptrs
                    src_ptr_remote = int(peer_comm_ptrs[src_rank]) + comm_buf_offset
                    dst_ptr_local = int(local_tensor.data_ptr())
                    if self.copy_engine_enabled:
                        launch_p2p_copy_engine(
                            src_ptr=src_ptr_remote,
                            dst_ptr=dst_ptr_local,
                            nbytes=nbytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled and nbytes % 16 == 0 and dst_ptr_local % 16 == 0:
                        launch_p2p_copy_tma(
                            src_ptr=src_ptr_remote,
                            dst_ptr=dst_ptr_local,
                            nbytes=nbytes,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(src_ptr_remote, dst_ptr_local, nbytes)
                        launch_pull_copy(
                            src_ptr=src_ptr_remote,
                            dst_ptr=dst_ptr_local,
                            nbytes=nbytes,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )
                # else: this rank participates in barriers only.

                launch_barrier(
                    token=post_tok,
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=row_ptr,
                    signal_ptrs_dev=sig_ptrs,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )

                # Receiver in push protocol pulls from own comm_buf into dst tensor.
                if protocol == "put" and role == "receiver":
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + nbytes
                    ]
                    local_tensor.view(torch.uint8).reshape(-1)[:nbytes].copy_(local_buf)

    # -------------------------------------------------- autotune dispatch
    def _dispatch_all_gather(
        self,
        *,
        policy: Policy,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
    ) -> tuple[int, int]:
        """Route an all_gather call according to an autotune policy.

        Returns ``(post_epoch, post_token)`` for caller bookkeeping.
        The fused kernel itself handles any alignment (picks a vec width
        of 16/8/4/1 plus a byte-tail at runtime), so no eligibility check
        is needed here. ``pipelined`` falls back to ``vec_pull`` when
        the chunk count collapses to 1.
        """
        path = policy.path
        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            if chunks <= 1:
                path = "vec_pull"

        if path == "fused":
            num_channels = int(
                policy.knobs.get("fused_num_channels", self.config.fused_num_channels)
            )
            chunk_size = int(
                policy.knobs.get("fused_chunk_size", self.config.fused_chunk_size)
            )
            pre_e = self._epochs.next_for_barrier(group.group_id)
            step_base_e = self._epochs.next_for_barrier(group.group_id)
            pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
            step_base = compute_token(
                self.session_nonce, group.group_id, step_base_e
            )
            with torch.cuda.stream(self.stream):
                launch_barrier(
                    token=pre_tok,
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=row_ptr,
                    signal_ptrs_dev=sig_ptrs,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )
                launch_all_gather_fused(
                    input_ptr=int(input_.data_ptr()),
                    output_ptr=int(output.data_ptr()),
                    comm_buf_ptrs_dev=cb_ptrs,
                    step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_global_rank=self.rank,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    world_size=self.world_size,
                    max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                    num_channels=num_channels,
                    chunk_bytes=chunk_size,
                    step_base=step_base,
                    alignment=self.config.alignment,
                )
            return step_base_e, step_base

        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            post_e = self._all_gather_pipelined(
                input_=input_,
                output=output,
                group=group,
                comm_buf_offset=comm_buf_offset,
                row_ptr=row_ptr,
                sig_ptrs=sig_ptrs,
                cb_ptrs=cb_ptrs,
                slice_bytes=slice_bytes,
                chunks=chunks,
            )
            post_tok = compute_token(self.session_nonce, group.group_id, post_e)
            return post_e, post_tok

        # One-shot stage + barrier + body + barrier paths.
        pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
        pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)
        with torch.cuda.stream(self.stream):
            local_buf = self.regions.comm_buf.tensor[
                comm_buf_offset : comm_buf_offset + slice_bytes
            ]
            local_buf.copy_(input_.view(torch.uint8).reshape(-1))
            launch_barrier(
                token=pre_tok,
                self_global_rank=self.rank,
                group_ranks_row_ptr=row_ptr,
                signal_ptrs_dev=sig_ptrs,
                edge_seq_ptr=self.edge_seq_ptr,
                group_size=group.size,
                world_size=self.world_size,
            )
            if path == "copy_engine":
                launch_all_gather_copy_engine(
                    peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks=group.ranks,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    stream=self.stream,
                )
            elif path == "tma":
                launch_all_gather_tma(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    num_sms=self.num_sms,
                )
            else:
                copy_sms = int(policy.knobs.get("copy_sms", self.copy_sms))
                vec = _vec_for(
                    self.regions.comm_buf.local_ptr,
                    int(output.data_ptr()),
                    slice_bytes,
                )
                launch_all_gather_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=copy_sms,
                )
            launch_barrier(
                token=post_tok,
                self_global_rank=self.rank,
                group_ranks_row_ptr=row_ptr,
                signal_ptrs_dev=sig_ptrs,
                edge_seq_ptr=self.edge_seq_ptr,
                group_size=group.size,
                world_size=self.world_size,
            )
        return post_e, post_tok

    def _dispatch_all2all(
        self,
        *,
        policy: Policy,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
        staged_bytes: int,
    ) -> tuple[int, int]:
        path = policy.path
        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            if chunks <= 1:
                path = "vec_pull"

        if path == "fused":
            num_channels = int(
                policy.knobs.get("fused_num_channels", self.config.fused_num_channels)
            )
            chunk_size = int(
                policy.knobs.get("fused_chunk_size", self.config.fused_chunk_size)
            )
            pre_e = self._epochs.next_for_barrier(group.group_id)
            step_base_e = self._epochs.next_for_barrier(group.group_id)
            pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
            step_base = compute_token(
                self.session_nonce, group.group_id, step_base_e
            )
            with torch.cuda.stream(self.stream):
                launch_barrier(
                    token=pre_tok,
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=row_ptr,
                    signal_ptrs_dev=sig_ptrs,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )
                launch_all2all_fused(
                    input_ptr=int(input_.data_ptr()),
                    output_ptr=int(output.data_ptr()),
                    comm_buf_ptrs_dev=cb_ptrs,
                    step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_global_rank=self.rank,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    world_size=self.world_size,
                    max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                    num_channels=num_channels,
                    chunk_bytes=chunk_size,
                    step_base=step_base,
                    alignment=self.config.alignment,
                )
            return step_base_e, step_base

        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            post_e = self._all2all_pipelined(
                input_=input_,
                output=output,
                group=group,
                comm_buf_offset=comm_buf_offset,
                row_ptr=row_ptr,
                sig_ptrs=sig_ptrs,
                cb_ptrs=cb_ptrs,
                slice_bytes=slice_bytes,
                chunks=chunks,
            )
            post_tok = compute_token(self.session_nonce, group.group_id, post_e)
            return post_e, post_tok

        pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
        pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)
        with torch.cuda.stream(self.stream):
            local_buf = self.regions.comm_buf.tensor[
                comm_buf_offset : comm_buf_offset + staged_bytes
            ]
            local_buf.copy_(
                input_.view(torch.uint8).reshape(-1)[:staged_bytes]
            )
            launch_barrier(
                token=pre_tok,
                self_global_rank=self.rank,
                group_ranks_row_ptr=row_ptr,
                signal_ptrs_dev=sig_ptrs,
                edge_seq_ptr=self.edge_seq_ptr,
                group_size=group.size,
                world_size=self.world_size,
            )
            if path == "copy_engine":
                launch_all2all_copy_engine(
                    peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks=group.ranks,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    stream=self.stream,
                )
            elif path == "tma":
                launch_all2all_tma(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    num_sms=self.num_sms,
                )
            else:
                copy_sms = int(policy.knobs.get("copy_sms", self.copy_sms))
                vec = _vec_for(
                    self.regions.comm_buf.local_ptr,
                    int(output.data_ptr()),
                    slice_bytes,
                )
                launch_all2all_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=copy_sms,
                )
            launch_barrier(
                token=post_tok,
                self_global_rank=self.rank,
                group_ranks_row_ptr=row_ptr,
                signal_ptrs_dev=sig_ptrs,
                edge_seq_ptr=self.edge_seq_ptr,
                group_size=group.size,
                world_size=self.world_size,
            )
        return post_e, post_tok

    # ------------------------------------------------- pipelined helpers
    def _effective_pipeline_chunks(
        self, axis_bytes: int, override: Optional[int]
    ) -> int:
        """Resolve the pipeline chunk count for a collective.

        ``axis_bytes`` is the per-peer chunkable axis: ``input.nbytes`` for
        all_gather and ``slice_bytes`` (= input.nbytes / group.size) for
        all2all. Returns 1 when pipelining is disabled, when the data path
        does not yet support chunking (TMA / copy-engine), or when
        ``axis_bytes`` is too small for the configured minimum chunk size.
        """
        if self.tma_enabled or self.copy_engine_enabled:
            return 1
        target = override if override is not None else self.config.pipeline_chunks
        target = min(int(target), self.config.max_pipeline_chunks)
        if target <= 1:
            return 1
        min_chunk = self.config.pipeline_min_chunk_bytes
        if axis_bytes < 2 * min_chunk:
            return 1
        # Cap chunks so each chunk has at least ``min_chunk`` bytes.
        max_by_size = max(1, axis_bytes // min_chunk)
        return max(1, min(target, max_by_size))

    @staticmethod
    def _chunk_layout(axis_bytes: int, chunks: int, align: int = 16) -> list[tuple[int, int]]:
        """Return ``[(offset, size)]`` covering ``axis_bytes`` in up to
        ``chunks`` aligned slices.

        The first ``chunks - 1`` chunks are aligned up to ``align`` bytes;
        the last chunk absorbs the remainder. This keeps every per-peer
        sub-buffer 16B-aligned so the vec kernel keeps its widest store.
        """
        assert chunks >= 1
        if chunks == 1 or axis_bytes <= 0:
            return [(0, axis_bytes)]
        nominal = (axis_bytes + chunks - 1) // chunks
        nominal = (nominal + align - 1) & ~(align - 1)
        layout: list[tuple[int, int]] = []
        off = 0
        while off < axis_bytes and len(layout) < chunks:
            size = min(nominal, axis_bytes - off)
            layout.append((off, size))
            off += size
        # If alignment rounding caused us to skip some bytes, absorb the
        # remainder into the last chunk.
        if off < axis_bytes:
            last_off, last_size = layout[-1]
            layout[-1] = (last_off, last_size + (axis_bytes - off))
        return layout

    def _all_gather_pipelined(
        self,
        *,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
        chunks: int,
    ) -> int:
        """Chunked all_gather. Returns the post-barrier epoch."""
        layout = self._chunk_layout(slice_bytes, chunks, align=self.config.alignment)
        actual_chunks = len(layout)
        max_chunks = self.config.max_pipeline_chunks
        assert actual_chunks <= max_chunks, (
            f"chunked all_gather: actual_chunks={actual_chunks} > max={max_chunks}"
        )

        # Reserve one epoch per pre-barrier chunk + one for the post.
        pre_epochs = [
            self._epochs.next_for_barrier(group.group_id) for _ in range(actual_chunks)
        ]
        post_e = self._epochs.next_for_barrier(group.group_id)
        pre_toks = [
            compute_token(self.session_nonce, group.group_id, e) for e in pre_epochs
        ]
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)

        # Fork copy_stream off main_stream. Two purposes in one event:
        #   1. Under CUDA-graph capture, this brings copy_stream into the
        #      active capture (a side stream is only captured once it waits
        #      on an event from the capture stream).
        #   2. The prior pulls and post-barrier are already in main_stream's
        #      history, so waiting on this event blocks staging until that
        #      prior work is done.
        fork_ev = self.stream.record_event()
        self.copy_stream.wait_event(fork_ev)

        src_u8 = input_.view(torch.uint8).reshape(-1)
        cb_local = self.regions.comm_buf.tensor
        output_ptr = int(output.data_ptr())
        dst_stride = slice_bytes  # peer-stride in output is full slice_bytes
        comm_buf_local_ptr = self.regions.comm_buf.local_ptr

        for k, (offset, size) in enumerate(layout):
            # Stage chunk k on copy_stream.
            with torch.cuda.stream(self.copy_stream):
                dst_buf = cb_local[
                    comm_buf_offset + offset : comm_buf_offset + offset + size
                ]
                dst_buf.copy_(src_u8[offset : offset + size])
            stage_ev = self.copy_stream.record_event()

            with torch.cuda.stream(self.stream):
                self.stream.wait_event(stage_ev)
                launch_barrier(
                    token=pre_toks[k],
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=row_ptr,
                    signal_ptrs_dev=sig_ptrs,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )

                vec = _vec_for(
                    comm_buf_local_ptr + offset,
                    output_ptr + offset,
                    size,
                )
                launch_all_gather_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=output_ptr + offset,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset + offset,
                    self_local_index=group.local_index,
                    slice_bytes=size,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=self.copy_sms,
                    dst_stride_bytes=dst_stride,
                )

        with torch.cuda.stream(self.stream):
            launch_barrier(
                token=post_tok,
                self_global_rank=self.rank,
                group_ranks_row_ptr=row_ptr,
                signal_ptrs_dev=sig_ptrs,
                edge_seq_ptr=self.edge_seq_ptr,
                group_size=group.size,
                world_size=self.world_size,
            )

        return post_e

    def _all2all_pipelined(
        self,
        *,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
        chunks: int,
    ) -> int:
        """Chunked all2all. Returns the post-barrier epoch.

        For all2all every chunk is a sub-slice of size ``chunk_bytes`` within
        each per-peer slice. The local stage of chunk k copies a single
        contiguous run of ``group.size * chunk_bytes`` bytes by issuing one
        copy per peer slice to the comm_buf at offset
        ``comm_buf_offset + peer_idx * slice_bytes + chunk_offset``.
        """
        layout = self._chunk_layout(slice_bytes, chunks, align=self.config.alignment)
        actual_chunks = len(layout)
        max_chunks = self.config.max_pipeline_chunks
        assert actual_chunks <= max_chunks, (
            f"chunked all2all: actual_chunks={actual_chunks} > max={max_chunks}"
        )

        pre_epochs = [
            self._epochs.next_for_barrier(group.group_id) for _ in range(actual_chunks)
        ]
        post_e = self._epochs.next_for_barrier(group.group_id)
        pre_toks = [
            compute_token(self.session_nonce, group.group_id, e) for e in pre_epochs
        ]
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)

        fork_ev = self.stream.record_event()
        self.copy_stream.wait_event(fork_ev)

        src_u8 = input_.view(torch.uint8).reshape(-1)
        cb_local = self.regions.comm_buf.tensor
        output_ptr = int(output.data_ptr())
        comm_buf_local_ptr = self.regions.comm_buf.local_ptr
        group_size = group.size

        for k, (offset, size) in enumerate(layout):
            # Stage chunk k on copy_stream: copy the chunk-bytes destined for
            # each peer from input[peer*slice_bytes + offset : ...] into
            # comm_buf[peer*slice_bytes + offset : ...]. The runs
            # within input and within comm_buf are not contiguous across
            # peers, so we issue ``group_size`` copies per chunk.
            with torch.cuda.stream(self.copy_stream):
                for peer_idx in range(group_size):
                    in_off = peer_idx * slice_bytes + offset
                    cb_off = comm_buf_offset + peer_idx * slice_bytes + offset
                    cb_local[cb_off : cb_off + size].copy_(
                        src_u8[in_off : in_off + size]
                    )
            stage_ev = self.copy_stream.record_event()

            with torch.cuda.stream(self.stream):
                self.stream.wait_event(stage_ev)
                launch_barrier(
                    token=pre_toks[k],
                    self_global_rank=self.rank,
                    group_ranks_row_ptr=row_ptr,
                    signal_ptrs_dev=sig_ptrs,
                    edge_seq_ptr=self.edge_seq_ptr,
                    group_size=group.size,
                    world_size=self.world_size,
                )

                vec = _vec_for(
                    comm_buf_local_ptr
                    + comm_buf_offset
                    + group.local_index * slice_bytes
                    + offset,
                    output_ptr + offset,
                    size,
                )
                launch_all2all_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=output_ptr + offset,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset + offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=self.copy_sms,
                    copy_bytes=size,
                    src_slice_stride=slice_bytes,
                    dst_stride_bytes=slice_bytes,
                )

        with torch.cuda.stream(self.stream):
            launch_barrier(
                token=post_tok,
                self_global_rank=self.rank,
                group_ranks_row_ptr=row_ptr,
                signal_ptrs_dev=sig_ptrs,
                edge_seq_ptr=self.edge_seq_ptr,
                group_size=group.size,
                world_size=self.world_size,
            )

        return post_e

    # --------------------------------------------- external stream integration
    def wait_for_external(self, event: torch.cuda.Event) -> None:
        """Insert a one-way edge: ``self.stream`` waits on ``event``."""
        self.stream.wait_event(event)

    def record_event(self, event: Optional[torch.cuda.Event] = None) -> torch.cuda.Event:
        """Record a completion event on ``self.stream`` and return it."""
        ev = event if event is not None else torch.cuda.Event()
        ev.record(self.stream)
        return ev


    # --- TRACE INSTRUMENTATION (added by trace_patch) -----------------------
    def _trace_record(self, op: str, group, extra: dict | None = None) -> None:
        if not hasattr(self, "_trace_ring"):
            import collections
            self._trace_ring = collections.deque(maxlen=8192)
            self._trace_seq = 0
        self._trace_seq += 1
        # Snapshot edge_seq for the peers in this group only (cheap, ~8 ints).
        edge_seq_host = self.edge_seq.cpu().tolist()
        entry = {
            "n": self._trace_seq,
            "rank": self.rank,
            "op": op,
            "group_id_hex": hex(group.group_id),
            "group_ranks": list(group.ranks),
            "edge_seq_for_peers": {
                int(p): int(edge_seq_host[p]) for p in group.ranks if p != self.rank
            },
            "slots_for_peers": {
                int(p): int(edge_seq_host[p]) & 1
                for p in group.ranks if p != self.rank
            },
        }
        if extra:
            entry.update(extra)
        self._trace_ring.append(entry)

    def dump_trace(self, suffix: str = "hang") -> str:
        import json
        out = Path(f"/tmp/gfc_trace_rank{self.rank}.{suffix}.json")
        data = list(self._trace_ring) if hasattr(self, "_trace_ring") else []
        # Snapshot signal_buf contents for postmortem cell inspection.
        try:
            sig = self.regions.signal_buf.tensor.cpu().tolist()
        except Exception as exc:
            sig = f"<failed: {exc!r}>"
        edge_seq_host = self.edge_seq.cpu().tolist()
        out.write_text(json.dumps({
            "rank": self.rank,
            "world_size": self.world_size,
            "edge_seq": edge_seq_host,
            "signal_buf_flat": sig,
            "trace_count": len(data),
            "trace": data,
        }, indent=2, default=str))
        return str(out)
    # --- END TRACE INSTRUMENTATION ------------------------------------------

    # -------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        """Drain, zero the signal/debug grids, world-barrier.

        After ``shutdown`` the runtime object is no longer usable for
        collectives. Repeated shutdown is a no-op.
        """
        if getattr(self, "_shutdown", False):
            return
        torch.cuda.current_stream(self.device).synchronize()
        self.stream.synchronize()
        self.copy_stream.synchronize()
        zero_signal_grids(self.regions)
        self.edge_seq.zero_()
        dist.barrier(group=self.world_pg)
        self._shutdown = True

    # -------------------------------------------------------------- accessors
    @property
    def comm_buf_ptrs_dev(self) -> int:
        return self.regions.comm_buf.buffer_ptrs_dev

    @property
    def signal_buf_ptrs_dev(self) -> int:
        return self.regions.signal_buf.buffer_ptrs_dev

    @property
    def edge_seq_ptr(self) -> int:
        return int(self.edge_seq.data_ptr())

    @property
    def step_pad_ptrs_dev(self) -> int:
        return self.regions.step_pad.buffer_ptrs_dev


# --------------------------------------------------------------------- helpers
def _validate_collective_args(
    input_: torch.Tensor,
    output: torch.Tensor,
    group: GroupDescriptor,
    device: torch.device,
    config: SymmetricCollectiveConfig,
    *,
    kind: str,
) -> None:
    assert group.local_index >= 0, f"this rank is not a member of group {group.ranks}"
    assert group.size <= config.max_group_size
    assert input_.is_cuda and output.is_cuda, f"{kind}: tensors must be on CUDA"
    assert input_.device == device, f"{kind}: input on {input_.device}, runtime on {device}"
    assert output.device == device, f"{kind}: output on {output.device}, runtime on {device}"
    assert input_.is_contiguous(), f"{kind}: input must be contiguous"
    assert output.is_contiguous(), f"{kind}: output must be contiguous"
    if kind == "all_gather":
        assert input_.nbytes <= config.max_collective_bytes, (
            f"{kind}: input.nbytes {input_.nbytes} > max_collective_bytes "
            f"{config.max_collective_bytes}"
        )
    elif kind == "all2all":
        assert output.nbytes <= config.max_collective_bytes, (
            f"{kind}: output.nbytes {output.nbytes} > max_collective_bytes "
            f"{config.max_collective_bytes}"
        )


def _vec_for(*ptrs_and_size: int) -> int:
    """Pick a vector width given pointer addresses and a payload size."""
    aligns = []
    for v in ptrs_and_size[:-1]:
        aligns.append(v & -v if v else 1 << 62)
    aligns.append(int(ptrs_and_size[-1]))
    return vec_width_bytes(*aligns)


def _with_debug(c: SymmetricCollectiveConfig, v: bool) -> SymmetricCollectiveConfig:
    return _replace(c, enable_debug_checks=v)


def _with_use_tma(c: SymmetricCollectiveConfig, v: bool) -> SymmetricCollectiveConfig:
    return _replace(c, use_tma=v)


def _with_use_copy_engine(c: SymmetricCollectiveConfig, v: bool) -> SymmetricCollectiveConfig:
    return _replace(c, use_copy_engine=v)


def _with_copy_sms(c: SymmetricCollectiveConfig, v: int) -> SymmetricCollectiveConfig:
    return _replace(c, copy_sms=v)


def _with_timeout(c: SymmetricCollectiveConfig, v: int) -> SymmetricCollectiveConfig:
    return _replace(c, timeout_ms=v)


def _replace(c: SymmetricCollectiveConfig, **fields) -> SymmetricCollectiveConfig:
    """``dataclasses.replace`` but without re-importing it everywhere."""
    import dataclasses

    return dataclasses.replace(c, **fields)


# ---------------------------------------------------------------- conveniences
def init_distributed_for_runtime() -> torch.device:
    """Initialize the bootstrap NCCL process group from torchrun env vars.

    Returns the CUDA device for this rank (``cuda:LOCAL_RANK``).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return device
