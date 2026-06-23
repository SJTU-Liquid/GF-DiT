# SPDX-License-Identifier: Apache-2.0

"""Real NCCL P2P data-plane test for runtime_v2 RESHARD.

Spawns 2 worker processes with a real torch.distributed NCCL world group,
constructs a minimal `_WorkerProcessRuntime` on each rank (skipping the
diffusion pipeline init), then invokes the actual `_execute_migration_transfer`
code path with a hand-built `ArtifactMigrationSchedule` of one tensor field.

The dst rank must end up with a new entry in its `local_artifacts` containing
the tensor exactly as it lived on the src rank.

Skipped on machines without at least 2 CUDA devices. There is no mock and no
gloo fallback; if NCCL P2P is unavailable the test fails loudly.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.runtime_v2.adapters.common import (
    output_artifact_layout,
    replicated_tensor_field,
)
from vllm_omni.diffusion.runtime_v2.codec import ArtifactLayoutCodec
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactMigrationSchedule,
    ArtifactValue,
    ExecutionGroupSpec,
    MigrateArtifactSpec,
    MigrateArtifactsPlan,
    OutputArtifactLayout,
    ParallelSpec,
    TaskKind,
    TensorMigrationEdge,
    TensorSliceSpec,
)


pytestmark = [pytest.mark.diffusion, pytest.mark.gpu]


@dataclass
class _MiniState:
    note: str
    device: torch.device | None = None
    payload: torch.Tensor | None = None


_MINI_CODEC_ID = "test.disaggregate.mini.v1"


class _MiniCodec(ArtifactLayoutCodec):
    @property
    def codec_id(self) -> str:
        return _MINI_CODEC_ID

    def describe_output(
        self,
        *,
        group: ExecutionGroupSpec,
        group_rank: int,
        request_metadata: Mapping[str, Any],
        artifact: ArtifactHandle,
    ) -> OutputArtifactLayout:
        # One replicated 4-element float32 tensor field, single-attribute path.
        return output_artifact_layout(
            artifact,
            (replicated_tensor_field("payload", (4,), torch.float32),),
        )

    def pack_metadata(self, *, value, layout):
        # Mirror Wan/Qwen: strip rank-local fields here so dst rank rebuilds.
        import dataclasses
        replacements = {f.field_path[0]: None for f in layout.tensors}
        replacements["device"] = None
        skeleton = dataclasses.replace(value, **replacements)
        import pickle as _pickle
        return _pickle.dumps(skeleton, protocol=_pickle.HIGHEST_PROTOCOL)

    def assemble(self, *, metadata_bytes, tensors, layout, device=None):
        if device is None:
            raise ValueError("_MiniCodec.assemble requires a device kwarg")
        state = super().assemble(
            metadata_bytes=metadata_bytes, tensors=tensors, layout=layout, device=device,
        )
        state.device = torch.device(device) if not isinstance(device, torch.device) else device
        return state


def _build_runtime_for_test(worker_rank: int):
    """Construct a minimal _WorkerProcessRuntime instance without loading any
    diffusion pipeline. We only exercise codec_registry, local_artifacts and the
    NCCL P2P data-plane code path.
    """
    from unittest.mock import MagicMock
    from multiprocessing import Pipe

    from vllm_omni.diffusion.runtime_v2.multiproc_worker import _WorkerProcessRuntime

    cmd_r, _cmd_w = Pipe(duplex=False)
    _evt_r, evt_w = Pipe(duplex=False)
    _res_r, res_w = Pipe(duplex=False)

    od_config = MagicMock()
    od_config.num_gpus = 2
    od_config.master_port = 29610

    runtime = _WorkerProcessRuntime(
        worker_rank=worker_rank,
        device_id=worker_rank,
        od_config=od_config,
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        dist_rank=worker_rank,
        local_rank=worker_rank,
        world_size=2,
        master_port=29610,
        group_id="g_src" if worker_rank == 0 else "g_dst",
        command_pipe_r=cmd_r,
        event_pipe_w=evt_w,
        result_pipe_w=res_w,
    )
    runtime.codec_registry.register(_MiniCodec())
    return runtime


def _build_schedule(*, request_id: str, src_rank: int, dst_rank: int, dst_artifact_id: str) -> ArtifactMigrationSchedule:
    # Schedule edges always carry the destination artifact id; the source-side
    # rename is resolved through MigrateArtifactSpec.effective_source_handle.
    edge = TensorMigrationEdge(
        artifact_id=dst_artifact_id,
        src_rank=src_rank,
        dst_rank=dst_rank,
        src_slice=TensorSliceSpec(field_path=("payload",), offset=(0,), shape=(4,), dtype="torch.float32"),
        dst_slice=TensorSliceSpec(field_path=("payload",), offset=(0,), shape=(4,), dtype="torch.float32"),
        local_copy=False,
    )
    return ArtifactMigrationSchedule(
        migrate_id="mig-test",
        request_id=request_id,
        src_group_id="g_src",
        dst_group_id="g_dst",
        src_ranks=(src_rank,),
        dst_ranks=(dst_rank,),
        participant_ranks=(min(src_rank, dst_rank), max(src_rank, dst_rank)),
        edges=(edge,),
    )


def _build_command(
    *,
    request_id: str,
    src_artifact_id: str,
    dst_artifact_id: str,
    schedule: ArtifactMigrationSchedule,
    metadata_bytes: bytes,
    src_group: ExecutionGroupSpec,
    dst_group: ExecutionGroupSpec,
):
    from vllm_omni.diffusion.runtime_v2.multiproc_worker import (
        MigrateArtifactsCommand,
        _MIGRATE_PHASE_EXECUTE_TRANSFER,
    )

    dst_handle = ArtifactHandle(
        request_id=request_id,
        artifact_id=dst_artifact_id,
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        codec_id=_MINI_CODEC_ID,
    )
    source_handle = (
        None
        if src_artifact_id == dst_artifact_id
        else ArtifactHandle(
            request_id=request_id,
            artifact_id=src_artifact_id,
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.WORKER_LOCAL,
            codec_id=_MINI_CODEC_ID,
        )
    )
    plan = MigrateArtifactsPlan(
        request_id=request_id,
        src_group_id="g_src",
        dst_group_id="g_dst",
        artifacts=(MigrateArtifactSpec(handle=dst_handle, source_handle=source_handle),),
        request_metadata={},
    )
    return MigrateArtifactsCommand(
        migrate_id=schedule.migrate_id,
        phase=_MIGRATE_PHASE_EXECUTE_TRANSFER,
        plan=plan,
        src_group=src_group,
        dst_group=dst_group,
        schedule=schedule,
        metadata_payloads=((dst_artifact_id, metadata_bytes),),
    )


def _worker_entry(
    local_rank: int,
    world_size: int,
    master_port: int,
    src_artifact_id: str,
    dst_artifact_id: str,
    result_queue,
):
    """Body of each spawned process. Performs a real NCCL P2P RESHARD between
    rank 0 (src) and rank 1 (dst), then reports success/failure via queue."""
    try:
        torch.cuda.set_device(local_rank)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
        )

        runtime = _build_runtime_for_test(worker_rank=local_rank)

        # Two non-overlapping single-rank groups for the test.
        src_group = ExecutionGroupSpec(
            group_id="g_src",
            ranks=(0,),
            parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
            supported_task_kinds=(TaskKind.TEXT_ENCODE,),
        )
        dst_group = ExecutionGroupSpec(
            group_id="g_dst",
            ranks=(1,),
            parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
            supported_task_kinds=(TaskKind.DIT_PREPARE,),
        )

        request_id = "req-test"
        device = torch.device(f"cuda:{local_rank}")
        expected = torch.arange(4, dtype=torch.float32, device=device)

        # Seed src-side artifact value (under SOURCE id) before migration.
        if local_rank == 0:
            src_handle = ArtifactHandle(
                request_id=request_id,
                artifact_id=src_artifact_id,
                kind=ArtifactKind.REQUEST_STATE,
                layout=ArtifactLayout.WORKER_LOCAL,
                codec_id=_MINI_CODEC_ID,
            )
            # Seed state.device to src device (cuda:0). The dst rank should
            # rebind it to its own device after assemble; test below asserts this.
            value = _MiniState(note="hello", device=device, payload=expected.clone())
            key = runtime._artifact_key(request_id, "g_src", src_artifact_id)
            runtime.local_artifacts[key] = ArtifactValue(handle=src_handle, value=value)
            # Pre-pack metadata as the BUILD_SCHEDULE phase would have done.
            layout = runtime.codec_registry.get(_MINI_CODEC_ID).describe_output(
                group=src_group, group_rank=0, request_metadata={}, artifact=src_handle,
            )
            metadata_bytes = runtime.codec_registry.get(_MINI_CODEC_ID).pack_metadata(
                value=value, layout=layout,
            )
        else:
            metadata_bytes = b""

        # Broadcast the metadata bytes from src (rank 0) to dst (rank 1) so dst
        # has the same skeleton bytes that the controller would have shipped.
        meta_list = [metadata_bytes if local_rank == 0 else None]
        torch.distributed.broadcast_object_list(meta_list, src=0)
        metadata_bytes = meta_list[0]
        assert metadata_bytes is not None and isinstance(metadata_bytes, (bytes, bytearray))

        schedule = _build_schedule(
            request_id=request_id, src_rank=0, dst_rank=1, dst_artifact_id=dst_artifact_id,
        )
        command = _build_command(
            request_id=request_id,
            src_artifact_id=src_artifact_id,
            dst_artifact_id=dst_artifact_id,
            schedule=schedule,
            metadata_bytes=bytes(metadata_bytes),
            src_group=src_group,
            dst_group=dst_group,
        )

        # Real NCCL P2P data-plane invocation.
        result = runtime._execute_migration_transfer(command)
        assert result.error is None, result.error

        if local_rank == 1:
            # dst rank stores under DESTINATION id.
            key = runtime._artifact_key(request_id, "g_dst", dst_artifact_id)
            holder = runtime.local_artifacts.get(key)
            assert holder is not None, (
                f"dst rank did not receive migrated artifact under {dst_artifact_id!r}; "
                f"keys={list(runtime.local_artifacts.keys())!r}"
            )
            assembled = holder.value
            assert assembled.note == "hello", f"metadata round-trip failed: note={assembled.note!r}"
            assert assembled.payload is not None
            assert assembled.payload.device.type == "cuda"
            assert assembled.payload.dtype == torch.float32
            torch.testing.assert_close(assembled.payload.cpu(), expected.cpu())
            # Critical: state.device must point at the dst rank's device, not
            # the src rank's. Without this rebinding, downstream tasks call
            # torch.randn(..., device=state.device) and land tensors on cuda:0.
            assert assembled.device is not None
            assert assembled.device.type == "cuda"
            assert assembled.device.index == local_rank, (
                f"state.device not rebound: got {assembled.device}, expected cuda:{local_rank}"
            )
        result_queue.put(("ok", local_rank))
    except Exception as exc:  # noqa: BLE001
        import traceback
        result_queue.put(("error", local_rank, f"{exc}\n{traceback.format_exc()}"))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("src_artifact_id", "dst_artifact_id", "master_port"),
    [
        # Same id on both sides (legacy / simplest case).
        ("state_text", "state_text", 29611),
        # Renamed across the boundary (the real Wan/Qwen disaggregate path).
        # This case used to fail with "source artifact not found" before the
        # MigrateArtifactSpec.source_handle plumbing landed.
        ("state_text", "state_text_dit", 29612),
    ],
    ids=["same_id", "renamed"],
)
def test_reshard_nccl_p2p_round_trips_tensor_artifact(
    src_artifact_id: str, dst_artifact_id: str, master_port: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("RESHARD NCCL data-plane test requires CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"RESHARD NCCL test requires >=2 GPUs, got {torch.cuda.device_count()}")

    mp_ctx = torch.multiprocessing.get_context("spawn")
    result_queue: torch.multiprocessing.Queue = mp_ctx.Queue()
    procs = []
    for rank in range(2):
        p = mp_ctx.Process(
            target=_worker_entry,
            args=(rank, 2, master_port, src_artifact_id, dst_artifact_id, result_queue),
            daemon=False,
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        if p.is_alive():
            p.terminate()
            raise RuntimeError(f"worker process pid={p.pid} timed out")
        assert p.exitcode == 0, f"worker process pid={p.pid} exit_code={p.exitcode}"

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    assert len(results) == 2, f"expected 2 worker results, got {results!r}"
    for r in results:
        assert r[0] == "ok", f"worker reported error: {r!r}"
