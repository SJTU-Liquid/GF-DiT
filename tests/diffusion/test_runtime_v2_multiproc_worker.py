# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
import queue
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.runtime_v2.multiproc_worker import (
    _WorkerProcessRuntime,
    MultiprocWorkerPool,
    FetchArtifactsCommand,
    FetchArtifactsResult,
    MigrateArtifactsCommand,
    MigrateArtifactsRankResult,
    ProcessDispatchTaskCommand,
    SerializedArtifactValue,
    _deserialize_artifact_value,
    _serialize_artifact_value,
)
from vllm_omni.diffusion.runtime_v2.adapters.common import output_artifact_layout, sharded_tensor_field
from vllm_omni.diffusion.runtime_v2.data_plane import (
    _SHARDED_FIELD_GLOBAL,
    _SHARDED_FIELD_LOCAL,
    _validate_sharded_field_status,
    build_layout_migration_schedule,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    ExecutionGroupSpec,
    InferenceTask,
    MigrateArtifactSpec,
    MigrateArtifactsPlan,
    ParallelSpec,
    TaskKind,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _make_topology() -> RuntimeTopology:
    return RuntimeTopology(
        workers=(
            WorkerSpec(worker_rank=0, device_id=0),
            WorkerSpec(worker_rank=1, device_id=1),
        ),
        groups=(
            ExecutionGroupSpec(
                group_id="g0",
                ranks=(0, 1),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
            ),
        ),
    )


def _make_task() -> InferenceTask:
    return InferenceTask(
        task_id="task-0",
        request_id="req-0",
        kind=TaskKind.DIT_STEP_CHUNK,
        group_id="g0",
        parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
    )


def _make_event(
    task: InferenceTask,
    *,
    worker_rank: int,
    kind: WorkerEventKind,
    message: str = "",
    metadata: dict | None = None,
) -> WorkerEvent:
    return WorkerEvent(
        event_id=f"{task.task_id}:{worker_rank}:{kind.value}",
        task_id=task.task_id,
        request_id=task.request_id,
        group_id=task.group_id or "",
        worker_rank=worker_rank,
        kind=kind,
        timestamp_ns=1000 + worker_rank,
        message=message,
        metadata=metadata or {},
    )


def test_task_end_event_is_emitted_only_after_all_ranks_report() -> None:
    pool = MultiprocWorkerPool(topology=_make_topology(), od_config=Mock())
    task = _make_task()
    pool._register_task_dispatch(task)

    assert pool._consume_event(_make_event(task, worker_rank=0, kind=WorkerEventKind.TASK_EXEC_END)) == []

    events = pool._consume_event(_make_event(task, worker_rank=1, kind=WorkerEventKind.TASK_EXEC_END))

    assert len(events) == 1
    event = events[0]
    assert event.kind == WorkerEventKind.TASK_EXEC_END
    assert event.worker_rank == 0
    assert event.metadata["completed_ranks"] == (0, 1)


def test_launch_end_uses_owner_metadata_after_all_ranks_report() -> None:
    pool = MultiprocWorkerPool(topology=_make_topology(), od_config=Mock())
    task = _make_task()
    pool._register_task_dispatch(task)

    owner_metadata = {"published_outputs": ("artifact-ref",)}
    assert (
        pool._consume_event(
            _make_event(
                task,
                worker_rank=0,
                kind=WorkerEventKind.TASK_LAUNCH_END,
                metadata=owner_metadata,
            )
        )
        == []
    )

    events = pool._consume_event(
        _make_event(
            task,
            worker_rank=1,
            kind=WorkerEventKind.TASK_LAUNCH_END,
            metadata={"published_outputs": ()},
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind == WorkerEventKind.TASK_LAUNCH_END
    assert event.metadata["published_outputs"] == ("artifact-ref",)
    assert event.metadata["completed_ranks"] == (0, 1)


def test_task_failure_is_reported_immediately_and_late_events_are_dropped() -> None:
    pool = MultiprocWorkerPool(topology=_make_topology(), od_config=Mock())
    task = _make_task()
    pool._register_task_dispatch(task)

    events = pool._consume_event(
        _make_event(
            task,
            worker_rank=1,
            kind=WorkerEventKind.TASK_FAILED,
            message="rank-1 failed",
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind == WorkerEventKind.TASK_FAILED
    assert event.worker_rank == 1
    assert event.metadata["failed_rank"] == 1
    assert pool._consume_event(_make_event(task, worker_rank=0, kind=WorkerEventKind.TASK_EXEC_END)) == []


def _register_text_encode_group(runtime, group_id: str = "g0") -> None:
    # Register a single-rank sp=1 execution group so _execute_task's setup
    # (_group_spec / _validate_task_parallel_spec) resolves it instead of
    # KeyError-ing. Injected directly to bypass _ensure_group_spec, which would
    # walk the Mock() od_config.parallel_config in these unit tests.
    runtime._execution_group_specs_by_id[group_id] = ExecutionGroupSpec(
        group_id=group_id,
        ranks=(0,),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        supported_task_kinds=(TaskKind.TEXT_ENCODE,),
    )


def test_worker_runtime_executes_tasks_inside_forward_context(mocker) -> None:
    runtime = _WorkerProcessRuntime(
        worker_rank=0,
        device_id=0,
        od_config=Mock(),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        command_pipe_r=Mock(),
        event_pipe_w=Mock(),
        result_pipe_w=Mock(),
    )
    runtime.fixed_session = Mock()
    runtime.worker_wrapper = Mock()
    runtime.worker_wrapper.worker = Mock(vllm_config="vllm-config", od_config="od-config")
    executor = Mock()
    output_handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="state-1",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    executor.execute.return_value = (ArtifactValue(handle=output_handle, value="state"),)
    runtime.executors = {TaskKind.TEXT_ENCODE: executor}
    _register_text_encode_group(runtime)
    task = InferenceTask(
        task_id="task-ctx",
        request_id="req-0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g0",
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        outputs=(output_handle,),
    )
    command = ProcessDispatchTaskCommand(task=task, inline_inputs=(), result_owner_rank=0)
    context_log: list[tuple[object, object]] = []

    def _fake_set_forward_context(*, vllm_config=None, omni_diffusion_config=None):
        context_log.append((vllm_config, omni_diffusion_config))
        return nullcontext()

    mocker.patch(
        "vllm_omni.diffusion.runtime_v2.multiproc_worker.set_forward_context",
        side_effect=_fake_set_forward_context,
    )

    runtime._execute_task(command)

    runtime.fixed_session.activate.assert_called_once_with()
    executor.execute.assert_called_once_with(task, {})
    assert context_log == [("vllm-config", "od-config")]


def test_execute_task_propagates_resolve_failure_instead_of_task_failed(mocker) -> None:
    # Regression: _resolve_inputs -> materialize_input_artifact issues a cross-rank
    # all_gather of sharded inputs. A failure there must PROPAGATE (so the worker
    # and its process group tear down and SP peers fail fast), not be swallowed
    # into a TASK_FAILED + continue, which would leave peers blocked forever inside
    # the collective and desync this worker for every later task.
    runtime = _WorkerProcessRuntime(
        worker_rank=0,
        device_id=0,
        od_config=Mock(),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        command_pipe_r=Mock(),
        event_pipe_w=Mock(),
        result_pipe_w=Mock(),
    )
    runtime.fixed_session = Mock()
    runtime.worker_wrapper = Mock()
    runtime.worker_wrapper.worker = Mock(vllm_config="vllm-config", od_config="od-config")
    executor = Mock()
    runtime.executors = {TaskKind.TEXT_ENCODE: executor}
    _register_text_encode_group(runtime)
    task = InferenceTask(
        task_id="task-resolve-fail",
        request_id="req-0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g0",
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
    )
    command = ProcessDispatchTaskCommand(task=task, inline_inputs=(), result_owner_rank=0)
    # Make local setup deterministically succeed so the test isolates the
    # input-resolution path: resolution then fails and must PROPAGATE.
    mocker.patch.object(runtime, "_group_spec", return_value=Mock())
    mocker.patch.object(runtime, "_validate_task_parallel_spec")
    mocker.patch.object(runtime, "_activate_group_session")
    mocker.patch.object(runtime, "_resolve_inputs", side_effect=RuntimeError("missing shard"))

    with pytest.raises(RuntimeError, match="missing shard"):
        runtime._execute_task(command)

    # The failure propagated rather than being downgraded to a TASK_FAILED event,
    # and the executor was never run.
    runtime.event_pipe_w.send.assert_not_called()
    executor.execute.assert_not_called()


def test_execute_task_reports_local_setup_failure_as_task_failed(mocker) -> None:
    # Local, rank-symmetric setup (session activation, validation) has no collective
    # in flight, so a failure stays recoverable: emit TASK_FAILED and return instead
    # of killing the worker. Input resolution is never reached.
    runtime = _WorkerProcessRuntime(
        worker_rank=0,
        device_id=0,
        od_config=Mock(),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        command_pipe_r=Mock(),
        event_pipe_w=Mock(),
        result_pipe_w=Mock(),
    )
    runtime.fixed_session = Mock()
    runtime.worker_wrapper = Mock()
    runtime.executors = {TaskKind.TEXT_ENCODE: Mock()}
    task = InferenceTask(
        task_id="task-setup-fail",
        request_id="req-0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g0",
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
    )
    command = ProcessDispatchTaskCommand(task=task, inline_inputs=(), result_owner_rank=0)
    # Drive the failure specifically from session activation (a local step), with
    # the earlier setup steps mocked to succeed.
    mocker.patch.object(runtime, "_group_spec", return_value=Mock())
    mocker.patch.object(runtime, "_validate_task_parallel_spec")
    mocker.patch.object(runtime, "_activate_group_session", side_effect=RuntimeError("activate boom"))
    resolve = mocker.patch.object(runtime, "_resolve_inputs")

    runtime._execute_task(command)  # must NOT raise

    sent = [c.args[0] for c in runtime.event_pipe_w.send.call_args_list]
    assert len(sent) == 1
    assert sent[0].kind == WorkerEventKind.TASK_FAILED
    assert "task setup failed" in sent[0].message
    resolve.assert_not_called()


def test_worker_runtime_executes_tasks_inside_inference_mode(mocker) -> None:
    runtime = _WorkerProcessRuntime(
        worker_rank=0,
        device_id=0,
        od_config=Mock(),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        command_pipe_r=Mock(),
        event_pipe_w=Mock(),
        result_pipe_w=Mock(),
    )
    runtime.fixed_session = Mock()
    runtime.worker_wrapper = Mock()
    runtime.worker_wrapper.worker = Mock(vllm_config="vllm-config", od_config=Mock())
    runtime.worker_wrapper.worker.od_config.parallel_config = Mock(use_hsdp=False)
    executor = Mock()
    output_handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="state-1",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    executor.execute.return_value = (ArtifactValue(handle=output_handle, value="state"),)
    runtime.executors = {TaskKind.TEXT_ENCODE: executor}
    _register_text_encode_group(runtime)
    task = InferenceTask(
        task_id="task-grad",
        request_id="req-0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g0",
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        outputs=(output_handle,),
    )
    command = ProcessDispatchTaskCommand(task=task, inline_inputs=(), result_owner_rank=0)
    grad_log: list[str] = []

    class _TrackedContext:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            grad_log.append(f"enter:{self.name}")
            return None

        def __exit__(self, exc_type, exc, tb):
            grad_log.append(f"exit:{self.name}")
            return False

    mocker.patch(
        "vllm_omni.diffusion.runtime_v2.multiproc_worker.set_forward_context",
        return_value=nullcontext(),
    )
    mocker.patch(
        "vllm_omni.diffusion.runtime_v2.multiproc_worker.torch.inference_mode",
        return_value=_TrackedContext("inference_mode"),
    )
    mocker.patch(
        "vllm_omni.diffusion.runtime_v2.multiproc_worker.torch.no_grad",
        return_value=_TrackedContext("no_grad"),
    )

    runtime._execute_task(command)

    executor.execute.assert_called_once_with(task, {})
    assert grad_log == ["enter:inference_mode", "exit:inference_mode"]


def test_worker_runtime_releases_artifacts_after_task_exec(mocker) -> None:
    runtime = _WorkerProcessRuntime(
        worker_rank=0,
        device_id=0,
        od_config=Mock(),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        command_pipe_r=Mock(),
        event_pipe_w=Mock(),
        result_pipe_w=Mock(),
    )
    runtime.fixed_session = Mock()
    runtime.worker_wrapper = Mock()
    runtime.worker_wrapper.worker = Mock(vllm_config="vllm-config", od_config=Mock())
    runtime.worker_wrapper.worker.od_config.parallel_config = Mock(use_hsdp=False)
    input_handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="state-in",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    output_handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="state-out",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    runtime.local_artifacts[(input_handle.request_id, "g0", input_handle.artifact_id)] = ArtifactValue(
        handle=input_handle,
        value="input-state",
    )
    executor = Mock()
    executor.execute.return_value = (ArtifactValue(handle=output_handle, value="output-state"),)
    runtime.executors = {TaskKind.TEXT_ENCODE: executor}
    _register_text_encode_group(runtime)
    task = InferenceTask(
        task_id="task-release",
        request_id="req-0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g0",
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        inputs=(input_handle,),
        outputs=(output_handle,),
    )
    command = ProcessDispatchTaskCommand(
        task=task,
        inline_inputs=(),
        result_owner_rank=0,
        release_after_exec_artifact_ids=("state-in",),
    )

    mocker.patch(
        "vllm_omni.diffusion.runtime_v2.multiproc_worker.set_forward_context",
        return_value=nullcontext(),
    )

    runtime._execute_task(command)

    assert (input_handle.request_id, "g0", input_handle.artifact_id) not in runtime.local_artifacts
    assert runtime.local_artifacts[(output_handle.request_id, "g0", output_handle.artifact_id)].value == "output-state"


def test_fetch_artifacts_deserializes_worker_payloads(mocker) -> None:
    pool = MultiprocWorkerPool(topology=_make_topology(), od_config=Mock())
    handle = Mock()
    handle.command_pipe_w = Mock()
    pool.worker_handles = {0: handle, 1: Mock()}
    artifact = ArtifactValue(
        handle=ArtifactHandle(
            request_id="req-0",
            artifact_id="artifact-0",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.HOST,
        ),
        value={"x": 1},
    )
    mocker.patch.object(pool, "start_fetch_artifacts", return_value="fetch-0")
    mocker.patch.object(
        pool,
        "poll_fetch_artifacts",
        side_effect=[
            None,
            FetchArtifactsResult(
                request_id="req-0",
                worker_rank=0,
                artifacts=(_serialize_artifact_value(artifact),),
                fetch_id="fetch-0",
            ),
        ],
    )
    mocker.patch("vllm_omni.diffusion.runtime_v2.multiproc_worker.time.sleep", return_value=None)

    result = pool.fetch_artifacts(request_id="req-0", group_id="g0", artifact_ids=("artifact-0",))

    pool.start_fetch_artifacts.assert_called_once_with(
        request_id="req-0",
        group_id="g0",
        artifact_ids=("artifact-0",),
    )
    assert len(result.artifacts) == 1
    assert isinstance(result.artifacts[0], ArtifactValue)
    assert result.artifacts[0].value == {"x": 1}


def test_worker_fetch_does_not_activate_group_session(mocker) -> None:
    runtime = _WorkerProcessRuntime(
        worker_rank=0,
        device_id=0,
        od_config=Mock(),
        parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
        command_pipe_r=Mock(),
        event_pipe_w=Mock(),
        result_pipe_w=Mock(),
    )
    handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="artifact-0",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
    )
    runtime.local_artifacts[(handle.request_id, "g0", handle.artifact_id)] = ArtifactValue(
        handle=handle,
        value={"x": 1},
    )
    activate = mocker.patch.object(runtime, "_maybe_activate_group_session")

    runtime._handle_fetch_artifacts(
        FetchArtifactsCommand(
            fetch_id="fetch-0",
            request_id="req-0",
            group_id="g0",
            artifact_ids=("artifact-0",),
        )
    )

    activate.assert_not_called()
    result = runtime.result_pipe_w.send.call_args.args[0]
    assert isinstance(result, FetchArtifactsResult)
    assert result.fetch_id == "fetch-0"
    assert result.error is None


def test_sharded_input_status_rejects_mixed_full_and_local_tensor_state() -> None:
    handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="state",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        codec_id="codec.state.v1",
    )
    field = sharded_tensor_field(
        "latents",
        (5,),
        torch.float32,
        shard_dim=0,
        shard_world_size=2,
        shard_rank=0,
    )

    with pytest.raises(RuntimeError, match="inconsistent per-rank materialization state"):
        _validate_sharded_field_status(
            artifact=handle,
            field=field,
            local_status=_SHARDED_FIELD_LOCAL,
            all_statuses=[_SHARDED_FIELD_LOCAL, _SHARDED_FIELD_GLOBAL],
            tensor=torch.zeros(field.local_shape),
        )


def test_serialized_artifact_value_exposes_value_property() -> None:
    serialized = SerializedArtifactValue(handle="h0", payload=pickle.dumps({"x": 1}, protocol=pickle.HIGHEST_PROTOCOL))
    assert serialized.value == {"x": 1}


def test_serialize_large_output_artifact_prefers_shm() -> None:
    output_handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="output",
        kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.HOST,
    )
    output_tensor = torch.arange(300_000, dtype=torch.float32).reshape(300, 1000)
    artifact = ArtifactValue(
        handle=output_handle,
        value=DiffusionOutput(output=output_tensor),
    )

    serialized = _serialize_artifact_value(artifact, prefer_shm_output=True)
    restored = _deserialize_artifact_value(serialized)

    assert serialized.transport == "shm"
    assert serialized.payload_nbytes > 0
    assert isinstance(restored.value, DiffusionOutput)
    assert isinstance(restored.value.output, torch.Tensor)
    assert torch.equal(restored.value.output, output_tensor)


def test_serialize_non_output_artifact_keeps_pickle_when_shm_preferred() -> None:
    artifact = ArtifactValue(
        handle=ArtifactHandle(
            request_id="req-0",
            artifact_id="state-0",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.WORKER_LOCAL,
        ),
        value={"x": 1},
    )

    serialized = _serialize_artifact_value(artifact, prefer_shm_output=True)

    assert serialized.transport == "pickle"
    assert serialized.value == {"x": 1}


def test_layout_migration_schedule_handles_irregular_2_to_4_shards() -> None:
    handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="state",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        codec_id="codec.state.v1",
    )
    src_group = ExecutionGroupSpec(
        group_id="g_sp2",
        ranks=(0, 1),
        parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
        supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
        ulysses_degree=2,
        ring_degree=1,
    )
    dst_group = ExecutionGroupSpec(
        group_id="g_sp4",
        ranks=(0, 1, 2, 3),
        parallel_spec=ParallelSpec(tp=1, sp=4, cfg=1),
        supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
        ulysses_degree=4,
        ring_degree=1,
    )
    src_layout_results = [
        MigrateArtifactsRankResult(
            migrate_id="mig",
            request_id="req-0",
            worker_rank=rank,
            phase="describe_layout",
            src_layouts=(
                output_artifact_layout(
                    handle,
                    (
                        sharded_tensor_field(
                            "latents",
                            (5,),
                            torch.float32,
                            shard_dim=0,
                            shard_world_size=2,
                            shard_rank=rank,
                        ),
                    ),
                ),
            ),
        )
        for rank in (0, 1)
    ]
    dst_layout_results = [
        MigrateArtifactsRankResult(
            migrate_id="mig",
            request_id="req-0",
            worker_rank=rank,
            phase="describe_layout",
            dst_layouts=(
                output_artifact_layout(
                    handle,
                    (
                        sharded_tensor_field(
                            "latents",
                            (5,),
                            torch.float32,
                            shard_dim=0,
                            shard_world_size=4,
                            shard_rank=rank,
                        ),
                    ),
                ),
            ),
        )
        for rank in (0, 1, 2, 3)
    ]
    command = MigrateArtifactsCommand(
        migrate_id="mig",
        phase="build_schedule",
        plan=MigrateArtifactsPlan(
            request_id="req-0",
            src_group_id="g_sp2",
            dst_group_id="g_sp4",
            artifacts=(MigrateArtifactSpec(handle=handle),),
        ),
        src_group=src_group,
        dst_group=dst_group,
        layout_results=tuple(src_layout_results + dst_layout_results),
    )

    schedule = build_layout_migration_schedule(command)

    edge_summary = [
        (
            edge.src_rank,
            edge.dst_rank,
            edge.src_slice.offset,
            edge.dst_slice.offset,
            edge.src_slice.shape,
        )
        for edge in schedule.edges
    ]
    assert edge_summary == [
        (0, 0, (0,), (0,), (2,)),
        (0, 1, (2,), (0,), (1,)),
        (1, 2, (0,), (0,), (1,)),
        (1, 3, (1,), (0,), (1,)),
    ]
    assert [edge.local_copy for edge in schedule.edges] == [True, False, False, False]


def test_poll_reads_forwarded_events_from_threadsafe_queue() -> None:
    pool = MultiprocWorkerPool(topology=_make_topology(), od_config=Mock())
    task = _make_task()
    event = _make_event(task, worker_rank=0, kind=WorkerEventKind.TASK_EXEC_BEGIN)
    pool._event_queue.put(event)

    events = pool.poll(timeout_s=0.0)

    assert events == [event]


def test_wait_for_result_reads_forwarded_results_from_threadsafe_queue() -> None:
    pool = MultiprocWorkerPool(topology=_make_topology(), od_config=Mock())
    pool._result_queues = {0: queue.Queue()}
    expected = FetchArtifactsResult(request_id="req-0", worker_rank=0, artifacts=())
    pool._result_queues[0].put(expected)

    result = pool._wait_for_result(leader_rank=0, timeout_s=0.1, expected_type=FetchArtifactsResult)

    assert result == expected
