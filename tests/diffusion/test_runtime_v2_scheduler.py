# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    FCFSSchedulerPolicy,
    GlobalScheduler,
    InMemoryArtifactStore,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


class _StaticCompiler:
    def __init__(self, plan: RequestExecutionPlan) -> None:
        self.plan = plan

    def compile_request(self, _request) -> RequestExecutionPlan:
        return self.plan


def _build_topology() -> RuntimeTopology:
    return RuntimeTopology(
        workers=(WorkerSpec(worker_rank=0, device_id=0),),
        groups=(
            ExecutionGroupSpec(
                group_id="g0",
                ranks=(0,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=(
                    TaskKind.TEXT_ENCODE,
                    TaskKind.FINALIZE,
                ),
            ),
        ),
    )


def _event(*, task: InferenceTask, kind: WorkerEventKind, metadata: dict | None = None) -> WorkerEvent:
    return WorkerEvent(
        event_id=f"{task.task_id}:{kind.value}",
        task_id=task.task_id,
        request_id=task.request_id,
        group_id=task.group_id or "g0",
        worker_rank=0,
        kind=kind,
        timestamp_ns=time.monotonic_ns(),
        metadata=metadata or {},
    )


def test_scheduler_piggybacks_final_input_release_on_task_dispatch() -> None:
    request_handle = ArtifactHandle(
        request_id="req-0",
        artifact_id="request",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    task = InferenceTask(
        task_id="req-0:text_encode:0",
        request_id="req-0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g0",
        parallel_spec=ParallelSpec(),
        status=TaskStatus.PENDING,
        inputs=(request_handle,),
    )
    plan = RequestExecutionPlan(
        request_id="req-0",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
        initial_artifacts=(ArtifactValue(handle=request_handle, value="request-state"),),
    )
    worker_pool = Mock()
    scheduler = GlobalScheduler(
        topology=_build_topology(),
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(_build_topology()),
    )

    scheduler.submit_request(object())

    worker_pool.dispatch.assert_called_once_with(
        task=task,
        inline_inputs=(ArtifactValue(handle=request_handle, value="request-state"),),
        release_after_exec_artifact_ids=("request",),
    )


def test_scheduler_dispatch_sets_task_parallel_spec_from_group() -> None:
    request_handle = ArtifactHandle(
        request_id="req-sp2",
        artifact_id="request",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.HOST,
    )
    task = InferenceTask(
        task_id="req-sp2:dit_step_chunk:0",
        request_id="req-sp2",
        kind=TaskKind.DIT_STEP_CHUNK,
        group_id="g_sp2",
        parallel_spec=ParallelSpec(),
        status=TaskStatus.PENDING,
        inputs=(request_handle,),
    )
    plan = RequestExecutionPlan(
        request_id="req-sp2",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
        initial_artifacts=(ArtifactValue(handle=request_handle, value="request-state"),),
    )
    topology = RuntimeTopology(
        workers=(WorkerSpec(worker_rank=0, device_id=0), WorkerSpec(worker_rank=1, device_id=1)),
        groups=(
            ExecutionGroupSpec(
                group_id="g_sp2",
                ranks=(0, 1),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
            ),
        ),
    )
    worker_pool = Mock()
    scheduler = GlobalScheduler(
        topology=topology,
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(topology),
    )

    scheduler.submit_request(object())

    assert task.parallel_spec == ParallelSpec(tp=1, sp=2, cfg=1)
    worker_pool.dispatch.assert_called_once()


def test_scheduler_evicts_request_after_collecting_terminal_output() -> None:
    output_handle = ArtifactHandle(
        request_id="req-1",
        artifact_id="output",
        kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.HOST,
        producer_task_id="req-1:finalize:0",
    )
    task = InferenceTask(
        task_id="req-1:finalize:0",
        request_id="req-1",
        kind=TaskKind.FINALIZE,
        group_id="g0",
        parallel_spec=ParallelSpec(),
        status=TaskStatus.PENDING,
        outputs=(output_handle,),
    )
    plan = RequestExecutionPlan(
        request_id="req-1",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
    )
    worker_pool = Mock()
    worker_pool.start_fetch_artifacts.return_value = "fetch-1"
    worker_pool.poll_fetch_artifacts.side_effect = [
        Mock(
            error=None,
            artifacts=(ArtifactValue(handle=output_handle, value="done"),),
            fetch_id="fetch-1",
        ),
    ]
    worker_pool.fetch_artifacts.return_value = Mock(
        error=None,
        artifacts=(ArtifactValue(handle=output_handle, value="done"),),
    )
    scheduler = GlobalScheduler(
        topology=_build_topology(),
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(_build_topology()),
    )

    scheduler.submit_request(object())
    scheduler._handle_worker_event(
        _event(
            task=task,
            kind=WorkerEventKind.TASK_LAUNCH_END,
            metadata={
                "published_outputs": (
                    WorkerLocalArtifactRef(handle=output_handle, group_id="g0", worker_rank=0),
                )
            },
        )
    )
    scheduler._handle_worker_event(_event(task=task, kind=WorkerEventKind.TASK_EXEC_END))

    status, payload = scheduler.get_request_status("req-1")
    assert status == "pending"
    assert payload is None

    status, payload = scheduler.get_request_status("req-1")

    assert status == "finished"
    assert payload == "done"
    worker_pool.start_fetch_artifacts.assert_called_once_with(
        request_id="req-1",
        group_id="g0",
        artifact_ids=("output",),
    )
    worker_pool.poll_fetch_artifacts.assert_called_with("fetch-1")
    worker_pool.evict_request.assert_called_once_with("req-1")
    # Controller-side state is freed at cleanup: the artifact store, the plan and
    # the per-task bookkeeping must not accumulate for the process lifetime.
    assert not [k for k in scheduler.artifact_store._values if k[0] == "req-1"]
    assert "req-1" not in scheduler.plans
    assert task.task_id not in scheduler.task_index
    # The delivered output is cached so a re-poll still answers "finished"; it is
    # only dropped by release_request (called by the run loops after delivery).
    assert scheduler._completed_outputs.get("req-1") == "done"
    finished_key = f"req-1:{WorkerEventKind.REQUEST_FINISHED.value}"
    assert finished_key in scheduler.released_requests
    scheduler.release_request("req-1")
    assert "req-1" not in scheduler._completed_outputs
    assert "req-1" in scheduler.cleaned_requests
    assert finished_key in scheduler.released_requests

    status, payload = scheduler.get_request_status("req-1")

    assert status == "pending"
    assert payload is None


def test_scheduler_reports_failed_when_output_fetch_errors() -> None:
    # Regression: a fetch that completes with an error must transition the request
    # to "failed" and free its state, NOT raise a raw RuntimeError out of
    # get_request_status (which would never record the failure, would leak the
    # plan/task bookkeeping, and would let a re-poll silently start a new fetch).
    output_handle = ArtifactHandle(
        request_id="req-err",
        artifact_id="output",
        kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.HOST,
        producer_task_id="req-err:finalize:0",
    )
    task = InferenceTask(
        task_id="req-err:finalize:0",
        request_id="req-err",
        kind=TaskKind.FINALIZE,
        group_id="g0",
        parallel_spec=ParallelSpec(),
        status=TaskStatus.PENDING,
        outputs=(output_handle,),
    )
    plan = RequestExecutionPlan(
        request_id="req-err",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
    )
    worker_pool = Mock()
    worker_pool.start_fetch_artifacts.return_value = "fetch-err"
    worker_pool.poll_fetch_artifacts.side_effect = [
        Mock(error="boom while fetching", artifacts=(), fetch_id="fetch-err"),
    ]
    scheduler = GlobalScheduler(
        topology=_build_topology(),
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(_build_topology()),
    )

    scheduler.submit_request(object())
    scheduler._handle_worker_event(
        _event(
            task=task,
            kind=WorkerEventKind.TASK_LAUNCH_END,
            metadata={
                "published_outputs": (
                    WorkerLocalArtifactRef(handle=output_handle, group_id="g0", worker_rank=0),
                )
            },
        )
    )
    scheduler._handle_worker_event(_event(task=task, kind=WorkerEventKind.TASK_EXEC_END))

    scheduler.get_request_status("req-err")  # starts the async fetch
    status, payload = scheduler.get_request_status("req-err")  # poll surfaces the error

    assert status == "failed"
    assert payload == "boom while fetching"
    # State is freed, not leaked, and there is no lingering pending fetch that a
    # re-poll could restart.
    assert "req-err" not in scheduler.plans
    assert task.task_id not in scheduler.task_index
    assert "req-err" not in scheduler.pending_output_fetches
    assert not [k for k in scheduler.artifact_store._values if k[0] == "req-err"]
    # The failure is durable across re-polls until the request is released.
    assert scheduler.get_request_status("req-err") == ("failed", "boom while fetching")
    scheduler.release_request("req-err")
    assert "req-err" not in scheduler.failed_requests
    assert scheduler.get_request_status("req-err") == ("pending", None)


def test_scheduler_ignores_late_request_failed_after_release() -> None:
    worker_pool = Mock()
    policy = Mock()
    policy.on_worker_event.return_value = ()
    scheduler = GlobalScheduler(
        topology=_build_topology(),
        worker_pool=worker_pool,
        compiler=_StaticCompiler(
            RequestExecutionPlan(
                request_id="unused",
                tasks={},
                terminal_task_ids=(),
            )
        ),
        artifact_store=InMemoryArtifactStore(),
        policy=policy,
    )
    event = WorkerEvent(
        event_id="late-fail",
        task_id="",
        request_id="req-failed",
        group_id="g0",
        worker_rank=0,
        kind=WorkerEventKind.REQUEST_FAILED,
        timestamp_ns=time.monotonic_ns(),
        message="boom",
    )

    scheduler._handle_worker_event(event)
    status, payload = scheduler.get_request_status("req-failed")

    assert status == "failed"
    assert payload == "boom"
    policy.on_worker_event.assert_called_once_with(event)
    failed_key = f"req-failed:{WorkerEventKind.REQUEST_FAILED.value}"
    assert failed_key in scheduler.released_requests

    scheduler.release_request("req-failed")
    assert "req-failed" not in scheduler.failed_requests
    assert "req-failed" in scheduler.cleaned_requests
    assert failed_key in scheduler.released_requests

    status, payload = scheduler.get_request_status("req-failed")

    assert status == "pending"
    assert payload is None
    policy.on_worker_event.reset_mock()

    scheduler._handle_worker_event(event)

    policy.on_worker_event.assert_not_called()


def test_scheduler_reports_finished_for_none_terminal_output() -> None:
    # Regression: completion must be signalled by the terminal task finishing, not
    # by "output is not None" -- otherwise a legitimately None output hangs forever.
    output_handle = ArtifactHandle(
        request_id="req-none",
        artifact_id="output",
        kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.HOST,
        producer_task_id="req-none:finalize:0",
    )
    task = InferenceTask(
        task_id="req-none:finalize:0",
        request_id="req-none",
        kind=TaskKind.FINALIZE,
        group_id="g0",
        parallel_spec=ParallelSpec(),
        status=TaskStatus.PENDING,
        outputs=(output_handle,),
    )
    plan = RequestExecutionPlan(
        request_id="req-none",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
    )
    worker_pool = Mock()
    worker_pool.start_fetch_artifacts.return_value = "fetch-none"
    worker_pool.poll_fetch_artifacts.side_effect = [
        Mock(error=None, artifacts=(ArtifactValue(handle=output_handle, value=None),), fetch_id="fetch-none"),
    ]
    scheduler = GlobalScheduler(
        topology=_build_topology(),
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(_build_topology()),
    )

    scheduler.submit_request(object())
    scheduler._handle_worker_event(
        _event(
            task=task,
            kind=WorkerEventKind.TASK_LAUNCH_END,
            metadata={
                "published_outputs": (
                    WorkerLocalArtifactRef(handle=output_handle, group_id="g0", worker_rank=0),
                )
            },
        )
    )
    scheduler._handle_worker_event(_event(task=task, kind=WorkerEventKind.TASK_EXEC_END))

    scheduler.get_request_status("req-none")  # starts the async fetch
    status, payload = scheduler.get_request_status("req-none")  # poll completes with None
    assert status == "finished"
    assert payload is None
