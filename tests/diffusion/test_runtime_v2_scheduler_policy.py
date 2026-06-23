# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from types import SimpleNamespace

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
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    EdfGreedySchedulerPolicy,
    FCFSSchedulerPolicy,
    SJFSchedulerPolicy,
    SRTFSchedulerPolicy,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _build_topology() -> RuntimeTopology:
    task_kinds: tuple[TaskKind, ...] = (
        TaskKind.TEXT_ENCODE,
        TaskKind.DIT_PREPARE,
        TaskKind.TIMESTEP_PREPARE,
        TaskKind.DIT_STEP_CHUNK,
        TaskKind.VAE_DECODE,
        TaskKind.FINALIZE,
    )
    return RuntimeTopology(
        workers=(WorkerSpec(worker_rank=0, device_id=0),),
        groups=(
            ExecutionGroupSpec(
                group_id="g0",
                ranks=(0,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg_parallel=False),
                supported_task_kinds=task_kinds,
            ),
        ),
    )


def _build_two_group_topology() -> RuntimeTopology:
    return RuntimeTopology(
        workers=(
            WorkerSpec(worker_rank=0, device_id=0),
            WorkerSpec(worker_rank=1, device_id=1),
            WorkerSpec(worker_rank=2, device_id=2),
        ),
        groups=(
            ExecutionGroupSpec(
                group_id="g0",
                ranks=(0, 1),
                parallel_spec=ParallelSpec(tp=2, sp=1, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
            ),
            ExecutionGroupSpec(
                group_id="g1",
                ranks=(2,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
            ),
        ),
    )


def _build_edf_topology(worker_count: int = 4) -> RuntimeTopology:
    all_task_kinds: tuple[TaskKind, ...] = (
        TaskKind.TEXT_ENCODE,
        TaskKind.DIT_PREPARE,
        TaskKind.TIMESTEP_PREPARE,
        TaskKind.DIT_STEP_CHUNK,
        TaskKind.VAE_DECODE,
        TaskKind.FINALIZE,
    )
    groups: list[ExecutionGroupSpec] = []
    if worker_count > 1:
        groups.append(
            ExecutionGroupSpec(
                group_id="g_world",
                ranks=tuple(range(worker_count)),
                parallel_spec=ParallelSpec(tp=1, sp=worker_count, cfg=1),
                supported_task_kinds=(TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK),
                ulysses_degree=worker_count,
                ring_degree=1,
            )
        )
    for rank in range(worker_count):
        groups.append(
            ExecutionGroupSpec(
                group_id=f"g_sp1_r{rank}",
                ranks=(rank,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=all_task_kinds,
            )
        )
    return RuntimeTopology(
        workers=tuple(WorkerSpec(worker_rank=rank, device_id=rank) for rank in range(worker_count)),
        groups=tuple(groups),
    )


def _build_task(
    *,
    request_id: str,
    task_idx: int,
    dependencies: tuple[str, ...] = (),
    seq_len: int,
    kind: TaskKind = TaskKind.DIT_STEP_CHUNK,
) -> InferenceTask:
    return InferenceTask(
        task_id=f"{request_id}:{kind.value}:{task_idx}",
        request_id=request_id,
        kind=kind,
        group_id=None,
        parallel_spec=ParallelSpec(),
        status=TaskStatus.PENDING,
        dependencies=dependencies,
        payload={
            "seq_len": seq_len,
            "request_stage": kind.value,
        },
    )


def _initial_artifact(request_id: str, **attrs) -> ArtifactValue:
    return ArtifactValue(
        handle=ArtifactHandle(
            request_id=request_id,
            artifact_id="request",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.HOST,
        ),
        value=SimpleNamespace(**attrs),
    )


def _event(
    *,
    task: InferenceTask,
    kind: WorkerEventKind,
    group_id: str = "g0",
) -> WorkerEvent:
    return WorkerEvent(
        event_id=f"{task.task_id}:{kind.value}",
        task_id=task.task_id,
        request_id=task.request_id,
        group_id=group_id,
        worker_rank=0,
        kind=kind,
        timestamp_ns=time.monotonic_ns(),
    )


def test_srtf_prefers_smaller_estimated_remaining_time() -> None:
    calls: list[tuple[ParallelSpec, int, str]] = []

    def estimator(parallel_spec: ParallelSpec, seq_len: int, request_stage: str) -> float:
        calls.append((parallel_spec, seq_len, request_stage))
        return float(seq_len)

    policy = SRTFSchedulerPolicy(
        topology=_build_topology(),
        task_runtime_estimator=estimator,
    )

    req_a_t0 = _build_task(request_id="req-a", task_idx=0, seq_len=100)
    req_a_t1 = _build_task(
        request_id="req-a",
        task_idx=1,
        dependencies=(req_a_t0.task_id,),
        seq_len=100,
    )
    req_b_t0 = _build_task(request_id="req-b", task_idx=0, seq_len=10)

    plan_a = RequestExecutionPlan(
        request_id="req-a",
        tasks={req_a_t0.task_id: req_a_t0, req_a_t1.task_id: req_a_t1},
        terminal_task_ids=(req_a_t1.task_id,),
    )
    plan_b = RequestExecutionPlan(
        request_id="req-b",
        tasks={req_b_t0.task_id: req_b_t0},
        terminal_task_ids=(req_b_t0.task_id,),
    )

    first_dispatch = list(policy.on_request_submitted(plan_a))
    assert [task.task_id for task in first_dispatch] == [req_a_t0.task_id]

    second_dispatch = list(policy.on_request_submitted(plan_b))
    assert second_dispatch == []

    assert list(policy.on_worker_event(_event(task=req_a_t0, kind=WorkerEventKind.TASK_LAUNCH_END))) == []

    third_dispatch = list(policy.on_tasks_runnable([req_a_t1]))
    assert [task.task_id for task in third_dispatch] == [req_b_t0.task_id]

    assert len(calls) == 3
    assert {seq_len for _, seq_len, _ in calls} == {10, 100}
    assert {stage for _, _, stage in calls} == {TaskKind.DIT_STEP_CHUNK.value}


def test_sjf_is_non_preemptive_and_keeps_running_request() -> None:
    # Same scenario as the SRTF preemption test: a long request (req-a) is in
    # flight when a much shorter request (req-b) becomes runnable. SRTF would
    # preempt at the task boundary and run req-b; non-preemptive SJF must keep
    # running req-a until it finishes.
    policy = SJFSchedulerPolicy(
        topology=_build_topology(),
        task_runtime_estimator=lambda _parallel_spec, seq_len, _stage: float(seq_len),
    )

    req_a_t0 = _build_task(request_id="req-a", task_idx=0, seq_len=100)
    req_a_t1 = _build_task(
        request_id="req-a",
        task_idx=1,
        dependencies=(req_a_t0.task_id,),
        seq_len=100,
    )
    req_b_t0 = _build_task(request_id="req-b", task_idx=0, seq_len=10)

    plan_a = RequestExecutionPlan(
        request_id="req-a",
        tasks={req_a_t0.task_id: req_a_t0, req_a_t1.task_id: req_a_t1},
        terminal_task_ids=(req_a_t1.task_id,),
    )
    plan_b = RequestExecutionPlan(
        request_id="req-b",
        tasks={req_b_t0.task_id: req_b_t0},
        terminal_task_ids=(req_b_t0.task_id,),
    )

    first_dispatch = list(policy.on_request_submitted(plan_a))
    assert [task.task_id for task in first_dispatch] == [req_a_t0.task_id]

    # req-b arrives while req-a is running; group is busy so nothing dispatches.
    assert list(policy.on_request_submitted(plan_b)) == []

    assert list(policy.on_worker_event(_event(task=req_a_t0, kind=WorkerEventKind.TASK_LAUNCH_END))) == []

    # req-a's next task becomes runnable. Non-preemptive: keep serving req-a
    # even though req-b is shorter and waiting.
    third_dispatch = list(policy.on_tasks_runnable([req_a_t1]))
    assert [task.task_id for task in third_dispatch] == [req_a_t1.task_id]


def test_sjf_admits_smallest_total_job_when_group_frees() -> None:
    # Three single-task requests bound to one group. A long request runs first;
    # while it is busy, a medium then a small request arrive. When the group
    # frees, SJF must admit the smallest *total* job (small) rather than the
    # earliest arrival (medium) -- proving selection-by-job-size, not FIFO.
    policy = SJFSchedulerPolicy(
        topology=_build_topology(),
        task_runtime_estimator=lambda _parallel_spec, seq_len, _stage: float(seq_len),
    )

    big = _build_task(request_id="req-big", task_idx=0, seq_len=100)
    mid = _build_task(request_id="req-mid", task_idx=0, seq_len=50)
    small = _build_task(request_id="req-small", task_idx=0, seq_len=5)
    plans = {
        rid: RequestExecutionPlan(
            request_id=rid, tasks={t.task_id: t}, terminal_task_ids=(t.task_id,)
        )
        for rid, t in (("req-big", big), ("req-mid", mid), ("req-small", small))
    }

    assert [t.task_id for t in policy.on_request_submitted(plans["req-big"])] == [big.task_id]
    # mid arrives before small, both while the group is busy with big.
    assert list(policy.on_request_submitted(plans["req-mid"])) == []
    assert list(policy.on_request_submitted(plans["req-small"])) == []

    # Finish big: launch + exec end, then request-finished frees the group.
    list(policy.on_worker_event(_event(task=big, kind=WorkerEventKind.TASK_LAUNCH_END)))
    assert list(policy.on_worker_event(_event(task=big, kind=WorkerEventKind.TASK_EXEC_END))) == []
    freed_dispatch = list(
        policy.on_worker_event(
            WorkerEvent(
                event_id="request:req-big:request_finished",
                task_id="",
                request_id="req-big",
                group_id="g0",
                worker_rank=-1,
                kind=WorkerEventKind.REQUEST_FINISHED,
                timestamp_ns=time.monotonic_ns(),
            )
        )
    )
    assert [t.task_id for t in freed_dispatch] == [small.task_id]


def test_sjf_breaks_equal_work_ties_by_arrival_not_request_id() -> None:
    # Regression: equal-work requests must be ordered FIFO (submission order), not
    # by lexicographic request_id (a random UUID in production -> starvation/
    # non-determinism). "req-zzz" arrives before "req-aaa" with identical total
    # work; if the policy tie-broke on request_id, "req-aaa" would win.
    policy = SJFSchedulerPolicy(
        topology=_build_topology(),
        task_runtime_estimator=lambda _parallel_spec, seq_len, _stage: float(seq_len),
    )
    big = _build_task(request_id="req-big", task_idx=0, seq_len=100)
    early = _build_task(request_id="req-zzz", task_idx=0, seq_len=10)
    late = _build_task(request_id="req-aaa", task_idx=0, seq_len=10)
    plans = {
        rid: RequestExecutionPlan(request_id=rid, tasks={t.task_id: t}, terminal_task_ids=(t.task_id,))
        for rid, t in (("req-big", big), ("req-zzz", early), ("req-aaa", late))
    }
    policy.on_request_submitted(plans["req-big"])
    assert list(policy.on_request_submitted(plans["req-zzz"])) == []  # arrives first
    assert list(policy.on_request_submitted(plans["req-aaa"])) == []  # arrives second
    list(policy.on_worker_event(_event(task=big, kind=WorkerEventKind.TASK_LAUNCH_END)))
    list(policy.on_worker_event(_event(task=big, kind=WorkerEventKind.TASK_EXEC_END)))
    freed = list(
        policy.on_worker_event(
            WorkerEvent(
                event_id="request:req-big:request_finished",
                task_id="",
                request_id="req-big",
                group_id="g0",
                worker_rank=-1,
                kind=WorkerEventKind.REQUEST_FINISHED,
                timestamp_ns=time.monotonic_ns(),
            )
        )
    )
    assert [t.request_id for t in freed] == ["req-zzz"]  # earliest arrival, not "req-aaa"


def test_srtf_runtime_accounting_is_idempotent_and_cleans_state() -> None:
    policy = SRTFSchedulerPolicy(
        topology=_build_topology(),
        task_runtime_estimator=lambda _parallel_spec, seq_len, _stage: float(seq_len),
    )

    t0 = _build_task(request_id="req-1", task_idx=0, seq_len=7)
    t1 = _build_task(request_id="req-1", task_idx=1, dependencies=(t0.task_id,), seq_len=5)
    plan = RequestExecutionPlan(
        request_id="req-1",
        tasks={t0.task_id: t0, t1.task_id: t1},
        terminal_task_ids=(t1.task_id,),
    )

    list(policy.on_request_submitted(plan))
    assert policy.request_remaining_work["req-1"] == 12.0

    list(policy.on_worker_event(_event(task=t0, kind=WorkerEventKind.TASK_LAUNCH_END)))
    assert policy.request_remaining_work["req-1"] == 5.0

    list(policy.on_worker_event(_event(task=t0, kind=WorkerEventKind.TASK_EXEC_END)))
    assert policy.request_remaining_work["req-1"] == 5.0

    list(
        policy.on_worker_event(
            WorkerEvent(
                event_id="request:req-1:request_finished",
                task_id="",
                request_id="req-1",
                group_id="g0",
                worker_rank=-1,
                kind=WorkerEventKind.REQUEST_FINISHED,
                timestamp_ns=time.monotonic_ns(),
            )
        )
    )
    assert "req-1" not in policy.request_remaining_work
    assert "req-1" not in policy.request_task_ids


def test_srtf_rejects_negative_estimate() -> None:
    policy = SRTFSchedulerPolicy(
        topology=_build_topology(),
        task_runtime_estimator=lambda _parallel_spec, _seq_len, _stage: -1.0,
    )
    task = _build_task(request_id="req-x", task_idx=0, seq_len=4)
    plan = RequestExecutionPlan(
        request_id="req-x",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
    )

    with pytest.raises(ValueError, match="must be >= 0"):
        list(policy.on_request_submitted(plan))


def test_fcfs_binds_group_at_submit_by_earliest_free() -> None:
    policy = FCFSSchedulerPolicy(
        topology=_build_two_group_topology(),
        task_runtime_estimator=lambda parallel_spec, seq_len, _stage: float(seq_len) / float(parallel_spec.tp),
    )
    req_a_t0 = _build_task(request_id="req-a", task_idx=0, seq_len=100)
    req_b_t0 = _build_task(request_id="req-b", task_idx=0, seq_len=60)
    plan_a = RequestExecutionPlan(
        request_id="req-a",
        tasks={req_a_t0.task_id: req_a_t0},
        terminal_task_ids=(req_a_t0.task_id,),
    )
    plan_b = RequestExecutionPlan(
        request_id="req-b",
        tasks={req_b_t0.task_id: req_b_t0},
        terminal_task_ids=(req_b_t0.task_id,),
    )

    first_dispatch = list(policy.on_request_submitted(plan_a))
    second_dispatch = list(policy.on_request_submitted(plan_b))

    assert first_dispatch[0].group_id == "g0"
    assert second_dispatch[0].group_id == "g1"
    assert policy.request_group["req-a"] == "g0"
    assert policy.request_group["req-b"] == "g1"


def test_fcfs_keeps_explicit_group_id() -> None:
    policy = FCFSSchedulerPolicy(
        topology=_build_two_group_topology(),
        task_runtime_estimator=lambda _parallel_spec, seq_len, _stage: float(seq_len),
    )
    pinned_task = _build_task(request_id="req-pin", task_idx=0, seq_len=20)
    pinned_task.group_id = "g1"
    pinned_plan = RequestExecutionPlan(
        request_id="req-pin",
        tasks={pinned_task.task_id: pinned_task},
        terminal_task_ids=(pinned_task.task_id,),
    )

    dispatch = list(policy.on_request_submitted(pinned_plan))

    assert dispatch[0].group_id == "g1"
    assert policy.request_group["req-pin"] == "g1"


def test_srtf_binds_group_at_submit_by_earliest_free() -> None:
    policy = SRTFSchedulerPolicy(
        topology=_build_two_group_topology(),
        task_runtime_estimator=lambda parallel_spec, seq_len, _stage: float(seq_len) / float(parallel_spec.tp),
    )
    req_a_t0 = _build_task(request_id="req-a", task_idx=0, seq_len=100)
    req_b_t0 = _build_task(request_id="req-b", task_idx=0, seq_len=60)
    plan_a = RequestExecutionPlan(
        request_id="req-a",
        tasks={req_a_t0.task_id: req_a_t0},
        terminal_task_ids=(req_a_t0.task_id,),
    )
    plan_b = RequestExecutionPlan(
        request_id="req-b",
        tasks={req_b_t0.task_id: req_b_t0},
        terminal_task_ids=(req_b_t0.task_id,),
    )

    first_dispatch = list(policy.on_request_submitted(plan_a))
    second_dispatch = list(policy.on_request_submitted(plan_b))

    assert first_dispatch[0].group_id == "g0"
    assert second_dispatch[0].group_id == "g1"
    assert policy.request_group["req-a"] == "g0"
    assert policy.request_group["req-b"] == "g1"


def test_srtf_keeps_explicit_group_id() -> None:
    policy = SRTFSchedulerPolicy(
        topology=_build_two_group_topology(),
        task_runtime_estimator=lambda _parallel_spec, seq_len, _stage: float(seq_len),
    )
    pinned_task = _build_task(request_id="req-pin", task_idx=0, seq_len=20)
    pinned_task.group_id = "g1"
    pinned_plan = RequestExecutionPlan(
        request_id="req-pin",
        tasks={pinned_task.task_id: pinned_task},
        terminal_task_ids=(pinned_task.task_id,),
    )

    dispatch = list(policy.on_request_submitted(pinned_plan))

    assert dispatch[0].group_id == "g1"
    assert policy.request_group["req-pin"] == "g1"


def test_edf_greedy_builds_dynamic_dit_group_from_free_ranks() -> None:
    topology = _build_edf_topology(worker_count=4)
    policy = EdfGreedySchedulerPolicy(topology=topology, allowed_sp_sizes=(4, 2, 1))
    policy.outstanding_per_rank[0] = 1
    task = _build_task(request_id="req-dyn", task_idx=0, seq_len=16)
    plan = RequestExecutionPlan(
        request_id="req-dyn",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
    )

    dispatched = list(policy.on_request_submitted(plan))

    assert len(dispatched) == 1
    assert dispatched[0].group_id == "edf_sp2_r1_2"
    assert topology.get_group("edf_sp2_r1_2").ranks == (1, 2)
    assert topology.get_group("edf_sp2_r1_2").parallel_spec.sp == 2


def test_edf_greedy_uses_single_rank_group_for_vae() -> None:
    topology = _build_edf_topology(worker_count=4)
    policy = EdfGreedySchedulerPolicy(topology=topology, allowed_sp_sizes=(4, 2, 1))
    task = _build_task(
        request_id="req-vae",
        task_idx=0,
        seq_len=16,
        kind=TaskKind.VAE_DECODE,
    )
    plan = RequestExecutionPlan(
        request_id="req-vae",
        tasks={task.task_id: task},
        terminal_task_ids=(task.task_id,),
    )

    dispatched = list(policy.on_request_submitted(plan))

    assert len(dispatched) == 1
    assert dispatched[0].group_id == "g_sp1_r0"
    assert topology.get_group(dispatched[0].group_id).ranks == (0,)


def test_edf_greedy_orders_ready_requests_by_deadline() -> None:
    topology = _build_edf_topology(worker_count=2)
    policy = EdfGreedySchedulerPolicy(topology=topology, allowed_sp_sizes=(1,))
    policy.outstanding_per_rank[0] = 1
    policy.outstanding_per_rank[1] = 1

    late = _build_task(request_id="req-late", task_idx=0, seq_len=16, kind=TaskKind.VAE_DECODE)
    early = _build_task(request_id="req-early", task_idx=0, seq_len=16, kind=TaskKind.VAE_DECODE)
    late_plan = RequestExecutionPlan(
        request_id="req-late",
        tasks={late.task_id: late},
        terminal_task_ids=(late.task_id,),
        initial_artifacts=(_initial_artifact("req-late", deadline_ms=200.0, priority=10),),
    )
    early_plan = RequestExecutionPlan(
        request_id="req-early",
        tasks={early.task_id: early},
        terminal_task_ids=(early.task_id,),
        initial_artifacts=(_initial_artifact("req-early", deadline_ms=100.0, priority=0),),
    )

    assert list(policy.on_request_submitted(late_plan)) == []
    assert list(policy.on_request_submitted(early_plan)) == []

    policy.outstanding_per_rank.pop(0)
    dispatched = list(policy.on_worker_event(_event(task=late, kind=WorkerEventKind.TASK_LAUNCH_END)))

    assert [task.request_id for task in dispatched] == ["req-early"]
    assert dispatched[0].group_id == "g_sp1_r0"
