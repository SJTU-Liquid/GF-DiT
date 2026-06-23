# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.runtime_v2.data_plane import (
    _MIGRATE_PHASE_ACCEPT_SCHEDULE,
    _MIGRATE_PHASE_BUILD_SCHEDULE,
    _MIGRATE_PHASE_DESCRIBE_LAYOUT,
    _MIGRATE_PHASE_EXECUTE_TRANSFER,
)
from vllm_omni.diffusion.runtime_v2.multiproc_worker import (
    MigrateArtifactsRankResult,
    MultiprocWorkerPool,
)
from vllm_omni.diffusion.runtime_v2.policies.edf_greedy import EdfGreedySchedulerPolicy
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ExecutionGroupSpec,
    InferenceTask,
    MigrateArtifactsPlan,
    MigrateArtifactSpec,
    ParallelSpec,
    RequestExecutionPlan,
    RESHARD_PAYLOAD_DST_GROUP_ID,
    RESHARD_PAYLOAD_SRC_GROUP_ID,
    TaskKind,
    TaskStatus,
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

_ALL_TASK_KINDS: tuple[TaskKind, ...] = (
    TaskKind.TEXT_ENCODE,
    TaskKind.DIT_PREPARE,
    TaskKind.TIMESTEP_PREPARE,
    TaskKind.DIT_STEP_CHUNK,
    TaskKind.VAE_DECODE,
    TaskKind.FINALIZE,
)


def _topology(n: int) -> RuntimeTopology:
    groups: list[ExecutionGroupSpec] = []
    for rank in range(n):
        groups.append(
            ExecutionGroupSpec(
                group_id=f"g_sp1_r{rank}",
                ranks=(rank,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=_ALL_TASK_KINDS,
            )
        )
    return RuntimeTopology(
        workers=tuple(WorkerSpec(worker_rank=r, device_id=r) for r in range(n)),
        groups=tuple(groups),
    )


def test_migration_hold_removes_then_restores_free_ranks() -> None:
    policy = EdfGreedySchedulerPolicy(_topology(4))
    assert policy._free_rank_set() == {0, 1, 2, 3}

    policy.acquire_migration_hold((1, 2))
    assert policy._free_rank_set() == {0, 3}

    policy.release_migration_hold((1, 2))
    assert policy._free_rank_set() == {0, 1, 2, 3}


class _StaticCompiler:
    def __init__(self, plan):
        self.plan = plan

    def compile_request(self, _request):
        return self.plan


class _CapturingPool:
    """Worker pool double that captures begin_migration callbacks so a test can
    complete/fail the migration explicitly."""
    def __init__(self) -> None:
        self.begun: list = []
        self.dispatched: list = []

    def begin_migration(self, plan, *, on_done, on_error, timeout_s=None):
        self.begun.append((plan, on_done, on_error))
        return f"m{len(self.begun)}"

    def pump_migrations(self):
        pass

    def has_pending_migrations(self):
        return False

    def dispatch(self, task, inline_inputs, release_after_exec_artifact_ids=()):
        self.dispatched.append(task)

    def poll(self, timeout_s=0.0):
        return []

    def evict_request(self, request_id):
        pass


def _two_group_topology() -> RuntimeTopology:
    return RuntimeTopology(
        workers=(WorkerSpec(worker_rank=0, device_id=0), WorkerSpec(worker_rank=1, device_id=1)),
        groups=(
            ExecutionGroupSpec(group_id="src", ranks=(0,),
                               parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                               supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,)),
            ExecutionGroupSpec(group_id="dst", ranks=(1,),
                               parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                               supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,)),
        ),
    )


def _migration_plan() -> MigrateArtifactsPlan:
    handle = ArtifactHandle(request_id="r", artifact_id="a", kind=ArtifactKind.TENSOR,
                            layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    return MigrateArtifactsPlan(
        request_id="r", src_group_id="src", dst_group_id="dst",
        artifacts=(MigrateArtifactSpec(handle=handle),), request_metadata={},
    )


def _drive_pool_migration(pool, results_by_phase):
    """Feed each phase's results into the migration queues, then pump."""
    for phase in (_MIGRATE_PHASE_DESCRIBE_LAYOUT, _MIGRATE_PHASE_BUILD_SCHEDULE,
                  _MIGRATE_PHASE_ACCEPT_SCHEDULE, _MIGRATE_PHASE_EXECUTE_TRANSFER):
        pool.pump_migrations()  # sends this phase
        for rank, result in results_by_phase[phase]:
            pool._migration_result_queues[rank].put(result)
    pool.pump_migrations()  # consume EXECUTE replies -> on_done


def test_pool_begin_migration_runs_async_and_calls_on_done() -> None:
    pool = MultiprocWorkerPool(topology=_two_group_topology(), od_config=Mock())
    # No real processes: stub command pipes + per-rank migration queues.
    pool.worker_handles = {r: Mock() for r in (0, 1)}
    pool._migration_result_queues = {0: queue.Queue(), 1: queue.Queue()}
    pool._raise_reader_error = lambda: None

    done: list[object] = []
    pool.begin_migration(_migration_plan(), on_done=lambda s: done.append(s),
                         on_error=lambda e: done.append(e))

    # The engine assigns its own migrate_id; capture it after begin.
    migrate_id = next(iter(pool._migration_engine._jobs))
    def res2(rank, phase, **kw):
        return MigrateArtifactsRankResult(migrate_id=migrate_id, request_id="r",
                                          worker_rank=rank, phase=phase, **kw)
    _drive_pool_migration(pool, {
        _MIGRATE_PHASE_DESCRIBE_LAYOUT: [(0, res2(0, _MIGRATE_PHASE_DESCRIBE_LAYOUT)),
                                         (1, res2(1, _MIGRATE_PHASE_DESCRIBE_LAYOUT))],
        _MIGRATE_PHASE_BUILD_SCHEDULE: [(0, res2(0, _MIGRATE_PHASE_BUILD_SCHEDULE, schedule="S"))],
        _MIGRATE_PHASE_ACCEPT_SCHEDULE: [(0, res2(0, _MIGRATE_PHASE_ACCEPT_SCHEDULE)),
                                         (1, res2(1, _MIGRATE_PHASE_ACCEPT_SCHEDULE))],
        _MIGRATE_PHASE_EXECUTE_TRANSFER: [(0, res2(0, _MIGRATE_PHASE_EXECUTE_TRANSFER)),
                                          (1, res2(1, _MIGRATE_PHASE_EXECUTE_TRANSFER))],
    })
    assert done == ["S"]
    # DESCRIBE command was actually sent to both participants' command pipes.
    assert pool.worker_handles[0].command_pipe_w.send.called
    assert pool.worker_handles[1].command_pipe_w.send.called


def test_reshard_task_migration_is_async_and_completes_on_callback() -> None:
    handle = ArtifactHandle(request_id="r", artifact_id="lat", kind=ArtifactKind.TENSOR,
                            layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    reshard = InferenceTask(
        task_id="r:reshard:0", request_id="r", kind=TaskKind.RESHARD,
        group_id=None, parallel_spec=ParallelSpec(), status=TaskStatus.READY,
        inputs=(handle,), outputs=(handle,),
        payload={RESHARD_PAYLOAD_SRC_GROUP_ID: "src", RESHARD_PAYLOAD_DST_GROUP_ID: "dst"},
    )
    plan = RequestExecutionPlan(request_id="r", tasks={reshard.task_id: reshard},
                                terminal_task_ids=(reshard.task_id,), initial_artifacts=())
    pool = _CapturingPool()
    store = InMemoryArtifactStore()
    store.put(handle, WorkerLocalArtifactRef(handle=handle, group_id="src", worker_rank=0))
    scheduler = GlobalScheduler(
        topology=_two_group_topology(), worker_pool=pool,
        compiler=_StaticCompiler(plan), artifact_store=store,
        policy=FCFSSchedulerPolicy(_two_group_topology()),
    )
    scheduler.task_index[reshard.task_id] = reshard
    scheduler.plans[reshard.request_id] = plan

    scheduler._dispatch_reshard_task(reshard)
    # Async: migration started, task RUNNING, no synthetic completion yet.
    assert len(pool.begun) == 1
    assert reshard.status == TaskStatus.RUNNING

    # Completing the migration publishes the dst artifact ref.
    _plan, on_done, _on_error = pool.begun[0]
    on_done(None)
    stored = store.get(handle)
    assert isinstance(stored, WorkerLocalArtifactRef) and stored.group_id == "dst"
    # The synthetic LAUNCH_END/EXEC_END chain drives the task to completion.
    assert reshard.status == TaskStatus.FINISHED


def test_implicit_migration_parks_task_and_dispatches_after_completion() -> None:
    # Input artifact lives on group "src"; the dependent task runs on group "dst".
    handle = ArtifactHandle(request_id="r", artifact_id="lat", kind=ArtifactKind.TENSOR,
                            layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    task = InferenceTask(
        task_id="r:dit:0", request_id="r", kind=TaskKind.DIT_STEP_CHUNK,
        group_id="dst", parallel_spec=ParallelSpec(), status=TaskStatus.READY,
        inputs=(handle,),
    )
    plan = RequestExecutionPlan(request_id="r", tasks={task.task_id: task},
                                terminal_task_ids=(task.task_id,), initial_artifacts=())
    pool = _CapturingPool()
    store = InMemoryArtifactStore()
    store.put(handle, WorkerLocalArtifactRef(handle=handle, group_id="src", worker_rank=0))
    policy = FCFSSchedulerPolicy(_two_group_topology())
    holds: list = []
    policy.acquire_migration_hold = lambda ranks: holds.append(("acq", tuple(ranks)))
    policy.release_migration_hold = lambda ranks: holds.append(("rel", tuple(ranks)))
    scheduler = GlobalScheduler(
        topology=_two_group_topology(), worker_pool=pool,
        compiler=_StaticCompiler(plan), artifact_store=store, policy=policy,
    )
    scheduler.task_index[task.task_id] = task
    scheduler.plans[task.request_id] = plan

    scheduler._dispatch_tasks([task])
    # Parked: migration started, hold acquired on src∪dst = {0,1}, NOT dispatched.
    assert len(pool.begun) == 1
    assert ("acq", (0, 1)) in holds
    assert pool.dispatched == []

    # Completing the migration releases the hold and dispatches the task.
    _plan, on_done, _on_error = pool.begun[0]
    on_done(None)
    assert ("rel", (0, 1)) in holds
    assert pool.dispatched == [task]


def test_implicit_migration_error_fails_task_and_releases_hold() -> None:
    handle = ArtifactHandle(request_id="r", artifact_id="lat", kind=ArtifactKind.TENSOR,
                            layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    task = InferenceTask(
        task_id="r:dit:0", request_id="r", kind=TaskKind.DIT_STEP_CHUNK,
        group_id="dst", parallel_spec=ParallelSpec(), status=TaskStatus.READY,
        inputs=(handle,),
    )
    plan = RequestExecutionPlan(request_id="r", tasks={task.task_id: task},
                                terminal_task_ids=(task.task_id,), initial_artifacts=())
    pool = _CapturingPool()
    store = InMemoryArtifactStore()
    store.put(handle, WorkerLocalArtifactRef(handle=handle, group_id="src", worker_rank=0))
    policy = FCFSSchedulerPolicy(_two_group_topology())
    holds: list = []
    policy.acquire_migration_hold = lambda ranks: holds.append(("acq", tuple(ranks)))
    policy.release_migration_hold = lambda ranks: holds.append(("rel", tuple(ranks)))
    scheduler = GlobalScheduler(
        topology=_two_group_topology(), worker_pool=pool,
        compiler=_StaticCompiler(plan), artifact_store=store, policy=policy,
    )
    scheduler.task_index[task.task_id] = task
    scheduler.plans[task.request_id] = plan

    scheduler._dispatch_tasks([task])
    _plan, _on_done, on_error = pool.begun[0]
    on_error(RuntimeError("boom"))

    # Hold released, task NOT dispatched, request marked failed.
    assert ("rel", (0, 1)) in holds
    assert pool.dispatched == []
    assert "r" in scheduler.failed_requests
    assert "boom" in scheduler.failed_requests["r"]
    assert task.task_id not in scheduler._parked_pending


def test_implicit_migration_partial_failure_does_not_dispatch() -> None:
    # Task with TWO cross-group inputs; one migration fails, the other later
    # succeeds. The late success must release its hold but NOT dispatch.
    h1 = ArtifactHandle(request_id="r", artifact_id="a1", kind=ArtifactKind.TENSOR,
                        layout=ArtifactLayout.WORKER_LOCAL, codec_id="c")
    h2 = ArtifactHandle(request_id="r", artifact_id="a2", kind=ArtifactKind.TENSOR,
                        layout=ArtifactLayout.WORKER_LOCAL, codec_id="c")
    task = InferenceTask(
        task_id="r:dit:0", request_id="r", kind=TaskKind.DIT_STEP_CHUNK,
        group_id="dst", parallel_spec=ParallelSpec(), status=TaskStatus.READY,
        inputs=(h1, h2),
    )
    plan = RequestExecutionPlan(request_id="r", tasks={task.task_id: task},
                                terminal_task_ids=(task.task_id,), initial_artifacts=())
    pool = _CapturingPool()
    store = InMemoryArtifactStore()
    store.put(h1, WorkerLocalArtifactRef(handle=h1, group_id="src", worker_rank=0))
    store.put(h2, WorkerLocalArtifactRef(handle=h2, group_id="src", worker_rank=0))
    scheduler = GlobalScheduler(
        topology=_two_group_topology(), worker_pool=pool,
        compiler=_StaticCompiler(plan), artifact_store=store,
        policy=FCFSSchedulerPolicy(_two_group_topology()),
    )
    scheduler.task_index[task.task_id] = task
    scheduler.plans[task.request_id] = plan

    scheduler._dispatch_tasks([task])
    assert len(pool.begun) == 2          # one migration per cross-group input
    assert scheduler._parked_pending[task.task_id] == 2

    # First input's migration fails -> task fails.
    _p0, _done0, on_error0 = pool.begun[0]
    on_error0(RuntimeError("boom"))
    assert "r" in scheduler.failed_requests
    assert task.task_id not in scheduler._parked_pending

    # Second input's migration later succeeds -> must NOT dispatch the failed task.
    _p1, on_done1, _err1 = pool.begun[1]
    on_done1(None)
    assert pool.dispatched == []


def test_implicit_migration_done_skips_dispatch_if_request_failed_externally() -> None:
    handle = ArtifactHandle(request_id="r", artifact_id="lat", kind=ArtifactKind.TENSOR,
                            layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    task = InferenceTask(
        task_id="r:dit:0", request_id="r", kind=TaskKind.DIT_STEP_CHUNK,
        group_id="dst", parallel_spec=ParallelSpec(), status=TaskStatus.READY,
        inputs=(handle,),
    )
    plan = RequestExecutionPlan(request_id="r", tasks={task.task_id: task},
                                terminal_task_ids=(task.task_id,), initial_artifacts=())
    pool = _CapturingPool()
    store = InMemoryArtifactStore()
    store.put(handle, WorkerLocalArtifactRef(handle=handle, group_id="src", worker_rank=0))
    scheduler = GlobalScheduler(
        topology=_two_group_topology(), worker_pool=pool,
        compiler=_StaticCompiler(plan), artifact_store=store,
        policy=FCFSSchedulerPolicy(_two_group_topology()),
    )
    scheduler.task_index[task.task_id] = task
    scheduler.plans[task.request_id] = plan

    scheduler._dispatch_tasks([task])
    # Simulate an external failure/eviction of the request while parked.
    scheduler.failed_requests["r"] = "failed elsewhere"
    _plan, on_done, _on_error = pool.begun[0]
    on_done(None)
    # Must not dispatch onto an already-failed/evicted request, and must release
    # the parked task's policy claim by failing it (P1) instead of leaking it.
    assert pool.dispatched == []
    assert task.task_id in scheduler.failed_tasks


class _RaisingPool(_CapturingPool):
    """begin_migration raises synchronously (e.g. dead reader / missing handle)."""
    def begin_migration(self, plan, *, on_done, on_error, timeout_s=None):
        raise RuntimeError("reader dead")


def _reshard_task():
    in_h = ArtifactHandle(request_id="r", artifact_id="lat", kind=ArtifactKind.TENSOR,
                          layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    out_h = ArtifactHandle(request_id="r", artifact_id="lat_dit", kind=ArtifactKind.TENSOR,
                           layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    reshard = InferenceTask(
        task_id="r:reshard:0", request_id="r", kind=TaskKind.RESHARD, group_id=None,
        parallel_spec=ParallelSpec(), status=TaskStatus.READY,
        inputs=(in_h,), outputs=(out_h,),
        payload={RESHARD_PAYLOAD_SRC_GROUP_ID: "src", RESHARD_PAYLOAD_DST_GROUP_ID: "dst"},
    )
    plan = RequestExecutionPlan(request_id="r", tasks={reshard.task_id: reshard},
                                terminal_task_ids=(reshard.task_id,), initial_artifacts=())
    return in_h, out_h, reshard, plan


def _reshard_scheduler(pool, in_h, reshard, plan):
    store = InMemoryArtifactStore()
    store.put(in_h, WorkerLocalArtifactRef(handle=in_h, group_id="src", worker_rank=0))
    scheduler = GlobalScheduler(
        topology=_two_group_topology(), worker_pool=pool,
        compiler=_StaticCompiler(plan), artifact_store=store,
        policy=FCFSSchedulerPolicy(_two_group_topology()),
    )
    scheduler.task_index[reshard.task_id] = reshard
    scheduler.plans[reshard.request_id] = plan
    return scheduler, store


def test_reshard_done_after_external_failure_does_not_publish() -> None:
    # P2: a late RESHARD success after the request failed must NOT publish outputs
    # (would zombie-dispatch downstream); it fails the task to release the slot.
    in_h, out_h, reshard, plan = _reshard_task()
    pool = _CapturingPool()
    scheduler, store = _reshard_scheduler(pool, in_h, reshard, plan)
    scheduler._dispatch_reshard_task(reshard)
    scheduler.failed_requests["r"] = "failed elsewhere"
    _plan, on_done, _on_error = pool.begun[0]
    on_done(None)
    assert not store.is_ready(out_h)              # dst output not published
    assert reshard.task_id in scheduler.failed_tasks


def test_reshard_begin_migration_startup_error_fails_task() -> None:
    # P2: begin_migration raising synchronously must fail the task, not escape.
    in_h, out_h, reshard, plan = _reshard_task()
    pool = _RaisingPool()
    scheduler, store = _reshard_scheduler(pool, in_h, reshard, plan)
    scheduler._dispatch_reshard_task(reshard)   # must not raise
    assert reshard.task_id in scheduler.failed_tasks
    assert "r" in scheduler.failed_requests


def test_implicit_begin_migration_startup_error_fails_task_and_releases_hold() -> None:
    # P2: implicit begin_migration raising must release the hold + fail the task.
    handle = ArtifactHandle(request_id="r", artifact_id="lat", kind=ArtifactKind.TENSOR,
                            layout=ArtifactLayout.WORKER_LOCAL, codec_id="latent")
    task = InferenceTask(task_id="r:dit:0", request_id="r", kind=TaskKind.DIT_STEP_CHUNK,
                         group_id="dst", parallel_spec=ParallelSpec(), status=TaskStatus.READY,
                         inputs=(handle,))
    plan = RequestExecutionPlan(request_id="r", tasks={task.task_id: task},
                                terminal_task_ids=(task.task_id,), initial_artifacts=())
    pool = _RaisingPool()
    store = InMemoryArtifactStore()
    store.put(handle, WorkerLocalArtifactRef(handle=handle, group_id="src", worker_rank=0))
    policy = FCFSSchedulerPolicy(_two_group_topology())
    holds: list = []
    policy.acquire_migration_hold = lambda ranks: holds.append(("acq", tuple(ranks)))
    policy.release_migration_hold = lambda ranks: holds.append(("rel", tuple(ranks)))
    scheduler = GlobalScheduler(topology=_two_group_topology(), worker_pool=pool,
                                compiler=_StaticCompiler(plan), artifact_store=store, policy=policy)
    scheduler.task_index[task.task_id] = task
    scheduler.plans[task.request_id] = plan
    scheduler._dispatch_tasks([task])           # must not raise
    assert ("acq", (0, 1)) in holds and ("rel", (0, 1)) in holds
    assert task.task_id in scheduler.failed_tasks


def test_poll_once_pumps_before_poll_and_caps_timeout_when_pending() -> None:
    # P2: pump before blocking poll, and cap the poll wait while migrations pend.
    calls: list = []

    class _RecPool:
        def pump_migrations(self):
            calls.append("pump")

        def has_pending_migrations(self):
            return True

        def poll(self, timeout_s=0.0):
            calls.append(("poll", timeout_s))
            return []

    empty_plan = RequestExecutionPlan(request_id="x", tasks={}, terminal_task_ids=(),
                                      initial_artifacts=())
    scheduler = GlobalScheduler(topology=_two_group_topology(), worker_pool=_RecPool(),
                                compiler=_StaticCompiler(empty_plan),
                                artifact_store=InMemoryArtifactStore(),
                                policy=FCFSSchedulerPolicy(_two_group_topology()))
    scheduler.poll_once(timeout_s=0.05)
    assert calls[0] == "pump"                    # pumped before poll
    assert ("poll", 0.001) in calls              # poll wait capped while pending
