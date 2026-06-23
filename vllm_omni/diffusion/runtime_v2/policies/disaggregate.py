# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.interfaces import SchedulerPolicy
from vllm_omni.diffusion.runtime_v2.protocol import (
    RESHARD_PAYLOAD_DST_GROUP_ID,
    RESHARD_PAYLOAD_SRC_GROUP_ID,
    InferenceTask,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


from vllm_omni.diffusion.runtime_v2.adapters.common import STAGE_AUX, STAGE_DIT

class DisaggregateSchedulerPolicy(SchedulerPolicy):
    """Static stage-disaggregated FCFS policy.

    Allows a single request to span multiple execution groups (one per stage):
    policies assign ``group_id`` as tasks become runnable, and GlobalScheduler
    migrates worker-local inputs on demand when a consumer runs on a different
    group from its producer. Dedicated RESHARD tasks are still supported for
    plans that explicitly contain them.

    Scheduling is intentionally minimal: per-group FIFO ready queue plus a
    dedicated single-slot RESHARD queue. This policy is independent from the
    request-level binding semantics of FCFS / SRTF and never falls back to
    them.
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        *,
        stage_group_ids: dict[str, str] | None = None,
        dit_step_schedule: tuple[dict[str, Any], ...] | None = None,
    ) -> None:
        self.topology = topology
        self.stage_group_ids = dict(stage_group_ids) if stage_group_ids else None
        self.dit_step_schedule = tuple(dict(item) for item in dit_step_schedule or ())
        self.group_ready: dict[str, deque[InferenceTask]] = {}
        self.reshard_ready: deque[InferenceTask] = deque()
        # Outstanding (dispatched-but-not-finished) task counters.
        self.outstanding_per_group: dict[str, int] = {}
        # Per-group exclusive reservation by an in-flight RESHARD. A RESHARD
        # claims BOTH its src_group_id and dst_group_id while running; normal
        # task dispatch on either of those groups is gated on the hold being 0.
        # This prevents NCCL P2P (migration data plane) from racing with the
        # compute kernels of an unrelated task on the same group, which in
        # earlier versions could deadlock NCCL.
        self.reshard_holds: dict[str, int] = {}
        # task_id -> (src_group_id, dst_group_id) for in-flight RESHARD tasks,
        # used by _release_task to drop the two reshard_holds entries.
        self.reshard_groups_by_task: dict[str, tuple[str, str]] = {}
        self.outstanding_reshards = 0
        # Conservative per-group concurrency for the first version. Each group
        # runs one task at a time so we don't oversubscribe a worker pool that
        # was sized for a single in-flight task per group.
        self.max_concurrent_per_group = 1
        self.max_concurrent_reshards = 1
        # task_id -> "__reshard__" or group_id, used to release counters on finish.
        self.dispatched_kind: dict[str, str] = {}
        self._group_order = {group.group_id: idx for idx, group in enumerate(self.topology.groups)}

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        for task in tasks:
            if task.group_id is None:
                task.group_id = self._select_group_for_task(task)
            self._validate_task(task)
            task.status = TaskStatus.READY
            if task.kind == TaskKind.RESHARD:
                self.reshard_ready.append(task)
            else:
                assert task.group_id is not None  # validated above
                self.group_ready.setdefault(task.group_id, deque()).append(task)
        return self._take_dispatchable_tasks()

    def _select_group_for_task(self, task: InferenceTask) -> str | None:
        if self.stage_group_ids is None:
            return None
        if task.kind in (TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE):
            return self.stage_group_ids.get(STAGE_AUX)
        if task.kind in (TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK):
            return self.stage_group_ids.get(STAGE_DIT)
        return None

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        if event.kind in (
            WorkerEventKind.TASK_EXEC_END,
            WorkerEventKind.TASK_FAILED,
        ):
            self._release_task(event.task_id)
        if event.kind in (
            WorkerEventKind.REQUEST_FINISHED,
            WorkerEventKind.REQUEST_FAILED,
        ):
            self._drop_request(event.request_id)
        return self._take_dispatchable_tasks()

    def _validate_task(self, task: InferenceTask) -> None:
        if task.kind == TaskKind.RESHARD:
            if task.group_id is not None:
                raise ValueError(
                    f"RESHARD task {task.task_id} must have group_id=None, got {task.group_id!r}"
                )
            src = task.payload.get(RESHARD_PAYLOAD_SRC_GROUP_ID)
            dst = task.payload.get(RESHARD_PAYLOAD_DST_GROUP_ID)
            if not isinstance(src, str) or not src:
                raise ValueError(
                    f"RESHARD task {task.task_id} requires payload[{RESHARD_PAYLOAD_SRC_GROUP_ID!r}]"
                )
            if not isinstance(dst, str) or not dst:
                raise ValueError(
                    f"RESHARD task {task.task_id} requires payload[{RESHARD_PAYLOAD_DST_GROUP_ID!r}]"
                )
            if src == dst:
                raise ValueError(
                    f"RESHARD task {task.task_id} src_group_id == dst_group_id == {src!r}"
                )
            # Validate referenced groups exist; surfaces topology mistakes early.
            self.topology.get_group(src)
            self.topology.get_group(dst)
            return
        if task.group_id is None:
            raise ValueError(
                f"DisaggregateSchedulerPolicy requires explicit group_id for non-RESHARD tasks; "
                f"task {task.task_id} kind={task.kind.value} has group_id=None"
            )
        group = self.topology.get_group(task.group_id)
        if task.kind not in group.supported_task_kinds:
            raise ValueError(
                f"task {task.task_id} kind={task.kind.value} is not supported by group {task.group_id!r} "
                f"(supports {[k.value for k in group.supported_task_kinds]!r})"
            )

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        out: list[InferenceTask] = []
        # Drain RESHARD first. RESHARD requires its src AND dst groups to be
        # completely idle (no normal task running, no other RESHARD holding
        # either group). If the head of the queue cannot be dispatched, we
        # don't peek further — keep the FIFO order so an earlier RESHARD can't
        # starve waiting for one of its groups while a later RESHARD with a
        # different pair sneaks past. This is intentionally conservative for
        # the first version; more sophisticated multi-RESHARD scheduling can
        # build on top.
        while (
            self.outstanding_reshards < self.max_concurrent_reshards
            and self.reshard_ready
        ):
            task = self.reshard_ready[0]
            src = task.payload[RESHARD_PAYLOAD_SRC_GROUP_ID]
            dst = task.payload[RESHARD_PAYLOAD_DST_GROUP_ID]
            if not self._can_dispatch_reshard(src, dst):
                break
            self.reshard_ready.popleft()
            task.status = TaskStatus.DISPATCHED
            self.outstanding_reshards += 1
            self.reshard_holds[src] = self.reshard_holds.get(src, 0) + 1
            self.reshard_holds[dst] = self.reshard_holds.get(dst, 0) + 1
            self.reshard_groups_by_task[task.task_id] = (src, dst)
            self.dispatched_kind[task.task_id] = "__reshard__"
            out.append(task)

        # Stable group iteration order for determinism in tests.
        for group in self.topology.groups:
            group_id = group.group_id
            queue = self.group_ready.get(group_id)
            if not queue:
                continue
            while (
                queue
                and self.outstanding_per_group.get(group_id, 0) < self.max_concurrent_per_group
                and self.reshard_holds.get(group_id, 0) == 0
            ):
                task = queue.popleft()
                task.status = TaskStatus.DISPATCHED
                self.outstanding_per_group[group_id] = self.outstanding_per_group.get(group_id, 0) + 1
                self.dispatched_kind[task.task_id] = group_id
                out.append(task)
            if not queue:
                self.group_ready.pop(group_id, None)
        return out

    def _can_dispatch_reshard(self, src_group_id: str, dst_group_id: str) -> bool:
        # Both endpoints must be completely free: no normal task currently
        # executing AND no other RESHARD currently holding either group.
        for g in (src_group_id, dst_group_id):
            if self.outstanding_per_group.get(g, 0) > 0:
                return False
            if self.reshard_holds.get(g, 0) > 0:
                return False
        return True

    def _release_task(self, task_id: str) -> None:
        target = self.dispatched_kind.pop(task_id, None)
        if target is None:
            return
        if target == "__reshard__":
            self.outstanding_reshards = max(0, self.outstanding_reshards - 1)
            groups = self.reshard_groups_by_task.pop(task_id, None)
            if groups is not None:
                for g in groups:
                    held = self.reshard_holds.get(g, 0)
                    if held <= 1:
                        self.reshard_holds.pop(g, None)
                    else:
                        self.reshard_holds[g] = held - 1
            return
        remaining = self.outstanding_per_group.get(target, 0)
        if remaining <= 1:
            self.outstanding_per_group.pop(target, None)
        else:
            self.outstanding_per_group[target] = remaining - 1

    def _drop_request(self, request_id: str) -> None:
        for group_id in list(self.group_ready.keys()):
            queue = self.group_ready[group_id]
            if not any(task.request_id == request_id for task in queue):
                continue
            self.group_ready[group_id] = deque(task for task in queue if task.request_id != request_id)
            if not self.group_ready[group_id]:
                self.group_ready.pop(group_id)
        if any(task.request_id == request_id for task in self.reshard_ready):
            self.reshard_ready = deque(task for task in self.reshard_ready if task.request_id != request_id)


class DynamicStepFCFSSchedulerPolicy(DisaggregateSchedulerPolicy):
    """FCFS policy for dynamic DiT step group switching.

    This policy intentionally has its own name/entrypoint instead of extending
    the user-visible meaning of ``disaggregate``. It accepts explicit per-task
    group placement from the compiler and uses rank-level resource accounting,
    so overlapping execution groups are mutually exclusive while disjoint groups
    can still make progress independently.
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        *,
        stage_group_ids: dict[str, str] | None = None,
        dit_step_schedule: tuple[dict[str, Any], ...] | None = None,
    ) -> None:
        super().__init__(
            topology=topology,
            stage_group_ids=stage_group_ids,
            dit_step_schedule=dit_step_schedule,
        )
        self.outstanding_per_rank: dict[int, int] = {}
        self.reshard_holds_by_rank: dict[int, int] = {}
        self.dispatched_ranks_by_task: dict[str, tuple[int, ...]] = {}

    def _select_group_for_task(self, task: InferenceTask) -> str | None:
        if task.kind == TaskKind.DIT_STEP_CHUNK and self.dit_step_schedule:
            if task.step_range is None:
                raise ValueError(f"dynamic_step_fcfs requires step_range for task {task.task_id}")
            step = task.step_range.start
            for item in self.dit_step_schedule:
                start = int(item.get("start", 0))
                end_raw = item.get("end")
                end = float("inf") if end_raw is None else int(end_raw)
                if start <= step < end:
                    return str(item["group_id"])
            raise ValueError(f"dynamic_step_fcfs has no DiT group for step {step}")
        return super()._select_group_for_task(task)

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        out: list[InferenceTask] = []
        while (
            self.outstanding_reshards < self.max_concurrent_reshards
            and self.reshard_ready
        ):
            task = self.reshard_ready[0]
            src = task.payload[RESHARD_PAYLOAD_SRC_GROUP_ID]
            dst = task.payload[RESHARD_PAYLOAD_DST_GROUP_ID]
            if not self._can_dispatch_reshard(src, dst):
                break
            self.reshard_ready.popleft()
            task.status = TaskStatus.DISPATCHED
            ranks = tuple(sorted(set(self.topology.get_group(src).ranks) | set(self.topology.get_group(dst).ranks)))
            self._claim_ranks(ranks, reshard=True)
            self.outstanding_reshards += 1
            self.reshard_groups_by_task[task.task_id] = (src, dst)
            self.dispatched_kind[task.task_id] = "__reshard__"
            self.dispatched_ranks_by_task[task.task_id] = ranks
            out.append(task)

        for group in self.topology.groups:
            group_id = group.group_id
            queue = self.group_ready.get(group_id)
            if not queue:
                continue
            ranks = tuple(group.ranks)
            while queue and self._can_dispatch_ranks(ranks):
                task = queue.popleft()
                task.status = TaskStatus.DISPATCHED
                self._claim_ranks(ranks, reshard=False)
                self.outstanding_per_group[group_id] = self.outstanding_per_group.get(group_id, 0) + 1
                self.dispatched_kind[task.task_id] = group_id
                self.dispatched_ranks_by_task[task.task_id] = ranks
                out.append(task)
            if not queue:
                self.group_ready.pop(group_id, None)
        return out

    def _can_dispatch_reshard(self, src_group_id: str, dst_group_id: str) -> bool:
        ranks = tuple(
            sorted(
                set(self.topology.get_group(src_group_id).ranks)
                | set(self.topology.get_group(dst_group_id).ranks)
            )
        )
        return self._can_dispatch_ranks(ranks)

    def _can_dispatch_ranks(self, ranks: tuple[int, ...]) -> bool:
        for rank in ranks:
            if self.outstanding_per_rank.get(rank, 0) > 0:
                return False
            if self.reshard_holds_by_rank.get(rank, 0) > 0:
                return False
        return True

    def _claim_ranks(self, ranks: tuple[int, ...], *, reshard: bool) -> None:
        target = self.reshard_holds_by_rank if reshard else self.outstanding_per_rank
        for rank in ranks:
            target[rank] = target.get(rank, 0) + 1

    def acquire_migration_hold(self, ranks: tuple[int, ...]) -> None:
        """Reserve ranks against future dispatch for the duration of an
        asynchronous migration. Mirrors the RESHARD task's rank reservation but
        is driven by the scheduler for the implicit-migration path. Held ranks
        are excluded from both dispatch eligibility (_can_dispatch_ranks) and
        the scheduler-visible free-rank view (_free_rank_set) via
        reshard_holds_by_rank."""
        self._claim_ranks(ranks, reshard=True)

    def release_migration_hold(self, ranks: tuple[int, ...]) -> None:
        """Release a hold taken by acquire_migration_hold; decrements
        reshard_holds_by_rank for each rank."""
        self._release_rank_counts(self.reshard_holds_by_rank, ranks)

    @staticmethod
    def _release_rank_counts(target: dict[int, int], ranks: tuple[int, ...]) -> None:
        for rank in ranks:
            remaining = target.get(rank, 0)
            if remaining <= 1:
                target.pop(rank, None)
            else:
                target[rank] = remaining - 1

    def _release_task(self, task_id: str) -> None:
        target = self.dispatched_kind.pop(task_id, None)
        if target is None:
            return
        ranks = self.dispatched_ranks_by_task.pop(task_id, ())
        if target == "__reshard__":
            self.outstanding_reshards = max(0, self.outstanding_reshards - 1)
            self.reshard_groups_by_task.pop(task_id, None)
            self._release_rank_counts(self.reshard_holds_by_rank, ranks)
            return
        remaining = self.outstanding_per_group.get(target, 0)
        if remaining <= 1:
            self.outstanding_per_group.pop(target, None)
        else:
            self.outstanding_per_group[target] = remaining - 1
        self._release_rank_counts(self.outstanding_per_rank, ranks)
