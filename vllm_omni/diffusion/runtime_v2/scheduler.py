# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2._env import env_flag
from vllm_omni.diffusion.runtime_v2.interfaces import ArtifactStore, SchedulerPolicy, TaskCompiler
from vllm_omni.diffusion.runtime_v2.protocol import (
    RESHARD_PAYLOAD_DST_GROUP_ID,
    RESHARD_PAYLOAD_SRC_GROUP_ID,
    ArtifactHandle,
    ArtifactValue,
    InferenceTask,
    MigrateArtifactSpec,
    MigrateArtifactsPlan,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


# Policy classes moved to policies/.  Re-exported here for back-compat
# with external imports (`from vllm_omni.diffusion.runtime_v2.scheduler
#   import FCFSSchedulerPolicy`, tests, benchmarks, etc).
from vllm_omni.diffusion.runtime_v2.policies import (  # noqa: E402, F401
    DisaggregateSchedulerPolicy,
    DynamicStepFCFSSchedulerPolicy,
    EdfBestFitSchedulerPolicy,
    EdfGreedySchedulerPolicy,
    FCFSSchedulerPolicy,
    SJFSchedulerPolicy,
    SRTFSchedulerPolicy,
    TaskRuntimeEstimator,
    WaveStressSchedulerPolicy,
    default_srtf_task_runtime_estimator,
)

logger = init_logger(__name__)

# While migrations are in flight, cap poll_once's blocking wait: migration phase
# results are routed outside the worker event queue, so worker_pool.poll() cannot
# wake on them. A short cap lets the loop pump them promptly.
_MIGRATION_PENDING_POLL_TIMEOUT_S = 0.001


class InMemoryArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], Any] = {}

    def put(self, artifact: ArtifactHandle, value: Any) -> None:
        self._values[(artifact.request_id, artifact.artifact_id)] = value

    def get(self, artifact: ArtifactHandle) -> Any:
        return self._values[(artifact.request_id, artifact.artifact_id)]

    def is_ready(self, artifact: ArtifactHandle) -> bool:
        return (artifact.request_id, artifact.artifact_id) in self._values

    def evict_request(self, request_id: str) -> None:
        keys = [key for key in self._values if key[0] == request_id]
        for key in keys:
            del self._values[key]



class GlobalScheduler:
    """Minimal event-driven task-graph scheduler for diffusion runtime_v2."""

    def __init__(
        self,
        topology: RuntimeTopology,
        worker_pool: Any,
        compiler: TaskCompiler,
        artifact_store: ArtifactStore,
        policy: SchedulerPolicy,
    ) -> None:
        self.topology = topology
        self.worker_pool = worker_pool
        self.compiler = compiler
        self.artifact_store = artifact_store
        self.policy = policy

        self.plans: dict[str, RequestExecutionPlan] = {}
        self.task_index: dict[str, InferenceTask] = {}
        self.pending_dependencies: dict[str, int] = {}
        self.dependents: dict[str, list[str]] = {}
        self._parked_pending: dict[str, int] = {}      # task_id -> outstanding migrations
        self.failed_requests: dict[str, str] = {}
        self.released_requests: set[str] = set()
        self.cleaned_requests: set[str] = set()
        self.failed_tasks: set[str] = set()
        self.artifact_refcounts: dict[tuple[str, str], int] = {}
        self.request_artifacts: dict[str, set[tuple[str, str]]] = {}
        self.pending_output_fetches: dict[str, _PendingOutputFetch] = {}
        # request_id -> delivered terminal output. Membership (NOT value identity)
        # is the "request finished" signal, so a legitimately None/falsy output
        # still reports finished, and the artifact store + per-request bookkeeping
        # can be evicted at cleanup without breaking a later get_request_status poll.
        self._completed_outputs: dict[str, Any] = {}
        self._sched_trace_enabled = env_flag("VLLM_RUNTIME_V2_SCHED_TRACE", False)

    def _trace(self, action: str, **fields: Any) -> None:
        if not self._sched_trace_enabled:
            return
        parts = [f"{k}={v}" for k, v in fields.items()]
        parts.append(f"mono_ns={time.monotonic_ns()}")
        logger.debug("runtime_v2 scheduler trace: %s %s", action, " ".join(parts))

    def start(self) -> None:
        self.worker_pool.start()

    def shutdown(self) -> None:
        self.worker_pool.shutdown()

    def submit_request(self, request: Any) -> str:
        plan = self.compiler.compile_request(request)
        self.plans[plan.request_id] = plan
        logger.info(
            "runtime_v2 plan compiled: request_id=%s tasks=%s terminal=%s metadata=%s",
            plan.request_id,
            len(plan.tasks),
            len(plan.terminal_task_ids),
            plan.metadata,
        )

        for artifact_value in plan.initial_artifacts:
            self.artifact_store.put(artifact_value.handle, artifact_value.value)

        request_artifacts: set[tuple[str, str]] = set()
        for task in plan.tasks.values():
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
                # Validate src/dst groups exist; raises clearly if missing.
                self.topology.get_group(src)
                self.topology.get_group(dst)
            elif task.group_id is not None:
                group = self.topology.get_group(task.group_id)
                if task.kind not in group.supported_task_kinds:
                    raise ValueError(
                        f"task {task.task_id} is pinned to unsupported group {task.group_id!r} for kind={task.kind}"
                    )
            elif self.topology.select_group_for_task(task.kind) is None:
                raise ValueError(f"no execution group supports task kind {task.kind}")
            self.task_index[task.task_id] = task
            self.pending_dependencies[task.task_id] = len(task.dependencies)
            for artifact in task.inputs:
                key = (artifact.request_id, artifact.artifact_id)
                self.artifact_refcounts[key] = self.artifact_refcounts.get(key, 0) + 1
                request_artifacts.add(key)
            for artifact in task.outputs:
                request_artifacts.add((artifact.request_id, artifact.artifact_id))
            for dependency in task.dependencies:
                self.dependents.setdefault(dependency, []).append(task.task_id)
        for artifact_value in plan.initial_artifacts:
            request_artifacts.add((artifact_value.handle.request_id, artifact_value.handle.artifact_id))
        self.request_artifacts[plan.request_id] = request_artifacts

        self._dispatch_tasks(self.policy.on_request_submitted(plan))
        return plan.request_id

    def execute(self, request: Any, timeout_s: float | None = None) -> Any:
        request_id = self.submit_request(request)
        return self.run_until_request_done(request_id=request_id, timeout_s=timeout_s)

    def run_until_request_done(self, request_id: str, timeout_s: float | None = None) -> Any:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            status, payload = self.get_request_status(request_id)
            if status == "failed":
                self.release_request(request_id)
                raise RuntimeError(str(payload))
            if status == "finished":
                self.release_request(request_id)
                return payload

            poll_timeout = 0.05
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"request {request_id} timed out")
                poll_timeout = min(poll_timeout, remaining)
            self.poll_once(timeout_s=poll_timeout)

    def poll_once(self, timeout_s: float = 0.0) -> None:
        poll_begin_ns = time.monotonic_ns()
        self._trace(
            "poll_begin",
            timeout_s=timeout_s,
            pending_fetches=len(self.pending_output_fetches),
            failed_requests=len(self.failed_requests),
        )
        # Advance in-flight migrations BEFORE blocking: their phase results are
        # routed outside the worker event queue, so they cannot wake
        # worker_pool.poll(). Pumping first sends the next phase promptly, and
        # capping the poll wait while migrations are pending picks up freshly
        # arrived phase results within ~1ms instead of a full poll interval.
        self.worker_pool.pump_migrations()
        if self.worker_pool.has_pending_migrations():
            timeout_s = min(timeout_s, _MIGRATION_PENDING_POLL_TIMEOUT_S)
        events = self.worker_pool.poll(timeout_s=timeout_s)
        self.worker_pool.pump_migrations()
        poll_end_ns = time.monotonic_ns()
        if not events:
            self._trace(
                "poll_end",
                event_count=0,
                elapsed_ns=max(0, poll_end_ns - poll_begin_ns),
            )
            return
        self._trace(
            "poll_end",
            event_count=len(events),
            elapsed_ns=max(0, poll_end_ns - poll_begin_ns),
        )
        for event in events:
            self._handle_worker_event(event)

    def get_request_status(self, request_id: str) -> tuple[str, Any | None]:
        """
        Returns:
            ("pending", None) if request is not done,
            ("finished", output) if request completed successfully,
            ("failed", error_message) if request failed.
        """
        if request_id in self.failed_requests:
            return "failed", self.failed_requests[request_id]
        if request_id in self._completed_outputs:
            return "finished", self._completed_outputs[request_id]
        if request_id not in self.plans and request_id not in self.pending_output_fetches:
            return "pending", None
        self._try_collect_request_output(request_id)
        if request_id in self._completed_outputs:
            return "finished", self._completed_outputs[request_id]
        if request_id in self.failed_requests:
            return "failed", self.failed_requests[request_id]
        return "pending", None

    def get_request_progress(self, request_id: str) -> dict[str, int]:
        plan = self.plans.get(request_id)
        if plan is None:
            return {
                "total_tasks": 0,
                "finished_tasks": 0,
                "running_tasks": 0,
                "pending_tasks": 0,
                "failed_tasks": 0,
            }
        total = len(plan.tasks)
        finished = 0
        running = 0
        pending = 0
        failed = 0
        for task in plan.tasks.values():
            status = self.task_index.get(task.task_id, task).status
            if status == TaskStatus.FINISHED:
                finished += 1
            elif status == TaskStatus.RUNNING:
                running += 1
            elif status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.DISPATCHED):
                pending += 1
            elif status == TaskStatus.FAILED:
                failed += 1
        return {
            "total_tasks": total,
            "finished_tasks": finished,
            "running_tasks": running,
            "pending_tasks": pending,
            "failed_tasks": failed,
        }

    def _dispatch_tasks(self, tasks: Iterable[InferenceTask]) -> None:
        for task in tasks:
            if task.kind == TaskKind.RESHARD:
                self._dispatch_reshard_task(task)
                continue
            if task.group_id is None:
                raise ValueError(f"task {task.task_id} has no assigned group_id before dispatch")
            group = self.topology.get_group(task.group_id)
            task.parallel_spec = group.parallel_spec

            cross_group = [
                artifact for artifact in task.inputs
                if self._needs_migration(task, artifact)
            ]
            if not cross_group:
                self._finalize_task_dispatch(task)
                continue
            self._parked_pending[task.task_id] = len(cross_group)
            for artifact in cross_group:
                self._begin_implicit_migration(task, artifact)

    def _needs_migration(self, task: InferenceTask, artifact: ArtifactHandle) -> bool:
        stored = self.artifact_store.get(artifact)
        return isinstance(stored, WorkerLocalArtifactRef) and task.group_id != stored.group_id

    def _begin_implicit_migration(self, task: InferenceTask, artifact: ArtifactHandle) -> None:
        stored = self.artifact_store.get(artifact)
        if not artifact.codec_id:
            raise RuntimeError(
                f"task {task.task_id} input {artifact.artifact_id!r} requires migration from "
                f"group {stored.group_id!r} to {task.group_id!r}, but has no codec_id")
        src_ranks = tuple(self.topology.get_group(stored.group_id).ranks)
        dst_ranks = tuple(self.topology.get_group(task.group_id).ranks)
        hold_ranks = tuple(sorted(set(src_ranks) | set(dst_ranks)))
        self.policy.acquire_migration_hold(hold_ranks)
        plan = MigrateArtifactsPlan(
            request_id=task.request_id,
            src_group_id=stored.group_id,
            dst_group_id=task.group_id,
            artifacts=(MigrateArtifactSpec(handle=artifact),),
            request_metadata=self.plans[task.request_id].metadata,
        )
        logger.info(
            "runtime_v2 implicit migration: request_id=%s task_id=%s artifact=%s "
            "src_group=%s dst_group=%s",
            task.request_id, task.task_id, artifact.artifact_id, stored.group_id, task.group_id,
        )

        def _on_done(_schedule, *, task=task, artifact=artifact,
                     dst_group_id=task.group_id, hold_ranks=hold_ranks) -> None:
            self.policy.release_migration_hold(hold_ranks)
            # Stop if a sibling input's migration already failed this task, or the
            # request was failed/evicted externally while this migration ran. The
            # parked task still holds the policy's dispatch claim (it was never
            # dispatched, so no EXEC_END/TASK_FAILED has released it), so fail it
            # once to release the claim instead of leaking the ranks/slot.
            if (task.task_id not in self._parked_pending
                    or task.request_id in self.failed_requests):
                self._parked_pending.pop(task.task_id, None)
                self._fail_parked_task(task, "request failed during implicit migration")
                return
            migrated = WorkerLocalArtifactRef(
                handle=artifact, group_id=dst_group_id,
                worker_rank=self.topology.get_group_leader(dst_group_id),
            )
            self.artifact_store.put(artifact, migrated)
            self._parked_pending[task.task_id] -= 1
            if self._parked_pending[task.task_id] == 0:
                self._parked_pending.pop(task.task_id, None)
                self._finalize_task_dispatch(task)

        def _on_error(exc, *, task=task, hold_ranks=hold_ranks) -> None:
            # Always release this migration's hold, then fail the task once so the
            # policy releases the dst-rank claim + in-flight slot it took when the
            # task was dispatched (before parking) -- same terminal path as RESHARD
            # errors. _fail_parked_task dedups via failed_tasks, so sibling-input
            # failures and begin-time raises are all safe.
            self.policy.release_migration_hold(hold_ranks)
            self._parked_pending.pop(task.task_id, None)
            self._fail_parked_task(task, f"implicit migration failed: {exc}")

        try:
            self.worker_pool.begin_migration(plan, on_done=_on_done, on_error=_on_error)
        except Exception as exc:  # noqa: BLE001 - startup raise (reader dead / missing handle)
            # begin_migration can raise before any callback is registered; route it
            # through the same failure path so the hold + policy claim are released.
            _on_error(exc)

    def _finalize_task_dispatch(self, task: InferenceTask) -> None:
        """Assemble inline inputs, apply refcount/release accounting, and dispatch.
        Called once all of the task's cross-group inputs have migrated, so every
        WorkerLocalArtifactRef input now lives on the task's own group."""
        inline_inputs: list[ArtifactValue] = []
        release_after_exec_artifact_ids: list[str] = []
        for artifact in task.inputs:
            stored = self.artifact_store.get(artifact)
            if not isinstance(stored, WorkerLocalArtifactRef):
                inline_inputs.append(ArtifactValue(handle=artifact, value=stored))
            key = (artifact.request_id, artifact.artifact_id)
            remaining = self.artifact_refcounts.get(key)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    self.artifact_refcounts.pop(key, None)
                    release_after_exec_artifact_ids.append(artifact.artifact_id)
                else:
                    self.artifact_refcounts[key] = remaining
        if task.step_range is not None:
            logger.debug(
                "runtime_v2 dispatch: request_id=%s task_id=%s kind=%s group=%s step_range=[%s,%s)",
                task.request_id,
                task.task_id,
                task.kind,
                task.group_id,
                task.step_range.start,
                task.step_range.end,
            )
        else:
            logger.debug(
                "runtime_v2 dispatch: request_id=%s task_id=%s kind=%s group=%s",
                task.request_id,
                task.task_id,
                task.kind,
                task.group_id,
            )
        self.worker_pool.dispatch(
            task=task,
            inline_inputs=tuple(inline_inputs),
            release_after_exec_artifact_ids=tuple(release_after_exec_artifact_ids),
        )

    def _dispatch_reshard_task(self, task: InferenceTask) -> None:
        # RESHARD is a collective artifact migration that spans src∪dst ranks.
        # It does NOT go through worker_pool.dispatch (which is per-task on a
        # single group). Instead we start an asynchronous migration via
        # begin_migration; its on_done/on_error callbacks (driven by
        # pump_migrations on the poll loop) fake LAUNCH_END/EXEC_END (or
        # TASK_FAILED) events so dependents see the migrated artifact and the
        # policy releases its in-flight slot. The control thread never blocks
        # waiting for the migration handshake.
        src_group_id = task.payload.get(RESHARD_PAYLOAD_SRC_GROUP_ID)
        dst_group_id = task.payload.get(RESHARD_PAYLOAD_DST_GROUP_ID)
        if not isinstance(src_group_id, str) or not src_group_id:
            raise ValueError(
                f"RESHARD task {task.task_id} payload missing {RESHARD_PAYLOAD_SRC_GROUP_ID!r}"
            )
        if not isinstance(dst_group_id, str) or not dst_group_id:
            raise ValueError(
                f"RESHARD task {task.task_id} payload missing {RESHARD_PAYLOAD_DST_GROUP_ID!r}"
            )
        if not task.inputs:
            raise ValueError(f"RESHARD task {task.task_id} requires at least one input artifact")
        if not task.outputs:
            raise ValueError(f"RESHARD task {task.task_id} requires at least one output artifact")
        if len(task.inputs) != len(task.outputs):
            raise ValueError(
                f"RESHARD task {task.task_id} input/output count mismatch: "
                f"{len(task.inputs)} inputs vs {len(task.outputs)} outputs"
            )

        artifact_specs: list[MigrateArtifactSpec] = []
        for input_handle, output_handle in zip(task.inputs, task.outputs, strict=True):
            if not output_handle.codec_id:
                raise ValueError(
                    f"RESHARD task {task.task_id} output {output_handle.artifact_id!r} "
                    "requires codec_id; the codec is needed to extract/assemble tensors during migration"
                )
            # Distinct source/destination handles let the migration control
            # plane locate the source value on src ranks even when the
            # compiler renamed the artifact across the boundary
            # (e.g. state_text -> state_text_dit).
            source_handle = (
                input_handle if input_handle.artifact_id != output_handle.artifact_id else None
            )
            artifact_specs.append(
                MigrateArtifactSpec(handle=output_handle, source_handle=source_handle)
            )

        for input_handle in task.inputs:
            stored = self.artifact_store.get(input_handle)
            if not isinstance(stored, WorkerLocalArtifactRef):
                raise RuntimeError(
                    f"RESHARD task {task.task_id} input {input_handle.artifact_id!r} is not a "
                    f"WorkerLocalArtifactRef (got {type(stored).__name__}); RESHARD only migrates "
                    "worker-local artifacts produced by a prior task."
                )
            if stored.group_id != src_group_id:
                raise RuntimeError(
                    f"RESHARD task {task.task_id} input {input_handle.artifact_id!r} lives on group "
                    f"{stored.group_id!r}, but payload says src_group_id={src_group_id!r}"
                )

        # Drop input refcounts now (reshard consumes them once).
        for artifact in task.inputs:
            key = (artifact.request_id, artifact.artifact_id)
            remaining = self.artifact_refcounts.get(key)
            if remaining is None:
                continue
            remaining -= 1
            if remaining <= 0:
                self.artifact_refcounts.pop(key, None)
            else:
                self.artifact_refcounts[key] = remaining

        plan = MigrateArtifactsPlan(
            request_id=task.request_id,
            src_group_id=src_group_id,
            dst_group_id=dst_group_id,
            artifacts=tuple(artifact_specs),
            request_metadata=self.plans[task.request_id].metadata,
        )
        logger.info(
            "runtime_v2 reshard begin: request_id=%s task_id=%s src_group=%s dst_group=%s artifacts=%s",
            task.request_id,
            task.task_id,
            src_group_id,
            dst_group_id,
            [spec.handle.artifact_id for spec in artifact_specs],
        )

        task.status = TaskStatus.RUNNING

        # _schedule is the wire-level transfer plan; RESHARD only needs to
        # publish the dst WorkerLocalArtifactRefs, so it is not inspected here.
        def _on_done(_schedule, *, task=task, dst_group_id=dst_group_id) -> None:
            # If the request was failed/evicted externally while this reshard was in
            # flight, do NOT publish outputs / decrement dependents (that would
            # dispatch downstream work for an evicted request). Fail the task once so
            # the policy releases the reshard hold + outstanding-reshard slot.
            if task.request_id in self.failed_requests or task.task_id in self.failed_tasks:
                self._fail_parked_task(task, "request failed during reshard")
                return
            dst_leader_rank = self.topology.get_group_leader(dst_group_id)
            published_outputs = tuple(
                WorkerLocalArtifactRef(handle=output_handle, group_id=dst_group_id,
                                       worker_rank=dst_leader_rank)
                for output_handle in task.outputs
            )
            logger.info(
                "runtime_v2 reshard done: request_id=%s task_id=%s dst_leader_rank=%s outputs=%s",
                task.request_id, task.task_id, dst_leader_rank,
                [ref.handle.artifact_id for ref in published_outputs],
            )
            self._fire_synthetic_event(
                task=task, kind=WorkerEventKind.TASK_LAUNCH_END, group_id=dst_group_id,
                metadata={"published_outputs": published_outputs},
            )
            self._fire_synthetic_event(
                task=task, kind=WorkerEventKind.TASK_EXEC_END, group_id=dst_group_id,
            )

        def _on_error(exc, *, task=task, dst_group_id=dst_group_id) -> None:
            logger.error(
                "runtime_v2 reshard failed: request_id=%s task_id=%s",
                task.request_id, task.task_id, exc_info=exc,
            )
            self._fail_parked_task(task, f"reshard task {task.task_id} failed: {exc}",
                                   group_id=dst_group_id)

        try:
            self.worker_pool.begin_migration(plan, on_done=_on_done, on_error=_on_error)
        except Exception as exc:  # noqa: BLE001 - startup raise (reader dead / missing handle)
            _on_error(exc)

    def _fail_parked_task(self, task: InferenceTask, reason: str, *, group_id: str | None = None) -> None:
        """Fire a synthetic TASK_FAILED for a task whose migration could not
        complete, or whose request failed while it was parked/in-flight. Deduped
        via failed_tasks, so it is safe to call from any migration callback. The
        TASK_FAILED routes through _handle_worker_event, which drives the policy to
        release the task's dispatch claim (ranks / in-flight slot) -- the only path
        that releases that claim for a task that was never dispatched to a worker."""
        if task.task_id in self.failed_tasks:
            return
        self._fire_synthetic_event(
            task=task,
            kind=WorkerEventKind.TASK_FAILED,
            group_id=group_id if group_id is not None else (task.group_id or ""),
            message=reason,
        )

    def _fire_synthetic_event(
        self,
        *,
        task: InferenceTask,
        kind: WorkerEventKind,
        group_id: str,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = WorkerEvent(
            event_id=f"reshard:{task.task_id}:{kind.value}",
            task_id=task.task_id,
            request_id=task.request_id,
            group_id=group_id,
            worker_rank=-1,
            kind=kind,
            timestamp_ns=time.monotonic_ns(),
            message=message,
            metadata=metadata or {},
        )
        self._handle_worker_event(event)

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        from vllm_omni.diffusion.runtime_v2.multiproc_worker import WorkerReadyMessage

        if isinstance(event, WorkerReadyMessage):
            # Post-startup WorkerReadyMessage is a worker-death notification:
            # the worker process raised before exiting and sent its traceback.
            # Surface it as a fatal error instead of crashing on .kind.
            message = event.message or f"runtime_v2 worker {event.worker_rank} exited unexpectedly"
            logger.error(
                "runtime_v2 worker died: worker_rank=%s status=%s message=%s",
                event.worker_rank,
                event.status,
                message,
            )
            raise RuntimeError(
                f"runtime_v2 worker {event.worker_rank} died (status={event.status!r}): {message}"
            )

        if event.kind == WorkerEventKind.REQUEST_FINISHED:
            if not self._mark_request_event_delivered(event.request_id, event.kind):
                return
            logger.info("runtime_v2 request finished: request_id=%s group=%s", event.request_id, event.group_id)
            self._dispatch_tasks(self.policy.on_worker_event(event))
            return
        if event.kind == WorkerEventKind.REQUEST_FAILED:
            if not self._mark_request_event_delivered(event.request_id, event.kind):
                return
            self.failed_requests.setdefault(event.request_id, event.message or f"request {event.request_id} failed")
            logger.error(
                "runtime_v2 request failed: request_id=%s group=%s message=%s",
                event.request_id,
                event.group_id,
                event.message,
            )
            self._dispatch_tasks(self.policy.on_worker_event(event))
            self._cleanup_request_worker_state(event.request_id)
            return

        task = self.task_index.get(event.task_id)
        if task is None:
            return

        if event.kind == WorkerEventKind.TASK_EXEC_BEGIN:
            task.status = TaskStatus.RUNNING
            if task.step_range is not None:
                logger.debug(
                    "runtime_v2 task begin: request_id=%s task_id=%s kind=%s step_range=[%s,%s)",
                    task.request_id,
                    task.task_id,
                    task.kind,
                    task.step_range.start,
                    task.step_range.end,
                )
            else:
                logger.debug(
                    "runtime_v2 task begin: request_id=%s task_id=%s kind=%s",
                    task.request_id,
                    task.task_id,
                    task.kind,
                )
            return

        if event.kind == WorkerEventKind.TASK_LAUNCH_BEGIN:
            logger.debug(
                "runtime_v2 task launch begin: request_id=%s task_id=%s kind=%s group=%s worker_rank=%s metadata=%s",
                task.request_id,
                task.task_id,
                task.kind,
                event.group_id,
                event.worker_rank,
                dict(event.metadata),
            )
            return

        if event.kind == WorkerEventKind.TASK_LAUNCH_END:
            logger.debug(
                "runtime_v2 task launch end: request_id=%s task_id=%s kind=%s group=%s worker_rank=%s metadata=%s",
                task.request_id,
                task.task_id,
                task.kind,
                event.group_id,
                event.worker_rank,
                dict(event.metadata),
            )
            self._dispatch_tasks(self.policy.on_worker_event(event))
            published_outputs = event.metadata.get("published_outputs", ())
            for artifact_ref in published_outputs:
                self.artifact_store.put(artifact_ref.handle, artifact_ref)

            # Skip dependent scheduling if the request already failed/was retired:
            # decrementing or dispatching downstream tasks of a dead request would
            # offer them to the policy (and the bookkeeping may already be gone).
            # Mirrors the RESHARD _on_done guard.
            newly_runnable: list[InferenceTask] = []
            if task.request_id not in self.failed_requests:
                for dependent_id in self.dependents.get(task.task_id, []):
                    if dependent_id not in self.pending_dependencies:
                        continue
                    self.pending_dependencies[dependent_id] -= 1
                    if self.pending_dependencies[dependent_id] == 0:
                        newly_runnable.append(self.task_index[dependent_id])
            self._dispatch_tasks(self.policy.on_tasks_runnable(newly_runnable))
            return

        if event.kind == WorkerEventKind.TASK_EXEC_END:
            task.status = TaskStatus.FINISHED
            logger.debug(
                "runtime_v2 task end: request_id=%s task_id=%s kind=%s",
                task.request_id,
                task.task_id,
                task.kind,
            )
            self._dispatch_tasks(self.policy.on_worker_event(event))
            if task.task_id in self.plans[task.request_id].terminal_task_ids:
                self._emit_request_event(
                    request_id=task.request_id,
                    group_id=event.group_id,
                    kind=WorkerEventKind.REQUEST_FINISHED,
                )
            return

        if event.kind == WorkerEventKind.TASK_FAILED:
            if task.task_id in self.failed_tasks:
                return
            self.failed_tasks.add(task.task_id)
            task.status = TaskStatus.FAILED
            self.failed_requests.setdefault(task.request_id, event.message or f"task {task.task_id} failed")
            logger.error(
                "runtime_v2 task failed: request_id=%s task_id=%s kind=%s message=%s",
                task.request_id,
                task.task_id,
                task.kind,
                event.message,
            )
            self._dispatch_tasks(self.policy.on_worker_event(event))
            self._emit_request_event(
                request_id=task.request_id,
                group_id=event.group_id,
                kind=WorkerEventKind.REQUEST_FAILED,
                message=event.message,
            )
            self._cleanup_request_worker_state(task.request_id)

    def _try_collect_request_output(self, request_id: str) -> Any | None:
        if request_id in self._completed_outputs:
            return self._completed_outputs[request_id]
        self._trace(
            "collect_probe",
            request_id=request_id,
            has_pending_fetch=request_id in self.pending_output_fetches,
            has_failure=request_id in self.failed_requests,
        )
        # Use _completed_outputs membership (not "output is not None") as the
        # finished signal: a fetch can legitimately complete with a None value, and
        # the finish path pops the plan, so we must NOT fall through to
        # _try_start_request_output_fetch afterwards (it would KeyError on the plan).
        self._try_finish_request_output_fetch(request_id)
        if request_id in self._completed_outputs:
            self._trace("collect_done", request_id=request_id, path="finish_fetch")
            return self._completed_outputs[request_id]
        if request_id not in self.plans:
            self._trace("collect_skip_no_plan", request_id=request_id)
            return None
        self._try_start_request_output_fetch(request_id)
        if request_id in self._completed_outputs:
            self._trace("collect_done", request_id=request_id, path="direct")
            return self._completed_outputs[request_id]
        self._trace("collect_pending", request_id=request_id)
        return None

    def _fail_request_from_fetch_error(self, request_id: str, message: str) -> None:
        # Record an output-fetch failure as a terminal "failed" status (mirroring
        # a REQUEST_FAILED worker event) and free the request's controller state.
        # Raising out of the fetch helpers instead would propagate a raw exception
        # through get_request_status without ever transitioning the request to
        # "failed", would leak the plan / artifact bookkeeping (cleanup never
        # runs), and would let a re-poll silently start a brand-new fetch.
        logger.error("runtime_v2 output fetch failed: request_id=%s error=%s", request_id, message)
        self.failed_requests.setdefault(request_id, message)
        self._cleanup_request_worker_state(request_id)

    def _try_start_request_output_fetch(self, request_id: str) -> Any | None:
        plan = self.plans[request_id]
        unfinished_terminal = [
            task_id
            for task_id in plan.terminal_task_ids
            if self.task_index[task_id].status != TaskStatus.FINISHED
        ]
        if unfinished_terminal:
            self._trace(
                "fetch_start_skip_terminals_not_finished",
                request_id=request_id,
                unfinished_terminal=len(unfinished_terminal),
            )
            return None

        if request_id in self.pending_output_fetches:
            self._trace("fetch_start_skip_already_pending", request_id=request_id)
            return None

        terminal_task = self.task_index[plan.terminal_task_ids[0]]
        if not terminal_task.outputs:
            self._trace("fetch_start_skip_no_output", request_id=request_id, terminal_task_id=terminal_task.task_id)
            return None
        output_handle = terminal_task.outputs[0]
        stored = self.artifact_store.get(output_handle)

        if isinstance(stored, WorkerLocalArtifactRef):
            if not hasattr(self.worker_pool, "start_fetch_artifacts"):
                fetch_begin_ns = time.monotonic_ns()
                fetch = self.worker_pool.fetch_artifacts(
                    request_id=request_id,
                    group_id=stored.group_id,
                    artifact_ids=(output_handle.artifact_id,),
                )
                fetch_done_ns = time.monotonic_ns()
                if fetch.error:
                    self._fail_request_from_fetch_error(request_id, str(fetch.error))
                    return None
                if not fetch.artifacts:
                    self._fail_request_from_fetch_error(
                        request_id,
                        f"runtime_v2 fetch returned empty artifacts: request_id={request_id}",
                    )
                    return None
                artifact_value = fetch.artifacts[0]
                self.artifact_store.put(artifact_value.handle, artifact_value.value)
                self._completed_outputs[request_id] = artifact_value.value
                self._cleanup_request_worker_state(request_id)
                self._trace(
                    "fetch_sync_completed",
                    request_id=request_id,
                    group_id=stored.group_id,
                    artifact_id=output_handle.artifact_id,
                    elapsed_ns=max(0, fetch_done_ns - fetch_begin_ns),
                )
                return artifact_value.value

            fetch_start_ns = time.monotonic_ns()
            fetch_id = self.worker_pool.start_fetch_artifacts(
                request_id=request_id,
                group_id=stored.group_id,
                artifact_ids=(output_handle.artifact_id,),
            )
            self.pending_output_fetches[request_id] = _PendingOutputFetch(
                fetch_id=fetch_id,
                output_handle=output_handle,
                group_id=stored.group_id,
                started_ns=fetch_start_ns,
            )
            logger.debug(
                "runtime_v2 output fetch started: request_id=%s group=%s artifact_id=%s fetch_id=%s",
                request_id,
                stored.group_id,
                output_handle.artifact_id,
                fetch_id,
            )
            self._trace(
                "fetch_async_started",
                request_id=request_id,
                group_id=stored.group_id,
                artifact_id=output_handle.artifact_id,
                fetch_id=fetch_id,
            )
            return None

        self._trace(
            "fetch_skip_worker_pull",
            request_id=request_id,
            artifact_id=output_handle.artifact_id,
            stored_type=type(stored).__name__,
        )
        self._completed_outputs[request_id] = stored
        self._cleanup_request_worker_state(request_id)
        return stored

    def _try_finish_request_output_fetch(self, request_id: str) -> Any | None:
        pending = self.pending_output_fetches.get(request_id)
        if pending is None:
            return None
        if not hasattr(self.worker_pool, "poll_fetch_artifacts"):
            self._trace("fetch_poll_skip_no_api", request_id=request_id, fetch_id=pending.fetch_id)
            return None

        fetch = self.worker_pool.poll_fetch_artifacts(pending.fetch_id)
        if fetch is None:
            self._trace("fetch_poll_pending", request_id=request_id, fetch_id=pending.fetch_id)
            return None

        fetch_done_ns = time.monotonic_ns()
        self.pending_output_fetches.pop(request_id, None)
        if fetch.error:
            self._fail_request_from_fetch_error(request_id, str(fetch.error))
            return None
        if not fetch.artifacts:
            self._fail_request_from_fetch_error(
                request_id,
                f"runtime_v2 fetch returned empty artifacts: request_id={request_id} fetch_id={pending.fetch_id}",
            )
            return None
        artifact_value = fetch.artifacts[0]
        self.artifact_store.put(artifact_value.handle, artifact_value.value)
        self._completed_outputs[request_id] = artifact_value.value
        self._cleanup_request_worker_state(request_id)
        logger.debug(
            "runtime_v2 output fetch completed: request_id=%s group=%s artifact_id=%s fetch_id=%s",
            request_id,
            pending.group_id,
            pending.output_handle.artifact_id,
            pending.fetch_id,
        )
        self._trace(
            "fetch_async_completed",
            request_id=request_id,
            group_id=pending.group_id,
            artifact_id=pending.output_handle.artifact_id,
            fetch_id=pending.fetch_id,
            elapsed_ns=max(0, fetch_done_ns - pending.started_ns),
        )
        return artifact_value.value

    def _mark_request_event_delivered(self, request_id: str, kind: WorkerEventKind) -> bool:
        key = f"{request_id}:{kind.value}"
        if key in self.released_requests:
            return False
        self.released_requests.add(key)
        return True

    def _emit_request_event(
        self,
        *,
        request_id: str,
        group_id: str,
        kind: WorkerEventKind,
        message: str = "",
    ) -> None:
        if not self._mark_request_event_delivered(request_id, kind):
            return
        self._dispatch_tasks(
            self.policy.on_worker_event(
                WorkerEvent(
                    event_id=f"request:{request_id}:{kind.value}",
                    task_id="",
                    request_id=request_id,
                    group_id=group_id,
                    worker_rank=-1,
                    kind=kind,
                    timestamp_ns=time.monotonic_ns(),
                    message=message,
                )
            )
        )

    def _cleanup_request_worker_state(self, request_id: str) -> None:
        if request_id in self.cleaned_requests:
            self._trace("cleanup_skip_already_done", request_id=request_id)
            return
        cleanup_begin_ns = time.monotonic_ns()
        self.cleaned_requests.add(request_id)
        pending = self.pending_output_fetches.pop(request_id, None)
        if pending is not None and hasattr(self.worker_pool, "discard_fetch"):
            self.worker_pool.discard_fetch(pending.fetch_id)
        removed_artifacts = 0
        for key in self.request_artifacts.pop(request_id, set()):
            self.artifact_refcounts.pop(key, None)
            removed_artifacts += 1
        self.worker_pool.evict_request(request_id)
        # Free controller-side per-request state. The terminal output (if any) has
        # already been copied into self._completed_outputs by the caller, so evicting
        # the artifact store here cannot strand a not-yet-delivered result; a later
        # get_request_status poll is served from _completed_outputs. Without this the
        # store, plan, and task bookkeeping grew unboundedly for the process lifetime.
        self.artifact_store.evict_request(request_id)
        plan = self.plans.pop(request_id, None)
        if plan is not None:
            for task_id in plan.tasks:
                self.task_index.pop(task_id, None)
                self.pending_dependencies.pop(task_id, None)
                self.dependents.pop(task_id, None)
                self._parked_pending.pop(task_id, None)
        cleanup_done_ns = time.monotonic_ns()
        self._trace(
            "cleanup_done",
            request_id=request_id,
            removed_artifacts=removed_artifacts,
            had_pending_fetch=pending is not None,
            elapsed_ns=max(0, cleanup_done_ns - cleanup_begin_ns),
        )

    def release_request(self, request_id: str) -> None:
        """Retire a request once its terminal status has been delivered to the
        caller. Frees the large per-request state (plan, artifact store, task
        bookkeeping, cached output) but intentionally KEEPS the lightweight
        cleanup/request-event tombstones (cleaned_requests / released_requests)
        so a duplicate or late worker event for this id is still suppressed; by
        design those string sets persist for the process lifetime (bounded only
        by the number of distinct request ids seen -- not the big objects).
        Idempotent and safe to call more than once. The sync run loops call this
        after returning the payload; an async consumer should call it at finalize."""
        self._cleanup_request_worker_state(request_id)
        self._completed_outputs.pop(request_id, None)
        self.failed_requests.pop(request_id, None)


@dataclass(frozen=True)
class _PendingOutputFetch:
    fetch_id: str
    output_handle: ArtifactHandle
    group_id: str
    started_ns: int = 0
