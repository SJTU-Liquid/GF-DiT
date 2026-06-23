# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.data_plane import (
    _MIGRATE_PHASE_ACCEPT_SCHEDULE,
    _MIGRATE_PHASE_BUILD_SCHEDULE,
    _MIGRATE_PHASE_DESCRIBE_LAYOUT,
    _MIGRATE_PHASE_EXECUTE_TRANSFER,
)

logger = init_logger(__name__)

# The phase that follows each one (None terminates the handshake).
_NEXT_PHASE = {
    _MIGRATE_PHASE_DESCRIBE_LAYOUT: _MIGRATE_PHASE_BUILD_SCHEDULE,
    _MIGRATE_PHASE_BUILD_SCHEDULE: _MIGRATE_PHASE_ACCEPT_SCHEDULE,
    _MIGRATE_PHASE_ACCEPT_SCHEDULE: _MIGRATE_PHASE_EXECUTE_TRANSFER,
    _MIGRATE_PHASE_EXECUTE_TRANSFER: None,
}


@dataclass
class MigrationJob:
    migrate_id: str
    migrate_seq: int
    plan: Any
    src_group: Any
    dst_group: Any
    participant_ranks: tuple[int, ...]
    src_leader_rank: int
    on_done: Callable[[Any], None]
    on_error: Callable[[Exception], None]
    deadline: float
    phase: str = _MIGRATE_PHASE_DESCRIBE_LAYOUT
    awaiting: bool = False              # True once the current phase command is sent
    pending_ranks: set[int] = field(default_factory=set)
    layout_results: list = field(default_factory=list)
    schedule: Any = None
    metadata_payloads: tuple = ()
    finished: bool = False
    transfer_overrun_warned: bool = False


class MigrationEngine:
    """Event-driven state machine for the multi-phase migration handshake.

    Pure: no pipes, no time source. The owner injects ``build_command(job, phase)``
    (constructs the transport command) and ``send(command, ranks)`` (delivers it),
    drains transport results, and calls ``pump(results, now)`` once per tick.
    """

    def __init__(
        self,
        *,
        build_command: Callable[[MigrationJob, str], Any],
        send: Callable[[Any, tuple[int, ...]], None],
    ) -> None:
        self._build_command = build_command
        self._send = send
        self._jobs: dict[str, MigrationJob] = {}
        self._rank_queues: dict[int, list[str]] = {}
        self._seq = 0

    def begin(
        self,
        *,
        plan: Any,
        src_group: Any,
        dst_group: Any,
        participant_ranks: tuple[int, ...],
        src_leader_rank: int,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        deadline: float,
        migrate_id: str | None = None,
    ) -> str:
        # BUILD targets only the src leader, so it must be a participant; the
        # caller derives participant_ranks as src∪dst, so this always holds.
        assert src_leader_rank in participant_ranks, (
            f"src_leader_rank {src_leader_rank} must be in participant_ranks "
            f"{participant_ranks}"
        )
        mid = migrate_id or str(uuid.uuid4())
        assert mid not in self._jobs, f"duplicate migrate_id {mid}"
        self._seq += 1
        job = MigrationJob(
            migrate_id=mid,
            migrate_seq=self._seq,
            plan=plan,
            src_group=src_group,
            dst_group=dst_group,
            participant_ranks=tuple(participant_ranks),
            src_leader_rank=src_leader_rank,
            on_done=on_done,
            on_error=on_error,
            deadline=deadline,
        )
        self._jobs[mid] = job
        for rank in participant_ranks:
            self._rank_queues.setdefault(rank, []).append(mid)
        logger.info(
            "runtime_v2 migration begin: migrate_id=%s seq=%s participants=%s",
            mid, job.migrate_seq, job.participant_ranks,
        )
        return mid

    def has_pending(self) -> bool:
        """True while any migration job is still in flight."""
        return bool(self._jobs)

    def pump(self, results, now: float) -> None:
        self._ingest_results(results)
        for job in sorted(self._jobs.values(), key=lambda j: j.migrate_seq):
            if job.finished:
                continue
            if now > job.deadline:
                # Once EXECUTE_TRANSFER has been dispatched (awaiting=True), the
                # workers are committed to a blocking cross-rank collective with no
                # abort path. Timing the job out here would retire it and free its
                # ranks, so the next queued migration would be sent to ranks still
                # stuck in the old collective -> control/data-plane divergence and a
                # cross-collective deadlock. Only the handshake phases (DESCRIBE /
                # BUILD / ACCEPT, no collective in flight) are timed out; a stuck
                # transfer instead surfaces via the request-level timeout.
                in_flight_transfer = (
                    job.phase == _MIGRATE_PHASE_EXECUTE_TRANSFER and job.awaiting
                )
                if not in_flight_transfer:
                    self._fail(job, TimeoutError(
                        f"migration {job.migrate_id} timed out in phase {job.phase}"))
                    continue
                if not job.transfer_overrun_warned:
                    job.transfer_overrun_warned = True
                    logger.warning(
                        "runtime_v2 migration %s exceeded deadline while EXECUTE_TRANSFER is "
                        "in flight (pending_ranks=%s); not timing out to avoid reusing ranks "
                        "still inside the transfer collective.",
                        job.migrate_id, sorted(job.pending_ranks),
                    )
            self._advance(job, now)

    # ---- internals ----

    def _ingest_results(self, results) -> None:
        for result in results:
            job = self._jobs.get(result.migrate_id)
            if job is None or job.finished:
                continue
            if result.error:
                self._fail(job, RuntimeError(
                    f"migration {job.migrate_id} phase {result.phase} failed on "
                    f"rank {result.worker_rank}: {result.error}"))
                continue
            if result.phase != job.phase:
                # A stale/duplicate reply must not crash the shared pump loop
                # (which drives every in-flight migration); fail just this job.
                self._fail(job, RuntimeError(
                    f"unexpected migration result: migrate_id={job.migrate_id} "
                    f"expected phase={job.phase} got {result.phase}"))
                continue
            job.pending_ranks.discard(result.worker_rank)
            if job.phase == _MIGRATE_PHASE_DESCRIBE_LAYOUT:
                job.layout_results.append(result)
            elif job.phase == _MIGRATE_PHASE_BUILD_SCHEDULE:
                job.schedule = result.schedule
                job.metadata_payloads = result.metadata_payloads

    def _advance(self, job: MigrationJob, now: float) -> None:
        if not self._is_oldest_on_all_ranks(job):
            return
        if job.awaiting:
            if job.pending_ranks:
                return                       # current phase still in flight
            nxt = _NEXT_PHASE[job.phase]
            if nxt is None:
                self._complete(job)
                return
            job.phase = nxt
            job.awaiting = False
        targets = self._targets(job)
        command = self._build_command(job, job.phase)
        job.pending_ranks = set(targets)
        job.awaiting = True
        self._send(command, targets)

    def _is_oldest_on_all_ranks(self, job: MigrationJob) -> bool:
        """A job may act only when it is the front (lowest seq) of every
        participant rank's FIFO queue. Retired jobs are removed from the queues,
        so the front is always the oldest still-pending migration."""
        for rank in job.participant_ranks:
            queue = self._rank_queues.get(rank)
            if not queue or queue[0] != job.migrate_id:
                return False
        return True

    def _targets(self, job: MigrationJob) -> tuple[int, ...]:
        if job.phase == _MIGRATE_PHASE_BUILD_SCHEDULE:
            return (job.src_leader_rank,)
        return job.participant_ranks

    def _complete(self, job: MigrationJob) -> None:
        self._retire(job)
        logger.info(
            "runtime_v2 migration done: migrate_id=%s seq=%s participants=%s",
            job.migrate_id, job.migrate_seq, job.participant_ranks,
        )
        job.on_done(job.schedule)

    def _fail(self, job: MigrationJob, exc: Exception) -> None:
        self._retire(job)
        logger.warning(
            "runtime_v2 migration failed: migrate_id=%s seq=%s phase=%s error=%s",
            job.migrate_id, job.migrate_seq, job.phase, exc,
        )
        job.on_error(exc)

    def _retire(self, job: MigrationJob) -> None:
        job.finished = True
        self._jobs.pop(job.migrate_id, None)
        for rank in job.participant_ranks:
            queue = self._rank_queues.get(rank)
            if queue and job.migrate_id in queue:
                queue.remove(job.migrate_id)
