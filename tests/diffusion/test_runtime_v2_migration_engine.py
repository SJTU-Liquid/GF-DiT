# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_omni.diffusion.runtime_v2.data_plane import (
    _MIGRATE_PHASE_ACCEPT_SCHEDULE,
    _MIGRATE_PHASE_BUILD_SCHEDULE,
    _MIGRATE_PHASE_DESCRIBE_LAYOUT,
    _MIGRATE_PHASE_EXECUTE_TRANSFER,
)
from vllm_omni.diffusion.runtime_v2.migration_engine import MigrationEngine

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Result:
    """Stand-in for MigrateArtifactsRankResult (duck-typed by the engine)."""
    migrate_id: str
    worker_rank: int
    phase: str
    schedule: object | None = None
    metadata_payloads: tuple = ()
    error: str | None = None


class _Recorder:
    """Captures (phase, ranks) sends and lets a test build fake commands."""
    def __init__(self) -> None:
        self.sends: list[tuple[str, tuple[int, ...]]] = []

    def build(self, job, phase):
        return (job.migrate_id, phase)

    def send(self, command, ranks):
        self.sends.append((command[1], tuple(ranks)))


def _engine(rec: _Recorder) -> MigrationEngine:
    return MigrationEngine(build_command=rec.build, send=rec.send)


def _begin(engine, rec, *, migrate_id_ranks, src_leader, on_done, on_error):
    # plan/src_group/dst_group are opaque to the engine; pass simple sentinels.
    return engine.begin(
        plan=None,
        src_group=None,
        dst_group=None,
        participant_ranks=tuple(migrate_id_ranks),
        src_leader_rank=src_leader,
        on_done=on_done,
        on_error=on_error,
        deadline=float("inf"),
    )


def test_single_migration_runs_all_four_phases_then_calls_on_done() -> None:
    rec = _Recorder()
    engine = _engine(rec)
    done: list[object] = []
    mid = _begin(
        engine, rec, migrate_id_ranks=(0, 1), src_leader=0,
        on_done=lambda sched: done.append(sched), on_error=lambda e: done.append(e),
    )

    # Pump 1: nothing sent yet until pump; DESCRIBE goes to all participants.
    engine.pump([], now=0.0)
    assert rec.sends == [(_MIGRATE_PHASE_DESCRIBE_LAYOUT, (0, 1))]

    # All DESCRIBE replies in -> next pump sends BUILD to src leader only.
    engine.pump(
        [_Result(mid, 0, _MIGRATE_PHASE_DESCRIBE_LAYOUT),
         _Result(mid, 1, _MIGRATE_PHASE_DESCRIBE_LAYOUT)],
        now=0.0,
    )
    assert rec.sends[-1] == (_MIGRATE_PHASE_BUILD_SCHEDULE, (0,))

    # BUILD reply carries the schedule -> ACCEPT to all participants.
    engine.pump([_Result(mid, 0, _MIGRATE_PHASE_BUILD_SCHEDULE, schedule="SCHED")], now=0.0)
    assert rec.sends[-1] == (_MIGRATE_PHASE_ACCEPT_SCHEDULE, (0, 1))

    # ACCEPT replies -> EXECUTE to all participants.
    engine.pump(
        [_Result(mid, 0, _MIGRATE_PHASE_ACCEPT_SCHEDULE),
         _Result(mid, 1, _MIGRATE_PHASE_ACCEPT_SCHEDULE)],
        now=0.0,
    )
    assert rec.sends[-1] == (_MIGRATE_PHASE_EXECUTE_TRANSFER, (0, 1))

    # EXECUTE replies -> on_done with the captured schedule.
    assert done == []
    engine.pump(
        [_Result(mid, 0, _MIGRATE_PHASE_EXECUTE_TRANSFER),
         _Result(mid, 1, _MIGRATE_PHASE_EXECUTE_TRANSFER)],
        now=0.0,
    )
    assert done == ["SCHED"]


def test_disjoint_migrations_advance_concurrently() -> None:
    rec = _Recorder()
    engine = _engine(rec)
    m1 = _begin(engine, rec, migrate_id_ranks=(0, 1), src_leader=0,
                on_done=lambda s: None, on_error=lambda e: None)
    m2 = _begin(engine, rec, migrate_id_ranks=(2, 3), src_leader=2,
                on_done=lambda s: None, on_error=lambda e: None)

    engine.pump([], now=0.0)
    # Both send DESCRIBE in the same pump — full concurrency for disjoint sets.
    assert (_MIGRATE_PHASE_DESCRIBE_LAYOUT, (0, 1)) in rec.sends
    assert (_MIGRATE_PHASE_DESCRIBE_LAYOUT, (2, 3)) in rec.sends
    assert m1 != m2


def test_overlapping_migrations_serialize_by_seq() -> None:
    rec = _Recorder()
    engine = _engine(rec)
    done: list[str] = []
    m1 = _begin(engine, rec, migrate_id_ranks=(0, 1), src_leader=0,
                on_done=lambda s: done.append("m1"), on_error=lambda e: None)
    m2 = _begin(engine, rec, migrate_id_ranks=(1, 2), src_leader=1,
                on_done=lambda s: done.append("m2"), on_error=lambda e: None)

    # m1 (older seq) owns shared rank 1; m2 must send NOTHING until m1 finishes.
    engine.pump([], now=0.0)
    assert rec.sends == [(_MIGRATE_PHASE_DESCRIBE_LAYOUT, (0, 1))]

    # Drive m1 to completion.
    engine.pump([_Result(m1, 0, _MIGRATE_PHASE_DESCRIBE_LAYOUT),
                 _Result(m1, 1, _MIGRATE_PHASE_DESCRIBE_LAYOUT)], now=0.0)
    engine.pump([_Result(m1, 0, _MIGRATE_PHASE_BUILD_SCHEDULE, schedule="S")], now=0.0)
    engine.pump([_Result(m1, 0, _MIGRATE_PHASE_ACCEPT_SCHEDULE),
                 _Result(m1, 1, _MIGRATE_PHASE_ACCEPT_SCHEDULE)], now=0.0)
    # m2 still gated while m1 holds rank 1.
    assert all(ranks != (1, 2) for _, ranks in rec.sends)
    engine.pump([_Result(m1, 0, _MIGRATE_PHASE_EXECUTE_TRANSFER),
                 _Result(m1, 1, _MIGRATE_PHASE_EXECUTE_TRANSFER)], now=0.0)
    assert done == ["m1"]

    # m1 retired -> m2 becomes oldest on rank 1 and finally sends DESCRIBE.
    engine.pump([], now=0.0)
    assert rec.sends[-1] == (_MIGRATE_PHASE_DESCRIBE_LAYOUT, (1, 2))
    assert m2  # referenced


def test_deadline_exceeded_calls_on_error_once() -> None:
    rec = _Recorder()
    engine = _engine(rec)
    errors: list[Exception] = []
    mid = engine.begin(
        plan=None, src_group=None, dst_group=None,
        participant_ranks=(0, 1), src_leader_rank=0,
        on_done=lambda s: None, on_error=lambda e: errors.append(e),
        deadline=10.0,
    )
    engine.pump([], now=0.0)          # within deadline -> DESCRIBE sent
    engine.pump([], now=20.0)         # past deadline -> on_error
    engine.pump([], now=30.0)         # retired -> no second call
    assert len(errors) == 1
    assert isinstance(errors[0], TimeoutError)
    assert mid


def test_worker_error_result_calls_on_error() -> None:
    rec = _Recorder()
    engine = _engine(rec)
    errors: list[Exception] = []
    mid = _begin(engine, rec, migrate_id_ranks=(0, 1), src_leader=0,
                 on_done=lambda s: None, on_error=lambda e: errors.append(e))
    engine.pump([], now=0.0)
    engine.pump([_Result(mid, 0, _MIGRATE_PHASE_DESCRIBE_LAYOUT, error="boom")], now=0.0)
    assert len(errors) == 1
    assert "boom" in str(errors[0])
