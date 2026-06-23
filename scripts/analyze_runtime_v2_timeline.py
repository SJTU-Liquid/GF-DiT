#!/usr/bin/env python3
"""Analyze vllm-omni runtime_v2 logs and diagnose scheduler behavior.

Compared with the sglang helper, this script adds scheduler diagnostics to answer:
"Why does SRTF look the same as FCFS in my run?"

Outputs:
1) raw task timeline trace (from `runtime_v2 worker task done`)
2) merged-by-request task timeline trace
3) text summary with scheduler diagnostics
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Timestamp formats seen in logs (the leading `[Stage-N]` / `(APIServer pid=K)` prefixes
# emitted by the orchestrator are tolerated; we anchor on the `[file.py:LINE]` suffix):
# - INFO 03-24 13:13:50 [worker.py:72] ...
# - [Stage-0] INFO 05-17 11:15:44 [scheduler.py:1569] ...
# - (APIServer pid=12345) DEBUG 05-17 11:15:16 [omni_stage.py:260] ...
# - [2026-03-24 13:13:50] ...
TS_SHORT_RE = re.compile(r"(?P<ts>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[\w+\.py:\d+\]")
TS_BRACKET_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]")

RUNNER_POLICY_RE = re.compile(r"runtime_v2 runner started: .* policy=(?P<policy>\w+)")
SUBMIT_RE = re.compile(
    r"runtime_v2 submit: request_id=(?P<request_id>\S+) "
    r"(?:model_class_name=(?P<model_class_name>\S+) )?"
    r"chunk=(?P<chunk>\S+) steps=(?P<steps>\S+) frames=(?P<frames>\S+) size=(?P<size>\S+)"
)
WAIT_FINISHED_RE = re.compile(
    r"runtime_v2 wait finished: request_id=(?P<request_id>\S+) elapsed=(?P<elapsed>[0-9]*\.?[0-9]+)s"
)

DISPATCH_RE = re.compile(
    r"runtime_v2 dispatch: request_id=(?P<request_id>\S+) "
    r"task_id=(?P<task_id>\S+) kind=(?P<kind>\S+) group=(?P<group>\S+)"
    r"(?: step_range=\[(?P<step_start>\d+),(?P<step_end>\d+)\))?"
)

WORKER_DONE_RE = re.compile(
    r"runtime_v2 worker task done: rank=(?P<rank>\d+) "
    r"request_id=(?P<request_id>\S+) task_id=(?P<task_id>\S+) kind=(?P<kind>\S+) "
    r"outputs=(?P<outputs>\d+) elapsed=(?P<elapsed>[0-9]*\.?[0-9]+)s"
    r"(?:\s+mono_ns=(?P<mono_ns>\d+))?"
    r"(?:\s+total_elapsed_ns=(?P<total_elapsed_ns>\d+))?"
)

SCHED_TRACE_RE = re.compile(
    r"runtime_v2 scheduler trace: (?P<action>\S+)(?P<rest>.*)$"
)

RESHARD_BEGIN_RE = re.compile(
    r"runtime_v2 reshard begin: request_id=(?P<request_id>\S+) "
    r"task_id=(?P<task_id>\S+) src_group=(?P<src_group>\S+) dst_group=(?P<dst_group>\S+)"
)

RESHARD_DONE_RE = re.compile(
    r"runtime_v2 reshard done: request_id=(?P<request_id>\S+) "
    r"task_id=(?P<task_id>\S+) dst_leader_rank=(?P<dst_leader_rank>\S+)"
)

RESHARD_FAILED_RE = re.compile(
    r"runtime_v2 reshard failed: request_id=(?P<request_id>\S+) task_id=(?P<task_id>\S+)"
)

WORKER_OP_BEGIN_RE = re.compile(
    r"runtime_v2 worker op begin: rank=(?P<rank>\d+) op=migrate_artifacts "
    r"migrate_id=(?P<migrate_id>\S+) phase=(?P<phase>\S+) mono_ns=(?P<mono_ns>\d+)"
)

WORKER_OP_DONE_RE = re.compile(
    r"runtime_v2 worker op done: rank=(?P<rank>\d+) op=migrate_artifacts "
    r"migrate_id=(?P<migrate_id>\S+) phase=(?P<phase>\S+) status=(?P<status>\S+) "
    r"elapsed_ns=(?P<elapsed_ns>\d+) mono_ns=(?P<mono_ns>\d+)"
)


@dataclass
class DispatchEvent:
    line_no: int
    ts: datetime
    request_id: str
    task_id: str
    kind: str
    group: str
    step_start: int | None
    step_end: int | None


@dataclass
class WorkerDoneEvent:
    line_no: int
    ts_end: datetime
    worker: int
    request_id: str
    task_id: str
    kind: str
    elapsed_s: float
    mono_ns: int | None = None
    total_elapsed_ns: int | None = None

    @property
    def ts_start(self) -> datetime:
        if self.total_elapsed_ns is not None:
            return self.ts_end - timedelta(microseconds=self.total_elapsed_ns / 1000)
        return self.ts_end - timedelta(seconds=self.elapsed_s)


@dataclass
class RequestStats:
    request_id: str
    submit_ts: datetime | None = None
    finish_ts: datetime | None = None
    submit_line: int | None = None
    finish_line: int | None = None
    dispatch_count: int = 0
    dispatch_dit_count: int = 0
    first_dispatch_line: int | None = None
    first_dispatch_ts: datetime | None = None


@dataclass
class ReshardSpan:
    request_id: str
    task_id: str
    begin_ts: datetime
    end_ts: datetime
    src_group: str
    dst_group: str
    dst_leader_rank: int | None
    failed: bool


@dataclass
class WorkerPhaseSpan:
    worker: int
    migrate_id: str
    phase: str
    begin_ts: datetime
    end_ts: datetime
    elapsed_ns: int
    status: str
    matched_begin: bool
    begin_mono_ns: int | None = None
    end_mono_ns: int | None = None


@dataclass
class SchedulerPollSpan:
    begin_mono_ns: int
    end_mono_ns: int
    elapsed_ns: int
    event_count: int
    begin_ts: datetime  # filled by mono-anchor pass


@dataclass
class SchedulerInstantEvent:
    action: str
    mono_ns: int
    fields: Dict[str, str]
    ts: datetime  # filled by mono-anchor pass


@dataclass
class MergedSlice:
    worker: int
    request_id: str
    start_ts: datetime
    end_ts: datetime
    task_count: int
    dit_step_count: int
    first_task_id: str
    last_task_id: str

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_ts - self.start_ts).total_seconds())


def _parse_ts(raw_line: str, year_hint: int) -> datetime | None:
    line = ANSI_ESCAPE_RE.sub("", raw_line)
    m = TS_SHORT_RE.search(line)
    if m:
        try:
            return datetime.strptime(f"{year_hint}-{m.group('ts')}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    m = TS_BRACKET_RE.match(line)
    if m:
        try:
            return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def _detect_year_hint(log_path: Path, cli_year: int | None) -> int:
    if cli_year is not None:
        return cli_year
    now_year = datetime.now().year
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            clean = ANSI_ESCAPE_RE.sub("", line)
            m = TS_BRACKET_RE.match(clean)
            if not m:
                continue
            ts = m.group("ts")
            if len(ts) >= 4 and ts[:4].isdigit():
                return int(ts[:4])
    return now_year


def _is_dit_chunk_kind(kind: str) -> bool:
    return "DIT_STEP_CHUNK" in kind or kind.endswith("dit_step_chunk")


def parse_events(
    log_path: Path, year_hint: int
) -> tuple[
    str | None,
    List[DispatchEvent],
    List[WorkerDoneEvent],
    Dict[str, RequestStats],
    List[ReshardSpan],
    List[WorkerPhaseSpan],
    List[SchedulerPollSpan],
    List[SchedulerInstantEvent],
]:
    policy_name: str | None = None
    dispatch_events: List[DispatchEvent] = []
    worker_done_events: List[WorkerDoneEvent] = []
    req_stats: Dict[str, RequestStats] = {}
    reshard_spans: List[ReshardSpan] = []
    worker_phase_spans: List[WorkerPhaseSpan] = []
    sched_poll_spans: List[SchedulerPollSpan] = []
    sched_instants: List[SchedulerInstantEvent] = []
    pending_reshard_begins: Dict[Tuple[str, str], Tuple[datetime, str, str]] = {}
    pending_worker_op_begins: Dict[Tuple[int, str, str], Tuple[datetime, int | None]] = {}
    failed_reshard_keys: set[Tuple[str, str]] = set()
    pending_poll_begin_mono: int | None = None

    def get_req(req_id: str) -> RequestStats:
        st = req_stats.get(req_id)
        if st is None:
            st = RequestStats(request_id=req_id)
            req_stats[req_id] = st
        return st

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = ANSI_ESCAPE_RE.sub("", raw_line)
            if "runtime_v2" not in line:
                continue

            ts = _parse_ts(line, year_hint)
            if ts is None:
                continue

            m = RUNNER_POLICY_RE.search(line)
            if m and policy_name is None:
                policy_name = m.group("policy")

            m = SUBMIT_RE.search(line)
            if m:
                req_id = m.group("request_id")
                st = get_req(req_id)
                if st.submit_ts is None:
                    st.submit_ts = ts
                    st.submit_line = line_no

            m = WAIT_FINISHED_RE.search(line)
            if m:
                req_id = m.group("request_id")
                st = get_req(req_id)
                st.finish_ts = ts
                st.finish_line = line_no

            m = DISPATCH_RE.search(line)
            if m:
                req_id = m.group("request_id")
                kind = m.group("kind")
                ev = DispatchEvent(
                    line_no=line_no,
                    ts=ts,
                    request_id=req_id,
                    task_id=m.group("task_id"),
                    kind=kind,
                    group=m.group("group"),
                    step_start=int(m.group("step_start")) if m.group("step_start") is not None else None,
                    step_end=int(m.group("step_end")) if m.group("step_end") is not None else None,
                )
                dispatch_events.append(ev)

                st = get_req(req_id)
                st.dispatch_count += 1
                if _is_dit_chunk_kind(kind):
                    st.dispatch_dit_count += 1
                if st.first_dispatch_line is None:
                    st.first_dispatch_line = line_no
                    st.first_dispatch_ts = ts

            m = WORKER_DONE_RE.search(line)
            if m:
                mono_ns_str = m.group("mono_ns")
                total_elapsed_str = m.group("total_elapsed_ns")
                worker_done_events.append(
                    WorkerDoneEvent(
                        line_no=line_no,
                        ts_end=ts,
                        worker=int(m.group("rank")),
                        request_id=m.group("request_id"),
                        task_id=m.group("task_id"),
                        kind=m.group("kind"),
                        elapsed_s=float(m.group("elapsed")),
                        mono_ns=int(mono_ns_str) if mono_ns_str else None,
                        total_elapsed_ns=int(total_elapsed_str) if total_elapsed_str else None,
                    )
                )

            m = RESHARD_BEGIN_RE.search(line)
            if m:
                pending_reshard_begins[(m.group("request_id"), m.group("task_id"))] = (
                    ts,
                    m.group("src_group"),
                    m.group("dst_group"),
                )
                continue

            m = RESHARD_FAILED_RE.search(line)
            if m:
                failed_reshard_keys.add((m.group("request_id"), m.group("task_id")))
                continue

            m = RESHARD_DONE_RE.search(line)
            if m:
                key = (m.group("request_id"), m.group("task_id"))
                begin = pending_reshard_begins.pop(key, None)
                if begin is None:
                    continue
                begin_ts, src_group, dst_group = begin
                dst_rank_raw = m.group("dst_leader_rank")
                try:
                    dst_rank = int(dst_rank_raw)
                except ValueError:
                    dst_rank = None
                reshard_spans.append(
                    ReshardSpan(
                        request_id=m.group("request_id"),
                        task_id=m.group("task_id"),
                        begin_ts=begin_ts,
                        end_ts=ts,
                        src_group=src_group,
                        dst_group=dst_group,
                        dst_leader_rank=dst_rank,
                        failed=key in failed_reshard_keys,
                    )
                )
                continue

            m = WORKER_OP_BEGIN_RE.search(line)
            if m:
                begin_mono = int(m.group("mono_ns"))
                pending_worker_op_begins[
                    (int(m.group("rank")), m.group("migrate_id"), m.group("phase"))
                ] = (ts, begin_mono)
                continue

            m = WORKER_OP_DONE_RE.search(line)
            if m:
                rank = int(m.group("rank"))
                migrate_id = m.group("migrate_id")
                phase = m.group("phase")
                elapsed_ns = int(m.group("elapsed_ns"))
                end_mono = int(m.group("mono_ns"))
                pair = pending_worker_op_begins.pop((rank, migrate_id, phase), None)
                matched = pair is not None
                if pair is None:
                    begin_ts = ts - timedelta(microseconds=elapsed_ns / 1000)
                    begin_mono = end_mono - elapsed_ns
                else:
                    begin_ts, begin_mono = pair
                end_ts = begin_ts + timedelta(microseconds=elapsed_ns / 1000)
                worker_phase_spans.append(
                    WorkerPhaseSpan(
                        worker=rank,
                        migrate_id=migrate_id,
                        phase=phase,
                        begin_ts=begin_ts,
                        end_ts=end_ts,
                        elapsed_ns=elapsed_ns,
                        status=m.group("status"),
                        matched_begin=matched,
                        begin_mono_ns=begin_mono,
                        end_mono_ns=end_mono,
                    )
                )
                continue

            m = SCHED_TRACE_RE.search(line)
            if m:
                action = m.group("action")
                rest = m.group("rest")
                fields_dict: Dict[str, str] = {}
                for part in rest.split():
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    fields_dict[k] = v
                mono_ns_str = fields_dict.pop("mono_ns", None)
                if mono_ns_str is None:
                    continue
                trace_mono_ns = int(mono_ns_str)
                if action == "poll_begin":
                    pending_poll_begin_mono = trace_mono_ns
                    continue
                if action == "poll_end":
                    poll_elapsed_ns = int(fields_dict.get("elapsed_ns", "0"))
                    event_count = int(fields_dict.get("event_count", "0"))
                    if pending_poll_begin_mono is not None:
                        begin_mono = pending_poll_begin_mono
                        pending_poll_begin_mono = None
                    else:
                        begin_mono = trace_mono_ns - poll_elapsed_ns
                    sched_poll_spans.append(
                        SchedulerPollSpan(
                            begin_mono_ns=begin_mono,
                            end_mono_ns=trace_mono_ns,
                            elapsed_ns=poll_elapsed_ns,
                            event_count=event_count,
                            begin_ts=ts,  # placeholder; re-stamped after anchor
                        )
                    )
                    continue
                sched_instants.append(
                    SchedulerInstantEvent(
                        action=action,
                        mono_ns=trace_mono_ns,
                        fields=fields_dict,
                        ts=ts,  # placeholder; re-stamped after anchor
                    )
                )
                continue

    dispatch_events.sort(key=lambda e: e.line_no)
    worker_done_events.sort(key=lambda e: (e.worker, e.line_no))
    reshard_spans.sort(key=lambda s: s.begin_ts)
    worker_phase_spans.sort(key=lambda p: (p.worker, p.begin_ts))
    sched_poll_spans.sort(key=lambda s: s.begin_mono_ns)
    sched_instants.sort(key=lambda s: s.mono_ns)

    # mono_ns ↔ wall-clock anchor pass. Wall-clock timestamps in vllm logs are
    # second-resolution which is far too coarse for sub-second event analysis
    # (causes phantom ~700ms gaps between adjacent sub-second tasks). We anchor
    # off the earliest event that carries both a parsed wall-clock and a mono_ns
    # field, then re-stamp every other mono_ns-bearing event in that domain.
    anchor: Tuple[int, datetime] | None = None
    for ev in worker_done_events:
        if ev.mono_ns is not None:
            anchor = (ev.mono_ns, ev.ts_end)
            break
    if anchor is None:
        for sp in worker_phase_spans:
            if sp.end_mono_ns is not None:
                anchor = (sp.end_mono_ns, sp.end_ts)
                break
    if anchor is None and sched_poll_spans:
        sp = sched_poll_spans[0]
        anchor = (sp.end_mono_ns, sp.begin_ts)
    if anchor is None and sched_instants:
        si = sched_instants[0]
        anchor = (si.mono_ns, si.ts)

    if anchor is not None:
        anchor_mono_ns, anchor_ts = anchor

        def mono_to_wall(mono_ns: int) -> datetime:
            return anchor_ts + timedelta(microseconds=(mono_ns - anchor_mono_ns) / 1000)

        for ev in worker_done_events:
            if ev.mono_ns is not None:
                ev.ts_end = mono_to_wall(ev.mono_ns)
        for sp in worker_phase_spans:
            if sp.begin_mono_ns is not None and sp.end_mono_ns is not None:
                sp.begin_ts = mono_to_wall(sp.begin_mono_ns)
                sp.end_ts = mono_to_wall(sp.end_mono_ns)
        for poll in sched_poll_spans:
            poll.begin_ts = mono_to_wall(poll.begin_mono_ns)
        for inst in sched_instants:
            inst.ts = mono_to_wall(inst.mono_ns)

    return (
        policy_name,
        dispatch_events,
        worker_done_events,
        req_stats,
        reshard_spans,
        worker_phase_spans,
        sched_poll_spans,
        sched_instants,
    )


def normalize_worker_done(events: Sequence[WorkerDoneEvent], idle_threshold_s: float) -> List[WorkerDoneEvent]:
    grouped: Dict[int, List[WorkerDoneEvent]] = defaultdict(list)
    for e in events:
        grouped[e.worker].append(e)

    out: List[WorkerDoneEvent] = []
    idle_threshold = timedelta(seconds=idle_threshold_s)

    for worker, items in grouped.items():
        items.sort(key=lambda e: e.line_no)
        prev_end: datetime | None = None
        for ev in items:
            observed_start = ev.ts_start
            if prev_end is None:
                start = observed_start
            else:
                if observed_start - prev_end > idle_threshold:
                    start = observed_start
                else:
                    start = max(prev_end, observed_start)
            duration = (
                timedelta(microseconds=ev.total_elapsed_ns / 1000)
                if ev.total_elapsed_ns is not None
                else timedelta(seconds=ev.elapsed_s)
            )
            end = start + duration
            prev_end = end
            out.append(
                WorkerDoneEvent(
                    line_no=ev.line_no,
                    ts_end=end,
                    worker=ev.worker,
                    request_id=ev.request_id,
                    task_id=ev.task_id,
                    kind=ev.kind,
                    elapsed_s=ev.elapsed_s,
                    mono_ns=ev.mono_ns,
                    total_elapsed_ns=ev.total_elapsed_ns,
                )
            )

    out.sort(key=lambda e: (e.worker, e.line_no))
    return out


def merge_consecutive_by_request(events: Sequence[WorkerDoneEvent]) -> List[MergedSlice]:
    grouped: Dict[int, List[WorkerDoneEvent]] = defaultdict(list)
    for e in events:
        grouped[e.worker].append(e)

    merged: List[MergedSlice] = []
    for worker, items in grouped.items():
        items.sort(key=lambda e: e.line_no)
        cur: MergedSlice | None = None
        for ev in items:
            if cur is None:
                cur = MergedSlice(
                    worker=worker,
                    request_id=ev.request_id,
                    start_ts=ev.ts_start,
                    end_ts=ev.ts_end,
                    task_count=1,
                    dit_step_count=1 if _is_dit_chunk_kind(ev.kind) else 0,
                    first_task_id=ev.task_id,
                    last_task_id=ev.task_id,
                )
                continue
            if ev.request_id == cur.request_id:
                cur.end_ts = max(cur.end_ts, ev.ts_end)
                cur.task_count += 1
                if _is_dit_chunk_kind(ev.kind):
                    cur.dit_step_count += 1
                cur.last_task_id = ev.task_id
            else:
                merged.append(cur)
                cur = MergedSlice(
                    worker=worker,
                    request_id=ev.request_id,
                    start_ts=ev.ts_start,
                    end_ts=ev.ts_end,
                    task_count=1,
                    dit_step_count=1 if _is_dit_chunk_kind(ev.kind) else 0,
                    first_task_id=ev.task_id,
                    last_task_id=ev.task_id,
                )
        if cur is not None:
            merged.append(cur)

    merged.sort(key=lambda s: (s.worker, s.start_ts, s.end_ts))
    return merged


def _short_id(s: str, n: int = 10) -> str:
    return s[:n]


def _to_trace_events(
    raw_events: Sequence[WorkerDoneEvent],
    merged_slices: Sequence[MergedSlice],
    reshard_spans: Sequence[ReshardSpan],
    worker_phase_spans: Sequence[WorkerPhaseSpan],
    sched_poll_spans: Sequence[SchedulerPollSpan],
    sched_instants: Sequence[SchedulerInstantEvent],
    include_raw: bool,
) -> List[dict]:
    starts = (
        [e.ts_start for e in raw_events]
        + [s.start_ts for s in merged_slices]
        + [r.begin_ts for r in reshard_spans]
        + [p.begin_ts for p in worker_phase_spans]
        + [sp.begin_ts for sp in sched_poll_spans]
        + [si.ts for si in sched_instants]
    )
    if not starts:
        return []
    t0 = min(starts)

    def ts_us(ts: datetime) -> int:
        return int(round((ts - t0).total_seconds() * 1_000_000))

    def dur_us(s: float) -> int:
        return max(1, int(round(s * 1_000_000)))

    workers = sorted(
        {e.worker for e in raw_events}
        | {s.worker for s in merged_slices}
        | {p.worker for p in worker_phase_spans}
    )
    tid: Dict[Tuple[Any, str], int] = {}
    cur_tid = 0
    for w in workers:
        tid[(w, "raw")] = cur_tid
        cur_tid += 1
        tid[(w, "merged")] = cur_tid
        cur_tid += 1
        tid[(w, "reshard")] = cur_tid
        cur_tid += 1
    tid[("scheduler", "reshard")] = cur_tid
    cur_tid += 1
    tid[("scheduler", "poll")] = cur_tid
    cur_tid += 1
    tid[("scheduler", "events")] = cur_tid
    cur_tid += 1

    trace: List[dict] = [
        {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "vllm-omni runtime_v2"}}
    ]
    for w in workers:
        if include_raw:
            trace.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 0,
                    "tid": tid[(w, "raw")],
                    "args": {"name": f"worker-{w} raw_tasks"},
                }
            )
        trace.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid[(w, "merged")],
                "args": {"name": f"worker-{w} merged_requests"},
            }
        )
        trace.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid[(w, "reshard")],
                "args": {"name": f"worker-{w} reshard_phases"},
            }
        )
    trace.append(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": tid[("scheduler", "reshard")],
            "args": {"name": "scheduler reshard"},
        }
    )
    trace.append(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": tid[("scheduler", "poll")],
            "args": {"name": "scheduler poll"},
        }
    )
    trace.append(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": tid[("scheduler", "events")],
            "args": {"name": "scheduler events"},
        }
    )

    if include_raw:
        for ev in raw_events:
            task_tail = ":".join(ev.task_id.split(":")[-2:])
            trace.append(
                {
                    "name": f"{_short_id(ev.request_id)}:{task_tail}",
                    "cat": "raw_task",
                    "ph": "X",
                    "pid": 0,
                    "tid": tid[(ev.worker, "raw")],
                    "ts": ts_us(ev.ts_start),
                    "dur": dur_us(ev.elapsed_s),
                    "args": {
                        "request_id": ev.request_id,
                        "task_id": ev.task_id,
                        "kind": ev.kind,
                        "worker": ev.worker,
                        "elapsed_s": ev.elapsed_s,
                    },
                }
            )

    for sl in merged_slices:
        trace.append(
            {
                "name": f"{_short_id(sl.request_id)} x{sl.task_count}",
                "cat": "merged_request",
                "ph": "X",
                "pid": 0,
                "tid": tid[(sl.worker, "merged")],
                "ts": ts_us(sl.start_ts),
                "dur": dur_us(sl.duration_s),
                "args": {
                    "request_id": sl.request_id,
                    "worker": sl.worker,
                    "task_count": sl.task_count,
                    "dit_step_count": sl.dit_step_count,
                    "first_task_id": sl.first_task_id,
                    "last_task_id": sl.last_task_id,
                },
            }
        )

    for p in worker_phase_spans:
        trace.append(
            {
                "name": p.phase,
                "cat": "reshard_phase",
                "ph": "X",
                "pid": 0,
                "tid": tid[(p.worker, "reshard")],
                "ts": ts_us(p.begin_ts),
                "dur": dur_us((p.end_ts - p.begin_ts).total_seconds()),
                "args": {
                    "worker": p.worker,
                    "migrate_id": p.migrate_id,
                    "phase": p.phase,
                    "status": p.status,
                    "elapsed_ns": p.elapsed_ns,
                    "matched_begin": p.matched_begin,
                },
            }
        )

    for r in reshard_spans:
        trace.append(
            {
                "name": f"{_short_id(r.request_id)} {r.src_group}->{r.dst_group}"
                + (" [FAILED]" if r.failed else ""),
                "cat": "reshard",
                "ph": "X",
                "pid": 0,
                "tid": tid[("scheduler", "reshard")],
                "ts": ts_us(r.begin_ts),
                "dur": dur_us((r.end_ts - r.begin_ts).total_seconds()),
                "args": {
                    "request_id": r.request_id,
                    "task_id": r.task_id,
                    "src_group": r.src_group,
                    "dst_group": r.dst_group,
                    "dst_leader_rank": r.dst_leader_rank,
                    "failed": r.failed,
                },
            }
        )

    # Derived migration spans, grouped by migrate_id across all ranks/phases. This
    # populates the scheduler reshard row even when only the data_plane logs are
    # present (e.g. implicit migrations on the edf_greedy path, which don't emit
    # `runtime_v2 reshard begin/done`).
    derived: Dict[str, Dict[str, Any]] = {}
    for p in worker_phase_spans:
        cur = derived.get(p.migrate_id)
        if cur is None:
            derived[p.migrate_id] = {
                "begin": p.begin_ts,
                "end": p.end_ts,
                "phases": {p.phase},
                "ranks": {p.worker},
                "phase_count": 1,
            }
        else:
            if p.begin_ts < cur["begin"]:
                cur["begin"] = p.begin_ts
            if p.end_ts > cur["end"]:
                cur["end"] = p.end_ts
            cur["phases"].add(p.phase)
            cur["ranks"].add(p.worker)
            cur["phase_count"] += 1
    for mig_id, d in derived.items():
        trace.append(
            {
                "name": f"migrate {mig_id[:8]} ({len(d['ranks'])}r,{len(d['phases'])}p)",
                "cat": "migration_derived",
                "ph": "X",
                "pid": 0,
                "tid": tid[("scheduler", "reshard")],
                "ts": ts_us(d["begin"]),
                "dur": dur_us((d["end"] - d["begin"]).total_seconds()),
                "args": {
                    "migrate_id": mig_id,
                    "phase_count": d["phase_count"],
                    "rank_count": len(d["ranks"]),
                    "phases": sorted(d["phases"]),
                },
            }
        )

    for poll in sched_poll_spans:
        end_ts = poll.begin_ts + timedelta(microseconds=poll.elapsed_ns / 1000)
        trace.append(
            {
                "name": f"poll ({poll.event_count}ev)" if poll.event_count else "poll (idle)",
                "cat": "scheduler_poll",
                "ph": "X",
                "pid": 0,
                "tid": tid[("scheduler", "poll")],
                "ts": ts_us(poll.begin_ts),
                "dur": dur_us((end_ts - poll.begin_ts).total_seconds()),
                "args": {
                    "elapsed_ns": poll.elapsed_ns,
                    "event_count": poll.event_count,
                },
            }
        )

    for inst in sched_instants:
        trace.append(
            {
                "name": inst.action,
                "cat": "scheduler_event",
                "ph": "i",
                "pid": 0,
                "tid": tid[("scheduler", "events")],
                "ts": ts_us(inst.ts),
                "s": "t",
                "args": dict(inst.fields),
            }
        )

    return trace


def _iter_live_spans(req_stats: Iterable[RequestStats], fallback_end: datetime | None) -> List[Tuple[datetime, datetime, str]]:
    spans: List[Tuple[datetime, datetime, str]] = []
    for st in req_stats:
        if st.submit_ts is None:
            continue
        end = st.finish_ts or fallback_end
        if end is None or end < st.submit_ts:
            continue
        spans.append((st.submit_ts, end, st.request_id))
    return spans


def _max_live_requests(spans: Sequence[Tuple[datetime, datetime, str]]) -> int:
    points: List[Tuple[datetime, int]] = []
    for start, end, _ in spans:
        points.append((start, +1))
        points.append((end, -1))
    points.sort(key=lambda x: (x[0], x[1]))
    live = 0
    max_live = 0
    for _, delta in points:
        live += delta
        if live > max_live:
            max_live = live
    return max_live


def _count_dispatch_switches(dispatch_events: Sequence[DispatchEvent], dit_only: bool) -> tuple[int, int]:
    prev_req: str | None = None
    switches = 0
    total = 0
    for ev in dispatch_events:
        if dit_only and (not _is_dit_chunk_kind(ev.kind)):
            continue
        if prev_req is not None and prev_req != ev.request_id:
            switches += 1
        prev_req = ev.request_id
        total += 1
    return switches, total


def _count_multi_live_dispatch(dispatch_events: Sequence[DispatchEvent], spans: Sequence[Tuple[datetime, datetime, str]]) -> int:
    cnt = 0
    for ev in dispatch_events:
        live = 0
        for s, e, _ in spans:
            if s <= ev.ts <= e:
                live += 1
                if live >= 2:
                    cnt += 1
                    break
    return cnt


def _format_summary(
    log_path: Path,
    policy_name: str | None,
    dispatch_events: Sequence[DispatchEvent],
    worker_done_events: Sequence[WorkerDoneEvent],
    merged_slices: Sequence[MergedSlice],
    req_stats: Dict[str, RequestStats],
    reshard_spans: Sequence[ReshardSpan],
    worker_phase_spans: Sequence[WorkerPhaseSpan],
    sched_poll_spans: Sequence[SchedulerPollSpan],
    sched_instants: Sequence[SchedulerInstantEvent],
    top_n: int,
) -> str:
    lines: List[str] = []
    lines.append(f"log_file: {log_path}")
    lines.append(f"policy_from_log: {policy_name or 'unknown'}")
    lines.append(f"requests_seen: {len(req_stats)}")
    lines.append(f"dispatch_events: {len(dispatch_events)}")
    lines.append(f"worker_done_events: {len(worker_done_events)}")
    lines.append(f"merged_worker_slices: {len(merged_slices)}")

    fallback_end = max((e.ts_end for e in worker_done_events), default=None)
    spans = _iter_live_spans(req_stats.values(), fallback_end)
    max_live = _max_live_requests(spans)
    lines.append(f"max_live_requests(submit..finish): {max_live}")

    sw_all, n_all = _count_dispatch_switches(dispatch_events, dit_only=False)
    sw_dit, n_dit = _count_dispatch_switches(dispatch_events, dit_only=True)
    lines.append(f"dispatch_request_switches(all): {sw_all}/{max(0, n_all - 1)}")
    lines.append(f"dispatch_request_switches(dit_only): {sw_dit}/{max(0, n_dit - 1)}")

    multi_live_dispatch = _count_multi_live_dispatch(dispatch_events, spans)
    lines.append(f"dispatch_events_when_live_requests>=2: {multi_live_dispatch}")

    # Heuristic diagnosis.
    lines.append("")
    lines.append("diagnosis:")
    reasons: List[str] = []
    if max_live <= 1:
        reasons.append("No request-level contention: max_live_requests <= 1.")
    if multi_live_dispatch == 0:
        reasons.append("No dispatch happened while >=2 requests were live.")
    if n_dit > 1 and sw_dit == 0:
        reasons.append("No cross-request interleaving in DIT_STEP_CHUNK dispatches.")
    if not reasons:
        reasons.append("SRTF had opportunities; inspect merged trace for subtle differences.")
    for r in reasons:
        lines.append(f"  - {r}")

    # Top requests by total worker execution time.
    req_worker_time: Dict[str, float] = defaultdict(float)
    req_worker_tasks: Dict[str, int] = defaultdict(int)
    for ev in worker_done_events:
        req_worker_time[ev.request_id] += ev.elapsed_s
        req_worker_tasks[ev.request_id] += 1

    top = sorted(req_worker_time.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    lines.append("")
    lines.append(f"top_requests_by_worker_elapsed_s (top {top_n}):")
    for i, (req_id, dur) in enumerate(top, start=1):
        st = req_stats.get(req_id)
        submit_line = st.submit_line if st else None
        first_dispatch_line = st.first_dispatch_line if st else None
        finish_line = st.finish_line if st else None
        lines.append(
            f"  {i:>2}. {req_id} elapsed_s={dur:.3f} tasks={req_worker_tasks[req_id]} "
            f"submit_line={submit_line} first_dispatch_line={first_dispatch_line} finish_line={finish_line}"
        )

    # First merged slices (quick eyeballing of interleaving).
    lines.append("")
    lines.append("first_30_merged_worker_slices:")
    for i, sl in enumerate(merged_slices[:30], start=1):
        lines.append(
            f"  {i:>2}. worker={sl.worker} req={_short_id(sl.request_id)} "
            f"dur_s={sl.duration_s:.3f} tasks={sl.task_count} dit_steps={sl.dit_step_count}"
        )

    # Reshard stats.
    lines.append("")
    lines.append(f"reshard_spans: {len(reshard_spans)}")
    lines.append(f"worker_phase_spans: {len(worker_phase_spans)}")
    if reshard_spans:
        total_reshard_s = sum((r.end_ts - r.begin_ts).total_seconds() for r in reshard_spans)
        failed = sum(1 for r in reshard_spans if r.failed)
        lines.append(f"reshard_total_s: {total_reshard_s:.3f}")
        lines.append(f"reshard_failed: {failed}")
        per_transition_count: Dict[Tuple[str, str], int] = defaultdict(int)
        per_transition_dur: Dict[Tuple[str, str], float] = defaultdict(float)
        for r in reshard_spans:
            k = (r.src_group, r.dst_group)
            per_transition_count[k] += 1
            per_transition_dur[k] += (r.end_ts - r.begin_ts).total_seconds()
        for k in sorted(per_transition_count.keys()):
            lines.append(
                f"  reshard {k[0]}->{k[1]}: count={per_transition_count[k]} total_s={per_transition_dur[k]:.3f}"
            )
    if worker_phase_spans:
        phase_count: Dict[str, int] = defaultdict(int)
        phase_total_ns: Dict[str, int] = defaultdict(int)
        phase_unmatched: Dict[str, int] = defaultdict(int)
        for p in worker_phase_spans:
            phase_count[p.phase] += 1
            phase_total_ns[p.phase] += p.elapsed_ns
            if not p.matched_begin:
                phase_unmatched[p.phase] += 1
        for phase in sorted(phase_count.keys()):
            unmatched = phase_unmatched.get(phase, 0)
            note = f" unmatched_begin={unmatched}" if unmatched else ""
            lines.append(
                f"  phase {phase}: count={phase_count[phase]} "
                f"total_ms={phase_total_ns[phase]/1e6:.3f}{note}"
            )

    # Scheduler trace stats (only present when VLLM_RUNTIME_V2_SCHED_TRACE=1).
    lines.append("")
    lines.append(f"scheduler_poll_spans: {len(sched_poll_spans)}")
    lines.append(f"scheduler_instant_events: {len(sched_instants)}")
    if sched_poll_spans:
        idle = sum(1 for p in sched_poll_spans if p.event_count == 0)
        durs = sorted(p.elapsed_ns for p in sched_poll_spans)
        p50 = durs[len(durs) // 2]
        p95 = durs[int(len(durs) * 0.95)] if len(durs) > 1 else durs[0]
        pmax = durs[-1]
        lines.append(
            f"  poll: idle={idle}/{len(sched_poll_spans)} "
            f"p50_us={p50/1000:.1f} p95_us={p95/1000:.1f} max_us={pmax/1000:.1f}"
        )
        # Inter-poll gap: end of one poll → begin of next. Spikes here mean the
        # scheduler thread is busy elsewhere (or stuck) between polls.
        gaps_ns: List[int] = []
        for i in range(1, len(sched_poll_spans)):
            gap = sched_poll_spans[i].begin_mono_ns - sched_poll_spans[i - 1].end_mono_ns
            if gap > 0:
                gaps_ns.append(gap)
        if gaps_ns:
            gaps_ns.sort()
            gmax = gaps_ns[-1]
            gp95 = gaps_ns[int(len(gaps_ns) * 0.95)] if len(gaps_ns) > 1 else gaps_ns[0]
            big = sum(1 for g in gaps_ns if g >= 100_000_000)  # >= 100ms
            lines.append(
                f"  inter_poll_gap: p95_ms={gp95/1e6:.3f} max_ms={gmax/1e6:.3f} ge_100ms_count={big}"
            )
    if sched_instants:
        action_count: Dict[str, int] = defaultdict(int)
        for inst in sched_instants:
            action_count[inst.action] += 1
        for action in sorted(action_count.keys()):
            lines.append(f"  action {action}: {action_count[action]}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze vllm-omni runtime_v2 logs and scheduler behavior.")
    parser.add_argument("--log-file", default="log.log", help="Path to log file (default: log.log)")
    parser.add_argument(
        "--trace-out",
        default=None,
        help="Chrome trace output (raw + merged). Default: <logstem>.trace.json",
    )
    parser.add_argument(
        "--merged-trace-out",
        default=None,
        help="Chrome trace output (merged only). Default: <logstem>.merged.trace.json",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Text summary output. Default: <logstem>.summary.txt",
    )
    parser.add_argument(
        "--idle-threshold-s",
        type=float,
        default=1.2,
        help="Treat <= this gap as log jitter when normalizing worker timelines",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top N requests in summary")
    parser.add_argument("--year", type=int, default=None, help="Fallback year for short timestamps")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    log_stem = log_path.stem or "runtime_v2_timeline"
    if args.trace_out is None:
        args.trace_out = f"{log_stem}.trace.json"
    if args.merged_trace_out is None:
        args.merged_trace_out = f"{log_stem}.merged.trace.json"
    if args.summary_out is None:
        args.summary_out = f"{log_stem}.summary.txt"

    year_hint = _detect_year_hint(log_path, args.year)
    (
        policy_name,
        dispatch_events,
        worker_done_events,
        req_stats,
        reshard_spans,
        worker_phase_spans,
        sched_poll_spans,
        sched_instants,
    ) = parse_events(log_path, year_hint)

    if (
        not dispatch_events
        and not worker_done_events
        and not reshard_spans
        and not worker_phase_spans
        and not sched_poll_spans
    ):
        raise RuntimeError("No runtime_v2 dispatch/worker_done/reshard events found in log.")

    normalized_done = normalize_worker_done(worker_done_events, idle_threshold_s=args.idle_threshold_s)
    merged = merge_consecutive_by_request(normalized_done)

    trace_payload = {
        "traceEvents": _to_trace_events(
            normalized_done,
            merged,
            reshard_spans,
            worker_phase_spans,
            sched_poll_spans,
            sched_instants,
            include_raw=True,
        ),
        "displayTimeUnit": "ms",
    }
    merged_trace_payload = {
        "traceEvents": _to_trace_events(
            normalized_done,
            merged,
            reshard_spans,
            worker_phase_spans,
            sched_poll_spans,
            sched_instants,
            include_raw=False,
        ),
        "displayTimeUnit": "ms",
    }

    trace_out = Path(args.trace_out)
    merged_trace_out = Path(args.merged_trace_out)
    summary_out = Path(args.summary_out)

    trace_out.write_text(json.dumps(trace_payload, ensure_ascii=False), encoding="utf-8")
    merged_trace_out.write_text(json.dumps(merged_trace_payload, ensure_ascii=False), encoding="utf-8")

    summary = _format_summary(
        log_path=log_path,
        policy_name=policy_name,
        dispatch_events=dispatch_events,
        worker_done_events=normalized_done,
        merged_slices=merged,
        req_stats=req_stats,
        reshard_spans=reshard_spans,
        worker_phase_spans=worker_phase_spans,
        sched_poll_spans=sched_poll_spans,
        sched_instants=sched_instants,
        top_n=args.top_n,
    )
    summary_out.write_text(summary, encoding="utf-8")

    print(f"policy_from_log: {policy_name or 'unknown'}")
    print(f"dispatch_events: {len(dispatch_events)}")
    print(f"worker_done_events: {len(normalized_done)}")
    print(f"merged_worker_slices: {len(merged)}")
    print(f"reshard_spans: {len(reshard_spans)}")
    print(f"worker_phase_spans: {len(worker_phase_spans)}")
    print(f"scheduler_poll_spans: {len(sched_poll_spans)}")
    print(f"scheduler_instant_events: {len(sched_instants)}")
    print(f"summary: {summary_out}")
    print(f"trace: {trace_out}")
    print(f"merged_trace: {merged_trace_out}")


if __name__ == "__main__":
    main()
