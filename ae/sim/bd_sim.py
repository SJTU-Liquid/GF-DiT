"""Breakdown driver: wraps runtime_v2_mixed_priority_runner with
(1) an NCCL-group-setup cost model injected into task durations, and
(2) reconfiguration / group / policy-overhead accounting.

Env:
  BD_SETUP_MODE   = none | uncached | cached     (default none == GFC, ~60us -> 0)
  BD_GFC_US       = per-new-descriptor cost in us when mode=none (default 60; applied like uncached)
  BD_OUT_EXTRA    = path for the extra-stats JSON (default <results-out>.extra.json)
All runner CLI args pass through unchanged.

NCCL cold setup (H20 8-GPU, torch 2.10 / NCCL 2.27.5, ae/figures/logs/nccl_subgroup_cost_h20_sizesweep.json,
total_create_ms_max = new_group + first collective): size2 218.1, 4 532.5, 6 664.2, 8 778.4 ms;
sizes 3/5/7 linearly interpolated. Per-group memory 484 MB (size 2) / 535 MB (size>=4).
The full-world group is treated as pre-created at startup (free) in every mode.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # artifact repo root
import benchmarks.diffusion.runtime_v2_mixed_priority_runner as runner  # noqa: E402
from benchmarks.diffusion.runtime_v2_policy_simulator import RuntimeV2PolicySimulator  # noqa: E402
from vllm_omni.diffusion.runtime_v2.protocol import TaskKind  # noqa: E402

SETUP_MS = {2: 218.1, 4: 532.5, 6: 664.2, 8: 778.4}
MEM_MB = {2: 484.0}


def setup_ms(size: int) -> float:
    if size in SETUP_MS:
        return SETUP_MS[size]
    ks = sorted(SETUP_MS)
    if size < ks[0]:
        return SETUP_MS[ks[0]]
    if size > ks[-1]:
        return SETUP_MS[ks[-1]]
    lo = max(k for k in ks if k < size); hi = min(k for k in ks if k > size)
    return SETUP_MS[lo] + (SETUP_MS[hi] - SETUP_MS[lo]) * (size - lo) / (hi - lo)


def mem_mb(size: int) -> float:
    return MEM_MB.get(size, 535.0)


MODE = os.environ.get("BD_SETUP_MODE", "none")
# world: PyTorch default new_group semantics -- every reconfiguration opens a
# WORLD barrier at decision time; no rank starts a new task after that; each rank
# joins when its running task ends; once all joined, EVERY rank pays the cold setup
# (x BD_SETUP_MULT), then scheduling resumes. Reconfigs arriving while a barrier is
# open queue behind it (sequential new_group calls).
SETUP_MULT = float(os.environ.get("BD_SETUP_MULT", "1"))
GFC_MS = float(os.environ.get("BD_GFC_US", "60")) / 1000.0
STATS: dict = {}


class BreakdownSim(RuntimeV2PolicySimulator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._world = tuple(sorted(w.worker_rank for w in self.topology.workers))
        # groups treated as pre-created at startup (free in every mode): the world
        # group plus, if BD_FREE_ALIGNED_SP=k, the aligned size-k replicas.
        self._free = {self._world}
        k = int(os.environ.get("BD_FREE_ALIGNED_SP", "0") or 0)
        if k > 1:
            n = len(self._world)
            self._free |= {tuple(range(i * k, (i + 1) * k)) for i in range(n // k)}
        self._req_last_ranks: dict[str, tuple] = {}
        self._created: set[tuple] = set()
        self._setup_events = 0
        self._setup_total_ms = 0.0
        self._setup_rank_ms = 0.0
        self._req_setup_ms: dict[str, float] = defaultdict(float)
        self._req_setup_n: dict[str, int] = defaultdict(int)
        self._now = 0.0
        self._gate_active = False
        self._gate_release_ms = 0.0
        self._req_last_disp: dict[str, tuple] = {}
        self._world_reconfigs = 0
        self._world_wait_ms: list[float] = []      # per-reconfig: decision -> its setup starts
        self._world_barrier_ms: list[float] = []   # per-gate: decision -> all ranks joined
        self._world_stall_rank_ms = 0.0            # sum over ranks of barrier idle + setup
        self._world_setup_ms_total = 0.0
        self._policy_calls = 0
        self._policy_s = 0.0
        pol = self.policy
        for name in ("on_request_submitted", "on_tasks_runnable", "on_worker_event", "on_round_tick"):
            fn = getattr(pol, name, None)
            if fn is None:
                continue

            def wrap(fn=fn):
                def inner(*x, **y):
                    t0 = time.perf_counter()
                    try:
                        return fn(*x, **y)
                    finally:
                        self._policy_s += time.perf_counter() - t0
                        self._policy_calls += 1
                return inner
            setattr(pol, name, wrap())

    # ---- NCCL-world barrier model -------------------------------------------
    def _poll_workers(self, now_ms):
        self._now = now_ms
        return super()._poll_workers(now_ms)

    def _next_startable_task(self):
        if MODE == "world" and self._gate_active:
            if self._now + 1e-9 >= self._gate_release_ms:
                self._gate_active = False
            else:
                return None
        return super()._next_startable_task()

    def _all_ranks_idle(self):
        if MODE == "world" and self._gate_active:
            return False
        return super()._all_ranks_idle()

    def _handle_round_tick(self, now_ms):
        if self._round_mode:
            return super()._handle_round_tick(now_ms)
        return []  # gate wake-up only

    def _dispatch(self, tasks, now_ms):
        tasks = list(tasks)
        if MODE == "world":
            self._now = max(self._now, now_ms)
            for task in tasks:
                self._world_check(task, now_ms)
        return super()._dispatch(tasks, now_ms)

    def _world_check(self, task, now_ms):
        if task.kind not in (TaskKind.DIT_STEP_CHUNK, TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE):
            return
        try:
            _, ranks = self._task_resource(task)
        except ValueError:
            return
        ranks = tuple(sorted(ranks))
        prev = self._req_last_disp.get(task.request_id)
        self._req_last_disp[task.request_id] = ranks
        if len(ranks) < 2 or ranks in self._free or prev == ranks:
            return
        S = setup_ms(len(ranks)) * SETUP_MULT
        n = len(self._world)
        if not self._gate_active:
            t1 = max([now_ms] + [self.rank_exec_free_ms[r] for r in self._world])
            idle = sum(t1 - max(now_ms, self.rank_exec_free_ms[r]) for r in self._world)
            self._world_stall_rank_ms += idle
            self._world_barrier_ms.append(t1 - now_ms)
            self._world_wait_ms.append(t1 - now_ms)
            self._gate_active = True
            self._gate_release_ms = t1 + S
        else:
            self._world_wait_ms.append(max(0.0, self._gate_release_ms - now_ms))
            self._gate_release_ms += S
        self._world_stall_rank_ms += n * S
        self._world_setup_ms_total += S
        self._world_reconfigs += 1
        self._req_setup_ms[task.request_id] += S
        self._req_setup_n[task.request_id] += 1
        self._created.add(ranks)
        self._push_event(time_ms=self._gate_release_ms, kind="round_tick")

    def _task_duration_ms(self, task, group_id):
        base = super()._task_duration_ms(task, group_id)
        if MODE == "world":
            return base
        if task.kind == TaskKind.RESHARD:
            return base
        try:
            _, ranks = self._task_resource(task)
        except ValueError:
            return base
        ranks = tuple(sorted(ranks))
        if len(ranks) < 2 or ranks in self._free:
            if task.kind in (TaskKind.DIT_STEP_CHUNK, TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE):
                self._req_last_ranks[task.request_id] = ranks
            return base
        extra = 0.0
        if MODE == "uncached":
            if self._req_last_ranks.get(task.request_id) != ranks:
                extra = setup_ms(len(ranks))
        elif MODE == "cached":
            if ranks not in self._created:
                extra = setup_ms(len(ranks))
        else:  # GFC: descriptor registration whenever the request moves to a new logical group
            if self._req_last_ranks.get(task.request_id) != ranks:
                extra = GFC_MS
        self._created.add(ranks)
        self._req_last_ranks[task.request_id] = ranks
        if extra > 0:
            self._setup_events += 1
            self._setup_total_ms += extra
            self._setup_rank_ms += extra * len(ranks)
            self._req_setup_ms[task.request_id] += extra
            self._req_setup_n[task.request_id] += 1
        return base + extra

    def run(self, requests):
        t0 = time.perf_counter()
        result = super().run(requests)
        wall = time.perf_counter() - t0
        # reconfiguration accounting from the timeline (DiT steps only)
        seqs = defaultdict(list)
        for ev in result.timeline:
            if ev.kind == TaskKind.DIT_STEP_CHUNK.value:
                a = ev.args or {}
                if a.get("step_start") is not None and a.get("step_end") is not None:
                    k = max(1, int(a["step_end"]) - int(a["step_start"]))
                else:
                    k = max(1, int(a.get("chunk_steps", 1) or 1))
                seqs[ev.request_id].append((ev.start_ms, tuple(sorted(ev.ranks)), k))
        n_req = len(seqs); steps = 0; rc = 0; rc_multi = 0; with_rc = 0; true_steps = 0; per_req_rc = []
        sizes = Counter(); distinct = set()
        for rid, s in seqs.items():
            s.sort()
            r = [x[1] for x in s]
            steps += len(r)
            true_steps += sum(x[2] for x in s)
            for x in r:
                sizes[len(x)] += 1; distinct.add(x)
            c = 0
            for a, b in zip(r, r[1:]):
                if a != b:
                    c += 1
                    if len(b) >= 2:
                        rc_multi += 1
            rc += c; with_rc += bool(c); per_req_rc.append(c)
        multi = {x for x in distinct if len(x) >= 2 and x not in self._free}
        per_gpu_n = Counter(); per_gpu_mb = Counter()
        for x in multi:
            for g in x:
                per_gpu_n[g] += 1; per_gpu_mb[g] += mem_mb(len(x))
        busy_rank_ms = sum((ev.end_ms - ev.start_ms) * len(ev.ranks) for ev in result.timeline
                           if ev.kind != "task_launch")
        affected = {r: v for r, v in self._req_setup_ms.items() if v > 0}
        arr = min(result.request_arrival_ms.values(), default=0.0)
        fin = max(result.request_finish_ms.values(), default=0.0)
        dur_s = (fin - arr) / 1000.0
        STATS.update({
            "mode": MODE,
            "world": len(self._world),
            "requests": n_req, "dit_steps": steps,
            "reconfigs": rc, "reconfigs_to_multirank": rc_multi,
            "frac_steps_reconfigured": rc / max(1, steps - n_req),
            "dit_steps_true": true_steps,
            "frac_step_boundaries_reconfigured": rc / max(1, true_steps - n_req),
            "reconfigs_per_req_max": max(per_req_rc) if per_req_rc else 0,
            "requests_with_reconfig": with_rc,
            "reconfigs_per_req_mean": rc / max(1, n_req),
            "reconfigs_per_s": rc / dur_s if dur_s else None,
            "step_size_hist": dict(sorted(sizes.items())),
            "distinct_rank_sets": len(distinct),
            "distinct_nonworld_multirank_sets": len(multi),
            "max_comms_per_gpu_if_cached": max(per_gpu_n.values()) if per_gpu_n else 0,
            "max_comm_mem_gb_per_gpu_if_cached": (max(per_gpu_mb.values()) / 1024.0) if per_gpu_mb else 0.0,
            "setup_events": self._setup_events,
            "setup_total_ms": self._setup_total_ms,
            "setup_rank_ms": self._setup_rank_ms,
            "busy_rank_ms": busy_rank_ms,
            "setup_frac_of_busy": self._setup_rank_ms / busy_rank_ms if busy_rank_ms else 0.0,
            "affected_requests": len(affected),
            "setup_per_affected_req_ms": (sum(affected.values()) / len(affected)) if affected else 0.0,
            "req_setup_ms": dict(affected),
            "world_reconfigs": self._world_reconfigs,
            "world_gates": len(self._world_barrier_ms),
            "world_barrier_ms_mean": (sum(self._world_barrier_ms) / len(self._world_barrier_ms)) if self._world_barrier_ms else 0.0,
            "world_wait_ms_mean": (sum(self._world_wait_ms) / len(self._world_wait_ms)) if self._world_wait_ms else 0.0,
            "world_stall_rank_ms": self._world_stall_rank_ms,
            "world_setup_ms_total": self._world_setup_ms_total,
            "gpu_time_rank_ms": len(self._world) * (fin - arr),
            "sim_wall_s": wall,
            "policy_calls": self._policy_calls,
            "policy_total_s": self._policy_s,
            "policy_us_per_call": 1e6 * self._policy_s / max(1, self._policy_calls),
            "duration_s": dur_s,
        })
        return result


runner.RuntimeV2PolicySimulator = BreakdownSim

if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("--results-out") + 1]
    sys.argv = ["runner"] + args
    runner.main()
    extra_path = os.environ.get("BD_OUT_EXTRA", out + ".extra.json")
    Path(extra_path).write_text(json.dumps(STATS, indent=1))
    d = json.load(open(out))
    d["bd_extra"] = STATS
    Path(out).write_text(json.dumps(d))
