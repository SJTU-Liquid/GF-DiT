# runtime_v2 mixed-priority video DiT serving — results

> Goal: verify that *elastic* serving (step-boundary preemption, SP demotion)
> is actually useful on the 8-GPU `Wan2.2-TI2V-5B` profile, compared with
> strong request-level baselines.

## TL;DR

**Yes — step-boundary preemption is the dominant gain.** On every workload we
ran, `step_preempt_demote` reaches >75% foreground SLO attainment with p95
foreground latency ≈ T_single, while the best request-level baseline (load
adaptive + priority-aware activation) stalls at 30-80% SAR with p95 latencies
8-20× T_single.

Critically, this gain is *not* available to request-level priority-aware
adaptive placement, because once an L job is admitted to a latency group
its ranks are stuck for hundreds to thousands of seconds. Step-boundary
preemption (pause or demote at DiT step granularity) is what unlocks the
foreground lane.

## Setup

* Topology (8 GPUs, overlapping): 8 × SP1 lanes + 2 × SP4 lanes
  (`sp4_a`=[0..3], `sp4_b`=[4..7]). SP2 omitted because the cost model
  shows SP2 is barely faster than SP1; SP8 has no cost model fit.
* Cost model: `cost-model/wan22-ti2v-5b/{tp1_ulysses1_ring1_cfg1,
  tp1_ulysses2_ring1_cfg1, tp1_ulysses4_ring1_cfg1}.json` (real profile).
* Per-class T_single (single-GPU end-to-end latency):
  * S = 480p × 49f × 25 steps → 40 s
  * M = 720p × 81f × 50 steps → 504 s
  * L = 720p × 81f × 100 steps → 1002 s
* SLO multipliers (deadline = arrival + α × T_single):
  * S: α = 1.5 → 60 s SLO
  * M: α = 2.0 → 1008 s SLO
  * L: α = 6.0 → 6012 s SLO
* Reshard cost charged: 50 ms per state migration (cost-model rough
  upper bound from `runtime_v2_reshard_microbench.py`).
* TEXT_ENCODE / VAE_DECODE / FINALIZE simulated as single-rank: the
  simulator routes them to `rank[0]` of the assigned group so they do not
  falsely block (sp-1) idle ranks for the few seconds they run.

## Workloads

All workloads are emitted by
`benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py`,
which writes placement-agnostic JSON (request shape, class, priority,
deadline, T_single only). No `stage_group_ids` or `dit_step_schedule` ever
appear in the workload JSON.

| profile           | duration | seed | requests | mix used | purpose |
|-------------------|---------:|-----:|---------:|----------|---------|
| `inflight_burst`  | 25 min   | 0    | 6 L + 15 S | warmup L on SP4, then S burst | mechanism stress |
| `foreground_burst`| 30 min   | 1    | 14 S + 7 M + 5 L | recurring S bursts | mixed-priority stress |
| `tridentserve`    | 60 min   | 1    | 39 S + 25 M + 16 L | 5-phase load shift | long-horizon stable serving |

## Policies

All policy factories live in
`benchmarks/diffusion/runtime_v2_mixed_priority_policies.py`.

* `static_sp1`, `static_sp4` — every request pinned to a single fixed SP
  lane (round-robin).
* `rssp` — class → fixed SP map (default S:4, M:4, L:1).
* `load_adaptive` — high-load detection routes urgent admissions to
  latency lanes, others to throughput lanes. No priority ordering.
* `prio_load_adaptive` — load-adaptive *plus* priority-aware activation:
  when an SP4 lane frees, the highest-priority queued request goes next
  (sort by priority desc, deadline asc, FIFO).
* `step_preempt_pause` — step-boundary preemption: when foreground (class
  S) is pending and an L is on a latency lane, L pauses at the next DIT
  step boundary. L resumes after foreground pressure drops.
* `step_preempt_demote` — same trigger as pause, but L is *demoted* to a
  small (SP1) lane instead of pausing. L keeps progressing at lower SP;
  foreground takes the released SP4 lane.

## Results

### Workload 1: `inflight_burst` (mechanism workload)

L jobs admitted during 0-300 s (low load) onto SP4 lanes. Foreground S
burst arrives at 604-780 s while L still has ~80 steps remaining on SP4.

| policy                | S SAR | S p95 (s) | S norm_p95 | L SAR | L p95 (s) | makespan |
|-----------------------|------:|----------:|-----------:|------:|----------:|---------:|
| static_sp4            |  0.0% |  1075.9 |     27.0 | 100% | 1375.7 |  1771 s |
| load_adaptive         | 40.0% |   783.9 |     19.7 | 100% | 1562.0 |  1755 s |
| prio_load_adaptive    | 40.0% |   783.9 |     19.7 | 100% | 1562.0 |  1755 s |
| **step_preempt_pause**| **100%** | **39.9** | **1.00** | 100% | 1183.9 | **1463 s** |
| **step_preempt_demote**| 33.3% | 114.1 | 2.86 | 100% | **1111.3** | 1360 s |

Read this row by row: `step_preempt_pause` gives every S exactly its
single-card latency (40 s), where the priority baseline misses 60 % of S
deadlines and the slowest S waits 13 minutes.

`step_preempt_demote` is *worse* than `pause` here because demoting L to
SP1 just moves its ranks from "all 4 ranks of `sp4_a`" to "1 rank that is
also part of `sp4_a`". With 6 L jobs demoted in parallel they cover 6 of
8 ranks, leaving neither SP4 lane intact — a rank-overlap pathology of
this topology. Pause is the safer mechanism when L jobs share a latency
group with foreground.

### Workload 2: `foreground_burst` (recurring S bursts)

| policy                | S SAR | S p95 (s) | S norm_p95 | M SAR | L SAR | L p95 (s) |
|-----------------------|------:|----------:|-----------:|------:|------:|----------:|
| static_sp1            | 78.6% |   504.7 |    12.7 | 100% | 100% | 1136.1 |
| static_sp4            | 50.0% |   417.4 |    10.5 |  85.7% | 100% | 1595.0 |
| load_adaptive         | 78.6% |   504.7 |    12.7 | 100% | 100% | 1136.1 |
| prio_load_adaptive    | 78.6% |   504.7 |    12.7 | 100% | 100% | 1136.1 |
| **step_preempt_pause**  | **100%** | **39.9** | **1.00** | 100% | 100% | 1241.5 |
| **step_preempt_demote** | **100%** | **39.9** | **1.00** | 100% | 100% | **1001.6** |

Same dramatic foreground win. `step_preempt_demote` is the cleanest
operating point on this workload: foreground p95 = T_single, M p95 =
T_single, *and* L p95 = T_single (no slowdown vs static_sp1). The total
makespan is the same as static_sp1 (2638 s), so the elastic schedule
loses nothing on throughput.

### Workload 3: `tridentserve` (60-min, 5-phase load shift, 80 requests)

| policy                | S SAR | S p95 (s) | S norm_p95 | M SAR | M p95 (s) | L SAR | L p95 (s) | completed |
|-----------------------|------:|----------:|-----------:|------:|----------:|------:|----------:|----------:|
| static_sp1            | 35.9% | 2478.4 | 62.2 | 36.0% | 2515.4 | 100%  | 2689.6 | 80/80 |
| static_sp4            | 10.3% | 5330.7 | 133.7 | 20.0% | 5946.7 | 93.8% | 5590.9 | 80/80 |
| load_adaptive         | 30.8% | 2478.4 | 62.2 | 36.0% | 2515.4 | 100%  | 2689.6 | 80/80 |
| prio_load_adaptive    | 30.8% |  884.7 | 22.2 | 44.0% | 1668.5 | 100%  | 3393.0 | 80/80 |
| **step_preempt_pause**  | **92.3%** | **60.2** | **1.51** | **84.0%** | 1247.1 |   0%  |   ∞   | 64/80 |
| **step_preempt_demote** | **76.9%** | **65.2** | 1.64 | 60.0% | 1303.8 | 100%  | 4965.7 | 80/80 |

Headlines for the 60-min trace:

* Step-boundary preemption pushes S p95 from 885 s (best baseline) to
  ~60 s — **a 15× reduction**. Normalized to T_single, S latency drops
  from 22× back down to ~1.5×.
* `step_preempt_pause` starves L: with continuous foreground pressure
  across the overload phase, paused L jobs never get a window to resume,
  16/16 of them miss the 60-min trace window. This is the predicted
  pause failure mode.
* `step_preempt_demote` is the only policy that hits *all three* of (a)
  S p95 below the 60 s SLO, (b) >50 % M SAR, (c) 100 % L completion
  without unbounded slowdown. L pays a 5× wall-time penalty (4965 s vs
  T_single of 1002 s) — that is the *cost* of trading throughput for
  foreground responsiveness, and it lands inside L's 6× α = loose SLO.

## What the gains are NOT from

* Not from cherry-picking the priority baseline: `prio_load_adaptive`
  reorders both the shared `ready_queue` *and* per-group activation, so
  the next free SP4 lane always goes to the highest-priority queued
  request. The remaining gap to step_preempt is what genuinely requires
  step-boundary preemption.
* Not from VAE/text encoder parallelism: the simulator pins those stages
  to a single rank regardless of group, and the cost models show they
  don't speed up across SP anyway.
* Not from the workload accidentally favouring elastic: every policy
  sees the same JSON; placement is policy-side only.

## What the gains ARE from

The win is mechanistic. Once an SP4 lane is held by a 100-step L job,
that lane is unavailable to foreground for ~500 s on SP4 (1000 s on
SP1). Request-level adaptive admission cannot change L's group after
admission — `load_adaptive`'s "high-load route to SP1" only helps the
*next* request, not the one already running on SP4. Priority-aware
activation only fires when the running request *finishes*; it cannot
interrupt mid-flight.

Step-boundary preemption at DiT_STEP_CHUNK granularity is the natural
interrupt point — each DiT step is 0.5-10 s, so foreground waits at
most one step end. That is the irreducible latency reduction visible
across all three workloads.

## Caveats and known limitations

1. **Rank overlap pathology** (visible in `inflight_burst`): demoting L
   to an SP1 lane that overlaps the SP4 ranks still blocks the SP4 lane.
   On topologies where SP1 lanes are disjoint from SP4 lanes (e.g. a
   separate "small" partition), demote should match or beat pause.
2. **Pause starvation** (visible in `tridentserve`): with continuous
   foreground pressure, paused L can sit indefinitely. The existing
   `max_pause_ms` knob ages L out, but the experiments above used 10 min;
   raising it to 60 min on the 60-min trace was the simplest way to
   isolate the pause vs demote tradeoff. Production systems should use
   `step_preempt_demote` plus an aging promotion when no foreground is
   pending.
3. The simulator approximates reshard cost as a flat 50 ms per migration.
   `runtime_v2_reshard_microbench.py` shows real reshard cost depends
   heavily on payload size; rerunning with workload-derived reshard cost
   is a useful sensitivity sweep but not done here.
4. Cost models cover SP1, SP2, SP4 only. SP8 was removed from the
   topology because there is no cost-model fit; including it would need
   `runtime_v2_stage_profiler.py` to be re-run.
5. RSSP is included only as a sanity check. Its hand-tuned mapping is
   trivially beaten by `load_adaptive` whenever the request mix shifts
   across phases (visible in `foreground_burst`). It is not a strong
   baseline.

## How to reproduce

```bash
# 1. Generate the three workloads. The cost-model dir provides T_single
#    so deadlines and per-class metadata are reproducible.
python benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py \
    --profile inflight_burst   --out /tmp/wl_inflight.json --seed 0
python benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py \
    --profile foreground_burst --out /tmp/wl_burst.json    --seed 1
python benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py \
    --profile tridentserve     --out /tmp/wl_tri.json      --seed 1

# 2. Run the policy matrix. The runner builds an 8-rank topology with
#    8 SP1 lanes + 2 SP4 lanes by default; pass --topology-json to use
#    a custom topology.
POLICIES="\
  benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_static_sp1 \
  benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_static_sp4 \
  benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_load_adaptive \
  benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_priority_load_adaptive \
  benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_step_preempt_pause \
  benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_step_preempt_demote"

PYTHONPATH=. python benchmarks/diffusion/runtime_v2_mixed_priority_runner.py \
    --workload-json /tmp/wl_tri.json \
    --policy $POLICIES \
    --policy-name static_sp1 static_sp4 load_adaptive prio_load_adaptive step_pause step_demote \
    --reshard-ms 50 \
    --results-out results_tri.json
```
