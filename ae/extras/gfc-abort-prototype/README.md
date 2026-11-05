# GFC dead-peer abort prototype (Sec. 5.1, "Failure handling")

**Status: prototype, not integrated into Plectra.** The paper only states that GFC
*permits* finer-grained recovery and that a prototype exists; no result in the
paper depends on it. It is included for completeness.

## What it does

GFC synchronizes overlapping groups with per-rank-pair (edge) barriers. Without the
prototype, a rank whose peer crashes spins in the barrier until the kernel watchdog
traps, which leaves a sticky CUDA error (`unspecified launch failure`) and an unusable
CUDA context; the whole worker must restart. The prototype makes the edge barrier also
poll a host-pinned *dead-peer mask*:

- `runtime.declare_dead(rank)` marks a crashed rank;
- a pending barrier on any group that contains it raises `GFCPeerFailure` instead
  of hanging (the abort is scoped to the whole group, patch 0002);
- after all survivors quiesce, `reset_edges(live_peers)` clears the edge state so
  the surviving ranks can form a new group and keep running collectives.

## Files

| file | content |
|---|---|
| `0001-feat-abort-a-barrier-wait-when-its-peer-is-declared-.patch` | dead-peer mask polled in the barrier spin; `declare_dead` / `check_abort` / `clear_abort` |
| `0002-fix-scope-the-dead-peer-abort-to-the-whole-group-and.patch` | group-scoped abort + `reset_edges` (fixes a hang in edge-scoped aborts) |
| `abort_test.py`, `run_abort_test.sh` | 3-GPU test: healthy phase, SIGKILL one rank mid-collective, check survivors raise, regroup {0,1} and verify correctness |
| `logs/run_kill9b_t1..t8` | 8 SIGKILL trials (3 GPUs); `logs/run_watchdog` the baseline without the prototype (3 s watchdog) |

## Reproduce (>= 3 GPUs with P2P)

The patches apply to GroupFree-Collective commit `bcffb02` (the version vendored in
`GroupFree-Collective/` is the earlier `150f3d2` used for the paper's experiments):

```bash
git clone https://github.com/tie-pilot-qxw/GroupFree-Collective.git /tmp/gfc-abort
cd /tmp/gfc-abort && git checkout bcffb02
git am <artifact>/ae/extras/gfc-abort-prototype/000*.patch
pip install -e .                                  # in a scratch environment
PYTHON=$(which python) bash <artifact>/ae/extras/gfc-abort-prototype/run_abort_test.sh 0,1,2 abort /tmp/abort_run external
```

Expected (as in `logs/`): every survivor prints
`"result": "GFCPeerFailure dead_ranks=[2]"` with `death_to_error_ms` of a few tens
of ms (34-65 ms in our 8 trials) and `"cuda_context_usable": true`, followed by a
correct collective on the regrouped `{0,1}`. With `watchdog` instead of `abort`
(`logs/run_watchdog`), survivors hang until the 3 s watchdog traps (error after
~5.4-5.8 s) and report `"cuda_context_usable": false`.
