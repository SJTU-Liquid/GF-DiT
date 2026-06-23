"""Construct a 'lazy-B' workload: two request types whose per-step times have an
awkward (near-coprime) ratio so NO fixed round length aligns both -> round-based
B always pays tail-idle (small tau) or admission latency (large tau), while the
continuous A packs them with neither. Sweeps tau for B and compares to A.

SP is pinned to 1 (EDF_GREEDY_SP_SIZES=1) to isolate the round-packing waste
from sequence-parallelism effects -- this is the pure fixed-interval problem.
"""
import json, os, sys, random
sys.path.insert(0, ".")
os.environ["EDF_GREEDY_SP_SIZES"] = "1"  # SP1-only: isolate round waste
from benchmarks.diffusion.runtime_v2_policy_simulator import (
    CostModelStore, RuntimeV2PolicySimulator, build_policy_factory, _request_from_mapping, SimRequest,
)
from benchmarks.diffusion.runtime_v2_mixed_priority_runner import _build_topology
from vllm_omni.diffusion.runtime_v2.protocol import ParallelSpec

COST = "cost-model/wan22-ti2v-5b-fullrange"
WORLD = int(os.environ.get("GEN_WORLD", "2"))      # GPU count
NUM_STEPS = 50
ALPHA = float(os.environ.get("GEN_ALPHA", "3.0"))  # deadline = arrival + ALPHA * T_single(SP1)
LOAD = float(os.environ.get("GEN_LOAD", "0.55"))   # offered load on the pool
N_REQ = int(os.environ.get("GEN_NREQ", "60"))
SEED = 42
# SP sizes the dynamic policies may synthesize. "1" => SP1-only (isolate round
# waste). "" => leave unset so policies use the full {1,2,...,W} ladder.
_SP = os.environ.get("GEN_SP", "1")
if _SP:
    os.environ["EDF_GREEDY_SP_SIZES"] = _SP
else:
    os.environ.pop("EDF_GREEDY_SP_SIZES", None)
# (label, H, W, frames)
TYPE_X = ("X", 480, 832, 33)   # fast step
TYPE_Y = ("Y", 480, 832, 81)   # slow step

cm = CostModelStore.load_dir(COST)

def step_ms(h, w, f):
    r = SimRequest(request_id="p", arrival_ms=0, height=h, width=w, num_frames=f,
                   num_steps=NUM_STEPS, denoise_chunk_size=1, text_seq_len=512, priority=0, group_id=None)
    m = cm.for_parallel_spec(ParallelSpec(tp=1, sp=1, cfg=1))
    return m.estimate_ms("dit_step_chunk", float(r.latent_seq_len)), r.latent_seq_len

sx, _ = step_ms(*TYPE_X[1:]); sy, _ = step_ms(*TYPE_Y[1:])
Tx = sx * NUM_STEPS; Ty = sy * NUM_STEPS
print(f"step_X={sx:.0f}ms step_Y={sy:.0f}ms  ratio={sy/sx:.3f}  T_X={Tx/1000:.0f}s T_Y={Ty/1000:.0f}s")

rng = random.Random(SEED)
mean_service = (Tx + Ty) / 2.0
lam = WORLD * LOAD / mean_service          # req per ms
items = []
t = 0.0
for i in range(N_REQ):
    t += rng.expovariate(lam)
    is_x = (i % 2 == 0)
    lbl, h, w, f = TYPE_X if is_x else TYPE_Y
    Tsingle = Tx if is_x else Ty
    dl = t + ALPHA * Tsingle
    items.append({
        "request_id": f"{lbl}_{i:04d}", "arrival_ms": round(t, 1),
        "height": h, "width": w, "num_frames": f, "num_steps": NUM_STEPS,
        "denoise_chunk_size": 1, "text_seq_len": 512,
        "request_class": lbl, "priority": 0,
        "deadline_ms": round(dl, 1), "client_deadline_ms": round(dl + 5000, 1),
    })
data = {
    "description": (
        "lazy-B demo: two request types with near-coprime per-step times (no "
        "fixed round length aligns both). SP1-only to isolate round-packing "
        "waste. Continuous A beats round-based B at every round length: small "
        "tau pays tail-idle + admission latency, large tau (alignment LCM) pays "
        "admission latency."
    ),
    "metadata": {
        "generator": "gen_lazyb_coprime_workload.py",
        "cost_model": COST, "world_size": WORLD, "sp_sizes": [1],
        "num_steps": NUM_STEPS, "alpha": ALPHA, "target_load": LOAD,
        "seed": SEED, "n_requests": N_REQ,
        "type_X": {"shape": list(TYPE_X[1:]), "step_ms": round(sx, 1), "T_single_ms": round(Tx, 1)},
        "type_Y": {"shape": list(TYPE_Y[1:]), "step_ms": round(sy, 1), "T_single_ms": round(Ty, 1)},
        "step_ratio": round(sy / sx, 4),
        "deadline_model": "scheduler = arrival + alpha*T_single(SP1); client = +5000ms",
    },
    "requests": items,
}
_spmode = (_SP.replace(",", "") if _SP else "full")
out_path = "out/stress/lazyb_coprime_sp%s_w%d_a%.1f_load%.2f_seed%d.json" % (
    _spmode, WORLD, ALPHA, LOAD, SEED)
json.dump(data, open(out_path, "w"), indent=1)
print(f"wrote {out_path}: {N_REQ} reqs, arrival span {items[-1]['arrival_ms']/1000:.0f}s, load~{LOAD}")

reqs = [_request_from_mapping(it) for it in items]

def run(spec, round_ms):
    topo = _build_topology(None, None, "minimal", world_size=WORLD)
    sim = RuntimeV2PolicySimulator(topology=topo, policy_factory=build_policy_factory(spec),
        cost_models=cm, reshard_ms=3.0, round_ms=round_ms)
    res = sim.run(reqs)
    met = tot = 0; mx = 0
    for it in items:
        f = res.request_finish_ms.get(str(it["request_id"])); tot += 1
        if f is not None: mx = max(mx, f)
        if f is not None and f <= it["client_deadline_ms"]: met += 1
    return 100.0 * met / tot, res.global_idle_frac * 100, mx / 1000

print(f"\n{'policy':<18}{'SAR':>7}{'idle%':>8}{'makespan':>10}")
sar, idle, mk = run("benchmarks.diffusion.policies:make_tetriserve_continuous", 0.0)
print(f"{'A continuous':<18}{sar:>6.1f}%{idle:>7.1f}%{mk:>9.0f}s")
for rms in (3000, 6000, 10000, 15000, 24000, 40000):
    sar, idle, mk = run("benchmarks.diffusion.policies:make_tetriserve_round", float(rms))
    print(f"{'B round '+str(rms//1000)+'s':<18}{sar:>6.1f}%{idle:>7.1f}%{mk:>9.0f}s")
