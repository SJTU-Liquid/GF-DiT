# Plectra — PPoPP 2027 Artifact

Artifact for *Plectra: Scheduling Parallelism for Diffusion Transformer Serving*
(PPoPP 2027). Plectra is a policy-programmable DiT-serving runtime (`runtime_v2`) inside a
fork of vLLM-Omni 0.17.0, plus Group-Free Collectives (GFC), a communication layer that
forms GPU groups online without `new_group`.

**Badges requested: Artifact Available, Artifact Reusable** (Functional as fallback).
We do not request Results Reproduced: the paper's real-serving results were measured on a
4×H20 and an 8×A100 server that we cannot make available to reviewers. Instead, every
number in the paper can be checked **on a CPU**:

- **[S] Simulator.** Claims that come from the simulator (Fig. 10 simulated bars,
  Sec. 6.6 scalability, Sec. 6.9 TetriServe) are re-run from scratch. Python standard
  library only.
- **[L] Raw logs.** Every figure and table measured on GPUs (Figs. 3, 6–10, Tables 1–2,
  Secs. 6.3–6.5) is regenerated from the raw benchmark outputs we shipped, using the
  paper's plotting scripts, and compared with the data behind the paper (must be
  identical, up to line endings).
- **[G] GPU scripts (optional).** Scripts that redo each measurement on your own GPUs
  (Sec. 6 below). Absolute numbers depend on the GPU model; see "Interpreting results".

## Contents

| path | content |
|---|---|
| `vllm_omni/diffusion/runtime_v2/` | Plectra runtime: task-graph scheduler, migration engine, policies (`policies/`), model adapters (`adapters/`), cost model |
| `vllm_omni/distributed/collective_runtime.py`, `GroupFree-Collective/` | GFC backend and the GFC library (vendored, commit `150f3d2`) |
| `benchmarks/diffusion/` | serving benchmark client, workload generator, simulator (`runtime_v2_policy_simulator.py`, `runtime_v2_mixed_priority_runner.py`) and simulator policies (`policies/`, incl. the TetriServe port) |
| `benchmarks/distributed/` | GFC vs. NCCL group-setup and collective micro-benchmarks |
| `scripts/` | launch scripts; `sweep_policies.sh` runs one server per policy |
| `ae/sim/` | [S] `run_claims.py`, the H20 Wan2.2 cost model and the paper's workloads |
| `ae/figures/` | [L] raw logs (`logs/`), the paper's plotting scripts (`scripts/`), expected data (`expected/`), `run_all.sh` |
| `ae/gpu/` | [G] scripts `00`–`07` |
| `ae/docker/` | `Dockerfile.cpu` (tiers S, L) and `Dockerfile.gpu` (tier G) |
| `ae/extras/gfc-abort-prototype/` | prototype for Sec. 5.1 "Failure handling" (not integrated) |
| `ae/paper/plectra-ppopp27.pdf` | the paper (camera-ready draft) |

## 1. Getting started (≈ 10 min, CPU only)

Either use Docker (from the artifact root):

```bash
docker build -f ae/docker/Dockerfile.cpu -t plectra-ae:cpu .
docker run --rm -it plectra-ae:cpu        # opens a shell in /workspace/plectra
```

or any Linux/macOS machine with Python ≥ 3.10; only the figures need two packages:

```bash
python3 -m venv /tmp/plectra-ae && . /tmp/plectra-ae/bin/activate
pip install matplotlib numpy              # tested: matplotlib 3.10.9, numpy 2.2.6
```

Then run the two checks:

```bash
python3 ae/sim/run_claims.py --quick      # < 1 min: claims S1, S2 (9 checks)
bash ae/figures/run_all.sh                # ~ 1 min: all figures and tables from the logs
```

Expected output: `run_claims.py --quick` ends with a table whose `check` column reads
`PASS` in all 9 rows (two further `info` rows are explained in Sec. 4), and `run_all.sh`
ends with

```
IDENTICAL gfc_collective_latency.csv
...
IDENTICAL workload_trace_summary.csv

ALL FIGURE/TABLE DATA MATCH THE PAPER
```

Plots are written to `ae/figures/figs/*.pdf` (compare them visually with the paper), the
simulator results to `ae/sim/out/` (summary in `ae/sim/out/summary.md`).

## 2. Claims and how to check them

Run `python3 ae/sim/run_claims.py` (no `--quick`) to add S3 and S4 (30 simulations). It
takes about **15–20 min with ≥ 6 cores**: each of the six 64-GPU simulations takes up to
~15 min (16 min on our 20-core machine; with fewer cores it takes proportionally
longer). `--jobs N` sets the parallelism; finished simulations are cached in
`ae/sim/out/`, and `--fresh` discards them. Expected: every row `PASS` or `info`, and the
last line `overall: PASS`.

| paper | claim | tier: command | expected |
|---|---|---|---|
| Abstract, Sec. 6.3 | EDF vs. Legacy: up to 6.01× throughput, −95.2% mean latency, −89.6% SLO violations | L: `run_all.sh` → `figs/headline_numbers.md` | 6.01× (A100 Qwen fg), 95.2% (A100 Qwen short), 89.6% (A100 Wan short) |
| Fig. 3 | the best parallelism depends on stage, request shape and load | L: `run_all.sh` → `figs/motivation.pdf` | identical to `expected/motivation_data.csv` |
| Table 1, Sec. 4 | NCCL subgroup setup: up to 778 ms first collective, ~0.5 GB/GPU per group; GFC registration ~60 µs, 0 MB | L: `run_all.sh` → `figs/table1_group_setup.md`; G: `ae/gpu/02_group_setup.sh` | 777.7 ms at size 8; GFC median 60.6 µs |
| Fig. 6, Sec. 6.3 | EDF has the highest SLO attainment in every setting (H20/A100 × Wan2.2/Qwen-Image × short/fg) | L: `run_all.sh` → `figs/main_result_combined_metric_grid.pdf`; G: `EXP=main ae/gpu/07_policy_runs.sh` | identical to `expected/main_result_combined.csv` |
| Fig. 7 | the two workloads | L: `run_all.sh` → `figs/workload_trace_paper.pdf` | identical to `expected/workload_trace_summary.csv` |
| Table 2, Sec. 6.4 | each mechanism adds SLO attainment: 27.9 → 55.7 → 57.4 → 70.5 → 83.6 → 85.2%; at 40% load EDF mean 50.5 s vs. EDF-SP1 81.3 s and FCFS-SP4 70.1 s | L: `run_all.sh` → `figs/table2_breakdown.md`; G: `EXP=breakdown ae/gpu/07_policy_runs.sh` | identical to `expected/table2_breakdown.md` |
| Sec. 6.5 | runtime overhead: Plectra FCFS-SP4 vs. Legacy SP4 | L: `run_all.sh` → `figs/legacy_vs_fcfs_sp4.pdf` | identical to `expected/legacy_vs_fcfs_sp4.csv` |
| Fig. 8, Sec. 6.6 | GFC all2all / all_gather latency vs. NCCL, 1 KB–64 MB | L: `run_all.sh` → `figs/gfc_collective_latency.pdf`; G: `ae/gpu/03_gfc_collectives.sh` | identical to `expected/gfc_collective_latency.csv` |
| Sec. 6.6 "Scalability" | 8 → 64 GPUs (simulated): reconfigurations 0.09 → 0.96 /s; world-synchronized `new_group` raises mean latency over GFC by 14% / 30%; caching all groups needs 70 communicators (~36 GB) per GPU | S: `run_claims.py` → claim S3 | `PASS` (mean of seeds 42–44) |
| Fig. 9, Sec. 6.7 | EDF wins at low and moderate load, SRTF-SP1 under sustained overload | L: `run_all.sh` → `figs/scaling_arrival_rate.pdf`; G: `EXP=arrival ae/gpu/07_policy_runs.sh` | identical to `expected/scaling_arrival_rate.csv` |
| Fig. 10, Sec. 6.8 | simulator matches real serving (H20 Wan2.2 fg) | real bars, L: `run_all.sh` → `figs/sim_vs_real_h20_fg.pdf`; simulated bars, S: claim S2 | `PASS`: EDF 89.2, SRTF-SP1 81.5, SRTF-SPmax 55.4, FCFS-SP1 49.2, FCFS-SP4 21.5 |
| Sec. 6.9 | TetriServe's policy 59.0% / 52.3% vs. EDF 91.8% / 89.2% (short / fg) | S: claim S1 | `PASS` |
| Sec. 4 | GFC is correct for overlapping groups | G: `ae/gpu/01_gfc_correctness.sh` | all tests pass |
| Sec. 5.1 | GFC permits peer-failure abort (prototype) | G: `ae/extras/gfc-abort-prototype/README.md` | `GFCPeerFailure` in tens of ms, CUDA context usable |

Figures 1, 2, 4 and 5 are diagrams.

## 3. Expected runtimes

| step | hardware | time |
|---|---|---|
| `run_claims.py --quick` | any CPU | < 1 min |
| `run_all.sh` | any CPU | ~1 min |
| `run_claims.py` (full) | ≥ 6 cores | 15–20 min |
| `ae/gpu/00`–`04` | ≥ 2 GPUs (4 for Fig. 8) | ~1 h total + model download |
| `ae/gpu/05` + `06` | the GPUs to evaluate | 1–2 h per model |
| `ae/gpu/07` | 4 GPUs | 3–6 h per experiment |

## 4. Interpreting results

**Simulator claims are deterministic.** S1, S2 and S3 reproduce the paper's numbers
exactly (S3 as the mean over three workload seeds, as in the paper).

**SRTF tie-breaking (S2 `info` rows, S4, Table 2 vs. Fig. 6).** When several requests
have equal remaining work, SRTF/SJF must break the tie. The Fig. 6 runs (May–June 2026)
broke ties by the random UUID request id; the released code (and the Table 2 runs,
October 2026) breaks ties by arrival order. This changes only SRTF/SJF, and only by
reordering equal-work requests; FCFS and EDF are unaffected. Consequently:

- S2 reproduces Fig. 10's SRTF bars with `ae/sim/ae_policies.py:make_srtf_legacy_tiebreak`,
  which restores the old tie-break on request ids randomized as in real serving. The
  `info` rows show the released policy on the same trace (SRTF-SP1 80.0%, SRTF-SPmax
  49.2%).
- S4 shows the spread on the short workload: the released policy versus the old
  tie-break under six random id seeds.
- On real GPUs, expect SRTF-SP1 on the short workload near Table 2 (70.5%) rather than
  Fig. 6 (82.0%). The paper notes this in Sec. 6.4.

**Code version.** This artifact is the release snapshot of our development tree plus the
SJF wiring used for Table 2. The Fig. 6/8/9 runs were made from earlier snapshots of the
same tree on the H20 and A100 servers; apart from the SRTF tie-break above, the
runtime, the policies and GFC are the same, but the release is not byte-identical to
those snapshots. The raw logs in `ae/figures/logs/` are the original outputs.

**Real-GPU reruns (tier G).** SLO deadlines and arrival rates are calibrated to the
platform's measured costs, so the paper's H20 workloads are meaningful only on 4×H20.
On other GPUs, first profile a cost model (`05_cost_model.sh`) and generate workloads
from it (`06_workloads.sh`); then compare *relative* results: EDF should attain the
highest SLO attainment, and Legacy the lowest. Repeated real runs differ by a few
percentage points (EDF on H20 Wan2.2 short: 86.9% in Fig. 6, 85.2% in Table 2).

**TetriServe** is evaluated by porting its scheduling algorithm to our policy interface
(`benchmarks/diffusion/policies/tetriserve.py`) and running it in our simulator under
identical traces and profiled costs (Sec. 6.2); it is not the authors' system.

**Supported and unsupported claims.** All quantitative claims listed in Sec. 2 can be
checked on a CPU (from the simulator or from the raw logs). Without H20/A100 servers,
reviewers cannot re-measure the real-serving numbers themselves; tier G provides the
scripts. The failure-handling prototype (Sec. 5.1) is a prototype, not part of Plectra's
evaluated system.

## 5. Running on GPUs (tier G, optional)

Requirements: NVIDIA GPUs with P2P/NVLink (GFC needs ≥ 2), ≥ 48 GB memory each for the
models, a driver that supports CUDA 12.9, and disk space for the Wan2.2-TI2V-5B and
Qwen-Image weights (downloaded from Hugging Face on first start).

```bash
docker build -f ae/docker/Dockerfile.gpu -t plectra-ae:gpu .
docker run --gpus all --ipc=host --shm-size 16g -it \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface plectra-ae:gpu
```

The image is `nvcr.io/nvidia/pytorch:25.06-py3` with a venv at `/opt/venv` holding the
paper's stack (PyTorch 2.10.0+cu129, vLLM 0.17.0, NCCL 2.27.5, Triton 3.6.0, this repo,
GFC); `ae/docker/gpu-resolved-packages.txt` lists every resolved package version. Without
Docker, replicate the three `uv pip install` lines of `Dockerfile.gpu` in a Python 3.12
venv.

All scripts read `GPUS` (default `0,1,2,3`), `NUM_GPUS`, `PORT` (8098), `RESULTS`
(`ae/gpu/results`), and `HF_HUB_OFFLINE` (set to 1 once the weights are cached).

| script | paper | what it does |
|---|---|---|
| `00_env_check.sh` | — | prints GPUs and versions, checks that Plectra and GFC import |
| `01_gfc_correctness.sh` | Sec. 4 | GFC test suite (overlapping groups, barriers, collectives) |
| `02_group_setup.sh` | Table 1 | NCCL `new_group` + first collective vs. GFC registration; writes `table1_group_setup.md` |
| `03_gfc_collectives.sh` | Fig. 8 | GFC vs. NCCL all2all / all_gather latency, CUDA graphs |
| `04_serve_smoke.sh` | — | starts Wan2.2 with Plectra (EDF best-fit + GFC) and generates one video |
| `05_cost_model.sh` | Sec. 5.4 | profiles the per-stage cost model (`MODEL=wan\|qwen`) |
| `06_workloads.sh` | Sec. 6.1 | generates the short and foreground-burst workloads from a cost model |
| `07_policy_runs.sh` | Fig. 6, Table 2, Fig. 9 | `EXP=main\|breakdown\|arrival` real serving runs |
| `summarize_runs.py` | — | one row per run: SLO %, mean / p95 latency, throughput |

`07_policy_runs.sh` covers Wan2.2; for Qwen-Image use `MODEL=qwen` in `05` and pass the
resulting cost model and workloads to `run_plectra` / `run_legacy` in `ae/gpu/common.sh`.

## 6. Reusing Plectra

**Serve with Plectra.** Plectra is opt-in on top of vLLM-Omni:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --num-gpus 4 \
  --enable-runtime-v2 --runtime-v2-scheduler-policy edf_best_fit \
  --runtime-v2-collective-backend gfc --runtime-v2-cost-model-dir <cost model dir>
```

See `ae/gpu/04_serve_smoke.sh` for a complete example and the top-level `README.md` for
all flags.

**Add a scheduling policy.** A policy implements `SchedulerPolicy`
(`vllm_omni/diffusion/runtime_v2/interfaces.py`): it receives request plans and worker
events and returns the tasks to launch, each bound to an execution group. Start from
`policies/fcfs.py` or `policies/srtf.py`; `policies/sjf.py` (non-preemptive SJF, ~90 lines
on top of SRTF) is a complete example. Registering a policy takes three small changes,
exactly those made for SJF: the factory in `runtime_v2/runner.py`, the allowed names in
`vllm_omni/diffusion/data.py`, and the CLI choices in `vllm_omni/entrypoints/cli/serve.py`.

**Evaluate a policy offline.** The simulator drives the same policy interface with a
profiled cost model, so a policy can be tried without GPUs:

```bash
python3 ae/sim/sim_shim.py benchmarks.diffusion.runtime_v2_mixed_priority_runner \
  --workload-json ae/sim/workloads/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json \
  --cost-model-dir ae/sim/cost-model/wan22-ti2v-5b-h20 --world-size 4 \
  --policy benchmarks.diffusion.policies.edf_best_fit:make_edf_best_fit \
  --results-out /tmp/edf.json
```

This takes a few seconds and prints per-class results (overall SLO attainment 91.8%, the
EDF number of Sec. 6.9). `--policy` takes any `module:factory`, and several at once;
`ae/sim/ae_policies.py` shows a policy defined outside the repository. `sim_shim.py` lets the simulator run without vLLM installed.

**Add a model.** A model adapter (`RuntimeV2Adapter`, see `runtime_v2/adapters/wan.py` and
`qwen_image.py`) splits a pipeline into text-encode, denoise-chunk and decode tasks and
declares the artifact layouts used for migration; register it with
`register_runtime_v2_adapter` (`runtime_v2/registry.py`), then profile its cost model with `ae/gpu/05_cost_model.sh`.

**New hardware or workloads.** `05_cost_model.sh` and `06_workloads.sh` produce a cost
model and calibrated workloads for any GPU; `06_workloads.sh` documents the generator
settings (classes, α, duration, target load).

**Use GFC on its own.** `GroupFree-Collective/` is a standalone library
(`pip install -e GroupFree-Collective`); see its README. It is also available at
<https://github.com/tie-pilot-qxw/GroupFree-Collective>.

## License

Apache License 2.0 (`LICENSE`), as vLLM-Omni.
