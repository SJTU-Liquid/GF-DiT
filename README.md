<div align="center">

# GF-DiT: Scheduling Parallelism for Diffusion Transformer Serving

**A policy-programmable runtime for *elastic* Diffusion-Transformer serving — built on [vLLM-Omni](https://github.com/vllm-project/vllm-omni).**

[**Paper** (arXiv:2606.13501)](https://arxiv.org/abs/2606.13501) ·
[**Group-Free Collectives** (`gfc`)](https://github.com/SJTU-Liquid/group-free-collectives) ·
[**vLLM-Omni**](https://github.com/vllm-project/vllm-omni)

</div>

---

> This repository is the research artifact for **GF-DiT**. GF-DiT is implemented as a
> task-centric elastic runtime (`runtime_v2`) inside a fork of vLLM-Omni, plus a
> group-free collective backend. It is **opt-in** — with `--enable-runtime-v2` omitted the
> server behaves exactly like stock vLLM-Omni.

## What is GF-DiT

Existing DiT servers pin a request's parallel configuration (TP/SP/CFG degrees, rank set)
at admission and hold it fixed for the request's entire denoising trajectory. But DiT
workloads are heterogeneous across requests, across the stages of one request, and across
changing system load — so any single static choice wastes GPUs and degrades service
quality.

GF-DiT treats **GPU parallelism as a first-class schedulable resource** and adapts the
parallelism of *running* requests to workload demand and service objectives, via three
ideas:

- **An asynchronous execution abstraction** that decomposes each request into independently
  schedulable *trajectory tasks* and enables online GPU reallocation.
- **Group-free collectives (GFC)** — a lightweight communication layer that forms and
  reconfigures arbitrary execution groups online at microsecond cost (no
  `new_group`/`destroy_process_group`).
- **Policy-programmable scheduling** — pluggable policies decide task order and execution
  layout per objective (throughput, latency, SLO).

Across Qwen-Image / WAN on H20 / H100 / A100, GF-DiT delivers up to **6.01× throughput**,
up to **95%** lower mean latency, and up to **90%** fewer SLO violations, while cutting
communication-group setup from **778 ms to ~60 µs**. See the
[paper](https://arxiv.org/abs/2606.13501) for the full methodology and results.

## Why elastic parallelism

No single static parallel degree is right for all requests — the optimal sequence-parallel
(SP) degree flips with sequence length. Per-step DiT-chunk latency (ms), Qwen-Image
(20B DiT), H20-native cost model (r² ≈ 1.0):

| class (latent)            | SP1     | SP2 | SP4     | optimal                    |
| ------------------------- | ------: | --: | ------: | -------------------------- |
| **S** — 512²  (1024 tok)  | **155** | 161 |     172 | SP1 — wider is *slower*    |
| **M** — 1024² (4096 tok)  |     605 | 350 | **207** | SP4 (~3×)                  |
| **L** — 1536² (9216 tok)  |    1557 | 880 | **464** | SP4 (~3.4×)                |

Sweet spots also differ *across stages* (text-encode / DiT / VAE decode) and *across time*
(load, queue depth, free-rank set). One admission-time decision is therefore Pareto-bad for
part of every workload — which is exactly what GF-DiT removes.

## How it works

GF-DiT lives under `vllm_omni/diffusion/runtime_v2/` plus the group-free collective backend.

| Component               | What it does                                                                    | Code |
| ----------------------- | ------------------------------------------------------------------------------ | ---- |
| Task-graph scheduler    | Event-driven dispatch of trajectory tasks; per-request lifecycle               | `runtime_v2/scheduler.py`, `runtime_v2/runner.py` |
| Elastic migration       | Layout-aware online artifact movement across execution groups (SP reshard)     | `runtime_v2/migration_engine.py`, `runtime_v2/data_plane.py` |
| Group-free collectives  | µs-cost online group formation/reconfig (`all_gather` / `all2all` / `p2p`)      | `distributed/collective_runtime.py` + [`gfc`](https://github.com/SJTU-Liquid/group-free-collectives) |
| Pluggable policies      | Choose task order + execution layout per objective                             | `runtime_v2/policies/` — FCFS, SRTF, SJF, EDF-greedy, EDF-best-fit, disaggregate, wave-stress |
| Cost model + simulator  | Per-stage latency model for deadline-aware policies; offline what-if analysis  | `runtime_v2/cost_model.py`, `benchmarks/diffusion/` |

## Quickstart

GF-DiT builds on vLLM-Omni. Install the base framework (see the upstream
[installation guide](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)),
then this fork:

```bash
# base framework + this fork
pip install -e .

# (optional) group-free collectives — enables the elastic group-reconfig path
pip install git+https://github.com/SJTU-Liquid/group-free-collectives.git
```

Serve a DiT model with the elastic runtime enabled:

```bash
vllm serve <your-dit-model> --omni \
  --num-gpus 4 \
  --enable-runtime-v2 \
  --runtime-v2-scheduler-policy edf_best_fit \
  --runtime-v2-collective-backend gfc \
  --runtime-v2-gfc-max-collective-mb 1024
```

Without `gfc` installed, use the dependency-free PyTorch backend — the task-graph scheduler
still runs, but without online group reconfiguration:

```bash
  --runtime-v2-collective-backend torch
```

vLLM-Omni exposes an OpenAI-compatible API; send a generation request (see the upstream
[docs](https://vllm-omni.readthedocs.io/en/latest/) for the full schema and supported
models):

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "<your-dit-model>", "prompt": "a red panda on a skateboard"}'
```

Ready-made launch scripts live in `scripts/` — e.g. `serve-adaptive.sh`, `serve-elastic.sh`,
`serve-edf-best.sh`, `serve-disaggragate.sh`, and the static baselines
`serve-baseline-sp4.sh` / `serve-base.sh`.

### Key flags

| Flag                                 | Meaning |
| ------------------------------------ | ------- |
| `--enable-runtime-v2`                | Turn on the GF-DiT task-graph runtime (opt-in) |
| `--runtime-v2-scheduler-policy`      | `fcfs` · `srtf` · `disaggregate` · `dynamic_step_fcfs` · `edf_greedy` · `edf_best_fit` · `wave_stress` |
| `--runtime-v2-collective-backend`    | `torch` (default, no extra deps) or `gfc` (elastic group reconfig) |
| `--runtime-v2-denoise-chunk-size`    | Denoise steps per schedulable task chunk |
| `--runtime-v2-gfc-max-collective-mb` | GFC symmetric-memory budget per rank (MiB) |

## Reproducing the paper results

Entry points (see the [paper](https://arxiv.org/abs/2606.13501) for the full setup and the
expected numbers):

- **Cost-model profiling** — `scripts/profile-stages.sh` (per-host, per-stage latency grid).
- **Serving benchmark** — `benchmarks/diffusion/diffusion_benchmark_serving.py`
  (throughput / latency / SLO under load).
- **Sim-vs-real grids** — `benchmarks/diffusion/gen_sim_vs_real_grid.py`.
- **GFC vs NCCL group cost** — `benchmarks/distributed/gfc_group_cost.py` versus
  `benchmarks/distributed/nccl_subgroup_cost.py` (the 778 ms → ~60 µs result).

## Relationship to vLLM-Omni

This repository is a fork of [vLLM-Omni](https://github.com/vllm-project/vllm-omni). GF-DiT
is additive: it lives under `vllm_omni/diffusion/runtime_v2/` and the GFC collective
backend, and is gated behind `--enable-runtime-v2`. Base installation, supported models, and
the OpenAI-compatible server all follow upstream vLLM-Omni — see their
[documentation](https://vllm-omni.readthedocs.io/en/latest/).

## Citation

If you use GF-DiT, please cite:

```bibtex
@misc{qiang2026gfditschedulingparallelismdiffusion,
      title={GF-DiT: Scheduling Parallelism for Diffusion Transformer Serving}, 
      author={Xinwei Qiang and Yifan Hu and Shixuan Sun and Jing Yang and Han Zhao and Chen Chen and Yu Feng and Jingwen Leng and Minyi Guo},
      year={2026},
      eprint={2606.13501},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2606.13501}, 
}
```

GF-DiT is built on vLLM-Omni; please also cite their work:

```bibtex
@misc{yin2026vllmomnifullydisaggregatedserving,
      title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models}, 
      author={Peiqi Yin and Jiangyun Zhu and Han Gao and Chenguang Zheng and Yongxiang Huang and Taichang Zhou and Ruirui Yang and Weizhi Liu and Weiqing Chen and Canlin Guo and Didan Deng and Zifeng Mo and Cong Wang and James Cheng and Roger Wang and Hongsheng Liu},
      year={2026},
      eprint={2602.02204},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2602.02204}, 
}
```

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file. GF-DiT inherits vLLM-Omni's
license.
