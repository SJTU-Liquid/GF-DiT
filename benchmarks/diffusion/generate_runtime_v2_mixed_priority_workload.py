#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mixed-priority Poisson workload for runtime_v2 video DiT serving.

Produces a placement-agnostic JSON workload with three request classes:

  S (foreground): short, urgent, non-preemptible, tight SLO
  M (normal):     medium, normal-priority, non-preemptible, normal SLO
  L (background): long, low-priority, preemptible, loose SLO

Arrival is a piecewise-Poisson process so the same workload exercises
low-load, moderate-load, and overload regimes. Each request carries
``profiled_optimal_latency_ms`` (single-card T_single) and
``deadline_ms = arrival_ms + alpha_class * T_single`` so policies can
reason about SLO slack without hard-coding shape information.

Use with:
  --policy benchmarks.diffusion.runtime_v2_step_preemptive_policy:make_demote_policy
  --workload-json runtime_v2_mixed_priority.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


# Class-level shape and step defaults. Denoising stays at the realistic 50
# steps for every class; size scales by resolution (and frame count, video),
# not step count.
#
# Wan video: S and M share 480p (S is a shorter clip); L steps up to 720p.
WAN_CLASS_SHAPES: dict[str, dict[str, int]] = {
    "S": {"height": 480, "width": 832, "num_frames": 49, "num_steps": 50},
    "M": {"height": 480, "width": 832, "num_frames": 81, "num_steps": 50},
    "L": {"height": 720, "width": 1280, "num_frames": 81, "num_steps": 50},
}
# Qwen Image: single-frame, separated purely by resolution. Latent token count
# = (H//16)*(W//16): S=1024, M=4096, L=9216 -> ~9x range for SP-allocation
# lever arm, mirroring the Wan S/M/L spread.
QWEN_CLASS_SHAPES: dict[str, dict[str, int]] = {
    "S": {"height": 512, "width": 512, "num_frames": 1, "num_steps": 50},
    "M": {"height": 1024, "width": 1024, "num_frames": 1, "num_steps": 50},
    "L": {"height": 1536, "width": 1536, "num_frames": 1, "num_steps": 50},
}
# Default (Wan) kept under the original name for back-compat with any importer.
CLASS_SHAPES = WAN_CLASS_SHAPES

CLASS_PRIORITY: dict[str, int] = {"S": 100, "M": 50, "L": 0}
CLASS_PREEMPTIBLE: dict[str, bool] = {"S": False, "M": False, "L": True}
# Default SLO multipliers (alpha) per class. ``alpha`` is multiplied with
# T_single(req) to define the deadline.
CLASS_ALPHA: dict[str, float] = {"S": 1.5, "M": 2.0, "L": 6.0}


def _class_shapes(adapter: str) -> dict[str, dict[str, int]]:
    return QWEN_CLASS_SHAPES if adapter == "qwen" else WAN_CLASS_SHAPES


def _resolve_adapter_kind(override: str, cost_options: Mapping[str, object], model: str) -> str:
    """Pick the adapter. ``auto`` prefers the cost model's recorded adapter,
    then falls back to sniffing the model id."""
    if override and override != "auto":
        return override
    recorded = str(cost_options.get("adapter") or "").lower()
    if recorded in ("wan", "qwen"):
        return recorded
    name = str(model).lower()
    if "qwen" in name:
        return "qwen"
    if "wan" in name:
        return "wan"
    raise SystemExit(
        f"could not infer adapter from cost model options/model={model!r}; pass --adapter wan|qwen"
    )


def _make_latent_seq_len_fn(adapter: str, model: str) -> Callable[[int, int, int], int]:
    """Return (height, width, num_frames) -> latent_seq_len, keyed identically
    to what the runtime adapter computes so cost-model lookups hit the right row."""
    if adapter == "qwen":
        # Mirrors QwenImageRuntimeV2Adapter._estimate_packed_latent_seq_len:
        # VAE spatial scale (8) then transformer patch (2) -> //16, single frame.
        def _qwen(height: int, width: int, num_frames: int = 1) -> int:
            return max(1, int(height) // 16) * max(1, int(width) // 16)

        return _qwen
    from vllm_omni.diffusion.runtime_v2.adapters.wan import WanLatentGeometry

    geometry = WanLatentGeometry.from_model(model)
    return geometry.dit_latent_seq_len


@dataclass(frozen=True)
class Phase:
    name: str
    start_s: float
    end_s: float
    arrival_rate_per_min: float
    mix: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runtime_v2_mixed_priority.json")
    parser.add_argument(
        "--adapter",
        choices=("auto", "wan", "qwen"),
        default="auto",
        help="model adapter; 'auto' reads the cost model's options.adapter then sniffs its model id. "
        "Selects class shapes (Wan video vs Qwen image) and latent geometry.",
    )
    parser.add_argument(
        "--cost-model-dir",
        default="cost-model/wan22-ti2v-5b",
        help="path with SP=1 cost-model JSON; used to derive T_single per class",
    )
    parser.add_argument("--sp1-cost-file", default="tp1_ulysses1_ring1_cfg1.json")
    parser.add_argument(
        "--profile",
        choices=("short", "tridentserve", "foreground_burst", "inflight_burst", "custom"),
        default="tridentserve",
        help="phased load/mix profile. short=10min sanity, tridentserve=60min dynamic mix",
    )
    parser.add_argument("--duration-s", type=float, default=None, help="override profile total duration")
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="scale Poisson rates uniformly (sweep load). Stacks with --num-gpus.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Auto-scale rates to N-GPU capacity. Built-in profiles are calibrated "
            "for an 8-GPU cluster (see comments in _build_phases); this flag "
            "multiplies arrival rates by num_gpus/8 so the workload tracks your "
            "actual hardware. Stacks with --rate-scale: effective_scale = "
            "rate_scale * num_gpus / 8."
        ),
    )
    parser.add_argument(
        "--target-load",
        type=float,
        default=None,
        help=(
            "Auto-calibrate arrival rates from the cost model to hit this offered "
            "load (work GPU-seconds / (num_gpus * arrival window)). 1.0 = exactly "
            "cluster capacity, 1.5 = 150%% (overload stress), 0.8 = underloaded. "
            "Preserves each profile's temporal shape and class mix, rescaling all "
            "phase rates by a single factor. Makes the workload hardware-portable: "
            "the same --target-load expresses the same load on any GPU once you "
            "pass that GPU's cost model. Requires --num-gpus. When set, the "
            "absolute profile rates / --rate-scale only define the shape; "
            "--target-load pins the intensity."
        ),
    )
    parser.add_argument(
        "--alpha-s", type=float, default=CLASS_ALPHA["S"], help="SLO multiplier for class S"
    )
    parser.add_argument(
        "--alpha-m", type=float, default=CLASS_ALPHA["M"], help="SLO multiplier for class M"
    )
    parser.add_argument(
        "--alpha-l", type=float, default=CLASS_ALPHA["L"], help="SLO multiplier for class L"
    )
    parser.add_argument(
        "--deadline-constant-ms",
        type=float,
        default=0.0,
        help=(
            "Fixed end-to-end overhead (ms) added on top of the scheduler "
            "deadline to form the CLIENT-observed deadline. The scheduler's "
            "deadline_ms = arrival + alpha*T_single covers only the GPU "
            "pipeline (text/dit/vae/finalize) the policy controls; the client "
            "additionally pays video post-processing/MP4-encode, IPC handoff, "
            "and the bench's ~2s poll interval before it sees the result. "
            "client_deadline_ms = deadline_ms + this constant, so the bench's "
            "SLO check is not penalized for fixed measurement tax the scheduler "
            "cannot influence. Sim/runtime EDF keep using deadline_ms."
        ),
    )
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--denoise-chunk-size", type=int, default=1)
    parser.add_argument(
        "--do-true-cfg",
        choices=("auto", "true", "false"),
        default="auto",
        help=(
            "Whether generated DiT requests run true CFG. 'auto' picks true for "
            "Wan (its serving default) and false for Qwen Image (the image bench "
            "sends no negative prompt, so serving runs no-CFG). false derives "
            "no-CFG DiT cost from the cost model when no explicit no-CFG curve "
            "exists."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_sp1_cost(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, float]], str, dict[str, object]]:
    path = Path(args.cost_model_dir) / args.sp1_cost_file
    data = json.loads(path.read_text())
    if data.get("parallelism", {}).get("sp", 1) != 1:
        raise ValueError(
            f"--sp1-cost-file must point at an SP=1 cost model, got parallelism={data.get('parallelism')}"
        )
    stages: dict[str, dict[str, float]] = {}
    for stage, body in data["stages"].items():
        coeffs = body["coeffs"]
        stages[stage] = {
            "a": float(coeffs.get("a", 0.0)),
            "b": float(coeffs.get("b", 0.0)),
            "c": float(coeffs.get("c", 0.0)),
            "x_name": str(body.get("x_name", "")),
        }
    model = str(data.get("model") or "")
    if not model:
        raise ValueError(
            f"cost model {path} has no 'model' field; cannot resolve latent "
            "geometry. Re-profile with the stage profiler."
        )
    options = dict(data.get("options", {}) or {})
    return stages, model, options


def _eval_quadratic(stage: Mapping[str, float], x: float) -> float:
    return max(0.0, stage["a"] * x * x + stage["b"] * x + stage["c"])


def _eval_stage(stages: dict[str, dict[str, float]], stage: str, x: float) -> float:
    if stage in stages:
        return _eval_quadratic(stages[stage], x)
    if stage == "dit_step_chunk_cfg" and "dit_step_chunk" in stages:
        return _eval_quadratic(stages["dit_step_chunk"], x)
    if stage == "dit_step_chunk_nocfg":
        if "dit_step_chunk_cfg" in stages:
            return _eval_quadratic(stages["dit_step_chunk_cfg"], x) * 0.5
        if "dit_step_chunk" in stages:
            return _eval_quadratic(stages["dit_step_chunk"], x) * 0.5
    raise KeyError(f"stage {stage!r} not found in cost model")


def _t_single_ms(
    shape: Mapping[str, int],
    text_seq_len: int,
    stages: dict[str, dict[str, float]],
    latent_seq_len: int,
    *,
    do_true_cfg: bool,
) -> float:
    height = shape["height"]
    width = shape["width"]
    frames = shape["num_frames"]
    steps = shape["num_steps"]
    voxels = height * width * frames
    dit_stage = "dit_step_chunk_cfg" if do_true_cfg else "dit_step_chunk_nocfg"
    text = _eval_stage(stages, "text_encode", text_seq_len)
    dit_prepare = _eval_stage(stages, "dit_prepare", latent_seq_len)
    timestep_prepare = _eval_stage(stages, "timestep_prepare", steps)
    dit_step = _eval_stage(stages, dit_stage, latent_seq_len)
    vae = _eval_stage(stages, "vae_decode", voxels)
    finalize = _eval_stage(stages, "finalize", 1.0)
    return text + dit_prepare + timestep_prepare + dit_step * steps + vae + finalize


def _build_phases(profile: str, duration_s: float | None, rate_scale: float) -> list[Phase]:
    if profile == "short":
        # 10-minute sanity profile. Rates calibrated for an 8-GPU video DiT
        # cluster where the average T_single is several minutes: even modest
        # rates already saturate capacity.
        phases = [
            Phase("warmup",      0.0,   60.0,  0.8, {"S": 0.50, "M": 0.35, "L": 0.15}),
            Phase("midload",    60.0,  360.0, 1.5, {"S": 0.55, "M": 0.35, "L": 0.10}),
            Phase("press",     360.0,  540.0, 2.5, {"S": 0.70, "M": 0.25, "L": 0.05}),
            Phase("drain",     540.0,  600.0, 0.0, {"S": 0.0,  "M": 0.0,  "L": 0.0}),
        ]
    elif profile == "tridentserve":
        # 60-minute dynamic mix. Total rate stays in the 0.6-2.5 req/min band,
        # which is roughly 60-180% of nominal 8-GPU capacity given that mid-
        # and large jobs each consume hundreds of GPU-seconds.
        phases = [
            Phase("warmup",      0.0,   600.0, 0.8, {"S": 0.50, "M": 0.35, "L": 0.15}),
            Phase("midload",    600.0, 1500.0, 1.4, {"S": 0.35, "M": 0.40, "L": 0.25}),
            Phase("overload", 1500.0, 2400.0, 2.4, {"S": 0.50, "M": 0.30, "L": 0.20}),
            Phase("bg_heavy",  2400.0, 3000.0, 1.2, {"S": 0.20, "M": 0.35, "L": 0.45}),
            Phase("cooldown",  3000.0, 3600.0, 0.6, {"S": 0.55, "M": 0.35, "L": 0.10}),
        ]
    elif profile == "foreground_burst":
        # 30-minute trace that exercises the in-flight burst mechanism: a
        # steady L background flow, then a foreground S burst hits while L is
        # already running.
        phases = [
            Phase("warmup",     0.0,   180.0, 0.5, {"S": 0.0,  "M": 0.4,  "L": 0.6}),
            Phase("burst1",    180.0,  360.0, 3.0, {"S": 0.85, "M": 0.10, "L": 0.05}),
            Phase("recover",   360.0,  900.0, 0.8, {"S": 0.30, "M": 0.30, "L": 0.40}),
            Phase("burst2",    900.0, 1080.0, 3.0, {"S": 0.85, "M": 0.10, "L": 0.05}),
            Phase("drain",    1080.0, 1800.0, 0.5, {"S": 0.40, "M": 0.30, "L": 0.30}),
        ]
    elif profile == "inflight_burst":
        # Mechanistic workload. L jobs arrive during a quiet phase and start
        # on the latency group; a foreground S burst then arrives several
        # minutes later, after L is well underway. This is the scenario
        # where step-boundary preemption / demotion is meant to help, by
        # moving in-flight L from SP4 to SP1 so S can use SP4 immediately.
        # ``rate_scale`` keeps Poisson rate small to make arrivals
        # deterministic across seeds.
        phases = [
            Phase("warmup_L",      0.0,  300.0, 0.6, {"S": 0.0, "M": 0.0, "L": 1.0}),
            Phase("quiet",       300.0,  600.0, 0.0, {"S": 0.0, "M": 0.0, "L": 0.0}),
            Phase("burst_S",     600.0,  780.0, 4.0, {"S": 1.0, "M": 0.0, "L": 0.0}),
            Phase("drain",       780.0, 1500.0, 0.0, {"S": 0.0, "M": 0.0, "L": 0.0}),
        ]
    else:
        raise ValueError(f"unknown profile {profile!r}")
    if rate_scale != 1.0:
        phases = [Phase(p.name, p.start_s, p.end_s, p.arrival_rate_per_min * rate_scale, p.mix) for p in phases]
    if duration_s is not None:
        scale = duration_s / phases[-1].end_s
        phases = [Phase(p.name, p.start_s * scale, p.end_s * scale, p.arrival_rate_per_min, p.mix) for p in phases]
    return phases


def _expected_offered_load(
    phases: list[Phase],
    class_t_single_s: Mapping[str, float],
    num_gpus: int,
) -> float:
    """Expected offered load = work GPU-seconds / (num_gpus * arrival window).

    Uses expected (rate * duration * mix) request counts rather than a sample,
    so the result is deterministic. T_single seconds doubles as GPU-seconds per
    request: an SP=k placement uses k GPUs for ~T_single/k wall-seconds, i.e.
    ~T_single GPU-seconds either way (communication overhead aside). Window is
    the span of phases that actually emit arrivals (rate > 0); a trailing drain
    phase does not inflate it.
    """
    total_work_s = 0.0
    active_start = None
    active_end = 0.0
    for p in phases:
        if p.arrival_rate_per_min <= 0.0:
            continue
        if active_start is None:
            active_start = p.start_s
        active_end = p.end_s
        expected_reqs = p.arrival_rate_per_min / 60.0 * (p.end_s - p.start_s)
        mix_total = sum(p.mix.values()) or 1.0
        for klass, weight in p.mix.items():
            reqs_c = expected_reqs * (weight / mix_total)
            total_work_s += reqs_c * class_t_single_s.get(klass, 0.0)
    if active_start is None:
        return 0.0
    window_s = max(active_end - active_start, 1e-9)
    return total_work_s / (num_gpus * window_s)


def _sample_class(rng: random.Random, mix: Mapping[str, float]) -> str:
    classes = list(mix.keys())
    weights = [mix[c] for c in classes]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("class mix sums to zero in phase")
    r = rng.random() * total
    cumulative = 0.0
    for c, w in zip(classes, weights):
        cumulative += w
        if r <= cumulative:
            return c
    return classes[-1]


def _sample_poisson_arrivals(rng: random.Random, phases: list[Phase]) -> list[tuple[float, str]]:
    """Generate (arrival_s, request_class) tuples from a piecewise-Poisson process."""
    out: list[tuple[float, str]] = []
    for phase in phases:
        rate_per_s = phase.arrival_rate_per_min / 60.0
        if rate_per_s <= 0.0:
            continue
        t = phase.start_s
        while True:
            # Exponential interarrival.
            u = rng.random()
            if u <= 0.0:
                u = 1e-12
            t = t + (-math.log(u) / rate_per_s)
            if t >= phase.end_s:
                break
            out.append((t, _sample_class(rng, phase.mix)))
    out.sort()
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    stages, model, cost_options = _load_sp1_cost(args)
    adapter = _resolve_adapter_kind(args.adapter, cost_options, model)
    class_shapes = _class_shapes(adapter)
    if args.do_true_cfg == "auto":
        do_true_cfg = adapter != "qwen"
    else:
        do_true_cfg = args.do_true_cfg == "true"
    # latent_seq_len must be keyed identically to the cost model's x-axis and
    # to what the runtime adapter computes, so derive geometry from the same
    # model rather than hardcoding VAE/patch factors.
    latent_seq_len_fn = _make_latent_seq_len_fn(adapter, model)
    # Built-in profile rates were calibrated for an 8-GPU cluster (see comments
    # in _build_phases). --num-gpus rescales them to the user's actual hardware
    # so the same profile expresses the same fraction of capacity; --rate-scale
    # remains an independent knob for load sweeps.
    gpu_scale = (args.num_gpus / 8.0) if args.num_gpus is not None else 1.0
    effective_rate_scale = args.rate_scale * gpu_scale
    phases = _build_phases(args.profile, args.duration_s, effective_rate_scale)

    # Auto-calibrate intensity from the cost model so the same --target-load
    # expresses the same offered load on any hardware (deadlines already scale
    # via T_single; without this the absolute Poisson rates do not).
    target_load_actual: float | None = None
    if args.target_load is not None:
        if args.num_gpus is None:
            raise SystemExit("--target-load requires --num-gpus (capacity needs the GPU count)")
        class_t_single_s = {}
        for klass, shape in class_shapes.items():
            lat = latent_seq_len_fn(shape["height"], shape["width"], shape["num_frames"])
            class_t_single_s[klass] = _t_single_ms(
                shape, args.text_seq_len, stages, lat, do_true_cfg=do_true_cfg
            ) / 1000.0
        current_load = _expected_offered_load(phases, class_t_single_s, args.num_gpus)
        if current_load <= 0.0:
            raise SystemExit("cannot calibrate --target-load: profile has zero expected work")
        k = args.target_load / current_load
        phases = [Phase(p.name, p.start_s, p.end_s, p.arrival_rate_per_min * k, p.mix) for p in phases]
        target_load_actual = _expected_offered_load(phases, class_t_single_s, args.num_gpus)

    arrivals = _sample_poisson_arrivals(rng, phases)
    alphas = {"S": args.alpha_s, "M": args.alpha_m, "L": args.alpha_l}

    class_counts: dict[str, int] = {"S": 0, "M": 0, "L": 0}
    requests: list[dict[str, object]] = []
    realized_work_s = 0.0
    for arrival_s, klass in arrivals:
        idx = class_counts[klass]
        class_counts[klass] += 1
        shape = class_shapes[klass]
        latent_seq_len = latent_seq_len_fn(
            shape["height"], shape["width"], shape["num_frames"]
        )
        t_single_ms = _t_single_ms(
            shape,
            args.text_seq_len,
            stages,
            latent_seq_len,
            do_true_cfg=do_true_cfg,
        )
        realized_work_s += t_single_ms / 1000.0
        deadline_ms = arrival_s * 1000.0 + alphas[klass] * t_single_ms
        client_deadline_ms = deadline_ms + args.deadline_constant_ms
        requests.append(
            {
                "request_id": f"{klass}_{idx:06d}",
                "request_class": klass,
                "arrival_ms": round(arrival_s * 1000.0, 3),
                "height": shape["height"],
                "width": shape["width"],
                "num_frames": shape["num_frames"],
                "num_steps": shape["num_steps"],
                "latent_seq_len": latent_seq_len,
                "denoise_chunk_size": args.denoise_chunk_size,
                "do_true_cfg": do_true_cfg,
                "text_seq_len": args.text_seq_len,
                "priority": CLASS_PRIORITY[klass],
                "preemptible": CLASS_PREEMPTIBLE[klass],
                "deadline_ms": round(deadline_ms, 3),
                "client_deadline_ms": round(client_deadline_ms, 3),
                "profiled_optimal_latency_ms": round(t_single_ms, 3),
            }
        )

    metadata = {
        "profile": args.profile,
        "model": model,
        "adapter": adapter,
        "do_true_cfg": do_true_cfg,
        "rate_scale": args.rate_scale,
        "num_gpus": args.num_gpus,
        "effective_rate_scale": effective_rate_scale,
        "target_load": args.target_load,
        "offered_load_expected": round(target_load_actual, 4) if target_load_actual is not None else None,
        "offered_load_realized": (
            round(
                realized_work_s
                / (
                    args.num_gpus
                    * max(
                        (max(r["arrival_ms"] for r in requests) - min(r["arrival_ms"] for r in requests)) / 1000.0,
                        1e-9,
                    )
                ),
                4,
            )
            if requests and args.num_gpus
            else None
        ),
        "alphas": alphas,
        "deadline_constant_ms": args.deadline_constant_ms,
        "phases": [
            {
                "name": p.name,
                "start_s": p.start_s,
                "end_s": p.end_s,
                "arrival_rate_per_min": p.arrival_rate_per_min,
                "mix": p.mix,
            }
            for p in phases
        ],
        "class_shapes": class_shapes,
        "class_counts": class_counts,
        "duration_s": phases[-1].end_s,
        "seed": args.seed,
    }

    workload = {
        "description": (
            f"Mixed-priority {adapter} DiT serving workload "
            f"(profile={args.profile}, rate_scale={args.rate_scale}, "
            f"num_gpus={args.num_gpus}, effective_rate_scale={effective_rate_scale:g}, "
            f"duration_s={metadata['duration_s']})."
        ),
        "metadata": metadata,
        "requests": requests,
    }
    Path(args.out).write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out} requests={len(requests)} "
        f"S={class_counts['S']} M={class_counts['M']} L={class_counts['L']} "
        f"duration_s={metadata['duration_s']:.0f} "
        f"effective_rate_scale={effective_rate_scale:g}"
    )


if __name__ == "__main__":
    main()
