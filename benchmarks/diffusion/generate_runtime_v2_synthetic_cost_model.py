#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic SP1/SP2/SP4/SP8 cost model for runtime_v2 simulations.

The empirical Wan2.2 cost model in ``cost-model/wan22-ti2v-5b/`` was fit on
samples up to ``latent_seq_len=18480``; running it at our serving shape
(720p x 81f gives latent=75600) extrapolates 4x past the fit range, and
the quadratic ``a`` term dominates and gives unrealistic results
(SP2 becomes slower than SP1 etc.). For policy ablations and elastic-SP
sweeps we want a clean, *convex sub-linear* speedup model that holds at
any latent length.

The synthetic model uses Amdahl-like efficiency at the DIT step:

    T_dit_step_ms(latent, sp) = base_per_latent_ms * latent / efficiency(sp)
                              + comm_overhead_ms * (sp - 1)

with ``efficiency(sp)`` configurable on the CLI. Default profile:

    sp=1: efficiency=1.00   ( T proportional to latent  )
    sp=2: efficiency=1.82   ( 0.91 per-rank, 2x parallel )
    sp=4: efficiency=3.33   ( 0.83 per-rank             )
    sp=8: efficiency=5.88   ( 0.74 per-rank             )

That curve is convex sub-linear (Amdahl with f ~= 0.1 sequential portion).
``base_per_latent_ms = 0.1323`` calibrated so SP1 at latent=75600 is
10,000ms / step — within ~5% of the empirical fit at that shape.

Other stages:
  TEXT_ENCODE / VAE_DECODE / FINALIZE are independent of SP (single-rank
  in the simulator), but we still emit a small SP-invariant cost.
  DIT_PREPARE and TIMESTEP_PREPARE are also small constants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_EFFICIENCY = {1: 1.00, 2: 1.82, 4: 3.33, 8: 5.88}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--sp-list",
        default="1,2,4,8",
        help="comma-separated SP degrees to emit",
    )
    parser.add_argument(
        "--efficiency",
        default=None,
        help="comma-separated efficiency overrides, e.g. '1:1.0,2:1.9,4:3.6,8:6.5'",
    )
    parser.add_argument(
        "--base-per-latent-ms",
        type=float,
        default=0.1323,
        help="ms per latent token at SP1 (DIT step). Default calibrated so SP1@latent=75600 = 10000ms.",
    )
    parser.add_argument(
        "--comm-overhead-ms",
        type=float,
        default=20.0,
        help="comm/overhead added per (sp - 1) at each DIT step",
    )
    parser.add_argument("--text-encode-ms", type=float, default=55.0)
    parser.add_argument("--vae-decode-ms-per-voxel", type=float, default=9.0e-8)
    parser.add_argument("--dit-prepare-ms", type=float, default=50.0)
    parser.add_argument("--timestep-prepare-ms", type=float, default=2.0)
    parser.add_argument("--finalize-ms", type=float, default=0.5)
    parser.add_argument("--model-name", default="synthetic-wan22-like")
    return parser.parse_args()


def _parse_efficiency(args: argparse.Namespace) -> dict[int, float]:
    if not args.efficiency:
        return dict(DEFAULT_EFFICIENCY)
    out: dict[int, float] = dict(DEFAULT_EFFICIENCY)
    for pair in args.efficiency.split(","):
        if not pair.strip():
            continue
        key, val = pair.split(":")
        out[int(key)] = float(val)
    return out


def _build_cost_model(
    sp: int,
    *,
    args: argparse.Namespace,
    efficiency: dict[int, float],
) -> dict:
    eff = efficiency.get(sp, float(sp))
    # DIT step chunk: T = b*latent + c (linear in latent, no quadratic).
    b_dit = args.base_per_latent_ms / eff
    c_dit = args.comm_overhead_ms * (sp - 1)
    # VAE: voxels-linear, SP-invariant (text/VAE pinned to rank[0] regardless).
    b_vae = args.vae_decode_ms_per_voxel
    c_vae = 100.0  # constant ramp
    stages = {
        "text_encode": {
            "x_name": "text_seq_len",
            "coeffs": {"a": 0.0, "b": 0.0, "c": args.text_encode_ms},
            "r2": 1.0,
            "samples": [[512.0, args.text_encode_ms]],
        },
        "dit_prepare": {
            "x_name": "latent_seq_len",
            "coeffs": {"a": 0.0, "b": 0.0, "c": args.dit_prepare_ms},
            "r2": 1.0,
            "samples": [[1.0, args.dit_prepare_ms]],
        },
        "timestep_prepare": {
            "x_name": "num_steps",
            "coeffs": {"a": 0.0, "b": 0.0, "c": args.timestep_prepare_ms},
            "r2": 1.0,
            "samples": [[1.0, args.timestep_prepare_ms]],
        },
        "dit_step_chunk": {
            "x_name": "latent_seq_len",
            "coeffs": {"a": 0.0, "b": b_dit, "c": c_dit},
            "r2": 1.0,
            "samples": [[1.0, b_dit + c_dit]],
        },
        "vae_decode": {
            "x_name": "voxels",
            "coeffs": {"a": 0.0, "b": b_vae, "c": c_vae},
            "r2": 1.0,
            "samples": [[1.0, b_vae + c_vae]],
        },
        "finalize": {
            "x_name": "ones",
            "coeffs": {"a": 0.0, "b": 0.0, "c": args.finalize_ms},
            "r2": 1.0,
            "samples": [[1.0, args.finalize_ms]],
        },
    }
    return {
        "model": args.model_name,
        "options": {
            "synthetic": True,
            "efficiency": eff,
            "base_per_latent_ms": args.base_per_latent_ms,
            "comm_overhead_ms": args.comm_overhead_ms,
        },
        "parallelism": {"tp": 1, "sp": sp, "ulysses": sp, "ring": 1, "cfg": 1},
        "stages": stages,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    efficiency = _parse_efficiency(args)
    sp_list = [int(s.strip()) for s in args.sp_list.split(",") if s.strip()]
    for sp in sp_list:
        data = _build_cost_model(sp, args=args, efficiency=efficiency)
        path = out_dir / f"tp1_ulysses{sp}_ring1_cfg1.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        # Print a quick summary at the user's serving shape.
        latent_720p_81f = (720 // 16) * (1280 // 16) * ((81 + 3) // 4)
        dit_step = data["stages"]["dit_step_chunk"]["coeffs"]["b"] * latent_720p_81f + \
                   data["stages"]["dit_step_chunk"]["coeffs"]["c"]
        print(
            f"wrote {path}: SP={sp} efficiency={efficiency.get(sp, sp):.2f} "
            f"dit_step_ms@720p81f={dit_step:.1f}"
        )


if __name__ == "__main__":
    main()
