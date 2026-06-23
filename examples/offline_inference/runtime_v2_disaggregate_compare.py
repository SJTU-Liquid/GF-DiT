# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness check for runtime_v2 disaggregate stage placement.

Runs the same Wan2.2 T2V prompt through two configurations in separate
subprocesses (so torch.distributed state can be fully torn down between runs),
saves the decoded frames as .npy, then compares them element-wise.

Configurations:
  * disagg   — 2 GPUs: rank 0 runs text_encode+VAE_decode+finalize (TP=1 SP=1),
               rank 1 runs DiT (TP=1 SP=1). Tests the RESHARD boundary in
               isolation, no sequence parallelism.
  * baseline — 1 GPU single group, single rank, all stages in one group, FCFS.
               This is the reference output we expect disagg to match.

Run:
  python examples/offline_inference/runtime_v2_disaggregate_compare.py

The two runs share the same seed, prompt, steps and image size. Differences
between the two outputs come from RESHARD round-tripping (D2D copy + pickle
of non-tensor metadata) and any host-side scheduling reordering; they should
be tiny but rarely zero (mixed-precision matmul is not strictly deterministic
across kernels chosen by NCCL stream interactions).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


_DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)


def _common_arg_keys() -> tuple[str, ...]:
    # Args that must be forwarded verbatim from the outer driver to each
    # subprocess run so both configurations see identical generation params.
    return (
        "model",
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "height",
        "width",
        "num_frames",
        "boundary_ratio",
        "flow_shift",
        "guidance_scale",
        "guidance_scale_high",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8, help="num_inference_steps (kept small for fast comparison)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=17, help="kept short for fast iteration")
    p.add_argument("--boundary-ratio", type=float, default=0.875)
    p.add_argument("--flow-shift", type=float, default=5.0)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--guidance-scale-high", type=float, default=4.0)
    p.add_argument("--disagg-output", default="/tmp/wan22_disagg_frames.npy")
    p.add_argument("--baseline-output", default="/tmp/wan22_baseline_frames.npy")
    p.add_argument(
        "--disagg-visible-devices",
        default="1,2",
        help="CUDA_VISIBLE_DEVICES override for the disaggregate run (1+1, needs 2 GPUs)",
    )
    p.add_argument(
        "--baseline-visible-devices",
        default="2",
        help="CUDA_VISIBLE_DEVICES override for the baseline run (single GPU)",
    )
    p.add_argument(
        "--skip-disagg",
        action="store_true",
        help="reuse an existing disagg .npy and only run the baseline + compare",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="reuse an existing baseline .npy and only run the disagg + compare",
    )
    p.add_argument(
        "--internal-run",
        action="store_true",
        help="(internal) when set, run a single configuration in this process",
    )
    p.add_argument("--mode", choices=("disagg", "baseline"))
    p.add_argument("--output", default=None)
    return p.parse_args()


def _run_inference(args: argparse.Namespace) -> None:
    """Single-configuration entry. Loads the model, runs one generate(),
    saves the decoded frames as a numpy array, exits.
    """
    import torch  # noqa: F401  # used by tensor handling in _frames_to_numpy

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput  # noqa: F401  # used by _extract_frames

    if args.mode == "disagg":
        # 1 + 1: aux on rank 0, dit on rank 1, no sequence parallelism.
        num_gpus = 2
        groups_json = json.dumps(
            [
                {
                    "group_id": "g_aux",
                    "ranks": [0],
                    "tp": 1,
                    "sp": 1,
                    "cfg": 1,
                    "supported_task_kinds": ["text_encode", "vae_decode", "finalize"],
                },
                {
                    "group_id": "g_dit",
                    "ranks": [1],
                    "tp": 1,
                    "sp": 1,
                    "cfg": 1,
                    "supported_task_kinds": ["dit_prepare", "timestep_prepare", "dit_step_chunk"],
                },
            ]
        )
        scheduler_policy = "disaggregate"
        # ParallelConfig is informational here; the actual per-group parallel
        # spec comes from runtime_v2_groups_json. We still need a valid object.
        parallel_config = DiffusionParallelConfig(
            ulysses_degree=1, ring_degree=1, cfg_parallel_size=1, tensor_parallel_size=1
        )
        disagg_kwargs = {
            "runtime_v2_disaggregate_aux_group_id": "g_aux",
            "runtime_v2_disaggregate_dit_group_id": "g_dit",
        }
    elif args.mode == "baseline":
        # Single GPU, single group, no parallelism. The reference output.
        num_gpus = 1
        groups_json = json.dumps(
            [
                {
                    "group_id": "g0",
                    "ranks": [0],
                    "tp": 1,
                    "sp": 1,
                    "cfg": 1,
                },
            ]
        )
        scheduler_policy = "fcfs"
        parallel_config = DiffusionParallelConfig(
            ulysses_degree=1, ring_degree=1, cfg_parallel_size=1, tensor_parallel_size=1
        )
        disagg_kwargs = {}
    else:
        raise ValueError(f"unknown internal mode: {args.mode!r}")

    print(f"=== internal run: mode={args.mode} num_gpus={num_gpus} ===", flush=True)
    print(f"groups_json={groups_json}", flush=True)

    omni = Omni(
        model=args.model,
        num_gpus=num_gpus,
        enable_runtime_v2=True,
        runtime_v2_scheduler_policy=scheduler_policy,
        runtime_v2_groups_json=groups_json,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        parallel_config=parallel_config,
        **disagg_kwargs,
    )

    # Pass `seed` as an int rather than a pre-built torch.Generator object.
    # The Generator is device-bound; round-tripping it through the stage-worker
    # subprocess (which lives in its own CUDA_VISIBLE_DEVICES sandbox) is
    # fragile and easily loses state, which means the dit_prepare task ends up
    # sampling fresh noise from the global RNG and the two runs diverge from
    # step 0. The text_encode executor reconstructs the generator from `seed`
    # locally on the worker side, which is reproducible across configurations.
    output = omni.generate(
        {"prompt": args.prompt, "negative_prompt": args.negative_prompt},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_high,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
        ),
    )

    frames = _extract_frames(output)
    if frames is None:
        raise RuntimeError("could not locate decoded frames in omni.generate output")
    np_frames = _frames_to_numpy(frames)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np_frames)
    print(f"saved frames: path={out_path} shape={np_frames.shape} dtype={np_frames.dtype}", flush=True)


def _extract_frames(output):
    """Walk the heterogeneous Omni output shape to find the actual frame tensor.
    Logic mirrors what examples/offline_inference/text_to_video.py does.
    """
    from vllm_omni.outputs import OmniRequestOutput

    cursor = output
    if isinstance(cursor, list):
        cursor = cursor[0] if cursor else None
    if isinstance(cursor, OmniRequestOutput):
        if cursor.is_pipeline_output and cursor.request_output is not None:
            inner = cursor.request_output
            if isinstance(inner, list):
                inner = inner[0] if inner else None
            if isinstance(inner, OmniRequestOutput):
                cursor = inner
        if isinstance(cursor, OmniRequestOutput) and cursor.images:
            cursor = cursor.images[0] if isinstance(cursor.images, list) else cursor.images
    if isinstance(cursor, tuple) and len(cursor) == 2:
        cursor = cursor[0]
    if isinstance(cursor, dict):
        cursor = cursor.get("frames") or cursor.get("video") or cursor
    return cursor


def _frames_to_numpy(frames) -> np.ndarray:
    import torch

    if isinstance(frames, torch.Tensor):
        t = frames.detach().cpu()
        if t.dim() == 5:
            t = t[0]
        if t.dim() == 4 and t.shape[0] in (3, 4):
            t = t.permute(1, 2, 3, 0)
        if t.is_floating_point():
            t = t.clamp(-1, 1) * 0.5 + 0.5
        return t.float().numpy()
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 5:
            arr = arr[0]
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 255.0
        return arr
    if isinstance(frames, list) and frames and isinstance(frames[0], (np.ndarray, torch.Tensor)):
        return np.stack([_frames_to_numpy(f) for f in frames], axis=0)
    raise TypeError(f"unsupported frame container type: {type(frames).__name__}")


def _spawn_run(
    args: argparse.Namespace,
    *,
    mode: str,
    output: str,
    visible_devices: str,
) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    cmd = [
        sys.executable,
        __file__,
        "--internal-run",
        "--mode",
        mode,
        "--output",
        output,
    ]
    for key in _common_arg_keys():
        cli_key = "--" + key.replace("_", "-")
        value = getattr(args, key)
        cmd.extend([cli_key, str(value)])
    print(f"\n=== launching {mode} subprocess on CUDA_VISIBLE_DEVICES={visible_devices} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def _compare(a_path: str, b_path: str) -> None:
    a = np.load(a_path)
    b = np.load(b_path)
    print("\n=== compare ===")
    print(f"disagg   : path={a_path} shape={a.shape} dtype={a.dtype}")
    print(f"baseline : path={b_path} shape={b.shape} dtype={b.dtype}")
    if a.shape != b.shape:
        print(f"!! SHAPE MISMATCH — refusing to diff")
        return
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)
    print(f"global   : max_abs={diff.max():.6e}  mean_abs={diff.mean():.6e}  rmse={np.sqrt((diff ** 2).mean()):.6e}")
    # MSE-style PSNR on [0,1] floating-point frames.
    mse = ((a64 - b64) ** 2).mean()
    if mse > 0:
        psnr = 10.0 * np.log10(1.0 / mse)
        print(f"PSNR     : {psnr:.2f} dB (higher = closer)")
    else:
        print("PSNR     : inf (bit-exact)")
    # Per-frame breakdown.
    n_frames = a.shape[0] if a.ndim >= 4 else 1
    print(f"per-frame (showing up to 8 of {n_frames}):")
    for i in range(min(8, n_frames)):
        fa = a64[i] if a.ndim >= 4 else a64
        fb = b64[i] if b.ndim >= 4 else b64
        d = np.abs(fa - fb)
        print(f"  frame[{i:3d}]: max_abs={d.max():.6e} mean_abs={d.mean():.6e}")


def main() -> None:
    args = parse_args()

    if args.internal_run:
        if args.mode is None or args.output is None:
            raise SystemExit("--internal-run requires --mode and --output")
        _run_inference(args)
        return

    if not args.skip_disagg:
        _spawn_run(args, mode="disagg", output=args.disagg_output, visible_devices=args.disagg_visible_devices)
    else:
        print(f"=== skipping disagg run, reusing {args.disagg_output} ===")
    if not args.skip_baseline:
        _spawn_run(args, mode="baseline", output=args.baseline_output, visible_devices=args.baseline_visible_devices)
    else:
        print(f"=== skipping baseline run, reusing {args.baseline_output} ===")
    _compare(args.disagg_output, args.baseline_output)


if __name__ == "__main__":
    main()
