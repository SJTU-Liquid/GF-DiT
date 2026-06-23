#!/usr/bin/env bash
# Profile Wan2.2 DiT stages across a grid that covers the real serving shape
# (720p x 81 frames), so the fitted cost model is valid at the serving point
# instead of extrapolating past the fit range.
#
# latent_seq_len is the DiT transformer sequence length, derived from the
# model's own config via WanLatentGeometry (VAE spatial/temporal compression
# then the transformer patch fold). For Wan2.2-TI2V-5B the grid below spans
# latent_seq_len ~245 -> 18480.
#
# Usage (4 GPUs minimum to also cover SP=4):
#   bash scripts/profile-stages-fullrange.sh
#
# Override:
#   bash scripts/profile-stages-fullrange.sh \
#       --out-dir cost-model/wan22-ti2v-5b-fullrange \
#       --sp 1,2,4,8 \
#       --warmup-iters 30 --bench-iters 50 \
#       --skip-sp1-largest         # drop 720x1280x81 for SP=1 only (OOM-prone)
#
# Notes:
#   * SP=1 at 720x1280x81 may OOM on <80GB GPUs. Use --skip-sp1-largest, or
#     drop SP=1 from --sp entirely if even smaller points blow memory.
#   * eager-mode per-call timings are stable; a few warmup iters settle the
#     cuBLAS/allocator first-call cost. Raise --bench-iters only if a stage's
#     fit r2 looks noisy.
#   * Defaults to --collective-backend gfc to match edf_best_fit serving (its
#     SP all-to-all is faster than torch). Pass torch only to profile that path.
#   * For a single shape with torch.compile pass --no-eager.
set -euo pipefail

OUT_DIR="cost-model/wan22-ti2v-5b-fullrange"
SP_LIST="1,2,4"
WARMUP_ITERS=5
BENCH_ITERS=20
AUX_WARMUP_ITERS=2
AUX_BENCH_ITERS=5
COLLECTIVE_BACKEND=gfc
MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
GUIDANCE_SCALE=4.0
GUIDANCE_SCALE_HIGH=""
ENFORCE_EAGER=1
SKIP_SP1_LARGEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)              MODEL="$2"; shift 2;;
    --out-dir)            OUT_DIR="$2"; shift 2;;
    --sp)                 SP_LIST="$2"; shift 2;;
    --warmup-iters)       WARMUP_ITERS="$2"; shift 2;;
    --bench-iters)        BENCH_ITERS="$2"; shift 2;;
    --aux-warmup-iters)   AUX_WARMUP_ITERS="$2"; shift 2;;
    --aux-bench-iters)    AUX_BENCH_ITERS="$2"; shift 2;;
    --collective-backend) COLLECTIVE_BACKEND="$2"; shift 2;;
    --guidance-scale)     GUIDANCE_SCALE="$2"; shift 2;;
    --guidance-scale-high) GUIDANCE_SCALE_HIGH="$2"; shift 2;;
    --no-eager)           ENFORCE_EAGER=0; shift 1;;
    --skip-sp1-largest)   SKIP_SP1_LARGEST=1; shift 1;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Grid: 10 points, latent_seq_len from 245 to 18480 (patch-folded).
#   240x240x17  = 245     (small-image floor)
#   240x480x17  = 525
#   480x480x17  = 1125
#   480x832x17  = 1950
#   720x720x17  = 2420
#   720x1280x17 = 4400
#   480x832x49  = 5070    (class-S serving shape)
#   480x832x81  = 8190
#   720x1280x49 = 11440
#   720x1280x81 = 18480   (class-M/L serving shape)
GRID_FULL='[
  {"height":240,"width":240,"num_frames":17},
  {"height":240,"width":480,"num_frames":17},
  {"height":480,"width":480,"num_frames":17},
  {"height":480,"width":832,"num_frames":17},
  {"height":720,"width":720,"num_frames":17},
  {"height":720,"width":1280,"num_frames":17},
  {"height":480,"width":832,"num_frames":49},
  {"height":480,"width":832,"num_frames":81},
  {"height":720,"width":1280,"num_frames":49},
  {"height":720,"width":1280,"num_frames":81}
]'

# Same grid minus the top point — only used for SP=1 if --skip-sp1-largest is set.
GRID_NO_TOP='[
  {"height":240,"width":240,"num_frames":17},
  {"height":240,"width":480,"num_frames":17},
  {"height":480,"width":480,"num_frames":17},
  {"height":480,"width":832,"num_frames":17},
  {"height":720,"width":720,"num_frames":17},
  {"height":720,"width":1280,"num_frames":17},
  {"height":480,"width":832,"num_frames":49},
  {"height":480,"width":832,"num_frames":81},
  {"height":720,"width":1280,"num_frames":49}
]'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Detect visible GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi
echo "detected $NUM_GPUS visible GPU(s)"
mkdir -p "$OUT_DIR"

EAGER_ARG=()
[[ "$ENFORCE_EAGER" -eq 1 ]] && EAGER_ARG=(--enforce-eager)
GUIDANCE_HIGH_ARG=()
[[ -n "$GUIDANCE_SCALE_HIGH" ]] && GUIDANCE_HIGH_ARG=(--guidance-scale-high "$GUIDANCE_SCALE_HIGH")

IFS=',' read -r -a SPS <<< "$SP_LIST"

for sp in "${SPS[@]}"; do
  if (( sp > NUM_GPUS )); then
    echo "skip SP=$sp (world=$sp > visible=$NUM_GPUS)"
    continue
  fi

  grid="$GRID_FULL"
  if (( sp == 1 )) && (( SKIP_SP1_LARGEST == 1 )); then
    grid="$GRID_NO_TOP"
    echo "SP=1: skipping 720x1280x81 (--skip-sp1-largest)"
  fi

  tag="tp1_ulysses${sp}_ring1_cfg1"
  out_path="$OUT_DIR/${tag}.json"
  effective_backend="$COLLECTIVE_BACKEND"
  if (( sp == 1 )) && [[ "$effective_backend" == "gfc" ]]; then
    effective_backend=torch
  fi
  echo
  echo "=== profiling $tag (world=$sp, backend=$effective_backend, warmup=$WARMUP_ITERS bench=$BENCH_ITERS guidance=$GUIDANCE_SCALE high=${GUIDANCE_SCALE_HIGH:-same}) ==="
  torchrun --standalone --nproc_per_node="$sp" \
    benchmarks/diffusion/runtime_v2_stage_profiler.py \
    --model "$MODEL" \
    --out "$out_path" \
    --tensor-parallel-size 1 \
    --ulysses-degree "$sp" \
    --ring-degree 1 \
    --cfg-parallel-size 1 \
    --warmup-iters "$WARMUP_ITERS" \
    --bench-iters "$BENCH_ITERS" \
    --aux-warmup-iters "$AUX_WARMUP_ITERS" \
    --aux-bench-iters "$AUX_BENCH_ITERS" \
    --collective-backend "$effective_backend" \
    --guidance-scale "$GUIDANCE_SCALE" \
    "${GUIDANCE_HIGH_ARG[@]}" \
    --grid-json "$grid" \
    "${EAGER_ARG[@]}"
done

echo
echo "all SP levels finished; outputs under $OUT_DIR"
echo "to use this cost-model in simulations, point your runner at $OUT_DIR"
