#!/usr/bin/env bash
# Profile Wan2.2 VAE decode latency across patch/tile-parallel degrees, so the
# motivation figure can show how the VAE stage scales with execution group size.
#
# Each PP level is one torchrun world (nproc_per_node = PP). The served
# TI2V-5B pipeline uses the plain single-rank VAE; this loads the distributed
# DistributedAutoencoderKLWan instead (see vae_patch_parallel_profiler.py).
#
# Usage (4 GPUs to cover PP up to 4):
#   bash scripts/profile-vae-pp.sh
#
# Override:
#   bash scripts/profile-vae-pp.sh --pp 1,2,4 --out-dir out/vae \
#       --warmup-iters 5 --bench-iters 20
set -euo pipefail

OUT_DIR="out/vae"
PP_LIST="1,2,4"
WARMUP_ITERS=3
BENCH_ITERS=10
MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
GRID_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2;;
    --out-dir)      OUT_DIR="$2"; shift 2;;
    --pp)           PP_LIST="$2"; shift 2;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2;;
    --bench-iters)  BENCH_ITERS="$2"; shift 2;;
    --grid-json)    GRID_JSON="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="python3"
[[ -x "$REPO_ROOT/.venv/bin/python" ]] && PY="$REPO_ROOT/.venv/bin/python"
TORCHRUN="torchrun"
[[ -x "$REPO_ROOT/.venv/bin/torchrun" ]] && TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi
echo "detected $NUM_GPUS visible GPU(s)"
mkdir -p "$OUT_DIR"

GRID_ARG=()
[[ -n "$GRID_JSON" ]] && GRID_ARG=(--grid-json "$GRID_JSON")

IFS=',' read -r -a PPS <<< "$PP_LIST"
for pp in "${PPS[@]}"; do
  if (( pp > NUM_GPUS )); then
    echo "skip PP=$pp (world=$pp > visible=$NUM_GPUS)"
    continue
  fi
  out_path="$OUT_DIR/vae_pp${pp}.json"
  echo
  echo "=== profiling VAE PP=$pp (warmup=$WARMUP_ITERS bench=$BENCH_ITERS) ==="
  if (( pp == 1 )); then
    "$PY" benchmarks/diffusion/vae_patch_parallel_profiler.py \
      --model "$MODEL" --out "$out_path" \
      --warmup-iters "$WARMUP_ITERS" --bench-iters "$BENCH_ITERS" \
      "${GRID_ARG[@]}"
  else
    "$TORCHRUN" --standalone --nproc_per_node="$pp" \
      benchmarks/diffusion/vae_patch_parallel_profiler.py \
      --model "$MODEL" --out "$out_path" \
      --warmup-iters "$WARMUP_ITERS" --bench-iters "$BENCH_ITERS" \
      "${GRID_ARG[@]}"
  fi
done

echo
echo "all PP levels finished; outputs under $OUT_DIR"
