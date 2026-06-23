#!/usr/bin/env bash
# Run runtime_v2_stage_profiler.py across the SP/CFG cartesian product, one
# JSON per (tp, sp, cfg) config. Skips configs that exceed visible GPUs.
#
# Usage:
#   bash scripts/profile-stages.sh \
#       --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
#       --out-dir cost-model/wan22-ti2v-5b
#
# Override the search grid:
#   bash scripts/profile-stages.sh --tp 1 --sp 1,2,4 --cfg 1
set -euo pipefail

MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
OUT_DIR="cost-model/wan22-ti2v-5b"
TP_LIST="1"
SP_LIST="1,2,4"          # interpreted as ulysses; we leave ring=1 by default
CFG_LIST="1"
RING_LIST="1"
EAGER_FLAG="--enforce-eager"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2"; shift 2;;
    --out-dir)  OUT_DIR="$2"; shift 2;;
    --tp)       TP_LIST="$2"; shift 2;;
    --sp)       SP_LIST="$2"; shift 2;;
    --ring)     RING_LIST="$2"; shift 2;;
    --cfg)      CFG_LIST="$2"; shift 2;;
    --no-eager) EAGER_FLAG=""; shift 1;;
    --)         shift; EXTRA_ARGS+=("$@"); break;;
    *)          EXTRA_ARGS+=("$1"); shift;;
  esac
done

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
if [[ "$VISIBLE_GPUS" == *","* ]]; then
  NUM_GPUS=$(echo "$VISIBLE_GPUS" | tr ',' '\n' | wc -l)
else
  NUM_GPUS="$VISIBLE_GPUS"
fi
echo "detected $NUM_GPUS visible GPU(s)"
mkdir -p "$OUT_DIR"

IFS=',' read -r -a TPS <<< "$TP_LIST"
IFS=',' read -r -a SPS <<< "$SP_LIST"
IFS=',' read -r -a RINGS <<< "$RING_LIST"
IFS=',' read -r -a CFGS <<< "$CFG_LIST"

for tp in "${TPS[@]}"; do
  for sp in "${SPS[@]}"; do
    for ring in "${RINGS[@]}"; do
      for cfg in "${CFGS[@]}"; do
        world=$(( tp * sp * cfg ))
        if (( world > NUM_GPUS )); then
          echo "skip tp=$tp sp=$sp ring=$ring cfg=$cfg (world=$world > visible=$NUM_GPUS)"
          continue
        fi
        # sp here is the *sequence* dim (ulysses*ring); the script reads
        # ulysses & ring separately, so split sp into ulysses given ring.
        if (( sp % ring != 0 )); then
          echo "skip tp=$tp sp=$sp ring=$ring (sp not divisible by ring)"
          continue
        fi
        ulysses=$(( sp / ring ))

        tag="tp${tp}_ulysses${ulysses}_ring${ring}_cfg${cfg}"
        out_path="$OUT_DIR/${tag}.json"
        echo
        echo "=== profiling $tag (world=$world) ==="
        torchrun --standalone --nproc_per_node="$world" \
          benchmarks/diffusion/runtime_v2_stage_profiler.py \
          --model "$MODEL" \
          --out "$out_path" \
          --tensor-parallel-size "$tp" \
          --ulysses-degree "$ulysses" \
          --ring-degree "$ring" \
          --cfg-parallel-size "$cfg" \
          $EAGER_FLAG \
          "${EXTRA_ARGS[@]}"
      done
    done
  done
done

echo
echo "all configurations finished; outputs under $OUT_DIR"
