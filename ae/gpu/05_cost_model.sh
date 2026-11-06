#!/usr/bin/env bash
# G5: profile the per-stage cost model on YOUR GPUs (the policies and the
# simulator need it; the paper's H20/A100 models are in ae/gpu/inputs). Runs on
# the GPUs in $GPUS only and writes
#   $RESULTS/cost-model/<model>-$TAG/tp1_ulysses<k>_ring1_cfg1.json   for k in $SP_SIZES
# (under the results directory, which the GPU container mounts from the host).
# ~1-2 h per model.  MODEL=wan|qwen (default wan), TAG (default local).
source "$(dirname "$0")/common.sh"
TAG="${TAG:-local}"
case "${MODEL:-wan}" in
  wan)  hf="$WAN_MODEL";  out="$RESULTS/cost-model/wan22-ti2v-5b-$TAG" ;;
  qwen) hf="$QWEN_MODEL"; out="$RESULTS/cost-model/qwen-image-$TAG" ;;
  *) echo "unknown MODEL=$MODEL (wan|qwen)"; exit 2 ;;
esac
log "profiling $hf on GPUs $GPUS (sp $SP_SIZES) -> $out"
# GFC for SP >= 2, as serving and the paper's cost models (the profiler uses torch at SP = 1).
CUDA_VISIBLE_DEVICES="$GPUS" bash scripts/profile-stages.sh --model "$hf" --out-dir "$out" --sp "$SP_SIZES" \
  --collective-backend gfc || exit 1
ls -la "$out"
