#!/usr/bin/env bash
# G5: profile the per-stage cost model on YOUR GPUs (the policies and the
# simulator need it; the paper's H20/A100 models are in ae/sim/cost-model and
# ae/figures/logs). Writes cost-model/<model>-<tag>/tp1_ulysses<k>_ring1_cfg1.json.
# ~1-2 h per model.  MODEL=wan|qwen
source "$(dirname "$0")/common.sh"
TAG="${TAG:-local}"
case "${MODEL:-wan}" in
  wan)  bash scripts/profile-stages.sh --model "$WAN_MODEL" --out-dir "cost-model/wan22-ti2v-5b-$TAG" --sp "$SP_SIZES" ;;
  qwen) bash scripts/profile-stages.sh --model "$QWEN_MODEL" --out-dir "cost-model/qwen-image-$TAG" --sp "$SP_SIZES" ;;
esac
