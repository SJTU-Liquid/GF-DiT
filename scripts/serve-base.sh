#!/usr/bin/env bash
# Launch the baseline single-group SP=2 runtime_v2 serve setup.
#
# Usage:
#   bash scripts/serve-base.sh                       # default torch backend
#   bash scripts/serve-base.sh gfc                   # GFC backend, default 128 MiB / 4 slots
#   GFC_MAX_MB=256 GFC_SLOTS=2 bash scripts/serve-base.sh gfc
#   COLLECTIVE_BACKEND=gfc bash scripts/serve-base.sh
#
# The GFC buffer is a per-rank symmetric-memory pool. It needs to fit the
# WORST-CASE single SP collective payload; varying request sizes just reuse it
# in time. Per-rank footprint ~= GFC_MAX_MB * GFC_SLOTS MiB.
set -euo pipefail

BACKEND="${1:-${COLLECTIVE_BACKEND:-torch}}"
case "$BACKEND" in
  torch|gfc) ;;
  *) echo "unsupported backend '$BACKEND' (expected: torch | gfc)" >&2; exit 2;;
esac

GFC_MAX_MB="${GFC_MAX_MB:-128}"
GFC_SLOTS="${GFC_SLOTS:-4}"

vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
    --port 8098 \
    --num-gpus 2 \
    --enable-runtime-v2 \
    --runtime-v2-scheduler-policy fcfs \
    --runtime-v2-collective-backend "$BACKEND" \
    --runtime-v2-gfc-max-collective-mb "$GFC_MAX_MB" \
    --runtime-v2-gfc-num-comm-slots "$GFC_SLOTS" \
    --runtime-v2-groups-json '[
      {"group_id":"g0","ranks":[0,1],"tp":1,"sp":2,
       "ulysses_degree":2,"ring_degree":1,"cfg":1}
    ]' \
    --boundary-ratio 0.875 \
    --flow-shift 5.0 --runtime-v2-gfc-max-collective-mb 1024
