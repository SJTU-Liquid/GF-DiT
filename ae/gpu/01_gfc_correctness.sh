#!/usr/bin/env bash
# G1 (Sec. 4): GFC correctness suite (overlapping groups, barriers, all2all,
# all_gather, p2p). Needs >= 2 GPUs with P2P / NVLink; tests needing more GPUs
# than $GPUS are skipped. ~10-20 min.
source "$(dirname "$0")/common.sh"
GFC_PY="$VENV_BIN/python" GFC_TORCHRUN="$VENV_BIN/torchrun" GFC_GPUS="$GPUS" \
  bash GroupFree-Collective/tests/run_all.sh 2>&1 | tee "$RESULTS/gfc_correctness.log"
