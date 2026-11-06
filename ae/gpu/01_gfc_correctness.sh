#!/usr/bin/env bash
# G1 (Sec. 4): GFC correctness suite (overlapping groups, barriers, all2all,
# all_gather, p2p). Needs >= 2 GPUs with P2P / NVLink; the overlapping-group
# tests need 3-4 GPUs and the largest stress tests 8; tests needing more GPUs
# than $GPUS are skipped (and counted below). ~5 min.
source "$(dirname "$0")/common.sh"
[ "$NUM_GPUS" -ge 2 ] || { log "FAILED: needs >= 2 GPUs (GPUS=$GPUS)"; exit 1; }
[ "$NUM_GPUS" -ge 4 ] || log "WARNING: with $NUM_GPUS GPUs the overlapping-group tests are skipped; use >= 4 GPUs"
GFC_PY="$VENV_BIN/python" GFC_TORCHRUN="$VENV_BIN/torchrun" GFC_GPUS="$GPUS" \
  bash GroupFree-Collective/tests/run_all.sh 2>&1 | tee "$RESULTS/gfc_correctness.log"
rc=${PIPESTATUS[0]}
log "$(grep -c '^=== torchrun' "$RESULTS/gfc_correctness.log") multi-GPU tests run, $(grep -c '^=== SKIP' "$RESULTS/gfc_correctness.log") skipped for lack of GPUs; exit $rc"
exit "$rc"
