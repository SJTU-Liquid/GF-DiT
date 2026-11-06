#!/usr/bin/env bash
# G7: real serving runs behind Fig. 6, Sec. 6.5, Fig. 10 (real bars), Table 2 / Sec. 6.4 and Fig. 9.
#
#   EXP=main       Fig. 6: EDF, SRTF-SP1, SRTF-SPmax, FCFS-SP1 and Legacy on the short and
#                  foreground-burst workloads, for MODEL=wan|qwen on PLATFORM=h20 (4 GPUs)
#                  or a100 (8 GPUs). For h20/wan also FCFS-SP4 (Sec. 6.5, Fig. 10).
#                                                  (~7 h Wan2.2, ~5 h Qwen-Image on 4x H20)
#   EXP=breakdown  Table 2 and Sec. 6.4 (H20 Wan2.2): Legacy, FCFS-SP1, SJF-SP1, SRTF-SP1,
#                  EDF-SP1 and EDF on the short workload, then the 40%-load runs
#                  (EDF, FCFS-SP4, EDF-SP1).                                     (~4 h)
#   EXP=load40     only the 40%-load runs of EXP=breakdown.                      (~1 h)
#   EXP=arrival    Fig. 9 (H20 Wan2.2): EDF and SRTF-SP1 on the short workload at target
#                  loads $LOADS.                                                 (~7 h)
#
# Inputs default to the paper's cost model and workloads for $PLATFORM-$MODEL (in
# ae/gpu/inputs); they are calibrated for that hardware and GPU count, and the script
# refuses to run them on a different GPU count (ALLOW_GPU_MISMATCH=1 overrides).
# load40 and arrival are H20 Wan2.2 experiments; elsewhere, or with your own COST /
# WL_SHORT / WL_FG, pass WL_LOAD40 / WL_ARRIVAL_DIR too.
# On other GPUs, point COST, WL_SHORT, WL_FG (and WL_LOAD40, WL_ARRIVAL_DIR) at the
# outputs of 05_cost_model.sh and 06_workloads.sh.
#
# Code and settings as in the paper's runs (override with the variables named):
#   LEGACY_PYTHONPATH  code for the Legacy server; the paper used upstream vLLM-Omni
#                      cf328eb8 (preset in the GPU image). Unset: this repository.
#   MIGRATION=paper|gfc|nccl  transport for state migration. 'paper' (default) uses
#                      NCCL P2P for EXP=main on h20/wan (as in Fig. 6/10) and GFC elsewhere.
#
# Results go to $RESULTS/<exp>-<platform>-<model>/. The exit status is non-zero if any
# run of this invocation failed; summarize_runs.py lists the runs and marks failed ones.
PLATFORM="${PLATFORM:-h20}"
# A100 results used all 8 GPUs; default to them unless GPUS is set.
[ "$PLATFORM" = a100 ] && export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
source "$(dirname "$0")/common.sh"
install_run_traps
EXP="${EXP:-main}"
MODEL="${MODEL:-wan}"
IN="$AE_DIR/gpu/inputs/$PLATFORM-$MODEL"
case "$PLATFORM-$MODEL" in
  h20-wan)   s=h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json;    f=h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json ;;
  h20-qwen)  s=qwenbat_short_tl1.20_seed42.json;                f=qwenbat_foreground_burst_tl0.90_seed42.json ;;
  a100-wan)  s=wanbat8a100_short_tl1.05_seed42.json;            f=wanbat8a100_foreground_burst_tl0.9_seed42.json ;;
  a100-qwen) s=qwenbat8a100_short_tl1.0_origalpha_seed42.json;  f=qwenbat8a100_fg_tl0.7_origalpha_seed42.json ;;
  *) echo "unknown PLATFORM-MODEL '$PLATFORM-$MODEL' (PLATFORM=h20|a100, MODEL=wan|qwen)"; exit 2 ;;
esac
CUSTOM_INPUTS="${COST:-}${WL_SHORT:-}${WL_FG:-}"  # non-empty if the user gave inputs
COST="${COST:-$IN/cost-model}"
WL_SHORT="${WL_SHORT:-$IN/$s}"
WL_FG="${WL_FG:-$IN/$f}"
LOADS="${LOADS:-0.8 1.0 1.1 1.2 1.4 1.6 1.8 2.0}"
OUT="$RESULTS/$EXP-$PLATFORM-$MODEL"

# The bundled load40 / arrival traces belong to the paper's H20 Wan2.2 cost model;
# elsewhere, or with user-given inputs, they must be given too.
if [ "$EXP" = breakdown ] || [ "$EXP" = load40 ] || [ "$EXP" = arrival ]; then
  what="$PLATFORM-$MODEL${CUSTOM_INPUTS:+ with your own COST/WL_SHORT/WL_FG}"
  if [ "$PLATFORM-$MODEL" = h20-wan ] && [ -z "$CUSTOM_INPUTS" ]; then
    WL_LOAD40="${WL_LOAD40:-$IN/h20_short_n4_tl0.4_a2.0_2.5_3.5_s42.json}"
    WL_ARRIVAL_DIR="${WL_ARRIVAL_DIR:-$IN/arrival}"
  elif [ "$EXP" = arrival ] && [ -z "${WL_ARRIVAL_DIR:-}" ]; then
    echo "the bundled arrival traces are for H20 Wan2.2 with the paper's cost model; for $what"
    echo "set WL_ARRIVAL_DIR (a directory; missing ${MODEL}_short_tl<load>_seed42.json files are"
    echo "generated from COST=$COST)"; exit 2
  elif [ "$EXP" != arrival ] && [ -z "${WL_LOAD40:-}" ]; then
    echo "the bundled 40%-load trace is for H20 Wan2.2 with the paper's cost model; for $what"
    echo "set WL_LOAD40, e.g. to \$RESULTS/workloads/${MODEL}_short_tl0.4_seed42.json from:"
    echo "  PLATFORM=$PLATFORM GPUS=$GPUS MODEL=$MODEL TL_SHORT=0.4 COST=$COST ae/gpu/06_workloads.sh"; exit 2
  fi
fi

if [ "$MODEL" = wan ]; then
  HF_MODEL="$WAN_MODEL"; TASK=t2v; BACKEND=v1/videos; SERVE_ARGS="$WAN_SERVE_ARGS"
else
  HF_MODEL="$QWEN_MODEL"; TASK=t2i; BACKEND=openai; SERVE_ARGS=""
fi

# State-migration transport (see header).
case "${MIGRATION:-paper}" in
  paper) { [ "$EXP" = main ] && [ "$PLATFORM-$MODEL" = h20-wan ]; } && mig=nccl || mig=gfc ;;
  gfc|nccl) mig="$MIGRATION" ;;
  *) echo "unknown MIGRATION=$MIGRATION (paper|gfc|nccl)"; exit 2 ;;
esac
[ "$mig" = nccl ] && export VLLM_RUNTIME_V2_MIGRATE_GFC_P2P=0 || export VLLM_RUNTIME_V2_MIGRATE_GFC_P2P=1

# Workloads are calibrated for a GPU count; refuse a mismatch unless overridden.
check_gpus() {  # workload.json
  local n
  n=$("$VENV_BIN/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"]["num_gpus"])' "$1" 2>/dev/null)
  if [ -n "$n" ] && [ "$n" != "$NUM_GPUS" ]; then
    log "$(basename "$1") is calibrated for $n GPUs but NUM_GPUS=$NUM_GPUS (GPUS=$GPUS)"
    [ "${ALLOW_GPU_MISMATCH:-0}" = 1 ] || { log "refusing to run; set GPUS accordingly or ALLOW_GPU_MISMATCH=1"; exit 2; }
  fi
}
case "$EXP" in
  main)      check_gpus "$WL_SHORT"; check_gpus "$WL_FG" ;;
  breakdown) check_gpus "$WL_SHORT"; check_gpus "$WL_LOAD40" ;;
  load40)    check_gpus "$WL_LOAD40" ;;
  arrival)   for tl in $LOADS; do w="$WL_ARRIVAL_DIR/wanscale4_short_tl${tl}_seed42.json"; [ -f "$w" ] && check_gpus "$w"; done ;;
  *) echo "unknown EXP=$EXP (main|breakdown|load40|arrival)"; exit 2 ;;
esac

FAILED=0
plectra() { run_plectra "$HF_MODEL" "$1" "$2" "$COST" "$TASK" "$BACKEND" "$3" $SERVE_ARGS || FAILED=1; }
legacy()  { run_legacy  "$HF_MODEL" "$1" "$2" "$TASK" "$BACKEND" $SERVE_ARGS || FAILED=1; }
load40() {
  plectra "$WL_LOAD40" "$OUT/load40" "edf_best_fit fcfs_sp4"
  SP_SIZES=1 plectra "$WL_LOAD40" "$OUT/load40_edfsp1" "edf_best_fit"
}

log "EXP=$EXP PLATFORM=$PLATFORM MODEL=$MODEL GPUS=$GPUS migration=$mig cost=$COST -> $OUT"
case "$EXP" in
  main)
    pols="edf_best_fit srtf_sp1 srtf fcfs_sp1"
    [ "$PLATFORM-$MODEL" = h20-wan ] && pols="$pols fcfs_sp4"
    for w in short fg; do
      wl=$([ $w = short ] && echo "$WL_SHORT" || echo "$WL_FG")
      plectra "$wl" "$OUT/$w" "$pols"
      legacy "$wl" "$OUT/$w"
    done ;;
  breakdown)
    legacy "$WL_SHORT" "$OUT/short"
    plectra "$WL_SHORT" "$OUT/short" "fcfs_sp1 sjf_sp1 srtf_sp1 edf_best_fit"
    SP_SIZES=1 plectra "$WL_SHORT" "$OUT/short_edfsp1" "edf_best_fit"
    load40 ;;
  load40)
    load40 ;;
  arrival)
    for tl in $LOADS; do
      wl="$WL_ARRIVAL_DIR/wanscale4_short_tl${tl}_seed42.json"
      if [ ! -f "$wl" ]; then
        # Generated with the duration of the paper's short workload (Wan2.2 1200 s, Qwen-Image 600 s).
        wl="$WL_ARRIVAL_DIR/${MODEL}_short_tl${tl}_seed42.json"
        dur=$([ "$MODEL" = qwen ] && echo 600 || echo 1200)
        if [ ! -f "$wl" ] && ! gen_workload "$MODEL" short "$dur" "$tl" "$wl" "$COST"; then
          for p in edf_best_fit srtf_sp1; do mark_failed "$OUT/tl$tl" "sweep_$p" "workload generation failed for $wl"; done
          FAILED=1; continue
        fi
      fi
      plectra "$wl" "$OUT/tl$tl" "edf_best_fit srtf_sp1"
    done ;;
esac
# The table covers everything under $OUT, including earlier invocations; the exit
# status reflects only this invocation's runs (tracked in FAILED above).
"$VENV_BIN/python" ae/gpu/summarize_runs.py "$OUT"; srs=$?
[ "$srs" = 0 ] || [ "$srs" = 3 ] || FAILED=1
[ "$FAILED" = 0 ] && log "ALL RUNS OF THIS INVOCATION SUCCEEDED" || log "SOME RUNS FAILED (see above)"
exit "$FAILED"
