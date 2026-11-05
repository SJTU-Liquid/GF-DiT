#!/usr/bin/env bash
# G7: real serving runs behind Fig. 6, Table 2 and Fig. 9 (Wan2.2).
#   EXP=main      Fig. 6 : EDF, SRTF-SP1, SRTF-SPmax, FCFS-SP1 + Legacy on short and fg  (~4 h on 4x H20)
#   EXP=breakdown Table 2: Legacy, FCFS-SP1, SJF-SP1, SRTF-SP1, EDF-SP1, EDF on short    (~3 h)
#   EXP=arrival   Fig. 9 : EDF and SRTF-SP1 on short workloads at increasing target load (~6 h)
# Inputs: a cost model (G5) and workloads (G6) for your platform; defaults use the
# paper's H20 files, which are only meaningful on 4x H20.
source "$(dirname "$0")/common.sh"
EXP="${EXP:-main}"
COST="${COST:-ae/sim/cost-model/wan22-ti2v-5b-h20}"
WL_SHORT="${WL_SHORT:-ae/sim/workloads/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json}"
WL_FG="${WL_FG:-ae/sim/workloads/h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json}"
OUT="$RESULTS/$EXP"
plectra() { run_plectra "$WAN_MODEL" "$1" "$2" "$COST" t2v v1/videos "$3" $WAN_SERVE_ARGS; }
legacy()  { run_legacy  "$WAN_MODEL" "$1" "$2" t2v v1/videos $WAN_SERVE_ARGS; }
case "$EXP" in
  main)
    for w in short fg; do
      wl=$([ $w = short ] && echo "$WL_SHORT" || echo "$WL_FG")
      plectra "$wl" "$OUT/$w" "edf_best_fit srtf_sp1 srtf fcfs_sp1"
      legacy "$wl" "$OUT/$w"
    done ;;
  breakdown)
    legacy "$WL_SHORT" "$OUT/short"
    plectra "$WL_SHORT" "$OUT/short" "fcfs_sp1 sjf_sp1 srtf_sp1 edf_best_fit"
    SP_SIZES=1 plectra "$WL_SHORT" "$OUT/short_edfsp1" "edf_best_fit" ;;
  arrival)
    for tl in ${LOADS:-0.8 1.0 1.1 1.2 1.4 1.6 1.8 2.0}; do
      wl="$RESULTS/workloads/wan_short_tl${tl}_seed42.json"
      [ -f "$wl" ] || "$VENV_BIN/python" benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py \
        --out "$wl" --cost-model-dir "$COST" --profile short --duration-s 1200 --num-gpus "$NUM_GPUS" \
        --target-load "$tl" --alpha-s 2.0 --alpha-m 2.5 --alpha-l 3.5 --deadline-constant-ms 5000 --seed 42
      plectra "$wl" "$OUT/tl$tl" "edf_best_fit srtf_sp1"
    done ;;
  *) echo "unknown EXP=$EXP (main|breakdown|arrival)"; exit 2 ;;
esac
"$VENV_BIN/python" ae/gpu/summarize_runs.py "$OUT"
