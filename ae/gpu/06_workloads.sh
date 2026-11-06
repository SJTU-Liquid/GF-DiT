#!/usr/bin/env bash
# G6: generate the short and foreground-burst workloads for YOUR platform from its
# cost model (05_cost_model.sh), with the paper's generator settings (Sec. 6.1).
#   MODEL=wan|qwen (default wan); COST=<cost model dir> (default: 05's output for MODEL);
#   TL_SHORT / TL_FG = target load. Workloads are generated for NUM_GPUS GPUs (from GPUS);
#   PLATFORM=a100 defaults GPUS to all 8, as in 07_policy_runs.sh.
# Target load is relative to the platform's estimated capacity; the paper used
#   H20  Wan2.2 short 1.2 (1200 s), fg 1.1 (1800 s);  Qwen-Image short 1.2 (600 s), fg 0.9 (1800 s)
#   A100 Wan2.2 short 1.05,         fg 0.9;           Qwen-Image short 1.0,         fg 0.7
# (originals with full metadata in ae/gpu/inputs/<platform>-<model>/).
# Pass the outputs to 07_policy_runs.sh as WL_SHORT / WL_FG, with COST.
PLATFORM="${PLATFORM:-h20}"
[ "$PLATFORM" = a100 ] && export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
source "$(dirname "$0")/common.sh"
MODEL="${MODEL:-wan}"
case "$MODEL" in
  wan)  COST="${COST:-$RESULTS/cost-model/wan22-ti2v-5b-local}"; DUR_SHORT=1200; TL_SHORT="${TL_SHORT:-1.2}"; TL_FG="${TL_FG:-1.1}" ;;
  qwen) COST="${COST:-$RESULTS/cost-model/qwen-image-local}"; DUR_SHORT=600;  TL_SHORT="${TL_SHORT:-1.2}"; TL_FG="${TL_FG:-0.9}" ;;
  *) echo "unknown MODEL=$MODEL (wan|qwen)"; exit 2 ;;
esac
[ -d "$COST" ] || { echo "cost model $COST not found: run 05_cost_model.sh or set COST"; exit 2; }
OUTW="${OUTW:-$RESULTS/workloads}"
log "generating $MODEL workloads for $NUM_GPUS GPUs from $COST -> $OUTW"
gen_workload "$MODEL" short "$DUR_SHORT" "$TL_SHORT" "$OUTW/${MODEL}_short_tl${TL_SHORT}_seed42.json" "$COST" || exit 1
gen_workload "$MODEL" foreground_burst 1800 "$TL_FG" "$OUTW/${MODEL}_fg_tl${TL_FG}_seed42.json" "$COST" || exit 1
ls -la "$OUTW"
