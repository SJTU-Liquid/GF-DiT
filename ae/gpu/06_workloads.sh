#!/usr/bin/env bash
# G6: generate the two Wan2.2 workloads (short, foreground burst) for YOUR
# platform from its cost model, with the same generator settings as the paper
# (Sec. 6.1; see ae/sim/workloads/*.json "metadata" for the H20 originals).
source "$(dirname "$0")/common.sh"
COST="${COST:-cost-model/wan22-ti2v-5b-local}"
OUTW="${OUTW:-$RESULTS/workloads}"; mkdir -p "$OUTW"
gen() {  # profile duration target_load
  "$VENV_BIN/python" benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py \
    --out "$OUTW/wan_${1}_tl${3}_seed42.json" --cost-model-dir "$COST" --profile "$1" --duration-s "$2" \
    --num-gpus "$NUM_GPUS" --target-load "$3" --alpha-s 2.0 --alpha-m 2.5 --alpha-l 3.5 \
    --deadline-constant-ms 5000 --seed 42
}
gen short 1200 1.2
gen foreground_burst 1800 1.1
ls -la "$OUTW"
