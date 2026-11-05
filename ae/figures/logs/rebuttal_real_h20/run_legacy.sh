#!/bin/bash
# Rebuttal real-H20 runs on the Wan short workload, queued serially (6 runs).
set -uo pipefail
cd /home/LOCAL/shixuan/xinwei/vllm-omni-private
export LD_LIBRARY_PATH=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=/home/LOCAL/shixuan/xinwei/.venv/bin
G=0,1,2,3; P=8198; M=Wan-AI/Wan2.2-TI2V-5B-Diffusers
WO=out/stress/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json
W2=out/stress/h20bat_short_tl1.20_a2.0_2.0_2.0_seed42.json
R=out/rebuttal
X="--boundary-ratio 0.875 --flow-shift 5.0"
log(){ echo "=== [$(date '+%F %T')] $*"; }
gpu_gate(){  # wait (up to 1 h) until each of GPUs 0-3 uses < 2 GB
  for i in $(seq 1 360); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | awk '$1>2048' | wc -l)
    [ "$busy" = 0 ] && return 0; sleep 10
  done
  log "GPU gate timeout"; return 1
}
sweep(){  # $1=workload $2=outdir $3=policies $4=sp_sizes
  gpu_gate || return; mkdir -p "$2"; log "SWEEP $3 sp=$4 $1 -> $2"
  GPUS=$G NUM_GPUS=4 SP_SIZES="$4" WORKLOAD="$1" NUM_PROMPTS=100000 MAX_CONC=256 \
    POLICIES="$3" COST_MODEL_DIR=cost-model/wan22-ti2v-5b-fullrange MODEL=$M TASK=t2v \
    BENCH_BACKEND=v1/videos SERVE_EXTRA_ARGS="$X" OUT_DIR="$2" BENCH_TIMEOUT=9000 PORT=$P VENV_BIN=$V \
    bash scripts/sweep_policies_rebuttal.sh
}
legacy(){  # $1=workload $2=outdir; vllm-omni-legacy code, SP4 Ulysses, as in out/legacy_baseline_sp4
  gpu_gate || return; mkdir -p "$2"; log "LEGACY $1 -> $2"
  CUDA_VISIBLE_DEVICES=$G PYTHONPATH=/home/LOCAL/shixuan/xinwei/vllm-omni-legacy setsid nohup \
    $V/vllm serve $M --omni --port $P --num-gpus 4 --ulysses-degree 4 $X --enforce-eager \
    > "$2/legacy_sp4_server.log" 2>&1 &
  S=$!; ok=no
  for i in $(seq 1 300); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$P/health)" = 200 ] && { ok=yes; break; }
    kill -0 $S 2>/dev/null || break; sleep 4
  done
  if [ $ok = yes ]; then
    log "legacy healthy -> bench"
    timeout 9000 $V/python benchmarks/diffusion/diffusion_benchmark_serving.py --backend v1/videos --task t2v \
      --model $M --dataset mixed_priority --workload-json "$1" --num-prompts 100000 --max-concurrency 256 \
      --port $P --guidance-scale 4.0 --output-file "$2/legacy_sp4_result.json" > "$2/legacy_sp4_bench.log" 2>&1
    log "legacy bench exit=$?"
  else
    log "legacy server unhealthy"
  fi
  kill -TERM -- -$S 2>/dev/null; sleep 20; kill -KILL -- -$S 2>/dev/null; sleep 10
}
log "REBUTTAL START"
legacy $WO $R/legacy_orig
sweep  $WO $R/short_orig "sjf_sp1" 1,2,4
sweep  $WO $R/short_orig_edfsp1 "edf_best_fit" 1
sweep  $W2 $R/short_a2 "srtf_sp1 edf_best_fit" 1,2,4
legacy $W2 $R/legacy_a2
log "REBUTTAL COMPLETE"
