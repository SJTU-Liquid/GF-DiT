#!/usr/bin/env bash
# Shared settings for the GPU experiments. Every variable can be overridden from
# the environment, e.g.  GPUS=0,1 NUM_GPUS=2 bash ae/gpu/04_serve_smoke.sh
set -uo pipefail

AE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$AE_DIR/.." && pwd)"
cd "$REPO"

# Python environment that has vllm 0.17.0 + this repo + gfc installed.
# In the GPU Docker image this is /opt/venv/bin.
export VENV_BIN="${VENV_BIN:-$(dirname "$(command -v python3)")}"
export GPUS="${GPUS:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"$GPUS")}"
export PORT="${PORT:-8098}"
export RESULTS="${RESULTS:-$AE_DIR/gpu/results}"
# Set HF_HUB_OFFLINE=1 once the weights are in the HF cache: vLLM-Omni otherwise
# checks the Hub on every server start and can hang behind a firewall.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export LD_LIBRARY_PATH="${KEEP_LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}"

WAN_MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
QWEN_MODEL="Qwen/Qwen-Image"
WAN_SERVE_ARGS="--boundary-ratio 0.875 --flow-shift 5.0"

# SP sizes the elastic EDF policy may choose from: powers of two up to NUM_GPUS.
default_sp_sizes() {
  local s=1 out=""
  while [ "$s" -le "$NUM_GPUS" ]; do out="${out:+$out,}$s"; s=$((s * 2)); done
  echo "$out"
}
export SP_SIZES="${SP_SIZES:-$(default_sp_sizes)}"

log() { echo "=== [$(date '+%F %T')] $*"; }

# Wait until GPUs in $GPUS are idle (< 2 GB used), at most one hour.
gpu_gate() {
  local i busy
  for i in $(seq 1 360); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | awk '$1 > 2048' | wc -l)
    [ "$busy" = 0 ] && return 0
    sleep 10
  done
  log "GPU gate timeout"; return 1
}

# Legacy baseline: stock vLLM-Omni fixed pipeline (no --enable-runtime-v2) with
# Ulysses SP over all GPUs, benchmarked like the Plectra runs.
#   run_legacy <model> <workload.json> <out_dir> <task> <backend> [serve args...]
run_legacy() {
  local model="$1" wl="$2" out="$3" task="$4" backend="$5"; shift 5
  gpu_gate || return 1
  mkdir -p "$out"
  log "LEGACY $model sp=$NUM_GPUS $wl -> $out"
  CUDA_VISIBLE_DEVICES="$GPUS" setsid "$VENV_BIN/vllm" serve "$model" --omni --port "$PORT" \
    --num-gpus "$NUM_GPUS" --ulysses-degree "$NUM_GPUS" --enforce-eager "$@" \
    > "$out/legacy_sp${NUM_GPUS}_server.log" 2>&1 &
  local pid=$! ok=no i
  for i in $(seq 1 300); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = 200 ] && { ok=yes; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 4
  done
  if [ "$ok" = yes ]; then
    timeout "${BENCH_TIMEOUT:-9000}" "$VENV_BIN/python" benchmarks/diffusion/diffusion_benchmark_serving.py \
      --backend "$backend" --task "$task" --model "$model" --dataset mixed_priority --workload-json "$wl" \
      --num-prompts 100000 --max-concurrency 256 --port "$PORT" --guidance-scale 4.0 \
      --output-file "$out/legacy_sp${NUM_GPUS}_result.json" > "$out/legacy_sp${NUM_GPUS}_bench.log" 2>&1
    log "legacy bench exit=$?"
  else
    log "legacy server failed to start; see $out/legacy_sp${NUM_GPUS}_server.log"
  fi
  kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null; sleep 5
}

# Plectra runs: one server per policy token (see scripts/sweep_policies.sh).
#   run_plectra <model> <workload.json> <out_dir> <cost_model_dir> <task> <backend> "<policies>" [serve args]
run_plectra() {
  local model="$1" wl="$2" out="$3" cost="$4" task="$5" backend="$6" policies="$7"; shift 7
  gpu_gate || return 1
  log "PLECTRA $policies sp=$SP_SIZES $wl -> $out"
  GPUS="$GPUS" NUM_GPUS="$NUM_GPUS" SP_SIZES="$SP_SIZES" WORKLOAD="$wl" NUM_PROMPTS=100000 MAX_CONC=256 \
    POLICIES="$policies" COST_MODEL_DIR="$cost" MODEL="$model" TASK="$task" BENCH_BACKEND="$backend" \
    SERVE_EXTRA_ARGS="$*" OUT_DIR="$out" BENCH_TIMEOUT="${BENCH_TIMEOUT:-9000}" PORT="$PORT" VENV_BIN="$VENV_BIN" \
    bash scripts/sweep_policies.sh
}
