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
export PATH="$VENV_BIN:$PATH"
export GPUS="${GPUS:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"$GPUS")}"
export PORT="${PORT:-8098}"
RESULTS="${RESULTS:-$AE_DIR/gpu/results}"
mkdir -p "$RESULTS" && RESULTS="$(cd "$RESULTS" && pwd)"
export RESULTS
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

source "$REPO/scripts/server_lifecycle.sh"

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

# mark_failed <dir> <run tag> <reason>: record a run that could not start (busy GPUs
# or port, no workload) so that summarize_runs.py lists it as FAILED; drops the files
# an earlier invocation left for that run.
mark_failed() {
  mkdir -p "$1"
  rm -f "$1/$2_result.json" "$1/$2_bench.log"
  echo "server not started: $3" > "$1/$2_server.log"
  echo "$3" > "$1/$2_failed.txt"
}

# Legacy baseline: the fixed-pipeline vLLM-Omni server (no --enable-runtime-v2) with
# Ulysses SP over all GPUs, benchmarked like the Plectra runs. The paper's Legacy
# numbers come from upstream vLLM-Omni at commit cf328eb8 (the parent of the
# runtime_v2 work), loaded through PYTHONPATH on top of the same venv; set
# LEGACY_PYTHONPATH to such a checkout (the GPU image has one in
# /opt/vllm-omni-legacy). If unset, this repository's code runs with runtime_v2 off.
# Returns non-zero if the server does not start, the benchmark fails, no result
# JSON is written, or the server dies during the benchmark.
#   run_legacy <model> <workload.json> <out_dir> <task> <backend> [serve args...]
LEGACY_PID=""   # session leader of a running Legacy server (for the traps below)
LEGACY_BPID=""  # its running benchmark
SWEEP_PID=""    # running scripts/sweep_policies.sh (run_plectra)
run_legacy() {
  local model="$1" wl="$2" out="$3" task="$4" backend="$5"; shift 5
  local tag="legacy_sp${NUM_GPUS}" rc=1 hs why="" guidance=()
  mkdir -p "$out"
  rm -f "$out/${tag}_result.json" "$out/${tag}_server.log" "$out/${tag}_bench.log" "$out/${tag}_failed.txt"
  gpu_gate || { mark_failed "$out" "$tag" "GPUs $GPUS busy (gpu_gate timeout)"; return 1; }
  if port_in_use "$PORT"; then
    log "LEGACY FAILED: port $PORT already in use"
    mark_failed "$out" "$tag" "port $PORT already in use"
    return 1
  fi
  # As in the paper's runs: guidance scale 4.0 for video (Wan2.2), model default for images.
  [ "$backend" = v1/videos ] && guidance=(--guidance-scale 4.0)
  log "LEGACY $model sp=$NUM_GPUS $wl -> $out (vllm_omni: ${LEGACY_PYTHONPATH:-this repository, runtime_v2 off})"
  {
    # Run from / so that, as for the vllm console script, the cwd is not on sys.path.
    echo "# Legacy code under test: $(cd / && PYTHONPATH="${LEGACY_PYTHONPATH:-}" "$VENV_BIN/python" -c \
      'import vllm_omni; print(vllm_omni.__file__)' 2>&1 | tail -1)"
  } > "$out/${tag}_server.log"
  CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="${LEGACY_PYTHONPATH:-}" setsid "$VENV_BIN/vllm" serve "$model" \
    --omni --port "$PORT" --num-gpus "$NUM_GPUS" --ulysses-degree "$NUM_GPUS" --enforce-eager "$@" \
    >> "$out/${tag}_server.log" 2>&1 &
  LEGACY_PID=$!
  wait_healthy "$LEGACY_PID" "$PORT" 300 4; hs=$?
  if [ "$hs" = 0 ]; then
    timeout "${BENCH_TIMEOUT:-9000}" "$VENV_BIN/python" benchmarks/diffusion/diffusion_benchmark_serving.py \
      --backend "$backend" --task "$task" --model "$model" --dataset mixed_priority --workload-json "$wl" \
      --num-prompts 100000 --max-concurrency 256 --port "$PORT" "${guidance[@]}" \
      --output-file "$out/${tag}_result.json" > "$out/${tag}_bench.log" 2>&1 &
    LEGACY_BPID=$!; wait "$LEGACY_BPID"; rc=$?; LEGACY_BPID=""
    log "legacy bench exit=$rc"
    if [ "$rc" != 0 ]; then
      why="benchmark exit $rc; see ${tag}_bench.log"
    elif [ ! -s "$out/${tag}_result.json" ]; then
      why="no result JSON; see ${tag}_bench.log"; rc=1
    elif why=$(run_broken_reason "$LEGACY_PID" "$PORT" "$out/${tag}_server.log" "$out/${tag}_result.json"); then
      why="$why; see ${tag}_server.log"; rc=1
    else
      why=""
    fi
  else
    why="$(health_failure "$hs"); see ${tag}_server.log"
  fi
  [ -n "$why" ] && log "LEGACY FAILED: $why"
  stop_session "$LEGACY_PID"; LEGACY_PID=""
  wait_port_free "$PORT" 60 || log "WARNING: port $PORT still in use"
  [ "$rc" = 0 ] || echo "$why" > "$out/${tag}_failed.txt"
  return "$rc"
}

# For scripts that run experiments (07): on Ctrl-C / SIGTERM / SIGHUP sent to the
# script, stop a running sweep (stop_runs sends it SIGTERM; it stops its own
# benchmark and server) and a running Legacy benchmark and server, then die from the
# same signal. A signal sent only to a sweep or Legacy process ends that run; 07
# records it as failed and continues with the next run.
stop_runs() {  # stop a running sweep (it cleans up itself) and Legacy server
  if [ -n "$SWEEP_PID" ]; then kill -TERM "$SWEEP_PID" 2>/dev/null; wait "$SWEEP_PID" 2>/dev/null; SWEEP_PID=""; fi
  [ -n "$LEGACY_BPID" ] && kill -TERM "$LEGACY_BPID" 2>/dev/null
  stop_session "$LEGACY_PID"; LEGACY_PID=""
}
install_run_traps() {
  trap 'trap "" INT TERM HUP PIPE; stop_runs' EXIT
  local sig
  for sig in INT TERM HUP; do
    trap "trap '' INT TERM HUP PIPE; log '$sig received: stopping' 2>/dev/null; stop_runs 2>/dev/null; trap - EXIT $sig; kill -$sig \$\$" "$sig"
  done
}

# Generate a workload with the paper's settings (Sec. 6.1): SLO factors alpha_S/M/L
# 2.0/2.5/3.5 and a 5 s deadline allowance for Wan2.2, 1.5/2.0/6.0 and 1 s for
# Qwen-Image; seed 42.
#   gen_workload <wan|qwen> <short|foreground_burst> <duration_s> <target_load> <out.json> <cost_model_dir>
gen_workload() {
  local model="$1" profile="$2" dur="$3" tl="$4" out="$5" cost="$6" alphas
  case "$model" in
    wan)  alphas=(--alpha-s 2.0 --alpha-m 2.5 --alpha-l 3.5 --deadline-constant-ms 5000) ;;
    qwen) alphas=(--alpha-s 1.5 --alpha-m 2.0 --alpha-l 6.0 --deadline-constant-ms 1000) ;;
    *) log "gen_workload: unknown model $model"; return 2 ;;
  esac
  mkdir -p "$(dirname "$out")"
  "$VENV_BIN/python" benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py \
    --out "$out" --adapter "$model" --cost-model-dir "$cost" --profile "$profile" --duration-s "$dur" \
    --num-gpus "$NUM_GPUS" --target-load "$tl" "${alphas[@]}" --seed 42
}

# Plectra runs: one server per policy token (see scripts/sweep_policies.sh);
# returns non-zero if any policy failed.
#   run_plectra <model> <workload.json> <out_dir> <cost_model_dir> <task> <backend> "<policies>" [serve args]
run_plectra() {
  local model="$1" wl="$2" out="$3" cost="$4" task="$5" backend="$6" policies="$7" p; shift 7
  if ! gpu_gate; then
    for p in $policies; do mark_failed "$out" "sweep_$p" "GPUs $GPUS busy (gpu_gate timeout)"; done
    return 1
  fi
  log "PLECTRA $policies sp=$SP_SIZES $wl -> $out"
  GPUS="$GPUS" NUM_GPUS="$NUM_GPUS" SP_SIZES="$SP_SIZES" WORKLOAD="$wl" NUM_PROMPTS=100000 MAX_CONC=256 \
    POLICIES="$policies" COST_MODEL_DIR="$cost" MODEL="$model" TASK="$task" BENCH_BACKEND="$backend" \
    SERVE_EXTRA_ARGS="$*" OUT_DIR="$out" BENCH_TIMEOUT="${BENCH_TIMEOUT:-9000}" PORT="$PORT" VENV_BIN="$VENV_BIN" \
    bash scripts/sweep_policies.sh &
  # Background + wait: the caller's traps run at once and forward the signal (stop_runs).
  SWEEP_PID=$!; wait "$SWEEP_PID"; local rc=$?; SWEEP_PID=""
  return "$rc"
}
