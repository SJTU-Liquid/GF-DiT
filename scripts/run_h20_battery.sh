#!/bin/bash
# Full mixed-priority policy battery: every staged workload profile x every
# policy, one serving run each. Designed to be launched DETACHED so an SSH /
# tunnel drop mid-run cannot kill it (the h20 tunnel is flaky).
#
# Detached launch on h20 (survives disconnect):
#   cd <repo>
#   mkdir -p out/h20battery
#   setsid nohup bash scripts/run_h20_battery.sh > out/h20battery/driver.log 2>&1 < /dev/null &
#   echo $! > out/h20battery/driver.pid     # remember the pid
#   tail -f out/h20battery/driver.log        # watch; Ctrl-C / disconnect is safe
#
# Reconnect later and check progress:
#   tail -n 40 out/h20battery/driver.log
#   ls out/h20battery/*/sweep_*_result.json
#
# Each (profile, policy) pair: spawn vllm serve, wait /health, run the bench
# against that workload, kill the server. Reuses scripts/sweep_policies.sh for
# the per-policy serve/bench/kill lifecycle; this script just loops profiles.
#
# Env knobs (h20 defaults shown):
#   VENV_BIN=/home/LOCAL/shixuan/xinwei/.venv/bin   # h20 shared venv
#   GPUS=3,4,5,6        # CUDA_VISIBLE_DEVICES (h20: GPU3 is MIG, these map to physical 4-7)
#   NUM_GPUS=4
#   SP_SIZES=1,2,4
#   COST_MODEL_DIR=cost-model/wan22-ti2v-5b-fullrange  # h20-native; must match what the workloads were calibrated with
#   POLICIES="edf_greedy edf_best_fit srtf_sp1 srtf_sp2 srtf_sp4 fcfs_sp1 fcfs_sp2 fcfs_sp4"
#   PORT=8098
#   OUT_ROOT=out/h20battery
#   BENCH_TIMEOUT=9000  # per-policy cap (s); tridentserve's 3600s arrival window needs a high cap on slow h20
#   WORKLOADS=<the 4 staged h20bat_*.json>   # newline/space separated
#
# RUNTIME: each (profile,policy) bench makespan ~= that profile's arrival
# window at load ~1.0, so the full 8-policy x 4-profile matrix is ~19h on h20
# (tridentserve's 3600s window dominates). It is detached, so that is fine --
# but to get partial results sooner, trim POLICIES or WORKLOADS, and note the
# profiles are ordered cheap-first so tridentserve runs last.
set -uo pipefail
cd "$(dirname "$0")/.."

# h20's ~/.bashrc prepends system CUDA 12.8 (/usr/local/cuda-12.8/.../lib) to
# LD_LIBRARY_PATH. Its libnvJitLink.so.12 (12.8.61) shadows the venv's
# pip-bundled 12.9 one and lacks __nvJitLinkGetErrorLogSize_12_9, which the
# pip nvidia-cusparse-cu12 (CUDA 12.9) requires -> vllm crashes on `import
# torch`. The venv bundles all CUDA libs with RPATH, so clear LD_LIBRARY_PATH
# and let each wheel find its own. Override by exporting KEEP_LD_LIBRARY_PATH=1.
if [ "${KEEP_LD_LIBRARY_PATH:-0}" != "1" ]; then
  export LD_LIBRARY_PATH=""
fi

VENV_BIN="${VENV_BIN:-/home/LOCAL/shixuan/xinwei/.venv/bin}"
GPUS="${GPUS:-3,4,5,6}"
NUM_GPUS="${NUM_GPUS:-4}"
SP_SIZES="${SP_SIZES:-1,2,4}"
COST_MODEL_DIR="${COST_MODEL_DIR:-cost-model/wan22-ti2v-5b-fullrange}"
# Model / task / backend. Defaults serve Wan2.2 video. For a Qwen Image battery:
#   MODEL=Qwen/Qwen-Image TASK=t2i BENCH_BACKEND=openai SERVE_EXTRA_ARGS="" \
#     COST_MODEL_DIR=cost-model/qwen-image \
#     WORKLOADS="$(ls out/stress/qwenbat_*.json)" \
#     setsid nohup bash scripts/run_h20_battery.sh > out/h20battery/driver.log 2>&1 < /dev/null &
MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
TASK="${TASK:-t2v}"
BENCH_BACKEND="${BENCH_BACKEND:-v1/videos}"
# Preserve explicit empty string so callers can disable the Wan defaults
# (Qwen image battery passes SERVE_EXTRA_ARGS=""). See note in sweep_policies.sh.
if [ "${SERVE_EXTRA_ARGS+x}" != "x" ]; then
  SERVE_EXTRA_ARGS="--boundary-ratio 0.875 --flow-shift 5.0"
fi
POLICIES="${POLICIES:-edf_greedy edf_best_fit srtf_sp1 srtf_sp2 srtf_sp4 fcfs_sp1 fcfs_sp2 fcfs_sp4}"
PORT="${PORT:-8098}"
OUT_ROOT="${OUT_ROOT:-out/h20battery}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-9000}"
WORKLOADS="${WORKLOADS:-
out/stress/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json
out/stress/h20bat_foreground_burst_tl1.10_a2.0_2.5_3.5_seed42.json
out/stress/h20bat_inflight_burst_tl0.90_a2.0_2.5_3.5_seed42.json
out/stress/h20bat_tridentserve_tl1.09_a2.0_2.5_3.5_seed42.json
}"

mkdir -p "$OUT_ROOT"
echo "=== [$(date '+%F %T')] h20 battery START ==="
echo "  model    : $MODEL (task=$TASK backend=$BENCH_BACKEND)"
echo "  policies : $POLICIES"
echo "  gpus     : $GPUS (num=$NUM_GPUS) sp_sizes=$SP_SIZES"
echo "  venv     : $VENV_BIN"
echo "  costmodel: $COST_MODEL_DIR"
echo "  out_root : $OUT_ROOT"

n_done=0
for WL in $WORKLOADS; do
  if [ ! -f "$WL" ]; then
    echo "=== [$(date '+%F %T')] SKIP (missing workload): $WL ==="
    continue
  fi
  tag=$(basename "$WL" .json)
  od="$OUT_ROOT/$tag"
  mkdir -p "$od"
  echo "=== [$(date '+%F %T')] WORKLOAD $tag -> $od ==="
  # NUM_PROMPTS huge so the whole workload is used regardless of profile size.
  GPUS="$GPUS" NUM_GPUS="$NUM_GPUS" SP_SIZES="$SP_SIZES" \
    WORKLOAD="$WL" NUM_PROMPTS=100000 MAX_CONC=256 \
    POLICIES="$POLICIES" COST_MODEL_DIR="$COST_MODEL_DIR" \
    MODEL="$MODEL" TASK="$TASK" BENCH_BACKEND="$BENCH_BACKEND" SERVE_EXTRA_ARGS="$SERVE_EXTRA_ARGS" \
    OUT_DIR="$od" BENCH_TIMEOUT="$BENCH_TIMEOUT" PORT="$PORT" \
    VENV_BIN="$VENV_BIN" \
    bash scripts/sweep_policies.sh
  n_done=$((n_done+1))
  echo "=== [$(date '+%F %T')] WORKLOAD $tag done ($n_done) ==="
done
echo "=== [$(date '+%F %T')] h20 battery COMPLETE ($n_done workloads) ==="
echo "Results: $OUT_ROOT/*/sweep_*_result.json"
