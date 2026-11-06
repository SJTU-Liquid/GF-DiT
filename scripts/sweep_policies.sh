#!/bin/bash
# Sweep scheduling policies on a single GPU host: one serving run per policy,
# bench against the same workload, save server log + benchmark result JSON.
#
# Usage:
#   scripts/sweep_policies.sh                          # 4-GPU defaults: paper's H20 Wan2.2 short workload
#   POLICIES="edf_best_fit srtf_sp1" scripts/sweep_policies.sh
#   GPUS=0,3 NUM_GPUS=2 SP_SIZES=1,2 WORKLOAD=<workload.json> COST_MODEL_DIR=<dir> \
#       POLICIES="edf_best_fit" scripts/sweep_policies.sh
# ae/gpu/07_policy_runs.sh drives this script for the paper's experiments.
#
# Env knobs (with defaults):
#   GPUS=0,1,2,3            # CUDA_VISIBLE_DEVICES
#   NUM_GPUS=4              # --num-gpus
#   SP_SIZES=1,2,4          # --runtime-v2-edf-greedy-sp-sizes (edf policies only)
#   PORT=8098
#   WORKLOAD=ae/gpu/inputs/h20-wan/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json
#   NUM_PROMPTS=100000      # cap on requests taken from WORKLOAD (default: all)
#   MAX_CONC=256
#   POLICIES="edf_best_fit edf_greedy fcfs srtf"
#     Tokens may carry a topology suffix: fcfs_sp1 / srtf_sp1 give NUM_GPUS
#     independent SP=1 lanes (max concurrency); fcfs_sp2 / srtf_sp2 give
#     ceil(NUM_GPUS/2) SP=2 lanes (the middle baseline); plain fcfs / srtf (or
#     _sp4) give one static SP=NUM_GPUS group (max per-request parallelism).
#     The runtime policy name is the token minus the suffix; output files keep
#     the full token so variants don't overwrite each other.
#   OUT_DIR=/tmp                              # where server/bench logs land
#   BENCH_TIMEOUT=9000                        # per-policy bench wall-clock cap (s)
#   STATIC_SP_GROUP_FOR_FCFS_SRTF=1           # if set, give fcfs/srtf a single static sp=NUM_GPUS group (no cost model)
#   COST_MODEL_DIR=ae/gpu/inputs/h20-wan/cost-model   # must match the GPUs and WORKLOAD
#
# Per policy this script:
#   1. Spawns vllm serve in background with the policy applied.
#   2. Waits for /health to return 200 (timeout 10 min; fails early if the server
#      exits or keeps answering 5xx).
#   3. Runs diffusion_benchmark_serving.py against the configured workload.
#   4. Stops the server's own session (the server and the workers it spawned;
#      no other process on the machine is touched).
#
# Exit status: 0 only if every policy produced a result JSON. A policy fails if
# the port is already in use, the server never becomes healthy, the benchmark
# exits non-zero (incl. BENCH_TIMEOUT), no result JSON is written, or the server
# died, became unhealthy or logged a worker failure during the benchmark, or no
# request completed; failures are listed at the end and the script exits 1.
# On Ctrl-C / SIGTERM / SIGHUP the current benchmark and server are stopped and
# the script dies from the same signal (so a caller running it in the foreground
# sees the interruption). ae/gpu/07_policy_runs.sh runs it in the background and
# stops it with SIGTERM when 07 itself is interrupted.
#
# Outputs (one set per policy P):
#   $OUT_DIR/sweep_${P}_server.log
#   $OUT_DIR/sweep_${P}_bench.log
#   $OUT_DIR/sweep_${P}_result.json
set -uo pipefail

cd "$(dirname "$0")/.."
source scripts/server_lifecycle.sh

GPUS="${GPUS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
SP_SIZES="${SP_SIZES:-1,2,4}"
PORT="${PORT:-8098}"
WORKLOAD="${WORKLOAD:-ae/gpu/inputs/h20-wan/h20bat_short_tl1.20_a2.0_2.5_3.5_seed42.json}"
NUM_PROMPTS="${NUM_PROMPTS:-100000}"
MAX_CONC="${MAX_CONC:-256}"
POLICIES="${POLICIES:-edf_best_fit edf_greedy fcfs srtf}"
OUT_DIR="${OUT_DIR:-/tmp}"
# Per-policy benchmark wall-clock cap (s); the paper's runs used 9000.
BENCH_TIMEOUT="${BENCH_TIMEOUT:-9000}"
COST_MODEL_DIR="${COST_MODEL_DIR:-ae/gpu/inputs/h20-wan/cost-model}"
STATIC_SP_GROUP_FOR_FCFS_SRTF="${STATIC_SP_GROUP_FOR_FCFS_SRTF:-1}"
# venv layout differs between machines; default to in-repo .venv, override for
# shared venv outside the repo (e.g. VENV_BIN=/path/to/shared/.venv/bin).
VENV_BIN="${VENV_BIN:-.venv/bin}"

# GFC autotune (opt-in via GFC_AUTOTUNE=1). Picks a tuned collective table by
# world size so the GFC backend dispatches the best kernel per (group_size,
# slice_bytes) instead of the default fused path. gfc reads it via the
# SYMM_COLL_AUTOTUNE_CONFIG env; only affects --runtime-v2-collective-backend gfc.
# Tables: per-group-size rules each come from that group's own-world autotune
# run -- elastic{4,8} cover sub-groups (gs2/4[/8]) for elastic EDF, 2r covers
# the world=2 single group. Default OFF so prior runs stay comparable.
if [ "${GFC_AUTOTUNE:-0}" = "1" ]; then
  AUTOTUNE_DIR="${AUTOTUNE_DIR:-GroupFree-Collective/benchmarks}"
  case "$NUM_GPUS" in
    8) _autotune_file="$AUTOTUNE_DIR/autotune_h20_elastic8.json";;
    4) _autotune_file="$AUTOTUNE_DIR/autotune_h20_elastic4.json";;
    2) _autotune_file="$AUTOTUNE_DIR/autotune_h20_2r.json";;
    *) _autotune_file="";;
  esac
  if [ -n "$_autotune_file" ] && [ -f "$_autotune_file" ]; then
    export SYMM_COLL_AUTOTUNE_CONFIG="$_autotune_file"
    echo "  GFC autotune ENABLED: $_autotune_file"
  else
    echo "  GFC autotune requested but no table for NUM_GPUS=$NUM_GPUS under $AUTOTUNE_DIR (default dispatch)"
  fi
fi

# Model / task / backend knobs. Defaults serve Wan2.2 video (--backend v1/videos);
# override for Qwen Image (here with the paper's H20 inputs):
#   MODEL=Qwen/Qwen-Image TASK=t2i BENCH_BACKEND=openai SERVE_EXTRA_ARGS="" \
#     COST_MODEL_DIR=ae/gpu/inputs/h20-qwen/cost-model \
#     WORKLOAD=ae/gpu/inputs/h20-qwen/qwenbat_short_tl1.20_seed42.json scripts/sweep_policies.sh
# SERVE_EXTRA_ARGS holds model-specific serve flags (Wan: --boundary-ratio/--flow-shift;
# Qwen needs none).
MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
TASK="${TASK:-t2v}"
BENCH_BACKEND="${BENCH_BACKEND:-v1/videos}"
# Use ${VAR+x} test (not ${VAR:-default}) so callers can pass an explicit
# empty string to mean "no extra args" -- needed for Qwen where the Wan
# --boundary-ratio/--flow-shift defaults are inappropriate. With ${:-},
# SERVE_EXTRA_ARGS="" would silently revert to the Wan defaults.
if [ "${SERVE_EXTRA_ARGS+x}" != "x" ]; then
  SERVE_EXTRA_ARGS="--boundary-ratio 0.875 --flow-shift 5.0"
fi

# One DiT group spanning all ranks at SP=NUM_GPUS (static sp4 baseline).
static_sp_group_json() {
  local ranks=""
  for ((r=0; r<NUM_GPUS; r++)); do
    if [ -z "$ranks" ]; then ranks="$r"; else ranks="$ranks,$r"; fi
  done
  printf '[{"group_id":"g0","ranks":[%s],"tp":1,"sp":%d,"ulysses_degree":%d,"ring_degree":1,"cfg":1,"supported_task_kinds":["text_encode","dit_prepare","timestep_prepare","dit_step_chunk","vae_decode","finalize"]}]' \
    "$ranks" "$NUM_GPUS" "$NUM_GPUS"
}

# NUM_GPUS independent SP=1 lanes (one rank each). fcfs/srtf bind each request
# to a single lane via the earliest-free heuristic, so NUM_GPUS requests run
# concurrently each on one rank -- the max-concurrency / min-per-request-parallelism
# baseline opposite static sp4.
sp1_groups_json() {
  local out="["
  for ((r=0; r<NUM_GPUS; r++)); do
    [ "$r" -gt 0 ] && out+=","
    out+=$(printf '{"group_id":"g%d","ranks":[%d],"tp":1,"sp":1,"ulysses_degree":1,"ring_degree":1,"cfg":1,"supported_task_kinds":["text_encode","dit_prepare","timestep_prepare","dit_step_chunk","vae_decode","finalize"]}' "$r" "$r")
  done
  out+="]"
  printf '%s' "$out"
}

# ceil(NUM_GPUS/2) independent SP=2 lanes (two consecutive ranks each). For
# NUM_GPUS=4: groups [0,1] and [2,3]. fcfs/srtf bind each request to one SP2
# lane -- the middle baseline between sp1 (max concurrency) and sp4 (one big
# group / max per-request parallelism). An odd trailing rank gets its own SP1.
sp2_groups_json() {
  local out="[" first=1 g=0 r=0 r2
  while [ "$r" -lt "$NUM_GPUS" ]; do
    [ "$first" -eq 1 ] || out+=","; first=0
    r2=$((r+1))
    if [ "$r2" -ge "$NUM_GPUS" ]; then
      out+=$(printf '{"group_id":"g%d","ranks":[%d],"tp":1,"sp":1,"ulysses_degree":1,"ring_degree":1,"cfg":1,"supported_task_kinds":["text_encode","dit_prepare","timestep_prepare","dit_step_chunk","vae_decode","finalize"]}' "$g" "$r")
      r=$((r+1))
    else
      out+=$(printf '{"group_id":"g%d","ranks":[%d,%d],"tp":1,"sp":2,"ulysses_degree":2,"ring_degree":1,"cfg":1,"supported_task_kinds":["text_encode","dit_prepare","timestep_prepare","dit_step_chunk","vae_decode","finalize"]}' "$g" "$r" "$r2")
      r=$((r+2))
    fi
    g=$((g+1))
  done
  out+="]"
  printf '%s' "$out"
}

# OUT_DIR may be a not-yet-existing nested path (e.g. a per-profile subdir from
# run_h20_battery.sh); create it so the server/bench log redirects below don't
# fail with "No such file or directory" (which silently skips the policy).
mkdir -p "$OUT_DIR"

FAILED=()
# fail_policy <reason>: record a failure of policy $P (also as sweep_<P>_failed.txt,
# which ae/gpu/summarize_runs.py reports).
fail_policy() { FAILED+=("$P: $1"); echo "$1" > "$OUT_DIR/sweep_${P}_failed.txt"; }
SPID=""   # session leader of the running server
BPID=""   # running benchmark (timeout wrapper)
# Exit (normal or not): stop benchmark and server, ignoring further signals and a
# closed stdout (e.g. `| tee` killed by the same Ctrl-C) so cleanup completes.
trap 'trap "" INT TERM HUP PIPE; [ -n "$BPID" ] && kill -TERM "$BPID" 2>/dev/null; stop_session "$SPID"' EXIT
# Signals: stop benchmark and server, then die from the same signal.
on_signal() {
  trap "" INT TERM HUP PIPE
  [ -n "$BPID" ] && kill -TERM "$BPID" 2>/dev/null
  echo "  [$(date '+%H:%M:%S')] $1 received: stopping benchmark and server" 2>/dev/null
  stop_session "$SPID" 2>/dev/null; SPID=""
  trap - EXIT "$1"
  kill -"$1" $$
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

for P in $POLICIES; do
  echo "=== [$(date '+%m-%d %H:%M:%S')] POLICY=$P start (GPUS=$GPUS NUM_GPUS=$NUM_GPUS WORKLOAD=$WORKLOAD) ==="
  rm -f "$OUT_DIR/sweep_${P}_server.log" "$OUT_DIR/sweep_${P}_bench.log" "$OUT_DIR/sweep_${P}_result.json" \
        "$OUT_DIR/sweep_${P}_failed.txt"

  # Policy token may carry a topology suffix (_sp1 / _sp4); the actual runtime
  # policy name is the token minus that suffix. Output files stay keyed by the
  # full token so variants don't clobber each other.
  base_policy="$P"
  extra_args=()
  case "$P" in
    fcfs_sp1|srtf_sp1|sjf_sp1)
      base_policy="${P%_sp1}"
      extra_args+=(--runtime-v2-groups-json "$(sp1_groups_json)" --runtime-v2-cost-model-dir "$COST_MODEL_DIR")
      ;;
    fcfs_sp2|srtf_sp2)
      base_policy="${P%_sp2}"
      extra_args+=(--runtime-v2-groups-json "$(sp2_groups_json)" --runtime-v2-cost-model-dir "$COST_MODEL_DIR")
      ;;
    fcfs|srtf|fcfs_sp4|srtf_sp4)
      base_policy="${P%_sp4}"
      if [ "$STATIC_SP_GROUP_FOR_FCFS_SRTF" = "1" ]; then
        extra_args+=(--runtime-v2-groups-json "$(static_sp_group_json)")
      else
        extra_args+=(--runtime-v2-cost-model-dir "$COST_MODEL_DIR" --runtime-v2-edf-greedy-sp-sizes "$SP_SIZES")
      fi
      ;;
    *)
      extra_args+=(--runtime-v2-cost-model-dir "$COST_MODEL_DIR" --runtime-v2-edf-greedy-sp-sizes "$SP_SIZES")
      ;;
  esac

  if port_in_use "$PORT"; then
    echo "  [$(date '+%H:%M:%S')] $P FAILED -- port $PORT already in use (another server?)"
    echo "server not started: port $PORT already in use" > "$OUT_DIR/sweep_${P}_server.log"
    fail_policy "port $PORT in use"; continue
  fi
  # setsid: the server and its workers get their own session, so cleanup below
  # can stop exactly them.
  CUDA_VISIBLE_DEVICES="$GPUS" setsid "$VENV_BIN/vllm" serve "$MODEL" --omni \
    --port "$PORT" --num-gpus "$NUM_GPUS" --enable-runtime-v2 \
    --runtime-v2-scheduler-policy "$base_policy" \
    --runtime-v2-collective-backend gfc --runtime-v2-gfc-max-collective-mb 1024 \
    "${extra_args[@]}" \
    $SERVE_EXTRA_ARGS --enforce-eager \
    > "$OUT_DIR/sweep_${P}_server.log" 2>&1 &
  SPID=$!
  wait_healthy "$SPID" "$PORT" 150 4; hs=$?
  if [ "$hs" = 0 ]; then
    echo "  [$(date '+%H:%M:%S')] $P healthy -> benchmark"
    # Background + wait, so a signal is handled at once (on_signal stops it).
    timeout "$BENCH_TIMEOUT" "$VENV_BIN/python" benchmarks/diffusion/diffusion_benchmark_serving.py \
      --backend "$BENCH_BACKEND" --task "$TASK" --model "$MODEL" --dataset mixed_priority \
      --workload-json "$WORKLOAD" \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONC" --port "$PORT" \
      --output-file "$OUT_DIR/sweep_${P}_result.json" \
      > "$OUT_DIR/sweep_${P}_bench.log" 2>&1 &
    BPID=$!; wait "$BPID"; rc=$?; BPID=""
    echo "  [$(date '+%H:%M:%S')] $P benchmark exit=$rc"
    if [ "$rc" -ne 0 ]; then
      fail_policy "benchmark exit $rc$([ "$rc" = 124 ] && echo " (BENCH_TIMEOUT=${BENCH_TIMEOUT}s)"); see sweep_${P}_bench.log"
    elif [ ! -s "$OUT_DIR/sweep_${P}_result.json" ]; then
      fail_policy "no result JSON; see sweep_${P}_bench.log"
    elif why=$(run_broken_reason "$SPID" "$PORT" "$OUT_DIR/sweep_${P}_server.log" "$OUT_DIR/sweep_${P}_result.json"); then
      # The client records request errors as failed requests and still exits 0.
      fail_policy "$why; see sweep_${P}_server.log"
    fi
  else
    echo "  [$(date '+%H:%M:%S')] $P FAILED -- $(health_failure "$hs")"
    grep -E 'Error|invalid|required|raise ' "$OUT_DIR/sweep_${P}_server.log" 2>/dev/null \
      | grep -viE 'futurewarning|deprecated' | head -4
    fail_policy "$(health_failure "$hs"); see sweep_${P}_server.log"
  fi
  stop_session "$SPID"; SPID=""
  wait_port_free "$PORT" 60 || echo "  WARNING: port $PORT still in use after stopping the server"
  echo "  [$(date '+%H:%M:%S')] $P done, server down"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== [$(date '+%m-%d %H:%M:%S')] SWEEP FINISHED WITH ${#FAILED[@]} FAILURE(S) ==="
  printf '  FAILED %s\n' "${FAILED[@]}"
  exit 1
fi
echo "=== [$(date '+%m-%d %H:%M:%S')] SWEEP COMPLETE ==="
