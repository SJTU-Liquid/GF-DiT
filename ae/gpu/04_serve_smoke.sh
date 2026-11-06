#!/usr/bin/env bash
# G4 (getting started on GPUs): serve Wan2.2-TI2V-5B with the Plectra runtime
# (EDF best-fit + GFC) and generate one short video. Needs >= 2 GPUs with
# >= 48 GB each and the Wan2.2 weights (downloaded on first start, ~30 GB). ~10 min.
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/serve_smoke"; mkdir -p "$OUT"
port_in_use "$PORT" && { log "FAILED: port $PORT already in use; set PORT"; exit 1; }
CUDA_VISIBLE_DEVICES="$GPUS" setsid "$VENV_BIN/vllm" serve "$WAN_MODEL" --omni --port "$PORT" \
  --num-gpus "$NUM_GPUS" --enforce-eager $WAN_SERVE_ARGS \
  --enable-runtime-v2 --runtime-v2-scheduler-policy edf_best_fit \
  --runtime-v2-collective-backend gfc --runtime-v2-gfc-max-collective-mb 1024 \
  --runtime-v2-cost-model-dir ae/sim/cost-model/wan22-ti2v-5b-h20 --runtime-v2-edf-greedy-sp-sizes "$SP_SIZES" \
  > "$OUT/server.log" 2>&1 &
PID=$!
trap 'stop_session "$PID"' EXIT
wait_healthy "$PID" "$PORT" 300 4; hs=$?
if [ "$hs" != 0 ]; then
  log "FAILED: $(health_failure "$hs"); see $OUT/server.log"
  tail -5 "$OUT/server.log"; exit 1
fi
log "server healthy; submitting one request"
ID=$(curl -sS -X POST "http://localhost:$PORT/v1/videos" -F "prompt=An orange cat wearing VR glasses." \
  -F seconds=3 -F size=832x480 -F fps=16 -F num_inference_steps=20 -F guidance_scale=4.0 -F seed=42 | \
  "$VENV_BIN/python" -c 'import json,sys; print(json.load(sys.stdin)["id"])')
[ -n "$ID" ] || { log "FAILED: request was not accepted; see $OUT/server.log"; exit 1; }
S=""
for i in $(seq 1 450); do  # up to 15 min
  S=$(curl -sS "http://localhost:$PORT/v1/videos/$ID" | "$VENV_BIN/python" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [ "$S" = completed ] || [ "$S" = failed ] && break; sleep 2
done
[ "$S" = completed ] || { log "FAILED: request status '$S'; see $OUT/server.log"; exit 1; }
curl -sS -L "http://localhost:$PORT/v1/videos/$ID/content" -o "$OUT/sample.mp4" && [ -s "$OUT/sample.mp4" ] \
  || { log "FAILED: could not download the video"; exit 1; }
log "OK: $OUT/sample.mp4 ($(stat -c %s "$OUT/sample.mp4") bytes); scheduler trace in $OUT/server.log"
grep -m3 -E 'runtime_v2 runner started|policy=' "$OUT/server.log"
