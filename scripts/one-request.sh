BASE_URL=http://localhost:8098

# 提交任务
curl -sS -X POST "${BASE_URL}/v1/videos" -H "Accept: application/json" \
  -F "prompt=An orange cat wearing VR glasses." \
  -F "seconds=5" \
  -F "size=832x480" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42" | tee /tmp/disagg_create.json
VID=$(jq -r '.id' /tmp/disagg_create.json)
# 轮询
while :; do
  S=$(curl -sS "${BASE_URL}/v1/videos/${VID}" | jq -r '.status')
  echo "status=$S"
  [ "$S" = completed ] && break
  [ "$S" = failed ]    && { curl -sS "${BASE_URL}/v1/videos/${VID}" | jq .; exit 1; }
  sleep 2
done
# 下载
curl -sS -L "${BASE_URL}/v1/videos/${VID}/content" -o adaptive.mp4
echo "saved: adaptive.mp4"