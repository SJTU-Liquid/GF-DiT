vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
    --port 8098 \
    --num-gpus 4 \
    --enable-runtime-v2 \
    --runtime-v2-scheduler-policy edf_greedy \
    --runtime-v2-collective-backend gfc \
    --runtime-v2-gfc-max-collective-mb 1024 \
    --runtime-v2-gfc-num-comm-slots 4 \
    --boundary-ratio 0.875 \
    --flow-shift 5.0 --enforce-eager