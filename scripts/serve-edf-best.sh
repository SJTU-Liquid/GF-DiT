#!/usr/bin/env bash
# EDF best-fit + GFC on 4 GPUs (Wan2.2-TI2V-5B). Run from the repository root.
# The cost model must match the serving GPUs (profile with ae/gpu/05_cost_model.sh);
# the default is the paper's H20 cost model.
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
      --port 8098 \
      --num-gpus 4 \
      --enable-runtime-v2 \
      --runtime-v2-scheduler-policy edf_best_fit \
      --runtime-v2-collective-backend gfc \
      --runtime-v2-gfc-max-collective-mb 1024 \
      --runtime-v2-cost-model-dir "${COST_MODEL_DIR:-ae/gpu/inputs/h20-wan/cost-model}" \
      --runtime-v2-edf-greedy-sp-sizes 1,2,4 \
      --boundary-ratio 0.875 \
      --flow-shift 5.0 \
      --enforce-eager