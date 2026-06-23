#!/usr/bin/env bash
# Baseline: SP=4 single group, fcfs policy, torch backend, enforce_eager.
# Measures "natural" per-chunk time of Wan2.2-TI2V-5B at user's workload
# without any dynamic-group/GFC gymnastics. Compare exec_only_ms in this
# run against wave_stress run to isolate per-call collective overhead.
set -euo pipefail

vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
      --port 8098 \
      --num-gpus 4 \
      --enable-runtime-v2 \
      --runtime-v2-scheduler-policy fcfs \
      --runtime-v2-collective-backend torch \
      --runtime-v2-groups-json '[
        {"group_id":"g0","ranks":[0,1,2,3],"tp":1,"sp":4,
         "ulysses_degree":4,"ring_degree":1,"cfg":1,
         "supported_task_kinds":["text_encode","dit_prepare","timestep_prepare","dit_step_chunk","vae_decode","finalize"]}
      ]' \
      --boundary-ratio 0.875 \
      --flow-shift 5.0 
