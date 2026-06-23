vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
      --port 8098 \
      --num-gpus 3 \
      --enable-runtime-v2 \
      --runtime-v2-scheduler-policy dynamic_step_fcfs \
      --runtime-v2-disaggregate-aux-group-id g_aux \
      --runtime-v2-disaggregate-dit-group-id g_dit_sp1 \
      --runtime-v2-dit-step-schedule '[
        {"start":0,"end":null,"group_id":"g_dit_sp1"}
      ]' \
      --runtime-v2-groups-json '[
        {"group_id":"g_aux","ranks":[0],"tp":1,"sp":1,"cfg":1,
         "supported_task_kinds":["text_encode","vae_decode","finalize"]},
        {"group_id":"g_dit_sp1","ranks":[1],"tp":1,"sp":1,
         "ulysses_degree":1,"ring_degree":1,"cfg":1,
         "supported_task_kinds":["dit_prepare","timestep_prepare","dit_step_chunk"]}
      ]' \
      --boundary-ratio 0.875 \
      --flow-shift 5.0