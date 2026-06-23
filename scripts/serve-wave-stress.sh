#!/usr/bin/env bash
# Launch the runtime_v2 wave-stress policy serve setup.
#
# Usage:
#   bash scripts/serve-wave-stress.sh
#
# Everything the policy needs is built DYNAMICALLY at runtime:
#   - stress_g01   ranks (0,1)   sp=2  -- registered by policy via topology.ensure_group
#   - stress_g12   ranks (1,2)   sp=2  -- registered by policy via topology.ensure_group
#   - stress_gfull ranks (0..3)  sp=4  -- declared statically below ONLY so the worker
#                                         pool has a primary session at startup;
#                                         the policy ensure_groups the same id/spec
#                                         idempotently
#
# Worker side materializes each group's _FixedParallelSession lazily on first
# dispatch via _ensure_group_spec, so adding G01/G12 from the policy at init
# time requires no NCCL handshake under the GFC backend.
#
# What this exercises:
#   - first warmup_reqs (default 5) requests run normally on stress_gfull
#     (the policy self-routes warmup tasks to its own aux_group_id)
#   - request 6..9 are held until 4 are queued, then activated as a wave:
#       wave[0] -> "cross_g01"  -> DIT chunks on dynamically-built G(0,1) (sp=2)
#       wave[1] -> "cross_g12"  -> DIT chunks on dynamically-built G(1,2) (sp=2)
#       wave[2] -> "cross_full" -> DIT chunks on stress_gfull            (sp=4)
#       wave[3] -> "pingpong"   -> DIT chunks rotate sp4 -> sp2 -> sp2 -> sp4
#   - wave DIT chunks are dispatched round-robin across the 4 wave requests,
#     forcing rank 1 (shared by all three SP groups) to switch SP session
#     on every consecutive task
#   - pingpong's same-request group rotation triggers automatic reshard via
#     scheduler's _migrate_worker_local_input_for_task
set -euo pipefail

vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
      --port 8098 \
      --num-gpus 4 \
      --enable-runtime-v2 \
      --runtime-v2-scheduler-policy wave_stress \
      --runtime-v2-collective-backend gfc \
      --runtime-v2-wave-stress-warmup-reqs 1 \
      --runtime-v2-wave-stress-wave-size 4 \
      --runtime-v2-groups-json '[
        {"group_id":"stress_gfull","ranks":[0,1,2,3],"tp":1,"sp":4,
         "ulysses_degree":4,"ring_degree":1,"cfg":1,
         "supported_task_kinds":["text_encode","dit_prepare","timestep_prepare","dit_step_chunk","vae_decode","finalize"]}
      ]' \
      --boundary-ratio 0.875 \
      --flow-shift 5.0 --enforce-eager
