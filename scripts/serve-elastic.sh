#!/usr/bin/env bash
# Launch the elastic dynamic-SP serve setup.
#
# Usage:
#   bash scripts/serve-elastic.sh             # default torch backend
#   bash scripts/serve-elastic.sh gfc         # group-free collective backend
#   COLLECTIVE_BACKEND=gfc bash scripts/serve-elastic.sh
#
# The two g_dit_* groups overlap on rank 1 (sp=1 -> sp=2 at step 10), so the
# sp=2 SP subgroup is created on demand. With backend=gfc that registration
# is a logical session and skips torch.distributed.new_group; with backend=
# torch it falls back to the static-group path.
set -euo pipefail

BACKEND="${1:-${COLLECTIVE_BACKEND:-torch}}"
case "$BACKEND" in
  torch|gfc) ;;
  *) echo "unsupported backend '$BACKEND' (expected: torch | gfc)" >&2; exit 2;;
esac

vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
      --port 8098 \
      --num-gpus 3 \
      --enable-runtime-v2 \
      --runtime-v2-scheduler-policy dynamic_step_fcfs \
      --runtime-v2-collective-backend "$BACKEND" \
      --runtime-v2-disaggregate-aux-group-id g_aux \
      --runtime-v2-disaggregate-dit-group-id g_dit_sp1 \
      --runtime-v2-dit-step-schedule '[
        {"start":0,"end":10,"group_id":"g_dit_sp1"},
        {"start":10,"end":null,"group_id":"g_dit_sp2"}
      ]' \
      --runtime-v2-groups-json '[
        {"group_id":"g_aux","ranks":[0],"tp":1,"sp":1,"cfg":1,
         "supported_task_kinds":["text_encode","vae_decode","finalize"]},
        {"group_id":"g_dit_sp1","ranks":[1],"tp":1,"sp":1,
         "ulysses_degree":1,"ring_degree":1,"cfg":1,
         "supported_task_kinds":["dit_prepare","timestep_prepare","dit_step_chunk"]},
        {"group_id":"g_dit_sp2","ranks":[1,2],"tp":1,"sp":2,
         "ulysses_degree":2,"ring_degree":1,"cfg":1,
         "supported_task_kinds":["dit_prepare","timestep_prepare","dit_step_chunk"]}
      ]' \
      --boundary-ratio 0.875 \
      --flow-shift 5.0
