#!/usr/bin/env bash
# G2 (Table 1, Sec. 4): NCCL subgroup setup cost vs. GFC group registration.
# Paper: 8x H20 for NCCL (group sizes 2,4,6,8), 2x H20 for GFC. Measures the even
# group sizes up to NUM_GPUS, so the full Table 1 needs GPUS=0,1,2,3,4,5,6,7. ~5 min.
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/group_setup"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPUS" "$VENV_BIN/torchrun" --standalone --nproc_per_node="$NUM_GPUS" \
  benchmarks/distributed/nccl_subgroup_cost.py --sizes "$(seq -s, 2 2 "$NUM_GPUS")" --churn-count 0 \
  --out "$OUT/nccl_subgroup_cost.json" \
  || { log "FAILED: NCCL subgroup benchmark"; exit 1; }
CUDA_VISIBLE_DEVICES="$(cut -d, -f1,2 <<<"$GPUS")" "$VENV_BIN/torchrun" --standalone --nproc_per_node=2 \
  benchmarks/distributed/gfc_group_cost.py --out "$OUT/gfc_group_cost.json" \
  || { log "FAILED: GFC group-cost benchmark"; exit 1; }
"$VENV_BIN/python" ae/figures/scripts/summarize_group_setup.py \
  --nccl "$OUT/nccl_subgroup_cost.json" --gfc "$OUT/gfc_group_cost.json" --out "$OUT/table1_group_setup.md"
