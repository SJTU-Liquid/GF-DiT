#!/usr/bin/env bash
# G2 (Table 1, Sec. 4): NCCL subgroup setup cost vs. GFC group registration.
# Paper: 8x H20 for NCCL (size 2..8), 2x H20 for GFC. ~5 min.
source "$(dirname "$0")/common.sh"
mkdir -p "$RESULTS/group_setup"
CUDA_VISIBLE_DEVICES="$GPUS" "$VENV_BIN/torchrun" --standalone --nproc_per_node="$NUM_GPUS" \
  benchmarks/distributed/nccl_subgroup_cost.py --out "$RESULTS/group_setup/nccl_subgroup_cost.json"
CUDA_VISIBLE_DEVICES="${GPUS%%,*},$(cut -d, -f2 <<<"$GPUS")" "$VENV_BIN/torchrun" --standalone --nproc_per_node=2 \
  benchmarks/distributed/gfc_group_cost.py --out "$RESULTS/group_setup/gfc_group_cost.json"
"$VENV_BIN/python" ae/figures/scripts/summarize_group_setup.py \
  --nccl "$RESULTS/group_setup/nccl_subgroup_cost.json" --gfc "$RESULTS/group_setup/gfc_group_cost.json" \
  --out "$RESULTS/group_setup/table1_group_setup.md"
