#!/usr/bin/env bash
# Run gfc benchmarks on the three diffusion-serving presets.
#
# Env vars:
#   GFC_PY        path to a python that imports torch/triton (default: vllm-omni-private venv)
#   GFC_TORCHRUN  path to torchrun (default: vllm-omni-private venv)
#   GFC_GPUS      CUDA_VISIBLE_DEVICES list (default: 1,2,3,4 to avoid GPU 0)
#   GFC_NPROC     nproc_per_node (default: 4)
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

py="${GFC_PY:-/workspace/vllm-omni-private/.venv/bin/python}"
torchrun="${GFC_TORCHRUN:-/workspace/vllm-omni-private/.venv/bin/torchrun}"
gpus="${GFC_GPUS:-1,2,3,4}"
nproc="${GFC_NPROC:-4}"
iters="${GFC_ITERS:-500}"
warmup="${GFC_WARMUP:-50}"

for preset in small_ctrl mid_act large_latent; do
    echo "=== preset=$preset nproc=$nproc ==="
    rm -rf /tmp/gfc_run
    CUDA_VISIBLE_DEVICES="$gpus" "$torchrun" --standalone --nproc_per_node="$nproc" \
        --redirects=3 --log-dir /tmp/gfc_run \
        benchmarks/bench_symm_collectives.py \
        --diffusion-preset "$preset" \
        --iters "$iters" --warmup "$warmup" \
        --compare-nccl-reference \
        --validate || true
    if [ -f /tmp/gfc_run/*/attempt_0/0/stdout.log ]; then
        cat /tmp/gfc_run/*/attempt_0/0/stdout.log
    fi
    echo
done
