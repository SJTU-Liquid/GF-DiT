#!/usr/bin/env bash
# Run gfc benchmarks on the three diffusion-serving presets.
#
# Env vars:
#   GFC_TORCHRUN  path to torchrun (default: torchrun on PATH)
#   GFC_GPUS      CUDA_VISIBLE_DEVICES list (default: 1,2,3,4 to avoid GPU 0)
#   GFC_NPROC     nproc_per_node (default: 4)
# Exits non-zero if any preset fails (the failing ranks' stderr is printed).
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

torchrun="${GFC_TORCHRUN:-$(command -v torchrun || true)}"
[ -n "$torchrun" ] && [ -x "$torchrun" ] || { echo "run_diffusion_shapes.sh: no torchrun found; set GFC_TORCHRUN" >&2; exit 1; }
gpus="${GFC_GPUS:-1,2,3,4}"
nproc="${GFC_NPROC:-4}"
iters="${GFC_ITERS:-500}"
warmup="${GFC_WARMUP:-50}"

rc=0
for preset in small_ctrl mid_act large_latent; do
    echo "=== preset=$preset nproc=$nproc ==="
    rm -rf /tmp/gfc_run
    CUDA_VISIBLE_DEVICES="$gpus" "$torchrun" --standalone --nproc_per_node="$nproc" \
        --redirects=3 --log-dir /tmp/gfc_run \
        benchmarks/bench_symm_collectives.py \
        --diffusion-preset "$preset" \
        --iters "$iters" --warmup "$warmup" \
        --compare-nccl-reference \
        --validate && ok=1 || ok=0
    for f in /tmp/gfc_run/*/attempt_0/0/stdout.log; do
        [ -f "$f" ] && cat "$f"
    done
    if [ "$ok" = 0 ]; then
        echo "=== preset=$preset FAILED; rank stderr: ==="
        tail -n 20 /tmp/gfc_run/*/attempt_0/*/stderr.log 2>/dev/null || true
        rc=1
    fi
    echo
done
exit "$rc"
