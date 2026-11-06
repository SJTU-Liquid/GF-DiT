#!/usr/bin/env bash
# Run the full gfc test suite. Each multi-process test is launched via
# torchrun. GPU set is selected with $GFC_GPUS (default 1,2,3,4) to avoid
# binding GPU 0 if it is already in use.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# python3 / torchrun of the active environment unless GFC_PY / GFC_TORCHRUN are set.
py="${GFC_PY:-$(command -v python3 || true)}"
torchrun="${GFC_TORCHRUN:-$(command -v torchrun || true)}"
[ -n "$py" ] && [ -x "$py" ] || { echo "run_all.sh: no python3 found; activate the environment with torch, triton and gfc, or set GFC_PY" >&2; exit 1; }
[ -n "$torchrun" ] && [ -x "$torchrun" ] || { echo "run_all.sh: no torchrun found; activate the environment with torch, or set GFC_TORCHRUN" >&2; exit 1; }
gpus="${GFC_GPUS:-1,2,3,4}"

# Number of comma-separated GPUs available.
nallgpus=$(awk -F, '{print NF}' <<<"$gpus")

run_single () {
    echo "=== pytest $1 ==="
    "$py" -m pytest "$1" -q
}

run_torchrun () {
    local nproc="$1" path="$2"
    if [ "$nproc" -gt "$nallgpus" ]; then
        echo "=== SKIP $path (need $nproc GPUs, have $nallgpus from \$GFC_GPUS) ==="
        return 0
    fi
    echo "=== torchrun --nproc_per_node=$nproc $path ==="
    rm -rf /tmp/gfc_run
    CUDA_VISIBLE_DEVICES="$gpus" "$torchrun" \
        --standalone --nproc_per_node="$nproc" \
        --tee=3 --log-dir /tmp/gfc_run "$path"
}

# Phase 1
run_single tests/test_skeleton_unit.py

# Phase 2
run_torchrun 2 tests/test_init.py
run_torchrun 4 tests/test_init.py

# Phase 3
run_torchrun 2 tests/test_remote_copy_kernel.py

# Phase 4
run_torchrun 2 tests/test_barrier_repeated.py
run_torchrun 4 tests/test_subgroup_barrier.py
run_torchrun 4 tests/test_barrier_stress.py
run_torchrun 8 tests/test_barrier_stress.py

# Phase 5
run_torchrun 2 tests/test_all_gather.py
run_torchrun 4 tests/test_all_gather.py

# Phase 6
run_torchrun 2 tests/test_all2all.py
run_torchrun 4 tests/test_all2all.py

# Phase 7
run_torchrun 2 tests/test_p2p.py

# Phase 8
run_torchrun 2 tests/test_cross_kernel_publication.py
run_torchrun 3 tests/test_overlap_order.py
run_torchrun 4 tests/test_complex_groups.py
run_torchrun 8 tests/test_complex_groups.py
run_torchrun 2 tests/test_buffer_reuse.py
run_torchrun 2 tests/test_perf_smoke.py
run_torchrun 4 tests/test_perf_smoke.py

# Phase 10
run_torchrun 2 tests/test_tma_probe.py

# Phase 12 perf (informational)
run_torchrun 4 tests/test_perf_smoke_tma.py
