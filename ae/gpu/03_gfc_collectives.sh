#!/usr/bin/env bash
# G3 (Fig. 8): steady-state GFC vs. NCCL latency for BF16 all2all / all_gather,
# CUDA-graph timing, 1 KB..64 MB per rank (same command as the paper's H20 run).
# Run alone on idle GPUs: concurrent jobs distort the timings. ~30 min on 4 GPUs.
# Exits non-zero if any measurement failed (printed as NA; errors in $OUT.err).
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/gfc_collectives_${NUM_GPUS}gpu.log"; : > "$OUT.err"
SIZES="1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864"
cd GroupFree-Collective
{
echo "START world=$NUM_GPUS cudagraph batch10/w3/i15  $(date)"
for coll in all2all allgather; do
  echo "## $coll"
  for b in $SIZES; do
    o=$(CUDA_VISIBLE_DEVICES="$GPUS" "$VENV_BIN/torchrun" --standalone --nproc_per_node="$NUM_GPUS" \
        benchmarks/bench_symm_collectives.py --collective "$coll" --bytes "$b" --dtype bf16 \
        --cuda-graph 1 --graph-nccl-reference 1 --graph-batch-iters 10 --iters 15 --warmup 3 \
        --compare-nccl-reference 2>>"$OUT.err")
    lat=$(grep -oP 'latency\s+p50=\K[0-9.]+' <<<"$o"); ker=$(grep -oP 'kernel_time\s+p50=\K[0-9.]+' <<<"$o")
    ncl=$(grep -oP 'nccl_reference\s+p50=\K[0-9.]+' <<<"$o"); rat=$(grep -oP 'ratio \K[0-9.]+' <<<"$o")
    printf '%s bytes=%s GFC_lat=%s GFC_ker=%s NCCL=%s gfc/nccl=%s\n' "$coll" "$b" "${lat:-NA}" "${ker:-NA}" "${ncl:-NA}" "${rat:-NA}"
  done
done
echo "=== DONE $(date) ==="
} | tee "$OUT"
nfail=$(grep -c '=NA' "$OUT")
[ "$nfail" = 0 ] || { log "FAILED: $nfail measurement(s) missing; see $OUT.err"; exit 1; }
