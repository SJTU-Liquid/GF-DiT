# gfc — Group-Free Symmetric Collectives

A Python + Triton runtime for diffusion / DiT serving that provides subgroup
collectives (`all_gather`, `all2all`, `p2p_put`, `p2p_get`, `barrier`) over
**arbitrary subsets of the global rank set** — without ever calling
`torch.distributed.new_group`.

After a one-time symmetric-memory rendezvous on the world group, every
collective is performed by Triton kernels that load from / store to remote
peer pointers exposed by PyTorch Symmetric Memory. NCCL is used only for the
initial bootstrap handshake; it is **not** on any collective data path.

## Quickstart

```python
# rank 0 / rank 1 / ... — launch with torchrun.
import torch
import torch.distributed as dist
from gfc import (
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
    init_distributed_for_runtime,
)

device = init_distributed_for_runtime()        # init NCCL, set CUDA device
config = SymmetricCollectiveConfig(
        max_group_size=4,
    max_collective_bytes=128 * 1024 * 1024,
)
runtime = SymmetricCollectiveRuntime(config, device=device)

# Register a subgroup. Order matters — `(0, 1, 3)` and `(3, 1, 0)` are
# *different* groups with different group_ids.
g = runtime.register_group((0, 1, 3))

# all_gather across the subgroup. Only ranks in the group participate.
if g.local_index >= 0:
    inp = torch.full((1024,), dist.get_rank(), dtype=torch.bfloat16, device=device)
    out = torch.empty(g.size * 1024, dtype=torch.bfloat16, device=device)
    runtime.all_gather(inp, out, g)
    runtime.stream.synchronize()
    # out[i * 1024 : (i + 1) * 1024] now holds rank `g.ranks[i]`'s input.
```

## Collectives provided

| Method | Direction | Notes |
|--------|-----------|-------|
| `runtime.barrier(g)` | n/a | release/acquire-sys barrier on group `g` |
| `runtime.all_gather(inp, out, g)` | remote → local | pull-model |
| `runtime.all2all(inp, out, g)` | remote → local | pull-model, slice `t` of rank `s`'s input is destined for `g.ranks[t]` |
| `runtime.p2p_put(dst_rank, src)` + `p2p_put_recv` | local → remote | push protocol (sender's `tl.store` to peer comm_buf) |
| `runtime.p2p_get(src_rank, dst)` + `p2p_get_serve` | remote → local | pull protocol |

## Stream model

Single-stream by design. `runtime.stream` is a dedicated CUDA stream owned by
the runtime; all collectives are enqueued onto it under a process-wide
submit lock. External integration via:

```python
ev = torch.cuda.Event(); ev.record(producer_stream)
runtime.wait_for_external(ev)
runtime.all_gather(inp, out, g)
ev_done = runtime.record_event()
consumer_stream.wait_event(ev_done)
```

The runtime does **not** call `torch.cuda.current_stream()` implicitly. v1
is single-stream; concurrent multi-stream submission is a v2 non-goal.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `SYMM_COLL_DEBUG=1` | Enable debug grids, group-consistency checks, watchdog dumps |
| `SYMM_COLL_TIMEOUT_MS=N` | Watchdog timeout override (default: `config.timeout_ms`) |
| `SYMM_COLL_DUMP_SIGNALS=1` | On watchdog fire, dump full signal grids to stderr |
| `SYMM_COLL_USE_TMA=1` | Equivalent to `config.use_tma=True` |
| `SYMM_COLL_USE_COPY_ENGINE=1` | Use host-enqueued `cudaMemcpyAsync` copy-engine path |
| `SYMM_COLL_COPY_SMS=N` | Persistent vec-copy CTA count (default: `config.copy_sms`, 24) |

## Testing

```bash
# Picks up GPUs from $GFC_GPUS (default: 1,2,3,4 so GPU 0 is left alone).
./tests/run_all.sh
```

Individual tests live under `tests/`. All multi-process tests are torchrun
scripts; the harness `tests/_harness.py` initialises NCCL from torchrun env.

## Benchmarks

```bash
torchrun --standalone --nproc_per_node=4 \
    benchmarks/bench_symm_collectives.py \
    --collective all2all --bytes 1048576 --iters 1000 --warmup 100 \
    --dtype bf16 --use-tma 0 \
    --cuda-graph 1 --graph-batch-iters 8 \
    --validate --compare-nccl-reference

# Copy-engine path for large messages.
torchrun --standalone --nproc_per_node=4 \
    benchmarks/bench_symm_collectives.py \
    --collective all2all --bytes 16777216 --iters 200 --warmup 50 \
    --dtype bf16 --copy-engine 1 \
    --cuda-graph 1 --graph-batch-iters 8 \
    --validate --compare-nccl-reference

# Add --cuda-graph 0 when you explicitly want to include Python/kernel-launch
# overhead in the measured GFC path.

# Diffusion preset sweep (small_ctrl / mid_act / large_latent).
./benchmarks/run_diffusion_shapes.sh
```

## Hardware notes

* Tested on 8× NVIDIA H100 80GB on NVLink P2P (SM 9.0).
* Software baseline (matches the spec): `torch==2.10.0+cu130`, `triton==3.6.0`,
  CUDA 13.0. The package itself only depends on `torch>=2.10` and `triton>=3.6`.
* TMA over peer pointers is gated by `tma_probe.probe_tma_supported`. The
  probe passes on the development hardware; on systems where it fails,
  initialising with `use_tma=True` raises `TMAUnsupportedError` (no silent
  fallback) and `use_tma=False` keeps everything working on the vec path.
