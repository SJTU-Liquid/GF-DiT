"""Phase-10 TMA probe + TMA all_gather equivalence.

Initialises the runtime with ``use_tma=True``. The runtime itself runs the
probe in __init__ and raises :class:`TMAUnsupportedError` if it fails. If
TMA passes, this test then runs a small all_gather (which is now routed
through the TMA path) and verifies the output against the expected pattern.

The probe failing on this hardware is an acceptable v1 outcome: the spec
documents that TMA-over-peer-pointers is unverified. The test exits 0 in
that case after printing the verdict.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_tma_probe.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)
from gfc.tma_probe import TMAUnsupportedError  # noqa: E402

from tests._harness import order_after_caller, rank_print, setup_nccl, teardown  # noqa: E402


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=1 * 1024 * 1024,
        use_tma=True,
    )
    try:
        runtime = SymmetricCollectiveRuntime(config, device=device)
    except TMAUnsupportedError as e:
        rank_print(f"TMA probe FAILED: {e}")
        rank_print("v1 vec-only fallback is correct; this is acceptable per spec.")
        teardown()
        return 0

    rank_print("TMA probe OK; runtime initialised with TMA path active.")

    g = runtime.register_group(tuple(range(world)))
    n_elems = 4096
    inp = torch.full((n_elems,), runtime.rank, dtype=torch.float32, device=device)
    out = torch.empty(world * n_elems, dtype=torch.float32, device=device)

    order_after_caller(runtime)
    runtime.all_gather(inp, out, g)
    runtime.stream.synchronize()

    for i in range(world):
        seg = out[i * n_elems : (i + 1) * n_elems]
        if not torch.all(seg == g.ranks[i]):
            first = int((seg != g.ranks[i]).nonzero(as_tuple=False)[0, 0])
            raise AssertionError(
                f"TMA all_gather seg {i} byte {first} got={float(seg[first])} "
                f"exp={g.ranks[i]}"
            )

    rank_print("ok TMA all_gather equivalence verified")
    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
