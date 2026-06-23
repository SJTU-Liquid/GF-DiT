# SPDX-License-Identifier: Apache-2.0
"""Real-GFC integration tests for the runtime_v2 collective backend.

Skipped unless the ``gfc`` package is installed and at least 2 CUDA devices are
available. No mocks: this spawns real worker processes, rendezvous on NCCL,
initializes ``SymmetricCollectiveRuntime``, and exercises all_gather / all2all.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model]

_GFC_AVAILABLE = importlib.util.find_spec("gfc") is not None


def _update_env(envs: dict[str, str]) -> None:
    for key, value in envs.items():
        os.environ[key] = value


def _gfc_world(local_rank: int, world_size: int, port: str):
    from vllm_omni.diffusion.distributed import collective_runtime
    from vllm_omni.diffusion.distributed.collective_runtime import (
        LogicalGroupHandle,
        all_gather_into_tensor,
        all_to_all_single,
        init_runtime_v2_collective_runtime,
        make_logical_group,
        shutdown_runtime_v2_collective_runtime,
    )

    _update_env(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
    )
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank,
    )
    return collective_runtime, device, make_logical_group, init_runtime_v2_collective_runtime, all_to_all_single, all_gather_into_tensor, shutdown_runtime_v2_collective_runtime, LogicalGroupHandle


def _all_to_all_worker(local_rank: int, world_size: int, port: str) -> None:
    try:
        (
            collective_runtime,
            device,
            make_logical_group,
            init_runtime,
            all_to_all_single,
            _,
            shutdown,
            LogicalGroupHandle,
        ) = _gfc_world(local_rank, world_size, port)
        init_runtime(
            backend="gfc",
            device=device,
            max_group_size=world_size,
        )
        group = make_logical_group(tuple(range(world_size)), group_id="sp")
        assert isinstance(group, LogicalGroupHandle)
        assert group.gfc_group is not None

        slice_len = 4
        input_tensor = torch.arange(
            local_rank * world_size * slice_len,
            (local_rank + 1) * world_size * slice_len,
            dtype=torch.float32,
            device=device,
        ).reshape(world_size, slice_len)
        output = torch.empty_like(input_tensor)
        all_to_all_single(output, input_tensor, group=group)

        expected = torch.empty_like(input_tensor)
        for peer in range(world_size):
            expected[peer] = torch.arange(
                peer * world_size * slice_len + local_rank * slice_len,
                peer * world_size * slice_len + (local_rank + 1) * slice_len,
                dtype=torch.float32,
                device=device,
            )
        torch.testing.assert_close(output, expected)

        shutdown()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _all_gather_worker(local_rank: int, world_size: int, port: str) -> None:
    try:
        (
            collective_runtime,
            device,
            make_logical_group,
            init_runtime,
            _,
            all_gather_into_tensor,
            shutdown,
            LogicalGroupHandle,
        ) = _gfc_world(local_rank, world_size, port)
        init_runtime(
            backend="gfc",
            device=device,
            max_group_size=world_size,
        )
        group = make_logical_group(tuple(range(world_size)), group_id="sp")
        assert group.gfc_group is not None

        local_chunk = torch.full((8,), float(local_rank + 1), dtype=torch.float32, device=device)
        gathered = torch.empty((world_size * 8,), dtype=torch.float32, device=device)
        all_gather_into_tensor(gathered, local_chunk, group=group)

        expected = torch.cat(
            [torch.full((8,), float(peer + 1), dtype=torch.float32, device=device) for peer in range(world_size)]
        )
        torch.testing.assert_close(gathered, expected)

        shutdown()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(not _GFC_AVAILABLE, reason="gfc package not installed")
@pytest.mark.parametrize("world_size", [2])
def test_gfc_all_to_all_single_matches_expected(world_size: int) -> None:
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"need {world_size} GPUs, have {available_gpus}")
    torch.multiprocessing.spawn(
        _all_to_all_worker,
        args=(world_size, "29610"),
        nprocs=world_size,
    )


def _static_pg_all_to_all_worker(local_rank: int, world_size: int, port: str) -> None:
    """Build a real torch subgroup, side-attach a GFC descriptor (the static
    startup pattern used by parallel_state), and verify that comm.py routes
    all_to_all_single through GFC instead of dist.all_to_all_single."""
    import torch.distributed as dist

    try:
        (
            collective_runtime,
            device,
            _make,
            init_runtime,
            all_to_all_single,
            _,
            shutdown,
            _LogicalGroupHandle,
        ) = _gfc_world(local_rank, world_size, port)
        init_runtime(
            backend="gfc",
            device=device,
            max_group_size=world_size,
        )

        # Build a real torch ProcessGroup the same way initialize_model_parallel_*
        # does at static startup, then side-attach the GFC descriptor via
        # register_static_gfc_subgroup. After this, the PG is callable with
        # both torch and GFC paths.
        ranks = list(range(world_size))
        torch_pg = dist.new_group(ranks=ranks, backend="nccl")
        descriptor = collective_runtime.register_static_gfc_subgroup(torch_pg)
        assert descriptor is not None, "register_static_gfc_subgroup should return a descriptor"
        assert collective_runtime._gfc_group(torch_pg) is descriptor, "side-table lookup should return descriptor"

        # Sentinel: if comm.py falls back to torch.all_to_all_single instead of
        # going through GFC, the test fails loudly.
        original_torch_a2a = dist.all_to_all_single

        def _trap(*args, **kwargs):
            raise AssertionError("comm.py routed all_to_all_single through torch despite GFC backend")

        dist.all_to_all_single = _trap  # type: ignore[assignment]
        try:
            slice_len = 4
            input_tensor = torch.arange(
                local_rank * world_size * slice_len,
                (local_rank + 1) * world_size * slice_len,
                dtype=torch.float32,
                device=device,
            ).reshape(world_size, slice_len)
            output = torch.empty_like(input_tensor)
            all_to_all_single(output, input_tensor, group=torch_pg)
        finally:
            dist.all_to_all_single = original_torch_a2a  # type: ignore[assignment]

        expected = torch.empty_like(input_tensor)
        for peer in range(world_size):
            expected[peer] = torch.arange(
                peer * world_size * slice_len + local_rank * slice_len,
                peer * world_size * slice_len + (local_rank + 1) * slice_len,
                dtype=torch.float32,
                device=device,
            )
        torch.testing.assert_close(output, expected)

        shutdown()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(not _GFC_AVAILABLE, reason="gfc package not installed")
@pytest.mark.parametrize("world_size", [2])
def test_gfc_routes_collectives_via_static_torch_subgroup(world_size: int) -> None:
    """The static-startup pattern: build torch PG, register GFC descriptor on
    the side. Verifies that SP collectives on the static torch PG do go through
    GFC (with a torch.all_to_all_single trap to make the assertion explicit)."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"need {world_size} GPUs, have {available_gpus}")
    torch.multiprocessing.spawn(
        _static_pg_all_to_all_worker,
        args=(world_size, "29612"),
        nprocs=world_size,
    )


@pytest.mark.skipif(not _GFC_AVAILABLE, reason="gfc package not installed")
@pytest.mark.parametrize("world_size", [2])
def test_gfc_all_gather_into_tensor_matches_expected(world_size: int) -> None:
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"need {world_size} GPUs, have {available_gpus}")
    torch.multiprocessing.spawn(
        _all_gather_worker,
        args=(world_size, "29611"),
        nprocs=world_size,
    )
