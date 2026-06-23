# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import collective_runtime
from vllm_omni.diffusion.distributed.collective_runtime import LogicalGroupHandle
from vllm_omni.diffusion.distributed.comm import RingComm
from vllm_omni.diffusion.runtime_v2.protocol import ExecutionGroupSpec, ParallelSpec, TaskKind
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]


def test_ring_comm_uses_ordered_global_ranks_for_peers() -> None:
    group = LogicalGroupHandle(ranks=(2, 0, 3), rank=0, group_id="ring")
    comm = RingComm(group)

    assert comm.rank == 1
    assert comm.world_size == 3
    assert comm.send_rank == 3
    assert comm.recv_rank == 2


def test_runtime_topology_ensure_group_registers_dynamic_group() -> None:
    topology = RuntimeTopology(
        workers=tuple(WorkerSpec(worker_rank=rank, device_id=rank) for rank in range(4)),
        groups=(
            ExecutionGroupSpec(
                group_id="g0",
                ranks=(0, 1),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
                ulysses_degree=2,
                ring_degree=1,
            ),
            ExecutionGroupSpec(
                group_id="g1",
                ranks=(2, 3),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
                ulysses_degree=2,
                ring_degree=1,
            ),
        ),
    )
    dynamic = ExecutionGroupSpec(
        group_id="g_dyn",
        ranks=(1, 2),
        parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
        supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
        ulysses_degree=2,
        ring_degree=1,
    )

    assert topology.ensure_group(dynamic) is dynamic
    assert topology.get_group("g_dyn") == dynamic
    assert [group.group_id for group in topology.get_groups_for_worker(1)] == ["g0", "g_dyn"]

    with pytest.raises(ValueError, match="different spec"):
        topology.ensure_group(
            ExecutionGroupSpec(
                group_id="g_dyn",
                ranks=(0, 3),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
                ulysses_degree=2,
                ring_degree=1,
            )
        )


def test_gfc_migration_runtime_env_uses_shared_bool_parser(monkeypatch) -> None:
    created = []

    class FakeSymmetricCollectiveConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSymmetricCollectiveRuntime:
        def __init__(self, config, *, device):
            created.append((config, device))

    fake_gfc = SimpleNamespace(
        SymmetricCollectiveConfig=FakeSymmetricCollectiveConfig,
        SymmetricCollectiveRuntime=FakeSymmetricCollectiveRuntime,
    )
    monkeypatch.setitem(sys.modules, "gfc", fake_gfc)
    monkeypatch.setenv("VLLM_RUNTIME_V2_MIGRATE_GFC_P2P", "y")
    monkeypatch.setattr(collective_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(collective_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(collective_runtime.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(collective_runtime, "_GFC_RUNTIME", None)
    monkeypatch.setattr(collective_runtime, "_GFC_MIGRATE_RUNTIME", None)

    collective_runtime.init_runtime_v2_collective_runtime(
        backend="gfc",
        device=torch.device("cuda", 0),
    )

    assert len(created) == 2
    assert collective_runtime.get_gfc_migrate_runtime() is not None
