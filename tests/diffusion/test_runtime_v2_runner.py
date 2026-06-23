# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.runtime_v2.protocol import TaskKind
from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _make_runner() -> RuntimeV2Runner:
    return RuntimeV2Runner.__new__(RuntimeV2Runner)


def _make_od_config(
    *,
    num_gpus: int = 8,
    runtime_v2_group_sizes=None,
    runtime_v2_groups_json=None,
    runtime_v2_scheduler_policy: str = "fcfs",
    runtime_v2_collective_backend: str = "torch",
    tp: int | None = None,
    sp: int = 1,
    cfg: int = 1,
):
    default_tp = num_gpus if tp is None else tp
    parallel_config = SimpleNamespace(
        tensor_parallel_size=default_tp,
        sequence_parallel_size=sp,
        cfg_parallel_size=cfg,
        world_size=num_gpus,
    )
    return SimpleNamespace(
        parallel_config=parallel_config,
        num_gpus=num_gpus,
        runtime_v2_group_sizes=runtime_v2_group_sizes,
        runtime_v2_groups_json=runtime_v2_groups_json,
        runtime_v2_scheduler_policy=runtime_v2_scheduler_policy,
        runtime_v2_collective_backend=runtime_v2_collective_backend,
    )


def test_wait_zero_timeout_polls_once_then_times_out(mocker) -> None:
    runner = _make_runner()
    status_iter = iter([("pending", None), ("pending", None)])
    runner.get_request_status = mocker.Mock(side_effect=lambda _request_id: next(status_iter))
    runner.poll_once = mocker.Mock()
    runner._release_request = mocker.Mock()

    with pytest.raises(TimeoutError):
        runner.wait("req-0", timeout_s=0.0)

    runner.poll_once.assert_called_once_with(timeout_s=0.0)
    runner._release_request.assert_not_called()


def test_wait_zero_timeout_returns_finished_payload_after_poll(mocker) -> None:
    runner = _make_runner()
    status_iter = iter([("pending", None), ("finished", "done")])
    runner.get_request_status = mocker.Mock(side_effect=lambda _request_id: next(status_iter))
    runner.poll_once = mocker.Mock()

    assert runner.wait("req-0", timeout_s=0.0) == "done"
    runner.poll_once.assert_called_once_with(timeout_s=0.0)


def test_wait_with_positive_timeout_polls_until_finished(mocker) -> None:
    runner = _make_runner()
    runner._poll_interval_s = 0.01
    runner.get_request_status = mocker.Mock(side_effect=[("pending", None), ("finished", "done")])
    runner.poll_once = mocker.Mock()

    assert runner.wait("req-0", timeout_s=1.0) == "done"
    runner.poll_once.assert_called_once_with(timeout_s=0.01)


def test_close_only_shuts_down_scheduler(mocker) -> None:
    runner = _make_runner()
    runner.scheduler = mocker.Mock()

    runner.close()

    runner.scheduler.shutdown.assert_called_once_with()


def test_build_topology_from_group_sizes() -> None:
    topology = RuntimeV2Runner._build_topology(
        _make_od_config(num_gpus=8, runtime_v2_group_sizes=[4, 2, 1, 1])
    )
    assert [group.group_id for group in topology.groups] == ["g0", "g1", "g2", "g3"]
    assert [group.ranks for group in topology.groups] == [(0, 1, 2, 3), (4, 5), (6,), (7,)]
    assert [group.parallel_spec.tp for group in topology.groups] == [4, 2, 1, 1]
    assert all(group.parallel_spec.sp == 1 for group in topology.groups)
    assert all(group.parallel_spec.cfg == 1 for group in topology.groups)
    assert all(TaskKind.FINALIZE in group.supported_task_kinds for group in topology.groups)


def test_build_topology_from_group_sizes_string_8x1() -> None:
    topology = RuntimeV2Runner._build_topology(
        _make_od_config(num_gpus=8, runtime_v2_group_sizes="1,1,1,1,1,1,1,1")
    )
    assert len(topology.groups) == 8
    assert [group.ranks for group in topology.groups] == [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]


def test_build_topology_edf_greedy_bootstraps_world_and_sp1_groups() -> None:
    topology = RuntimeV2Runner._build_topology(
        _make_od_config(
            num_gpus=4,
            runtime_v2_scheduler_policy="edf_greedy",
            runtime_v2_collective_backend="gfc",
        )
    )

    assert [group.group_id for group in topology.groups] == [
        "g_world",
        "g_sp1_r0",
        "g_sp1_r1",
        "g_sp1_r2",
        "g_sp1_r3",
    ]
    assert topology.get_group("g_world").ranks == (0, 1, 2, 3)
    assert topology.get_group("g_world").parallel_spec.sp == 4
    assert topology.get_group("g_sp1_r2").ranks == (2,)


def test_build_topology_groups_json_takes_priority_over_group_sizes() -> None:
    topology = RuntimeV2Runner._build_topology(
        _make_od_config(
            num_gpus=8,
            runtime_v2_group_sizes=[4, 4],
            runtime_v2_groups_json=[{"size": 8, "tp": 2, "ulysses_degree": 2, "ring_degree": 2, "cfg": 1}],
        )
    )
    assert len(topology.groups) == 1
    assert topology.groups[0].group_id == "g0"
    assert topology.groups[0].ranks == tuple(range(8))
    assert topology.groups[0].parallel_spec.tp == 2
    assert topology.groups[0].parallel_spec.sp == 4
    assert topology.groups[0].ulysses_degree == 2
    assert topology.groups[0].ring_degree == 2


def test_build_topology_allows_overlapping_group_ranks() -> None:
    topology = RuntimeV2Runner._build_topology(
        _make_od_config(
            num_gpus=4,
            runtime_v2_groups_json=[
                {"group_id": "g0", "ranks": [0, 1], "tp": 1, "sp": 2},
                {"group_id": "g1", "ranks": [1, 2, 3], "tp": 1, "sp": 3},
            ],
        )
    )

    assert [group.group_id for group in topology.get_groups_for_worker(1)] == ["g0", "g1"]


def test_build_topology_rejects_missing_worker_coverage() -> None:
    with pytest.raises(ValueError, match="missing ranks"):
        RuntimeV2Runner._build_topology(
            _make_od_config(
                num_gpus=4,
                runtime_v2_groups_json=[{"group_id": "g0", "ranks": [0, 1], "tp": 2}],
            )
        )


def test_build_topology_rejects_invalid_parallel_product() -> None:
    with pytest.raises(ValueError, match="parallel mismatch"):
        RuntimeV2Runner._build_topology(
            _make_od_config(
                num_gpus=4,
                runtime_v2_groups_json=[{"group_id": "g0", "size": 4, "tp": 2, "sp": 1, "cfg": 1}],
            )
        )


def test_build_topology_rejects_mismatched_sp_and_ulysses_ring() -> None:
    with pytest.raises(ValueError, match="sequence parallel mismatch"):
        RuntimeV2Runner._build_topology(
            _make_od_config(
                num_gpus=4,
                runtime_v2_groups_json=[
                    {"group_id": "g0", "size": 4, "tp": 1, "sp": 4, "ulysses_degree": 1, "ring_degree": 2, "cfg": 1}
                ],
            )
        )


def test_gfc_collective_backend_rejects_tp_groups() -> None:
    od_config = _make_od_config(
        num_gpus=2,
        runtime_v2_collective_backend="gfc",
        runtime_v2_groups_json=[{"group_id": "g0", "ranks": [0, 1], "tp": 2, "sp": 1, "cfg": 1}],
    )
    topology = RuntimeV2Runner._build_topology(od_config)

    with pytest.raises(ValueError, match="tp=1"):
        RuntimeV2Runner._validate_collective_backend(od_config, topology)
