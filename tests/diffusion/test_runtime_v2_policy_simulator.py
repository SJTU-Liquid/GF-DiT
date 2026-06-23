# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys

from benchmarks.diffusion.runtime_v2_policy_simulator import (
    CostModelKey,
    CostModelStore,
    RuntimeV2PolicySimulator,
    SimRequest,
    build_chrome_trace_events,
    build_policy_factory,
    build_sim_plan,
    build_topology,
    generate_synthetic_workload,
)


def _write_cost_model(path, *, tp=1, sp=1, ulysses=1, ring=1, cfg=1, constant=1.0) -> None:
    stages = {}
    for stage, x_name in {
        "text_encode": "text_seq_len",
        "dit_prepare": "latent_seq_len",
        "timestep_prepare": "num_steps",
        "dit_step_chunk": "latent_seq_len",
        "vae_decode": "voxels",
        "finalize": "ones",
    }.items():
        stages[stage] = {
            "x_name": x_name,
            "coeffs": {"a": 0.0, "b": 0.0, "c": constant},
            "r2": 1.0,
            "samples": [[1.0, constant]],
        }
    path.write_text(
        json.dumps(
            {
                "model": "sim",
                "parallelism": {
                    "tp": tp,
                    "sp": sp,
                    "ulysses": ulysses,
                    "ring": ring,
                    "cfg": cfg,
                },
                "options": {},
                "stages": stages,
            }
        )
    )


def test_cost_model_loader_matches_group_parallelism(tmp_path) -> None:
    _write_cost_model(tmp_path / "sp1.json", sp=1, ulysses=1)
    _write_cost_model(tmp_path / "sp2.json", sp=2, ulysses=2)

    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(
        _args(cost_model_dir=str(tmp_path), ulysses_degree=2, world_size=2),
        store,
    )

    assert set(store.models) == {
        CostModelKey(tp=1, sp=1, ulysses=1, ring=1, cfg=1),
        CostModelKey(tp=1, sp=2, ulysses=2, ring=1, cfg=1),
    }
    assert {worker.worker_rank for worker in topology.workers} == {0, 1}
    assert [store.for_group(group).key.sp for group in topology.groups] == [2]


def test_cost_model_derives_cfg_and_nocfg_dit_stages(tmp_path) -> None:
    _write_cost_model(tmp_path / "sp1.json", constant=10.0)

    store = CostModelStore.load_dir(tmp_path)
    model = next(iter(store.models.values()))

    assert model.estimate_ms("dit_step_chunk_cfg", 1.0) == 10.0
    assert model.estimate_ms("dit_step_chunk_nocfg", 1.0) == 5.0
    assert model.stage_x_name("dit_step_chunk_nocfg") == "latent_seq_len"


def test_explicit_groups_json_controls_multiple_candidate_groups(tmp_path) -> None:
    _write_cost_model(tmp_path / "sp1.json", sp=1, ulysses=1)
    _write_cost_model(tmp_path / "sp2.json", sp=2, ulysses=2)
    _write_cost_model(tmp_path / "sp4.json", sp=4, ulysses=4)

    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(
        _args(
            world_size=4,
            groups_json=json.dumps(
                [
                    {"group_id": "sp1_a", "ranks": [0], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "sp1_b", "ranks": [1], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "sp2", "ranks": [2, 3], "tp": 1, "ulysses_degree": 2, "ring_degree": 1},
                    {"group_id": "sp4", "ranks": [0, 1, 2, 3], "tp": 1, "ulysses_degree": 4, "ring_degree": 1},
                ]
            ),
        ),
        store,
    )

    assert len(topology.workers) == 4
    assert [group.ranks for group in topology.groups] == [
        (0,),
        (1,),
        (2, 3),
        (0, 1, 2, 3),
    ]


def test_sim_plan_uses_explicit_x_name_features() -> None:
    request = SimRequest(
        request_id="r0",
        arrival_ms=0.0,
        height=720,
        width=1280,
        num_frames=81,
        num_steps=5,
        denoise_chunk_size=1,
        text_seq_len=512,
    )

    plan = build_sim_plan(request)

    timestep = plan.tasks["r0:timestep_prepare:0"]
    vae = plan.tasks["r0:vae_decode:0"]
    assert timestep.payload["seq_len"] == 5
    assert timestep.payload["sim_features"]["num_steps"] == 5
    assert vae.payload["seq_len"] == 720 * 1280 * 81
    assert vae.payload["sim_features"]["voxels"] == 720 * 1280 * 81


def test_sim_plan_can_mark_dit_chunk_nocfg() -> None:
    request = SimRequest(
        request_id="r0",
        arrival_ms=0.0,
        height=480,
        width=832,
        num_frames=17,
        num_steps=1,
        denoise_chunk_size=1,
        text_seq_len=512,
        do_true_cfg=False,
    )

    plan = build_sim_plan(request)
    task = plan.tasks["r0:dit_step_chunk:0"]

    assert task.payload["do_true_cfg"] is False
    assert task.payload["cost_model_stage"] == "dit_step_chunk_nocfg"


def test_fcfs_simulation_produces_predictable_single_rank_timeline(tmp_path) -> None:
    _write_cost_model(tmp_path / "sp1.json", constant=1.0)
    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(_args(cost_model_dir=str(tmp_path)), store)
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=build_policy_factory("fcfs"),
        cost_models=store,
        exec_start_delay_ms=0.0,
    )
    requests = [
        SimRequest("r0", 0.0, 480, 832, 17, 1, 1, 512),
        SimRequest("r1", 0.0, 480, 832, 17, 1, 1, 512),
    ]

    result = simulator.run(requests)

    assert result.request_finish_ms == {"r0": 6.0, "r1": 12.0}
    assert len(result.timeline) == 24
    exec_events = [event for event in result.timeline if event.kind != "task_launch"]
    assert len(exec_events) == 12
    assert [(event.request_id, event.start_ms, event.end_ms) for event in exec_events[:6]] == [
        ("r0", 0.0, 1.0),
        ("r0", 1.0, 2.0),
        ("r0", 2.0, 3.0),
        ("r0", 3.0, 4.0),
        ("r0", 4.0, 5.0),
        ("r0", 5.0, 6.0),
    ]


def test_text_vae_finalize_use_only_one_rank_of_sp_group(tmp_path) -> None:
    """Text encoder, VAE decode and finalize must not block (sp-1) extra ranks
    in the assigned SP group. We assert that when an SP2 plan runs, those
    three stages only consume rank[0] of the group, while DiT_step_chunk
    consumes all ranks."""
    _write_cost_model(tmp_path / "sp2.json", sp=2, ulysses=2, constant=1.0)
    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(
        _args(cost_model_dir=str(tmp_path), ulysses_degree=2, world_size=2),
        store,
    )
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=build_policy_factory("fcfs"),
        cost_models=store,
    )
    result = simulator.run([SimRequest("r0", 0.0, 480, 832, 17, 2, 1, 512)])

    ranks_by_kind: dict[str, set[int]] = {}
    for event in result.timeline:
        ranks_by_kind.setdefault(event.kind, set()).update(event.ranks)
    # Single-rank stages: only rank[0] of the SP2 group is used.
    for kind in ("text_encode", "vae_decode", "finalize"):
        assert ranks_by_kind.get(kind) == {0}, (kind, ranks_by_kind.get(kind))
    # DiT step chunk is full SP2 (ranks 0 and 1).
    assert ranks_by_kind.get("dit_step_chunk") == {0, 1}


def test_chrome_trace_has_rank_lanes(tmp_path) -> None:
    _write_cost_model(tmp_path / "sp1.json")
    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(_args(cost_model_dir=str(tmp_path)), store)
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=build_policy_factory("fcfs"),
        cost_models=store,
    )

    result = simulator.run([SimRequest("r0", 0.0, 480, 832, 17, 1, 1, 512)])
    events = build_chrome_trace_events(topology, result.timeline)

    assert any(
        event["ph"] == "M" and event["name"] == "thread_name" and event["args"]["name"] == "rank 0"
        for event in events
    )
    task_events = [event for event in events if event.get("ph") == "X"]
    assert task_events
    assert {event["tid"] for event in task_events} == {1}
    assert {event["cname"] for event in task_events}
    assert task_events[0]["args"]["request_id"] == "r0"
    assert "queue_wait_ms" in task_events[0]["args"]


def test_synthetic_arrival_ms_is_interarrival_without_burst() -> None:
    requests = generate_synthetic_workload(_args(num_requests=3, arrival_ms=7.5))

    assert [request.arrival_ms for request in requests] == [0.0, 7.5, 15.0]


def test_custom_policy_factory_loads_from_module(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "custom_policy.py"
    module_path.write_text(
        "from vllm_omni.diffusion.runtime_v2.scheduler import FCFSSchedulerPolicy\n"
        "def make_policy(topology, task_runtime_estimator):\n"
        "    return FCFSSchedulerPolicy(topology=topology, task_runtime_estimator=task_runtime_estimator)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("custom_policy", None)

    factory = build_policy_factory("custom_policy:make_policy")

    assert callable(factory)


def test_edf_greedy_cost_sort_handles_missing_topology_group(tmp_path) -> None:
    """Regression: edf_greedy's cost-aware SP picker must not raise when the
    simulator's topology lacks a pre-built group for every allowed_sp_size.

    With ``allowed_sp_sizes={1,2,4}`` and a minimal topology that contains
    only SP1 lanes, the picker still queries the estimator for SP2/SP4 to
    sort candidates by cost. Pre-fix the estimator walked
    ``topology.groups`` and raised ``KeyError`` because no SP2/SP4 group
    existed yet (those are created on demand by ``_ensure_group_for_ranks``
    AFTER the SP choice is made). The fix routes estimation through
    ``CostModelStore.for_parallel_spec`` directly.
    """
    _write_cost_model(tmp_path / "sp1.json", sp=1, ulysses=1, constant=100.0)
    _write_cost_model(tmp_path / "sp2.json", sp=2, ulysses=2, constant=80.0)
    _write_cost_model(tmp_path / "sp4.json", sp=4, ulysses=4, constant=50.0)
    store = CostModelStore.load_dir(tmp_path)
    # Minimal topology: 4 SP1 lanes only. No SP2 or SP4 group present;
    # edf_greedy will materialize one on demand AFTER the cost-sort picks.
    topology = build_topology(
        _args(
            cost_model_dir=str(tmp_path),
            world_size=4,
            groups_json=json.dumps(
                [
                    {
                        "group_id": f"sp1_{r}",
                        "ranks": [r],
                        "tp": 1,
                        "ulysses_degree": 1,
                        "ring_degree": 1,
                    }
                    for r in range(4)
                ]
            ),
        ),
        store,
    )
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=build_policy_factory(
            "benchmarks.diffusion.policies.edf_greedy:make_edf_greedy"
        ),
        cost_models=store,
        exec_start_delay_ms=0.0,
    )
    result = simulator.run(
        [
            SimRequest(
                request_id="S_0",
                arrival_ms=0.0,
                height=512,
                width=512,
                num_frames=1,
                num_steps=1,
                denoise_chunk_size=1,
                text_seq_len=64,
                priority=100,
                deadline_ms=10000.0,
                profiled_optimal_latency_ms=200.0,
                request_class="S",
            )
        ]
    )
    # Survived the cost-sort lookup AND dispatched to a real group.
    assert result.request_finish_ms.get("S_0", 0.0) > 0.0


def test_primal_dual_rolling_policy_loads_and_uses_parallel_group(tmp_path, monkeypatch) -> None:
    _write_cost_model(tmp_path / "sp1.json", sp=1, ulysses=1, constant=10.0)
    _write_cost_model(tmp_path / "sp2.json", sp=2, ulysses=2, constant=1.0)
    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(
        _args(
            cost_model_dir=str(tmp_path),
            world_size=2,
            groups_json=json.dumps(
                [
                    {"group_id": "sp1_0", "ranks": [0], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "sp1_1", "ranks": [1], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "sp2", "ranks": [0, 1], "tp": 1, "ulysses_degree": 2, "ring_degree": 1},
                ]
            ),
        ),
        store,
    )
    monkeypatch.setenv("PD_ROLL_GPU_TIME_PENALTY", "0")
    monkeypatch.setenv("PD_ROLL_TOP_M_REQUESTS", "2")
    monkeypatch.setenv("PD_ROLL_TOP_G_GROUPS", "3")
    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=build_policy_factory(
            "benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_pd_rolling"
        ),
        cost_models=store,
        exec_start_delay_ms=0.0,
    )

    result = simulator.run(
        [
            SimRequest(
                request_id="S_0",
                arrival_ms=0.0,
                height=512,
                width=512,
                num_frames=1,
                num_steps=1,
                denoise_chunk_size=1,
                text_seq_len=64,
                priority=100,
                deadline_ms=100.0,
                profiled_optimal_latency_ms=60.0,
                request_class="S",
            )
        ]
    )

    assert result.request_finish_ms["S_0"] > 0.0
    assert any(
        event.kind == "dit_step_chunk" and event.ranks == (0, 1)
        for event in result.timeline
    )


def _args(**overrides):
    defaults = {
        "groups_json": None,
        "group_sizes": None,
        "world_size": None,
        "tp": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "cfg": 1,
        "num_requests": 1,
        "arrival_ms": 0.0,
        "burst_size": 1,
        "burst_gap_ms": 0.0,
        "shape_pool": "480x832x17",
        "num_steps": 1,
        "denoise_chunk_size": 1,
        "text_seq_len": 512,
        "seed": 0,
        "group_id": None,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()


def test_explicit_reshard_with_auto_reshard_distinguishes_synthetic(tmp_path) -> None:
    """Regression: simulator-injected (synthetic) RESHARDs vs policy-dispatched
    (explicit, from the plan) RESHARDs must be handled differently.

    An aux->dit stage-group plan emits an EXPLICIT reshard_text_to_dit task; with
    reshard_ms>0 the dit->vae boundary also triggers the simulator's AUTOMATIC
    (synthetic) reshard. The fix must:
      [P1] forward the EXPLICIT reshard's TASK_EXEC_END to the policy (dynamic
           policies need it to release reshard holds), while still hiding the
           SYNTHETIC reshard's events from the policy.
      [P2] record a reshard's completed group as its real dst (not the synthetic
           "src->dst"), so the dit consumer of the explicit reshard does not see
           a phantom producer group and crash trying to re-migrate from it.
    """
    from vllm_omni.diffusion.runtime_v2.policies.disaggregate import (
        DynamicStepFCFSSchedulerPolicy,
    )
    from vllm_omni.diffusion.runtime_v2.protocol import TaskKind, WorkerEventKind

    _write_cost_model(tmp_path / "sp1.json", sp=1, ulysses=1, constant=10.0)
    _write_cost_model(tmp_path / "sp2.json", sp=2, ulysses=2, constant=8.0)
    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(
        _args(
            cost_model_dir=str(tmp_path),
            world_size=2,
            groups_json=json.dumps(
                [
                    {"group_id": "sp1_0", "ranks": [0], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "sp1_1", "ranks": [1], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "sp2_0", "ranks": [0, 1], "tp": 1, "ulysses_degree": 2, "ring_degree": 1},
                ]
            ),
        ),
        store,
    )
    seen: list[tuple] = []

    class _RecordingStageDynFCFS(DynamicStepFCFSSchedulerPolicy):
        def build_sim_plan(self, *, request, default_builder):
            return default_builder(request, stage_group_ids=self.stage_group_ids)

        def on_worker_event(self, event):
            seen.append((event.kind, event.task_id))
            return super().on_worker_event(event)

    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=lambda topo, est: _RecordingStageDynFCFS(
            topo, stage_group_ids={"aux": "sp1_0", "dit": "sp2_0"}
        ),
        cost_models=store,
        reshard_ms=50.0,
        exec_start_delay_ms=0.0,
    )
    result = simulator.run([SimRequest("r0", 0.0, 512, 512, 1, 2, 1, 64)])

    # Completes: explicit reshard not stalled (P1); dit consumer didn't crash on
    # a phantom "src->dst" source group (P2).
    assert result.request_finish_ms.get("r0", 0.0) > 0.0
    # [P1] explicit reshard's completion IS forwarded to the policy ...
    assert any(
        k == WorkerEventKind.TASK_EXEC_END and "reshard_text_to_dit" in tid
        for k, tid in seen
    ), f"explicit reshard TASK_EXEC_END not seen by policy: {seen}"
    # ... but the synthetic (auto) reshard's events are NOT.
    assert not any(
        k == WorkerEventKind.TASK_EXEC_END and "::reshard" in tid for k, tid in seen
    ), "synthetic reshard event leaked to the policy"


def test_reshard_samples_enable_automatic_dependency_migration(tmp_path) -> None:
    """A samples-only reshard model must still inject automatic migrations.

    ``--reshard-samples-json`` is documented as replacing the flat
    ``--reshard-ms`` duration model, so reshard_ms=0 cannot disable automatic
    dependency migrations when samples are present.
    """
    from vllm_omni.diffusion.runtime_v2.protocol import TaskKind

    _write_cost_model(tmp_path / "sp1.json", sp=1, ulysses=1, constant=1.0)
    store = CostModelStore.load_dir(tmp_path)
    topology = build_topology(
        _args(
            cost_model_dir=str(tmp_path),
            world_size=2,
            groups_json=json.dumps(
                [
                    {"group_id": "g0", "ranks": [0], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                    {"group_id": "g1", "ranks": [1], "tp": 1, "ulysses_degree": 1, "ring_degree": 1},
                ]
            ),
        ),
        store,
    )

    class _AlternatingGroupPolicy:
        def on_request_submitted(self, plan):
            roots = [task for task in plan.tasks.values() if not task.dependencies]
            return self.on_tasks_runnable(roots)

        def on_tasks_runnable(self, tasks):
            out = []
            for task in tasks:
                if task.kind in (TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE):
                    task.group_id = "g0"
                else:
                    task.group_id = "g1"
                out.append(task)
            return out

        def on_worker_event(self, event):
            return []

    simulator = RuntimeV2PolicySimulator(
        topology=topology,
        policy_factory=lambda topo, est: _AlternatingGroupPolicy(),
        cost_models=store,
        reshard_ms=0.0,
        reshard_samples=[7.0],
        exec_start_delay_ms=0.0,
    )
    result = simulator.run([SimRequest("r0", 0.0, 512, 512, 1, 1, 1, 64)])

    synthetic_reshards = [
        event for event in result.timeline
        if event.kind == "reshard" and "::reshard" in event.task_id
    ]
    assert synthetic_reshards
    assert {event.end_ms - event.start_ms for event in synthetic_reshards} == {7.0}
