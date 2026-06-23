# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for runtime_v2 stage-disaggregated scheduling.

Covers spec sections 9.1-9.4:
- policy factory selection (fcfs / srtf / disaggregate)
- DisaggregateSchedulerPolicy admits multi-group requests
- compiler stage placement + RESHARD insertion at encode/dit and dit/vae boundaries
- GlobalScheduler raises a clear error when a normal task references a
  cross-group worker-local artifact without a RESHARD task in between.

The NCCL data plane test lives in tests/diffusion/test_runtime_v2_reshard_nccl.py
because it requires real GPUs.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.runtime_v2.adapters.common import (
    STAGE_AUX,
    STAGE_DIT,
    build_chunked_dit_plan,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    RESHARD_PAYLOAD_DST_GROUP_ID,
    RESHARD_PAYLOAD_SRC_GROUP_ID,
    StepRange,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    DisaggregateSchedulerPolicy,
    DynamicStepFCFSSchedulerPolicy,
    FCFSSchedulerPolicy,
    GlobalScheduler,
    InMemoryArtifactStore,
    SRTFSchedulerPolicy,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology, WorkerSpec


pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _build_two_stage_topology() -> RuntimeTopology:
    """Aux on rank 0, dit on ranks 1+2, non-overlapping."""
    return RuntimeTopology(
        workers=(
            WorkerSpec(worker_rank=0, device_id=0),
            WorkerSpec(worker_rank=1, device_id=1),
            WorkerSpec(worker_rank=2, device_id=2),
        ),
        groups=(
            ExecutionGroupSpec(
                group_id="g_aux",
                ranks=(0,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=(
                    TaskKind.TEXT_ENCODE,
                    TaskKind.VAE_DECODE,
                    TaskKind.FINALIZE,
                ),
            ),
            ExecutionGroupSpec(
                group_id="g_dit",
                ranks=(1, 2),
                parallel_spec=ParallelSpec(tp=2, sp=1, cfg=1),
                supported_task_kinds=(
                    TaskKind.DIT_PREPARE,
                    TaskKind.TIMESTEP_PREPARE,
                    TaskKind.DIT_STEP_CHUNK,
                ),
            ),
        ),
    )


# -- 9.1 policy selection ----------------------------------------------------

def test_runner_factory_selects_disaggregate_policy() -> None:
    from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

    topology = _build_two_stage_topology()
    policy = RuntimeV2Runner._build_scheduler_policy(
        topology=topology,
        scheduler_policy="disaggregate",
        task_runtime_estimator=None,
    )
    assert isinstance(policy, DisaggregateSchedulerPolicy)


def test_runner_factory_selects_dynamic_step_policy() -> None:
    from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

    topology = RuntimeTopology(
        workers=(
            WorkerSpec(worker_rank=0, device_id=0),
            WorkerSpec(worker_rank=1, device_id=1),
            WorkerSpec(worker_rank=2, device_id=2),
        ),
        groups=(
            ExecutionGroupSpec(
                group_id="g_aux",
                ranks=(0,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=(TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE),
            ),
            ExecutionGroupSpec(
                group_id="g_dit",
                ranks=(1, 2),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK),
                ulysses_degree=2,
                ring_degree=1,
            ),
        ),
    )
    policy = RuntimeV2Runner._build_scheduler_policy(
        topology=topology,
        scheduler_policy="dynamic_step_fcfs",
        task_runtime_estimator=None,
    )
    assert isinstance(policy, DynamicStepFCFSSchedulerPolicy)


def test_runner_factory_keeps_fcfs_and_srtf_unchanged() -> None:
    from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

    topology = _build_two_stage_topology()
    fcfs = RuntimeV2Runner._build_scheduler_policy(
        topology=topology, scheduler_policy="fcfs", task_runtime_estimator=None,
    )
    srtf = RuntimeV2Runner._build_scheduler_policy(
        topology=topology, scheduler_policy="srtf", task_runtime_estimator=None,
    )
    assert isinstance(fcfs, FCFSSchedulerPolicy)
    assert isinstance(srtf, SRTFSchedulerPolicy)


def test_fcfs_still_rejects_multi_group_requests() -> None:
    """Spec: FCFS request-level binding must remain unchanged."""
    topology = _build_two_stage_topology()
    policy = FCFSSchedulerPolicy(topology=topology)
    text_task = InferenceTask(
        task_id="r0:text_encode:0",
        request_id="r0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g_aux",
        parallel_spec=ParallelSpec(),
    )
    dit_task = InferenceTask(
        task_id="r0:dit_prepare:0",
        request_id="r0",
        kind=TaskKind.DIT_PREPARE,
        group_id="g_dit",
        parallel_spec=ParallelSpec(),
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={text_task.task_id: text_task, dit_task.task_id: dit_task},
        terminal_task_ids=(dit_task.task_id,),
    )
    with pytest.raises(ValueError, match="multiple groups"):
        list(policy.on_request_submitted(plan))


# -- 9.2 disaggregate policy admits multi-group request ---------------------

def test_disaggregate_policy_admits_multi_group_request() -> None:
    topology = _build_two_stage_topology()
    policy = DisaggregateSchedulerPolicy(topology=topology)

    text_task = InferenceTask(
        task_id="r0:text_encode:0",
        request_id="r0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g_aux",
        parallel_spec=ParallelSpec(),
    )
    dit_task = InferenceTask(
        task_id="r0:dit_prepare:0",
        request_id="r0",
        kind=TaskKind.DIT_PREPARE,
        group_id="g_dit",
        parallel_spec=ParallelSpec(),
        dependencies=(text_task.task_id,),
    )
    vae_task = InferenceTask(
        task_id="r0:vae_decode:0",
        request_id="r0",
        kind=TaskKind.VAE_DECODE,
        group_id="g_aux",
        parallel_spec=ParallelSpec(),
        dependencies=(dit_task.task_id,),
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={
            text_task.task_id: text_task,
            dit_task.task_id: dit_task,
            vae_task.task_id: vae_task,
        },
        terminal_task_ids=(vae_task.task_id,),
    )
    dispatched = list(policy.on_request_submitted(plan))
    # Only text_task is a root (no deps); it should be dispatched on g_aux.
    assert [t.task_id for t in dispatched] == [text_task.task_id]
    assert dispatched[0].group_id == "g_aux"
    assert dispatched[0].status == TaskStatus.DISPATCHED


def test_dynamic_step_policy_blocks_overlapping_rank_groups() -> None:
    topology = RuntimeTopology(
        workers=tuple(WorkerSpec(worker_rank=i, device_id=i) for i in range(4)),
        groups=(
            ExecutionGroupSpec(
                group_id="g_sp2",
                ranks=(0, 1),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
                ulysses_degree=2,
                ring_degree=1,
            ),
            ExecutionGroupSpec(
                group_id="g_sp4",
                ranks=(0, 1, 2, 3),
                parallel_spec=ParallelSpec(tp=1, sp=4, cfg=1),
                supported_task_kinds=(TaskKind.DIT_STEP_CHUNK,),
                ulysses_degree=4,
                ring_degree=1,
            ),
        ),
    )
    policy = DynamicStepFCFSSchedulerPolicy(topology=topology)
    sp2_task = InferenceTask(
        task_id="r0:dit_step_chunk:0",
        request_id="r0",
        kind=TaskKind.DIT_STEP_CHUNK,
        group_id="g_sp2",
        parallel_spec=ParallelSpec(),
    )
    sp4_task = InferenceTask(
        task_id="r1:dit_step_chunk:0",
        request_id="r0",
        kind=TaskKind.DIT_STEP_CHUNK,
        group_id="g_sp4",
        parallel_spec=ParallelSpec(),
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={sp2_task.task_id: sp2_task, sp4_task.task_id: sp4_task},
        terminal_task_ids=(sp2_task.task_id, sp4_task.task_id),
    )

    first = list(policy.on_request_submitted(plan))

    assert [task.task_id for task in first] == [sp2_task.task_id]
    assert policy.outstanding_per_rank == {0: 1, 1: 1}
    assert [task.task_id for task in policy.group_ready["g_sp4"]] == [sp4_task.task_id]

    second = list(
        policy.on_worker_event(
            WorkerEvent(
                event_id="sp2-end",
                task_id=sp2_task.task_id,
                request_id="r0",
                group_id="g_sp2",
                worker_rank=0,
                kind=WorkerEventKind.TASK_EXEC_END,
                timestamp_ns=time.monotonic_ns(),
            )
        )
    )

    assert [task.task_id for task in second] == [sp4_task.task_id]
    assert policy.outstanding_per_rank == {0: 1, 1: 1, 2: 1, 3: 1}


def test_disaggregate_policy_validates_reshard_payload() -> None:
    topology = _build_two_stage_topology()
    policy = DisaggregateSchedulerPolicy(topology=topology)
    bad = InferenceTask(
        task_id="r0:reshard_text_to_dit:0",
        request_id="r0",
        kind=TaskKind.RESHARD,
        group_id=None,
        parallel_spec=ParallelSpec(),
        payload={RESHARD_PAYLOAD_SRC_GROUP_ID: "g_aux"},  # missing dst
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={bad.task_id: bad},
        terminal_task_ids=(bad.task_id,),
    )
    with pytest.raises(ValueError, match=RESHARD_PAYLOAD_DST_GROUP_ID):
        list(policy.on_request_submitted(plan))


def test_disaggregate_policy_dispatches_reshard_separately() -> None:
    """RESHARD goes through a dedicated queue. The endpoint groups (src+dst)
    must be idle of normal tasks at dispatch time, so a RESHARD becomes
    runnable while text_encode is still on aux must wait until text_encode
    finishes. This locks the no-overlap-with-normal-tasks contract.
    """
    topology = _build_two_stage_topology()
    policy = DisaggregateSchedulerPolicy(topology=topology)
    text_task = InferenceTask(
        task_id="r0:text_encode:0",
        request_id="r0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g_aux",
        parallel_spec=ParallelSpec(),
    )
    reshard = InferenceTask(
        task_id="r0:reshard_text_to_dit:0",
        request_id="r0",
        kind=TaskKind.RESHARD,
        group_id=None,
        parallel_spec=ParallelSpec(),
        dependencies=(text_task.task_id,),
        payload={
            RESHARD_PAYLOAD_SRC_GROUP_ID: "g_aux",
            RESHARD_PAYLOAD_DST_GROUP_ID: "g_dit",
        },
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={text_task.task_id: text_task, reshard.task_id: reshard},
        terminal_task_ids=(reshard.task_id,),
    )
    dispatched = list(policy.on_request_submitted(plan))
    assert [t.task_id for t in dispatched] == [text_task.task_id]
    assert policy.outstanding_per_group.get("g_aux") == 1

    # RESHARD made runnable while text_encode is still on aux: must NOT dispatch.
    second = list(policy.on_tasks_runnable([reshard]))
    assert second == []
    assert reshard in policy.reshard_ready
    assert policy.outstanding_reshards == 0
    assert policy.reshard_holds == {}

    # text_encode finishes -> aux is idle -> reshard dispatches and claims both groups.
    third = list(
        policy.on_worker_event(
            WorkerEvent(
                event_id="text-end",
                task_id=text_task.task_id,
                request_id="r0",
                group_id="g_aux",
                worker_rank=0,
                kind=WorkerEventKind.TASK_EXEC_END,
                timestamp_ns=time.monotonic_ns(),
            )
        )
    )
    assert [t.task_id for t in third] == [reshard.task_id]
    assert third[0].kind == TaskKind.RESHARD
    assert third[0].group_id is None
    assert policy.outstanding_reshards == 1
    assert policy.reshard_holds == {"g_aux": 1, "g_dit": 1}
    assert policy.reshard_groups_by_task[reshard.task_id] == ("g_aux", "g_dit")


def test_disaggregate_reshard_blocks_normal_task_on_held_group() -> None:
    """While a RESHARD is in flight it must hold BOTH endpoint groups, so a
    normal task targeted at either endpoint must wait. Without this gate, a
    DIT_PREPARE on g_dit could race the NCCL P2P kernels of the RESHARD on
    the same GPU and deadlock NCCL.
    """
    topology = _build_two_stage_topology()
    policy = DisaggregateSchedulerPolicy(topology=topology)
    reshard = InferenceTask(
        task_id="r0:reshard_text_to_dit:0",
        request_id="r0",
        kind=TaskKind.RESHARD,
        group_id=None,
        parallel_spec=ParallelSpec(),
        payload={
            RESHARD_PAYLOAD_SRC_GROUP_ID: "g_aux",
            RESHARD_PAYLOAD_DST_GROUP_ID: "g_dit",
        },
    )
    dit_task = InferenceTask(
        task_id="r0:dit_prepare:0",
        request_id="r0",
        kind=TaskKind.DIT_PREPARE,
        group_id="g_dit",
        parallel_spec=ParallelSpec(),
    )

    # Both groups idle => RESHARD dispatches and claims them.
    first = list(policy.on_tasks_runnable([reshard]))
    assert [t.task_id for t in first] == [reshard.task_id]
    assert policy.reshard_holds == {"g_aux": 1, "g_dit": 1}

    # Normal task on g_dit must be blocked by the hold.
    blocked = list(policy.on_tasks_runnable([dit_task]))
    assert blocked == []
    assert "g_dit" in policy.group_ready

    # RESHARD finishes -> hold released -> dit_task dispatches.
    after_release = list(
        policy.on_worker_event(
            WorkerEvent(
                event_id="reshard-end",
                task_id=reshard.task_id,
                request_id="r0",
                group_id="g_dit",
                worker_rank=-1,
                kind=WorkerEventKind.TASK_EXEC_END,
                timestamp_ns=time.monotonic_ns(),
            )
        )
    )
    assert [t.task_id for t in after_release] == [dit_task.task_id]
    assert policy.reshard_holds == {}
    assert policy.outstanding_reshards == 0
    assert policy.reshard_groups_by_task == {}


# -- 9.3 compiler stage placement + RESHARD insertion -----------------------

def test_compiler_default_path_unchanged() -> None:
    plan = build_chunked_dit_plan(
        request_id="r0",
        request_value="<request>",
        request_type="test",
        num_steps=2,
        chunk_size=1,
        priority=0,
        group_id="g0",
        text_seq_len=8,
        latent_seq_len=16,
        state_codec_id="codec.state.v1",
        decoded_codec_id="codec.decoded.v1",
    )
    # No stage_group_ids -> all tasks share the legacy group_id, no RESHARD.
    assert all(t.group_id == "g0" for t in plan.tasks.values())
    assert all(t.kind != TaskKind.RESHARD for t in plan.tasks.values())


def test_compiler_default_path_allows_unpinned_tasks_without_dit_schedule() -> None:
    plan = build_chunked_dit_plan(
        request_id="r0",
        request_value="<request>",
        request_type="test",
        num_steps=2,
        chunk_size=1,
        priority=0,
        group_id=None,
        text_seq_len=8,
        latent_seq_len=16,
        state_codec_id="codec.state.v1",
        decoded_codec_id="codec.decoded.v1",
    )
    # Legacy fcfs/srtf plans may leave tasks unpinned and let the scheduler
    # choose a group from the topology. A DiT step schedule is not required.
    assert all(t.group_id is None for t in plan.tasks.values())
    assert all(t.kind != TaskKind.RESHARD for t in plan.tasks.values())


def test_compiler_disaggregate_inserts_reshard_at_both_boundaries() -> None:
    plan = build_chunked_dit_plan(
        request_id="r0",
        request_value="<request>",
        request_type="test",
        num_steps=4,
        chunk_size=2,
        priority=0,
        group_id=None,
        text_seq_len=8,
        latent_seq_len=16,
        state_codec_id="codec.state.v1",
        decoded_codec_id="codec.decoded.v1",
        stage_group_ids={STAGE_AUX: "g_aux", STAGE_DIT: "g_dit"},
    )
    by_id = plan.tasks

    # Per-task group_id assignment per spec.
    assert by_id["r0:text_encode:0"].group_id == "g_aux"
    assert by_id["r0:dit_prepare:0"].group_id == "g_dit"
    assert by_id["r0:timestep_prepare:0"].group_id == "g_dit"
    for task in by_id.values():
        if task.kind == TaskKind.DIT_STEP_CHUNK:
            assert task.group_id == "g_dit"
    assert by_id["r0:vae_decode:0"].group_id == "g_aux"
    assert by_id["r0:finalize:0"].group_id == "g_aux"

    # RESHARD tasks exist at the two boundaries with correct payload.
    text_to_dit = by_id["r0:reshard_text_to_dit:0"]
    assert text_to_dit.kind == TaskKind.RESHARD
    assert text_to_dit.group_id is None
    assert text_to_dit.payload[RESHARD_PAYLOAD_SRC_GROUP_ID] == "g_aux"
    assert text_to_dit.payload[RESHARD_PAYLOAD_DST_GROUP_ID] == "g_dit"
    assert text_to_dit.dependencies == ("r0:text_encode:0",)
    assert [a.artifact_id for a in text_to_dit.inputs] == ["state_text"]
    assert [a.artifact_id for a in text_to_dit.outputs] == ["state_text_dit"]

    dit_to_aux = by_id["r0:reshard_dit_to_aux:0"]
    assert dit_to_aux.kind == TaskKind.RESHARD
    assert dit_to_aux.group_id is None
    assert dit_to_aux.payload[RESHARD_PAYLOAD_SRC_GROUP_ID] == "g_dit"
    assert dit_to_aux.payload[RESHARD_PAYLOAD_DST_GROUP_ID] == "g_aux"
    assert [a.artifact_id for a in dit_to_aux.inputs] == ["state_denoised"]
    assert [a.artifact_id for a in dit_to_aux.outputs] == ["state_denoised_aux"]

    # Downstream consumers must point at the RESHARD outputs, not the source.
    dit_prepare = by_id["r0:dit_prepare:0"]
    assert dit_prepare.dependencies == ("r0:reshard_text_to_dit:0",)
    assert [a.artifact_id for a in dit_prepare.inputs] == ["state_text_dit"]

    vae_decode = by_id["r0:vae_decode:0"]
    assert vae_decode.dependencies == ("r0:reshard_dit_to_aux:0",)
    assert [a.artifact_id for a in vae_decode.inputs] == ["state_denoised_aux"]


def test_compiler_dynamic_step_schedule_inserts_step_reshards() -> None:
    plan = build_chunked_dit_plan(
        request_id="r0",
        request_value="<request>",
        request_type="test",
        num_steps=4,
        chunk_size=1,
        priority=0,
        group_id=None,
        text_seq_len=8,
        latent_seq_len=16,
        state_codec_id="codec.state.v1",
        decoded_codec_id="codec.decoded.v1",
        stage_group_ids={STAGE_AUX: "g_aux", STAGE_DIT: "g_sp2"},
        dit_step_schedule=(
            {"start": 0, "end": 2, "group_id": "g_sp2"},
            {"start": 2, "end": None, "group_id": "g_sp4"},
        ),
    )
    by_id = plan.tasks

    assert by_id["r0:dit_step_chunk:0"].group_id == "g_sp2"
    assert by_id["r0:dit_step_chunk:1"].group_id == "g_sp2"
    assert by_id["r0:dit_step_chunk:2"].group_id == "g_sp4"
    assert by_id["r0:dit_step_chunk:3"].group_id == "g_sp4"

    step_reshard = by_id["r0:reshard_dit_step:2"]
    assert step_reshard.kind == TaskKind.RESHARD
    assert step_reshard.payload[RESHARD_PAYLOAD_SRC_GROUP_ID] == "g_sp2"
    assert step_reshard.payload[RESHARD_PAYLOAD_DST_GROUP_ID] == "g_sp4"
    assert step_reshard.dependencies == ("r0:dit_step_chunk:1",)
    assert by_id["r0:dit_step_chunk:2"].dependencies == ("r0:reshard_dit_step:2",)

    dit_to_aux = by_id["r0:reshard_dit_to_aux:0"]
    assert dit_to_aux.payload[RESHARD_PAYLOAD_SRC_GROUP_ID] == "g_sp4"
    assert by_id["r0:vae_decode:0"].dependencies == ("r0:reshard_dit_to_aux:0",)
    assert by_id["r0:text_encode:0"].payload["request_metadata"]["dit_step_schedule"][1]["group_id"] == "g_sp4"


def test_dynamic_policy_owns_optional_dit_step_schedule_placement() -> None:
    topology = RuntimeTopology(
        workers=tuple(WorkerSpec(worker_rank=i, device_id=i) for i in range(3)),
        groups=(
            ExecutionGroupSpec(
                group_id="g_aux",
                ranks=(0,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=(TaskKind.TEXT_ENCODE, TaskKind.VAE_DECODE, TaskKind.FINALIZE),
            ),
            ExecutionGroupSpec(
                group_id="g_sp1",
                ranks=(1,),
                parallel_spec=ParallelSpec(tp=1, sp=1, cfg=1),
                supported_task_kinds=(TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK),
            ),
            ExecutionGroupSpec(
                group_id="g_sp2",
                ranks=(1, 2),
                parallel_spec=ParallelSpec(tp=1, sp=2, cfg=1),
                supported_task_kinds=(TaskKind.DIT_PREPARE, TaskKind.TIMESTEP_PREPARE, TaskKind.DIT_STEP_CHUNK),
                ulysses_degree=2,
                ring_degree=1,
            ),
        ),
    )
    policy = DynamicStepFCFSSchedulerPolicy(
        topology=topology,
        stage_group_ids={STAGE_AUX: "g_aux", STAGE_DIT: "g_sp1"},
        dit_step_schedule=(
            {"start": 0, "end": 2, "group_id": "g_sp1"},
            {"start": 2, "end": None, "group_id": "g_sp2"},
        ),
    )
    early = InferenceTask(
        task_id="r0:dit_step_chunk:0",
        request_id="r0",
        kind=TaskKind.DIT_STEP_CHUNK,
        group_id=None,
        parallel_spec=ParallelSpec(),
        step_range=StepRange(start=0, end=1),
    )
    late = InferenceTask(
        task_id="r0:dit_step_chunk:2",
        request_id="r0",
        kind=TaskKind.DIT_STEP_CHUNK,
        group_id=None,
        parallel_spec=ParallelSpec(),
        step_range=StepRange(start=2, end=3),
    )

    policy.on_tasks_runnable((early, late))

    assert early.group_id == "g_sp1"
    assert late.group_id == "g_sp2"


def test_codec_pack_strips_state_device_and_assemble_requires_device() -> None:
    """Lock the strip-then-rebuild contract for state codecs: src side must
    not ship state.device (would land on the wrong GPU after migration), and
    assemble must refuse to run without a device kwarg so anyone who forgets
    to override assemble in a new codec hits a loud error, not a silent bug.
    """
    import dataclasses
    import pickle

    import torch

    from vllm_omni.diffusion.runtime_v2.adapters.common import (
        output_artifact_layout,
        replicated_tensor_field,
    )
    from vllm_omni.diffusion.runtime_v2.adapters.wan import (
        WAN_STATE_CODEC_ID,
        WanRuntimeState,
        WanStateLayoutCodec,
    )
    from vllm_omni.diffusion.runtime_v2.protocol import (
        ArtifactHandle,
        ArtifactKind,
        ArtifactLayout,
    )

    class _FakePipe:
        # Minimal pipeline shape used only by the codec; no real model.
        class _TextEncoder:
            dtype = torch.float32

            class _Cfg:
                hidden_size = 16

            config = _Cfg()

        text_encoder = _TextEncoder()
        transformer = None
        transformer_2 = None

    state = WanRuntimeState(
        request=None,  # type: ignore[arg-type]  # codec pack only inspects when non-None
        prompt="x",
        negative_prompt=None,
        height=64,
        width=64,
        num_frames=1,
        num_steps=1,
        output_type="np",
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        guidance_low=1.0,
        guidance_high=1.0,
        boundary_timestep=0.5,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )
    handle = ArtifactHandle(
        request_id="r0",
        artifact_id="state_text",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        codec_id=WAN_STATE_CODEC_ID,
    )
    layout = output_artifact_layout(
        handle,
        (replicated_tensor_field("prompt_embeds", (1, 1, 16), torch.float32),),
    )
    codec = WanStateLayoutCodec(_FakePipe())

    payload = codec.pack_metadata(value=state, layout=layout)
    skeleton = pickle.loads(payload)
    # Critical: src must NOT ship state.device; otherwise dst would inherit cuda:0.
    assert skeleton.device is None, (
        f"pack_metadata leaked rank-local device into skeleton: got {skeleton.device!r}"
    )
    assert skeleton.scheduler is None
    assert skeleton.prompt_embeds is None

    # assemble without device kwarg must refuse (defends against silent device-bind bugs).
    with pytest.raises(ValueError, match="device"):
        codec.assemble(
            metadata_bytes=payload,
            tensors={("prompt_embeds",): torch.zeros(1, 1, 16, dtype=torch.float32)},
            layout=layout,
        )


def test_codec_pack_rejects_undeclared_tensor_field() -> None:
    """If a state value carries a tensor field that the codec layout did not
    declare, pack_metadata MUST raise instead of silently pickling it. Pickling
    a CUDA tensor leaks its src-rank device across the migration boundary and
    causes "Expected all tensors to be on the same device" deep inside the
    model. This is a real regression that bit Wan T2V when negative_prompt
    was unset but guidance > 1.
    """
    import dataclasses

    import torch

    from vllm_omni.diffusion.runtime_v2.adapters.common import (
        output_artifact_layout,
        replicated_tensor_field,
    )
    from vllm_omni.diffusion.runtime_v2.adapters.wan import (
        WAN_STATE_CODEC_ID,
        WanRuntimeState,
        WanStateLayoutCodec,
    )
    from vllm_omni.diffusion.runtime_v2.protocol import (
        ArtifactHandle,
        ArtifactKind,
        ArtifactLayout,
    )

    class _FakePipe:
        class _TextEncoder:
            dtype = torch.float32

            class _Cfg:
                hidden_size = 16

            config = _Cfg()

        text_encoder = _TextEncoder()
        transformer = None
        transformer_2 = None

    state = WanRuntimeState(
        request=None,  # type: ignore[arg-type]
        prompt="x",
        negative_prompt=None,
        height=64,
        width=64,
        num_frames=1,
        num_steps=1,
        output_type="np",
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        guidance_low=4.0,
        guidance_high=4.0,
        boundary_timestep=0.5,
        prompt_embeds=torch.zeros(1, 1, 16),
        # Stash a tensor here that the layout does NOT cover (simulates the
        # negative_prompt_embeds leak when do_true_cfg metadata was wrong).
        negative_prompt_embeds=torch.zeros(1, 1, 16),
    )
    handle = ArtifactHandle(
        request_id="r0",
        artifact_id="state_text",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        codec_id=WAN_STATE_CODEC_ID,
    )
    # Layout intentionally OMITS negative_prompt_embeds so we can prove the guard fires.
    layout = output_artifact_layout(
        handle,
        (replicated_tensor_field("prompt_embeds", (1, 1, 16), torch.float32),),
    )
    codec = WanStateLayoutCodec(_FakePipe())

    with pytest.raises(RuntimeError, match="negative_prompt_embeds"):
        codec.pack_metadata(value=state, layout=layout)


def test_compiler_disaggregate_requires_both_stage_keys() -> None:
    with pytest.raises(ValueError, match="aux"):
        build_chunked_dit_plan(
            request_id="r0",
            request_value=None,
            request_type="test",
            num_steps=1,
            chunk_size=1,
            priority=0,
            group_id=None,
            text_seq_len=4,
            latent_seq_len=4,
            state_codec_id="c",
            decoded_codec_id="c",
            stage_group_ids={STAGE_DIT: "g_dit"},
        )


# -- 9.4 dynamic cross-group artifact migration ------------------------------

class _StaticCompiler:
    def __init__(self, plan: RequestExecutionPlan) -> None:
        self.plan = plan

    def compile_request(self, _request) -> RequestExecutionPlan:
        return self.plan


def test_global_scheduler_migrates_cross_group_artifact_at_dispatch_time() -> None:
    topology = _build_two_stage_topology()
    # Construct a request where DIT_PREPARE on g_dit consumes a worker-local
    # artifact produced on g_aux. The scheduler should migrate it when the
    # consumer is dispatched, so dynamic/elastic policies do not have to
    # pre-insert RESHARD nodes at compile time.
    text_handle = ArtifactHandle(
        request_id="r0",
        artifact_id="state_text",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        producer_task_id="r0:text_encode:0",
        codec_id="codec.state.v1",
    )
    text_task = InferenceTask(
        task_id="r0:text_encode:0",
        request_id="r0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g_aux",
        parallel_spec=ParallelSpec(),
        outputs=(text_handle,),
    )
    dit_task = InferenceTask(
        task_id="r0:dit_prepare:0",
        request_id="r0",
        kind=TaskKind.DIT_PREPARE,
        group_id="g_dit",
        parallel_spec=ParallelSpec(),
        dependencies=(text_task.task_id,),
        inputs=(text_handle,),
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={text_task.task_id: text_task, dit_task.task_id: dit_task},
        terminal_task_ids=(dit_task.task_id,),
    )

    artifact_store = InMemoryArtifactStore()
    worker_pool = Mock()
    scheduler = GlobalScheduler(
        topology=topology,
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=artifact_store,
        policy=DisaggregateSchedulerPolicy(topology=topology),
    )
    scheduler.submit_request(object())
    # text_encode dispatched to aux. Simulate worker publishing a worker-local
    # artifact via TASK_LAUNCH_END so dit_prepare becomes runnable.
    text_event = WorkerEvent(
        event_id="e0",
        task_id=text_task.task_id,
        request_id="r0",
        group_id="g_aux",
        worker_rank=0,
        kind=WorkerEventKind.TASK_LAUNCH_END,
        timestamp_ns=time.monotonic_ns(),
        metadata={
            "published_outputs": (
                WorkerLocalArtifactRef(
                    handle=text_handle, group_id="g_aux", worker_rank=0
                ),
            )
        },
    )
    scheduler._handle_worker_event(text_event)

    worker_pool.begin_migration.assert_called_once()
    migrate_plan = worker_pool.begin_migration.call_args.args[0]
    assert migrate_plan.request_id == "r0"
    assert migrate_plan.src_group_id == "g_aux"
    assert migrate_plan.dst_group_id == "g_dit"
    assert [spec.handle.artifact_id for spec in migrate_plan.artifacts] == ["state_text"]
    # Async: the dst artifact is published only when the migration completes.
    on_done = worker_pool.begin_migration.call_args.kwargs["on_done"]
    on_done(None)
    migrated = artifact_store.get(text_handle)
    assert isinstance(migrated, WorkerLocalArtifactRef)
    assert migrated.group_id == "g_dit"


def test_reshard_failure_fails_request_and_releases_policy_state() -> None:
    """When worker_pool.migrate_artifacts raises (e.g. real NCCL hang
    converted to timeout, or sharded codec not yet implemented), the RESHARD
    task must be cleanly accounted as a failure: the request enters the
    failed state, the policy releases its reshard_holds + outstanding_reshards,
    and the scheduler does not hang waiting for further events.
    """
    topology = _build_two_stage_topology()
    text_handle = ArtifactHandle(
        request_id="r0",
        artifact_id="state_text",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        producer_task_id="r0:text_encode:0",
        codec_id="codec.v1",
    )
    reshard_out_handle = ArtifactHandle(
        request_id="r0",
        artifact_id="state_text_dit",
        kind=ArtifactKind.REQUEST_STATE,
        layout=ArtifactLayout.WORKER_LOCAL,
        producer_task_id="r0:reshard_text_to_dit:0",
        codec_id="codec.v1",
    )
    text_task = InferenceTask(
        task_id="r0:text_encode:0",
        request_id="r0",
        kind=TaskKind.TEXT_ENCODE,
        group_id="g_aux",
        parallel_spec=ParallelSpec(),
        outputs=(text_handle,),
    )
    reshard_task = InferenceTask(
        task_id="r0:reshard_text_to_dit:0",
        request_id="r0",
        kind=TaskKind.RESHARD,
        group_id=None,
        parallel_spec=ParallelSpec(),
        dependencies=(text_task.task_id,),
        inputs=(text_handle,),
        outputs=(reshard_out_handle,),
        payload={
            RESHARD_PAYLOAD_SRC_GROUP_ID: "g_aux",
            RESHARD_PAYLOAD_DST_GROUP_ID: "g_dit",
        },
    )
    plan = RequestExecutionPlan(
        request_id="r0",
        tasks={text_task.task_id: text_task, reshard_task.task_id: reshard_task},
        terminal_task_ids=(reshard_task.task_id,),
    )

    policy = DisaggregateSchedulerPolicy(topology=topology)
    artifact_store = InMemoryArtifactStore()
    worker_pool = Mock()

    scheduler = GlobalScheduler(
        topology=topology,
        worker_pool=worker_pool,
        compiler=_StaticCompiler(plan),
        artifact_store=artifact_store,
        policy=policy,
    )
    scheduler.submit_request(object())
    # text_encode publishes its output (WorkerLocalArtifactRef on g_aux) via LAUNCH_END.
    # This makes reshard runnable but it must wait for text_encode to finish on aux.
    text_launch_end = WorkerEvent(
        event_id="text-launch-end",
        task_id=text_task.task_id,
        request_id="r0",
        group_id="g_aux",
        worker_rank=0,
        kind=WorkerEventKind.TASK_LAUNCH_END,
        timestamp_ns=time.monotonic_ns(),
        metadata={
            "published_outputs": (
                WorkerLocalArtifactRef(handle=text_handle, group_id="g_aux", worker_rank=0),
            )
        },
    )
    scheduler._handle_worker_event(text_launch_end)
    # text_encode finishes -> aux idle -> reshard dispatches -> begin_migration called.
    text_exec_end = WorkerEvent(
        event_id="text-exec-end",
        task_id=text_task.task_id,
        request_id="r0",
        group_id="g_aux",
        worker_rank=0,
        kind=WorkerEventKind.TASK_EXEC_END,
        timestamp_ns=time.monotonic_ns(),
    )
    scheduler._handle_worker_event(text_exec_end)

    # Reshard started exactly one async migration; the failure arrives via callback.
    assert worker_pool.begin_migration.call_count == 1
    on_error = worker_pool.begin_migration.call_args.kwargs["on_error"]
    on_error(RuntimeError("simulated NCCL hang -> timeout"))
    # Request is now in the failed state with an informative message.
    assert "r0" in scheduler.failed_requests
    assert "simulated NCCL hang" in scheduler.failed_requests["r0"]
    status, payload = scheduler.get_request_status("r0")
    assert status == "failed"
    assert "simulated NCCL hang" in str(payload)
    # Policy state is fully released so the scheduler does not deadlock on a
    # phantom in-flight reshard. Both endpoint groups are free to take new work.
    assert policy.outstanding_reshards == 0
    assert policy.reshard_holds == {}
    assert policy.reshard_groups_by_task == {}
