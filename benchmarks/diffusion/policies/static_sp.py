# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from vllm_omni.diffusion.runtime_v2.protocol import (
    ExecutionGroupSpec,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskKind,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.scheduler import (
    DynamicStepFCFSSchedulerPolicy,
    FCFSSchedulerPolicy,
    SJFSchedulerPolicy,
    SRTFSchedulerPolicy,
    TaskRuntimeEstimator,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


from benchmarks.diffusion.policies.common import (
    _SingleGroupPolicyBase,
    _request_class_from_value,
    _request_deadline_ms,
    _request_priority,
    _groups_with_sp,
)

class StaticSPPolicy(_SingleGroupPolicyBase):
    """Force every request to a single fixed DiT SP lane (round-robin)."""

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        target_sp: int,
    ) -> None:
        super().__init__(topology=topology, task_runtime_estimator=task_runtime_estimator)
        self._target_sp = int(target_sp)
        self._dit_group_ids = _groups_with_sp(topology, self._target_sp)
        if not self._dit_group_ids:
            raise ValueError(f"static policy needs at least one group at SP={self._target_sp}")
        self._dit_cursor = 0

    def _pick_dit_group_id(self, plan: RequestExecutionPlan) -> str:
        group_id = self._dit_group_ids[self._dit_cursor % len(self._dit_group_ids)]
        self._dit_cursor += 1
        return group_id


def make_fcfs(topology, task_runtime_estimator):
    """Plain FCFS over whatever groups the (possibly --allowed-groups-restricted)
    topology exposes. Restrict to SP1 lanes or a single SP-N group via
    --allowed-groups to get the static-SP baselines, mirroring how the real
    runtime sweep pins fcfs to a groups-json."""
    return FCFSSchedulerPolicy(topology, task_runtime_estimator)


def make_srtf(topology, task_runtime_estimator):
    """Plain SRTF (shortest-remaining-time-first) over the topology's groups.
    Pair with --allowed-groups to fix the SP, same as make_fcfs."""
    return SRTFSchedulerPolicy(topology, task_runtime_estimator)


def make_sjf(topology, task_runtime_estimator):
    """Non-preemptive SJF (shortest-job-first by total estimated work) over the
    topology's groups. Pair with --allowed-groups / --topology-json to fix the
    SP, same as make_fcfs / make_srtf."""
    return SJFSchedulerPolicy(topology, task_runtime_estimator)


def make_static_sp1(topology, task_runtime_estimator):
    return StaticSPPolicy(topology, task_runtime_estimator, target_sp=1)


def make_static_sp2(topology, task_runtime_estimator):
    return StaticSPPolicy(topology, task_runtime_estimator, target_sp=2)


def make_static_sp4(topology, task_runtime_estimator):
    return StaticSPPolicy(topology, task_runtime_estimator, target_sp=4)


def make_static_sp8(topology, task_runtime_estimator):
    return StaticSPPolicy(topology, task_runtime_estimator, target_sp=8)
