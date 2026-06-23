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

class RSSPPolicy(_SingleGroupPolicyBase):
    """Per-class fixed DiT SP, no in-flight changes."""

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
        *,
        class_sp: dict[str, int] | None = None,
    ) -> None:
        super().__init__(topology=topology, task_runtime_estimator=task_runtime_estimator)
        defaults = {"S": 4, "M": 4, "L": 1}
        if class_sp:
            defaults.update({k.upper(): int(v) for k, v in class_sp.items()})
        self._class_sp = defaults
        self._class_groups: dict[str, tuple[str, ...]] = {}
        for klass, sp in self._class_sp.items():
            groups = _groups_with_sp(topology, sp) or _groups_with_sp(topology, 1)
            if not groups:
                raise ValueError(f"RSSP: no DiT group for class {klass!r} at SP={sp}")
            self._class_groups[klass] = groups
        self._cursors: dict[str, int] = {klass: 0 for klass in self._class_sp}

    def _pick_dit_group_id(self, plan: RequestExecutionPlan) -> str:
        klass = _request_class_from_value(plan)
        groups = self._class_groups.get(klass) or self._class_groups["M"]
        cursor = self._cursors.get(klass, 0)
        self._cursors[klass] = cursor + 1
        return groups[cursor % len(groups)]


def make_rssp(topology, task_runtime_estimator):
    raw = os.getenv("RSSP_CLASS_SP", "S:4,M:4,L:1")
    class_sp: dict[str, int] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        key, value = pair.split(":")
        class_sp[key.strip().upper()] = int(value)
    return RSSPPolicy(topology, task_runtime_estimator, class_sp=class_sp)
