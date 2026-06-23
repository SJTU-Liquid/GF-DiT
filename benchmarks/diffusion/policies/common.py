# SPDX-License-Identifier: Apache-2.0
"""Shared helpers + _SingleGroupPolicyBase for the simulator policies."""

from __future__ import annotations

from vllm_omni.diffusion.runtime_v2.protocol import RequestExecutionPlan
from vllm_omni.diffusion.runtime_v2.scheduler import FCFSSchedulerPolicy
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology


def _groups_with_sp(topology: RuntimeTopology, sp: int) -> tuple[str, ...]:
    return tuple(
        group.group_id
        for group in topology.groups
        if int(group.parallel_spec.sp) == sp
    )


def _request_class_from_value(plan: RequestExecutionPlan) -> str:
    value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
    raw = getattr(value, "request_class", None)
    if raw:
        return str(raw).upper()
    prefix = plan.request_id.split("_", 1)[0].upper()
    return prefix if prefix in {"S", "M", "L"} else "M"


def _request_priority(plan: RequestExecutionPlan) -> int:
    value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
    return int(getattr(value, "priority", 0) or 0)


def _request_deadline_ms(plan: RequestExecutionPlan) -> float | None:
    value = plan.initial_artifacts[0].value if plan.initial_artifacts else None
    raw = getattr(value, "deadline_ms", None)
    return float(raw) if raw is not None else None


class _SingleGroupPolicyBase(FCFSSchedulerPolicy):
    """FCFS-based policy that picks one DiT group per request.

    Subclasses override ``_pick_dit_group_id``; this base class wires that
    decision into ``_bind_request_group`` so plan tasks are pinned at
    submission time. Tasks remain in a single group across the request,
    matching the FCFSSchedulerPolicy / SRTFSchedulerPolicy contract.
    """

    def _pick_dit_group_id(self, plan: RequestExecutionPlan) -> str:  # pragma: no cover
        raise NotImplementedError

    def _bind_request_group(self, plan: RequestExecutionPlan) -> str:
        explicit_group_id = self._resolve_explicit_request_group(plan)
        candidate_group_ids = tuple(
            group.group_id
            for group in self.topology.groups
            if all(task.kind in group.supported_task_kinds for task in plan.tasks.values())
        )
        if explicit_group_id is not None:
            if explicit_group_id not in candidate_group_ids:
                raise ValueError(
                    f"request {plan.request_id} is pinned to group {explicit_group_id!r}, "
                    "but that group does not support all task kinds"
                )
            selected = explicit_group_id
        else:
            if not candidate_group_ids:
                raise ValueError(f"no execution group supports all tasks of {plan.request_id}")
            selected = self._pick_dit_group_id(plan)
            if selected not in candidate_group_ids:
                raise ValueError(
                    f"policy {type(self).__name__} picked unsupported group "
                    f"{selected!r} for {plan.request_id}"
                )
        self.request_group[plan.request_id] = selected
        for task in plan.tasks.values():
            if task.group_id is None:
                task.group_id = selected
            elif task.group_id != selected:
                raise ValueError(
                    f"request {plan.request_id} task group mismatch: "
                    f"{task.group_id!r} != {selected!r}"
                )
        return selected

    @staticmethod
    def _resolve_explicit_request_group(plan: RequestExecutionPlan) -> str | None:
        explicit = {task.group_id for task in plan.tasks.values() if task.group_id is not None}
        if not explicit:
            return None
        if len(explicit) != 1:
            raise ValueError(
                f"request {plan.request_id} has tasks pinned to multiple groups: {sorted(explicit)!r}"
            )
        return next(iter(explicit))
