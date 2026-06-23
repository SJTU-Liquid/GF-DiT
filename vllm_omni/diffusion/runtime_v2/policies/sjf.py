# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from vllm_omni.diffusion.runtime_v2.policies.srtf import SRTFSchedulerPolicy
from vllm_omni.diffusion.runtime_v2.protocol import InferenceTask, RequestExecutionPlan
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

from vllm_omni.diffusion.runtime_v2.policies.common import TaskRuntimeEstimator


class SJFSchedulerPolicy(SRTFSchedulerPolicy):
    """Non-preemptive Shortest-Job-First.

    Reuses SRTF's request/group binding and runtime accounting but changes the
    two properties that make SRTF "SRTF":

    * **Orders by total job size, not remaining work.** A request is ranked by
      the estimated runtime of its *whole* plan, fixed at submission, so its
      priority never shifts as it makes progress. (Among requests that have not
      started yet this equals their remaining work, which is why the value is
      snapshotted lazily the first time a request is considered.)
    * **Non-preemptive.** Once a group starts serving a request it stays pinned
      to that request until it finishes. A shorter request that arrives mid-flight
      waits, rather than displacing the running one at the next task boundary
      (which is exactly what SRTF would do).
    """

    def __init__(
        self,
        topology: RuntimeTopology,
        task_runtime_estimator: TaskRuntimeEstimator | None = None,
    ) -> None:
        super().__init__(topology, task_runtime_estimator)
        # request_id -> total estimated work, snapshotted at submission (before any
        # task is charged) so the ordering key is the whole job size and never
        # shifts as the request makes progress.
        self.request_total_work: dict[str, float] = {}

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        result = super().on_request_submitted(plan)
        # super() set request_remaining_work to the full plan estimate and has not
        # charged any task yet (charging happens later on TASK_LAUNCH_END), so this
        # is the true total job size. Snapshotting here rather than lazily avoids
        # capturing an already-reduced remaining value if the request is first
        # considered after partial progress.
        self.request_total_work[plan.request_id] = self.request_remaining_work.get(
            plan.request_id, float("inf")
        )
        return result

    def _select_shortest_request(
        self,
        group_id: str,
        by_request: dict[str, deque[InferenceTask]],
    ) -> str | None:
        if not by_request:
            return None
        active_request = self.active_request_by_group.get(group_id)
        if active_request is not None:
            # In-flight request holds the group (non-preemptive). active_request_by_group
            # only carries a value between a request's first dispatch and its
            # REQUEST_FINISHED, so a non-None entry means that request is still
            # running here. If its next task is not runnable yet, leave the group
            # idle rather than admitting a different request.
            return active_request if active_request in by_request else None
        # Group is free: admit the waiting request with the smallest total job.
        for request_id in by_request:
            if request_id not in self.request_total_work:
                self.request_total_work[request_id] = self.request_remaining_work.get(
                    request_id, float("inf")
                )
        return min(
            by_request.keys(),
            key=lambda request_id: (
                self.request_total_work.get(request_id, float("inf")),
                self.request_submit_seq.get(request_id, self._submit_counter),
                request_id,
            ),
        )

    def _cleanup_request_work(self, request_id: str, group_id: str | None = None) -> None:
        self.request_total_work.pop(request_id, None)
        super()._cleanup_request_work(request_id=request_id, group_id=group_id)
