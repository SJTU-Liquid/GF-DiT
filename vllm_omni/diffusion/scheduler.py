# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from typing import Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_mq = getattr(self, "mq", None)
        if existing_mq is not None and not existing_mq.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self._lock = threading.Lock()
        # Serialize result-queue dequeue without blocking state updates/submits.
        self._recv_lock = threading.Lock()
        self._sync_generation_lock = threading.Lock()

        # Initialize single MessageQueue for all message types (generation & RPC)
        # Assuming all readers are local for now as per current launch_engine implementation
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )

        self.result_mq = None
        self._submission_counter = 0
        self._submitted_request_ids: set[str] = set()
        self._completed_generation_outputs: dict[str, DiffusionOutput] = {}
        self._pending_rpc_responses: deque[Any] = deque()

    def initialize_result_queue(self, handle):
        # Initialize MessageQueue for receiving results
        # We act as rank 0 reader for this queue
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized result MessageQueue")

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Backward-compatible sync API: submit then wait for completion."""
        # Keep legacy add_req serialization semantics.
        with self._sync_generation_lock:
            submission_id = self.submit_req(request)
            return self.collect_req(submission_id)

    def submit_req(self, request: OmniDiffusionRequest) -> str:
        """Submit a generation request and return scheduler request id."""
        with self._lock:
            self._submission_counter += 1
            submission_id = f"diff-{self._submission_counter}"
            self._submitted_request_ids.add(submission_id)
            rpc_request = {
                "type": "rpc",
                "method": "generate",
                "args": (request,),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
                "scheduler_req_id": submission_id,
            }
            self.mq.enqueue(rpc_request)
            return submission_id

    def collect_req(self, submission_id: str, timeout_s: float | None = None) -> DiffusionOutput:
        """Collect completion for a previously submitted request."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._lock:
                if submission_id in self._completed_generation_outputs:
                    output = self._completed_generation_outputs.pop(submission_id)
                    self._submitted_request_ids.discard(submission_id)
                    if output.error:
                        raise RuntimeError("worker error")
                    return output

            dequeue_timeout = self._compute_poll_timeout(deadline)
            with self._recv_lock:
                msg = self._dequeue_result_message(timeout_s=dequeue_timeout)
            if msg is not None:
                with self._lock:
                    self._route_result_message(msg, default_submission_id=submission_id)
                    if submission_id in self._completed_generation_outputs:
                        output = self._completed_generation_outputs.pop(submission_id)
                        self._submitted_request_ids.discard(submission_id)
                        if output.error:
                            raise RuntimeError("worker error")
                        return output

            if deadline is not None and time.monotonic() >= deadline:
                # Non-blocking polls (timeout <= 0) are expected to timeout
                # frequently while requests are still running; avoid noisy logs.
                if timeout_s is not None and timeout_s > 0:
                    logger.warning(
                        "Timeout waiting for response from scheduler (submission_id=%s, timeout_s=%.3f).",
                        submission_id,
                        timeout_s,
                    )
                raise TimeoutError("Scheduler did not respond in time.")

    def enqueue_rpc_request(self, rpc_request: dict[str, Any], timeout_s: float | None = None) -> None:
        lock_timeout = -1 if timeout_s is None else max(0.0, timeout_s)
        acquired = self._lock.acquire(timeout=lock_timeout)
        if not acquired:
            raise TimeoutError("RPC call timed out waiting for scheduler lock.")
        try:
            self.mq.enqueue(rpc_request)
        finally:
            self._lock.release()

    def dequeue_rpc_response(self, timeout_s: float | None = None) -> Any:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._pending_rpc_responses:
                    return self._pending_rpc_responses.popleft()

            dequeue_timeout = self._compute_poll_timeout(deadline)
            # Bound the _recv_lock acquisition by the poll budget rather than
            # blocking unconditionally: otherwise a caller with a deadline can be
            # wedged indefinitely behind another thread that holds _recv_lock while
            # blocked on result_mq (e.g. an add_req waiting on a dead worker), and
            # would never reach its own timeout check below. Failing to acquire just
            # retries the loop, which re-checks the deadline.
            if self._recv_lock.acquire(timeout=max(0.0, dequeue_timeout)):
                try:
                    msg = self._dequeue_result_message(timeout_s=dequeue_timeout)
                finally:
                    self._recv_lock.release()
            else:
                msg = None
            if msg is not None:
                with self._lock:
                    self._route_result_message(msg, prefer_rpc=True)
                    if self._pending_rpc_responses:
                        return self._pending_rpc_responses.popleft()

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("RPC response timed out.")

    def _compute_poll_timeout(self, deadline: float | None) -> float | None:
        if deadline is None:
            return 0.05
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.0
        return min(0.05, remaining)

    def _dequeue_result_message(self, timeout_s: float | None) -> Any | None:
        if self.result_mq is None:
            raise RuntimeError("Result queue not initialized")
        try:
            if timeout_s is None:
                return self.result_mq.dequeue()
            return self.result_mq.dequeue(timeout=timeout_s)
        except (TimeoutError, zmq.error.Again):
            return None

    def _route_result_message(
        self,
        msg: Any,
        default_submission_id: str | None = None,
        *,
        prefer_rpc: bool = False,
    ) -> None:
        # New protocol: generation result envelope with explicit submission id.
        if isinstance(msg, dict) and msg.get("type") == "generation_result":
            submission_id = msg.get("scheduler_req_id") or default_submission_id
            output = msg.get("output")
            if submission_id is None:
                logger.warning("Dropped generation_result without scheduler_req_id.")
                return
            if isinstance(output, DiffusionOutput):
                try:
                    unpack_diffusion_output_shm(output)
                except Exception as e:
                    logger.warning("SHM unpack failed (data may already be inline): %s", e)
                self._completed_generation_outputs[submission_id] = output
                return
            if isinstance(output, dict) and output.get("status") == "error":
                self._completed_generation_outputs[submission_id] = DiffusionOutput(error=str(output.get("error")))
                return
            self._completed_generation_outputs[submission_id] = DiffusionOutput(
                error=f"unexpected generation output type: {type(output)!r}"
            )
            return

        # Legacy worker error dict for generation path.
        if isinstance(msg, dict) and msg.get("status") == "error" and default_submission_id is not None:
            self._completed_generation_outputs[default_submission_id] = DiffusionOutput(error=str(msg.get("error")))
            return

        # Legacy fallback: raw DiffusionOutput without envelope.
        if isinstance(msg, DiffusionOutput):
            if prefer_rpc:
                self._pending_rpc_responses.append(msg)
                return
            submission_id = default_submission_id
            if submission_id is None and len(self._submitted_request_ids) == 1:
                submission_id = next(iter(self._submitted_request_ids))
            if submission_id is not None:
                try:
                    unpack_diffusion_output_shm(msg)
                except Exception as e:
                    logger.warning("SHM unpack failed (data may already be inline): %s", e)
                self._completed_generation_outputs[submission_id] = msg
                return

        # Everything else is treated as an RPC response.
        self._pending_rpc_responses.append(msg)

    def has_pending_generation_result(self, submission_id: str) -> bool:
        with self._lock:
            return submission_id in self._completed_generation_outputs

    def has_submitted_request(self, submission_id: str) -> bool:
        with self._lock:
            return submission_id in self._submitted_request_ids

    def close(self):
        """Closes the socket and terminates the context."""
        self.mq = None
        self.result_mq = None
        self._submitted_request_ids.clear()
        self._completed_generation_outputs.clear()
        self._pending_rpc_responses.clear()
