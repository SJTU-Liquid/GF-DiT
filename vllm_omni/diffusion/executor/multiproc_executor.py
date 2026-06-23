import multiprocessing as mp
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.worker import WorkerProc

logger = init_logger(__name__)


def _close_runtime_v2_runner(runner: Any) -> None:
    if runner is None:
        return
    try:
        runner.close()
    except Exception as exc:
        logger.warning("Failed to close runtime_v2 runner during finalization: %s", exc)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.scheduler is not None:
            try:
                for _ in range(self.scheduler.num_workers):
                    self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
                self.scheduler.close()
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)
        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(30)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(30)


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self.runtime_v2_runner = None
        # Serializes collective_rpc: RPC responses land in a single unkeyed FIFO
        # (scheduler._pending_rpc_responses) with no correlation id, so two
        # concurrent RPCs could pop each other's replies. Holding this across the
        # whole enqueue+collect restores the atomicity the pre-refactor code had
        # (it held scheduler._lock for the full enqueue+dequeue span).
        self._rpc_lock = threading.Lock()

        if self._should_use_runtime_v2():
            from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

            self.runtime_v2_runner = RuntimeV2Runner(
                pipeline=None,
                default_step_chunk_size=int(getattr(self.od_config, "runtime_v2_denoise_chunk_size", 1) or 1),
                scheduler_policy=str(getattr(self.od_config, "runtime_v2_scheduler_policy", "fcfs") or "fcfs").lower(),
                omni_diffusion_config=self.od_config,
            )
            self.resources = BackgroundResources(scheduler=None, processes=None)
            self._finalizer = weakref.finalize(self, _close_runtime_v2_runner, self.runtime_v2_runner)
            return

        # Initialize scheduler
        self.scheduler = Scheduler()
        self.scheduler.initialize(self.od_config)
        broadcast_handle = self.scheduler.get_broadcast_handle()

        # Launch workers
        processes, result_handle = self._launch_workers(broadcast_handle)

        if result_handle is not None:
            self.scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")

        self._processes = processes

        self.resources = BackgroundResources(scheduler=self.scheduler, processes=self._processes)
        self._finalizer = weakref.finalize(self, self.resources)

    def _should_use_runtime_v2(self) -> bool:
        if not bool(getattr(self.od_config, "enable_runtime_v2", False)):
            return False
        from vllm_omni.diffusion.runtime_v2.registry import supports_runtime_v2_model

        model_class_name = getattr(self.od_config, "model_class_name", None)
        if not supports_runtime_v2_model(model_class_name):
            raise ValueError(f"runtime_v2 does not support model_class_name={model_class_name!r}")
        return True

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Extract worker_extension_cls and custom_pipeline_args from od_config
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = mp.Process(
                target=WorkerProc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                    worker_extension_cls,
                    custom_pipeline_args,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handle = None
        for writer in scheduler_pipe_writers:
            writer.close()

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
                processes[i].join()
                logger.error(f"Exit code: {processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError("Initialization failed. Please see the error messages above.")

            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)
            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handle

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if self.runtime_v2_runner is not None:
            return self.runtime_v2_runner.execute(request)
        return self.scheduler.add_req(request)

    def submit_req(self, request: OmniDiffusionRequest) -> str:
        if self.runtime_v2_runner is not None:
            return self.runtime_v2_runner.submit(request)
        return self.scheduler.submit_req(request)

    def collect_req(self, request_id: str, timeout_s: float | None = None) -> DiffusionOutput:
        if self.runtime_v2_runner is not None:
            return self.runtime_v2_runner.wait(request_id, timeout_s=timeout_s)
        return self.scheduler.collect_req(request_id, timeout_s=timeout_s)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        if self.runtime_v2_runner is not None:
            if self._closed:
                raise RuntimeError("DiffusionExecutor is closed.")
            return self.runtime_v2_runner.collective_rpc(
                method,
                timeout_s=timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
            )
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Serialize the whole enqueue + response-collection span: RPC responses
            # land in a single unkeyed FIFO, so concurrent RPCs could otherwise pop
            # each other's replies. Acquire with the caller's deadline rather than
            # an unbounded wait, so a timed RPC is never wedged indefinitely behind
            # another RPC that is itself blocked on a slow/dead worker -- it fails
            # fast on the lock with a clear message instead of acquiring at its
            # deadline and then immediately timing out the enqueue. (timeout=None
            # callers still wait, as requested.)
            lock_timeout = -1 if deadline is None else max(0.0, deadline - time.monotonic())
            if not self._rpc_lock.acquire(timeout=lock_timeout):
                raise TimeoutError(f"RPC call to {method} timed out waiting for the RPC lock.")
            try:
                # Broadcast RPC request to all workers via unified message queue.
                enqueue_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                self.scheduler.enqueue_rpc_request(rpc_request, timeout_s=enqueue_timeout)

                # Determine which workers we expect responses from
                num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

                responses = []
                for _ in range(num_responses):
                    dequeue_timeout = None if deadline is None else max(0, deadline - time.monotonic())
                    try:
                        response = self.scheduler.dequeue_rpc_response(timeout_s=dequeue_timeout)

                        # Check if response indicates an error
                        if isinstance(response, dict) and response.get("status") == "error":
                            raise RuntimeError(
                                f"Worker failed with error '{response.get('error')}', "
                                "please check the stack trace above for the root cause"
                            )

                        responses.append(response)
                    except TimeoutError as e:
                        raise TimeoutError(f"RPC call to {method} timed out.") from e

                return responses[0] if unique_reply_rank is not None else responses
            finally:
                self._rpc_lock.release()

        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        if self.runtime_v2_runner is not None:
            self.runtime_v2_runner.check_health()
            return
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        if self.runtime_v2_runner is not None:
            self.runtime_v2_runner.close()
            self.runtime_v2_runner = None
            if hasattr(self, "_finalizer"):
                self._finalizer.detach()
            return
        self._finalizer()
