# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Diffusion Worker for vLLM-Omni.

Handles GPU infrastructure initialization and delegates model operations
to DiffusionModelRunner.
"""

import gc
import multiprocessing as mp
import os
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import zmq
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.mem_utils import GiB_bytes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel_from_execution_groups,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.collective_runtime import init_runtime_v2_collective_runtime
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.profiler import CurrentProfiler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.lora.request import LoRARequest
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

logger = init_logger(__name__)


@dataclass
class _PendingRuntimeV2Submission:
    scheduler_req_id: str
    should_reply: bool


class DiffusionWorker:
    """
    A worker that manages GPU infrastructure and delegates to the model runner.

    This class handles infrastructure initialization only:
    - Device setup (CUDA device selection)
    - Distributed environment (NCCL, model parallel)
    - Memory management (sleep/wake)

    All model-related operations (loading, compilation, execution) are
    delegated to DiffusionModelRunner.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
        *,
        device_id: int | None = None,
        world_size: int | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        skip_load_model: bool = False,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device_id = int(rank if device_id is None else device_id)
        resolved_world_size = world_size if world_size is not None else getattr(self.od_config, "num_gpus", 1)
        self.distributed_world_size = int(resolved_world_size or 1)
        self.distributed_master_addr = str(master_addr or "localhost")
        resolved_master_port = self.od_config.master_port if master_port is None else master_port
        self.distributed_master_port = int(resolved_master_port or 30005)
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: DiffusionModelRunner | None = None
        self.runtime_v2_runner = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        self.init_device()
        # Create model runner
        self.model_runner = DiffusionModelRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        if not skip_load_model:
            self.load_model(load_format=self.od_config.diffusion_load_format)
            self.init_lora_manager()
        logger.info(f"Worker {self.rank}: Initialization complete.")

    def init_device(self) -> None:
        """Initialize the device and distributed environment."""
        world_size = self.distributed_world_size
        rank = self.rank

        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = self.distributed_master_addr
        os.environ["MASTER_PORT"] = str(self.distributed_master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Setup device
        self.device = current_omni_platform.get_torch_device(self.device_id)
        current_omni_platform.set_device(self.device)

        # Create vllm_config for parallel configuration
        vllm_config = VllmConfig(compilation_config=CompilationConfig())
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        vllm_config.parallel_config.enable_expert_parallel = self.od_config.parallel_config.enable_expert_parallel
        self.vllm_config = vllm_config

        # Initialize distributed environment
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            init_distributed_environment(world_size=world_size, rank=rank)
            logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")

            parallel_config = self.od_config.parallel_config
            init_runtime_v2_collective_runtime(
                backend=getattr(self.od_config, "runtime_v2_collective_backend", "torch"),
                device=self.device,
                max_group_size=min(world_size, 16),
                max_collective_bytes=int(
                    getattr(self.od_config, "runtime_v2_gfc_max_collective_mb", 128)
                )
                * 1024
                * 1024,
            )
            runtime_v2_execution_groups = getattr(self.od_config, "runtime_v2_execution_groups", None)
            runtime_v2_worker_group_id = getattr(self.od_config, "runtime_v2_worker_group_id", None)
            if runtime_v2_execution_groups:
                # runtime_v2 shared-world path:
                # build TP/SP/CFG/... sub-groups from explicit execution-group
                # topology while all workers stay in one global world.
                initialize_model_parallel_from_execution_groups(
                    execution_groups=runtime_v2_execution_groups,
                    current_group_id=runtime_v2_worker_group_id,
                    enable_expert_parallel=parallel_config.enable_expert_parallel,
                )
            else:
                # Legacy path: derive model-parallel groups from scalar config.
                initialize_model_parallel(
                    data_parallel_size=parallel_config.data_parallel_size,
                    cfg_parallel_size=parallel_config.cfg_parallel_size,
                    sequence_parallel_size=parallel_config.sequence_parallel_size,
                    ulysses_degree=parallel_config.ulysses_degree,
                    ring_degree=parallel_config.ring_degree,
                    tensor_parallel_size=parallel_config.tensor_parallel_size,
                    pipeline_parallel_size=parallel_config.pipeline_parallel_size,
                    fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
                    hsdp_replicate_size=parallel_config.hsdp_replicate_size if parallel_config.use_hsdp else 1,
                    enable_expert_parallel=parallel_config.enable_expert_parallel,
                )
            init_workspace_manager(self.device)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None) -> None:
        """Load the diffusion model using DiffusionModelRunner."""
        self._close_runtime_v2_runner()
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )
        process_memory = get_process_gpu_memory(self.local_rank)
        if process_memory is not None:
            logger.info(
                "Worker %d: Process-scoped GPU memory after model loading: %.2f GiB.",
                self.rank,
                process_memory / GiB_bytes,
            )

        # When load_format is "dummy", pipeline will init with custom pipeline later
        if load_format != "dummy":
            assert self.model_runner.pipeline is not None
            self._maybe_init_runtime_v2_runner()

    def init_lora_manager(self) -> None:
        """Initialize the LoRA manager for this worker."""
        if self.model_runner.pipeline is None:
            return
        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.model_runner.pipeline,
            device=self.device,
            dtype=self.od_config.dtype,
            max_cached_adapters=self.od_config.max_cpu_loras,
            lora_path=self.od_config.lora_path,
            lora_scale=self.od_config.lora_scale,
        )

    def generate(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Generate output for the given requests."""
        return self.execute_model(request, self.od_config)

    @classmethod
    def start_profile(cls, trace_path_template: str) -> str:
        """Start profiling for this GPU worker."""
        return CurrentProfiler.start(trace_path_template)

    @classmethod
    def stop_profile(cls) -> dict | None:
        """Stop profiling and return the result dictionary."""
        return CurrentProfiler.stop()

    def _activate_lora_adapter(self, req: OmniDiffusionRequest) -> None:
        if self.lora_manager is None:
            return
        try:
            self.lora_manager.set_active_adapter(req.sampling_params.lora_request, req.sampling_params.lora_scale)
        except Exception as exc:
            if req.sampling_params.lora_request is not None:
                raise
            logger.warning("LoRA activation skipped: %s", exc)

    def execute_model(self, req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """Execute a forward pass by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        self._activate_lora_adapter(req)
        if self.runtime_v2_runner is not None:
            denoise_chunk_size = int(getattr(self.od_config, "runtime_v2_denoise_chunk_size", 1) or 1)
            request_id = self.runtime_v2_submit(req, denoise_chunk_size=denoise_chunk_size)
            return self.runtime_v2_wait(request_id)
        return self.model_runner.execute_model(req)

    def runtime_v2_submit(self, req: OmniDiffusionRequest, denoise_chunk_size: int | None = None) -> str:
        if self.runtime_v2_runner is None:
            raise RuntimeError("runtime_v2 runner is not enabled on this worker")
        if len(req.prompts) != 1:
            raise ValueError("runtime_v2 currently supports single-prompt requests")
        if denoise_chunk_size is None:
            denoise_chunk_size = int(getattr(self.od_config, "runtime_v2_denoise_chunk_size", 1) or 1)
        return self.runtime_v2_runner.submit(req, denoise_chunk_size=denoise_chunk_size)

    def runtime_v2_wait(self, request_id: str, timeout_s: float | None = None) -> DiffusionOutput:
        if self.runtime_v2_runner is None:
            raise RuntimeError("runtime_v2 runner is not enabled on this worker")
        return self.runtime_v2_runner.wait(request_id, timeout_s=timeout_s)

    def runtime_v2_poll_once(self, timeout_s: float = 0.0) -> None:
        if self.runtime_v2_runner is None:
            raise RuntimeError("runtime_v2 runner is not enabled on this worker")
        self.runtime_v2_runner.poll_once(timeout_s=timeout_s)

    def runtime_v2_get_request_status(self, request_id: str) -> tuple[str, Any | None]:
        if self.runtime_v2_runner is None:
            raise RuntimeError("runtime_v2 runner is not enabled on this worker")
        return self.runtime_v2_runner.get_request_status(request_id)

    def runtime_v2_release_request(self, request_id: str) -> None:
        # Direct call (not a getattr soft-probe): RuntimeV2Runner always defines
        # release_request, so a missing method should fail loudly rather than
        # silently no-op and silently reintroduce the unbounded-state leak.
        if self.runtime_v2_runner is not None:
            self.runtime_v2_runner.release_request(request_id)

    def runtime_v2_entrypoint_submit(self, req: OmniDiffusionRequest, denoise_chunk_size: int | None = None) -> str:
        self._activate_lora_adapter(req)
        return self.runtime_v2_submit(req, denoise_chunk_size=denoise_chunk_size)

    def load_weights(self, weights) -> set[str]:
        """Load weights by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.load_weights(weights)

    def remove_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.remove_adapter(adapter_id)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # NOTE (Alex): We have not implemented the API routing
        # for the frontend server yet.
        return self.lora_manager.add_adapter(lora_request)

    def list_loras(self) -> list[int]:
        return self.lora_manager.list_adapters()

    def pin_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.pin_adapter(adapter_id)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep, offloading model weights.

        Args:
            level: Sleep level. Level 1 offloads weights, level 2 also saves buffers.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        process_memory_before_sleep = get_process_gpu_memory(self.local_rank)
        free_bytes_before_sleep = None
        if process_memory_before_sleep is None:
            free_bytes_before_sleep = current_omni_platform.get_free_memory()

        # Save the buffers before level 2 sleep
        if level == 2 and self.model_runner is not None:
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        process_memory_after_sleep = get_process_gpu_memory(self.local_rank)
        if process_memory_before_sleep is not None and process_memory_after_sleep is not None:
            freed_bytes = process_memory_before_sleep - process_memory_after_sleep
            used_bytes = process_memory_after_sleep
            accounting_scope = "process-scoped"
        else:
            free_bytes_after_sleep = current_omni_platform.get_free_memory()
            assert free_bytes_before_sleep is not None
            device_id = self.device.index if self.device.index is not None else 0
            total = current_omni_platform.get_device_total_memory(device_id)
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
            used_bytes = total - free_bytes_after_sleep
            accounting_scope = "device-scoped fallback"
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode (%s) freed %.2f GiB memory, %.2f GiB memory is still in use.",
            accounting_scope,
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get memory pool context for sleep mode support."""
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup distributed environment."""
        self._close_runtime_v2_runner()
        destroy_distributed_env()

    def _supports_runtime_v2(self) -> bool:
        from vllm_omni.diffusion.runtime_v2.registry import supports_runtime_v2_model

        return supports_runtime_v2_model(getattr(self.od_config, "model_class_name", None))

    def _maybe_init_runtime_v2_runner(self) -> None:
        if not bool(getattr(self.od_config, "enable_runtime_v2", False)):
            return
        if getattr(self.od_config, "distributed_executor_backend", None) == "mp":
            logger.info(
                "Worker %s: skipping local runtime_v2 runner because multiproc executor owns the centralized scheduler.",
                self.rank,
            )
            return
        if not self._supports_runtime_v2():
            raise ValueError(
                "runtime_v2 does not support "
                f"model_class_name={getattr(self.od_config, 'model_class_name', None)!r}"
            )
        if self.model_runner is None or self.model_runner.pipeline is None:
            return
        from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

        denoise_chunk_size = int(getattr(self.od_config, "runtime_v2_denoise_chunk_size", 1) or 1)
        scheduler_policy = str(getattr(self.od_config, "runtime_v2_scheduler_policy", "fcfs") or "fcfs").lower()
        self.runtime_v2_runner = RuntimeV2Runner(
            pipeline=self.model_runner.pipeline,
            default_step_chunk_size=denoise_chunk_size,
            scheduler_policy=scheduler_policy,
            vllm_config=self.vllm_config,
            omni_diffusion_config=self.od_config,
        )
        logger.info(
            "Worker %s: runtime_v2 enabled (denoise_chunk_size=%s, policy=%s).",
            self.rank,
            denoise_chunk_size,
            scheduler_policy,
        )

    def _close_runtime_v2_runner(self) -> None:
        runner = self.runtime_v2_runner
        if runner is None:
            return
        try:
            runner.close()
        except Exception as exc:
            logger.warning("Worker %s: failed to close runtime_v2 runner cleanly: %s", self.rank, exc)
        finally:
            self.runtime_v2_runner = None

    def _can_execute_with_runtime_v2(self, req: OmniDiffusionRequest) -> bool:
        if self.runtime_v2_runner is None:
            return False
        if len(req.prompts) != 1:
            raise ValueError(f"runtime_v2 currently supports single-prompt requests, got {len(req.prompts)} prompts")
        return True


class CustomPipelineWorkerExtension:
    def re_init_pipeline(self, custom_pipeline_args: dict[str, Any]) -> None:
        """
        Re-initialize the pipeline with custom arguments.

        Args:
            custom_pipeline_args: Dictionary of arguments for custom pipeline initialization
        """

        # Clean up old pipeline
        if self.model_runner.pipeline is not None:
            del self.model_runner.pipeline
            gc.collect()
            torch.cuda.empty_cache()

        # Get custom pipeline class name
        custom_pipeline_name = custom_pipeline_args["pipeline_class"]

        # Use the DiffusionWorker's load_model method which handles the forward context
        self.load_model(
            load_format="custom_pipeline",
            custom_pipeline_name=custom_pipeline_name,
        )
        self.init_lora_manager()


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        broadcast_handle,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        self.od_config = od_config
        self.gpu_id = gpu_id

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)

        # Initialize MessageQueue reader from handle
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        self.result_mq = None
        self.result_mq_handle = None

        # Setup result sender (only for rank 0)
        if gpu_id == 0:
            self.result_mq = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
            self.result_mq_handle = self.result_mq.export_handle()
            logger.info(f"Worker {gpu_id} created result MessageQueue")

        assert od_config.master_port is not None

        # Create worker using WorkerWrapperBase for extension support
        self.worker = self._create_worker(gpu_id, od_config, worker_extension_cls, custom_pipeline_args)
        self._running = True
        self._pending_runtime_v2: dict[str, _PendingRuntimeV2Submission] = {}
        self._runtime_v2_active_poll_s = 0.01
        self._runtime_v2_idle_poll_s = 0.05

    def _create_worker(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        worker_extension_cls: str | None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> DiffusionWorker:
        """Create a worker instance. Override in subclasses for different worker types."""
        wrapper = WorkerWrapperBase(
            gpu_id=gpu_id,
            od_config=od_config,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
        )
        return wrapper

    def return_result(self, output: DiffusionOutput | dict[str, Any]):
        """Reply to client, only on rank 0."""
        if self.result_mq is not None:
            target = output
            if isinstance(output, DiffusionOutput):
                try:
                    pack_diffusion_output_shm(output)
                except Exception as e:
                    logger.warning("SHM pack failed, falling back to raw enqueue: %s", e)
            elif isinstance(output, dict):
                payload = output.get("output")
                if isinstance(payload, DiffusionOutput):
                    try:
                        pack_diffusion_output_shm(payload)
                    except Exception as e:
                        logger.warning("SHM pack failed, falling back to raw enqueue: %s", e)
                target = output
            self.result_mq.enqueue(target)

    def recv_message(self, timeout_s: float | None = None):
        """Receive messages from broadcast queue."""
        if timeout_s is None:
            return self.mq.dequeue(indefinite=True)
        return self.mq.dequeue(timeout=timeout_s)

    def _resolve_rpc_routing(self, rpc_request: dict) -> tuple[bool, bool]:
        output_rank = rpc_request.get("output_rank")
        exec_all_ranks = rpc_request.get("exec_all_ranks", False)
        should_execute = exec_all_ranks or output_rank is None or output_rank == self.gpu_id
        should_reply = (output_rank is None or output_rank == self.gpu_id) and self.result_mq is not None
        return should_execute, should_reply

    def execute_rpc(self, rpc_request: dict) -> tuple[object | None, bool]:
        """Execute an RPC request and indicate whether to reply."""
        method = rpc_request["method"]
        args = rpc_request.get("args", ())
        kwargs = rpc_request.get("kwargs", {})
        should_execute, should_reply = self._resolve_rpc_routing(rpc_request)

        if not should_execute:
            return None, False

        try:
            # Use execute_method from WorkerWrapperBase for consistent method resolution
            result = self.worker.execute_method(method, *args, **kwargs)
            return result, should_reply
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            raise e

    def _try_handle_runtime_v2_generate_rpc(self, rpc_request: dict) -> bool:
        if rpc_request.get("method") != "generate":
            return False

        args = rpc_request.get("args", ())
        if not args or not isinstance(args[0], OmniDiffusionRequest):
            return False
        req = args[0]
        should_execute, should_reply = self._resolve_rpc_routing(rpc_request)
        if not should_execute:
            return True
        try:
            if not self.worker._can_execute_with_runtime_v2(req):
                return False
        except Exception as exc:
            logger.error("runtime_v2 request validation failed: %s", exc, exc_info=True)
            if should_reply and self.result_mq is not None:
                self.return_result(
                    {
                        "type": "generation_result",
                        "scheduler_req_id": rpc_request.get("scheduler_req_id"),
                        "output": DiffusionOutput(error=str(exc)),
                    }
                )
            return True
        if req.sampling_params.lora_request is not None:
            message = "runtime_v2 entrypoint mode does not support LoRA requests yet"
            logger.error(message)
            if should_reply and self.result_mq is not None:
                self.return_result(
                    {
                        "type": "generation_result",
                        "scheduler_req_id": rpc_request.get("scheduler_req_id"),
                        "output": DiffusionOutput(error=message),
                    }
                )
            return True

        try:
            runtime_request_id = self.worker.runtime_v2_entrypoint_submit(req)
            scheduler_req_id = str(rpc_request.get("scheduler_req_id") or runtime_request_id)
            self._pending_runtime_v2[runtime_request_id] = _PendingRuntimeV2Submission(
                scheduler_req_id=scheduler_req_id,
                should_reply=bool(should_reply),
            )
            logger.info(
                "runtime_v2 entrypoint submit: scheduler_req_id=%s runtime_request_id=%s pending=%d",
                scheduler_req_id,
                runtime_request_id,
                len(self._pending_runtime_v2),
            )
            return True
        except Exception as exc:
            logger.error("runtime_v2 entrypoint submit failed: %s", exc, exc_info=True)
            if should_reply and self.result_mq is not None:
                self.return_result(
                    {
                        "type": "generation_result",
                        "scheduler_req_id": rpc_request.get("scheduler_req_id"),
                        "output": DiffusionOutput(error=str(exc)),
                    }
                )
            return True

    def _poll_runtime_v2_scheduler(self, timeout_s: float) -> None:
        if self.worker.runtime_v2_runner is None:
            return
        try:
            self.worker.runtime_v2_poll_once(timeout_s=timeout_s)
        except Exception as exc:
            logger.error("runtime_v2 entrypoint poll failed: %s", exc, exc_info=True)
            self._fail_all_pending_runtime_v2(reason=f"runtime_v2 poll failed: {exc}")
            self._running = False

    def _drain_runtime_v2_completions(self) -> None:
        if not self._pending_runtime_v2:
            return
        for runtime_request_id in list(self._pending_runtime_v2.keys()):
            pending = self._pending_runtime_v2.get(runtime_request_id)
            if pending is None:
                continue
            try:
                status, payload = self.worker.runtime_v2_get_request_status(runtime_request_id)
            except Exception as exc:
                logger.error(
                    "runtime_v2 entrypoint get_request_status failed: request_id=%s error=%s",
                    runtime_request_id,
                    exc,
                    exc_info=True,
                )
                status, payload = "failed", str(exc)

            if status == "pending":
                continue

            if status == "failed":
                output = DiffusionOutput(error=str(payload))
            elif status == "finished":
                output = payload if isinstance(payload, DiffusionOutput) else DiffusionOutput(output=payload)
            else:
                continue

            if pending.should_reply and self.result_mq is not None:
                self.return_result(
                    {
                        "type": "generation_result",
                        "scheduler_req_id": pending.scheduler_req_id,
                        "output": output,
                    }
                )
            self._release_runtime_v2_request(runtime_request_id)
            self._pending_runtime_v2.pop(runtime_request_id, None)
            logger.info(
                "runtime_v2 entrypoint complete: scheduler_req_id=%s runtime_request_id=%s pending=%d",
                pending.scheduler_req_id,
                runtime_request_id,
                len(self._pending_runtime_v2),
            )

    def _fail_all_pending_runtime_v2(self, reason: str) -> None:
        for runtime_request_id, pending in list(self._pending_runtime_v2.items()):
            logger.warning(
                "runtime_v2 force-failing pending request: scheduler_req_id=%s runtime_request_id=%s reason=%s",
                pending.scheduler_req_id,
                runtime_request_id,
                reason,
            )
            if pending.should_reply and self.result_mq is not None:
                self.return_result(
                    {
                        "type": "generation_result",
                        "scheduler_req_id": pending.scheduler_req_id,
                        "output": DiffusionOutput(error=f"{reason}: request_id={runtime_request_id}"),
                    }
                )
            self._release_runtime_v2_request(runtime_request_id)
            self._pending_runtime_v2.pop(runtime_request_id, None)

    def _release_runtime_v2_request(self, runtime_request_id: str) -> None:
        try:
            self.worker.runtime_v2_release_request(runtime_request_id)
        except Exception as exc:
            logger.warning(
                "runtime_v2 request release failed: request_id=%s error=%s",
                runtime_request_id,
                exc,
                exc_info=True,
            )

    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers."""
        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            if self.worker.runtime_v2_runner is not None:
                self._poll_runtime_v2_scheduler(timeout_s=0.0)
                self._drain_runtime_v2_completions()
                if not self._running:
                    break

            msg = None
            recv_timeout_s = None
            if self.worker.runtime_v2_runner is not None:
                recv_timeout_s = (
                    self._runtime_v2_active_poll_s if self._pending_runtime_v2 else self._runtime_v2_idle_poll_s
                )
            try:
                msg = self.recv_message(timeout_s=recv_timeout_s)
            except (TimeoutError, zmq.error.Again):
                continue
            except Exception as e:
                logger.error(
                    f"Error receiving message in worker loop: {e}",
                    exc_info=True,
                )
                continue

            if msg is None:
                continue
            if hasattr(msg, "__len__") and len(msg) == 0:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            # Route message based on type
            if isinstance(msg, dict) and msg.get("type") == "rpc":
                if self._try_handle_runtime_v2_generate_rpc(msg):
                    continue
                try:
                    result, should_reply = self.execute_rpc(msg)
                    if should_reply:
                        if msg.get("method") == "generate":
                            self.return_result(
                                {
                                    "type": "generation_result",
                                    "scheduler_req_id": msg.get("scheduler_req_id"),
                                    "output": result,
                                }
                            )
                        else:
                            self.return_result(result)
                except Exception as e:
                    logger.error(f"Error processing RPC: {e}", exc_info=True)
                    if self.result_mq is not None:
                        if msg.get("method") == "generate":
                            self.return_result(
                                {
                                    "type": "generation_result",
                                    "scheduler_req_id": msg.get("scheduler_req_id"),
                                    "output": DiffusionOutput(error=str(e)),
                                }
                            )
                        else:
                            self.return_result(DiffusionOutput(error=str(e)))

            elif isinstance(msg, dict) and msg.get("type") == "shutdown":
                logger.info("Worker %s: Received shutdown message", self.gpu_id)
                self._running = False
                continue

            else:
                # Handle generation request
                try:
                    output = self.worker.execute_model(msg, self.od_config)
                except Exception as e:
                    logger.error(
                        f"Error executing forward in event loop: {e}",
                        exc_info=True,
                    )
                    output = DiffusionOutput(error=str(e))

                try:
                    self.return_result(output)
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error sending reply: {e}")
                    continue

        self._fail_all_pending_runtime_v2(reason="worker shutdown")
        logger.info("event loop terminated.")
        try:
            self.worker.shutdown()
        except Exception as exc:
            logger.warning("Worker %s: Shutdown encountered an error: %s", self.gpu_id, exc)
        self.context.term()

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> None:
        """Worker initialization and execution loops."""
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        worker_proc = WorkerProc(
            od_config,
            gpu_id=rank,
            broadcast_handle=broadcast_handle,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
                "result_handle": worker_proc.result_mq_handle if rank == 0 else None,
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")


class WorkerWrapperBase:
    """
    Wrapper base class that creates DiffusionWorker with optional worker_extension_cls support.
    This enables dynamic inheritance for DiffusionWorker to extend with custom functionality.
    """

    def __init__(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        *,
        local_rank: int | None = None,
        rank: int | None = None,
        device_id: int | None = None,
        world_size: int | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        base_worker_class: type = DiffusionWorker,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        """
        Initialize WorkerWrapperBase with support for worker extensions.

        Args:
            gpu_id: GPU device ID
            od_config: OmniDiffusionConfig configuration
            local_rank: Optional local rank override for distributed session
            rank: Optional distributed rank override
            device_id: Optional device ID override used for CUDA device binding
            world_size: Optional distributed world size override
            master_addr: Optional distributed master address override
            master_port: Optional distributed master port override
            worker_extension_cls: Optional qualified name of worker extension class
            custom_pipeline_args: Optional arguments for custom pipeline initialization
        """
        self.gpu_id = gpu_id
        self.od_config = od_config
        self.base_worker_class = base_worker_class
        self.worker_extension_cls = worker_extension_cls
        self.custom_pipeline_args = custom_pipeline_args

        # Prepare worker class with extension support
        worker_class = self._prepare_worker_class()

        # Create the actual worker instance
        # When custom_pipeline_args is provided, skip initial model loading
        # since re_init_pipeline will handle it. This avoids allocating memory
        # through CuMemAllocator twice, which causes assertion failures in
        # sleep mode.
        worker_local_rank = gpu_id if local_rank is None else int(local_rank)
        worker_rank = gpu_id if rank is None else int(rank)
        worker_device_id = gpu_id if device_id is None else int(device_id)
        self.worker = worker_class(
            local_rank=worker_local_rank,
            rank=worker_rank,
            od_config=od_config,
            device_id=worker_device_id,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            skip_load_model=(self.custom_pipeline_args is not None),
        )

        # Re-initialize pipeline with custom pipeline if provided
        if self.custom_pipeline_args is not None:
            self.worker.re_init_pipeline(self.custom_pipeline_args)

    def _prepare_worker_class(self) -> type:
        """
        Prepare the worker class with optional extension.
        Dynamically extends GPUWorker with worker_extension_cls if provided.

        Returns:
            The worker class (potentially extended)
        """
        worker_class = self.base_worker_class

        # If custom_pipeline_args is provided, use CustomPipelineWorkerExtension
        if self.custom_pipeline_args is not None:
            # Set worker_extension_cls to CustomPipelineWorkerExtension if not already set
            if self.worker_extension_cls is None:
                self.worker_extension_cls = CustomPipelineWorkerExtension

        if self.worker_extension_cls:
            if isinstance(self.worker_extension_cls, str):
                worker_extension_cls = resolve_obj_by_qualname(self.worker_extension_cls)
            else:
                worker_extension_cls = self.worker_extension_cls
            extended_calls = []

            if worker_extension_cls not in worker_class.__bases__:
                # Check for conflicts between worker and extension
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    if hasattr(worker_class, attr):
                        logger.warning(
                            f"Worker class {worker_class} already has attribute "
                            f"{attr}, which may conflict with worker extension "
                            f"class {worker_extension_cls}."
                        )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)

                # Dynamically inherit the worker extension class
                class_name = f"{worker_class.__name__}With{worker_extension_cls.__name__}"
                worker_class = type(class_name, (worker_extension_cls, worker_class), {})
                logger.info(
                    "Created extended worker class %s from %s for extended calls %s",
                    class_name,
                    worker_extension_cls,
                    extended_calls,
                )

        return worker_class

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Generate output for the given requests.

        Args:
            requests: List of diffusion requests

        Returns:
            DiffusionOutput with generated results
        """
        return self.worker.generate(requests)

    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.

        Args:
            reqs: List of diffusion requests
            od_config: OmniDiffusionConfig configuration

        Returns:
            DiffusionOutput with generated results
        """
        return self.worker.execute_model(reqs, od_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterable of (name, tensor) tuples

        Returns:
            Set of loaded weight names
        """
        return self.worker.load_weights(weights)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache.
                Currently only support level 1.

        Returns:
            True on success
        """
        return self.worker.sleep(level)

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.

        Returns:
            True on success
        """
        return self.worker.wake_up(tags)

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        return self.worker.shutdown()

    def execute_method(self, method: str | bytes, *args, **kwargs) -> Any:
        """
        Execute a method on the worker.

        Args:
            method: Method name (str) or serialized callable (bytes)

        Returns:
            Result of the method execution (type depends on the method)

        Raises:
            Exception: If method execution fails
        """
        try:
            # Method resolution order:
            # 1. If method is defined in this class, it will be called directly
            # 2. Otherwise, since we define `__getattr__` and redirect attribute
            #    query to `self.worker`, the method will be called on the worker
            assert isinstance(method, str), "Method must be str"
            func = getattr(self.worker, method)
            return func(*args, **kwargs)

        except Exception as e:
            msg = f"Error executing method {method!r}. This might cause issues in distributed execution."
            logger.exception(msg)
            raise e

    def __getattr__(self, attr: str):
        """Delegate attribute access to the wrapped worker."""
        return getattr(self.worker, attr)
