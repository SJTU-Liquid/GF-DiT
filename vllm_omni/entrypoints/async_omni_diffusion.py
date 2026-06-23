# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Async entrypoint for vLLM-Omni diffusion model inference.

Provides an asynchronous interface for running diffusion models,
enabling concurrent request handling and streaming generation.
"""

import asyncio
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import CollectedDiffusionResult, DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.stage_utils import maybe_dump_to_shm
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.lora.request import LoRARequest
from vllm_omni.metrics import StageRequestStats, count_tokens_from_outputs
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)
_CPU_THREAD_TRACE_MIN_NS = 10_000_000

_RuntimeV2ResultMeta = dict[str, Any]
_RuntimeV2DirectResultConfig = dict[str, Any]
_PostprocessWorkItem = tuple[str, CollectedDiffusionResult, int, _RuntimeV2ResultMeta]
_PostprocessResultItem = tuple[str, OmniRequestOutput | None, str | None, int, int, int]


def _finalize_collected_in_subprocess(
    collected: CollectedDiffusionResult,
    od_config: OmniDiffusionConfig,
) -> list[OmniRequestOutput]:
    """Run DiffusionEngine.finalize_collected in a dedicated subprocess."""
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.registry import get_diffusion_post_process_func
    from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm

    # Build a lightweight engine shell: finalize_collected/_build_outputs only
    # need od_config + post_process_func, not executor/model weights.
    unpack_diffusion_output_shm(collected.output)
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = od_config
    engine.post_process_func = get_diffusion_post_process_func(od_config)
    engine.pre_process_func = None
    return DiffusionEngine.finalize_collected(engine, collected)


def _encode_video_outputs_inplace(
    result: OmniRequestOutput,
    collected: CollectedDiffusionResult,
) -> None:
    """Encode video frames to base64 MP4 inside the postprocess subprocess.

    Moves MP4 muxing off the API server event loop. Only the no-audio case is
    handled here; audio (e.g. LTX-2) keeps its raw frames and is encoded on the
    API-server path. On failure the raw frames are left intact so the API server
    can still encode.
    """
    sampling_params = getattr(collected.request, "sampling_params", None)
    num_frames = getattr(sampling_params, "num_frames", None)
    if not num_frames or int(num_frames) <= 1:
        return  # not a video request
    if not getattr(result, "images", None):
        return
    if (result.multimodal_output or {}).get("audio") is not None:
        return  # leave audio muxing to the API-server encode path
    fps = int(
        getattr(sampling_params, "fps", None)
        or getattr(sampling_params, "frame_rate", None)
        or 24
    )
    from vllm_omni.entrypoints.openai.video_api_utils import encode_request_output_videos_b64

    video_b64 = encode_request_output_videos_b64(result.images, fps)
    custom = dict(result.custom_output)
    custom["video_b64"] = video_b64
    result.custom_output = custom
    # Drop raw frames so only the small encoded payload crosses the IPC boundary.
    result.images = []


def _count_prompt_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    total = 0
    for out in engine_outputs:
        try:
            prompt_token_ids = getattr(out, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                total += len(prompt_token_ids)
        except Exception:
            pass
    return total


def _make_stage_request_stats(
    req_output: list[Any],
    stage_gen_time_ms: float,
    batch_id: int,
    batch_size: int,
    rx_decode_time_ms: float,
    rx_transfer_bytes: int,
    rx_in_flight_time_ms: float,
) -> StageRequestStats:
    return StageRequestStats(
        num_tokens_in=_count_prompt_tokens_from_outputs(req_output),
        num_tokens_out=count_tokens_from_outputs(req_output),
        stage_gen_time_ms=stage_gen_time_ms,
        batch_id=batch_id,
        batch_size=batch_size,
        rx_decode_time_ms=rx_decode_time_ms,
        rx_transfer_bytes=rx_transfer_bytes,
        rx_in_flight_time_ms=rx_in_flight_time_ms,
        stage_stats=None,
    )


def _output_strip_for_stage(
    output: OmniRequestOutput,
    *,
    final_output: bool,
    final_output_type: str | None,
) -> OmniRequestOutput:
    if final_output and final_output_type != "text":
        return output
    if getattr(output, "finished", False):
        return output
    mm_output = getattr(output, "multimodal_output", None)
    if mm_output is not None:
        output.multimodal_output = {}
    custom_out = getattr(output, "_custom_output", None)
    if custom_out is not None:
        output._custom_output = {}
    outputs = getattr(output, "outputs", None)
    if outputs is not None:
        for out in outputs:
            if getattr(out, "multimodal_output", None):
                out.multimodal_output = {}
            if getattr(out, "_custom_output", None):
                out._custom_output = {}
    return output


def _build_direct_stage_result_message(
    *,
    request_id: str,
    output: OmniRequestOutput,
    meta: _RuntimeV2ResultMeta,
    cfg: _RuntimeV2DirectResultConfig,
) -> dict[str, Any]:
    start_ts = float(meta.get("start_ts", time.time()))
    stage_gen_time_ms = max(0.0, (time.time() - start_ts) * 1000.0)
    rx_decode_time_ms = float(meta.get("rx_decode_time_ms", 0.0))
    rx_transfer_bytes = int(meta.get("rx_transfer_bytes", 0))
    rx_in_flight_time_ms = float(meta.get("rx_in_flight_time_ms", 0.0))
    batch_id = int(meta.get("batch_id", 0))
    batch_size = int(meta.get("batch_size", 1))

    stripped = _output_strip_for_stage(
        output,
        final_output=bool(cfg.get("final_output", False)),
        final_output_type=cfg.get("final_output_type"),
    )
    req_outputs = [stripped]
    metrics = _make_stage_request_stats(
        req_output=req_outputs,
        stage_gen_time_ms=stage_gen_time_ms,
        batch_id=batch_id,
        batch_size=batch_size,
        rx_decode_time_ms=rx_decode_time_ms,
        rx_transfer_bytes=rx_transfer_bytes,
        rx_in_flight_time_ms=rx_in_flight_time_ms,
    )
    use_shm, payload = maybe_dump_to_shm(req_outputs, int(cfg.get("shm_threshold_bytes", 65536)))
    message: dict[str, Any] = {
        "request_id": request_id,
        "stage_id": int(cfg["stage_id"]),
        "metrics": metrics,
    }
    if use_shm:
        message["engine_outputs_shm"] = payload
    else:
        message["engine_outputs"] = payload
    return message


def _runtime_v2_postprocess_loop(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    od_config: OmniDiffusionConfig,
    direct_result_queue: Any | None = None,
    direct_result_cfg: _RuntimeV2DirectResultConfig | None = None,
    direct_result_zmq_endpoint: str | None = None,
) -> None:
    direct_enabled = direct_result_cfg is not None and (
        direct_result_queue is not None or direct_result_zmq_endpoint is not None
    )
    stage_id = int(direct_result_cfg["stage_id"]) if direct_enabled else -1
    direct_queue: Any | None = direct_result_queue
    zmq_ctx = None
    if direct_enabled and direct_queue is None and direct_result_zmq_endpoint is not None:
        import zmq

        from vllm_omni.entrypoints.zmq_utils import create_zmq_queue

        zmq_ctx = zmq.Context()
        direct_queue = create_zmq_queue(zmq_ctx, direct_result_zmq_endpoint, zmq.PUSH)
    if direct_enabled and direct_queue is None:
        raise RuntimeError("runtime_v2 direct result channel is not configured")
    try:
        while True:
            item = in_queue.get()
            if item is None:
                return
            work_item: _PostprocessWorkItem = item
            request_id, collected, enqueued_ns, result_meta = work_item
            start_ns = time.monotonic_ns()
            try:
                result_list = _finalize_collected_in_subprocess(collected, od_config)
                if not result_list:
                    raise RuntimeError(f"Diffusion collect returned no outputs for request {request_id}")
                result = result_list[0]
                if not result.request_id:
                    result.request_id = request_id
                # Encode video off the API event loop. Standalone mode always
                # delivers a final result; in direct/staged mode this applies
                # only to the final stage output — an intermediate stage keeps
                # raw frames for the downstream stage to consume.
                if (not direct_enabled) or bool(direct_result_cfg.get("final_output")):
                    try:
                        _encode_video_outputs_inplace(result, collected)
                    except Exception as exc:
                        logger.warning(
                            "runtime_v2 postprocess video encode failed for %s; "
                            "falling back to API-server encode: %s",
                            request_id,
                            exc,
                        )
                if direct_enabled:
                    direct_queue.put(
                        _build_direct_stage_result_message(
                            request_id=request_id,
                            output=result,
                            meta=result_meta,
                            cfg=direct_result_cfg,
                        )
                    )
                else:
                    out_queue.put((request_id, result, None, enqueued_ns, start_ns, time.monotonic_ns()))
            except BaseException as exc:
                err_msg = f"Diffusion postprocess failed: {exc}"
                if direct_enabled:
                    direct_queue.put(
                        {
                            "request_id": request_id,
                            "stage_id": stage_id,
                            "error": err_msg,
                        }
                    )
                else:
                    out_queue.put(
                        (
                            request_id,
                            None,
                            err_msg,
                            enqueued_ns,
                            start_ns,
                            time.monotonic_ns(),
                        )
                    )
    finally:
        if direct_result_zmq_endpoint is not None and direct_queue is not None:
            close_fn = getattr(direct_queue, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if zmq_ctx is not None:
            try:
                zmq_ctx.term()
            except Exception:
                pass


class AsyncOmniDiffusion:
    """Async entry point for vLLM-Omni diffusion model inference.

    This class provides an asynchronous interface for running diffusion models,
    enabling concurrent request handling. It wraps the DiffusionEngine and
    provides async methods for image generation.

    Args:
        model: Model name or path to load
        od_config: Optional OmniDiffusionConfig. If not provided, it will be
            created from kwargs
        **kwargs: Additional keyword arguments passed to OmniDiffusionConfig

    Example:
        >>> async_diffusion = AsyncOmniDiffusion(model="Qwen/Qwen-Image")
        >>> result = await async_diffusion.generate(
        ...     prompt="A beautiful sunset over the ocean",
        ...     request_id="req-1",
        ... )
        >>> print(result.images)
    """

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig | None = None,
        **kwargs: Any,
    ):
        self.model = model

        # Capture stage info from kwargs before they might be filtered out
        stage_id = kwargs.get("stage_id")
        engine_input_source = kwargs.get("engine_input_source")
        cfg_kv_collect_func = kwargs.pop("cfg_kv_collect_func", None)
        runtime_v2_worker_mode = bool(kwargs.pop("runtime_v2_worker_mode", False))
        runtime_v2_direct_result_queue = kwargs.pop("runtime_v2_direct_result_queue", None)
        runtime_v2_direct_result_zmq_endpoint = kwargs.pop("runtime_v2_direct_result_zmq_endpoint", None)
        runtime_v2_direct_result_stage_id = kwargs.pop("runtime_v2_direct_result_stage_id", None)
        runtime_v2_direct_result_shm_threshold_bytes = int(kwargs.pop("runtime_v2_direct_result_shm_threshold_bytes", 65536))
        runtime_v2_direct_result_final_output = bool(kwargs.pop("runtime_v2_direct_result_final_output", False))
        runtime_v2_direct_result_final_output_type = kwargs.pop("runtime_v2_direct_result_final_output_type", None)

        # Build config
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(model=model, **kwargs)
        elif isinstance(od_config, dict):
            # If config is dict, check it too (priority to kwargs if both exist)
            if stage_id is None:
                stage_id = od_config.get("stage_id")
            if engine_input_source is None:
                engine_input_source = od_config.get("engine_input_source")
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Inject stage info into omni_kv_config if present
        if stage_id is not None:
            self.od_config.omni_kv_config.setdefault("stage_id", stage_id)
        if engine_input_source is not None:
            self.od_config.omni_kv_config.setdefault("engine_input_source", engine_input_source)

        # Diffusers-style models expose `model_index.json` with `_class_name`.
        # Non-diffusers models (e.g. Bagel, NextStep) only have `config.json`,
        # so we fall back to reading that and mapping model_type manually.
        try:
            config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
            if config_dict is not None:
                if od_config.model_class_name is None:
                    od_config.model_class_name = config_dict.get("_class_name", None)
                od_config.update_multimodal_support()

                tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
                od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            od_config.tf_model_config = TransformerConfig.from_dict(cfg)
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            # Bagel/NextStep models don't have a model_index.json, so we set the pipeline class name manually
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif model_type == "nextstep":
                if od_config.model_class_name is None:
                    od_config.model_class_name = "NextStep11Pipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif architectures and len(architectures) == 1:
                od_config.model_class_name = architectures[0]
            else:
                raise

        if cfg_kv_collect_func is not None:
            od_config.cfg_kv_collect_func = cfg_kv_collect_func

        # Initialize engine
        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)
        uses_runtime_v2_runner = bool(getattr(self.engine.executor, "runtime_v2_runner", None) is not None)

        # When runtime_v2 is active, keep the scheduler behind a single
        # background control thread, independent of whether this object is used
        # inside OmniStage worker mode or standalone async serving.
        self._runtime_v2_worker_mode = runtime_v2_worker_mode or uses_runtime_v2_runner
        self._control_executor = None if self._runtime_v2_worker_mode else ThreadPoolExecutor(max_workers=1)
        self._collect_executor = None if self._runtime_v2_worker_mode else ThreadPoolExecutor(max_workers=32)
        self._runtime_v2_direct_result_queue = runtime_v2_direct_result_queue
        self._runtime_v2_direct_result_zmq_endpoint = (
            str(runtime_v2_direct_result_zmq_endpoint)
            if runtime_v2_direct_result_zmq_endpoint is not None
            else None
        )
        self._runtime_v2_direct_result_cfg: _RuntimeV2DirectResultConfig | None = None
        self._runtime_v2_direct_output_enabled = False
        self._submit_futures: dict[str, asyncio.Future[str] | None] = {}
        self._pending_submissions: dict[str, str] = {}
        self._pending_lock = asyncio.Lock()
        self._runtime_v2_submit_queue: queue.Queue[tuple[str, OmniDiffusionRequest]] | None = None
        self._runtime_v2_postprocess_in_queue: mp.Queue | None = None
        self._runtime_v2_postprocess_out_queue: mp.Queue | None = None
        self._runtime_v2_postprocess_processes: list[mp.Process] = []
        # Protects runtime_v2 shared maps touched by control/caller threads.
        self._runtime_v2_state_lock = threading.Lock()
        self._runtime_v2_request_result_meta: dict[str, _RuntimeV2ResultMeta] = {}
        self._runtime_v2_active_requests: set[str] = set()
        self._runtime_v2_submission_ids: dict[str, str] = {}
        self._runtime_v2_postprocess_inflight: set[str] = set()
        self._runtime_v2_cancelled_requests: set[str] = set()
        self._runtime_v2_completed_results: dict[str, OmniRequestOutput] = {}
        self._runtime_v2_failed_requests: dict[str, BaseException] = {}
        self._runtime_v2_stop = threading.Event()
        self._runtime_v2_background_error: BaseException | None = None
        self._runtime_v2_control_thread: threading.Thread | None = None
        self._runtime_v2_control_trace_enabled = str(os.environ.get("VLLM_RUNTIME_V2_CONTROL_TRACE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._closed = False

        if self._runtime_v2_worker_mode:
            ctx = mp.get_context("spawn")
            has_direct_channel = (
                runtime_v2_direct_result_queue is not None or self._runtime_v2_direct_result_zmq_endpoint is not None
            )
            if runtime_v2_direct_result_stage_id is not None and has_direct_channel:
                self._runtime_v2_direct_result_cfg = {
                    "stage_id": int(runtime_v2_direct_result_stage_id),
                    "shm_threshold_bytes": runtime_v2_direct_result_shm_threshold_bytes,
                    "final_output": runtime_v2_direct_result_final_output,
                    "final_output_type": runtime_v2_direct_result_final_output_type,
                }
                self._runtime_v2_direct_output_enabled = True
            subprocess_direct_result_queue = (
                self._runtime_v2_direct_result_queue
                if self._runtime_v2_direct_output_enabled and self._runtime_v2_direct_result_zmq_endpoint is None
                else None
            )
            self._runtime_v2_submit_queue = queue.Queue()
            self._runtime_v2_postprocess_in_queue = ctx.Queue()
            self._runtime_v2_postprocess_out_queue = ctx.Queue()
            # Pool of postprocess+encode subprocesses. All processes pull from
            # the shared in_queue, so a free worker grabs the next item — this
            # load-balances naturally and avoids head-of-line blocking when one
            # encode runs long.
            num_gpus = int(self.od_config.num_gpus or 1)
            # Validated >= 1 in OmniDiffusionConfig.__post_init__ (None coerced to 2,
            # < 1 rejected), so read it directly; the getattr/`or` default only guards a
            # config object built without that validation (e.g. in tests).
            workers_per_gpu = int(getattr(self.od_config, "runtime_v2_postprocess_workers_per_gpu", 2) or 2)
            pool_size = max(1, num_gpus * workers_per_gpu)
            for idx in range(pool_size):
                process = ctx.Process(
                    target=_runtime_v2_postprocess_loop,
                    args=(
                        self._runtime_v2_postprocess_in_queue,
                        self._runtime_v2_postprocess_out_queue,
                        self.od_config,
                        subprocess_direct_result_queue,
                        self._runtime_v2_direct_result_cfg,
                        self._runtime_v2_direct_result_zmq_endpoint,
                    ),
                    name=f"runtime-v2-postprocess-{idx}",
                    daemon=True,
                )
                process.start()
                self._runtime_v2_postprocess_processes.append(process)
            self._runtime_v2_control_thread = threading.Thread(
                target=self._runtime_v2_control_loop,
                name="runtime-v2-control",
                daemon=True,
            )
            self._runtime_v2_control_thread.start()

        logger.info(
            "AsyncOmniDiffusion initialized with model: %s (runtime_v2_worker_mode=%s, "
            "runtime_v2_postprocess_pool_size=%s, runtime_v2_direct_output_enabled=%s)",
            model,
            self._runtime_v2_worker_mode,
            len(self._runtime_v2_postprocess_processes),
            self._runtime_v2_direct_output_enabled,
        )

    def _build_request(
        self,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str | None = None,
        lora_request: LoRARequest | None = None,
    ) -> tuple[str, OmniDiffusionRequest]:
        if request_id is None:
            request_id = f"diff-{uuid.uuid4().hex[:16]}"

        if sampling_params.guidance_scale:
            sampling_params.guidance_scale_provided = True

        if lora_request is not None:
            sampling_params.lora_request = lora_request

        request = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[request_id],
        )
        return request_id, request

    def _ensure_runtime_v2_threads_healthy(self) -> None:
        err = self._runtime_v2_background_error
        if err is not None:
            raise RuntimeError("runtime_v2 async diffusion background thread failed") from err
        processes = self._runtime_v2_postprocess_processes
        if self._runtime_v2_worker_mode and processes and not self._runtime_v2_stop.is_set():
            dead = [p for p in processes if not p.is_alive()]
            if dead:
                raise RuntimeError(
                    "runtime_v2 async diffusion postprocess subprocess exited unexpectedly "
                    f"(dead={len(dead)}/{len(processes)}, exitcodes={[p.exitcode for p in dead]})"
                )

    def _drain_runtime_v2_postprocess_results(self) -> int:
        if self._runtime_v2_direct_output_enabled:
            return 0
        out_queue = self._runtime_v2_postprocess_out_queue
        if out_queue is None:
            return 0
        drained = 0
        while True:
            try:
                item = out_queue.get_nowait()
            except queue.Empty:
                return drained
            result_item: _PostprocessResultItem = item
            request_id, result, error_msg, enqueued_ns, started_ns, done_ns = result_item
            drained += 1
            now_ns = time.monotonic_ns()
            dropped = False
            with self._runtime_v2_state_lock:
                if request_id in self._runtime_v2_cancelled_requests:
                    dropped = True
                    self._runtime_v2_cancelled_requests.discard(request_id)
                elif error_msg is not None:
                    error = RuntimeError(error_msg)
                    self._runtime_v2_failed_requests[request_id] = error
                elif result is not None:
                    self._runtime_v2_completed_results[request_id] = result
                self._runtime_v2_postprocess_inflight.discard(request_id)
                self._runtime_v2_request_result_meta.pop(request_id, None)
            total_elapsed_ns = max(0, now_ns - enqueued_ns)
            subprocess_elapsed_ns = max(0, done_ns - started_ns)
            recv_overhead_ns = max(0, now_ns - done_ns)
            if dropped:
                self._control_trace(
                    "postprocess_late_drop",
                    request_id=request_id,
                    elapsed_ns=recv_overhead_ns,
                    total_elapsed_ns=total_elapsed_ns,
                    subprocess_elapsed_ns=subprocess_elapsed_ns,
                )
                self._thread_trace(
                    "postprocess_late_drop",
                    request_id=request_id,
                    elapsed_ns=recv_overhead_ns,
                    total_elapsed_ns=total_elapsed_ns,
                    subprocess_elapsed_ns=subprocess_elapsed_ns,
                )
                if self._runtime_v2_direct_output_enabled:
                    self._runtime_v2_finalize_request(request_id)
                continue
            self._control_trace(
                "postprocess_result_recv",
                request_id=request_id,
                status="error" if error_msg is not None else "ok",
                elapsed_ns=recv_overhead_ns,
                total_elapsed_ns=total_elapsed_ns,
                subprocess_elapsed_ns=subprocess_elapsed_ns,
            )
            self._thread_trace(
                "postprocess_result_recv",
                request_id=request_id,
                status="error" if error_msg is not None else "ok",
                elapsed_ns=recv_overhead_ns,
                total_elapsed_ns=total_elapsed_ns,
                subprocess_elapsed_ns=subprocess_elapsed_ns,
            )
            if self._runtime_v2_direct_output_enabled:
                self._runtime_v2_finalize_request(request_id)

    def _runtime_v2_finalize_request(self, request_id: str) -> None:
        with self._runtime_v2_state_lock:
            self._runtime_v2_active_requests.discard(request_id)
            self._runtime_v2_request_result_meta.pop(request_id, None)
            self._runtime_v2_submission_ids.pop(request_id, None)
            self._runtime_v2_postprocess_inflight.discard(request_id)
            self._runtime_v2_cancelled_requests.discard(request_id)
            self._runtime_v2_completed_results.pop(request_id, None)
            self._runtime_v2_failed_requests.pop(request_id, None)

    def _control_trace(self, action: str, **fields: Any) -> None:
        if not self._runtime_v2_control_trace_enabled:
            return
        if "elapsed_ns" not in fields and fields.get("status") != "error":
            return
        elapsed_ns = fields.get("elapsed_ns")
        if isinstance(elapsed_ns, int) and elapsed_ns < _CPU_THREAD_TRACE_MIN_NS:
            return
        parts = [f"{k}={v}" for k, v in fields.items()]
        parts.append(f"mono_ns={time.monotonic_ns()}")
        logger.info("runtime_v2 control trace: action=%s %s", action, " ".join(parts))

    def _thread_trace(self, action: str, **fields: Any) -> None:
        if not self._runtime_v2_control_trace_enabled:
            return
        elapsed_ns = fields.get("elapsed_ns")
        if isinstance(elapsed_ns, int) and elapsed_ns < _CPU_THREAD_TRACE_MIN_NS:
            return
        thread = threading.current_thread()
        parts = [f"thread={thread.name}", f"tid={thread.ident}", f"action={action}"]
        parts.extend(f"{k}={v}" for k, v in fields.items())
        parts.append(f"mono_ns={time.monotonic_ns()}")
        logger.info("runtime_v2 cpu thread trace: %s", " ".join(parts))

    def _emit_direct_stage_error(self, request_id: str, message: str) -> None:
        if not self._runtime_v2_direct_output_enabled:
            return
        out_q = self._runtime_v2_direct_result_queue
        cfg = self._runtime_v2_direct_result_cfg
        if out_q is None or cfg is None:
            return
        out_q.put(
            {
                "request_id": request_id,
                "stage_id": int(cfg["stage_id"]),
                "error": message,
            }
        )

    def _runtime_v2_control_loop(self) -> None:
        submit_queue = self._runtime_v2_submit_queue
        postprocess_in_queue = self._runtime_v2_postprocess_in_queue
        assert submit_queue is not None
        assert postprocess_in_queue is not None
        try:
            while not self._runtime_v2_stop.is_set():
                drained = 0 if self._runtime_v2_direct_output_enabled else self._drain_runtime_v2_postprocess_results()
                did_work = drained > 0
                while True:
                    try:
                        request_id, request = submit_queue.get_nowait()
                    except queue.Empty:
                        break
                    cancelled = False
                    with self._runtime_v2_state_lock:
                        if request_id in self._runtime_v2_cancelled_requests:
                            cancelled = True
                    if cancelled:
                        if self._runtime_v2_direct_output_enabled:
                            self._runtime_v2_finalize_request(request_id)
                        continue
                    did_work = True
                    try:
                        submit_begin_ns = time.monotonic_ns()
                        submission_id = self.engine.submit(request)
                        submit_done_ns = time.monotonic_ns()
                        submit_elapsed_ns = max(0, submit_done_ns - submit_begin_ns)
                        self._control_trace(
                            "submit",
                            request_id=request_id,
                            status="ok",
                            elapsed_ns=submit_elapsed_ns,
                        )
                        self._thread_trace(
                            "submit",
                            request_id=request_id,
                            status="ok",
                            elapsed_ns=submit_elapsed_ns,
                        )
                    except BaseException as exc:
                        with self._runtime_v2_state_lock:
                            self._runtime_v2_failed_requests[request_id] = RuntimeError(f"Diffusion submit failed: {exc}")
                        self._emit_direct_stage_error(request_id, f"Diffusion submit failed: {exc}")
                        if self._runtime_v2_direct_output_enabled:
                            self._runtime_v2_finalize_request(request_id)
                        self._control_trace(
                            "submit",
                            request_id=request_id,
                            status="error",
                            error_type=type(exc).__name__,
                        )
                        continue
                    with self._runtime_v2_state_lock:
                        self._runtime_v2_submission_ids[request_id] = submission_id

                with self._runtime_v2_state_lock:
                    inflight = tuple(self._runtime_v2_submission_ids.items())
                    postprocess_inflight = len(self._runtime_v2_postprocess_inflight)
                self._control_trace(
                    "poll_cycle",
                    inflight=len(inflight),
                    postprocess_inflight=postprocess_inflight,
                    did_work=did_work,
                )
                for request_id, submission_id in inflight:
                    cancelled = False
                    with self._runtime_v2_state_lock:
                        if request_id in self._runtime_v2_cancelled_requests:
                            self._runtime_v2_submission_ids.pop(request_id, None)
                            cancelled = True
                    if cancelled:
                        if self._runtime_v2_direct_output_enabled:
                            self._runtime_v2_finalize_request(request_id)
                        continue
                    collect_begin_ns = time.monotonic_ns()
                    try:
                        collected = self.engine.collect_raw(submission_id, timeout_s=0.0)
                    except TimeoutError:
                        collect_done_ns = time.monotonic_ns()
                        collect_elapsed_ns = max(0, collect_done_ns - collect_begin_ns)
                        self._control_trace(
                            "collect_raw_poll",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="timeout",
                            elapsed_ns=collect_elapsed_ns,
                            inflight=len(inflight),
                        )
                        self._thread_trace(
                            "collect_raw_poll",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="timeout",
                            elapsed_ns=collect_elapsed_ns,
                            inflight=len(inflight),
                        )
                        continue
                    except BaseException as exc:
                        collect_done_ns = time.monotonic_ns()
                        collect_elapsed_ns = max(0, collect_done_ns - collect_begin_ns)
                        with self._runtime_v2_state_lock:
                            self._runtime_v2_submission_ids.pop(request_id, None)
                            self._runtime_v2_failed_requests[request_id] = RuntimeError(f"Diffusion collect failed: {exc}")
                        self._emit_direct_stage_error(request_id, f"Diffusion collect failed: {exc}")
                        if self._runtime_v2_direct_output_enabled:
                            self._runtime_v2_finalize_request(request_id)
                        self._control_trace(
                            "collect_raw_poll",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="error",
                            error_type=type(exc).__name__,
                            elapsed_ns=collect_elapsed_ns,
                            inflight=len(inflight),
                        )
                        self._thread_trace(
                            "collect_raw_poll",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="error",
                            error_type=type(exc).__name__,
                            elapsed_ns=collect_elapsed_ns,
                            inflight=len(inflight),
                        )
                        did_work = True
                        continue
                    collect_done_ns = time.monotonic_ns()
                    collect_elapsed_ns = max(0, collect_done_ns - collect_begin_ns)
                    with self._runtime_v2_state_lock:
                        self._runtime_v2_submission_ids.pop(request_id, None)
                        result_meta = dict(self._runtime_v2_request_result_meta.get(request_id, {}))
                    enqueue_start_ns = time.monotonic_ns()
                    try:
                        postprocess_in_queue.put_nowait((request_id, collected, enqueue_start_ns, result_meta))
                        enqueue_done_ns = time.monotonic_ns()
                        enqueue_elapsed_ns = max(0, enqueue_done_ns - enqueue_start_ns)
                        if self._runtime_v2_direct_output_enabled:
                            self._runtime_v2_finalize_request(request_id)
                        else:
                            with self._runtime_v2_state_lock:
                                self._runtime_v2_postprocess_inflight.add(request_id)
                        self._control_trace(
                            "postprocess_enqueue",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="ok",
                            elapsed_ns=enqueue_elapsed_ns,
                            queue_wait_ns=enqueue_elapsed_ns,
                        )
                        self._thread_trace(
                            "postprocess_enqueue",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="ok",
                            elapsed_ns=enqueue_elapsed_ns,
                        )
                    except BaseException as exc:
                        with self._runtime_v2_state_lock:
                            self._runtime_v2_failed_requests[request_id] = RuntimeError(
                                f"Diffusion postprocess enqueue failed: {exc}"
                            )
                        self._emit_direct_stage_error(request_id, f"Diffusion postprocess enqueue failed: {exc}")
                        if self._runtime_v2_direct_output_enabled:
                            self._runtime_v2_finalize_request(request_id)
                        self._control_trace(
                            "postprocess_enqueue",
                            request_id=request_id,
                            submission_id=submission_id,
                            status="error",
                            error_type=type(exc).__name__,
                        )
                        did_work = True
                        continue
                    self._control_trace(
                        "collect_raw_poll",
                        request_id=request_id,
                        submission_id=submission_id,
                        status="ready",
                        elapsed_ns=collect_elapsed_ns,
                        inflight=len(inflight),
                    )
                    self._thread_trace(
                        "collect_raw_poll",
                        request_id=request_id,
                        submission_id=submission_id,
                        status="ready",
                        elapsed_ns=collect_elapsed_ns,
                        inflight=len(inflight),
                    )
                    did_work = True

                if not did_work:
                    sleep_begin_ns = time.monotonic_ns()
                    time.sleep(0.001)
                    sleep_done_ns = time.monotonic_ns()
                    sleep_elapsed_ns = max(0, sleep_done_ns - sleep_begin_ns)
                    self._control_trace(
                        "idle_sleep",
                        elapsed_ns=sleep_elapsed_ns,
                    )
                    self._thread_trace("idle_sleep", elapsed_ns=sleep_elapsed_ns)
        except BaseException as exc:  # pragma: no cover - defensive path
            self._runtime_v2_background_error = exc
            logger.exception("runtime_v2 async diffusion control loop failed")
            self._runtime_v2_stop.set()

    async def submit(
        self,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str | None = None,
        lora_request: LoRARequest | None = None,
        runtime_v2_result_meta: _RuntimeV2ResultMeta | None = None,
    ) -> str:
        """Submit a generation request and return request_id immediately."""
        request_id, request = self._build_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        logger.debug("Submitting generation request %s", request_id)

        if self._runtime_v2_worker_mode:
            self._ensure_runtime_v2_threads_healthy()
            if self._runtime_v2_direct_output_enabled:
                with self._runtime_v2_state_lock:
                    if request_id in self._runtime_v2_active_requests:
                        raise RuntimeError(f"Duplicate diffusion request_id submitted: {request_id}")
                    self._runtime_v2_active_requests.add(request_id)
                    self._runtime_v2_request_result_meta[request_id] = dict(runtime_v2_result_meta or {})
                assert self._runtime_v2_submit_queue is not None
                self._runtime_v2_submit_queue.put((request_id, request))
                return request_id
            async with self._pending_lock:
                if request_id in self._pending_submissions:
                    raise RuntimeError(f"Duplicate diffusion request_id submitted: {request_id}")
                self._pending_submissions[request_id] = ""
                self._submit_futures[request_id] = None
            assert self._runtime_v2_submit_queue is not None
            self._runtime_v2_submit_queue.put((request_id, request))
            return request_id
        else:
            loop = asyncio.get_running_loop()
            async with self._pending_lock:
                if request_id in self._pending_submissions:
                    raise RuntimeError(f"Duplicate diffusion request_id submitted: {request_id}")
                # Reserve request_id early to avoid racy double-submit on concurrent calls.
                self._pending_submissions[request_id] = ""
                self._submit_futures[request_id] = loop.run_in_executor(
                    self._control_executor,
                    self.engine.submit,
                    request,
                )
            try:
                return request_id
            except Exception as e:
                async with self._pending_lock:
                    self._pending_submissions.pop(request_id, None)
                    self._submit_futures.pop(request_id, None)
                logger.error("Submit failed for request %s: %s", request_id, e)
                raise RuntimeError(f"Diffusion submit failed: {e}") from e

    async def collect(self, request_id: str, timeout_s: float | None = None) -> OmniRequestOutput:
        """Collect a previously submitted request."""
        if self._runtime_v2_worker_mode and self._runtime_v2_direct_output_enabled:
            raise RuntimeError("collect() is disabled when runtime_v2_direct_output_enabled is true")
        async with self._pending_lock:
            has_request = request_id in self._pending_submissions and request_id in self._submit_futures
            submission_id = self._pending_submissions.get(request_id)
            submit_future = self._submit_futures.get(request_id)
        if not has_request or submission_id is None:
            raise KeyError(f"Unknown diffusion request_id: {request_id}")

        if self._runtime_v2_worker_mode:
            deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
            while True:
                self._ensure_runtime_v2_threads_healthy()
                with self._runtime_v2_state_lock:
                    error = self._runtime_v2_failed_requests.get(request_id)
                    result = self._runtime_v2_completed_results.get(request_id)
                if error is not None:
                    async with self._pending_lock:
                        self._pending_submissions.pop(request_id, None)
                        self._submit_futures.pop(request_id, None)
                    self._runtime_v2_finalize_request(request_id)
                    logger.error("Collect failed for request %s: %s", request_id, error)
                    raise RuntimeError(str(error)) from error
                if result is not None:
                    async with self._pending_lock:
                        self._pending_submissions.pop(request_id, None)
                        self._submit_futures.pop(request_id, None)
                    self._runtime_v2_finalize_request(request_id)
                    return result
                if timeout_s is not None and timeout_s <= 0:
                    raise TimeoutError(f"Diffusion collect timed out for request {request_id}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Diffusion collect timed out for request {request_id}")
                collect_wait_begin_ns = time.monotonic_ns()
                await asyncio.sleep(0.001)
                collect_wait_done_ns = time.monotonic_ns()
                self._thread_trace(
                    "collect_wait_sleep",
                    request_id=request_id,
                    elapsed_ns=max(0, collect_wait_done_ns - collect_wait_begin_ns),
                )
        else:
            remaining_timeout = timeout_s
            if not submission_id:
                submit_start = asyncio.get_running_loop().time()
                try:
                    if timeout_s is None:
                        submission_id = await submit_future
                    else:
                        submission_id = await asyncio.wait_for(submit_future, timeout=timeout_s)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Diffusion collect timed out waiting submit for request {request_id}")
                except Exception as e:
                    async with self._pending_lock:
                        self._pending_submissions.pop(request_id, None)
                        self._submit_futures.pop(request_id, None)
                    logger.error("Submit future failed for request %s: %s", request_id, e)
                    raise RuntimeError(f"Diffusion submit future failed: {e}") from e

                if timeout_s is not None:
                    elapsed = asyncio.get_running_loop().time() - submit_start
                    remaining_timeout = max(0.0, timeout_s - elapsed)
                async with self._pending_lock:
                    self._pending_submissions[request_id] = submission_id

            loop = asyncio.get_running_loop()
            try:
                result_list = await loop.run_in_executor(
                    self._collect_executor,
                    self.engine.collect,
                    submission_id,
                    remaining_timeout,
                )
            except TimeoutError:
                raise
            except Exception as e:
                async with self._pending_lock:
                    self._pending_submissions.pop(request_id, None)
                    self._submit_futures.pop(request_id, None)
                logger.error("Collect failed for request %s: %s", request_id, e)
                raise RuntimeError(f"Diffusion collect failed: {e}") from e

        async with self._pending_lock:
            self._pending_submissions.pop(request_id, None)
            self._submit_futures.pop(request_id, None)

        if not result_list:
            raise RuntimeError(f"Diffusion collect returned no outputs for request {request_id}")
        result = result_list[0]
        if not result.request_id:
            result.request_id = request_id
        return result

    async def try_collect(self, request_id: str, timeout_s: float = 0.0) -> OmniRequestOutput | None:
        """Try to collect a submitted request without blocking."""
        try:
            return await self.collect(request_id, timeout_s=timeout_s)
        except TimeoutError:
            return None

    async def generate(
        self,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str | None = None,
        lora_request: LoRARequest | None = None,
    ) -> OmniRequestOutput:
        """Generate images asynchronously from a text prompt."""
        request_id = await self.submit(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        return await self.collect(request_id)

    async def generate_stream(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate images with streaming progress updates.

        Currently, diffusion models don't support true streaming, so this
        yields a single result after generation completes. Future implementations
        may support step-by-step progress updates.

        Args:
            prompt: Text prompt describing the desired image
            request_id: Optional unique identifier for tracking the request
            **kwargs: Additional generation parameters

        Yields:
            OmniRequestOutput with generation progress/results
        """
        result = await self.generate(prompt=prompt, request_id=request_id, **kwargs)
        yield result

    def close(self) -> None:
        """Close the engine and release resources.

        Should be called when done using the AsyncOmniDiffusion instance.
        """
        if self._closed:
            return
        self._closed = True
        self._pending_submissions.clear()
        self._submit_futures.clear()
        with self._runtime_v2_state_lock:
            self._runtime_v2_active_requests.clear()
            self._runtime_v2_request_result_meta.clear()
        self._runtime_v2_stop.set()

        try:
            if self._runtime_v2_control_thread is not None and self._runtime_v2_control_thread.is_alive():
                self._runtime_v2_control_thread.join(timeout=5.0)
        except Exception as e:
            logger.warning("Error stopping runtime_v2 background threads: %s", e)

        try:
            # One None sentinel per pooled subprocess; each consumes exactly one.
            if self._runtime_v2_postprocess_in_queue is not None:
                for _ in self._runtime_v2_postprocess_processes:
                    self._runtime_v2_postprocess_in_queue.put_nowait(None)
            for process in self._runtime_v2_postprocess_processes:
                if process.is_alive():
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5.0)
            if self._runtime_v2_postprocess_in_queue is not None:
                self._runtime_v2_postprocess_in_queue.close()
            if self._runtime_v2_postprocess_out_queue is not None:
                self._runtime_v2_postprocess_out_queue.close()
        except Exception as e:
            logger.warning("Error stopping runtime_v2 postprocess subprocess: %s", e)

        try:
            self.engine.close()
        except Exception as e:
            logger.warning("Error closing diffusion engine: %s", e)

        try:
            if self._control_executor is not None:
                self._control_executor.shutdown(wait=False)
            if self._collect_executor is not None:
                self._collect_executor.shutdown(wait=False)
        except Exception as e:
            logger.warning("Error shutting down executor: %s", e)

        logger.info("AsyncOmniDiffusion closed")

    def shutdown(self) -> None:
        """Alias for close() method."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request."""
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        async with self._pending_lock:
            for rid in request_ids:
                self._pending_submissions.pop(rid, None)
                submit_future = self._submit_futures.pop(rid, None)
                if submit_future is not None:
                    submit_future.cancel()
                if self._runtime_v2_worker_mode:
                    with self._runtime_v2_state_lock:
                        self._runtime_v2_cancelled_requests.add(rid)
                        self._runtime_v2_active_requests.discard(rid)
                        self._runtime_v2_request_result_meta.pop(rid, None)
                        self._runtime_v2_submission_ids.pop(rid, None)
                        self._runtime_v2_postprocess_inflight.discard(rid)
                        self._runtime_v2_completed_results.pop(rid, None)
                        self._runtime_v2_failed_requests.pop(rid, None)
                else:
                    self._runtime_v2_finalize_request(rid)
        self.engine.abort(request_id)

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return not self._closed

    @property
    def runtime_v2_direct_output_enabled(self) -> bool:
        return self._runtime_v2_direct_output_enabled

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self._closed

    async def remove_lora(self, adapter_id: int) -> bool:
        """Remove a LoRA"""
        if self._control_executor is None:
            return self.engine.collective_rpc("remove_lora", None, (adapter_id,), {}, None)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._control_executor,
            self.engine.collective_rpc,
            "remove_lora",
            None,
            (adapter_id,),
            {},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA adapter"""
        if self._control_executor is None:
            return self.engine.collective_rpc(
                "add_lora",
                None,
                (),
                {"lora_request": lora_request},
                None,
            )
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._control_executor,
            self.engine.collective_rpc,
            "add_lora",
            None,
            (),
            {"lora_request": lora_request},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def list_loras(self) -> list[int]:
        """List all registered LoRA adapter IDs."""
        if self._control_executor is None:
            results = self.engine.collective_rpc("list_loras", None, (), {}, None)
            if not isinstance(results, list):
                return results or []
            merged: set[int] = set()
            for part in results:
                merged.update(part or [])
            return sorted(merged)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._control_executor,
            self.engine.collective_rpc,
            "list_loras",
            None,
            (),
            {},
            None,
        )
        # collective_rpc returns list from workers; flatten unique ids
        if not isinstance(results, list):
            return results or []
        merged: set[int] = set()
        for part in results:
            merged.update(part or [])
        return sorted(merged)

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        if self._control_executor is None:
            return self.engine.collective_rpc("pin_lora", None, (), {"adapter_id": lora_id}, None)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._control_executor,
            self.engine.collective_rpc,
            "pin_lora",
            None,
            (),
            {"adapter_id": lora_id},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def start_profile(self, trace_filename: str | None = None) -> None:
        """Start profiling for the diffusion model.

        Args:
            trace_filename: Optional base filename for trace files.
                           If None, a timestamp-based name will be generated.
        """
        if self._control_executor is None:
            self.engine.start_profile(trace_filename)
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._control_executor,
            self.engine.start_profile,
            trace_filename,
        )

    async def stop_profile(self) -> dict:
        """Stop profiling and return profiling results.

        Returns:
            Dictionary containing paths to trace and table files.
        """
        if self._control_executor is None:
            return self.engine.stop_profile()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._control_executor,
            self.engine.stop_profile,
        )
