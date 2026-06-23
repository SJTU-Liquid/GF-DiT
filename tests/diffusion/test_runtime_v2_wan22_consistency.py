# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import gc
from dataclasses import replace
from typing import Any

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2 import RuntimeV2Runner
from vllm_omni.diffusion.runtime_v2.adapters.wan import WanRuntimeRequest, WanTaskCompiler
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

MODEL = os.environ.get("VLLM_TEST_WAN22_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")

pytestmark = [pytest.mark.diffusion, pytest.mark.advanced_model]


def _build_sampling_params() -> OmniDiffusionSamplingParams:
    # Keep the consistency test lightweight enough for ~32GiB cards.
    test_height = int(os.environ.get("VLLM_TEST_WAN22_HEIGHT", "320"))
    test_width = int(os.environ.get("VLLM_TEST_WAN22_WIDTH", "320"))
    test_num_frames = int(os.environ.get("VLLM_TEST_WAN22_NUM_FRAMES", "1"))
    test_num_steps = int(os.environ.get("VLLM_TEST_WAN22_NUM_STEPS", "2"))
    # Use latent comparison by default to keep memory pressure low.
    test_output_type = os.environ.get("VLLM_TEST_WAN22_OUTPUT_TYPE", "latent")
    return OmniDiffusionSamplingParams(
        height=test_height,
        width=test_width,
        num_frames=test_num_frames,
        num_inference_steps=test_num_steps,
        guidance_scale=1.0,
        guidance_scale_provided=True,
        seed=42,
        output_type=test_output_type,
    )


def _build_request(
    request_id: str,
    *,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> OmniDiffusionRequest:
    sampling_params = replace(_build_sampling_params())
    if num_inference_steps is not None:
        sampling_params.num_inference_steps = num_inference_steps
    if guidance_scale is not None:
        sampling_params.guidance_scale = guidance_scale
        sampling_params.guidance_scale_provided = True
    return OmniDiffusionRequest(
        prompts=[{"prompt": "A cat sitting on a table", "negative_prompt": ""}],
        sampling_params=sampling_params,
        request_ids=[request_id],
    )


def _load_wan22_worker() -> DiffusionWorker:
    od_config = OmniDiffusionConfig.from_kwargs(
        model=MODEL,
        model_class_name="WanPipeline",
        dtype="bfloat16",
        num_gpus=1,
        boundary_ratio=0.875,
        flow_shift=5.0,
        enforce_eager=True,
    )
    return DiffusionWorker(local_rank=0, rank=0, od_config=od_config)


def _build_engine(
    *,
    enable_runtime_v2: bool,
    parallel_config: dict[str, Any] | None = None,
) -> DiffusionEngine:
    parallel_config = dict(parallel_config or {})
    num_gpus = (
        int(parallel_config.get("tensor_parallel_size", 1))
        * int(parallel_config.get("sequence_parallel_size", 1))
        * int(parallel_config.get("cfg_parallel_size", 1))
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        model=MODEL,
        model_class_name="WanPipeline",
        dtype="bfloat16",
        num_gpus=num_gpus,
        boundary_ratio=0.875,
        flow_shift=5.0,
        enforce_eager=True,
        output_type="latent",
        enable_runtime_v2=enable_runtime_v2,
        runtime_v2_denoise_chunk_size=1,
        parallel_config=parallel_config,
    )
    return DiffusionEngine(od_config)


def _extract_output_tensor(output) -> torch.Tensor:
    assert output is not None
    assert output.error is None
    assert isinstance(output.output, torch.Tensor)
    return output.output.detach().cpu()


def _run_engine_once(
    *,
    enable_runtime_v2: bool,
    request_id: str,
    parallel_config: dict[str, Any] | None = None,
    guidance_scale: float | None = None,
) -> torch.Tensor:
    engine = _build_engine(enable_runtime_v2=enable_runtime_v2, parallel_config=parallel_config)
    try:
        request = _build_request(request_id, guidance_scale=guidance_scale)
        return _extract_output_tensor(engine.add_req_and_wait_for_response(request))
    finally:
        engine.close()
        gc.collect()
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()


def test_wan22_runtime_v2_compiler_emits_step_chunks() -> None:
    request = WanRuntimeRequest(
        diffusion_request=_build_request("req-compiler", num_inference_steps=2),
        denoise_chunk_size=2,
    )
    plan = WanTaskCompiler().compile_request(request)
    denoise_tasks = [task for task in plan.tasks.values() if task.kind.name == "DIT_STEP_CHUNK"]
    assert len(denoise_tasks) == 1
    assert (denoise_tasks[0].step_range.start, denoise_tasks[0].step_range.end) == (0, 2)


@pytest.mark.core_model
@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_wan22_runtime_v2_matches_legacy_on_real_gpu() -> None:
    # This test intentionally uses real Wan2.2 weights on GPU and compares:
    # - legacy path: DiffusionModelRunner.execute_model
    # - runtime_v2 path: RuntimeV2Runner task graph
    if not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()):
        pytest.skip("Wan2.2 runtime_v2 consistency test requires CUDA/ROCm GPU.")

    worker = _load_wan22_worker()
    model_runner = worker.model_runner
    assert model_runner is not None and model_runner.pipeline is not None
    runtime_v2 = RuntimeV2Runner(
        pipeline=model_runner.pipeline,
        default_step_chunk_size=1,
        vllm_config=worker.vllm_config,
        omni_diffusion_config=worker.od_config,
    )
    try:
        legacy_req = _build_request("req-legacy")
        v2_req = _build_request("req-v2")

        legacy_output_gpu = model_runner.execute_model(legacy_req).output
        assert isinstance(legacy_output_gpu, torch.Tensor)
        legacy_output = legacy_output_gpu.detach().cpu()
        del legacy_output_gpu
        gc.collect()
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        v2_output = runtime_v2.execute(v2_req, denoise_chunk_size=1, timeout_s=3600).output
        assert isinstance(v2_output, torch.Tensor)
        v2_output = v2_output.detach().cpu()

        assert legacy_output.shape == v2_output.shape
        assert torch.allclose(v2_output, legacy_output, atol=1e-3, rtol=1e-3)
    finally:
        runtime_v2.close()
        worker.shutdown()


@pytest.mark.core_model
@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_wan22_runtime_v2_engine_matches_legacy_engine_on_real_gpu() -> None:
    if not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()):
        pytest.skip("Wan2.2 runtime_v2 engine consistency test requires CUDA/ROCm GPU.")

    legacy_output = _run_engine_once(enable_runtime_v2=False, request_id="req-engine-legacy")
    v2_output = _run_engine_once(enable_runtime_v2=True, request_id="req-engine-v2")

    assert legacy_output.shape == v2_output.shape
    torch.testing.assert_close(v2_output, legacy_output, atol=1e-3, rtol=1e-3)


@pytest.mark.core_model
@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize(
    ("case_name", "parallel_config", "guidance_scale"),
    [
        pytest.param(
            "sp2",
            {
                "tensor_parallel_size": 1,
                "sequence_parallel_size": 2,
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_parallel_size": 1,
            },
            1.0,
            id="sp2_cfg1",
        ),
        pytest.param(
            "cfg2",
            {
                "tensor_parallel_size": 1,
                "sequence_parallel_size": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "cfg_parallel_size": 2,
            },
            5.0,
            id="sp1_cfg2",
        ),
    ],
)
def test_wan22_runtime_v2_engine_matches_legacy_engine_multi_gpu(
    case_name: str,
    parallel_config: dict[str, Any],
    guidance_scale: float,
) -> None:
    if not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()):
        pytest.skip("Wan2.2 runtime_v2 multi-GPU consistency test requires CUDA/ROCm GPU.")

    required_gpus = (
        int(parallel_config["tensor_parallel_size"])
        * int(parallel_config["sequence_parallel_size"])
        * int(parallel_config["cfg_parallel_size"])
    )
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < required_gpus:
        pytest.skip(f"Test case {case_name} requires {required_gpus} GPUs but only {available_gpus} available.")

    legacy_output = _run_engine_once(
        enable_runtime_v2=False,
        request_id=f"req-{case_name}-legacy",
        parallel_config=parallel_config,
        guidance_scale=guidance_scale,
    )
    v2_output = _run_engine_once(
        enable_runtime_v2=True,
        request_id=f"req-{case_name}-runtime-v2",
        parallel_config=parallel_config,
        guidance_scale=guidance_scale,
    )

    assert legacy_output.shape == v2_output.shape
    torch.testing.assert_close(v2_output, legacy_output, atol=1e-3, rtol=1e-3)
