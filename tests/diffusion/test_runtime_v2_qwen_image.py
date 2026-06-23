# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2.adapters.qwen_image import (
    QwenImageDecodedLayoutCodec,
    QwenImageRuntimeRequest,
    QwenImageRuntimeState,
    QwenImageRuntimeV2Adapter,
    QwenImageTaskCompiler,
    QwenImageTimestepPrepareExecutor,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ExecutionGroupSpec,
    ParallelSpec,
    TaskKind,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

QWEN_MODEL = os.environ.get("VLLM_TEST_QWEN_IMAGE_MODEL")


def _build_sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=5,
        guidance_scale=1.0,
        guidance_scale_provided=True,
        true_cfg_scale=1.0,
        seed=42,
        output_type="latent",
    )


def _build_request(request_id: str, *, num_inference_steps: int | None = None) -> OmniDiffusionRequest:
    sampling_params = replace(_build_sampling_params())
    if num_inference_steps is not None:
        sampling_params.num_inference_steps = num_inference_steps
    return OmniDiffusionRequest(
        prompts=[{"prompt": "A cat sitting on a table", "negative_prompt": ""}],
        sampling_params=sampling_params,
        request_ids=[request_id],
    )


class _DummyScheduler:
    def __init__(self) -> None:
        self.config = {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.begin_index = -1
        self.last_sigmas = None

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        **_: object,
    ) -> None:
        self.last_sigmas = sigmas
        if timesteps is not None:
            data = torch.tensor(timesteps, dtype=torch.float32)
        elif sigmas is not None:
            data = torch.tensor(sigmas, dtype=torch.float32)
        else:
            steps = int(num_inference_steps or 1)
            data = torch.arange(steps, 0, -1, dtype=torch.float32)
        if device is not None:
            data = data.to(device)
        self.timesteps = data

    def set_begin_index(self, index: int) -> None:
        self.begin_index = index

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, return_dict: bool = False):
        del noise_pred, t, return_dict
        return (latents,)


class _DummyPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.default_sample_size = 128
        self.vae_scale_factor = 8
        self.transformer = SimpleNamespace(in_channels=16, guidance_embeds=False, do_true_cfg=False)
        self.scheduler = _DummyScheduler()
        self.vae = SimpleNamespace(
            dtype=torch.float32,
            config=SimpleNamespace(
                latents_mean=[0.0, 0.0, 0.0, 0.0],
                latents_std=[1.0, 1.0, 1.0, 1.0],
                z_dim=4,
            ),
        )
        self._num_timesteps = 0

    def check_inputs(self, *args, **kwargs) -> None:
        del args, kwargs

    def check_cfg_parallel_validity(self, true_cfg_scale: float, has_neg_prompt: bool) -> bool:
        del true_cfg_scale, has_neg_prompt
        return True

    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        max_sequence_length: int = 1024,
    ):
        del prompt, prompt_embeds, prompt_embeds_mask
        embeds = torch.ones((num_images_per_prompt, min(max_sequence_length, 4), 8), dtype=torch.float32)
        mask = torch.ones((num_images_per_prompt, min(max_sequence_length, 4)), dtype=torch.long)
        return embeds, mask

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        del num_channels_latents, height, width, generator, latents
        return torch.zeros((batch_size, 4, 8), dtype=dtype, device=device)

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        del batch_size, num_channels_latents, height, width
        return latents

    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        del height, width, vae_scale_factor
        return latents


def test_qwen_image_runtime_v2_compiler_emits_step_chunks() -> None:
    request = QwenImageRuntimeRequest(
        diffusion_request=_build_request("req-compiler", num_inference_steps=5),
        denoise_chunk_size=2,
    )

    plan = QwenImageTaskCompiler().compile_request(request)

    denoise_tasks = [task for task in plan.tasks.values() if task.kind.name == "DIT_STEP_CHUNK"]
    assert len(denoise_tasks) == 3
    assert [(task.step_range.start, task.step_range.end) for task in denoise_tasks] == [(0, 2), (2, 4), (4, 5)]
    assert plan.metadata["request_type"] == "qwen_image_runtime_v2"


def test_qwen_image_timestep_prepare_uses_request_local_scheduler() -> None:
    pipeline = _DummyPipeline()
    executor = QwenImageTimestepPrepareExecutor(pipeline)
    task = SimpleNamespace(inputs=(SimpleNamespace(artifact_id="state"),), outputs=(SimpleNamespace(),))
    state = QwenImageRuntimeState(
        request=_build_request("req-state", num_inference_steps=3),
        prompt="cat",
        negative_prompt="",
        height=256,
        width=256,
        num_steps=3,
        output_type="latent",
        max_sequence_length=512,
        device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=1.0,
        true_cfg_scale=1.0,
        latents=torch.zeros((1, 4, 8), dtype=torch.float32),
    )

    outputs = executor.execute(task, {"state": state})
    next_state = outputs[0].value

    assert next_state.scheduler is not pipeline.scheduler
    assert isinstance(next_state.timesteps, torch.Tensor)
    assert next_state.scheduler.begin_index == 0
    assert pipeline.scheduler.begin_index == -1
    assert next_state.scheduler.last_sigmas is not None


def test_qwen_image_adapter_builds_all_standard_executors() -> None:
    adapter = QwenImageRuntimeV2Adapter()
    executors = adapter.build_executors(_DummyPipeline())

    assert set(executors) == set(adapter.supported_task_kinds)


def test_qwen_image_decoded_latent_layout_uses_packed_channels() -> None:
    pipeline = _DummyPipeline()
    codec = QwenImageDecodedLayoutCodec(pipeline)
    artifact = ArtifactHandle(
        request_id="req-latent",
        artifact_id="decoded",
        kind=ArtifactKind.OUTPUT,
        layout=ArtifactLayout.WORKER_LOCAL,
        codec_id=codec.codec_id,
    )
    group = ExecutionGroupSpec(
        group_id="g0",
        ranks=(0,),
        parallel_spec=ParallelSpec(),
        supported_task_kinds=(TaskKind.FINALIZE,),
    )

    layout = codec.describe_output(
        group=group,
        group_rank=0,
        request_metadata={
            "num_images_per_prompt": 1,
            "height": 256,
            "width": 256,
            "latent_seq_len": 4,
            "output_type": "latent",
        },
        artifact=artifact,
    )

    assert layout.tensors[0].field_path == ("value",)
    assert layout.tensors[0].local_shape == (1, 4, pipeline.transformer.in_channels)


@pytest.mark.slow
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_qwen_image_runtime_v2_engine_matches_legacy_engine_on_real_gpu() -> None:
    if not QWEN_MODEL:
        pytest.skip("Set VLLM_TEST_QWEN_IMAGE_MODEL to enable the qwen-image runtime_v2 consistency test.")
    if not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()):
        pytest.skip("QwenImage runtime_v2 consistency test requires CUDA/ROCm GPU.")

    def _build_engine(enable_runtime_v2: bool) -> DiffusionEngine:
        od_config = OmniDiffusionConfig.from_kwargs(
            model=QWEN_MODEL,
            model_class_name="QwenImagePipeline",
            dtype="bfloat16",
            num_gpus=1,
            enforce_eager=True,
            output_type="latent",
            enable_runtime_v2=enable_runtime_v2,
            runtime_v2_denoise_chunk_size=1,
        )
        return DiffusionEngine(od_config)

    legacy_engine = _build_engine(enable_runtime_v2=False)
    runtime_v2_engine = _build_engine(enable_runtime_v2=True)
    try:
        legacy_output = legacy_engine.add_req_and_wait_for_response(_build_request("req-qwen-legacy")).output
        runtime_v2_output = runtime_v2_engine.add_req_and_wait_for_response(_build_request("req-qwen-v2")).output
        assert isinstance(legacy_output, torch.Tensor)
        assert isinstance(runtime_v2_output, torch.Tensor)
        torch.testing.assert_close(
            runtime_v2_output.detach().cpu(),
            legacy_output.detach().cpu(),
            atol=1e-3,
            rtol=1e-3,
        )
    finally:
        legacy_engine.close()
        runtime_v2_engine.close()
