# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2.adapters.wan import (
    WanRuntimeRequest,
    WanTaskCompiler,
    _default_guidance_scale_from_pipeline,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def test_unprovided_guidance_keeps_secondary_guidance_unset() -> None:
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "a video", "negative_prompt": ""}],
        sampling_params=OmniDiffusionSamplingParams(),
    )

    assert not req.sampling_params.guidance_scale_provided
    assert req.sampling_params.guidance_scale == 1.0
    assert req.sampling_params.guidance_scale_2 is None


def test_explicit_guidance_defaults_secondary_guidance_to_primary() -> None:
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "a video", "negative_prompt": ""}],
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=4.0),
    )

    assert req.sampling_params.guidance_scale_provided
    assert req.sampling_params.guidance_scale == 4.0
    assert req.sampling_params.guidance_scale_2 == 4.0


def test_runtime_v2_wan_uses_pipeline_guidance_default() -> None:
    class DummyTi2VPipeline:
        def forward(self, req, guidance_scale: float = 5.0):
            raise NotImplementedError

    assert _default_guidance_scale_from_pipeline(DummyTi2VPipeline()) == 5.0


def test_wan_task_compiler_marks_dit_cost_stage_by_cfg() -> None:
    compiler = WanTaskCompiler(default_guidance_scale=5.0)
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "a video", "negative_prompt": ""}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=["r0"],
    )

    plan = compiler.compile_request(WanRuntimeRequest(req))

    assert plan.tasks["r0:dit_step_chunk:0"].payload["cost_model_stage"] == "dit_step_chunk_cfg"


def test_wan_task_compiler_marks_no_cfg_cost_stage() -> None:
    compiler = WanTaskCompiler(default_guidance_scale=5.0)
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "a video", "negative_prompt": ""}],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            guidance_scale=1.0,
        ),
        request_ids=["r0"],
    )

    plan = compiler.compile_request(WanRuntimeRequest(req))

    assert plan.tasks["r0:dit_step_chunk:0"].payload["cost_model_stage"] == "dit_step_chunk_nocfg"
