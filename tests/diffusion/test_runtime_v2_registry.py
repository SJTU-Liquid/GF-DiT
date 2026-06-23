# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.runtime_v2 import (
    get_registered_runtime_v2_adapters,
    get_runtime_v2_adapter,
    supports_runtime_v2_model,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_runtime_v2_registry_contains_wan_and_qwen_image() -> None:
    adapters = get_registered_runtime_v2_adapters()

    assert "WanPipeline" in adapters
    assert "QwenImagePipeline" in adapters
    assert supports_runtime_v2_model("WanPipeline") is True
    assert supports_runtime_v2_model("QwenImagePipeline") is True
    assert supports_runtime_v2_model("FluxPipeline") is False


def test_get_runtime_v2_adapter_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="does not support"):
        get_runtime_v2_adapter("FluxPipeline")


def test_multiproc_executor_runtime_v2_fails_fast_for_unsupported_model() -> None:
    executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(enable_runtime_v2=True, model_class_name="FluxPipeline")

    with pytest.raises(ValueError, match="does not support"):
        executor._should_use_runtime_v2()


def test_multiproc_executor_runtime_v2_accepts_registered_model() -> None:
    executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(enable_runtime_v2=True, model_class_name="QwenImagePipeline")

    assert executor._should_use_runtime_v2() is True
