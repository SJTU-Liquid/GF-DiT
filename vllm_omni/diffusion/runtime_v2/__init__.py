# SPDX-License-Identifier: Apache-2.0

from vllm_omni.diffusion.runtime_v2.interfaces import RuntimeV2Adapter
from vllm_omni.diffusion.runtime_v2.registry import (
    get_registered_runtime_v2_adapters,
    get_runtime_v2_adapter,
    register_runtime_v2_adapter,
    supports_runtime_v2_model,
)

__all__ = [
    "RuntimeV2Adapter",
    "RuntimeV2Runner",
    "get_registered_runtime_v2_adapters",
    "get_runtime_v2_adapter",
    "register_runtime_v2_adapter",
    "supports_runtime_v2_model",
]


def __getattr__(name: str):
    if name == "RuntimeV2Runner":
        from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

        return RuntimeV2Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
