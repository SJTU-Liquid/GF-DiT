# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from vllm_omni.diffusion.runtime_v2.interfaces import RuntimeV2Adapter

_RUNTIME_V2_ADAPTERS: dict[str, RuntimeV2Adapter] = {}
_DEFAULT_ADAPTERS_REGISTERED = False


def _ensure_default_adapters_registered() -> None:
    global _DEFAULT_ADAPTERS_REGISTERED
    if _DEFAULT_ADAPTERS_REGISTERED:
        return

    from vllm_omni.diffusion.runtime_v2.adapters.qwen_image import QwenImageRuntimeV2Adapter
    from vllm_omni.diffusion.runtime_v2.adapters.wan import WanRuntimeV2Adapter

    register_runtime_v2_adapter(WanRuntimeV2Adapter())
    register_runtime_v2_adapter(QwenImageRuntimeV2Adapter())
    _DEFAULT_ADAPTERS_REGISTERED = True


def register_runtime_v2_adapter(adapter: RuntimeV2Adapter) -> None:
    model_class_name = str(adapter.model_class_name)
    if not model_class_name:
        raise ValueError("runtime_v2 adapter must define a non-empty model_class_name")
    if model_class_name in _RUNTIME_V2_ADAPTERS:
        raise ValueError(f"runtime_v2 adapter already registered for model_class_name={model_class_name!r}")
    _RUNTIME_V2_ADAPTERS[model_class_name] = adapter


def get_runtime_v2_adapter(model_class_name: str | None) -> RuntimeV2Adapter:
    _ensure_default_adapters_registered()
    model_class_name = str(model_class_name or "")
    if model_class_name in _RUNTIME_V2_ADAPTERS:
        return _RUNTIME_V2_ADAPTERS[model_class_name]
    supported = ", ".join(sorted(_RUNTIME_V2_ADAPTERS)) or "<none>"
    raise ValueError(
        "runtime_v2 does not support "
        f"model_class_name={model_class_name!r}. Registered adapters: {supported}"
    )


def supports_runtime_v2_model(model_class_name: str | None) -> bool:
    _ensure_default_adapters_registered()
    return str(model_class_name or "") in _RUNTIME_V2_ADAPTERS


def get_registered_runtime_v2_adapters() -> dict[str, RuntimeV2Adapter]:
    _ensure_default_adapters_registered()
    return dict(_RUNTIME_V2_ADAPTERS)
