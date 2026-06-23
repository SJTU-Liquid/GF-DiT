# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints import utils as utils_module
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "riverclouds/qwen_image_random"


def _noop_inline_engine(self, model, stage_config, kwargs):
    self._inline_diffusion = False
    self._inline_engine = None


def test_default_stage_config_includes_cache_backend(monkeypatch):
    """Ensure cache_backend/cache_config are preserved in default diffusion stage."""
    monkeypatch.setattr(utils_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(utils_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)
    monkeypatch.setattr(AsyncOmni, "_init_inline_diffusion_engine", _noop_inline_engine)

    omni = AsyncOmni(
        model=MODEL,
        cache_backend="cache_dit",
        cache_config='{"Fn_compute_blocks": 2}',
        vae_use_slicing=True,
        ulysses_degree=2,
    )

    stage_cfg = omni.stage_configs[0]
    engine_args = stage_cfg.engine_args

    assert engine_args.get("cache_backend") == "cache_dit"
    cache_config = engine_args.get("cache_config")
    assert cache_config["Fn_compute_blocks"] == 2
    assert engine_args.get("vae_use_slicing") is True
    parallel_config = engine_args.get("parallel_config")
    if hasattr(parallel_config, "get"):
        ulysses_degree = parallel_config.get("ulysses_degree")
    else:
        ulysses_degree = getattr(parallel_config, "ulysses_degree", None)
    assert ulysses_degree == 2


def test_default_cache_config_used_when_missing(monkeypatch):
    """Ensure default cache_config is applied when cache_backend is set."""
    monkeypatch.setattr(utils_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(utils_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)
    monkeypatch.setattr(AsyncOmni, "_init_inline_diffusion_engine", _noop_inline_engine)

    omni = AsyncOmni(
        model=MODEL,
        cache_backend="cache_dit",
    )

    engine_args = omni.stage_configs[0].engine_args
    cache_config = engine_args.get("cache_config")
    assert cache_config is not None
    assert cache_config["Fn_compute_blocks"] == 1


def test_default_stage_devices_from_sequence_parallel(monkeypatch):
    """Ensure devices list reflects sequence parallel size when no parallel_config is provided."""
    monkeypatch.setattr(utils_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(utils_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)
    monkeypatch.setattr(AsyncOmni, "_init_inline_diffusion_engine", _noop_inline_engine)

    omni = AsyncOmni(
        model=MODEL,
        ulysses_degree=2,
        ring_degree=2,
    )

    stage_cfg = omni.stage_configs[0]
    runtime = stage_cfg.runtime
    if hasattr(runtime, "get"):
        devices = runtime.get("devices")
    else:
        devices = getattr(runtime, "devices", None)
    assert devices == "0,1,2,3"


def test_default_stage_config_includes_runtime_v2(monkeypatch):
    """Ensure runtime_v2 knobs are preserved in default diffusion stage."""
    monkeypatch.setattr(utils_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(utils_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)
    monkeypatch.setattr(AsyncOmni, "_init_inline_diffusion_engine", _noop_inline_engine)

    omni = AsyncOmni(
        model=MODEL,
        enable_runtime_v2=True,
        runtime_v2_denoise_chunk_size=3,
        runtime_v2_scheduler_policy="srtf",
        runtime_v2_collective_backend="gfc",
        runtime_v2_group_sizes="4,2,1,1",
        runtime_v2_groups_json=(
            '[{"size":4,"tp":4,"ulysses_degree":1,"ring_degree":1},'
            '{"size":2,"tp":2,"ulysses_degree":1,"ring_degree":1},'
            '{"size":1,"tp":1,"ulysses_degree":1,"ring_degree":1},'
            '{"size":1,"tp":1,"ulysses_degree":1,"ring_degree":1}]'
        ),
        runtime_v2_dit_step_schedule='[{"start":0,"end":2,"group_id":"g0"}]',
    )

    stage_cfg = omni.stage_configs[0]
    engine_args = stage_cfg.engine_args
    assert engine_args.get("enable_runtime_v2") is True
    assert engine_args.get("runtime_v2_denoise_chunk_size") == 3
    assert engine_args.get("runtime_v2_scheduler_policy") == "srtf"
    assert engine_args.get("runtime_v2_collective_backend") == "gfc"
    assert engine_args.get("runtime_v2_group_sizes") == "4,2,1,1"
    assert isinstance(engine_args.get("runtime_v2_groups_json"), str)
    assert isinstance(engine_args.get("runtime_v2_dit_step_schedule"), str)
