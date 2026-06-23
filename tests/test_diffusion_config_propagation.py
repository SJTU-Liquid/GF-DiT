# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that parallel_config survives the create_default_diffusion roundtrip.

Regression tests for https://github.com/vllm-project/vllm-omni/issues/1862
"""

from collections.abc import Mapping

import torch

from vllm_omni.config.stage_config import StageConfigFactory
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)


def _roundtrip_diffusion_config(**kwargs) -> OmniDiffusionConfig:
    """Simulate the real path: create_default_diffusion → OmniDiffusionConfig.

    Does NOT manually reconstruct parallel_config — relies on
    OmniDiffusionConfig.__post_init__ to handle the dict, just like
    the production code path does.
    """
    stages = StageConfigFactory.create_default_diffusion(kwargs)
    engine_args = dict(stages[0]["engine_args"])
    return OmniDiffusionConfig.from_kwargs(**engine_args)


class TestParallelConfigPropagation:
    """Core regression tests: parallel_config must survive serialization."""

    def test_tp2_roundtrip(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        od = _roundtrip_diffusion_config(model="test-model", parallel_config=pc)
        assert od.parallel_config.tensor_parallel_size == 2
        assert od.parallel_config.world_size == 2

    def test_tp4_devices_and_config(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=4)
        stages = StageConfigFactory.create_default_diffusion({"parallel_config": pc, "model": "x"})
        assert stages[0]["runtime"]["devices"] == "0,1,2,3"

        # Let __post_init__ reconstruct from dict (real code path)
        ea = dict(stages[0]["engine_args"])
        od = OmniDiffusionConfig.from_kwargs(**ea)
        assert od.parallel_config.tensor_parallel_size == 4
        assert od.parallel_config.world_size == 4

    def test_sp_config_roundtrip(self):
        pc = DiffusionParallelConfig(
            tensor_parallel_size=2,
            ulysses_degree=2,
            ring_degree=1,
        )
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.parallel_config.ulysses_degree == 2
        assert od.parallel_config.ring_degree == 1

    def test_cfg_parallel_roundtrip(self):
        pc = DiffusionParallelConfig(cfg_parallel_size=2)
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.parallel_config.cfg_parallel_size == 2
        assert od.parallel_config.world_size == 2

    def test_no_parallel_config_defaults_to_tp1(self):
        od = _roundtrip_diffusion_config(model="x")
        assert od.parallel_config.tensor_parallel_size == 1
        assert od.parallel_config.world_size == 1

    def test_num_gpus_derived_from_world_size(self):
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        od = _roundtrip_diffusion_config(model="x", parallel_config=pc)
        assert od.num_gpus == 2


class TestCreateDefaultDiffusion:
    """Verify engine_args structure from create_default_diffusion."""

    def test_parallel_config_serialized_as_dict(self):
        """The key fix: parallel_config must appear in engine_args as a dict."""
        pc = DiffusionParallelConfig(tensor_parallel_size=2)
        stages = StageConfigFactory.create_default_diffusion({"model": "x", "parallel_config": pc})
        ea = stages[0]["engine_args"]
        assert "parallel_config" in ea
        assert isinstance(ea["parallel_config"], Mapping)
        assert ea["parallel_config"]["tensor_parallel_size"] == 2

    def test_dtype_serialized_as_string(self):
        stages = StageConfigFactory.create_default_diffusion({"dtype": torch.float16, "model": "x"})
        assert stages[0]["engine_args"]["dtype"] == "torch.float16"

    def test_cache_backend_defaults_to_none(self):
        stages = StageConfigFactory.create_default_diffusion({"model": "x"})
        assert stages[0]["engine_args"]["cache_backend"] == "none"

    def test_single_gpu_default_devices(self):
        stages = StageConfigFactory.create_default_diffusion({"model": "x"})
        assert stages[0]["runtime"]["devices"] == "0"

    def test_extra_kwargs_forwarded(self):
        stages = StageConfigFactory.create_default_diffusion(
            {"model": "x", "enforce_eager": True, "lora_path": "/tmp/lora"}
        )
        ea = stages[0]["engine_args"]
        assert ea["enforce_eager"] is True
        assert ea["lora_path"] == "/tmp/lora"

    def test_runtime_v2_config_forwarded(self):
        stages = StageConfigFactory.create_default_diffusion(
            {
                "model": "x",
                "enable_runtime_v2": True,
                "runtime_v2_denoise_chunk_size": 3,
                "runtime_v2_scheduler_policy": "srtf",
                "runtime_v2_collective_backend": "gfc",
                "runtime_v2_group_sizes": "4,2,1,1",
                "runtime_v2_groups_json": (
                    '[{"size":4,"tp":4,"ulysses_degree":1,"ring_degree":1},'
                    '{"size":2,"tp":2,"ulysses_degree":1,"ring_degree":1},'
                    '{"size":1,"tp":1,"ulysses_degree":1,"ring_degree":1},'
                    '{"size":1,"tp":1,"ulysses_degree":1,"ring_degree":1}]'
                ),
                "runtime_v2_dit_step_schedule": '[{"start":0,"end":2,"group_id":"g0"}]',
            }
        )
        ea = dict(stages[0]["engine_args"])
        assert ea["enable_runtime_v2"] is True
        assert ea["runtime_v2_denoise_chunk_size"] == 3
        assert ea["runtime_v2_scheduler_policy"] == "srtf"
        assert ea["runtime_v2_collective_backend"] == "gfc"
        assert ea["runtime_v2_group_sizes"] == "4,2,1,1"
        assert isinstance(ea["runtime_v2_groups_json"], str)
        assert isinstance(ea["runtime_v2_dit_step_schedule"], str)

        od = OmniDiffusionConfig.from_kwargs(**ea)
        assert od.enable_runtime_v2 is True
        assert od.runtime_v2_denoise_chunk_size == 3
        assert od.runtime_v2_scheduler_policy == "srtf"
        assert od.runtime_v2_collective_backend == "gfc"
        assert od.runtime_v2_group_sizes == [4, 2, 1, 1]
        assert isinstance(od.runtime_v2_groups_json, list)
        assert len(od.runtime_v2_groups_json) == 4
        assert od.runtime_v2_dit_step_schedule == [{"start": 0, "end": 2, "group_id": "g0"}]


class TestRuntimeV2StageConfigPrecedence:
    """Regression tests for CLI-defaults versus YAML runtime_v2 settings."""

    @staticmethod
    def _runtime_v2_stage():
        return create_config(
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {"devices": "0,1"},
                "engine_args": {
                    "enable_runtime_v2": True,
                    "runtime_v2_denoise_chunk_size": 8,
                    "runtime_v2_scheduler_policy": "edf_best_fit",
                    "runtime_v2_collective_backend": "gfc",
                    "runtime_v2_gfc_max_collective_mb": 512,
                },
            }
        )

    def test_yaml_runtime_v2_settings_survive_omitted_cli_flags(self, monkeypatch) -> None:
        from vllm_omni.entrypoints import omni as omni_module

        stage = self._runtime_v2_stage()

        def fake_loader(_model, _stage_configs_path, _kwargs, default_stage_cfg_factory=None):
            del default_stage_cfg_factory
            return "/tmp/stage.yaml", [stage]

        monkeypatch.setattr(omni_module, "load_and_resolve_stage_configs", fake_loader)
        omni_base = omni_module.OmniBase.__new__(omni_module.OmniBase)

        _config_path, stage_configs = omni_base._resolve_stage_configs(
            "test-model",
            {"stage_configs_path": "/tmp/stage.yaml"},
        )

        engine_args = stage_configs[0].engine_args
        assert engine_args.enable_runtime_v2 is True
        assert engine_args.runtime_v2_denoise_chunk_size == 8
        assert engine_args.runtime_v2_scheduler_policy == "edf_best_fit"
        assert engine_args.runtime_v2_collective_backend == "gfc"
        assert engine_args.runtime_v2_gfc_max_collective_mb == 512

    def test_explicit_runtime_v2_kwargs_override_yaml(self, monkeypatch) -> None:
        from vllm_omni.entrypoints import omni as omni_module

        stage = self._runtime_v2_stage()

        def fake_loader(_model, _stage_configs_path, _kwargs, default_stage_cfg_factory=None):
            del default_stage_cfg_factory
            return "/tmp/stage.yaml", [stage]

        monkeypatch.setattr(omni_module, "load_and_resolve_stage_configs", fake_loader)
        omni_base = omni_module.OmniBase.__new__(omni_module.OmniBase)

        _config_path, stage_configs = omni_base._resolve_stage_configs(
            "test-model",
            {
                "stage_configs_path": "/tmp/stage.yaml",
                "enable_runtime_v2": False,
                "runtime_v2_denoise_chunk_size": 1,
                "runtime_v2_scheduler_policy": "fcfs",
                "runtime_v2_collective_backend": "torch",
                "runtime_v2_gfc_max_collective_mb": 128,
            },
        )

        engine_args = stage_configs[0].engine_args
        assert engine_args.enable_runtime_v2 is False
        assert engine_args.runtime_v2_denoise_chunk_size == 1
        assert engine_args.runtime_v2_scheduler_policy == "fcfs"
        assert engine_args.runtime_v2_collective_backend == "torch"
        assert engine_args.runtime_v2_gfc_max_collective_mb == 128
