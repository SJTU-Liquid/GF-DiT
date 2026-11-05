# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.cli.serve import OmniServeCommand

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_RUNTIME_V2_DEFAULTED_FLAGS = (
    "enable_runtime_v2",
    "runtime_v2_denoise_chunk_size",
    "runtime_v2_scheduler_policy",
    "runtime_v2_collective_backend",
    "runtime_v2_gfc_max_collective_mb",
    "runtime_v2_wave_stress_warmup_reqs",
    "runtime_v2_wave_stress_wave_size",
)


def _parse_serve_args(extra_args: list[str]):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)
    return parser.parse_args(["serve", "test-model", "--omni", *extra_args])


def test_runtime_v2_cli_defaults_are_not_present_when_omitted() -> None:
    args = _parse_serve_args([])

    for key in _RUNTIME_V2_DEFAULTED_FLAGS:
        assert not hasattr(args, key)


def test_runtime_v2_cli_flags_are_present_when_explicit() -> None:
    args = _parse_serve_args(
        [
            "--enable-runtime-v2",
            "--runtime-v2-denoise-chunk-size",
            "3",
            "--runtime-v2-scheduler-policy",
            "srtf",
            "--runtime-v2-collective-backend",
            "gfc",
            "--runtime-v2-gfc-max-collective-mb",
            "256",
            "--runtime-v2-wave-stress-warmup-reqs",
            "9",
            "--runtime-v2-wave-stress-wave-size",
            "2",
        ]
    )

    assert args.enable_runtime_v2 is True
    assert args.runtime_v2_denoise_chunk_size == 3
    assert args.runtime_v2_scheduler_policy == "srtf"
    assert args.runtime_v2_collective_backend == "gfc"
    assert args.runtime_v2_gfc_max_collective_mb == 256
    assert args.runtime_v2_wave_stress_warmup_reqs == 9
    assert args.runtime_v2_wave_stress_wave_size == 2
