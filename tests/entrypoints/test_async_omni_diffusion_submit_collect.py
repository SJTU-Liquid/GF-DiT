# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


class _DummyConfig:
    def __init__(self) -> None:
        self.model = "dummy-model"
        self.model_class_name = None
        self.omni_kv_config = {}
        self.tf_model_config = None
        self.cfg_kv_collect_func = None

    def update_multimodal_support(self) -> None:
        pass


class _FakeEngine:
    def __init__(self) -> None:
        self._submit_idx = 0
        self.submitted_ids: list[str] = []
        self.submitted_requests = []
        self.collect_calls: list[tuple[str, float | None]] = []
        self.raise_timeout_on_collect = False

    def submit(self, request):
        self._submit_idx += 1
        sid = f"sid-{self._submit_idx}"
        self.submitted_ids.append(sid)
        self.submitted_requests.append(request)
        return sid

    def collect(self, submission_id: str, timeout_s: float | None = None):
        self.collect_calls.append((submission_id, timeout_s))
        if self.raise_timeout_on_collect:
            raise TimeoutError("collect timeout")
        return [OmniRequestOutput.from_diffusion(request_id="", images=[])]

    def collective_rpc(self, *args, **kwargs):
        return [True]

    def close(self):
        return None

    def abort(self, request_id):
        return None


def _patch_init(monkeypatch, fake_engine: _FakeEngine) -> None:
    def _fake_get_hf_file_to_dict(filename: str, model: str):
        if filename == "model_index.json":
            return {"_class_name": "QwenImagePipeline"}
        if filename == "transformer/config.json":
            return {}
        return None

    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni_diffusion.get_hf_file_to_dict",
        _fake_get_hf_file_to_dict,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni_diffusion.DiffusionEngine.make_engine",
        staticmethod(lambda cfg: fake_engine),
    )


@pytest.mark.asyncio
async def test_submit_collect_roundtrip(monkeypatch):
    fake_engine = _FakeEngine()
    _patch_init(monkeypatch, fake_engine)

    diffusion = AsyncOmniDiffusion(model="dummy-model", od_config=_DummyConfig())
    try:
        sampling_params = OmniDiffusionSamplingParams()
        rid = await diffusion.submit(prompt="p", sampling_params=sampling_params, request_id="req-1")
        assert rid == "req-1"
        assert "req-1" in diffusion._pending_submissions
        assert len(fake_engine.submitted_requests) == 1

        out = await diffusion.collect("req-1")
        assert out.request_id == "req-1"
        assert "req-1" not in diffusion._pending_submissions
        assert fake_engine.collect_calls == [("sid-1", None)]
    finally:
        diffusion.close()


@pytest.mark.asyncio
async def test_collect_timeout_keeps_pending_request(monkeypatch):
    fake_engine = _FakeEngine()
    fake_engine.raise_timeout_on_collect = True
    _patch_init(monkeypatch, fake_engine)

    diffusion = AsyncOmniDiffusion(model="dummy-model", od_config=_DummyConfig())
    try:
        sampling_params = OmniDiffusionSamplingParams()
        await diffusion.submit(prompt="p", sampling_params=sampling_params, request_id="req-timeout")
        with pytest.raises(TimeoutError):
            await diffusion.collect("req-timeout", timeout_s=0.1)
        assert "req-timeout" in diffusion._pending_submissions
    finally:
        diffusion.close()


@pytest.mark.asyncio
async def test_generate_uses_submit_collect_path(monkeypatch):
    fake_engine = _FakeEngine()
    _patch_init(monkeypatch, fake_engine)

    diffusion = AsyncOmniDiffusion(model="dummy-model", od_config=_DummyConfig())
    try:
        out = await diffusion.generate(
            prompt={"prompt": "p"},
            sampling_params=OmniDiffusionSamplingParams(),
            request_id="req-generate",
        )
        assert out.request_id == "req-generate"
        assert fake_engine.submitted_ids == ["sid-1"]
        assert fake_engine.collect_calls == [("sid-1", None)]
    finally:
        diffusion.close()

