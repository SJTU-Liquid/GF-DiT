# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for DiffusionWorker class.

This module tests the DiffusionWorker implementation:
- load_weights: Loading model weights
- sleep: Putting worker into sleep mode (levels 1 and 2)
- wake_up: Waking worker from sleep mode
"""

from unittest.mock import call

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.worker.diffusion_worker import (
    _PendingRuntimeV2Submission,
    DiffusionWorker,
    WorkerProc,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def mock_od_config(mocker: MockerFixture):
    """Create a mock OmniDiffusionConfig."""
    config = mocker.Mock()
    config.num_gpus = 1
    config.master_port = 12345
    config.enable_sleep_mode = False
    config.cache_backend = None
    config.cache_config = None
    config.model = "test-model"
    config.model_class_name = "WanPipeline"
    config.enable_runtime_v2 = False
    config.runtime_v2_denoise_chunk_size = 1
    return config


@pytest.fixture
def mock_gpu_worker(mocker: MockerFixture, mock_od_config):
    """Create a DiffusionWorker with mocked initialization."""
    mocker.patch.object(DiffusionWorker, "init_device")
    mocker.patch.object(DiffusionWorker, "load_model")
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config)
    # Mock the model_runner with pipeline
    worker.model_runner = mocker.Mock()
    worker.model_runner.pipeline = mocker.Mock()
    worker.device = torch.device("cuda", 0)
    worker._sleep_saved_buffers = {}
    return worker


class TestDiffusionWorkerLoadWeights:
    """Test DiffusionWorker.load_weights method."""

    def test_load_weights_calls_pipeline(self, mocker: MockerFixture, mock_gpu_worker):
        """Test that load_weights delegates to model_runner.load_weights."""
        # Setup mock weights
        mock_weights = [
            ("layer1.weight", torch.randn(10, 10)),
            ("layer2.weight", torch.randn(20, 20)),
        ]
        expected_loaded = {"layer1.weight", "layer2.weight"}

        # Configure model_runner mock
        mock_gpu_worker.model_runner.load_weights = mocker.Mock(return_value=expected_loaded)

        # Call load_weights
        result = mock_gpu_worker.load_weights(mock_weights)

        # Verify model_runner.load_weights was called with the weights
        mock_gpu_worker.model_runner.load_weights.assert_called_once_with(mock_weights)
        assert result == expected_loaded

    def test_load_weights_empty_iterable(self, mocker: MockerFixture, mock_gpu_worker):
        """Test load_weights with empty weights iterable."""
        mock_gpu_worker.model_runner.load_weights = mocker.Mock(return_value=set())

        result = mock_gpu_worker.load_weights([])

        mock_gpu_worker.model_runner.load_weights.assert_called_once_with([])
        assert result == set()


class TestDiffusionWorkerSleep:
    """Test DiffusionWorker.sleep method."""

    def test_sleep_level_1(self, mocker: MockerFixture, mock_gpu_worker):
        """Test sleep mode level 1 (offload weights only)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")

        # Setup process-scoped memory mocks
        # Before sleep: 3GB used
        # After sleep: 1GB used (freed 2GB)
        mock_get_process_memory.side_effect = [
            3 * 1024**3,
            1 * 1024**3,
        ]

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()

        # Call sleep with level 1
        result = mock_gpu_worker.sleep(level=1)

        # Verify sleep was called with correct tags
        mock_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        assert result is True
        # Verify buffers were NOT saved (level 1 doesn't save buffers)
        assert len(mock_gpu_worker._sleep_saved_buffers) == 0

    def test_sleep_level_2(self, mocker: MockerFixture, mock_gpu_worker):
        """Test sleep mode level 2 (offload all, save buffers)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")

        # Setup process-scoped memory mocks
        mock_get_process_memory.side_effect = [
            5 * 1024**3,  # Before sleep
            1 * 1024**3,  # After sleep (freed 4GB)
        ]

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()

        # Mock pipeline buffers
        mock_buffer1 = torch.randn(10, 10)
        mock_buffer2 = torch.randn(20, 20)
        mock_gpu_worker.model_runner.pipeline.named_buffers = mocker.Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call sleep with level 2
        result = mock_gpu_worker.sleep(level=2)

        # Verify sleep was called with empty tags (offload all)
        mock_allocator.sleep.assert_called_once_with(offload_tags=tuple())
        assert result is True

        # Verify buffers were saved
        assert len(mock_gpu_worker._sleep_saved_buffers) == 2
        assert "buffer1" in mock_gpu_worker._sleep_saved_buffers
        assert "buffer2" in mock_gpu_worker._sleep_saved_buffers

    def test_sleep_memory_freed_validation(self, mocker: MockerFixture, mock_gpu_worker):
        """Test that sleep validates memory was actually freed."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")

        # Simulate process memory increase (should trigger assertion error)
        mock_get_process_memory.side_effect = [
            1 * 1024**3,  # Before sleep: 1GB used
            3 * 1024**3,  # After sleep: 3GB used (negative freed)
        ]

        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()

        # This should raise an assertion error
        with pytest.raises(AssertionError, match="Memory usage increased after sleeping"):
            mock_gpu_worker.sleep(level=1)

    def test_sleep_falls_back_to_device_memory_when_nvml_unavailable(self, mocker: MockerFixture, mock_gpu_worker):
        """Test sleep uses device-scoped fallback when NVML is unavailable."""

        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mock_platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")
        mock_get_process_memory.side_effect = [None, None]
        mock_platform.get_free_memory.side_effect = [
            1 * 1024**3,  # Before sleep
            3 * 1024**3,  # After sleep
        ]
        mock_platform.get_device_total_memory.return_value = 8 * 1024**3

        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()

        result = mock_gpu_worker.sleep(level=1)

        mock_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        assert result is True


class TestDiffusionWorkerWakeUp:
    """Test DiffusionWorker.wake_up method."""

    def test_wake_up_without_buffers(self, mocker: MockerFixture, mock_gpu_worker):
        """Test wake_up without saved buffers (level 1 sleep)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.wake_up = mocker.Mock()

        # Ensure no saved buffers
        mock_gpu_worker._sleep_saved_buffers = {}

        # Call wake_up
        result = mock_gpu_worker.wake_up(tags=["weights"])

        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(["weights"])
        assert result is True

    def test_wake_up_with_buffers(self, mocker: MockerFixture, mock_gpu_worker):
        """Test wake_up with saved buffers (level 2 sleep)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.wake_up = mocker.Mock()

        # Create saved buffers
        saved_buffer1 = torch.randn(10, 10)
        saved_buffer2 = torch.randn(20, 20)
        mock_gpu_worker._sleep_saved_buffers = {
            "buffer1": saved_buffer1,
            "buffer2": saved_buffer2,
        }

        # Mock pipeline buffers (these will be restored)
        mock_buffer1 = mocker.Mock()
        mock_buffer1.data = mocker.Mock()
        mock_buffer2 = mocker.Mock()
        mock_buffer2.data = mocker.Mock()

        mock_gpu_worker.model_runner.pipeline.named_buffers = mocker.Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call wake_up
        result = mock_gpu_worker.wake_up(tags=None)

        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(None)

        # Verify buffers were restored
        mock_buffer1.data.copy_.assert_called_once()
        mock_buffer2.data.copy_.assert_called_once()

        # Verify saved buffers were cleared
        assert len(mock_gpu_worker._sleep_saved_buffers) == 0
        assert result is True

    def test_wake_up_partial_buffer_restore(self, mocker: MockerFixture, mock_gpu_worker):
        """Test wake_up only restores buffers that were saved."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.wake_up = mocker.Mock()

        # Only save buffer1, not buffer2
        saved_buffer1 = torch.randn(10, 10)
        mock_gpu_worker._sleep_saved_buffers = {
            "buffer1": saved_buffer1,
        }

        # Mock pipeline has both buffers
        mock_buffer1 = mocker.Mock()
        mock_buffer1.data = mocker.Mock()
        mock_buffer2 = mocker.Mock()
        mock_buffer2.data = mocker.Mock()

        mock_gpu_worker.model_runner.pipeline.named_buffers = mocker.Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call wake_up
        result = mock_gpu_worker.wake_up()

        # Verify only buffer1 was restored
        mock_buffer1.data.copy_.assert_called_once()
        # buffer2 should NOT be restored since it wasn't saved
        mock_buffer2.data.copy_.assert_not_called()

        assert result is True


class TestDiffusionWorkerRuntimeV2:
    """Test runtime_v2 routing in DiffusionWorker."""

    def test_execute_model_uses_runtime_v2_when_enabled(self, mocker: MockerFixture, mock_gpu_worker):
        req = mocker.Mock()
        req.prompts = [{"prompt": "cat"}]

        mock_gpu_worker.od_config.runtime_v2_denoise_chunk_size = 2
        mock_runtime_v2_runner = mocker.Mock()
        expected_output = mocker.Mock()
        mock_runtime_v2_runner.submit.return_value = "req-id"
        mock_runtime_v2_runner.wait.return_value = expected_output
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner
        mock_gpu_worker.model_runner.execute_model = mocker.Mock()

        result = mock_gpu_worker.execute_model(req, mock_gpu_worker.od_config)

        mock_runtime_v2_runner.submit.assert_called_once_with(req, denoise_chunk_size=2)
        mock_runtime_v2_runner.wait.assert_called_once_with("req-id", timeout_s=None)
        mock_gpu_worker.model_runner.execute_model.assert_not_called()
        assert result == expected_output

    def test_execute_model_rejects_multi_prompt_requests(self, mocker: MockerFixture, mock_gpu_worker):
        req = mocker.Mock()
        req.prompts = ["p0", "p1"]

        mock_runtime_v2_runner = mocker.Mock()
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner

        with pytest.raises(ValueError, match="single-prompt"):
            mock_gpu_worker.execute_model(req, mock_gpu_worker.od_config)

        mock_runtime_v2_runner.submit.assert_not_called()

    def test_shutdown_closes_runtime_v2_runner(self, mocker: MockerFixture, mock_gpu_worker):
        mock_runtime_v2_runner = mocker.Mock()
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner
        mock_destroy = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.destroy_distributed_env")

        mock_gpu_worker.shutdown()

        mock_runtime_v2_runner.close.assert_called_once_with()
        mock_destroy.assert_called_once_with()

    def test_runtime_v2_submit_delegation(self, mocker: MockerFixture, mock_gpu_worker):
        req = mocker.Mock()
        req.prompts = [{"prompt": "cat"}]
        mock_runtime_v2_runner = mocker.Mock()
        mock_runtime_v2_runner.submit.return_value = "req-id"
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner
        mock_gpu_worker.od_config.runtime_v2_denoise_chunk_size = 3

        request_id = mock_gpu_worker.runtime_v2_submit(req)

        mock_runtime_v2_runner.submit.assert_called_once_with(req, denoise_chunk_size=3)
        assert request_id == "req-id"

    def test_runtime_v2_wait_delegation(self, mocker: MockerFixture, mock_gpu_worker):
        mock_runtime_v2_runner = mocker.Mock()
        expected_output = mocker.Mock()
        mock_runtime_v2_runner.wait.return_value = expected_output
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner

        output = mock_gpu_worker.runtime_v2_wait("req-id", timeout_s=12.0)

        mock_runtime_v2_runner.wait.assert_called_once_with("req-id", timeout_s=12.0)
        assert output == expected_output

    def test_runtime_v2_poll_once_delegation(self, mocker: MockerFixture, mock_gpu_worker):
        mock_runtime_v2_runner = mocker.Mock()
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner

        mock_gpu_worker.runtime_v2_poll_once(timeout_s=0.02)

        mock_runtime_v2_runner.poll_once.assert_called_once_with(timeout_s=0.02)

    def test_runtime_v2_get_request_status_delegation(self, mocker: MockerFixture, mock_gpu_worker):
        mock_runtime_v2_runner = mocker.Mock()
        expected = ("pending", None)
        mock_runtime_v2_runner.get_request_status.return_value = expected
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner

        status = mock_gpu_worker.runtime_v2_get_request_status("req-id")

        mock_runtime_v2_runner.get_request_status.assert_called_once_with("req-id")
        assert status == expected

    def test_runtime_v2_release_request_delegation(self, mocker: MockerFixture, mock_gpu_worker):
        mock_runtime_v2_runner = mocker.Mock()
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner

        mock_gpu_worker.runtime_v2_release_request("req-id")

        mock_runtime_v2_runner.release_request.assert_called_once_with("req-id")

    def test_runtime_v2_entrypoint_submit_activates_lora(self, mocker: MockerFixture, mock_gpu_worker):
        req = mocker.Mock()
        req.prompts = [{"prompt": "cat"}]
        req.sampling_params = mocker.Mock()
        req.sampling_params.lora_request = None
        req.sampling_params.lora_scale = 0.9

        mock_runtime_v2_runner = mocker.Mock()
        mock_runtime_v2_runner.submit.return_value = "req-id"
        mock_gpu_worker.runtime_v2_runner = mock_runtime_v2_runner
        mock_gpu_worker.lora_manager = mocker.Mock()

        request_id = mock_gpu_worker.runtime_v2_entrypoint_submit(req, denoise_chunk_size=4)

        mock_gpu_worker.lora_manager.set_active_adapter.assert_called_once_with(None, 0.9)
        mock_runtime_v2_runner.submit.assert_called_once_with(req, denoise_chunk_size=4)
        assert request_id == "req-id"

    def test_supports_runtime_v2_uses_registry(self, mock_gpu_worker):
        mock_gpu_worker.od_config.model_class_name = "QwenImagePipeline"
        assert mock_gpu_worker._supports_runtime_v2() is True
        mock_gpu_worker.od_config.model_class_name = "FluxPipeline"
        assert mock_gpu_worker._supports_runtime_v2() is False

    def test_maybe_init_runtime_v2_runner_fails_for_unsupported_model(
        self, mocker: MockerFixture, mock_gpu_worker
    ):
        mock_gpu_worker.od_config.enable_runtime_v2 = True
        mock_gpu_worker.od_config.model_class_name = "FluxPipeline"

        with pytest.raises(ValueError, match="does not support"):
            mock_gpu_worker._maybe_init_runtime_v2_runner()


class TestWorkerProcRuntimeV2Cleanup:
    """Test runtime_v2 entrypoint cleanup in WorkerProc."""

    @staticmethod
    def _make_proc(mocker: MockerFixture, pending_ids: list[str]) -> WorkerProc:
        proc = WorkerProc.__new__(WorkerProc)
        proc.worker = mocker.Mock()
        proc.result_mq = None
        proc._pending_runtime_v2 = {
            request_id: _PendingRuntimeV2Submission(
                scheduler_req_id=f"scheduler-{idx}",
                should_reply=False,
            )
            for idx, request_id in enumerate(pending_ids)
        }
        return proc

    @pytest.mark.parametrize("status,payload", [("finished", "done"), ("failed", "boom")])
    def test_drain_runtime_v2_completions_releases_terminal_request(
        self,
        mocker: MockerFixture,
        status: str,
        payload: object,
    ) -> None:
        proc = self._make_proc(mocker, ["runtime-1"])
        proc.worker.runtime_v2_get_request_status.return_value = (status, payload)

        proc._drain_runtime_v2_completions()

        proc.worker.runtime_v2_release_request.assert_called_once_with("runtime-1")
        assert proc._pending_runtime_v2 == {}

    def test_fail_all_pending_runtime_v2_releases_each_request(self, mocker: MockerFixture) -> None:
        proc = self._make_proc(mocker, ["runtime-1", "runtime-2"])

        proc._fail_all_pending_runtime_v2(reason="worker shutdown")

        proc.worker.runtime_v2_release_request.assert_has_calls(
            [call("runtime-1"), call("runtime-2")],
            any_order=True,
        )
        assert proc._pending_runtime_v2 == {}
