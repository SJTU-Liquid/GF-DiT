# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2.adapters.common import (
    STANDARD_TASK_KINDS,
    build_chunked_dit_plan,
    output_artifact_layout,
    replicated_tensor_field,
)
from vllm_omni.diffusion.runtime_v2.codec import ArtifactLayoutCodec
from vllm_omni.diffusion.runtime_v2.interfaces import RuntimeV2Adapter, TaskCompiler, WorkerExecutor
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactValue,
    ExecutionGroupSpec,
    InferenceTask,
    OutputArtifactLayout,
    TaskKind,
    TensorFieldLayout,
)

logger = init_logger(__name__)

_DEFAULT_QWEN_HEIGHT = 1024
_DEFAULT_QWEN_WIDTH = 1024
_DEFAULT_QWEN_VAE_SCALE_FACTOR = 8
QWEN_IMAGE_STATE_CODEC_ID = "qwen_image.runtime_state.v1"
QWEN_IMAGE_DECODED_CODEC_ID = "qwen_image.decoded.v1"


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _pad_prompt_to_len(
    embeds: torch.Tensor,
    mask: torch.Tensor | None,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Right-pad (B, S, H) embeds and (B, S) mask along the sequence dim to
    ``target_len`` so they match the codec's fixed text_seq_len layout. The
    mask tail is zeroed, so ``mask.sum()`` (i.e. txt_seq_lens) still reports the
    real token count and the denoise executor can slice back to it."""
    cur = embeds.shape[1]
    if cur >= target_len:
        return embeds, mask
    pad = target_len - cur
    embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad))
    if mask is not None:
        mask = torch.nn.functional.pad(mask, (0, pad))
    return embeds, mask


def _trim_prompt_to_seq_lens(
    embeds: torch.Tensor | None,
    mask: torch.Tensor | None,
    seq_lens: list[int] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Inverse of :func:`_pad_prompt_to_len`: slice padded (B, S, H) embeds and
    (B, S) mask back to the real token count (max over batch txt_seq_lens) so
    the transformer processes only real tokens. A no-op when embeds is None or
    already at/under the real length."""
    if embeds is None or not seq_lens:
        return embeds, mask
    real = int(max(seq_lens))
    if real >= embeds.shape[1]:
        return embeds, mask
    return embeds[:, :real], (mask[:, :real] if mask is not None else None)


def _retrieve_timesteps(
    scheduler: Any,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"scheduler {scheduler.__class__} does not support custom timesteps")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        current_timesteps = scheduler.timesteps
        num_inference_steps = len(current_timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"scheduler {scheduler.__class__} does not support custom sigmas")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        current_timesteps = scheduler.timesteps
        num_inference_steps = len(current_timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        current_timesteps = scheduler.timesteps
    return current_timesteps, int(num_inference_steps)


def _tensor_tree_to_device(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    # Use strict type check so dict/list/tuple subclasses (e.g. diffusers
    # FrozenDict, NamedTuple) are preserved as-is. Those typically hold
    # only scalar config and would lose their subclass type — and break
    # attribute access like ``config.prediction_type`` — if rebuilt as
    # plain dict/list/tuple.
    if type(value) is list:
        return [_tensor_tree_to_device(item, device) for item in value]
    if type(value) is tuple:
        return tuple(_tensor_tree_to_device(item, device) for item in value)
    if type(value) is dict:
        return {key: _tensor_tree_to_device(item, device) for key, item in value.items()}
    return value


def _scheduler_metadata_copy(scheduler: Any | None) -> Any | None:
    if scheduler is None:
        return None
    scheduler_copy = copy.copy(scheduler)
    for name, value in vars(scheduler).items():
        setattr(scheduler_copy, name, _tensor_tree_to_device(value, "cpu"))
    return scheduler_copy


def _sched_field(field_path: tuple[str, ...], tensor: torch.Tensor) -> TensorFieldLayout:
    return replicated_tensor_field(field_path, tuple(tensor.shape), tensor.dtype)


def _scheduler_skeleton(scheduler: Any | None) -> Any | None:
    """Picklable shallow copy of the scheduler with every on-device tensor
    nulled. Those tensors (sigmas/timesteps/solver-history, ...) migrate via
    P2P -- see ``QwenImageStateLayoutCodec.migration_extra_fields`` -- so they
    must NOT ride the pack_metadata pickle (a 40 MB D2H on the migration
    critical path). CPU tensors built in the scheduler ``__init__`` are small
    and stay in the skeleton. Raises if a device tensor still remains."""
    if scheduler is None:
        return None
    skeleton = copy.copy(scheduler)
    for name, attr in vars(scheduler).items():
        if isinstance(attr, torch.Tensor):
            if attr.is_cuda:
                setattr(skeleton, name, None)
        elif type(attr) is list and any(
            isinstance(it, torch.Tensor) and it.is_cuda for it in attr
        ):
            setattr(
                skeleton,
                name,
                [
                    None if (isinstance(it, torch.Tensor) and it.is_cuda) else it
                    for it in attr
                ],
            )
    for name, attr in vars(skeleton).items():
        scan = [attr] if isinstance(attr, torch.Tensor) else (
            attr if type(attr) is list else ()
        )
        for it in scan:
            if isinstance(it, torch.Tensor) and it.is_cuda:
                raise RuntimeError(
                    f"scheduler attribute {name!r} still holds an on-device "
                    f"tensor (device={it.device}) after skeleton stripping -- "
                    "it must migrate via migration_extra_fields, not be pickled"
                )
    return skeleton


def _move_scheduler_tensors_(scheduler: Any | None, device: torch.device) -> None:
    if scheduler is None:
        return
    for name, value in vars(scheduler).items():
        setattr(scheduler, name, _tensor_tree_to_device(value, device))


@dataclass(frozen=True)
class QwenImageRuntimeRequest:
    diffusion_request: OmniDiffusionRequest
    request_id: str = ""
    denoise_chunk_size: int = 1
    priority: int = 0
    group_id: str | None = None

    def __post_init__(self) -> None:
        if self.denoise_chunk_size < 1:
            raise ValueError(f"denoise_chunk_size must be >= 1, got {self.denoise_chunk_size}")

        req = self.diffusion_request
        if len(req.request_ids) == 0:
            req.request_ids = [str(uuid.uuid4())]
        if not self.request_id:
            object.__setattr__(self, "request_id", req.request_ids[0])
        if req.request_ids[0] != self.request_id:
            req.request_ids[0] = self.request_id


@dataclass
class QwenImageDecodedValue:
    """Wrap the raw decoded tensor so the QwenImage decoded codec's
    field_path=('value',) resolves through this struct to the tensor.
    Mirror of WanDecodedValue; see that class for the failure mode.
    """

    value: torch.Tensor | None = None


@dataclass
class QwenImageRuntimeState:
    request: OmniDiffusionRequest
    prompt: str
    negative_prompt: str | None
    height: int
    width: int
    num_steps: int
    output_type: str | None
    max_sequence_length: int
    device: torch.device
    dtype: torch.dtype
    guidance_scale: float
    true_cfg_scale: float
    prompt_embeds: torch.Tensor | None = None
    prompt_embeds_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    img_shapes: list[list[tuple[int, int, int]]] = field(default_factory=list)
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None
    guidance: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    scheduler: Any | None = None
    do_true_cfg: bool = False
    attention_kwargs: dict[str, Any] = field(default_factory=dict)


class QwenImageTaskCompiler(TaskCompiler):
    def __init__(
        self,
        default_denoise_chunk_size: int = 1,
    ) -> None:
        if default_denoise_chunk_size < 1:
            raise ValueError("default_denoise_chunk_size must be >= 1")
        self.default_denoise_chunk_size = default_denoise_chunk_size

    def compile_request(self, request: Any):
        if not isinstance(request, QwenImageRuntimeRequest):
            raise TypeError(f"unsupported request type: {type(request)!r}")

        req = request.diffusion_request
        request_id = request.request_id
        num_steps = int(req.sampling_params.num_inference_steps or 50)
        if num_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_steps}")
        text_seq_len = int(req.sampling_params.max_sequence_length or 512)
        if text_seq_len < 1:
            raise ValueError(f"max_sequence_length must be >= 1, got {text_seq_len}")
        latent_seq_len = self._estimate_packed_latent_seq_len(req)
        chunk_size = int(request.denoise_chunk_size or self.default_denoise_chunk_size)
        _prompt, negative_prompt = _extract_single_prompt(req)
        true_cfg_scale = float(req.sampling_params.true_cfg_scale or 4.0)
        num_images_per_prompt = (
            int(req.sampling_params.num_outputs_per_prompt) if req.sampling_params.num_outputs_per_prompt > 0 else 1
        )

        plan = build_chunked_dit_plan(
            request_id=request_id,
            request_value=req,
            request_type="qwen_image_runtime_v2",
            num_steps=num_steps,
            chunk_size=chunk_size,
            priority=request.priority,
            group_id=request.group_id,
            text_seq_len=text_seq_len,
            latent_seq_len=latent_seq_len,
            state_codec_id=QWEN_IMAGE_STATE_CODEC_ID,
            decoded_codec_id=QWEN_IMAGE_DECODED_CODEC_ID,
            metadata={
                "height": max(1, int(req.sampling_params.height or _DEFAULT_QWEN_HEIGHT)),
                "width": max(1, int(req.sampling_params.width or _DEFAULT_QWEN_WIDTH)),
                "num_images_per_prompt": num_images_per_prompt,
                "do_true_cfg": true_cfg_scale > 1 and negative_prompt is not None,
                "output_type": req.sampling_params.output_type or "pil",
            },
        )
        logger.info(
            "runtime_v2 compile: request_id=%s adapter=qwen_image steps=%s chunk=%s denoise_tasks=%s total_tasks=%s",
            request_id,
            num_steps,
            chunk_size,
            sum(1 for task in plan.tasks.values() if task.kind == TaskKind.DIT_STEP_CHUNK),
            len(plan.tasks),
        )
        return plan

    @staticmethod
    def _estimate_packed_latent_seq_len(req: OmniDiffusionRequest) -> int:
        height = max(1, int(req.sampling_params.height or _DEFAULT_QWEN_HEIGHT))
        width = max(1, int(req.sampling_params.width or _DEFAULT_QWEN_WIDTH))
        packed_h = max(1, height // (_DEFAULT_QWEN_VAE_SCALE_FACTOR * 2))
        packed_w = max(1, width // (_DEFAULT_QWEN_VAE_SCALE_FACTOR * 2))
        return max(1, packed_h * packed_w)


def _extract_single_prompt(req: OmniDiffusionRequest) -> tuple[str, str | None]:
    if len(req.prompts) != 1:
        raise ValueError("runtime_v2 currently supports exactly one prompt per request")
    prompt_data = req.prompts[0]
    if isinstance(prompt_data, str):
        return prompt_data, None
    prompt = str(prompt_data.get("prompt") or "")
    negative_prompt = prompt_data.get("negative_prompt")
    if negative_prompt is None:
        return prompt, None
    return prompt, str(negative_prompt)


class QwenImageStateLayoutCodec(ArtifactLayoutCodec):
    codec_id = QWEN_IMAGE_STATE_CODEC_ID

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def describe_output(
        self,
        *,
        group: ExecutionGroupSpec,
        group_rank: int,
        request_metadata: Mapping[str, Any],
        artifact: ArtifactHandle,
    ) -> OutputArtifactLayout:
        self._validate(group, group_rank, artifact)
        batch = self._metadata_int(request_metadata, "num_images_per_prompt", 1)
        text_seq_len = self._metadata_int(request_metadata, "text_seq_len", 512)
        latent_seq_len = self._metadata_int(request_metadata, "latent_seq_len", 1)
        num_steps = self._metadata_int(request_metadata, "num_steps", 1)
        prompt_dtype = self._prompt_dtype()
        fields: list[TensorFieldLayout] = [
            replicated_tensor_field(
                "prompt_embeds",
                (batch, text_seq_len, self._prompt_hidden_size()),
                prompt_dtype,
            ),
            replicated_tensor_field("prompt_embeds_mask", (batch, text_seq_len), torch.long),
        ]
        if bool(request_metadata.get("do_true_cfg", False)):
            fields.extend(
                (
                    replicated_tensor_field(
                        "negative_prompt_embeds",
                        (batch, text_seq_len, self._prompt_hidden_size()),
                        prompt_dtype,
                    ),
                    replicated_tensor_field("negative_prompt_embeds_mask", (batch, text_seq_len), torch.long),
                )
            )

        artifact_id = artifact.artifact_id
        has_latents = artifact_id == "state_latent" or artifact_id == "state_timestep" or artifact_id.startswith(
            "state_denoised"
        )
        if has_latents:
            # state.latents holds the PACKED latents (2x2-patchified), so the
            # channel dim is transformer.in_channels, not in_channels // 4 (the
            # pre-pack VAE channel count). Store REPLICATED (full) rather than
            # sp_sharded: the denoise executor already materializes the full
            # latent every step (model.forward does its own SP split/gather
            # internally), so sharding storage only adds a redundant
            # gather-on-materialize + reshard-on-store at each task boundary.
            # Full storage matches the single-card baseline layout and mirrors
            # WAN's cbf54c1. See [[todo-qwen-sp-layout]] Stage A.
            latent_channels = int(self.pipeline.transformer.in_channels)
            fields.append(
                replicated_tensor_field(
                    "latents",
                    (batch, latent_seq_len, latent_channels),
                    prompt_dtype,
                )
            )
            if bool(getattr(self.pipeline.transformer, "guidance_embeds", False)):
                fields.append(replicated_tensor_field("guidance", (batch,), torch.float32))

        has_timesteps = artifact_id == "state_timestep" or artifact_id.startswith("state_denoised")
        if has_timesteps:
            fields.append(replicated_tensor_field("timesteps", (num_steps,), torch.float32))

        return output_artifact_layout(artifact, tuple(fields))

    def pack_metadata(self, *, value: Any, layout: OutputArtifactLayout) -> bytes:
        # Declared tensor fields ride P2P (describe_output); scheduler on-device
        # tensors ride P2P too (migration_extra_fields) and are nulled in the
        # skeleton. Nothing on-device travels through this pickle.
        self._assert_no_undeclared_tensor_fields(value, layout.tensors)
        replacements: dict[str, Any] = {f.field_path[0]: None for f in layout.tensors}
        replacements["scheduler"] = _scheduler_skeleton(getattr(value, "scheduler", None))
        replacements["device"] = None
        skeleton = dataclasses.replace(value, **replacements)
        request = skeleton.request
        if request is not None:
            sp = getattr(request, "sampling_params", None)
            if sp is not None and getattr(sp, "generator", None) is not None:
                clean_sp = dataclasses.replace(sp, generator=None)
                clean_request = dataclasses.replace(request, sampling_params=clean_sp)
                skeleton.request = clean_request
        return pickle.dumps(skeleton, protocol=pickle.HIGHEST_PROTOCOL)

    def migration_extra_fields(
        self, *, value: Any, layout: OutputArtifactLayout
    ) -> tuple[TensorFieldLayout, ...]:
        # The diffusers scheduler keeps on-device tensors (sigmas, timesteps,
        # solver history, ...) that describe_output does not declare -- their
        # count/shape is runtime denoise state, not request metadata. Enumerate
        # them from the live value on the src leader so each migrates via P2P
        # instead of a ~40 MB D2H + pickle in pack_metadata (the bimodal slow
        # path on heavy/L migrations).
        scheduler = getattr(value, "scheduler", None)
        if scheduler is None:
            return ()
        declared = {f.field_path for f in layout.tensors}
        fields: list[TensorFieldLayout] = []
        for name, attr in vars(scheduler).items():
            if isinstance(attr, torch.Tensor):
                if attr.is_cuda:
                    fields.append(_sched_field(("scheduler", name), attr))
            elif type(attr) is list:
                for idx, item in enumerate(attr):
                    if isinstance(item, torch.Tensor) and item.is_cuda:
                        fields.append(_sched_field(("scheduler", name, str(idx)), item))
        return tuple(f for f in fields if f.field_path not in declared)

    def assemble(
        self,
        *,
        metadata_bytes: bytes,
        tensors: Mapping[tuple[str, ...], Any],
        layout: OutputArtifactLayout,
        device: Any = None,
    ) -> Any:
        if device is None:
            raise ValueError(
                "QwenImageStateLayoutCodec.assemble requires a device kwarg so the "
                "dst rank can rebind state.device after migration"
            )
        state = super().assemble(
            metadata_bytes=metadata_bytes, tensors=tensors, layout=layout, device=device,
        )
        # migration_extra_fields tensors (scheduler device tensors) arrive in
        # `tensors` but are not in layout.tensors; write them back into the
        # nulled scheduler-skeleton slots by their (nested) field path.
        declared = {f.field_path for f in layout.tensors}
        for field_path, tensor in tensors.items():
            path = tuple(field_path)
            if path not in declared:
                self._write_field(state, path, tensor)
        state.device = torch.device(device) if not isinstance(device, torch.device) else device
        _move_scheduler_tensors_(state.scheduler, state.device)
        if state.scheduler is not None and state.timesteps is not None:
            state.scheduler.timesteps = state.timesteps
        return state

    def _validate(self, group: ExecutionGroupSpec, group_rank: int, artifact: ArtifactHandle) -> None:
        if group_rank < 0 or group_rank >= len(group.ranks):
            raise ValueError(f"group_rank {group_rank} out of range for group {group.group_id}")
        if artifact.codec_id != self.codec_id:
            raise ValueError(
                f"qwen-image state codec cannot describe artifact {artifact.artifact_id} "
                f"with codec_id={artifact.codec_id!r}"
            )

    @staticmethod
    def _metadata_int(metadata: Mapping[str, Any], key: str, default: int) -> int:
        value = int(metadata.get(key, default))
        if value < 1:
            raise ValueError(f"{key} must be >= 1, got {value}")
        return value

    def _prompt_dtype(self) -> torch.dtype:
        text_encoder = getattr(self.pipeline, "text_encoder", None)
        dtype = getattr(text_encoder, "dtype", None)
        if dtype is not None:
            return dtype
        return getattr(self.pipeline.transformer, "dtype", torch.float32)

    def _prompt_hidden_size(self) -> int:
        candidates = (
            getattr(getattr(self.pipeline, "text_encoder", None), "config", None),
            getattr(self.pipeline, "text_encoder", None),
            getattr(getattr(self.pipeline, "transformer", None), "config", None),
            getattr(self.pipeline, "transformer", None),
        )
        for obj in candidates:
            if obj is None:
                continue
            for attr in ("hidden_size", "d_model", "text_hidden_size", "joint_attention_dim", "caption_channels"):
                value = getattr(obj, attr, None)
                if value is not None:
                    return int(value)
        raise ValueError("cannot infer qwen-image prompt hidden size")


class QwenImageDecodedLayoutCodec(ArtifactLayoutCodec):
    codec_id = QWEN_IMAGE_DECODED_CODEC_ID

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def describe_output(
        self,
        *,
        group: ExecutionGroupSpec,
        group_rank: int,
        request_metadata: Mapping[str, Any],
        artifact: ArtifactHandle,
    ) -> OutputArtifactLayout:
        if group_rank < 0 or group_rank >= len(group.ranks):
            raise ValueError(f"group_rank {group_rank} out of range for group {group.group_id}")
        if artifact.codec_id != self.codec_id:
            raise ValueError(
                f"qwen-image decoded codec cannot describe artifact {artifact.artifact_id} "
                f"with codec_id={artifact.codec_id!r}"
            )
        batch = QwenImageStateLayoutCodec._metadata_int(request_metadata, "num_images_per_prompt", 1)
        height = QwenImageStateLayoutCodec._metadata_int(request_metadata, "height", _DEFAULT_QWEN_HEIGHT)
        width = QwenImageStateLayoutCodec._metadata_int(request_metadata, "width", _DEFAULT_QWEN_WIDTH)
        if request_metadata.get("output_type") == "latent":
            latent_seq_len = QwenImageStateLayoutCodec._metadata_int(request_metadata, "latent_seq_len", 1)
            latent_channels = int(self.pipeline.transformer.in_channels)
            dtype = getattr(getattr(self.pipeline, "text_encoder", None), "dtype", torch.float32)
            tensors = (replicated_tensor_field("value", (batch, latent_seq_len, latent_channels), dtype),)
        else:
            dtype = getattr(self.pipeline.vae, "dtype", torch.float32)
            tensors = (replicated_tensor_field("value", (batch, 3, height, width), dtype),)
        return output_artifact_layout(artifact, tensors)


class _BaseQwenImageExecutor(WorkerExecutor):
    def __init__(self, pipeline: Any, output_codecs: Mapping[str, ArtifactLayoutCodec] | None = None) -> None:
        self.pipeline = pipeline
        self._output_codecs = dict(output_codecs or {})

    @property
    def output_codecs(self) -> Mapping[str, ArtifactLayoutCodec]:
        return self._output_codecs

    @staticmethod
    def _single_input(task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> Any:
        if len(task.inputs) != 1:
            raise ValueError(f"{task.kind} expects exactly one input artifact")
        artifact_id = task.inputs[0].artifact_id
        if artifact_id not in resolved_inputs:
            raise KeyError(f"missing resolved input for artifact {artifact_id}")
        return resolved_inputs[artifact_id]

    @staticmethod
    def _scheduler_step_maybe_with_cfg(
        scheduler: Any,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        latents: torch.Tensor,
        do_true_cfg: bool,
    ) -> torch.Tensor:
        cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
        if cfg_parallel_ready:
            cfg_group = get_cfg_group()
            cfg_rank = get_classifier_free_guidance_rank()
            if cfg_rank == 0:
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = latents.contiguous()
            cfg_group.broadcast(latents, src=0)
            return latents
        return scheduler.step(noise_pred, t, latents, return_dict=False)[0]


class QwenImageTextEncodeExecutor(_BaseQwenImageExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        req = cast(OmniDiffusionRequest, self._single_input(task, resolved_inputs))
        prompt, negative_prompt = _extract_single_prompt(req)

        height = int(req.sampling_params.height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor)
        width = int(req.sampling_params.width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor)
        num_steps = int(req.sampling_params.num_inference_steps or 50)
        if num_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_steps}")
        output_type = req.sampling_params.output_type or "pil"
        max_sequence_length = int(req.sampling_params.max_sequence_length or 512)
        true_cfg_scale = float(req.sampling_params.true_cfg_scale or 4.0)
        guidance_scale = (
            float(req.sampling_params.guidance_scale)
            if req.sampling_params.guidance_scale_provided
            else 1.0
        )
        num_images_per_prompt = (
            int(req.sampling_params.num_outputs_per_prompt) if req.sampling_params.num_outputs_per_prompt > 0 else 1
        )
        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._attention_kwargs = {}
        self.pipeline._current_timestep = None
        self.pipeline._interrupt = False

        self.pipeline.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_embeds_mask=None,
            negative_prompt_embeds_mask=None,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=max_sequence_length,
        )

        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(req.sampling_params.seed)
        req.sampling_params.generator = generator

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.pipeline.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.pipeline.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=None,
                prompt_embeds_mask=None,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # Qwen's encode_prompt returns variable-length embeddings (the real
        # token count, unpadded), but the state codec declares prompt_embeds at
        # the fixed text_seq_len=max_sequence_length. Cross-SP-group migration
        # slices tensors by the declared shape, so an unpadded (B, real, H)
        # tensor overruns. Pad to max_sequence_length here (with a zeroed mask
        # tail) so storage/migration match the declared layout; the denoise
        # executor slices back to the real length via txt_seq_lens so the
        # transformer compute (and the cost model) stay at the true length.
        prompt_embeds, prompt_embeds_mask = _pad_prompt_to_len(
            prompt_embeds, prompt_embeds_mask, max_sequence_length
        )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds, negative_prompt_embeds_mask = _pad_prompt_to_len(
                negative_prompt_embeds, negative_prompt_embeds_mask, max_sequence_length
            )

        state = QwenImageRuntimeState(
            request=req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_steps=num_steps,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
            device=self.pipeline.device,
            dtype=prompt_embeds.dtype,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            do_true_cfg=do_true_cfg,
        )
        return (ArtifactValue(handle=task.outputs[0], value=state),)


class QwenImageLatentPrepareExecutor(_BaseQwenImageExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        state = cast(QwenImageRuntimeState, self._single_input(task, resolved_inputs))
        req = state.request

        # See WanLatentPrepareExecutor: reconstruct the generator on this rank
        # when the migration pipeline stripped it, so cross-group runs match
        # the single-group baseline noise.
        if req.sampling_params.generator is None and req.sampling_params.seed is not None:
            req.sampling_params.generator = torch.Generator(device=self.pipeline.device).manual_seed(
                int(req.sampling_params.seed)
            )

        num_channels_latents = self.pipeline.transformer.in_channels // 4
        state.latents = self.pipeline.prepare_latents(
            batch_size=state.prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=state.height,
            width=state.width,
            dtype=state.prompt_embeds.dtype,
            device=state.device,
            generator=req.sampling_params.generator,
            latents=req.sampling_params.latents,
        )
        state.img_shapes = [[(1, state.height // self.pipeline.vae_scale_factor // 2, state.width // self.pipeline.vae_scale_factor // 2)]]
        state.txt_seq_lens = (
            state.prompt_embeds_mask.sum(dim=1).tolist() if state.prompt_embeds_mask is not None else None
        )
        state.negative_txt_seq_lens = (
            state.negative_prompt_embeds_mask.sum(dim=1).tolist()
            if state.negative_prompt_embeds_mask is not None
            else None
        )
        attention_kwargs = getattr(self.pipeline, "attention_kwargs", None)
        state.attention_kwargs = dict(attention_kwargs) if attention_kwargs is not None else {}
        if self.pipeline.transformer.guidance_embeds:
            guidance = torch.full([1], state.guidance_scale, dtype=torch.float32)
            state.guidance = guidance.expand(state.latents.shape[0])
        else:
            state.guidance = None
        return (ArtifactValue(handle=task.outputs[0], value=state),)


class QwenImageTimestepPrepareExecutor(_BaseQwenImageExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        state = cast(QwenImageRuntimeState, self._single_input(task, resolved_inputs))
        if state.latents is None:
            raise RuntimeError("state.latents is None in timestep_prepare stage")

        state.scheduler = copy.deepcopy(self.pipeline.scheduler)
        scheduler_config = state.scheduler.config
        sigmas = state.request.sampling_params.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / state.num_steps, state.num_steps)
        mu = _calculate_shift(
            state.latents.shape[1],
            scheduler_config.get("base_image_seq_len", 256),
            scheduler_config.get("max_image_seq_len", 4096),
            scheduler_config.get("base_shift", 0.5),
            scheduler_config.get("max_shift", 1.15),
        )
        state.timesteps, state.num_steps = _retrieve_timesteps(
            state.scheduler,
            state.num_steps,
            device=state.device,
            sigmas=sigmas,
            mu=mu,
        )
        if hasattr(state.scheduler, "set_begin_index"):
            state.scheduler.set_begin_index(0)
        self.pipeline._num_timesteps = len(state.timesteps)
        return (ArtifactValue(handle=task.outputs[0], value=state),)


class QwenImageDenoiseExecutor(_BaseQwenImageExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        if task.step_range is None:
            raise ValueError("dit_step_chunk requires step_range")

        state = cast(QwenImageRuntimeState, self._single_input(task, resolved_inputs))
        if state.timesteps is None or state.latents is None or state.scheduler is None:
            raise RuntimeError("state is not ready for qwen-image denoising")

        self.pipeline.transformer.do_true_cfg = state.do_true_cfg
        start = max(0, task.step_range.start)
        end = min(task.step_range.end, len(state.timesteps))
        # prompt_embeds is stored padded to max_sequence_length (so cross-group
        # migration matches the fixed codec layout). Slice back to the real
        # token count before the transformer, which uses encoder_hidden_states.
        # shape[1] as the text length -- keeping compute (and the cost model) at
        # the true length instead of the padded one.
        pos_embeds, pos_mask = _trim_prompt_to_seq_lens(
            state.prompt_embeds, state.prompt_embeds_mask, state.txt_seq_lens
        )
        neg_embeds, neg_mask = _trim_prompt_to_seq_lens(
            state.negative_prompt_embeds, state.negative_prompt_embeds_mask, state.negative_txt_seq_lens
        )
        for idx in range(start, end):
            t = state.timesteps[idx]
            self.pipeline._current_timestep = t
            timestep = t.expand(state.latents.shape[0]).to(device=state.latents.device, dtype=state.latents.dtype)
            positive_kwargs = {
                "hidden_states": state.latents,
                "timestep": timestep / 1000,
                "guidance": state.guidance,
                "encoder_hidden_states_mask": pos_mask,
                "encoder_hidden_states": pos_embeds,
                "img_shapes": state.img_shapes,
                "txt_seq_lens": state.txt_seq_lens,
                "return_dict": False,
                "attention_kwargs": state.attention_kwargs,
            }
            negative_kwargs = None
            if state.do_true_cfg:
                negative_kwargs = {
                    "hidden_states": state.latents,
                    "timestep": timestep / 1000,
                    "guidance": state.guidance,
                    "encoder_hidden_states_mask": neg_mask,
                    "encoder_hidden_states": neg_embeds,
                    "img_shapes": state.img_shapes,
                    "txt_seq_lens": state.negative_txt_seq_lens,
                    "return_dict": False,
                    "attention_kwargs": state.attention_kwargs,
                }
            noise_pred = self.pipeline.predict_noise_maybe_with_cfg(
                state.do_true_cfg,
                state.true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                True,
                None,
            )
            state.latents = self._scheduler_step_maybe_with_cfg(
                state.scheduler,
                noise_pred,
                t,
                state.latents,
                state.do_true_cfg,
            )

        if end >= len(state.timesteps):
            self.pipeline._current_timestep = None
            state.scheduler = None

        return (ArtifactValue(handle=task.outputs[0], value=state),)


class QwenImageDecodeExecutor(_BaseQwenImageExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        state = cast(QwenImageRuntimeState, self._single_input(task, resolved_inputs))
        if state.latents is None:
            raise RuntimeError("state.latents is None in decode stage")

        if state.output_type == "latent":
            output = state.latents
        else:
            latents = self.pipeline._unpack_latents(state.latents, state.height, state.width, self.pipeline.vae_scale_factor)
            latents = latents.to(self.pipeline.vae.dtype)
            latents_mean = (
                torch.tensor(self.pipeline.vae.config.latents_mean)
                .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                1.0
                / torch.tensor(self.pipeline.vae.config.latents_std)
                .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std + latents_mean
            output = self.pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return (ArtifactValue(handle=task.outputs[0], value=QwenImageDecodedValue(value=output)),)


class QwenImageFinalizeExecutor(_BaseQwenImageExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        decoded = self._single_input(task, resolved_inputs)
        output = decoded.value if isinstance(decoded, QwenImageDecodedValue) else decoded
        return (ArtifactValue(handle=task.outputs[0], value=DiffusionOutput(output=output)),)


class QwenImageRuntimeV2Adapter(RuntimeV2Adapter):
    model_class_name = "QwenImagePipeline"

    @property
    def supported_task_kinds(self) -> tuple[TaskKind, ...]:
        return STANDARD_TASK_KINDS

    def normalize_request(self, request: Any, denoise_chunk_size: int) -> QwenImageRuntimeRequest:
        if isinstance(request, QwenImageRuntimeRequest):
            return request
        if not isinstance(request, OmniDiffusionRequest):
            raise TypeError(f"unsupported runtime_v2 request type: {type(request)!r}")
        return QwenImageRuntimeRequest(diffusion_request=request, denoise_chunk_size=denoise_chunk_size)

    def build_task_compiler(
        self,
        default_denoise_chunk_size: int,
        *,
        od_config: Any = None,
        pipeline: Any = None,
    ) -> TaskCompiler:
        return QwenImageTaskCompiler(
            default_denoise_chunk_size=default_denoise_chunk_size,
        )

    def build_executors(self, pipeline: Any) -> dict[TaskKind, WorkerExecutor]:
        self.validate_pipeline(pipeline, getattr(pipeline, "od_config", None))
        state_codec = QwenImageStateLayoutCodec(pipeline)
        decoded_codec = QwenImageDecodedLayoutCodec(pipeline)
        return {
            TaskKind.TEXT_ENCODE: QwenImageTextEncodeExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.DIT_PREPARE: QwenImageLatentPrepareExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.TIMESTEP_PREPARE: QwenImageTimestepPrepareExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.DIT_STEP_CHUNK: QwenImageDenoiseExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.VAE_DECODE: QwenImageDecodeExecutor(
                pipeline,
                {decoded_codec.codec_id: decoded_codec},
            ),
            TaskKind.FINALIZE: QwenImageFinalizeExecutor(pipeline),
        }

    def validate_pipeline(self, pipeline: Any, od_config: Any) -> None:
        if pipeline is None:
            raise ValueError("runtime_v2 qwen-image adapter requires an initialized pipeline")
        for attr in (
            "check_inputs",
            "check_cfg_parallel_validity",
            "encode_prompt",
            "prepare_latents",
            "default_sample_size",
            "device",
            "transformer",
            "scheduler",
            "vae",
            "vae_scale_factor",
            "_pack_latents",
            "_unpack_latents",
        ):
            if not hasattr(pipeline, attr):
                raise ValueError(f"qwen-image runtime_v2 adapter requires pipeline.{attr}")
