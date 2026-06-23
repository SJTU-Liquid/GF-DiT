# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import inspect
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import PIL.Image
import torch
from diffusers.video_processor import VideoProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import retrieve_latents
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
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)
WAN_STATE_CODEC_ID = "wan.runtime_state.v1"
WAN_DECODED_CODEC_ID = "wan.decoded.v1"


@dataclass(frozen=True)
class WanRuntimeRequest:
    diffusion_request: OmniDiffusionRequest
    request_id: str = ""
    denoise_chunk_size: int = 1
    priority: int = 0
    group_id: str | None = None

    def __post_init__(self) -> None:
        if self.denoise_chunk_size < 1:
            raise ValueError(f"denoise_chunk_size must be >=1, got {self.denoise_chunk_size}")

        req = self.diffusion_request
        if len(req.request_ids) == 0:
            req.request_ids = [str(uuid.uuid4())]
        if not self.request_id:
            object.__setattr__(self, "request_id", req.request_ids[0])
        if req.request_ids[0] != self.request_id:
            req.request_ids[0] = self.request_id


@dataclass
class WanDecodedValue:
    """Wrap the raw decoded tensor so the WAN_DECODED codec's
    field_path=('value',) resolves through this struct to the tensor.

    Without the wrapper, the data_plane unwraps ArtifactValue.value (the raw
    tensor) before calling _read_field, and _read_field then tries
    `getattr(Tensor, 'value')` -> AttributeError. Concurrent cross-group
    consumption of the decoded artifact (e.g. VAE on group A, finalize on
    group B under dynamic group synthesis) triggers the migration path that
    exposes this. WanFinalizeExecutor unwraps on consume.
    """

    value: torch.Tensor | None = None


@dataclass
class WanRuntimeState:
    request: OmniDiffusionRequest
    prompt: str
    negative_prompt: str | None
    height: int
    width: int
    num_frames: int
    num_steps: int
    output_type: str | None
    device: torch.device
    dtype: torch.dtype
    guidance_low: float
    guidance_high: float
    boundary_timestep: float
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    attention_kwargs: dict[str, Any] = field(default_factory=dict)
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    scheduler: Any | None = None
    latent_condition: torch.Tensor | None = None
    first_frame_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class WanLatentGeometry:
    """Pixel -> DiT-token geometry for a Wan model.

    All fields come from the served model's own config (VAE compression and
    transformer patch size); nothing here is model-specific magic. The cost
    model is keyed by ``dit_latent_seq_len``, so the profiler, the runtime
    compiler and the workload generator must all resolve geometry the same
    way or the cost lookup lands on the wrong row.
    """

    vae_scale_spatial: int
    vae_scale_temporal: int
    patch_t: int
    patch_h: int
    patch_w: int

    @staticmethod
    def _cfg_get(config: Any, name: str) -> Any:
        # diffusers exposes config as a plain dict (load_config), a FrozenDict,
        # or a Config namespace object depending on the call path — the latter
        # is not subscriptable, so fall back to attribute access.
        if isinstance(config, Mapping):
            return config[name]
        return getattr(config, name)

    @classmethod
    def from_configs(cls, vae_config: Any, transformer_config: Any) -> "WanLatentGeometry":
        patch = list(cls._cfg_get(transformer_config, "patch_size"))
        return cls(
            vae_scale_spatial=int(cls._cfg_get(vae_config, "scale_factor_spatial")),
            vae_scale_temporal=int(cls._cfg_get(vae_config, "scale_factor_temporal")),
            patch_t=int(patch[0]),
            patch_h=int(patch[1]),
            patch_w=int(patch[2]),
        )

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "WanLatentGeometry":
        # Use the pipeline-level vae scale factors and flattened
        # transformer_config that the worker-side codec already relies on,
        # rather than reaching into pipeline.vae.config / .transformer.config
        # (whose type varies: dict, FrozenDict, or attribute-only Config).
        transformer_config = getattr(pipeline, "transformer_config", None)
        if transformer_config is None:
            transformer_config = pipeline.transformer.config
        patch = list(cls._cfg_get(transformer_config, "patch_size"))
        return cls(
            vae_scale_spatial=int(pipeline.vae_scale_factor_spatial),
            vae_scale_temporal=int(pipeline.vae_scale_factor_temporal),
            patch_t=int(patch[0]),
            patch_h=int(patch[1]),
            patch_w=int(patch[2]),
        )

    @classmethod
    def from_model(cls, model_name_or_path: str) -> "WanLatentGeometry":
        """Resolve geometry by loading only the VAE/transformer config.json
        (no weights) for ``model_name_or_path`` — works offline from cache."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        vae_config = AutoencoderKLWan.load_config(model_name_or_path, subfolder="vae")
        transformer_config = WanTransformer3DModel.load_config(
            model_name_or_path, subfolder="transformer"
        )
        return cls.from_configs(vae_config, transformer_config)

    @classmethod
    def default(cls) -> "WanLatentGeometry":
        # Fallback for the Wan2.2 family when no config source is reachable.
        # The runtime always resolves the real geometry from the served
        # model; this only keeps cost-model-less paths (fcfs/srtf) working.
        return cls(vae_scale_spatial=16, vae_scale_temporal=4, patch_t=1, patch_h=2, patch_w=2)

    def dit_latent_seq_len(self, height: int, width: int, num_frames: int) -> int:
        """DiT transformer sequence length for a pixel-space request shape:
        VAE-compress, then fold by the transformer patch size."""
        h_lat = max(1, int(height) // self.vae_scale_spatial)
        w_lat = max(1, int(width) // self.vae_scale_spatial)
        f_lat = max(1, (int(num_frames) + self.vae_scale_temporal - 1) // self.vae_scale_temporal)
        return max(
            1,
            (f_lat // self.patch_t) * (h_lat // self.patch_h) * (w_lat // self.patch_w),
        )


class WanTaskCompiler(TaskCompiler):
    def __init__(
        self,
        default_denoise_chunk_size: int = 1,
        *,
        latent_geometry: WanLatentGeometry | None = None,
        default_guidance_scale: float = 4.0,
        scheduler: Any | None = None,
        boundary_ratio: float | None = None,
        num_train_timesteps: int = 1000,
    ) -> None:
        if default_denoise_chunk_size < 1:
            raise ValueError("default_denoise_chunk_size must be >= 1")
        self.default_denoise_chunk_size = default_denoise_chunk_size
        self.latent_geometry = latent_geometry or WanLatentGeometry.default()
        self.default_guidance_scale = float(default_guidance_scale)
        self.scheduler = scheduler
        self.boundary_ratio = boundary_ratio
        self.num_train_timesteps = int(num_train_timesteps)

    def compile_request(self, request: Any):
        if not isinstance(request, WanRuntimeRequest):
            raise TypeError(f"unsupported request type: {type(request)!r}")

        req = request.diffusion_request
        request_id = request.request_id
        num_steps = int(req.sampling_params.num_inference_steps or 40)
        if num_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_steps}")
        text_seq_len = int(req.sampling_params.max_sequence_length or 512)
        if text_seq_len < 1:
            raise ValueError(f"max_sequence_length must be >= 1, got {text_seq_len}")
        latent_seq_len = self._estimate_dit_latent_seq_len(req)
        _prompt, negative_prompt = _extract_prompt(req)
        num_outputs = int(req.sampling_params.num_outputs_per_prompt or 1)
        guidance_low = (
            req.sampling_params.guidance_scale
            if req.sampling_params.guidance_scale_provided
            else self.default_guidance_scale
        )
        guidance_high = (
            float(req.sampling_params.guidance_scale_2)
            if req.sampling_params.guidance_scale_2 is not None
            else float(guidance_low)
        )
        dit_step_do_true_cfg = self._dit_step_do_true_cfg(
            req=req,
            num_steps=num_steps,
            guidance_low=float(guidance_low),
            guidance_high=float(guidance_high),
        )
        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else {}
        raw_image = multi_modal_data.get("image")
        has_image = len(raw_image) > 0 and raw_image[0] is not None if isinstance(raw_image, list) else raw_image is not None

        chunk_size = int(request.denoise_chunk_size or self.default_denoise_chunk_size)
        plan = build_chunked_dit_plan(
            request_id=request_id,
            request_value=req,
            request_type="wan_runtime_v2",
            num_steps=num_steps,
            chunk_size=chunk_size,
            priority=request.priority,
            group_id=request.group_id,
            text_seq_len=text_seq_len,
            latent_seq_len=latent_seq_len,
            state_codec_id=WAN_STATE_CODEC_ID,
            decoded_codec_id=WAN_DECODED_CODEC_ID,
            dit_step_do_true_cfg=dit_step_do_true_cfg,
            metadata={
                "height": max(1, int(req.sampling_params.height or 480)),
                "width": max(1, int(req.sampling_params.width or 832)),
                "num_frames": max(1, int(req.sampling_params.num_frames or 81)),
                "num_outputs_per_prompt": num_outputs,
                # CFG path is taken whenever guidance > 1, regardless of whether
                # the user supplied a negative_prompt: encode_prompt falls back to
                # the empty string and still allocates negative_prompt_embeds. This
                # metadata flag controls whether the codec includes that tensor in
                # the migration layout, so getting it wrong causes pickle to leak
                # the src-rank cuda tensor across to dst.
                "do_true_cfg": float(guidance_low) > 1.0 or float(guidance_high) > 1.0,
                "has_image": has_image,
                "output_type": req.sampling_params.output_type or "np",
            },
        )
        logger.info(
            "runtime_v2 compile: request_id=%s adapter=wan steps=%s chunk=%s denoise_tasks=%s total_tasks=%s",
            request_id,
            num_steps,
            chunk_size,
            sum(1 for task in plan.tasks.values() if task.kind == TaskKind.DIT_STEP_CHUNK),
            len(plan.tasks),
        )
        return plan

    def _estimate_dit_latent_seq_len(self, req: OmniDiffusionRequest) -> int:
        return self.latent_geometry.dit_latent_seq_len(
            height=int(req.sampling_params.height or 480),
            width=int(req.sampling_params.width or 832),
            num_frames=int(req.sampling_params.num_frames or 81),
        )

    def _dit_step_do_true_cfg(
        self,
        *,
        req: OmniDiffusionRequest,
        num_steps: int,
        guidance_low: float,
        guidance_high: float,
    ) -> tuple[bool, ...]:
        if guidance_low > 1.0 and guidance_high > 1.0:
            return (True,) * num_steps
        if guidance_low <= 1.0 and guidance_high <= 1.0:
            return (False,) * num_steps

        boundary_ratio = self.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = req.sampling_params.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = 0.875
        boundary_timestep = float(boundary_ratio) * float(self.num_train_timesteps)

        timesteps = self._scheduler_timesteps(num_steps)
        if timesteps is None:
            # Conservative fallback for unusual scheduler implementations: if
            # any phase uses CFG, avoid under-estimating deadline pressure.
            return (True,) * num_steps

        out: list[bool] = []
        for timestep in timesteps:
            current_guidance = guidance_high if timestep < boundary_timestep else guidance_low
            out.append(current_guidance > 1.0)
        return tuple(out)

    def _scheduler_timesteps(self, num_steps: int) -> tuple[float, ...] | None:
        if self.scheduler is None:
            return None
        try:
            scheduler = copy.deepcopy(self.scheduler)
            try:
                scheduler.set_timesteps(num_steps, device="cpu")
            except TypeError:
                scheduler.set_timesteps(num_steps)
            timesteps = getattr(scheduler, "timesteps", None)
            if timesteps is None:
                return None
            return tuple(float(t.item() if hasattr(t, "item") else t) for t in timesteps)
        except Exception:
            return None


def _extract_prompt(req: OmniDiffusionRequest) -> tuple[str, str | None]:
    if len(req.prompts) != 1:
        raise ValueError("runtime_v2 currently supports exactly one prompt per request")
    prompt_data = req.prompts[0]
    if isinstance(prompt_data, str):
        return prompt_data, None
    return str(prompt_data.get("prompt", "")), cast(str | None, prompt_data.get("negative_prompt"))


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
    """A picklable shallow copy of the scheduler with every on-device tensor
    nulled.

    Those tensors (UniPC ``model_outputs`` solver history, ``last_sample``,
    ``sigmas``, ...) migrate via P2P -- see ``WanStateLayoutCodec
    .migration_extra_fields``. CPU tensors built in the scheduler ``__init__``
    (``betas``/``alphas``/...) are small and stay in the skeleton. Raises if a
    device tensor remains: pickling it would be a slow D2H on the migration
    critical path and would leak the src cuda device to the dst rank.
    """
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


def _default_guidance_scale_from_pipeline(pipeline: Any) -> float:
    forward = getattr(pipeline, "forward", None)
    if forward is None:
        return 4.0
    try:
        default = inspect.signature(forward).parameters["guidance_scale"].default
    except (KeyError, TypeError, ValueError):
        return 4.0
    if default is inspect.Parameter.empty:
        return 4.0
    if isinstance(default, (tuple, list)):
        default = default[0] if default else 4.0
    try:
        return float(default)
    except (TypeError, ValueError):
        return 4.0


class WanStateLayoutCodec(ArtifactLayoutCodec):
    codec_id = WAN_STATE_CODEC_ID

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
        batch = self._metadata_int(request_metadata, "num_outputs_per_prompt", 1)
        text_seq_len = self._metadata_int(request_metadata, "text_seq_len", 512)
        num_steps = self._metadata_int(request_metadata, "num_steps", 1)
        dtype = self._transformer_dtype()
        fields: list[TensorFieldLayout] = [
            replicated_tensor_field(
                "prompt_embeds",
                (batch, text_seq_len, self._text_hidden_size()),
                dtype,
            )
        ]
        if bool(request_metadata.get("do_true_cfg", False)):
            fields.append(
                replicated_tensor_field(
                    "negative_prompt_embeds",
                    (batch, text_seq_len, self._text_hidden_size()),
                    dtype,
                )
            )

        artifact_id = artifact.artifact_id
        has_latents = artifact_id == "state_latent" or artifact_id == "state_timestep" or artifact_id.startswith(
            "state_denoised"
        )
        if has_latents:
            latent_shape = self._latent_shape(request_metadata)
            # Wan transformer's _sp_plan splits hidden_states inside forward
            # (SequenceParallelSplitHook on blocks.0) and gathers at proj_out,
            # so the executor must receive a full latent on every rank. Storing
            # the latent replicated lets materialize/shard_output short-circuit
            # at task boundaries; the alternative (sharded storage) would force
            # an all_gather every task only to immediately re-split inside the
            # model on a different axis. RESHARD still works on replicated
            # sources via select_source_fields_for_dst.
            fields.append(
                replicated_tensor_field("latents", latent_shape, torch.float32)
            )
            if bool(request_metadata.get("has_image", False)) and bool(getattr(self.pipeline, "expand_timesteps", False)):
                fields.extend(
                    (
                        replicated_tensor_field(
                            "latent_condition", latent_shape, torch.float32
                        ),
                        replicated_tensor_field(
                            "first_frame_mask",
                            (1, 1, *latent_shape[2:]),
                            torch.float32,
                        ),
                    )
                )

        has_timesteps = artifact_id == "state_timestep" or artifact_id.startswith("state_denoised")
        if has_timesteps:
            # Wan2.2 ships with a UniPC-style scheduler whose `set_timesteps`
            # produces an int64 (torch.long) timestep tensor (integer step
            # indices in [0, num_train_timesteps)). Declaring float32 here used
            # to cause the dit->vae RESHARD to abort with a dtype-mismatch.
            fields.append(replicated_tensor_field("timesteps", (num_steps,), torch.int64))

        return output_artifact_layout(artifact, tuple(fields))

    def pack_metadata(self, *, value: Any, layout: OutputArtifactLayout) -> bytes:
        # Build a small, picklable skeleton. Tensors never travel through this
        # pickle:
        #   * declared tensor fields    -> migrated via P2P (describe_output)
        #   * scheduler device tensors  -> migrated via P2P (migration_extra_fields);
        #     the skeleton keeps the scheduler object with those tensors nulled
        #   * device                    -> dst rebinds from its own cuda device
        #   * generator                 -> dst reseeds from sampling_params.seed
        # Pickling an on-device tensor here would be a 40+ MB D2H + serialize on
        # the migration critical path and would leak the src device, so codecs
        # must route device tensors through migration, not metadata.
        self._assert_no_undeclared_tensor_fields(value, layout.tensors)
        replacements: dict[str, Any] = {f.field_path[0]: None for f in layout.tensors}
        replacements["device"] = None
        replacements["scheduler"] = _scheduler_skeleton(getattr(value, "scheduler", None))
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
        # The diffusers scheduler keeps solver-history tensors on the GPU
        # (UniPC model_outputs, last_sample, sigmas, timesteps, ...). They are
        # not in describe_output -- their count/shape is runtime denoising
        # state, not request metadata -- so enumerate them from the live value
        # on the src leader. Each becomes a replicated P2P-migrated field, so a
        # ~40 MB solver history rides NCCL/GFC instead of a D2H + pickle.
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
                        fields.append(
                            _sched_field(("scheduler", name, str(idx)), item)
                        )
        return tuple(f for f in fields if f.field_path not in declared)

    def assemble(
        self,
        *,
        metadata_bytes: bytes,
        tensors: Mapping[tuple[str, ...], Any],
        layout: OutputArtifactLayout,
        device: Any = None,
    ) -> Any:
        # WanRuntimeState.device was stripped to None at pack time. Dst rank
        # MUST rebind to its own device; otherwise downstream tasks like
        # dit_prepare allocate fresh tensors on the wrong GPU.
        if device is None:
            raise ValueError(
                "WanStateLayoutCodec.assemble requires a device kwarg so the "
                "dst rank can rebind state.device after migration"
            )
        state = super().assemble(
            metadata_bytes=metadata_bytes, tensors=tensors, layout=layout, device=device,
        )
        # super().assemble injected the describe_output fields. Tensors from
        # migration_extra_fields (scheduler solver history) are also in
        # `tensors` but not in `layout.tensors`; write them back into the
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
                f"wan state codec cannot describe artifact {artifact.artifact_id} "
                f"with codec_id={artifact.codec_id!r}"
            )

    @staticmethod
    def _metadata_int(metadata: Mapping[str, Any], key: str, default: int) -> int:
        value = int(metadata.get(key, default))
        if value < 1:
            raise ValueError(f"{key} must be >= 1, got {value}")
        return value

    @staticmethod
    def _config_value(config: Any, name: str, default: Any | None = None) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    def _transformer_dtype(self) -> torch.dtype:
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is not None and getattr(transformer, "dtype", None) is not None:
            return transformer.dtype
        transformer_2 = getattr(self.pipeline, "transformer_2", None)
        if transformer_2 is not None and getattr(transformer_2, "dtype", None) is not None:
            return transformer_2.dtype
        return getattr(getattr(self.pipeline, "text_encoder", None), "dtype", torch.float32)

    def _text_hidden_size(self) -> int:
        candidates = (
            getattr(getattr(self.pipeline, "text_encoder", None), "config", None),
            getattr(self.pipeline, "text_encoder", None),
            getattr(self.pipeline, "transformer_config", None),
        )
        for obj in candidates:
            if obj is None:
                continue
            for attr in ("hidden_size", "d_model", "text_hidden_size", "cross_attention_dim"):
                value = self._config_value(obj, attr)
                if value is not None:
                    return int(value)
        raise ValueError("cannot infer wan text hidden size")

    def _normalized_size(self, request_metadata: Mapping[str, Any]) -> tuple[int, int, int]:
        height = self._metadata_int(request_metadata, "height", 480)
        width = self._metadata_int(request_metadata, "width", 832)
        num_frames = self._metadata_int(request_metadata, "num_frames", 81)
        patch_size = self._config_value(getattr(self.pipeline, "transformer_config", None), "patch_size", (1, 2, 2))
        mod_value = int(self.pipeline.vae_scale_factor_spatial) * int(patch_size[1])
        height = max(mod_value, (height // mod_value) * mod_value)
        width = max(mod_value, (width // mod_value) * mod_value)
        temporal = int(self.pipeline.vae_scale_factor_temporal)
        if num_frames % temporal != 1:
            num_frames = num_frames // temporal * temporal + 1
        return height, width, max(num_frames, 1)

    def _latent_shape(self, request_metadata: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
        batch = self._metadata_int(request_metadata, "num_outputs_per_prompt", 1)
        height, width, num_frames = self._normalized_size(request_metadata)
        config = getattr(self.pipeline, "transformer_config", None)
        if bool(request_metadata.get("has_image", False)) and bool(getattr(self.pipeline, "expand_timesteps", False)):
            channels = int(self._config_value(config, "out_channels"))
        else:
            channels = int(self._config_value(config, "in_channels"))
        latent_frames = (num_frames - 1) // int(self.pipeline.vae_scale_factor_temporal) + 1
        latent_height = height // int(self.pipeline.vae_scale_factor_spatial)
        latent_width = width // int(self.pipeline.vae_scale_factor_spatial)
        return (batch, channels, latent_frames, latent_height, latent_width)


class WanDecodedLayoutCodec(ArtifactLayoutCodec):
    codec_id = WAN_DECODED_CODEC_ID

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self._state_codec = WanStateLayoutCodec(pipeline)

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
                f"wan decoded codec cannot describe artifact {artifact.artifact_id} "
                f"with codec_id={artifact.codec_id!r}"
            )
        if request_metadata.get("output_type") == "latent":
            tensors = (replicated_tensor_field("value", self._state_codec._latent_shape(request_metadata), torch.float32),)
        else:
            batch = WanStateLayoutCodec._metadata_int(request_metadata, "num_outputs_per_prompt", 1)
            height, width, num_frames = self._state_codec._normalized_size(request_metadata)
            dtype = getattr(self.pipeline.vae, "dtype", torch.float32)
            tensors = (replicated_tensor_field("value", (batch, 3, num_frames, height, width), dtype),)
        return output_artifact_layout(artifact, tensors)


class _BaseWanExecutor(WorkerExecutor):
    def __init__(self, pipeline: Any, output_codecs: Mapping[str, ArtifactLayoutCodec] | None = None) -> None:
        self.pipeline = pipeline
        self._output_codecs = dict(output_codecs or {})
        self._default_guidance_scale = _default_guidance_scale_from_pipeline(pipeline)

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


class WanTextEncodeExecutor(_BaseWanExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        req = cast(OmniDiffusionRequest, self._single_input(task, resolved_inputs))
        prompt, negative_prompt = _extract_prompt(req)

        height = int(req.sampling_params.height or 480)
        width = int(req.sampling_params.width or 832)
        frame_num = int(req.sampling_params.num_frames or 81)
        num_steps = int(req.sampling_params.num_inference_steps or 40)
        output_type = req.sampling_params.output_type or "np"

        patch_size = self.pipeline.transformer_config.patch_size
        mod_value = self.pipeline.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        guidance_scale = (
            req.sampling_params.guidance_scale
            if req.sampling_params.guidance_scale_provided
            else self._default_guidance_scale
        )
        guidance_low = float(guidance_scale)
        guidance_high = (
            float(req.sampling_params.guidance_scale_2)
            if req.sampling_params.guidance_scale_2 is not None
            else guidance_low
        )

        boundary_ratio = self.pipeline.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = req.sampling_params.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = 0.875

        self.pipeline.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale_2=guidance_high,
            boundary_ratio=boundary_ratio,
        )

        if frame_num % self.pipeline.vae_scale_factor_temporal != 1:
            frame_num = frame_num // self.pipeline.vae_scale_factor_temporal * self.pipeline.vae_scale_factor_temporal + 1
        frame_num = max(frame_num, 1)

        if self.pipeline.transformer is not None:
            dtype = self.pipeline.transformer.dtype
        elif self.pipeline.transformer_2 is not None:
            dtype = self.pipeline.transformer_2.dtype
        else:
            dtype = self.pipeline.text_encoder.dtype

        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(req.sampling_params.seed)
        req.sampling_params.generator = generator

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_low > 1.0 or guidance_high > 1.0,
            num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
            max_sequence_length=req.sampling_params.max_sequence_length or 512,
            device=self.pipeline.device,
            dtype=dtype,
        )

        state = WanRuntimeState(
            request=req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=frame_num,
            num_steps=num_steps,
            output_type=output_type,
            device=self.pipeline.device,
            dtype=dtype,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=float(boundary_ratio) * self.pipeline.scheduler.config.num_train_timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        return (ArtifactValue(handle=task.outputs[0], value=state),)


class WanLatentPrepareExecutor(_BaseWanExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        state = cast(WanRuntimeState, self._single_input(task, resolved_inputs))
        req = state.request

        # When this task runs on the dit group after a RESHARD from aux, the
        # pickle-stripped state has request.sampling_params.generator=None
        # (torch.Generator is not safe to round-trip across the migration
        # boundary). Reconstruct it on this rank's device from `seed` so the
        # initial latent noise matches the single-group baseline. Without this,
        # prepare_latents falls back to the global RNG and disagg/baseline
        # outputs diverge completely.
        if req.sampling_params.generator is None and req.sampling_params.seed is not None:
            req.sampling_params.generator = torch.Generator(device=self.pipeline.device).manual_seed(
                int(req.sampling_params.seed)
            )

        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None

        image: PIL.Image.Image | torch.Tensor | None
        if raw_image is None:
            image = None
        elif isinstance(raw_image, str):
            image = PIL.Image.open(raw_image)
        else:
            image = cast(PIL.Image.Image | torch.Tensor, raw_image)

        if self.pipeline.expand_timesteps and image is not None:
            video_processor = VideoProcessor(vae_scale_factor=self.pipeline.vae_scale_factor_spatial)
            if isinstance(image, PIL.Image.Image):
                image = image.resize((state.width, state.height), PIL.Image.Resampling.LANCZOS)
                image_tensor = video_processor.preprocess(image, height=state.height, width=state.width)
            else:
                image_tensor = image

            num_channels_latents = self.pipeline.transformer_config.out_channels
            batch_size = state.prompt_embeds.shape[0]
            state.latents = self.pipeline.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=state.height,
                width=state.width,
                num_frames=state.num_frames,
                dtype=torch.float32,
                device=state.device,
                generator=req.sampling_params.generator,
                latents=req.sampling_params.latents,
            )

            image_tensor = image_tensor.unsqueeze(2).to(device=state.device, dtype=self.pipeline.vae.dtype)
            latent_condition = retrieve_latents(self.pipeline.vae.encode(image_tensor), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            latents_mean = (
                torch.tensor(self.pipeline.vae.config.latents_mean)
                .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = (
                1.0
                / torch.tensor(self.pipeline.vae.config.latents_std)
                .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            state.latent_condition = (latent_condition - latents_mean) * latents_std
            state.latent_condition = state.latent_condition.to(torch.float32)

            first_frame_mask = torch.ones(
                1,
                1,
                state.latents.shape[2],
                state.latents.shape[3],
                state.latents.shape[4],
                dtype=torch.float32,
                device=state.device,
            )
            first_frame_mask[:, :, 0] = 0
            state.first_frame_mask = first_frame_mask
        else:
            num_channels_latents = self.pipeline.transformer_config.in_channels
            state.latents = self.pipeline.prepare_latents(
                batch_size=state.prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=state.height,
                width=state.width,
                num_frames=state.num_frames,
                dtype=torch.float32,
                device=state.device,
                generator=req.sampling_params.generator,
                latents=req.sampling_params.latents,
            )

        return (ArtifactValue(handle=task.outputs[0], value=state),)


class WanTimestepPrepareExecutor(_BaseWanExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        state = cast(WanRuntimeState, self._single_input(task, resolved_inputs))
        state.scheduler = copy.deepcopy(self.pipeline.scheduler)
        state.scheduler.set_timesteps(state.num_steps, device=state.device)
        state.timesteps = state.scheduler.timesteps
        self.pipeline._num_timesteps = len(state.timesteps)
        return (ArtifactValue(handle=task.outputs[0], value=state),)


class WanDenoiseExecutor(_BaseWanExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        if task.step_range is None:
            raise ValueError("dit_step_chunk requires step_range")

        state = cast(WanRuntimeState, self._single_input(task, resolved_inputs))
        if state.timesteps is None or state.latents is None or state.scheduler is None:
            raise RuntimeError("state is not ready for denoising")

        start = max(0, task.step_range.start)
        end = min(task.step_range.end, len(state.timesteps))
        attention_kwargs = state.attention_kwargs or {}

        for idx in range(start, end):
            t = state.timesteps[idx]
            self.pipeline._current_timestep = t

            if t < state.boundary_timestep:
                current_guidance_scale = state.guidance_high
                if self.pipeline.transformer_2 is not None:
                    current_model = self.pipeline.transformer_2
                elif self.pipeline.transformer is not None:
                    current_model = self.pipeline.transformer
                else:
                    raise RuntimeError("No transformer available for low-noise stage")
            else:
                current_guidance_scale = state.guidance_low
                if self.pipeline.transformer is not None:
                    current_model = self.pipeline.transformer
                elif self.pipeline.transformer_2 is not None:
                    current_model = self.pipeline.transformer_2
                else:
                    raise RuntimeError("No transformer available for high-noise stage")

            if self.pipeline.expand_timesteps and state.latent_condition is not None and state.first_frame_mask is not None:
                latent_model_input = (1 - state.first_frame_mask) * state.latent_condition + state.first_frame_mask * state.latents
                latent_model_input = latent_model_input.to(state.dtype)
                patch_size = self.pipeline.transformer_config.patch_size
                patch_height = state.latents.shape[3] // patch_size[1]
                patch_width = state.latents.shape[4] // patch_size[2]
                patch_mask = state.first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]
                temp_ts = (patch_mask[0][0] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(state.latents.shape[0], -1)
            else:
                latent_model_input = state.latents.to(state.dtype)
                timestep = t.expand(state.latents.shape[0])

            do_true_cfg = current_guidance_scale > 1.0 and state.negative_prompt_embeds is not None
            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": state.prompt_embeds,
                "attention_kwargs": attention_kwargs,
                "return_dict": False,
                "current_model": current_model,
            }
            negative_kwargs = None
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": state.negative_prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }

            noise_pred = self.pipeline.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=current_guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )
            state.latents = self._scheduler_step_maybe_with_cfg(
                state.scheduler,
                noise_pred,
                t,
                state.latents,
                do_true_cfg,
            )

        if end >= len(state.timesteps):
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()
            self.pipeline._current_timestep = None
            state.scheduler = None
            if self.pipeline.expand_timesteps and state.latent_condition is not None and state.first_frame_mask is not None:
                state.latents = (1 - state.first_frame_mask) * state.latent_condition + state.first_frame_mask * state.latents

        return (ArtifactValue(handle=task.outputs[0], value=state),)


class WanDecodeExecutor(_BaseWanExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        state = cast(WanRuntimeState, self._single_input(task, resolved_inputs))
        if state.latents is None:
            raise RuntimeError("state.latents is None in decode stage")

        if state.output_type == "latent":
            output = state.latents
        else:
            latents = state.latents.to(self.pipeline.vae.dtype)
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
            output = self.pipeline.vae.decode(latents, return_dict=False)[0]

        return (ArtifactValue(handle=task.outputs[0], value=WanDecodedValue(value=output)),)


class WanFinalizeExecutor(_BaseWanExecutor):
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        decoded = self._single_input(task, resolved_inputs)
        # WanDecodeExecutor wraps the raw tensor in WanDecodedValue so the
        # codec's field_path=('value',) navigates correctly during migration.
        output = decoded.value if isinstance(decoded, WanDecodedValue) else decoded
        return (ArtifactValue(handle=task.outputs[0], value=DiffusionOutput(output=output)),)


class WanRuntimeV2Adapter(RuntimeV2Adapter):
    model_class_name = "WanPipeline"

    @property
    def supported_task_kinds(self) -> tuple[TaskKind, ...]:
        return STANDARD_TASK_KINDS

    def normalize_request(self, request: Any, denoise_chunk_size: int) -> WanRuntimeRequest:
        if isinstance(request, WanRuntimeRequest):
            return request
        if not isinstance(request, OmniDiffusionRequest):
            raise TypeError(f"unsupported runtime_v2 request type: {type(request)!r}")
        return WanRuntimeRequest(diffusion_request=request, denoise_chunk_size=denoise_chunk_size)

    def build_task_compiler(
        self,
        default_denoise_chunk_size: int,
        *,
        od_config: Any = None,
        pipeline: Any = None,
    ) -> TaskCompiler:
        return WanTaskCompiler(
            default_denoise_chunk_size=default_denoise_chunk_size,
            latent_geometry=self._resolve_latent_geometry(od_config),
            default_guidance_scale=(
                _default_guidance_scale_from_pipeline(pipeline)
                if pipeline is not None
                else 4.0
            ),
            scheduler=getattr(pipeline, "scheduler", None) if pipeline is not None else None,
            boundary_ratio=getattr(pipeline, "boundary_ratio", None) if pipeline is not None else None,
            num_train_timesteps=self._resolve_num_train_timesteps(pipeline),
        )

    @staticmethod
    def _resolve_latent_geometry(od_config: Any) -> WanLatentGeometry:
        model = getattr(od_config, "model", None) if od_config is not None else None
        if not model:
            logger.warning(
                "runtime_v2 wan adapter: no model in od_config; falling back to "
                "default Wan2.2 latent geometry for cost-model keying"
            )
            return WanLatentGeometry.default()
        try:
            return WanLatentGeometry.from_model(str(model))
        except Exception as exc:  # config load is best-effort; warn and degrade
            logger.warning(
                "runtime_v2 wan adapter: could not resolve latent geometry from "
                "model config %r (%s); falling back to Wan2.2 default",
                model,
                exc,
            )
            return WanLatentGeometry.default()

    @staticmethod
    def _resolve_num_train_timesteps(pipeline: Any) -> int:
        scheduler = getattr(pipeline, "scheduler", None)
        config = getattr(scheduler, "config", None)
        return int(getattr(config, "num_train_timesteps", 1000) or 1000)

    def build_executors(self, pipeline: Any) -> dict[TaskKind, WorkerExecutor]:
        self.validate_pipeline(pipeline, getattr(pipeline, "od_config", None))
        state_codec = WanStateLayoutCodec(pipeline)
        decoded_codec = WanDecodedLayoutCodec(pipeline)
        return {
            TaskKind.TEXT_ENCODE: WanTextEncodeExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.DIT_PREPARE: WanLatentPrepareExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.TIMESTEP_PREPARE: WanTimestepPrepareExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.DIT_STEP_CHUNK: WanDenoiseExecutor(
                pipeline,
                {state_codec.codec_id: state_codec},
            ),
            TaskKind.VAE_DECODE: WanDecodeExecutor(
                pipeline,
                {decoded_codec.codec_id: decoded_codec},
            ),
            TaskKind.FINALIZE: WanFinalizeExecutor(pipeline),
        }

    def validate_pipeline(self, pipeline: Any, od_config: Any) -> None:
        if pipeline is None:
            raise ValueError("runtime_v2 wan adapter requires an initialized pipeline")
        for attr in ("encode_prompt", "prepare_latents", "check_inputs", "vae", "scheduler"):
            if not hasattr(pipeline, attr):
                raise ValueError(f"wan runtime_v2 adapter requires pipeline.{attr}")
