# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ExecutionGroupSpec,
    OutputArtifactLayout,
    TensorFieldLayout,
)


class ArtifactLayoutCodec(ABC):
    @property
    @abstractmethod
    def codec_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def describe_output(
        self,
        *,
        group: ExecutionGroupSpec,
        group_rank: int,
        request_metadata: Mapping[str, Any],
        artifact: ArtifactHandle,
    ) -> OutputArtifactLayout:
        raise NotImplementedError

    # ---- artifact migration data plane helpers --------------------------------
    #
    # The migration EXECUTE_TRANSFER phase splits an artifact value into:
    #   1) tensor fields (described by `OutputArtifactLayout.tensors`), which
    #      travel via NCCL P2P send/recv between src/dst ranks.
    #   2) non-tensor metadata, which travels through the multiprocessing
    #      control pipe as pickled bytes.
    #
    # Subclasses that need to drop transient state (e.g. a non-picklable
    # diffusers scheduler held inside the value) should override
    # `pack_metadata` to return a sanitized skeleton. The default impl handles
    # dataclass and dict values whose tensor field paths are single-level
    # attribute / key names.
    def pack_metadata(self, *, value: Any, layout: OutputArtifactLayout) -> bytes:
        # Loud guard: any non-None Tensor on the value object that is NOT in
        # layout.tensors would otherwise be pickle-serialized into the skeleton
        # bytes, dragging its src-rank cuda device along to the dst rank. This
        # is the kind of bug that surfaces as "Expected all tensors to be on
        # the same device" deep inside the model. Force codec authors to either
        # add the field to describe_output (so it migrates via NCCL) or null it
        # before pack_metadata.
        self._assert_no_undeclared_tensor_fields(value, layout.tensors)
        skeleton = self._strip_tensor_fields(value, layout.tensors)
        return pickle.dumps(skeleton, protocol=pickle.HIGHEST_PROTOCOL)

    def migration_extra_fields(
        self, *, value: Any, layout: OutputArtifactLayout
    ) -> tuple[TensorFieldLayout, ...]:
        """On-device tensors reachable from a live artifact value that
        ``describe_output`` did not declare.

        The base codec declares every migratable tensor in describe_output, so
        there are none. Codecs that keep device tensors inside nested objects
        (e.g. a diffusers scheduler's solver history) override this so those
        tensors migrate via P2P from the live src value rather than being
        D2H-copied and pickled into the metadata skeleton. Called by the
        migration controller on the src leader, which has the live value.
        """
        return ()

    @staticmethod
    def _assert_no_undeclared_tensor_fields(
        value: Any, fields: tuple[TensorFieldLayout, ...]
    ) -> None:
        # Lazy import torch to keep this module import cheap on CPU-only paths.
        try:
            import torch as _torch
        except ImportError:  # pragma: no cover - torch is a hard dep elsewhere
            return
        declared_paths = {tuple(f.field_path) for f in fields}
        for attr_name, attr_value in ArtifactLayoutCodec._iter_value_attrs(value):
            if not isinstance(attr_value, _torch.Tensor):
                continue
            if (attr_name,) in declared_paths:
                continue
            raise RuntimeError(
                f"codec pack_metadata: attribute {attr_name!r} on artifact value is a "
                f"torch.Tensor (device={attr_value.device}, dtype={attr_value.dtype}, shape={tuple(attr_value.shape)}) "
                "but the codec layout did not declare it. Pickling this would leak the "
                "src-rank device to the dst rank. Either include this field in "
                "describe_output so it migrates via NCCL, or null it on the src side "
                "before pack_metadata is called."
            )

    @staticmethod
    def _iter_value_attrs(value: Any):
        """Yield (name, value) for top-level attributes of a dataclass or dict.

        The default implementation only checks single-level attributes/keys; if
        a codec stores tensors deeper than one level the codec must override
        pack_metadata.
        """
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                yield field.name, getattr(value, field.name)
            return
        if isinstance(value, dict):
            for key, val in value.items():
                yield key, val

    def assemble(
        self,
        *,
        metadata_bytes: bytes,
        tensors: Mapping[tuple[str, ...], Any],
        layout: OutputArtifactLayout,
        device: Any = None,
    ) -> Any:
        """Rebuild the artifact value on a dst rank.

        `device` is the dst rank's CUDA device. When the skeleton carries a
        `torch.device`-typed field (e.g. WanRuntimeState.device), subclasses
        should rebind that field to `device` so downstream tasks that allocate
        fresh tensors via `torch.randn(..., device=state.device)` land on the
        right GPU. The default implementation only injects tensor fields and
        ignores `device`.
        """
        skeleton = pickle.loads(metadata_bytes)
        return self._inject_tensor_fields(skeleton, tensors, layout.tensors)

    def extract_tensors(
        self,
        *,
        value: Any,
        layout: OutputArtifactLayout,
    ) -> dict[tuple[str, ...], Any]:
        out: dict[tuple[str, ...], Any] = {}
        for field in layout.tensors:
            out[field.field_path] = self._read_field(value, field.field_path)
        return out

    @staticmethod
    def _strip_tensor_fields(value: Any, fields: tuple[TensorFieldLayout, ...]) -> Any:
        # Return a shallow copy of `value` with each tensor field path set to None
        # so the skeleton is picklable independent of CUDA tensor handling.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            skeleton = copy.copy(value)
            for field in fields:
                ArtifactLayoutCodec._write_field(skeleton, field.field_path, None)
            return skeleton
        if isinstance(value, dict):
            skeleton = dict(value)
            for field in fields:
                ArtifactLayoutCodec._write_field(skeleton, field.field_path, None)
            return skeleton
        # For arbitrary objects, fall back to deepcopy and clear attributes.
        skeleton = copy.copy(value)
        for field in fields:
            ArtifactLayoutCodec._write_field(skeleton, field.field_path, None)
        return skeleton

    @staticmethod
    def _inject_tensor_fields(
        skeleton: Any,
        tensors: Mapping[tuple[str, ...], Any],
        fields: tuple[TensorFieldLayout, ...],
    ) -> Any:
        for field in fields:
            tensor = tensors.get(field.field_path)
            if tensor is None:
                raise KeyError(
                    f"missing tensor for field_path={field.field_path} during artifact assembly"
                )
            ArtifactLayoutCodec._write_field(skeleton, field.field_path, tensor)
        return skeleton

    @staticmethod
    def _read_field(value: Any, field_path: tuple[str, ...]) -> Any:
        cursor: Any = value
        for part in field_path:
            if isinstance(cursor, dict):
                cursor = cursor[part]
            elif isinstance(cursor, (list, tuple)):
                cursor = cursor[int(part)]
            else:
                cursor = getattr(cursor, part)
        return cursor

    @staticmethod
    def _write_field(value: Any, field_path: tuple[str, ...], new_value: Any) -> None:
        if not field_path:
            raise ValueError("field_path must be non-empty")
        cursor: Any = value
        for part in field_path[:-1]:
            if isinstance(cursor, dict):
                cursor = cursor[part]
            elif isinstance(cursor, (list, tuple)):
                cursor = cursor[int(part)]
            else:
                cursor = getattr(cursor, part)
        leaf = field_path[-1]
        if isinstance(cursor, dict):
            cursor[leaf] = new_value
        elif isinstance(cursor, list):
            cursor[int(leaf)] = new_value
        else:
            setattr(cursor, leaf, new_value)


class ArtifactLayoutCodecRegistry:
    def __init__(self) -> None:
        self._codecs: dict[str, ArtifactLayoutCodec] = {}

    def register(self, codec: ArtifactLayoutCodec) -> None:
        if not codec.codec_id:
            raise ValueError("artifact layout codec requires codec_id")
        existing = self._codecs.get(codec.codec_id)
        if existing is not None:
            if existing is codec:
                return
            raise ValueError(f"duplicate artifact layout codec: {codec.codec_id}")
        self._codecs[codec.codec_id] = codec

    def get(self, codec_id: str) -> ArtifactLayoutCodec:
        try:
            return self._codecs[codec_id]
        except KeyError:
            raise KeyError(f"unknown artifact layout codec: {codec_id}") from None

    def describe_output(
        self,
        *,
        group: ExecutionGroupSpec,
        group_rank: int,
        request_metadata: Mapping[str, Any],
        artifact: ArtifactHandle,
    ) -> OutputArtifactLayout:
        if not artifact.codec_id:
            raise ValueError(f"artifact {artifact.artifact_id} has no codec_id")
        return self.get(artifact.codec_id).describe_output(
            group=group,
            group_rank=group_rank,
            request_metadata=request_metadata,
            artifact=artifact,
        )
