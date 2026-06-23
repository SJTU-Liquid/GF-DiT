# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
    ):
        super().__init__()
        self.attn_backend = get_attn_backend(-1)
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )
        # Instantiate fallback backend for float32 support
        self.sdpa_fallback = SDPABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )
        self.backend_pref = None

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.causal = causal

        self.ring_runner = None
        self._ring_runner_cache_key = None
        self._parallel_strategy_cache_key = None
        self._parallel_strategy_cache = None

        try:
            config = get_forward_context().omni_diffusion_config
            self.backend_pref = config.attention_backend
        except Exception:
            pass
        # Fallback strategy when SP is not active (outside sharded regions)
        self._no_parallel_strategy = NoParallelAttention()

    def _get_active_parallel_strategy(self):
        """Get the parallel strategy for the currently active runtime group.

        Returns NoParallelAttention if we're outside an SP sharded region
        (e.g., in noise_refiner/context_refiner before unified_prepare in Z-Image).
        runtime_v2 may otherwise switch this worker from an SP4 DiT task to an
        SP2 DiT task from another request without a RESHARD on this worker, so
        the strategy type must be selected from the live group/config per call.
        """
        if is_forward_context_available():
            ctx = get_forward_context()
            if not ctx.sp_active:
                return self._no_parallel_strategy

        key = self._parallel_strategy_key()
        if key != self._parallel_strategy_cache_key or self._parallel_strategy_cache is None:
            self._parallel_strategy_cache = build_parallel_attention_strategy(
                scatter_idx=self.scatter_idx,
                gather_idx=self.gather_idx,
                use_sync=self.use_sync,
            )
            self._parallel_strategy_cache_key = key
        return self._parallel_strategy_cache

    def _parallel_strategy_key(self):
        try:
            config = get_forward_context().omni_diffusion_config
            parallel_config = config.parallel_config
            sp_group = get_sp_group()
            return (
                int(getattr(parallel_config, "sequence_parallel_size", 1) or 1),
                int(getattr(parallel_config, "ulysses_degree", 1) or 1),
                int(getattr(parallel_config, "ring_degree", 1) or 1),
                int(getattr(sp_group, "world_size", 1) or 1),
                int(getattr(sp_group, "ulysses_world_size", 1) or 1),
                int(getattr(sp_group, "ring_world_size", 1) or 1),
            )
        except Exception:
            return None

    def _get_active_ring_runner(self):
        try:
            config = get_forward_context().omni_diffusion_config
            if int(getattr(config.parallel_config, "ring_degree", 1) or 1) <= 1:
                return None
            sp_group = get_sp_group()
            if int(getattr(sp_group, "ring_world_size", 1) or 1) <= 1:
                return None
            key = (
                int(getattr(config.parallel_config, "ring_degree", 1) or 1),
                int(getattr(sp_group, "world_size", 1) or 1),
                int(getattr(sp_group, "ulysses_world_size", 1) or 1),
                int(getattr(sp_group, "ring_world_size", 1) or 1),
            )
            if key != self._ring_runner_cache_key or self.ring_runner is None:
                self.ring_runner = RingParallelAttention(sp_group)
                self._ring_runner_cache_key = key
            return self.ring_runner
        except Exception:
            return None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # Get the appropriate parallel strategy based on SP active state
        strategy = self._get_active_parallel_strategy()

        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

        # 2. Kernel Execution (Computation)
        ring_runner = self._get_active_ring_runner()
        if ring_runner is not None and strategy is not self._no_parallel_strategy:
            out = self._run_ring_attention(ring_runner, query, key, value, attn_metadata)
        else:
            out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = strategy.post_attention(out, ctx)

        return out

    def _run_local_attention(self, query, key, value, attn_metadata):
        if query.dtype == torch.float32:
            logger.warning_once(
                f"Only SDPA supports float32. Overriding user config {type(self.attention)} "
                f"attention_backend='{self.backend_pref}' to 'sdpa' for dtype={query.dtype}."
            )
            return self.sdpa_fallback.forward(query, key, value, attn_metadata)

        # Fallback to standard attention
        return self.attention.forward(query, key, value, attn_metadata)

    def _run_ring_attention(self, ring_runner, query, key, value, attn_metadata):
        return ring_runner.run_attention(
            query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
        )
