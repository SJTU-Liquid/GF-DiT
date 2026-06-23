# SPDX-License-Identifier: Apache-2.0
"""Cost-model store for runtime_v2 scheduling.

Loads per-(tp, sp, ulysses, ring, cfg) cost-model JSON files produced by
``benchmarks/diffusion/runtime_v2_stage_profiler.py`` (or the synthetic
generator). Each file declares stage coefficients ``a*x^2 + b*x + c``; the
scheduler asks ``estimate_ms(stage, x)`` to predict per-task wallclock.

The simulator at ``benchmarks/diffusion/runtime_v2_policy_simulator.py`` uses
the same store via a back-compat re-export so simulator and production share
exactly one set of cost-model semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.diffusion.runtime_v2.protocol import ExecutionGroupSpec, ParallelSpec


# Stages with no parallel implementation: they execute on a single rank
# regardless of the DiT SP degree, so they are profiled only in the SP1 cost
# model. A parallel (sp>1) cost model legitimately omits them, and a static
# SP-N policy (e.g. SRTF/FCFS pinned to one big group) that runs them should
# price them at their SP1 cost rather than fail.
_NON_PARALLEL_STAGES = frozenset(
    {"text_encode", "dit_prepare", "timestep_prepare", "vae_decode", "finalize"}
)


@dataclass(frozen=True)
class CostModelKey:
    tp: int
    sp: int
    ulysses: int
    ring: int
    cfg: int

    @classmethod
    def from_cost_model(cls, data: Mapping[str, Any]) -> "CostModelKey":
        parallelism = data.get("parallelism", {})
        return cls(
            tp=int(parallelism.get("tp", 1)),
            sp=int(parallelism.get("sp", 1)),
            ulysses=int(parallelism.get("ulysses", parallelism.get("sp", 1))),
            ring=int(parallelism.get("ring", 1)),
            cfg=int(parallelism.get("cfg", 1)),
        )

    @classmethod
    def from_group(cls, group: ExecutionGroupSpec) -> "CostModelKey":
        return cls(
            tp=int(group.parallel_spec.tp),
            sp=int(group.parallel_spec.sp),
            ulysses=int(group.ulysses_degree),
            ring=int(group.ring_degree),
            cfg=int(group.parallel_spec.cfg),
        )

    @classmethod
    def from_parallel_spec(cls, spec: ParallelSpec, *, ulysses: int | None = None, ring: int = 1) -> "CostModelKey":
        """Build a key for a synthesized SP-N group with no explicit topology entry.

        Used by EDF best-fit at dispatch time: it has a ParallelSpec(tp=1, sp=k,
        cfg=1) in hand but the synthesized ExecutionGroupSpec may not exist yet.
        Defaults ``ulysses = spec.sp`` and ``ring = 1`` to mirror the runtime_v2
        EDF runners which always emit ulysses-only groups.
        """
        return cls(
            tp=int(spec.tp),
            sp=int(spec.sp),
            ulysses=int(ulysses if ulysses is not None else spec.sp),
            ring=int(ring),
            cfg=int(spec.cfg),
        )


@dataclass
class StageCostModel:
    source_path: Path
    key: CostModelKey
    stages: dict[str, dict[str, Any]]
    # SP1 sibling for the same (tp, cfg, ring). Non-parallel aux stages absent
    # from a parallel model are priced here. None for the SP1 model itself.
    sp1_fallback: "StageCostModel | None" = None

    def stage_x_name(self, stage: str) -> str:
        stages, resolved, _scale = self._resolve_stage(stage)
        return str(stages[resolved].get("x_name", "seq_len"))

    def estimate_ms(self, stage: str, x: float) -> float:
        stages, resolved, scale = self._resolve_stage(stage)
        coeffs = stages[resolved]["coeffs"]
        a = float(coeffs.get("a", 0.0))
        b = float(coeffs.get("b", 0.0))
        c = float(coeffs.get("c", 0.0))
        return max(0.0, (a * x * x + b * x + c) * scale)

    def _resolve_stage(
        self, stage: str
    ) -> tuple[dict[str, dict[str, Any]], str, float]:
        stage = str(stage)
        if stage in self.stages:
            return self.stages, stage, 1.0
        if stage == "dit_step_chunk_cfg":
            if "dit_step_chunk" in self.stages:
                return self.stages, "dit_step_chunk", 1.0
        elif stage == "dit_step_chunk_nocfg":
            if "dit_step_chunk_cfg" in self.stages:
                return self.stages, "dit_step_chunk_cfg", 0.5
            if "dit_step_chunk" in self.stages:
                return self.stages, "dit_step_chunk", 0.5
        # Non-parallel aux stages live only in the SP1 model; a parallel model
        # prices them at their (parallelism-invariant) SP1 cost.
        if stage in _NON_PARALLEL_STAGES and self.sp1_fallback is not None:
            return self.sp1_fallback._resolve_stage(stage)
        raise KeyError(f"stage {stage!r} not found in cost model {self.source_path}")


@dataclass
class CostModelStore:
    models: dict[CostModelKey, StageCostModel]

    @classmethod
    def load_dir(cls, path: str | Path) -> "CostModelStore":
        root = Path(path)
        models: dict[CostModelKey, StageCostModel] = {}
        for cost_path in sorted(root.glob("*.json")):
            data = json.loads(cost_path.read_text())
            key = CostModelKey.from_cost_model(data)
            if key in models:
                raise ValueError(
                    f"duplicate cost model for {key}: {models[key].source_path} and {cost_path}"
                )
            models[key] = StageCostModel(
                source_path=cost_path,
                key=key,
                stages=dict(data["stages"]),
            )
        if not models:
            raise ValueError(f"no cost-model JSON files found under {root}")
        # Wire each parallel model to its SP1 sibling so non-parallel aux stages
        # (text_encode, vae_decode, ...) that are only profiled at SP1 resolve
        # for static SP-N policies that run them on a big group.
        for key, model in models.items():
            if key.sp == 1 and key.ulysses == 1:
                continue
            sp1 = models.get(
                CostModelKey(tp=key.tp, sp=1, ulysses=1, ring=key.ring, cfg=key.cfg)
            )
            if sp1 is not None:
                model.sp1_fallback = sp1
        return cls(models=models)

    def for_group(self, group: ExecutionGroupSpec) -> StageCostModel:
        key = CostModelKey.from_group(group)
        return self._lookup(key, label=f"group {group.group_id!r}")

    def for_parallel_spec(self, spec: ParallelSpec, *, ulysses: int | None = None, ring: int = 1) -> StageCostModel:
        key = CostModelKey.from_parallel_spec(spec, ulysses=ulysses, ring=ring)
        return self._lookup(key, label=f"parallel_spec={spec}")

    def _lookup(self, key: CostModelKey, *, label: str) -> StageCostModel:
        try:
            return self.models[key]
        except KeyError as exc:
            known = ", ".join(str(k) for k in sorted(self.models, key=lambda item: (item.tp, item.sp, item.cfg)))
            raise KeyError(f"no cost model for {label} key={key}; known keys: {known}") from exc

    def available_sp_sizes(self, *, tp: int = 1, cfg: int = 1, ring: int = 1) -> tuple[int, ...]:
        """SP sizes covered by this store for the given (tp, cfg, ring) — useful
        for the runner to clamp allowed_sp_sizes to what the cost-model supports.
        """
        return tuple(
            sorted({k.sp for k in self.models if k.tp == tp and k.cfg == cfg and k.ring == ring})
        )


__all__ = [
    "CostModelKey",
    "StageCostModel",
    "CostModelStore",
]
