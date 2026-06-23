# SPDX-License-Identifier: Apache-2.0
"""Simulator-side scheduling policies (one class per submodule).

The shim at ``benchmarks/diffusion/runtime_v2_mixed_priority_policies.py``
re-exports these so existing CLI policy specs (`module:make_X`) keep working.
"""

from benchmarks.diffusion.policies.clean_edf import (
    CleanEdfPreemptivePolicy,
    make_clean_edf_demote,
    make_clean_edf_pause,
)
from benchmarks.diffusion.policies.common import (
    _SingleGroupPolicyBase,
    _groups_with_sp,
    _request_class_from_value,
    _request_deadline_ms,
    _request_priority,
)
from benchmarks.diffusion.policies.edf_best_fit import (
    EdfBestFitPolicy,
    _EdfBestFitRequestMeta,
    make_edf_best_fit,
)
from benchmarks.diffusion.policies.edf_greedy import (
    EdfGreedyPolicy,
    _EdfGreedyRequestMeta,
    make_edf_greedy,
)
from benchmarks.diffusion.policies.load_adaptive import (
    LoadAdaptivePolicy,
    PriorityLoadAdaptivePolicy,
    make_load_adaptive,
    make_priority_load_adaptive,
)
from benchmarks.diffusion.policies.primal_dual_rolling import (
    PrimalDualRollingPlannerPolicy,
    make_pd_rolling,
    make_primal_dual_rolling_planner,
)
from benchmarks.diffusion.policies.rssp import RSSPPolicy, make_rssp
from benchmarks.diffusion.policies.static_sp import (
    StaticSPPolicy,
    make_fcfs,
    make_sjf,
    make_srtf,
    make_static_sp1,
    make_static_sp2,
    make_static_sp4,
    make_static_sp8,
)
from benchmarks.diffusion.policies.step_preempt import (
    StepPreemptiveSingleGroupPolicy,
    make_step_preempt_demote,
    make_step_preempt_pause,
)
from benchmarks.diffusion.policies.tetriserve import (
    TetriServeContinuousPolicy,
    TetriServeRoundPolicy,
    make_tetriserve_continuous,
    make_tetriserve_round,
)

__all__ = [
    # base
    "_SingleGroupPolicyBase",
    "_groups_with_sp",
    "_request_class_from_value",
    "_request_deadline_ms",
    "_request_priority",
    # policies
    "StaticSPPolicy", "RSSPPolicy",
    "LoadAdaptivePolicy", "PriorityLoadAdaptivePolicy",
    "StepPreemptiveSingleGroupPolicy",
    "EdfGreedyPolicy", "_EdfGreedyRequestMeta",
    "EdfBestFitPolicy", "_EdfBestFitRequestMeta",
    "CleanEdfPreemptivePolicy",
    "PrimalDualRollingPlannerPolicy",
    "TetriServeContinuousPolicy", "TetriServeRoundPolicy",
    # factories
    "make_static_sp1", "make_static_sp2", "make_static_sp4", "make_static_sp8",
    "make_fcfs", "make_srtf", "make_sjf",
    "make_rssp",
    "make_load_adaptive", "make_priority_load_adaptive",
    "make_step_preempt_pause", "make_step_preempt_demote",
    "make_primal_dual_rolling_planner", "make_pd_rolling",
    "make_edf_greedy",
    "make_edf_best_fit",
    "make_clean_edf_pause", "make_clean_edf_demote",
    "make_tetriserve_continuous", "make_tetriserve_round",
]
