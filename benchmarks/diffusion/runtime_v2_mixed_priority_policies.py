# SPDX-License-Identifier: Apache-2.0
"""Back-compat shim: original module re-exports each policy from
``benchmarks/diffusion/policies/`` so CLI invocations using the path
``benchmarks.diffusion.runtime_v2_mixed_priority_policies:make_X`` keep
working without modification.
"""

from benchmarks.diffusion.policies import *  # noqa: F401, F403
from benchmarks.diffusion.policies import (  # noqa: F401
    _SingleGroupPolicyBase,
    _EdfBestFitRequestMeta,
    _EdfGreedyRequestMeta,
    _groups_with_sp,
    _request_class_from_value,
    _request_deadline_ms,
    _request_priority,
)
