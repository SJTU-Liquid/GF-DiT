"""Policy variants used only to reproduce figures exactly.

``make_srtf_legacy_tiebreak`` restores the SRTF tie-break used when the paper's
Fig. 6 / Fig. 10 SRTF numbers were produced (May-June 2026): requests with equal
estimated remaining work are ordered by ``request_id`` alone. The released policy
(``vllm_omni/diffusion/runtime_v2/policies/srtf.py``) instead orders them by
arrival, which removes a starvation tail. Real serving assigns random request
ids, so the legacy rule is a random tie-break; the simulator reproduces it by
running on workloads whose ids are md5-randomized (see ``run_claims.py``).
"""
from vllm_omni.diffusion.runtime_v2.policies.srtf import SRTFSchedulerPolicy


class _LegacyTieBreakSRTF(SRTFSchedulerPolicy):
    def on_request_submitted(self, plan):
        out = super().on_request_submitted(plan)
        # A constant submit sequence makes request_id the effective tie-break,
        # exactly as before the arrival-order fix.
        self.request_submit_seq[plan.request_id] = 0
        return out


def make_srtf_legacy_tiebreak(topology, task_runtime_estimator):
    return _LegacyTieBreakSRTF(topology, task_runtime_estimator)
