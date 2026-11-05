"""Run Plectra's policy simulator on CPU without installing vLLM.

The simulator and the scheduling policies only need the Python standard library,
but importing ``vllm_omni`` normally executes its package ``__init__`` (which
patches vLLM) and ``vllm.logger``. This shim pre-registers lightweight package
objects for ``vllm_omni`` / ``vllm_omni.diffusion`` and a minimal ``vllm.logger``,
then runs the requested module as ``__main__``.

Usage (from the artifact repo root):
    python ae/sim/sim_shim.py <module> [args...]
e.g.
    python ae/sim/sim_shim.py benchmarks.diffusion.runtime_v2_mixed_priority_runner --help
"""
import logging
import os
import runpy
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
# Repo root (vllm_omni, benchmarks) and ae/sim (ae_policies, bd_sim) are importable.
sys.path[:0] = [REPO, HERE]


def _pkg(name: str, path: str) -> None:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


# Skip vllm_omni/__init__.py and vllm_omni/diffusion/__init__.py side effects.
_pkg("vllm_omni", os.path.join(REPO, "vllm_omni"))
_pkg("vllm_omni.diffusion", os.path.join(REPO, "vllm_omni", "diffusion"))

# Minimal vllm.logger replacement.
_vllm = types.ModuleType("vllm")
_vllm.__path__ = []
_vlog = types.ModuleType("vllm.logger")
_vlog.init_logger = logging.getLogger
_vllm.logger = _vlog
sys.modules["vllm"] = _vllm
sys.modules["vllm.logger"] = _vlog
logging.basicConfig(level=logging.WARNING)

runpy.run_module(module, run_name="__main__", alter_sys=True)
