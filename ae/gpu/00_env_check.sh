#!/usr/bin/env bash
# G0: print the GPU / software environment and check that Plectra imports.
source "$(dirname "$0")/common.sh"
nvidia-smi -L
nvidia-smi topo -m 2>/dev/null | head -20
"$VENV_BIN/python" - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for m in ("vllm", "vllm_omni", "gfc", "triton", "diffusers"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        print(m, "IMPORT FAILED:", e)
from vllm_omni.diffusion.runtime_v2.policies import edf_best_fit, srtf, sjf, fcfs  # noqa: F401
print("runtime_v2 policies import OK")
PY
