#!/usr/bin/env bash
# G0: print the GPU / software environment and check that Plectra imports.
source "$(dirname "$0")/common.sh"
nvidia-smi -L || { log "FAILED: nvidia-smi"; exit 1; }
nvidia-smi topo -m 2>/dev/null | head -20
CUDA_VISIBLE_DEVICES="$GPUS" "$VENV_BIN/python" - <<'PY'
import importlib, os, sys
import torch
n = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.version.cuda, "visible gpus", n)
failed = []
for m in ("vllm", "vllm_omni", "gfc", "triton", "diffusers"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        print(m, "IMPORT FAILED:", e); failed.append(m)
from vllm_omni.diffusion.runtime_v2.policies import edf_best_fit, srtf, sjf, fcfs  # noqa: F401
print("runtime_v2 policies import OK")
want = int(os.environ.get("NUM_GPUS", "1"))
if failed or n < want:
    sys.exit(f"FAILED: imports failed={failed}, visible GPUs {n} < NUM_GPUS {want}" if n < want else f"FAILED: imports failed={failed}")
print("OK")
PY
