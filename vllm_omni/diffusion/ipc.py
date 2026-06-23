# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Used by Hop1 (GPU worker <-> scheduler) to avoid pickling large video tensors
through the MessageQueue. Tensors above ``_SHM_TENSOR_THRESHOLD`` are copied
into a named shared-memory segment; only a lightweight metadata dict is
serialised through the queue.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput

_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB


def _tensor_to_shm(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).
    """
    from multiprocessing import shared_memory

    import numpy as np

    pinned_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
    pinned_tensor.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream().synchronize()  # Ensure the copy is complete before accessing pinned memory
    # numpy has no bfloat16 dtype: transport bf16 as raw uint16 and restore the
    # torch dtype on unpack. Without this the shm pack raises ("unsupported
    # ScalarType BFloat16") and the caller falls back to pickling the *CUDA*
    # tensor via CUDA IPC, which OOMs the postprocess consumer on a near-full GPU.
    np_src = pinned_tensor.view(torch.uint16) if pinned_tensor.dtype == torch.bfloat16 else pinned_tensor
    arr = np_src.numpy()
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_arr, arr)
    handle = {
        "__tensor_shm__": True,
        "name": shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(tensor.dtype),
        "numpy_dtype": str(arr.dtype),
        "nbytes": nbytes,
    }
    shm.close()
    return handle


def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle and free the segment."""
    from multiprocessing import shared_memory

    import numpy as np

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        np_dtype = np.dtype(handle["numpy_dtype"])
        arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[: handle["nbytes"]])
        tensor = torch.from_numpy(arr.copy())
        if handle.get("torch_dtype") == "torch.bfloat16":
            tensor = tensor.view(torch.bfloat16)
    finally:
        shm.close()
        shm.unlink()
    return tensor


def pack_diffusion_output_shm(output: DiffusionOutput) -> DiffusionOutput:
    """Replace large tensors in *output* with shared-memory handles.

    The DiffusionOutput is modified **in-place** so that the (now lightweight)
    object can be serialised cheaply through a MessageQueue.
    """
    if output.output is not None and isinstance(output.output, torch.Tensor):
        if output.output.nelement() * output.output.element_size() > _SHM_TENSOR_THRESHOLD:
            output.output = _tensor_to_shm(output.output)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        if output.trajectory_latents.nelement() * output.trajectory_latents.element_size() > _SHM_TENSOR_THRESHOLD:
            output.trajectory_latents = _tensor_to_shm(output.trajectory_latents)
    return output


def unpack_diffusion_output_shm(output: DiffusionOutput) -> DiffusionOutput:
    """Reconstruct tensors from shared-memory handles produced by ``pack_diffusion_output_shm``."""
    if isinstance(output.output, dict) and output.output.get("__tensor_shm__"):
        output.output = _tensor_from_shm(output.output)
    if isinstance(output.trajectory_latents, dict) and output.trajectory_latents.get("__tensor_shm__"):
        output.trajectory_latents = _tensor_from_shm(output.trajectory_latents)
    return output
