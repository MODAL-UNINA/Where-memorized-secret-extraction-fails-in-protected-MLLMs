"""Select one physical GPU before PyTorch initializes CUDA."""
from __future__ import annotations

import argparse
import os
import sys


def add_gpu_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--gpu-id", type=int, required=required,
                        help="physical GPU index shown by nvidia-smi; only this GPU is exposed")
    parser.add_argument("--disable-cudnn", action="store_true",
                        help="temporary fallback for a broken cuDNN installation; may be slower")


def configure_gpu(gpu_id: int, disable_cudnn: bool) -> dict[str, object]:
    if gpu_id < 0:
        raise ValueError("--gpu-id must be a non-negative physical GPU index")
    if "torch" in sys.modules:
        import torch
        # A later stage in the same process may reselect the GPU an earlier stage selected.
        if torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_id):
            raise RuntimeError("CUDA was initialized before --gpu-id could select the GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"--gpu-id {gpu_id} did not expose exactly one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"--gpu-id {gpu_id} does not support bfloat16")
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = False
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    info = {"gpu_id_physical": gpu_id, "gpu_id_visible": 0,
            "gpu_name": torch.cuda.get_device_name(0),
            "free_gpu_gib_at_start": round(free_bytes / 2**30, 2),
            "total_gpu_gib": round(total_bytes / 2**30, 2),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled)}
    print(f"GPU selection: {info}", flush=True)
    return info
