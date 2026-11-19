import os

import torch


def get_device() -> str:
    """Pick the compute device: $DEVICE if set, else CUDA, then Apple MPS, then CPU."""
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
