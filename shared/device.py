"""Hardware device selection for M1 Mac Mini (MPS) / CUDA / CPU."""

from __future__ import annotations


def get_device(prefer_mps: bool = True) -> str:
    """Return a device string suitable for DeepChem / PyTorch.

    Preference order on Apple Silicon: mps -> cpu (unless prefer_mps is False).
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_device(prefer_mps: bool = True):
    """Return a torch.device for native PyTorch training loops."""
    import torch

    name = get_device(prefer_mps=prefer_mps)
    return torch.device(name)
