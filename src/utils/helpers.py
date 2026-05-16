"""Utility helper functions."""

import random
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device for PyTorch (GPU if available, else CPU).

    Returns:
        torch.device: Selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_model_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
