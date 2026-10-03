"""
Copyright (c) 2025 by Argonne National Laboratory.
All rights reserved.
"""

import random
import numpy as np
import torch


def select_cuda_device_index(local_rank, visible_device_count, device_type="CUDA"):
    """Resolve a process-visible accelerator index; retain the legacy API name.

    This supports two common launch modes:
    - Full-node visibility (e.g. Perlmutter/Polaris): each process sees all
      GPUs, so ``local_rank`` selects the GPU.
    - Single-device visibility (e.g. Frontier with ROCR_VISIBLE_DEVICES set
      per rank): each process sees exactly one device at index 0.

    Args:
        local_rank (int): Node-local process rank.
        visible_device_count (int): Number of process-visible accelerator devices.
        device_type (str): Device label used in diagnostics (default: CUDA).

    Returns:
        int: Device index in the process-visible device list.

    Raises:
        ValueError: If ``local_rank`` is outside the visible-device range when
            more than one device is visible.
    """
    try:
        local_rank = int(local_rank)
    except (TypeError, ValueError):
        local_rank = 0

    try:
        visible_device_count = int(visible_device_count)
    except (TypeError, ValueError):
        visible_device_count = 0

    if visible_device_count <= 1:
        return 0

    if local_rank < 0:
        return 0

    if local_rank >= visible_device_count:
        raise ValueError(
            f"LOCAL_RANK={local_rank} exceeds visible {device_type} device count "
            f"({visible_device_count})."
        )

    return local_rank


def select_device(local_rank=0, device=None):
    """Select CUDA, XPU, or CPU and bind the process-visible accelerator."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "xpu" if torch.xpu.is_available() else "cpu"
        )
    device = torch.device(device)
    if device.type in {"cuda", "xpu"}:
        accelerator = getattr(torch, device.type)
        index = device.index
        if index is None:
            index = select_cuda_device_index(
                local_rank, accelerator.device_count(), device.type.upper()
            )
        accelerator.set_device(index)
        device = torch.device(device.type, index)
    return device


def set_seed(seed):
    """Set random seeds for reproducibility across all backends.

    Configures Python ``random``, NumPy, PyTorch CPU, and (if available)
    PyTorch CUDA random number generators. Also sets cuDNN to deterministic
    mode.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dict_agg(stats, key, value, op='concat'):
    """Aggregate a value into a dictionary entry by summing or concatenating.

    If *key* already exists in *stats*, the value is combined with the
    existing entry using the specified operation.  Otherwise, the value is
    stored directly.

    Args:
        stats (dict): Dictionary to update in place.
        key (str): Key in the dictionary.
        value (numpy.ndarray): Value to add or concatenate.
        op (str): Operation type -- ``'sum'`` for element-wise addition,
            ``'concat'`` for ``numpy.concatenate`` along axis 0.

    Raises:
        NotImplementedError: If *op* is not ``'sum'`` or ``'concat'``.
    """
    # Modifies stats in place
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value
