"""Load labeled ROME receiver inputs from NumPy archives."""

import numpy as np
import torch


def load_recordings(path):
    with np.load(path, allow_pickle=False) as data:
        if not {"received", "labels", "levels"}.issubset(data.files):
            raise ValueError("The NPZ file must contain received, labels, and levels.")
        received, labels, levels = (data[name] for name in ("received", "labels", "levels"))
    if received.ndim != 4 or received.shape[1:] != (24, 8, 8) or len(received) == 0:
        raise ValueError("received must have shape (N, 24, 8, 8) with N > 0.")
    if not np.issubdtype(received.dtype, np.floating):
        raise ValueError("received must contain real-valued floating point features.")
    received = np.asarray(received, dtype=np.float32, order="C")
    if not np.isfinite(received).all():
        raise ValueError("received must contain finite float32 values.")
    for name, values, maximum in (("labels", labels, 9), ("levels", levels, 3)):
        if values.shape != (len(received),) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array of length N.")
        if values.min() < 0 or values.max() > maximum:
            raise ValueError(f"{name} must be in [0, {maximum}].")
    return (
        torch.from_numpy(received),
        torch.from_numpy(labels.astype(np.int64)),
        torch.from_numpy(levels.astype(np.int64)),
    )
