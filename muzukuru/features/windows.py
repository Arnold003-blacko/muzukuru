"""Sliding-window segmentation of pose feature sequences."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np


def iter_windows(
    features: np.ndarray,
    *,
    length: int = 90,
    stride: int = 30,
    min_valid_ratio: float = 0.5,
    valid: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (start_index, window) for each sliding window.

    Windows shorter than `length` at the tail are skipped (no padding in training
    sample generation; pad only inside the dataset if needed).
    """
    t = features.shape[0]
    if t < length:
        return
    if valid is None:
        valid = np.ones(t, dtype=bool)

    for start in range(0, t - length + 1, stride):
        end = start + length
        window = features[start:end]
        vratio = float(valid[start:end].mean())
        if vratio < min_valid_ratio:
            continue
        yield start, window.astype(np.float32)


def window_label_from_clip(clip_label: int) -> int:
    """Clip-level labels apply to every window from that clip."""
    return int(clip_label)


def build_window_index(
    sequences: list[dict[str, Any]],
    *,
    length: int,
    stride: int,
    min_valid_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Build a flat list of window descriptors for Dataset indexing.

    Each sequence dict needs:
      features: (T, F)
      valid: (T,)
      label: int
      subject_id: str
      session_id: str
      clip_id: str
    """
    index: list[dict[str, Any]] = []
    for seq in sequences:
        feats = seq["features"]
        valid = seq.get("valid")
        for start, _ in iter_windows(
            feats,
            length=length,
            stride=stride,
            min_valid_ratio=min_valid_ratio,
            valid=valid,
        ):
            index.append(
                {
                    "clip_id": seq["clip_id"],
                    "subject_id": seq["subject_id"],
                    "session_id": seq.get("session_id", seq["clip_id"]),
                    "label": window_label_from_clip(seq["label"]),
                    "start": start,
                    "length": length,
                    "features": feats,  # shared reference; Dataset will slice
                    "valid": valid,
                }
            )
    return index
