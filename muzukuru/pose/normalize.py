"""Pose coordinate normalization (camera-distance and body-size invariant)."""

from __future__ import annotations

import numpy as np

from muzukuru.config import (
    MP_LEFT_HIP,
    MP_LEFT_SHOULDER,
    MP_RIGHT_HIP,
    MP_RIGHT_SHOULDER,
    NUM_LANDMARKS,
)


def hip_center(landmarks: np.ndarray) -> np.ndarray:
    """(T, 3) or (3,) mid-hip from left/right hip xyz."""
    return 0.5 * (landmarks[..., MP_LEFT_HIP, :3] + landmarks[..., MP_RIGHT_HIP, :3])


def shoulder_center(landmarks: np.ndarray) -> np.ndarray:
    return 0.5 * (
        landmarks[..., MP_LEFT_SHOULDER, :3] + landmarks[..., MP_RIGHT_SHOULDER, :3]
    )


def torso_length(landmarks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Euclidean distance between shoulder-center and hip-center."""
    sc = shoulder_center(landmarks)
    hc = hip_center(landmarks)
    length = np.linalg.norm(sc - hc, axis=-1)
    return np.maximum(length, eps)


def normalize_landmarks(
    landmarks: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Subtract hip-center, scale by torso length.

    Args:
        landmarks: (T, 33, 4) or (33, 4) with xyz + visibility
    Returns:
        Same shape; xyz normalized, visibility unchanged.
    """
    single = landmarks.ndim == 2
    if single:
        landmarks = landmarks[None, ...]

    xyz = landmarks[..., :3].copy()
    vis = landmarks[..., 3:4]
    hc = hip_center(landmarks)[..., None, :]  # (T, 1, 3)
    scale = torso_length(landmarks, eps=eps)[..., None, None]  # (T, 1, 1)
    xyz = (xyz - hc) / scale
    out = np.concatenate([xyz, vis], axis=-1).astype(np.float32)
    return out[0] if single else out


def person_bbox_aspect(landmarks: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Height/width ratio of the 2D keypoint bounding box per frame.

    landmarks: (T, 33, 4)
    returns: (T,) aspect = height / width
    """
    t = landmarks.shape[0]
    aspects = np.ones(t, dtype=np.float32)
    xy = landmarks[..., :2]
    vis = landmarks[..., 3]

    for i in range(t):
        if valid_mask is not None and not valid_mask[i]:
            aspects[i] = 1.0
            continue
        m = vis[i] > 0.3
        if m.sum() < 2:
            aspects[i] = 1.0
            continue
        pts = xy[i, m]
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        w = max(float(maxs[0] - mins[0]), 1e-6)
        h = max(float(maxs[1] - mins[1]), 1e-6)
        aspects[i] = h / w
    return aspects
