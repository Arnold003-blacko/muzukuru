"""Per-frame feature engineering for temporal fall classification."""

from __future__ import annotations

from typing import Any

import numpy as np

from muzukuru.config import MP_NOSE, NUM_LANDMARKS
from muzukuru.pose.normalize import (
    hip_center,
    normalize_landmarks,
    person_bbox_aspect,
    shoulder_center,
)


def _finite_diff(x: np.ndarray) -> np.ndarray:
    """First-order frame-to-frame difference; pad first frame with zeros."""
    d = np.diff(x, axis=0, prepend=x[:1])
    d[0] = 0.0
    return d.astype(np.float32)


def torso_angle_from_vertical(landmarks_norm: np.ndarray) -> np.ndarray:
    """
    Angle (radians) between shoulder→hip vector and vertical (0, 1, 0) in image space.

    MediaPipe y increases downward; vertical down is (0, 1). Upright ≈ 0, horizontal ≈ π/2.
    landmarks_norm: (T, 33, 4) already hip-centered / torso-scaled.
    """
    sc = shoulder_center(landmarks_norm)
    hc = hip_center(landmarks_norm)
    # Vector from shoulder to hip (down the torso)
    v = hc - sc
    # Angle with image vertical (positive y)
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_unit = v / np.maximum(v_norm, 1e-6)
    cos_a = np.clip((v_unit * vertical).sum(axis=-1), -1.0, 1.0)
    return np.arccos(cos_a).astype(np.float32)


def head_center(landmarks: np.ndarray) -> np.ndarray:
    """Approximate head center as nose xyz (robust enough for velocity)."""
    return landmarks[..., MP_NOSE, :3].astype(np.float32)


def compute_frame_features(
    landmarks: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    include_visibility: bool = False,
    include_acceleration: bool = True,
    include_torso_angle: bool = True,
    include_bbox_aspect: bool = True,
    include_com_height: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build per-frame feature matrix from raw MediaPipe landmarks.

    Args:
        landmarks: (T, 33, 4) x,y,z,visibility in image-normalized coords
        valid: (T,) pose detection mask

    Returns:
        features: (T, F) float32
        meta: feature layout description
    """
    t = landmarks.shape[0]
    if valid is None:
        valid = np.ones(t, dtype=bool)

    norm = normalize_landmarks(landmarks)
    parts: list[np.ndarray] = []
    layout: list[str] = []

    # Normalized keypoints
    if include_visibility:
        kp = norm.reshape(t, NUM_LANDMARKS * 4)
        layout.append(f"keypoints_xyzw:{NUM_LANDMARKS * 4}")
    else:
        kp = norm[..., :3].reshape(t, NUM_LANDMARKS * 3)
        layout.append(f"keypoints_xyz:{NUM_LANDMARKS * 3}")
    parts.append(kp)

    hc = hip_center(norm)
    hd = head_center(norm)

    # Velocities (frame-to-frame displacement in normalized space)
    v_hip = _finite_diff(hc)
    v_head = _finite_diff(hd)
    parts.extend([v_hip, v_head])
    layout.extend(["vel_hip:3", "vel_head:3"])

    if include_acceleration:
        a_hip = _finite_diff(v_hip)
        a_head = _finite_diff(v_head)
        parts.extend([a_hip, a_head])
        layout.extend(["acc_hip:3", "acc_head:3"])

    if include_torso_angle:
        angle = torso_angle_from_vertical(norm)[:, None]
        parts.append(angle)
        layout.append("torso_angle:1")

    if include_bbox_aspect:
        # Aspect from raw (pre-normalize) coords so image geometry is meaningful
        aspect = person_bbox_aspect(landmarks, valid)[:, None]
        parts.append(aspect)
        layout.append("bbox_aspect:1")

    if include_com_height:
        # In MediaPipe, smaller y = higher in frame. Use negated hip y after
        # normalization so larger values ≈ higher center of mass.
        com_h = (-hc[:, 1])[:, None].astype(np.float32)
        parts.append(com_h)
        layout.append("com_height:1")

    features = np.concatenate(parts, axis=-1).astype(np.float32)
    # Zero out frames without a detected pose
    features[~valid] = 0.0

    meta = {
        "feature_dim": int(features.shape[1]),
        "layout": layout,
        "num_frames": t,
    }
    return features, meta


def feature_dim_from_cfg(feat_cfg: dict) -> int:
    """Predict feature dimension from config (for model init before data load)."""
    dim = NUM_LANDMARKS * (4 if feat_cfg.get("include_visibility") else 3)
    dim += 6  # hip + head velocity
    if feat_cfg.get("include_acceleration", True):
        dim += 6
    if feat_cfg.get("include_torso_angle", True):
        dim += 1
    if feat_cfg.get("include_bbox_aspect", True):
        dim += 1
    if feat_cfg.get("include_com_height", True):
        dim += 1
    return dim
