"""Detect rapid center-of-mass descent to separate voluntary lying from falls."""

from __future__ import annotations

import numpy as np

from muzukuru.config import MP_LEFT_HIP, MP_RIGHT_HIP


def hip_center_y(landmarks: np.ndarray) -> np.ndarray:
    """(T,) image-normalized hip-center y (larger = lower in frame for MediaPipe)."""
    return 0.5 * (
        landmarks[:, MP_LEFT_HIP, 1] + landmarks[:, MP_RIGHT_HIP, 1]
    ).astype(np.float32)


def rapid_descent_mask(
    landmarks: np.ndarray,
    *,
    window: int = 15,
    drop_threshold: float = 0.12,
) -> np.ndarray:
    """
    True on frames belonging to a rapid downward hip transition.

    drop_threshold: minimum increase in hip y over `window` frames
    (MediaPipe y grows downward).
    """
    t = landmarks.shape[0]
    y = hip_center_y(landmarks)
    mask = np.zeros(t, dtype=bool)
    if t < window + 1:
        return mask
    for i in range(window, t):
        drop = y[i] - y[i - window]
        if drop >= drop_threshold:
            mask[i - window : i + 1] = True
    return mask


def sequence_has_rapid_descent(
    landmarks: np.ndarray,
    *,
    window: int = 15,
    drop_threshold: float = 0.12,
    min_frames: int = 5,
) -> bool:
    m = rapid_descent_mask(
        landmarks, window=window, drop_threshold=drop_threshold
    )
    return int(m.sum()) >= min_frames


def refine_posture_label(
    base_label: str,
    landmarks: np.ndarray,
    *,
    activity_id: int | None = None,
) -> str:
    """
    Sitting/laying → normal_activity unless preceded by rapid descent.

    If rapid descent is present:
      - primary event labeled as fall
      - long still laying after descent can be prolonged_immobility (activity 11)
    """
    if base_label != "normal_activity":
        return base_label
    if not sequence_has_rapid_descent(landmarks):
        return "normal_activity"

    # Rapid descent into floor contact
    if activity_id == 11:
        # Laying after a fall-like drop: treat clip as fall (dynamics) —
        # prolonged_immobility reserved for self-recorded long motionless clips
        # unless the descent occupies <20% and most frames are flat immobility.
        y = hip_center_y(landmarks)
        still = np.abs(np.diff(y, prepend=y[:1])) < 0.005
        if still.mean() > 0.7 and sequence_has_rapid_descent(landmarks):
            return "prolonged_immobility"
        return "fall"
    if activity_id == 8:
        return "fall"
    return "fall"
