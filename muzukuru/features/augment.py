"""Training-time augmentation for skeleton sequences."""

from __future__ import annotations

from typing import Any

import numpy as np

from muzukuru.config import NUM_LANDMARKS

# MediaPipe left/right pairs for horizontal flip of XYZ keypoint block
_LR_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


def _keypoints_width(include_visibility: bool) -> int:
    return NUM_LANDMARKS * (4 if include_visibility else 3)


def horizontal_flip_features(
    x: np.ndarray,
    *,
    include_visibility: bool = False,
) -> np.ndarray:
    """
    Mirror skeleton in the keypoint block: negate x, swap L/R joints.
    Velocity/acceleration x components are also negated.
    """
    out = x.copy()
    t, f = out.shape
    kw = _keypoints_width(include_visibility)
    per = 4 if include_visibility else 3
    kp = out[:, :kw].reshape(t, NUM_LANDMARKS, per)

    # Swap L/R
    for l, r in _LR_PAIRS:
        tmp = kp[:, l].copy()
        kp[:, l] = kp[:, r]
        kp[:, r] = tmp
    kp[:, :, 0] *= -1.0  # negate x
    out[:, :kw] = kp.reshape(t, kw)

    # Derived blocks after keypoints: vel_hip(3), vel_head(3), [acc...], ...
    # Negate x component of each 3-vector block that is velocity/acceleration.
    cursor = kw
    for _ in range(4):  # hip/head vel + hip/head acc (acc may be absent; safe if short)
        if cursor + 3 > f:
            break
        # Heuristic: first four 3-vectors after keypoints are vel/acc
        block = out[:, cursor : cursor + 3]
        # Only flip if this looks like xyz motion (always for our layout)
        if cursor < kw + 12:
            block[:, 0] *= -1.0
            out[:, cursor : cursor + 3] = block
        cursor += 3

    return out


def rotate_keypoints_2d(
    x: np.ndarray,
    degrees: float,
    *,
    include_visibility: bool = False,
) -> np.ndarray:
    """Rotate normalized keypoint (and motion) xy by ±degrees around origin."""
    out = x.copy()
    rad = np.deg2rad(degrees)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    kw = _keypoints_width(include_visibility)
    per = 4 if include_visibility else 3
    t = out.shape[0]
    kp = out[:, :kw].reshape(t, NUM_LANDMARKS, per)
    kp[:, :, :2] = kp[:, :, :2] @ R.T
    out[:, :kw] = kp.reshape(t, kw)

    # Rotate xy of vel/acc vectors
    cursor = kw
    f = out.shape[1]
    while cursor + 3 <= min(f, kw + 12):
        xy = out[:, cursor : cursor + 2]
        out[:, cursor : cursor + 2] = xy @ R.T
        cursor += 3
    return out


def temporal_jitter(
    x: np.ndarray,
    rng: np.random.Generator,
    max_frames: int = 3,
) -> np.ndarray:
    """Randomly drop or duplicate a few frames, then resample to original length."""
    t, f = x.shape
    if t < 4 or max_frames <= 0:
        return x
    n = int(rng.integers(1, max_frames + 1))
    out = x.copy()
    for _ in range(n):
        i = int(rng.integers(0, out.shape[0]))
        if rng.random() < 0.5 and out.shape[0] > t // 2:
            out = np.delete(out, i, axis=0)
        else:
            out = np.insert(out, i, out[i], axis=0)
    # Resample to original length
    idx = np.linspace(0, out.shape[0] - 1, t).astype(np.int64)
    return out[idx]


def add_gaussian_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    std: float,
    *,
    include_visibility: bool = False,
) -> np.ndarray:
    """Noise on keypoint coordinates only."""
    if std <= 0:
        return x
    out = x.copy()
    kw = _keypoints_width(include_visibility)
    noise = rng.normal(0.0, std, size=out[:, :kw].shape).astype(np.float32)
    out[:, :kw] += noise
    return out


def speed_variation(
    x: np.ndarray,
    rng: np.random.Generator,
    speed_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Stretch/compress temporal axis then resample to fixed length."""
    t = x.shape[0]
    factor = float(rng.uniform(speed_range[0], speed_range[1]))
    new_t = max(2, int(round(t / factor)))
    # Linear interpolation along time
    old_idx = np.linspace(0, t - 1, new_t)
    # Sample then map back to t
    sampled = np.stack(
        [np.interp(old_idx, np.arange(t), x[:, j]) for j in range(x.shape[1])],
        axis=1,
    ).astype(np.float32)
    new_idx = np.linspace(0, new_t - 1, t)
    out = np.stack(
        [np.interp(new_idx, np.arange(new_t), sampled[:, j]) for j in range(x.shape[1])],
        axis=1,
    ).astype(np.float32)
    return out


def augment_window(
    x: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    *,
    include_visibility: bool = False,
) -> np.ndarray:
    """Apply configured augmentations to one (T, F) window."""
    out = x
    if rng.random() < cfg.get("horizontal_flip_prob", 0.5):
        out = horizontal_flip_features(out, include_visibility=include_visibility)

    rot_max = float(cfg.get("rotation_deg", 10))
    if rot_max > 0:
        deg = float(rng.uniform(-rot_max, rot_max))
        out = rotate_keypoints_2d(out, deg, include_visibility=include_visibility)

    if rng.random() < cfg.get("temporal_jitter_prob", 0.3):
        out = temporal_jitter(
            out, rng, max_frames=int(cfg.get("temporal_jitter_max_frames", 3))
        )

    noise_std = float(cfg.get("gaussian_noise_std", 0.01))
    if noise_std > 0:
        out = add_gaussian_noise(
            out, rng, noise_std, include_visibility=include_visibility
        )

    if rng.random() < cfg.get("speed_variation_prob", 0.3):
        sr = cfg.get("speed_range", [0.85, 1.15])
        out = speed_variation(out, rng, speed_range=(float(sr[0]), float(sr[1])))

    return out.astype(np.float32)
