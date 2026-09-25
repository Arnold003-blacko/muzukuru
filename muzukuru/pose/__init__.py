"""MediaPipe Pose extraction package.

Heavy deps (cv2, mediapipe) load only when extract helpers are used.
"""

from muzukuru.pose.normalize import normalize_landmarks

__all__ = [
    "extract_pose_sequence",
    "save_pose_npz",
    "load_pose_npz",
    "normalize_landmarks",
]


def __getattr__(name: str):
    if name in {"extract_pose_sequence", "save_pose_npz", "load_pose_npz"}:
        from muzukuru.pose import extract as _extract

        return getattr(_extract, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
