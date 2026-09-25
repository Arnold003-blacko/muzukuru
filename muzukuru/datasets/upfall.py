"""UP-Fall (Zenodo improved 3D skeletons) ingest."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from muzukuru.datasets.descent import refine_posture_label
from muzukuru.datasets.labels import UPFALL_ACTIVITY_MAP, UPFALL_POSTURE_ACTIVITIES, to_id
from muzukuru.features.engineering import compute_frame_features

NAME_RE = re.compile(
    r"C(?P<cam>\d+)S(?P<sub>\d+)A(?P<act>\d+)T(?P<trial>\d+)", re.IGNORECASE
)

ZENODO_FILES = [
    "SUBJECT1.zip",
    "SUBJECT2.zip",
    "SUBJECT3.zip",
    "SUBJECT4.zip",
    "SUBJECT5.zip",
]
ZENODO_BASE = "https://zenodo.org/records/12773013/files"


def download_upfall(dest: Path) -> Path:
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    for name in ZENODO_FILES:
        out = dest / name
        if out.exists() and out.stat().st_size > 10_000:
            print(f"[upfall] exists {name}")
            continue
        url = f"{ZENODO_BASE}/{name}?download=1"
        print(f"[upfall] downloading {name} ...")
        urllib.request.urlretrieve(url, out)
    return dest


def extract_zips(zip_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for zpath in sorted(zip_dir.glob("SUBJECT*.zip")):
        print(f"[upfall] extracting {zpath.name}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out_dir)
    return out_dir


def csv_to_landmarks(df: pd.DataFrame) -> np.ndarray:
    """
    Convert Joint{{1..33}}_{{X,Y,Z}} (+ optional visibility) → (T, 33, 4).

    Visibility defaults to 1.0 when columns are absent; 0.0 if XYZ all ~0.
    """
    t = len(df)
    lm = np.zeros((t, 33, 4), dtype=np.float32)
    for j in range(1, 34):
        x = df[f"Joint{j}_X"].to_numpy(dtype=np.float32)
        y = df[f"Joint{j}_Y"].to_numpy(dtype=np.float32)
        z = df[f"Joint{j}_Z"].to_numpy(dtype=np.float32)
        vis_col = f"Joint{j}_V"
        if vis_col in df.columns:
            v = df[vis_col].to_numpy(dtype=np.float32)
        else:
            v = np.ones(t, dtype=np.float32)
            dead = (np.abs(x) + np.abs(y) + np.abs(z)) < 1e-8
            v[dead] = 0.0
        lm[:, j - 1, 0] = x
        lm[:, j - 1, 1] = y
        lm[:, j - 1, 2] = z
        lm[:, j - 1, 3] = v
    return lm


def parse_clip_name(stem: str) -> dict[str, int] | None:
    m = NAME_RE.search(stem)
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


def _build_clip(
    *,
    landmarks: np.ndarray,
    valid: np.ndarray,
    label_name: str,
    feat_cfg: dict,
    subject_id: str,
    session_id: str,
    clip_id: str,
    activity_id: int,
) -> dict[str, Any] | None:
    if landmarks.shape[0] < 16:
        return None
    features, fmeta = compute_frame_features(
        landmarks,
        valid,
        include_visibility=bool(feat_cfg.get("include_visibility", False)),
        include_acceleration=bool(feat_cfg.get("include_acceleration", True)),
        include_torso_angle=bool(feat_cfg.get("include_torso_angle", True)),
        include_bbox_aspect=bool(feat_cfg.get("include_bbox_aspect", True)),
        include_com_height=bool(feat_cfg.get("include_com_height", True)),
    )
    return {
        "landmarks": landmarks,
        "valid": valid,
        "features": features,
        "feature_dim": fmeta["feature_dim"],
        "layout": fmeta["layout"],
        "label": to_id(label_name),
        "label_name": label_name,
        "subject_id": subject_id,
        "session_id": session_id,
        "clip_id": clip_id,
        "source": "upfall",
        "activity_id": activity_id,
    }


def _impact_spans(impact: np.ndarray) -> tuple[int | None, int | None]:
    """Return first/last index where impact==1, or (None, None)."""
    idx = np.flatnonzero(impact.astype(bool))
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def convert_csv_clip(
    csv_path: Path,
    *,
    feat_cfg: dict,
    split_by_impact: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert one UP-Fall CSV into one or more labeled clips.

    When split_by_impact is True (fall activities A1–A5):
      - pre-impact → normal_activity
      - impact neighborhood → fall / collapse (from activity map)
      - long post-impact still region → prolonged_immobility
    """
    meta = parse_clip_name(csv_path.stem)
    if meta is None:
        return []
    act = meta["act"]
    if act not in UPFALL_ACTIVITY_MAP:
        return []

    df = pd.read_csv(csv_path)
    landmarks = csv_to_landmarks(df)
    valid = landmarks[:, :, 3].mean(axis=1) > 0.1
    subject_id = f"upfall_s{meta['sub']}"
    session_id = f"{subject_id}_c{meta['cam']}"
    base = UPFALL_ACTIVITY_MAP[act]
    event_label = base
    if act in UPFALL_POSTURE_ACTIVITIES:
        event_label = refine_posture_label(base, landmarks, activity_id=act)

    if not split_by_impact or act > 5 or "LABEL" not in df.columns:
        clip = _build_clip(
            landmarks=landmarks,
            valid=valid,
            label_name=event_label,
            feat_cfg=feat_cfg,
            subject_id=subject_id,
            session_id=session_id,
            clip_id=f"upfall_{csv_path.stem}",
            activity_id=act,
        )
        return [clip] if clip else []

    impact = df["LABEL"].to_numpy()
    first_i, last_i = _impact_spans(impact)
    t = landmarks.shape[0]
    out: list[dict[str, Any]] = []

    if first_i is None:
        # No impact marked: keep whole clip as mapped event label
        clip = _build_clip(
            landmarks=landmarks,
            valid=valid,
            label_name=event_label,
            feat_cfg=feat_cfg,
            subject_id=subject_id,
            session_id=session_id,
            clip_id=f"upfall_{csv_path.stem}",
            activity_id=act,
        )
        return [clip] if clip else []

    # Pre-impact ADL context
    pre_end = max(0, first_i - 2)
    if pre_end >= 24:
        clip = _build_clip(
            landmarks=landmarks[:pre_end],
            valid=valid[:pre_end],
            label_name="normal_activity",
            feat_cfg=feat_cfg,
            subject_id=subject_id,
            session_id=session_id,
            clip_id=f"upfall_{csv_path.stem}_pre",
            activity_id=act,
        )
        if clip:
            out.append(clip)

    # Impact neighborhood (pad a little before/after)
    ev_start = max(0, first_i - 10)
    ev_end = min(t, last_i + 15)
    if ev_end - ev_start >= 24:
        clip = _build_clip(
            landmarks=landmarks[ev_start:ev_end],
            valid=valid[ev_start:ev_end],
            label_name=event_label,
            feat_cfg=feat_cfg,
            subject_id=subject_id,
            session_id=session_id,
            clip_id=f"upfall_{csv_path.stem}_impact",
            activity_id=act,
        )
        if clip:
            out.append(clip)

    # Post-impact immobility (if enough remaining frames)
    post_start = min(t, last_i + 15)
    if t - post_start >= 40:
        clip = _build_clip(
            landmarks=landmarks[post_start:],
            valid=valid[post_start:],
            label_name="prolonged_immobility",
            feat_cfg=feat_cfg,
            subject_id=subject_id,
            session_id=session_id,
            clip_id=f"upfall_{csv_path.stem}_post",
            activity_id=act,
        )
        if clip:
            out.append(clip)

    return out


def ingest_upfall(
    extracted_dir: Path,
    *,
    feat_cfg: dict,
) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for csv_path in sorted(extracted_dir.rglob("*.csv")):
        clips.extend(convert_csv_clip(csv_path, feat_cfg=feat_cfg))
    print(f"[upfall] converted {len(clips)} clips from {extracted_dir}")
    return clips
