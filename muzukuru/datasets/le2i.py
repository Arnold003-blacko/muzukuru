"""Le2i Fall Detection Dataset ingest (video + Annotation_files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from muzukuru.datasets.labels import LE2I_FALL, to_id
from muzukuru.features.engineering import compute_frame_features
from muzukuru.pose.extract import extract_pose_sequence


def find_le2i_roots(root: Path) -> list[Path]:
    """Return scene folders that contain Videos / Annotation_files."""
    roots: list[Path] = []
    if not root.exists():
        return roots
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "Videos").is_dir() or (p / "Annotation_files").is_dir():
            roots.append(p)
        # Alternate layout: Coffee_room_01/ with .avi directly
        avis = list(p.glob("*.avi")) + list(p.glob("*.mp4"))
        if avis and (p / "Annotation_files").is_dir():
            roots.append(p)
    # Unique
    return sorted(set(roots))


def parse_le2i_annotation(path: Path) -> tuple[int | None, int | None]:
    """
    Le2i annotation: first two lines are fall start and fall end frame numbers.
    Returns (start, end) or (None, None) if no fall / unreadable.
    """
    try:
        lines = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()
        ]
    except OSError:
        return None, None
    if len(lines) < 2:
        return None, None
    try:
        start, end = int(lines[0]), int(lines[1])
    except ValueError:
        return None, None
    if start <= 0 and end <= 0:
        return None, None
    return start, end


def _video_for_stem(scene: Path, stem: str) -> Path | None:
    for folder in (scene / "Videos", scene):
        if not folder.exists():
            continue
        for ext in (".avi", ".mp4", ".mov", ".mkv"):
            cand = folder / f"{stem}{ext}"
            if cand.exists():
                return cand
            # "video (1)" style
            for p in folder.glob(f"*{stem}*{ext}"):
                return p
    return None


def _annotation_for_video(scene: Path, video_stem: str) -> Path | None:
    ann_dir = scene / "Annotation_files"
    if not ann_dir.exists():
        return None
    exact = ann_dir / f"{video_stem}.txt"
    if exact.exists():
        return exact
    matches = list(ann_dir.glob(f"*{video_stem}*.txt"))
    return matches[0] if matches else None


def label_windows_from_fall_span(
    n_frames: int,
    fall_start: int | None,
    fall_end: int | None,
) -> str:
    """Clip-level label: fall if any annotated fall span exists."""
    if fall_start is None or fall_end is None:
        return "normal_activity"
    if fall_end <= 0:
        return "normal_activity"
    return LE2I_FALL


def convert_le2i_video(
    video_path: Path,
    ann_path: Path | None,
    *,
    subject_id: str,
    feat_cfg: dict,
    pose_cfg: dict,
    max_frames: int | None = 400,
) -> dict[str, Any] | None:
    fall_start = fall_end = None
    if ann_path is not None:
        fall_start, fall_end = parse_le2i_annotation(ann_path)

    pose = extract_pose_sequence(
        video_path,
        model_complexity=int(pose_cfg.get("model_complexity", 1)),
        min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        max_frames=max_frames,
    )
    if pose["frame_count"] < 30:
        return None

    label_name = label_windows_from_fall_span(
        pose["frame_count"], fall_start, fall_end
    )
    features, fmeta = compute_frame_features(
        pose["landmarks"],
        pose["valid"],
        include_visibility=bool(feat_cfg.get("include_visibility", False)),
        include_acceleration=bool(feat_cfg.get("include_acceleration", True)),
        include_torso_angle=bool(feat_cfg.get("include_torso_angle", True)),
        include_bbox_aspect=bool(feat_cfg.get("include_bbox_aspect", True)),
        include_com_height=bool(feat_cfg.get("include_com_height", True)),
    )
    clip_id = f"le2i_{subject_id}_{video_path.stem}".replace(" ", "_")
    return {
        "landmarks": pose["landmarks"],
        "valid": pose["valid"],
        "features": features,
        "feature_dim": fmeta["feature_dim"],
        "layout": fmeta["layout"],
        "label": to_id(label_name),
        "label_name": label_name,
        "subject_id": f"le2i_{subject_id}",
        "session_id": f"le2i_{subject_id}",
        "clip_id": clip_id,
        "source": "le2i",
        "fall_start": fall_start,
        "fall_end": fall_end,
    }


def ingest_le2i(
    root: Path,
    *,
    feat_cfg: dict,
    pose_cfg: dict,
    max_videos: int | None = None,
    max_frames: int | None = 400,
) -> list[dict[str, Any]]:
    if not root.exists():
        print(f"[le2i] root missing: {root}")
        print(
            "  Place Le2i under data/external/le2i/ "
            "(Home_01, Coffee_room_01, ...) or download FallDataset.zip from UBFC."
        )
        return []

    scenes = find_le2i_roots(root)
    if not scenes:
        # Treat root itself as a flat video folder
        scenes = [root]

    clips: list[dict[str, Any]] = []
    count = 0
    for scene in scenes:
        subject = scene.name
        videos: list[Path] = []
        for folder in (scene / "Videos", scene):
            if folder.exists():
                videos.extend(folder.glob("*.avi"))
                videos.extend(folder.glob("*.mp4"))
        videos = sorted(set(videos))
        for vp in videos:
            if max_videos is not None and count >= max_videos:
                break
            ann = _annotation_for_video(scene, vp.stem)
            print(f"[le2i] {vp.relative_to(root) if vp.is_relative_to(root) else vp}")
            try:
                clip = convert_le2i_video(
                    vp,
                    ann,
                    subject_id=subject,
                    feat_cfg=feat_cfg,
                    pose_cfg=pose_cfg,
                    max_frames=max_frames,
                )
            except Exception as e:
                print(f"[le2i] skip {vp.name}: {e}")
                continue
            if clip is not None:
                clips.append(clip)
                count += 1
        if max_videos is not None and count >= max_videos:
            break

    print(f"[le2i] converted {len(clips)} clips")
    return clips
