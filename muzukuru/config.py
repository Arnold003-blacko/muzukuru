"""Configuration loading and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

CLASS_NAMES = [
    "normal_activity",
    "fall",
    "collapse",
    "prolonged_immobility",
    "abnormal_repetitive_movement",
]

# MediaPipe Pose landmark indices used for derived features
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_NOSE = 0
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
NUM_LANDMARKS = 33


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against project root
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        p = Path(value)
        if not p.is_absolute():
            paths[key] = str(ROOT / p)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("poses", "windows", "checkpoints", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
