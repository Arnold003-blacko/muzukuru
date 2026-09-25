"""Canonical 5-class taxonomy and per-dataset label maps."""

from __future__ import annotations

CLASS_NAMES = [
    "normal_activity",
    "fall",
    "collapse",
    "prolonged_immobility",
    "abnormal_repetitive_movement",
]

CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Original UP-Fall activity IDs (filename A#) → Muzukuru class
# Sitting (8) and Laying (11) are provisional; rapid-descent override applied later.
UPFALL_ACTIVITY_MAP: dict[int, str] = {
    1: "fall",  # falling forward using hands
    2: "fall",  # falling forward using knees
    3: "fall",  # falling backward
    4: "fall",  # falling sideward
    5: "collapse",  # falling attempting to sit in empty chair
    6: "normal_activity",  # walking
    7: "normal_activity",  # standing
    8: "normal_activity",  # sitting (unless rapid descent)
    9: "normal_activity",  # picking up object
    10: "normal_activity",  # jumping
    11: "normal_activity",  # laying (unless rapid descent → fall / immobility)
}

# Activities that may be re-labeled when a rapid-descent transition is detected
UPFALL_POSTURE_ACTIVITIES = {8, 11}  # sitting, laying

# UR Fall binary labels from folder/name heuristics
URFALL_NAME_MAP = {
    "fall": "fall",
    "adl": "normal_activity",
    "not_fall": "normal_activity",
    "not-fall": "normal_activity",
    "nofall": "normal_activity",
}

# Le2i: annotation gives fall start/end frames; outside → normal_activity
LE2I_DEFAULT = "normal_activity"
LE2I_FALL = "fall"


def to_id(name: str) -> int:
    if name not in CLASS_TO_ID:
        raise KeyError(f"Unknown class '{name}'. Expected one of {CLASS_NAMES}")
    return CLASS_TO_ID[name]
