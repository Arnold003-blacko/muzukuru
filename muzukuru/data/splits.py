"""Subject/session-level train/val/test splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def group_ids(index: list[dict[str, Any]], group_by: str = "subject") -> list[str]:
    key = "subject_id" if group_by == "subject" else "session_id"
    return [str(item[key]) for item in index]


def split_groups(
    groups: list[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, set[str]]:
    """
    Split unique groups into train/val/test without leakage across sets.
    """
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = [unique[i] for i in order]

    n = len(shuffled)
    if n < 3:
        # Degenerate: put all in train, empty val/test (caller should warn)
        return {"train": set(shuffled), "val": set(), "test": set()}

    # Guarantee at least one group in each split when n >= 3
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, n - n_train - n_val)
    # Rebalance if rounding ate the test (or train) share
    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test = max(1, n_test - 1)
    # Assign remainder to train
    assigned = n_train + n_val + n_test
    if assigned < n:
        n_train += n - assigned

    train = set(shuffled[:n_train])
    val = set(shuffled[n_train : n_train + n_val])
    test = set(shuffled[n_train + n_val : n_train + n_val + n_test])
    assert len(train | val | test) == n
    assert not (train & val or train & test or val & test)
    _ = test_ratio
    return {"train": train, "val": val, "test": test}


def partition_index(
    index: list[dict[str, Any]],
    *,
    group_by: str = "subject",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    groups = group_ids(index, group_by=group_by)
    parts = split_groups(
        groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    key = "subject_id" if group_by == "subject" else "session_id"
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for item, g in zip(index, groups):
        for split_name, members in parts.items():
            if g in members:
                out[split_name].append(item)
                break
    return out


def class_counts(index: list[dict[str, Any]], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for item in index:
        counts[int(item["label"])] += 1
    return counts


def inverse_frequency_weights(
    counts: np.ndarray,
    *,
    eps: float = 1.0,
) -> np.ndarray:
    """Class weights ∝ 1 / frequency; zeros get weight 0."""
    weights = np.zeros_like(counts, dtype=np.float32)
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = 1.0 / (c + eps)
    # Normalize so mean weight over present classes ≈ 1
    present = weights[weights > 0]
    if present.size:
        weights = weights / present.mean()
    return weights.astype(np.float32)


def summarize_split(parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, items in parts.items():
        by_subj: dict[str, int] = defaultdict(int)
        by_label: dict[int, int] = defaultdict(int)
        for it in items:
            by_subj[str(it["subject_id"])] += 1
            by_label[int(it["label"])] += 1
        summary[name] = {
            "n_windows": len(items),
            "n_subjects": len(by_subj),
            "subjects": sorted(by_subj.keys()),
            "label_counts": dict(sorted(by_label.items())),
        }
    return summary
