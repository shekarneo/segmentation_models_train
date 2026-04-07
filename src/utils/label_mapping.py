#!/usr/bin/env python3

"""
Load dataset label mapping (class names and IDs) from label_mapping.json.
Used by config, train, and infer so class names/IDs are never hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABEL_MAPPING_FILENAME = "label_mapping.json"

# BGR colors for overlay: index 0 = background (black, not drawn); indices 1+ = foreground
DEFAULT_PALETTE = [
    (0, 0, 0),       # 0 background (unused for overlay fill; class 0 skipped)
    (0, 255, 0),     # 1 green
    (0, 0, 255),     # 2 red
    (255, 165, 0),   # 3 orange
    (255, 0, 255),   # 4 magenta
    (0, 255, 255),   # 5 yellow
    (128, 0, 128),   # 6 purple
    (0, 128, 128),   # 7 teal
]
# Foreground-only slice so class_id 1 -> green, 2 -> red, etc. (never black)
FOREGROUND_PALETTE = DEFAULT_PALETTE[1:]


def load_label_mapping(dataset_root: Path) -> dict[str, Any] | None:
    """
    Load label_mapping.json from dataset_root. Returns None if missing.
    Returns dict with: num_classes (int), id_to_label (dict int->str), label_to_id (dict str->int).
    id_to_label keys are integers (0, 1, 2, ...); JSON may store them as string keys.
    """
    path = Path(dataset_root) / LABEL_MAPPING_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return None
    id_to_label_raw = raw.get("id_to_label", {})
    # Normalize to int keys
    id_to_label = {}
    for k, v in id_to_label_raw.items():
        try:
            id_to_label[int(k)] = str(v)
        except (ValueError, TypeError):
            continue
    if not id_to_label:
        return None
    num_classes = max(id_to_label.keys(), default=0) + 1
    if 0 not in id_to_label:
        id_to_label[0] = "background"
    label_to_id = raw.get("label_to_id", {})
    return {
        "num_classes": num_classes,
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
    }


def get_class_labels_for_wandb(mapping: dict[str, Any] | None, num_classes: int) -> dict[int, str]:
    """Return {class_id: name} for WandB mask logging. Falls back to class_0, class_1 if no mapping."""
    if mapping and mapping.get("id_to_label"):
        labels = dict(mapping["id_to_label"])
        if 0 not in labels:
            labels[0] = "background"
        return {i: labels.get(i, f"class_{i}") for i in range(num_classes)}
    return {i: ("background" if i == 0 else f"class_{i}") for i in range(num_classes)}


def get_overlay_colors(num_classes: int) -> dict[int, tuple[int, int, int]]:
    """BGR color per class id. Class 0 = black (not drawn); foreground from FOREGROUND_PALETTE (green, red, ...)."""
    out = {0: (0, 0, 0)}
    for i in range(1, num_classes):
        out[i] = FOREGROUND_PALETTE[(i - 1) % len(FOREGROUND_PALETTE)]
    return out
