#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "defect_data"


def count_pure_background(split: str) -> int:
    mask_dir = DATASET_ROOT / "masks" / split
    if not mask_dir.exists():
        return 0
    count = 0
    total = 0
    for p in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        total += 1
        if not (mask > 0).any():
            count += 1
    print(f"{split}: pure_background={count} / total={total}")
    return count


def main() -> None:
    for split in ("train", "val", "test"):
        count_pure_background(split)


if __name__ == "__main__":
    main()

