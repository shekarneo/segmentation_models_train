#!/usr/bin/env python3

"""
Analyze class-wise pixel distribution in prepared segmentation masks.

This script expects the dataset prepared by tools/prepare_dataset.py, i.e.:
    dataset/defect_data/images/{train,val,test}/...
    dataset/defect_data/masks/{train,val,test}/...

It reads the semantic masks (PNG, class IDs as uint8) and aggregates:
    - pixel counts per class (including background=0)
    - pixel ratios per class (overall + foreground-only)
for each split and for the whole dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "defect_data"
RAW_DATASET_ROOT = PROJECT_ROOT / "dataset" / "stage1_pseudomasks_sam2"


def load_label_mapping(dataset_root: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load id_to_label and label_to_id from label_mapping.json under dataset_root."""
    mapping_path = dataset_root / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"label_mapping.json not found under {dataset_root}")
    with open(mapping_path, "r") as f:
        data = json.load(f)
    raw_id_to_label = data.get("id_to_label", {})
    id_to_label: Dict[int, str] = {}
    for k, v in raw_id_to_label.items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        id_to_label[cid] = str(v)
    if 0 not in id_to_label:
        id_to_label[0] = "background"
    label_to_id = {str(k): int(v) for k, v in data.get("label_to_id", {}).items()}
    return id_to_label, label_to_id


def _rasterize_labelme_shape(
    mask: np.ndarray,
    points: list[list[float]],
    class_id: int,
) -> None:
    """Draw one polygon into mask (in-place). points = [[x,y], ...]."""
    if len(points) < 3:
        return
    pts = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [np.int32(pts)], class_id)


def mask_from_labelme(
    json_path: Path,
    label_to_id: Dict[str, int],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Build semantic mask from LabelMe JSON (shapes with label + points).
    This mirrors tools/prepare_dataset.mask_from_labelme.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return mask

    raw_h, raw_w = data.get("imageHeight"), data.get("imageWidth")
    json_h = height if (raw_h is None or not isinstance(raw_h, (int, float)) or raw_h < 1) else int(raw_h)
    json_w = width if (raw_w is None or not isinstance(raw_w, (int, float)) or raw_w < 1) else int(raw_w)

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        class_id = label_to_id.get(label, 0)
        if class_id <= 0:
            continue
        points = shape.get("points", [])
        if not points:
            continue
        if (json_h, json_w) != (height, width):
            scale_y, scale_x = height / max(1, json_h), width / max(1, json_w)
            points = [[p[0] * scale_x, p[1] * scale_y] for p in points]
        _rasterize_labelme_shape(mask, points, class_id)
    return mask


def analyze_split(mask_dir: Path) -> Tuple[Dict[int, int], int, int, int]:
    """
    Aggregate class-wise pixel counts for all masks in mask_dir.

    Returns:
        counts: dict[class_id] -> pixel_count
        total_pixels: total pixels across all masks (including background)
        total_fg_pixels: total foreground pixels (class_id > 0)
        num_images: number of mask images processed
    """
    counts: Dict[int, int] = defaultdict(int)
    total_pixels = 0
    total_fg_pixels = 0
    num_images = 0

    if not mask_dir.exists():
        return {}, 0, 0, 0

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        num_images += 1
        unique, cts = np.unique(mask, return_counts=True)
        for cid, n in zip(unique.tolist(), cts.tolist()):
            counts[int(cid)] += int(n)
            total_pixels += int(n)
            if cid > 0:
                total_fg_pixels += int(n)

    return dict(counts), total_pixels, total_fg_pixels, num_images


def analyze_labelme_split(
    raw_root: Path,
    split: str,
    label_to_id: Dict[str, int],
) -> Tuple[Dict[int, int], int, int, int]:
    """
    Aggregate class-wise pixel counts by rasterizing LabelMe JSONs directly,
    mirroring the logic used when preparing PNG masks.

    Returns:
        counts: dict[class_id] -> pixel_count
        total_pixels: total pixels across all masks (including background)
        total_fg_pixels: total foreground pixels (class_id > 0)
        num_images: number of image+JSON pairs processed
    """
    counts: Dict[int, int] = defaultdict(int)
    total_pixels = 0
    total_fg_pixels = 0
    num_images = 0

    split_dir = raw_root / split
    if not split_dir.exists():
        return {}, 0, 0, 0

    # Discover image / LabelMe pairs by stem, similar to tools/prepare_dataset.discover_labelme
    images_by_stem: Dict[str, Path] = {}
    labelme_by_stem: Dict[str, Path] = {}

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in split_dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        stem = path.stem
        if suffix in image_exts:
            images_by_stem[stem] = path
        elif suffix == ".json" and not path.name.endswith("_meta.json"):
            labelme_by_stem[stem] = path

    common_stems = sorted(set(images_by_stem.keys()) & set(labelme_by_stem.keys()))
    if not common_stems:
        return {}, 0, 0, 0

    for stem in common_stems:
        img_path = images_by_stem[stem]
        json_path = labelme_by_stem[stem]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        mask = mask_from_labelme(json_path, label_to_id, img_h, img_w)
        num_images += 1

        unique, cts = np.unique(mask, return_counts=True)
        for cid, n in zip(unique.tolist(), cts.tolist()):
            counts[int(cid)] += int(n)
            total_pixels += int(n)
            if cid > 0:
                total_fg_pixels += int(n)

    return dict(counts), total_pixels, total_fg_pixels, num_images


def print_distribution(
    split_name: str,
    counts: Dict[int, int],
    total_pixels: int,
    total_fg_pixels: int,
    id_to_label: Dict[int, str],
) -> None:
    if total_pixels == 0:
        print(f"\n[{split_name}] No pixels (no masks found).")
        return

    print(f"\n[{split_name}]")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Foreground pixels (>0): {total_fg_pixels} "
          f"({100.0 * total_fg_pixels / total_pixels:.3f}% of all pixels)")

    # Stable ordering by class ID
    for cid in sorted(counts.keys()):
        n = counts[cid]
        label = id_to_label.get(cid, f"class_{cid}")
        ratio_all = 100.0 * n / total_pixels if total_pixels else 0.0
        ratio_fg = (
            100.0 * n / total_fg_pixels if (total_fg_pixels and cid > 0) else 0.0
        )
        print(
            f"    class {cid:2d} ({label:>15s}): "
            f"pixels={n:12d}, "
            f"ratio_all={ratio_all:7.3f}%, "
            f"ratio_fg={ratio_fg:7.3f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze class-wise pixel distribution in segmentation masks."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Path to prepared dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_DATASET_ROOT,
        help=f"Path to raw LabelMe dataset root (default: {RAW_DATASET_ROOT})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="List of splits to analyze (default: train val test).",
    )
    parser.add_argument(
        "--compare-labelme",
        action="store_true",
        help="Also compute distribution directly from LabelMe JSONs and compare to prepared masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    raw_root: Path = args.raw_root
    splits = list(args.splits)

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root '{dataset_root}' does not exist.")
    if args.compare_labelme and not raw_root.exists():
        raise SystemExit(f"Raw LabelMe root '{raw_root}' does not exist.")

    id_to_label, label_to_id = load_label_mapping(dataset_root)

    # Aggregate over all splits (prepared masks)
    global_counts: Dict[int, int] = defaultdict(int)
    global_total_pixels = 0
    global_total_fg_pixels = 0

    # Aggregate over all splits (LabelMe)
    lm_global_counts: Dict[int, int] = defaultdict(int)
    lm_global_total_pixels = 0
    lm_global_total_fg_pixels = 0

    for split in splits:
        mask_dir = dataset_root / "masks" / split
        counts, total_px, total_fg_px, num_images = analyze_split(mask_dir)
        print(f"\n=== Split '{split}' (prepared PNG masks) ===")
        print(f"Mask directory: {mask_dir}")
        print(f"Number of mask images: {num_images}")
        print_distribution(split, counts, total_px, total_fg_px, id_to_label)

        for cid, n in counts.items():
            global_counts[cid] += n
        global_total_pixels += total_px
        global_total_fg_pixels += total_fg_px

        if args.compare_labelme:
            lm_counts, lm_total_px, lm_total_fg_px, lm_num = analyze_labelme_split(
                raw_root, split, label_to_id
            )
            print(f"\n--- Split '{split}' (LabelMe JSONs, rasterized on the fly) ---")
            print(f"Raw LabelMe directory: {raw_root / split}")
            print(f"Number of image+JSON pairs: {lm_num}")
            print_distribution(
                f"{split} [LabelMe]",
                lm_counts,
                lm_total_px,
                lm_total_fg_px,
                id_to_label,
            )

            # Per-class comparison
            print(f"\nDifference (LabelMe - PNG) per class for split '{split}':")
            all_cids = sorted(set(counts.keys()) | set(lm_counts.keys()))
            for cid in all_cids:
                png_n = counts.get(cid, 0)
                lm_n = lm_counts.get(cid, 0)
                delta = lm_n - png_n
                label = id_to_label.get(cid, f"class_{cid}")
                print(
                    f"  class {cid:2d} ({label:>15s}): "
                    f"LabelMe={lm_n:12d}, PNG={png_n:12d}, delta={delta:12d}"
                )

            for cid, n in lm_counts.items():
                lm_global_counts[cid] += n
            lm_global_total_pixels += lm_total_px
            lm_global_total_fg_pixels += lm_total_fg_px

    print("\n=== Overall (all splits combined) - prepared PNG masks ===")
    print_distribution(
        "overall_png",
        dict(global_counts),
        global_total_pixels,
        global_total_fg_pixels,
        id_to_label,
    )

    if args.compare_labelme:
        print("\n=== Overall (all splits combined) - LabelMe JSONs ===")
        print_distribution(
            "overall_labelme",
            dict(lm_global_counts),
            lm_global_total_pixels,
            lm_global_total_fg_pixels,
            id_to_label,
        )

        print("\nDifference (LabelMe - PNG) per class overall:")
        all_cids = sorted(set(global_counts.keys()) | set(lm_global_counts.keys()))
        for cid in all_cids:
            png_n = global_counts.get(cid, 0)
            lm_n = lm_global_counts.get(cid, 0)
            delta = lm_n - png_n
            label = id_to_label.get(cid, f"class_{cid}")
            print(
                f"  class {cid:2d} ({label:>15s}): "
                f"LabelMe={lm_n:12d}, PNG={png_n:12d}, delta={delta:12d}"
            )


if __name__ == "__main__":
    main()

