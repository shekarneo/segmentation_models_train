#!/usr/bin/env python3

"""
Prepare dataset for semantic segmentation training. Masks and labels are taken
from LabelMe JSONs (primary). Optionally load mask from .npy when provided.

Input (LabelMe): per-image JSON with shapes[].label and shapes[].points (polygons).
Optional: .npy mask files (e.g. SAM2) + _meta.json for instance->label when --use-npy.

Output: dataset/defect_data/ images/{split}/ masks/{split}/

Usage (from deeplabv3_plus root):
    python run.py stage=prepare stage.raw_dir=path/to/labelme_splits
    python run.py stage=prepare stage.raw_dir=path/to/raw stage.use_npy=true stage.out_dir=path/to/out
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV (cv2) is required but not installed.") from exc

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Iterable, *_, **__) -> Iterable:
        return iterable

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Path setup for src/stages/prepare.py
_STAGES_DIR = Path(__file__).resolve().parent
_SRC_DIR = _STAGES_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.image_utils import apply_clahe_bgr as _apply_clahe_bgr

RAW_DATASET_PATH = _PROJECT_ROOT / "dataset" / "stage1_pseudomasks_sam2"
OUTPUT_DATASET_PATH = _PROJECT_ROOT / "dataset" / "defect_data"


def _load_config():
    try:
        from config import get_default_config
        return get_default_config()
    except Exception:
        return None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Class IDs and names are built from data (LabelMe / _meta); label_mapping.json is the single source of truth.

def save_label_mapping(out_root: Path, label_to_id: Dict[str, int]) -> None:
    """Write label_mapping.json into out_root (used by finetune/infer/evaluate)."""
    if not label_to_id:
        return
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
    if 0 not in id_to_label:
        id_to_label[0] = "background"
    payload = {
        "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
        "id_to_label": {str(k): str(v) for k, v in sorted(id_to_label.items(), key=lambda kv: kv[0])},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "label_mapping.json"
    path.write_text(json.dumps(payload, indent=2))
    logging.info("Wrote %s", path)


def _collapse_to_binary(mask: np.ndarray) -> np.ndarray:
    """Collapse semantic mask to binary: 0=background, 1=any defect."""
    return (mask > 0).astype(np.uint8)


def _sanitize_class_for_filename(label: str) -> str:
    """Sanitize class label for use in filename (no spaces or path-unsafe chars)."""
    s = str(label).strip().replace(" ", "_")
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s) or "class"


def _instance_key_from_meta(inst: Dict) -> str:
    """Unique key for this instance from pseudomask only: 'label' string or 'class_<class_id>'."""
    label = str(inst.get("label", "")).strip()
    if label:
        return label
    if "class_id" in inst:
        return "class_" + str(int(inst["class_id"]))
    return ""


def collect_label_mapping_from_labelme(raw_root: Path, splits: List[str]) -> Dict[str, int]:
    """Build label -> class ID (1, 2, ...) from LabelMe JSON shapes[].label only. 0 = background."""
    seen: set = set()
    for split in splits:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        for jpath in split_dir.rglob("*.json"):
            if jpath.name.endswith("_meta.json"):
                continue
            try:
                with open(jpath, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            for shape in data.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                if label:
                    seen.add(label)
    if not seen:
        return {}
    return {k: i + 1 for i, k in enumerate(sorted(seen))}


def collect_label_mapping_from_json_paths(json_paths: List[Path]) -> Dict[str, int]:
    """Build label -> class ID from a list of LabelMe JSON paths. 0 = background."""
    seen: set = set()
    for jpath in json_paths:
        try:
            with open(jpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label:
                seen.add(label)
    if not seen:
        return {}
    return {k: i + 1 for i, k in enumerate(sorted(seen))}


def collect_label_mapping_from_pseudomasks(raw_root: Path, splits: List[str]) -> Dict[str, int]:
    """
    Build label -> class ID (1, 2, ...) from pseudomask _meta.json only. 0 = background.
    Used when loading masks from .npy (optional).
    """
    seen: set = set()
    for split in splits:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        for meta_path in split_dir.rglob("*_meta.json"):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                continue
            for inst in meta.get("instances", []):
                key = _instance_key_from_meta(inst)
                if key:
                    seen.add(key)
    if not seen:
        return {}
    unique_sorted = sorted(seen)
    return {key: i + 1 for i, key in enumerate(unique_sorted)}


def _rasterize_labelme_shape(
    mask: np.ndarray,
    points: List[List[float]],
    class_id: int,
) -> None:
    """Draw one polygon into mask (in-place). points = [[x,y], ...]."""
    if len(points) < 3:
        return
    pts = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [np.int32(pts)], class_id)


def mask_from_labelme(json_path: Path, label_to_id: Dict[str, int], height: int, width: int) -> np.ndarray:
    """Build semantic mask from LabelMe JSON (shapes with label + points). Returns (H,W) uint8, 0=bg."""
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


def discover_files(split_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Discover image paths and .npy mask paths by stem (for optional --use-npy)."""
    images_by_stem: Dict[str, Path] = {}
    masks_by_stem: Dict[str, Path] = {}

    for path in split_dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        stem = path.stem

        if suffix in IMAGE_EXTENSIONS:
            images_by_stem[stem] = path
        elif suffix == ".npy":
            if stem.endswith("_masks"):
                base_stem = stem[: -len("_masks")]
            else:
                base_stem = stem
            masks_by_stem[base_stem] = path

    return images_by_stem, masks_by_stem


def discover_labelme(split_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path]]:
    """Discover (image, labelme_json) pairs by stem; also return optional .npy by stem. Excludes *_meta.json."""
    images_by_stem: Dict[str, Path] = {}
    labelme_by_stem: Dict[str, Path] = {}
    npy_by_stem: Dict[str, Path] = {}

    for path in split_dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        stem = path.stem

        if suffix in IMAGE_EXTENSIONS:
            images_by_stem[stem] = path
        elif suffix == ".json" and not path.name.endswith("_meta.json"):
            labelme_by_stem[stem] = path
        elif suffix == ".npy":
            base = stem[: -len("_masks")] if stem.endswith("_masks") else stem
            npy_by_stem[base] = path

    return images_by_stem, labelme_by_stem, npy_by_stem


def discover_all_pairs_under(raw_root: Path) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    """
    Discover all (stem, img_path, json_path, npy_path) under raw_root recursively.
    npy_path is None if no .npy found. Only includes pairs that have both image and LabelMe JSON.
    """
    images_by_stem: Dict[Tuple[Path, str], Path] = {}  # (parent_dir, stem) -> path
    labelme_by_stem: Dict[Tuple[Path, str], Path] = {}
    npy_by_stem: Dict[Tuple[Path, str], Optional[Path]] = {}

    for path in raw_root.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        stem = path.stem
        key = (path.parent, stem)

        if suffix in IMAGE_EXTENSIONS:
            images_by_stem[key] = path
        elif suffix == ".json" and not path.name.endswith("_meta.json"):
            labelme_by_stem[key] = path
        elif suffix == ".npy":
            base_stem = stem[: -len("_masks")] if stem.endswith("_masks") else stem
            base_key = (path.parent, base_stem)
            npy_by_stem[base_key] = path

    pairs: List[Tuple[str, Path, Path, Optional[Path]]] = []
    for key in labelme_by_stem:
        if key not in images_by_stem:
            continue
        stem = key[1]
        img_path = images_by_stem[key]
        json_path = labelme_by_stem[key]
        npy_path = npy_by_stem.get(key)
        pairs.append((stem, img_path, json_path, npy_path))
    return pairs


def merge_mask_channels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"Expected (N, H, W) for multi-channel mask, got shape {mask.shape}")
    n, h, w = mask.shape
    merged = np.zeros((h, w), dtype=np.uint8)
    for i in range(n):
        channel = mask[i]
        if channel.ndim != 2:
            raise ValueError(f"Expected 2D channel at index {i}, got shape {channel.shape}")
        merged[channel > 0] = i + 1
    return merged


def instance_to_semantic(
    instance_mask: np.ndarray,
    mask_path: Path,
    label_to_id: Dict[str, int],
) -> np.ndarray:
    """Convert instance mask to semantic mask using only pseudomask _meta.json; class IDs from label_to_id (from pseudomasks)."""
    h, w = instance_mask.shape[:2]
    semantic = np.zeros((h, w), dtype=np.uint8)

    if mask_path.stem.endswith("_masks"):
        base_stem = mask_path.stem[: -len("_masks")]
    else:
        base_stem = mask_path.stem
    meta_path = mask_path.with_name(f"{base_stem}_meta.json")

    if not meta_path.exists():
        logging.warning(
            "Metadata file '%s' not found for mask '%s'. Falling back to binary mask (foreground=1).",
            meta_path,
            mask_path,
        )
        semantic[instance_mask > 0] = 1
        return semantic

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as exc:
        logging.warning(
            "Failed to read metadata '%s' for mask '%s': %s. Falling back to binary mask (foreground=1).",
            meta_path,
            mask_path,
            exc,
        )
        semantic[instance_mask > 0] = 1
        return semantic

    instances = meta.get("instances", [])
    if not instances:
        semantic[instance_mask > 0] = 1
        return semantic

    for inst in instances:
        inst_id = 0
        for key in ("id", "instance_id", "index"):
            if key in inst:
                inst_id = int(inst[key])
                break
        key = _instance_key_from_meta(inst)
        class_id = label_to_id.get(key, 0)
        if inst_id <= 0 or class_id <= 0:
            continue
        semantic[instance_mask == inst_id] = class_id

    return semantic


def load_and_convert_mask(mask_path: Path, label_to_id: Dict[str, int]) -> np.ndarray:
    arr = np.load(mask_path)

    if arr.ndim == 2:
        mask_2d = instance_to_semantic(arr, mask_path, label_to_id)
    elif arr.ndim == 3:
        mask_2d = merge_mask_channels(arr)
    else:
        raise ValueError(
            f"Unsupported mask shape {arr.shape} in '{mask_path}'. Expected (H, W) or (N, H, W)."
        )

    mask_2d = np.nan_to_num(mask_2d, nan=0.0, posinf=0.0, neginf=0.0)
    mask_2d = np.clip(mask_2d, 0, 255).astype(np.uint8)
    return mask_2d


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_tile_positions(
    height: int, width: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (sx, sy, crop_w, crop_h) for each tile.

    Logic mirrors the tiling in the infer stage:
    - stride = tile_size - overlap
    - for each step, compute x2 = min(step + tile_size, width)
      and x1 = max(0, x2 - tile_size) so tiles at the right/bottom
      edges are still tile_size wide/tall and are shifted left/up.
    """
    stride = max(1, tile_size - overlap)
    positions: List[Tuple[int, int, int, int]] = []
    for y_step in range(0, height, stride):
        for x_step in range(0, width, stride):
            x2 = min(x_step + tile_size, width)
            y2 = min(y_step + tile_size, height)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            crop_w = x2 - x1
            crop_h = y2 - y1
            positions.append((x1, y1, crop_w, crop_h))
    return positions


def _pad_tile(crop_img: np.ndarray, crop_mask: np.ndarray, tile_size: int):
    """Pad partial edge tiles to tile_size x tile_size so all tiles are uniform.
    Image is padded with BORDER_REFLECT_101; mask padded with 0 (background)."""
    h, w = crop_img.shape[:2]
    if h < tile_size or w < tile_size:
        pad_h = tile_size - h
        pad_w = tile_size - w
        crop_img = cv2.copyMakeBorder(crop_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        crop_mask = cv2.copyMakeBorder(crop_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return crop_img, crop_mask


def process_samples_list(
    out_root: Path,
    split_name: str,
    samples: List[Tuple[str, Path, Path, Optional[Path]]],
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    use_npy_optional: bool = False,
    save_per_class_masks: bool = False,
    tile_enabled: bool = False,
    tile_size: int = 1024,
    tile_overlap: int = 256,
    clahe_enabled: bool = False,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    crop_object_enabled: bool = False,
    crop_object_padding: int = 0,
    binary_mode: bool = False,
) -> None:
    """Write masks and copy images for a pre-assigned list of (stem, img_path, json_path, npy_path)."""
    if not samples:
        logging.warning("No samples for split '%s'.", split_name)
        return
    out_images_dir = out_root / "images" / split_name
    out_masks_dir = out_root / "masks" / split_name
    ensure_dir(out_images_dir)
    ensure_dir(out_masks_dir)
    n_with_npy = sum(1 for _, __, ___, npy in samples if npy is not None)
    logging.info(
        "Processing split '%s' (ratio): %d pairs, %d with optional .npy.",
        split_name,
        len(samples),
        n_with_npy,
    )
    for stem, img_path, json_path, npy_path in tqdm(samples, desc=split_name):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            logging.warning("Unable to read image '%s'. Skipping.", img_path)
            continue
        if clahe_enabled:
            image = _apply_clahe_bgr(image, clahe_clip, clahe_grid)
        img_h, img_w = image.shape[:2]
        if use_npy_optional and npy_path is not None:
            try:
                mask = load_and_convert_mask(npy_path, label_to_id)
            except Exception as exc:
                logging.warning("Failed to load .npy for '%s', using LabelMe raster: %s", stem, exc)
                mask = mask_from_labelme(json_path, label_to_id, img_h, img_w)
        else:
            mask = mask_from_labelme(json_path, label_to_id, img_h, img_w)
        m_h, m_w = mask.shape[:2]
        if (img_h, img_w) != (m_h, m_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if crop_object_enabled:
            import sys
            if str(_PROJECT_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(_PROJECT_ROOT / "src"))
            from utils.image_utils import get_object_crop_bbox
            cx1, cy1, cx2, cy2 = get_object_crop_bbox(image, padding=crop_object_padding)
            image = image[cy1:cy2, cx1:cx2]
            mask = mask[cy1:cy2, cx1:cx2]
            img_h, img_w = image.shape[:2]

        if binary_mode:
            mask = _collapse_to_binary(mask)

        if tile_enabled:
            positions = _get_tile_positions(img_h, img_w, tile_size, tile_overlap)
            suffix = img_path.suffix or ".png"
            for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(positions):
                crop_img = image[sy : sy + crop_h, sx : sx + crop_w]
                crop_mask = mask[sy : sy + crop_h, sx : sx + crop_w]
                crop_img, crop_mask = _pad_tile(crop_img, crop_mask, tile_size)
                tile_stem = f"{stem}_t{tile_idx:04d}"
                out_mask_path = out_masks_dir / f"{tile_stem}.png"
                if not cv2.imwrite(str(out_mask_path), crop_mask):
                    logging.error("Failed to write mask PNG '%s'.", out_mask_path)
                    continue
                if save_per_class_masks:
                    for cid, cname in id_to_label.items():
                        binary = (crop_mask == cid).astype(np.uint8) * 255
                        safe_name = _sanitize_class_for_filename(cname)
                        per_class_path = out_masks_dir / f"{tile_stem}_{cid}_{safe_name}.png"
                        cv2.imwrite(str(per_class_path), binary)
                out_img_path = out_images_dir / f"{tile_stem}{suffix}"
                if not cv2.imwrite(str(out_img_path), crop_img):
                    logging.error("Failed to write image '%s'.", out_img_path)
        else:
            out_mask_path = out_masks_dir / f"{stem}.png"
            if not cv2.imwrite(str(out_mask_path), mask):
                logging.error("Failed to write mask PNG '%s'.", out_mask_path)
                continue
            if save_per_class_masks:
                for cid, cname in id_to_label.items():
                    binary = (mask == cid).astype(np.uint8) * 255
                    safe_name = _sanitize_class_for_filename(cname)
                    per_class_path = out_masks_dir / f"{stem}_{cid}_{safe_name}.png"
                    cv2.imwrite(str(per_class_path), binary)
            out_img_path = out_images_dir / img_path.name
            if crop_object_enabled or clahe_enabled:
                if out_img_path.suffix.lower() in [".jpg", ".jpeg"]:
                    if not cv2.imwrite(str(out_img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                        logging.error("Failed to write image '%s'.", out_img_path)
                else:
                    if not cv2.imwrite(str(out_img_path), image):
                        logging.error("Failed to write image '%s'.", out_img_path)
            else:
                if out_img_path.resolve() != img_path.resolve():
                    try:
                        shutil.copy2(img_path, out_img_path)
                    except Exception as exc:
                        logging.error("Failed to copy image '%s' -> '%s': %s", img_path, out_img_path, exc)


def process_split(
    raw_root: Path,
    out_root: Path,
    split: str,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    use_labelme: bool = True,
    use_npy_optional: bool = False,
    save_per_class_masks: bool = False,
    tile_enabled: bool = False,
    tile_size: int = 1024,
    tile_overlap: int = 256,
    clahe_enabled: bool = False,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    crop_object_enabled: bool = False,
    crop_object_padding: int = 0,
    binary_mode: bool = False,
) -> None:
    split_dir = raw_root / split
    if not split_dir.exists():
        logging.warning("Split directory '%s' does not exist. Skipping.", split_dir)
        return

    out_images_dir = out_root / "images" / split
    out_masks_dir = out_root / "masks" / split
    ensure_dir(out_images_dir)
    ensure_dir(out_masks_dir)

    if use_labelme:
        images_by_stem, labelme_by_stem, npy_by_stem = discover_labelme(split_dir)
        common_stems = sorted(set(images_by_stem.keys()) & set(labelme_by_stem.keys()))
        missing_json = sorted(set(images_by_stem.keys()) - set(labelme_by_stem.keys()))
        missing_img = sorted(set(labelme_by_stem.keys()) - set(images_by_stem.keys()))
        for stem in missing_json:
            logging.warning("Image without LabelMe JSON for stem '%s' in split '%s'.", stem, split)
        for stem in missing_img:
            logging.warning("LabelMe JSON without image for stem '%s' in split '%s'.", stem, split)
        if not common_stems:
            logging.warning("No image+LabelMe JSON pairs in split '%s'.", split)
            return
        logging.info(
            "Processing split '%s' (LabelMe%s): %d pairs, %d with optional .npy.",
            split,
            " + optional .npy" if use_npy_optional else "",
            len(common_stems),
            sum(1 for s in common_stems if s in npy_by_stem),
        )
        for stem in tqdm(common_stems, desc=f"{split}"):
            img_path = images_by_stem[stem]
            json_path = labelme_by_stem[stem]
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                logging.warning("Unable to read image '%s'. Skipping.", img_path)
                continue
            if clahe_enabled:
                image = _apply_clahe_bgr(image, clahe_clip, clahe_grid)
            img_h, img_w = image.shape[:2]
            if use_npy_optional and stem in npy_by_stem:
                try:
                    mask = load_and_convert_mask(npy_by_stem[stem], label_to_id)
                except Exception as exc:
                    logging.warning("Failed to load .npy for '%s', using LabelMe raster: %s", stem, exc)
                    mask = mask_from_labelme(json_path, label_to_id, img_h, img_w)
            else:
                mask = mask_from_labelme(json_path, label_to_id, img_h, img_w)
            m_h, m_w = mask.shape[:2]
            if (img_h, img_w) != (m_h, m_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            if crop_object_enabled:
                import sys
                if str(_PROJECT_ROOT / "src") not in sys.path:
                    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
                from utils.image_utils import get_object_crop_bbox
                cx1, cy1, cx2, cy2 = get_object_crop_bbox(image, padding=crop_object_padding)
                image = image[cy1:cy2, cx1:cx2]
                mask = mask[cy1:cy2, cx1:cx2]
                img_h, img_w = image.shape[:2]

            if binary_mode:
                mask = _collapse_to_binary(mask)

            if tile_enabled:
                positions = _get_tile_positions(img_h, img_w, tile_size, tile_overlap)
                suffix = img_path.suffix or ".png"
                for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(positions):
                    crop_img = image[sy : sy + crop_h, sx : sx + crop_w]
                    crop_mask = mask[sy : sy + crop_h, sx : sx + crop_w]
                    crop_img, crop_mask = _pad_tile(crop_img, crop_mask, tile_size)
                    tile_stem = f"{stem}_t{tile_idx:04d}"
                    out_mask_path = out_masks_dir / f"{tile_stem}.png"
                    if not cv2.imwrite(str(out_mask_path), crop_mask):
                        logging.error("Failed to write mask PNG '%s'.", out_mask_path)
                        continue
                    if save_per_class_masks:
                        for cid, cname in id_to_label.items():
                            binary = (crop_mask == cid).astype(np.uint8) * 255
                            safe_name = _sanitize_class_for_filename(cname)
                            per_class_path = out_masks_dir / f"{tile_stem}_{cid}_{safe_name}.png"
                            cv2.imwrite(str(per_class_path), binary)
                    out_img_path = out_images_dir / f"{tile_stem}{suffix}"
                    if not cv2.imwrite(str(out_img_path), crop_img):
                        logging.error("Failed to write image '%s'.", out_img_path)
            else:
                out_mask_path = out_masks_dir / f"{stem}.png"
                if not cv2.imwrite(str(out_mask_path), mask):
                    logging.error("Failed to write mask PNG '%s'.", out_mask_path)
                    continue
                if save_per_class_masks:
                    for cid, cname in id_to_label.items():
                        binary = (mask == cid).astype(np.uint8) * 255
                        safe_name = _sanitize_class_for_filename(cname)
                        per_class_path = out_masks_dir / f"{stem}_{cid}_{safe_name}.png"
                        cv2.imwrite(str(per_class_path), binary)
                out_img_path = out_images_dir / img_path.name
                if crop_object_enabled or clahe_enabled:
                    if out_img_path.suffix.lower() in [".jpg", ".jpeg"]:
                        if not cv2.imwrite(str(out_img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                            logging.error("Failed to write image '%s'.", out_img_path)
                    else:
                        if not cv2.imwrite(str(out_img_path), image):
                            logging.error("Failed to write image '%s'.", out_img_path)
                else:
                    if out_img_path.resolve() != img_path.resolve():
                        try:
                            shutil.copy2(img_path, out_img_path)
                        except Exception as exc:
                            logging.error("Failed to copy image '%s' -> '%s': %s", img_path, out_img_path, exc)
        return

    # Legacy: .npy-only mode (no LabelMe)
    images_by_stem, masks_by_stem = discover_files(split_dir)
    if not masks_by_stem:
        logging.warning("No .npy masks found in '%s'.", split_dir)
    common_stems = sorted(set(masks_by_stem.keys()) & set(images_by_stem.keys()))
    missing_images = sorted(set(masks_by_stem.keys()) - set(images_by_stem.keys()))
    missing_masks = sorted(set(images_by_stem.keys()) - set(masks_by_stem.keys()))
    for stem in missing_images:
        logging.warning("Mask found without matching image for stem '%s' in split '%s'.", stem, split)
    for stem in missing_masks:
        logging.warning("Image found without matching mask for stem '%s' in split '%s'.", stem, split)
    if not common_stems:
        logging.warning("No matching image-mask pairs in split '%s'.", split)
        return
    logging.info(
        "Processing split '%s' (.npy only): %d pairs.",
        split,
        len(common_stems),
    )
    for stem in tqdm(common_stems, desc=f"{split}"):
        img_path = images_by_stem[stem]
        mask_path = masks_by_stem[stem]
        try:
            mask = load_and_convert_mask(mask_path, label_to_id)
        except Exception as exc:
            logging.error("Failed to load/convert mask '%s': %s", mask_path, exc)
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            logging.warning("Unable to read image '%s'. Skipping pair.", img_path)
            continue
        if clahe_enabled:
            image = _apply_clahe_bgr(image, clahe_clip, clahe_grid)
        img_h, img_w = image.shape[:2]
        m_h, m_w = mask.shape[:2]
        if (img_h, img_w) != (m_h, m_w):
            logging.warning(
                "Shape mismatch for '%s' in split '%s': image (%d, %d), mask (%d, %d).",
                stem, split, img_h, img_w, m_h, m_w,
            )

        if crop_object_enabled:
            import sys
            if str(_PROJECT_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(_PROJECT_ROOT / "src"))
            from utils.image_utils import get_object_crop_bbox
            cx1, cy1, cx2, cy2 = get_object_crop_bbox(image, padding=crop_object_padding)
            image = image[cy1:cy2, cx1:cx2]
            mask = mask[cy1:cy2, cx1:cx2]
            img_h, img_w = image.shape[:2]

        if binary_mode:
            mask = _collapse_to_binary(mask)

        if tile_enabled:
            positions = _get_tile_positions(img_h, img_w, tile_size, tile_overlap)
            suffix = img_path.suffix or ".png"
            for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(positions):
                crop_img = image[sy : sy + crop_h, sx : sx + crop_w]
                crop_mask = mask[sy : sy + crop_h, sx : sx + crop_w]
                crop_img, crop_mask = _pad_tile(crop_img, crop_mask, tile_size)
                tile_stem = f"{stem}_t{tile_idx:04d}"
                out_mask_path = out_masks_dir / f"{tile_stem}.png"
                if not cv2.imwrite(str(out_mask_path), crop_mask):
                    logging.error("Failed to write mask PNG '%s'.", out_mask_path)
                    continue
                if save_per_class_masks:
                    for cid, cname in id_to_label.items():
                        binary = (crop_mask == cid).astype(np.uint8) * 255
                        safe_name = _sanitize_class_for_filename(cname)
                        per_class_path = out_masks_dir / f"{tile_stem}_{cid}_{safe_name}.png"
                        cv2.imwrite(str(per_class_path), binary)
                out_img_path = out_images_dir / f"{tile_stem}{suffix}"
                if not cv2.imwrite(str(out_img_path), crop_img):
                    logging.error("Failed to write image '%s'.", out_img_path)
        else:
            out_mask_path = out_masks_dir / f"{stem}.png"
            if not cv2.imwrite(str(out_mask_path), mask):
                logging.error("Failed to write mask PNG '%s'.", out_mask_path)
                continue
            if save_per_class_masks:
                for cid, cname in id_to_label.items():
                    binary = (mask == cid).astype(np.uint8) * 255
                    safe_name = _sanitize_class_for_filename(cname)
                    per_class_path = out_masks_dir / f"{stem}_{cid}_{safe_name}.png"
                    cv2.imwrite(str(per_class_path), binary)
            out_img_path = out_images_dir / img_path.name
            if crop_object_enabled or clahe_enabled:
                if out_img_path.suffix.lower() in [".jpg", ".jpeg"]:
                    if not cv2.imwrite(str(out_img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                        logging.error("Failed to write image '%s'.", out_img_path)
                else:
                    if not cv2.imwrite(str(out_img_path), image):
                        logging.error("Failed to write image '%s'.", out_img_path)
            else:
                if out_img_path.resolve() != img_path.resolve():
                    try:
                        shutil.copy2(img_path, out_img_path)
                    except Exception as exc:
                        logging.error("Failed to copy image '%s' -> '%s': %s", img_path, out_img_path, exc)


def verify_masks(out_root: Path, split: str, num_sample: int = 5) -> None:
    """Read back a few masks and log unique class IDs and counts (to confirm non-black masks)."""
    mask_dir = out_root / "masks" / split
    if not mask_dir.exists():
        return
    masks_list = sorted(mask_dir.glob("*.png"))
    if not masks_list:
        return
    n = min(num_sample, len(masks_list))
    for i, mask_path in enumerate(masks_list[:n]):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        fg_pixels = (mask > 0).sum()
        logging.info(
            "[%s] %s: classes %s, counts %s, foreground_px=%d (%.1f%%)",
            split,
            mask_path.name,
            unique.tolist(),
            counts.tolist(),
            int(fg_pixels),
            100.0 * fg_pixels / total if total else 0,
        )
    with_fg = 0
    for p in masks_list:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None and (m > 0).any():
            with_fg += 1
    logging.info(
        "[%s] Summary: %d masks total, %d with at least one non-background pixel. (Masks use values 0,1,2 so they look almost black in viewers; use --visualize to see overlays.)",
        split,
        len(masks_list),
        with_fg,
    )


def collect_output_samples(
    out_root: Path,
    splits: Sequence[str],
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for split in splits:
        img_dir = out_root / "images" / split
        mask_dir = out_root / "masks" / split
        if not img_dir.exists() or not mask_dir.exists():
            continue
        masks_by_stem = {p.stem: p for p in mask_dir.glob("*.png") if p.is_file()}
        for img_path in img_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = img_path.stem
            mask_path = masks_by_stem.get(stem)
            if mask_path is not None:
                pairs.append((img_path, mask_path))
    return pairs


def visualize_random_sample(
    out_root: Path,
    splits: Sequence[str] = ("train", "val", "test"),
    num_samples: int = 1,
) -> None:
    if plt is None:
        logging.warning("matplotlib is not installed. Visualization is disabled.")
        return

    pairs = collect_output_samples(out_root, splits)
    if not pairs:
        logging.warning("No image-mask pairs found in output dataset for visualization.")
        return

    num_samples = min(num_samples, len(pairs))
    samples = random.sample(pairs, num_samples)

    for img_path, mask_path in samples:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            logging.warning("Failed to load image or mask: '%s', '%s'.", img_path, mask_path)
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        scaled_mask = (mask.astype(np.float32) * (255.0 / max(mask.max(), 1.0))).astype(np.uint8)
        colored_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Visualization: {img_path.name}")
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="viridis")
        plt.title("Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(overlay_rgb)
        plt.title("Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAM2 .npy masks to PNG and prepare dataset for segmentation."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATASET_PATH,
        help=f"Path to raw dataset root (default: {RAW_DATASET_PATH})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DATASET_PATH,
        help=f"Path to output dataset root (default: {OUTPUT_DATASET_PATH})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="List of dataset splits to process.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="After processing, visualize random image/mask overlays.",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=1,
        help="Number of random samples to visualize when --visualize is set.",
    )
    parser.add_argument(
        "--no-labelme",
        action="store_true",
        help="Disable LabelMe; use only .npy + _meta.json (legacy SAM2-only mode).",
    )
    parser.add_argument(
        "--use-npy",
        action="store_true",
        help="When using LabelMe, optionally load mask from .npy when present (e.g. SAM2); labels still from LabelMe JSON. Instance->class for .npy uses _meta.json.",
    )
    parser.add_argument(
        "--save-per-class-masks",
        action="store_true",
        help="Also save one binary mask per class per image as {stem}_{id}_{class}.png (e.g. img_1_scratch.png).",
    )
    parser.add_argument(
        "--tile-dataset",
        action="store_true",
        help="Tile images and masks into fixed-size patches (size/overlap from CLI or stage prepare config).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size when --tile-dataset is enabled (default: prepare_tile_size from config).",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Tile overlap when --tile-dataset is enabled (default: prepare_tile_overlap from config).",
    )
    parser.add_argument(
        "--clahe-enabled",
        action="store_true",
        help="Apply CLAHE to images before mask generation/tiling (useful for consistent contrast).",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (only used if --clahe-enabled).",
    )
    parser.add_argument(
        "--clahe-tile-grid",
        type=int,
        nargs=2,
        default=[8, 8],
        metavar=("GX", "GY"),
        help="CLAHE tile grid size, two ints (only used if --clahe-enabled).",
    )
    parser.add_argument(
        "--crop-object-enabled",
        action="store_true",
        help="Crop the main white object (device) before tiling and mask writing.",
    )
    parser.add_argument(
        "--crop-object-padding",
        type=int,
        default=0,
        help="Extra padding (pixels) around cropped object bbox.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs="+",
        default=None,
        metavar="RATIO",
        help="Split dataset by ratios (e.g. 0.8 0.1 0.1 for train/val/test). Uses --splits for names. Ignores existing train/val/test dirs; discovers all image+JSON pairs under --raw-dir and splits them. Requires LabelMe mode (no --no-labelme).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test split when using --split-ratios (default: 42).",
    )
    parser.add_argument(
        "--binary-mode",
        action="store_true",
        help="Collapse all non-background classes to one foreground class (0=background, 1=defect).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point: parse argv, resolve paths, build label mapping, process each split, verify, optional visualize."""
    args = parse_args(argv)
    raw_root = args.raw_dir if args.raw_dir.is_absolute() else _PROJECT_ROOT / args.raw_dir
    out_root = args.out_dir if args.out_dir.is_absolute() else _PROJECT_ROOT / args.out_dir
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    use_labelme = not args.no_labelme
    splits = list(args.splits) if args.splits else ["train", "val", "test"]
    binary_mode = bool(getattr(args, "binary_mode", False))

    # Stage/CLI controls (no longer rely on configs/default.yaml)
    cfg = _load_config()
    clahe_enabled = bool(getattr(args, "clahe_enabled", False)) or (bool(cfg.clahe_enabled) if cfg else False)
    clahe_clip = float(getattr(args, "clahe_clip_limit", 2.0))
    clahe_grid_arg = getattr(args, "clahe_tile_grid", [8, 8])
    clahe_grid = (int(clahe_grid_arg[0]), int(clahe_grid_arg[1])) if clahe_grid_arg else (8, 8)
    crop_object_enabled = bool(getattr(args, "crop_object_enabled", False)) or (bool(cfg.crop_object_enabled) if cfg else False)
    crop_object_padding = int(getattr(args, "crop_object_padding", 0))

    tile_size = int(args.tile_size) if getattr(args, "tile_size", None) is not None else (int(cfg.prepare_tile_size) if cfg else 1024)
    tile_overlap = int(args.tile_overlap) if getattr(args, "tile_overlap", None) is not None else (int(cfg.prepare_tile_overlap) if cfg else 256)
    tile_enabled = getattr(args, "tile_dataset", False)

    # If split ratios are provided: discover all pairs under raw_root (flat folder supported) and split by ratio.
    if args.split_ratios is not None:
        ratios = list(args.split_ratios)
        if len(ratios) != len(splits):
            raise ValueError(
                f"--split-ratios length ({len(ratios)}) must match --splits length ({len(splits)}). "
                f"Got splits={splits}, ratios={ratios}"
            )
        all_pairs = discover_all_pairs_under(raw_root)
        if not all_pairs:
            logging.error("No (image, LabelMe json) pairs found under %s", raw_root)
            return
        # Label mapping from all JSONs so ids are consistent across splits.
        if use_labelme:
            label_to_id = collect_label_mapping_from_json_paths([p[2] for p in all_pairs])
        else:
            label_to_id = collect_label_mapping_from_pseudomasks(raw_root, splits)
        if not label_to_id:
            logging.error("No labels found under %s (LabelMe mode=%s).", raw_root, use_labelme)
            return
        output_label_to_id = {"defect": 1} if binary_mode else label_to_id
        id_to_label = {v: k for k, v in output_label_to_id.items()}
        save_label_mapping(out_root, output_label_to_id)
        if binary_mode:
            logging.info("Binary mode enabled: all non-background labels collapsed to class id 1 (defect).")

        rng = random.Random(int(args.split_seed))
        shuffled = list(all_pairs)
        rng.shuffle(shuffled)
        n = len(shuffled)
        counts = [int(round(r * n)) for r in ratios]
        # Fix rounding so sum(counts) == n
        diff = n - sum(counts)
        if diff != 0:
            counts[0] += diff
        idx = 0
        for split_name, c in zip(splits, counts):
            split_samples = shuffled[idx : idx + max(0, c)]
            idx += max(0, c)
            process_samples_list(
                out_root,
                split_name,
                split_samples,
                label_to_id,
                id_to_label,
                use_npy_optional=args.use_npy,
                save_per_class_masks=args.save_per_class_masks,
                tile_enabled=tile_enabled,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                clahe_enabled=clahe_enabled,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                crop_object_enabled=crop_object_enabled,
                crop_object_padding=crop_object_padding,
                binary_mode=binary_mode,
            )
            verify_masks(out_root, split_name, num_sample=5)
    else:
        # Split-subfolder mode: expects raw_root/{train,val,test}/...
        if use_labelme:
            label_to_id = collect_label_mapping_from_labelme(raw_root, splits)
        else:
            label_to_id = collect_label_mapping_from_pseudomasks(raw_root, splits)
        if not label_to_id:
            logging.error("No labels found under %s (expected raw_root/<split>/*.json).", raw_root)
            return
        output_label_to_id = {"defect": 1} if binary_mode else label_to_id
        id_to_label = {v: k for k, v in output_label_to_id.items()}
        save_label_mapping(out_root, output_label_to_id)
        if binary_mode:
            logging.info("Binary mode enabled: all non-background labels collapsed to class id 1 (defect).")

        for split in splits:
            process_split(
                raw_root,
                out_root,
                split,
                label_to_id,
                id_to_label,
                use_labelme=use_labelme,
                use_npy_optional=args.use_npy,
                save_per_class_masks=args.save_per_class_masks,
                tile_enabled=tile_enabled,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                clahe_enabled=clahe_enabled,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                crop_object_enabled=crop_object_enabled,
                crop_object_padding=crop_object_padding,
                binary_mode=binary_mode,
            )
            verify_masks(out_root, split, num_sample=5)

    logging.info("Dataset preparation finished.")
    if args.visualize:
        visualize_random_sample(out_root, splits=splits, num_samples=getattr(args, "num_visualizations", 1))


def run_prepare(cfg) -> None:
    """Run dataset preparation from Hydra stage config. cfg is OmegaConf/DictConfig with cfg.stage.*"""
    from pathlib import Path
    stage = cfg.stage
    root = Path(cfg.paths.root) if hasattr(cfg, "paths") and hasattr(cfg.paths, "root") else _PROJECT_ROOT
    raw_dir = stage.get("raw_dir", str(RAW_DATASET_PATH))
    out_dir = stage.get("out_dir", str(OUTPUT_DATASET_PATH))
    raw_dir = Path(raw_dir) if not isinstance(raw_dir, Path) else raw_dir
    out_dir = Path(out_dir) if not isinstance(out_dir, Path) else out_dir
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    argv = ["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)]
    if stage.get("splits"):
        splits = list(stage.splits)
        if splits:
            argv.extend(["--splits", *[str(s) for s in splits]])
    if stage.get("visualize", False):
        argv.append("--visualize")
    if stage.get("use_npy", False):
        argv.append("--use-npy")
    if stage.get("save_per_class_masks", False):
        argv.append("--save-per-class-masks")
    if stage.get("prepare_tile_enabled", False):
        argv.append("--tile-dataset")
    tile_sz = stage.get("tile_size") or stage.get("prepare_tile_size")
    if tile_sz is not None:
        argv.extend(["--tile-size", str(tile_sz)])
    tile_ov = stage.get("tile_overlap") or stage.get("prepare_tile_overlap")
    if tile_ov is not None:
        argv.extend(["--tile-overlap", str(tile_ov)])
    if stage.get("clahe_enabled", False):
        argv.append("--clahe-enabled")
        if stage.get("clahe_clip_limit") is not None:
            argv.extend(["--clahe-clip-limit", str(stage.clahe_clip_limit)])
        if stage.get("clahe_tile_grid") is not None and len(stage.clahe_tile_grid) >= 2:
            argv.extend(["--clahe-tile-grid", str(stage.clahe_tile_grid[0]), str(stage.clahe_tile_grid[1])])
    if stage.get("crop_object_enabled", False):
        argv.append("--crop-object-enabled")
        if stage.get("crop_object_padding") is not None:
            argv.extend(["--crop-object-padding", str(stage.crop_object_padding)])
    if stage.get("split_ratios") is not None:
        ratios = list(stage.split_ratios)
        if ratios:
            argv.extend(["--split-ratios", *[str(r) for r in ratios]])
    if stage.get("split_seed") is not None:
        argv.extend(["--split-seed", str(stage.split_seed)])
    if stage.get("binary_mode", False):
        argv.append("--binary-mode")
    if stage.get("no_labelme", False):
        argv.append("--no-labelme")
    main(argv)



def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p

def run(cfg) -> None:
    from omegaconf import DictConfig
    root = _resolve_path(cfg.paths.root, Path.cwd())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    run_prepare(cfg)
