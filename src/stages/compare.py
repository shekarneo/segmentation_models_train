#!/usr/bin/env python3

"""
Compare segmentation model predictions (tiled infer) to a reference pixel mask.

Reference GT (pick one path):
  - SAM2 (stage.use_sam2=true): bbox-prompted SAM2 masks from LabelMe boxes; optional refinement.
  - LabelMe polygons (use_sam2=false, gt_source=labelme): mask_from_labelme on full-res image, then
    the same pipeline as prepare/infer when enabled: full-image CLAHE on BGR, then white-object crop,
    then slice image + GT mask.
  - PNG masks (use_sam2=false, gt_source=png): load gt_masks_dir/{stem}.png (grayscale class ids).

The model architecture comes from stage config / checkpoint (not tied to a single backbone name).
Outputs: pixel IoU/Dice, TP/FP/FN overlays, and comparison visualizations.

Usage:
  python run.py stage=compare
  python run.py stage=compare stage.use_sam2=false stage.gt_source=labelme
  python run.py stage=compare stage.use_sam2=false stage.gt_masks_dir=/path/to/gt_masks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv

import cv2
import numpy as np
import yaml
import torch

# Path setup for src/stages/compare.py
_STAGES_DIR = Path(__file__).resolve().parent
_SRC_DIR = _STAGES_DIR.parent
PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import Config, get_default_config
from utils.image_utils import apply_clahe_bgr
from .infer import _get_tile_positions
from .prepare import mask_from_labelme

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "Consensus_Mask_Reviewer_Test"
# SAM2 checkpoints: use jnj-sam2-pipeline in repo (checkpoints/base_models/sam2_hiera_large.pt)
SAM2_CHECKPOINT_DIR = PROJECT_ROOT / "jnj-sam2-pipeline" / "checkpoints" / "base_models"
DEFAULT_SAM2_CHECKPOINT = SAM2_CHECKPOINT_DIR / "sam2_hiera_large.pt"

# Color coding (BGR) - same as eval_detections_boxes_vs_masks.py
# GT: green, TP: orange, FP: red, FN: blue
COLORS = {
    "GT": (0, 255, 0),
    "TP": (0, 165, 255),
    "FP": (0, 0, 255),
    "FN": (255, 0, 0),
}


def _sanitize_label_for_filename(label: str) -> str:
    """Make a safe filename fragment from a class label."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)


def _pipeline_root(root: Path) -> Path:
    """Locate local jnj-sam2-pipeline root (same logic as pseudomask stage)."""
    pipeline_root = root / "jnj-sam2-pipeline"
    pipeline_root = Path(os.environ.get("SAM2_PIPELINE_ROOT", str(pipeline_root)))
    pipeline_root = pipeline_root.resolve()
    pipeline_src = pipeline_root / "src"
    if not (pipeline_src / "__init__.py").exists():
        raise FileNotFoundError(
            f"SAM2 pipeline not found at '{pipeline_src}'. "
            "Expected a local copy at jnj-sam2-pipeline/src or set SAM2_PIPELINE_ROOT."
        )
    return pipeline_root


def _import_sam2_pipeline(pipeline_root: Path):
    """
    Import the local jnj-sam2-pipeline 'src/' package under an isolated alias so:
    - its internal relative imports work
    - it does not collide with this repo's 'src' package
    (mirrors src/stages/pseudomask.py).
    """
    pkg = "sam2_pipeline"
    if pkg in sys.modules:
        return sys.modules[pkg]
    src_dir = pipeline_root / "src"
    init_py = src_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg,
        init_py,
        submodule_search_locations=[str(src_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for SAM2 pipeline at '{src_dir}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    return module


def _resolve_paths(cfg: Config) -> None:
    # dataset_root is only used as a legacy fallback for label_mapping.json; label_mapping_path is preferred.
    if getattr(cfg, "dataset_root", None) is not None and not cfg.dataset_root.is_absolute():
        cfg.dataset_root = PROJECT_ROOT / cfg.dataset_root
    if getattr(cfg, "label_mapping_path", None) is not None:
        p = Path(cfg.label_mapping_path)
        if not p.is_absolute():
            cfg.label_mapping_path = PROJECT_ROOT / p


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_intensity_image_from_cfg(
    image_bgr: np.ndarray,
    cv_cfg: dict | None,
) -> np.ndarray:
    """
    Build a single-channel intensity image using the same options as refine_sam2_masks.py:
      - intensity.mode: "gray" (default) or "hsv_v"
      - intensity.clahe.enabled: bool
      - intensity.clahe.clip_limit: float
      - intensity.clahe.tile_grid_size: int
    """
    if cv_cfg is None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mode_cfg = cv_cfg.get("intensity", {})
    mode = str(mode_cfg.get("mode", "gray")).lower()

    if mode == "hsv_v":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        intensity = hsv[:, :, 2]
    else:
        intensity = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe_cfg = mode_cfg.get("clahe", {})
    if bool(clahe_cfg.get("enabled", False)):
        clip = float(clahe_cfg.get("clip_limit", 2.0))
        tile = int(clahe_cfg.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        intensity = clahe.apply(intensity)

    return intensity


def _threshold_dark_regions_from_cfg(
    intensity: np.ndarray,
    thr_cfg: dict | None,
) -> np.ndarray:
    """
    Threshold dark regions, mirroring refine_sam2_masks.py:
      - threshold.method: "global", "otsu", "adaptive_mean", "adaptive_gaussian"
      - threshold.value: int (for global)
      - threshold.invert: bool
      - threshold.block_size: odd int (for adaptive)
      - threshold.C: int (for adaptive)
    """
    if thr_cfg is None:
        # Default to Otsu + invert (dark=1) if no config is provided.
        _, bin_mask = cv2.threshold(
            intensity, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return (bin_mask > 0).astype(np.uint8)

    method = str(thr_cfg.get("method", "otsu")).lower()
    value = int(thr_cfg.get("value", 40))
    invert = bool(thr_cfg.get("invert", True))
    block_size = int(thr_cfg.get("block_size", 31))
    C = int(thr_cfg.get("C", 5))

    if method == "global":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, bin_mask = cv2.threshold(intensity, value, 255, thresh_type)
    elif method == "otsu":
        _, bin_mask = cv2.threshold(
            intensity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if invert:
            bin_mask = cv2.bitwise_not(bin_mask)
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if method == "adaptive_mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        bin_mask = cv2.adaptiveThreshold(
            intensity,
            255,
            adaptive_method,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )
        if invert:
            bin_mask = cv2.bitwise_not(bin_mask)
    else:
        # Fallback to simple Otsu invert if method is unknown
        _, bin_mask = cv2.threshold(
            intensity, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    return (bin_mask > 0).astype(np.uint8)


def discover_image_labelme_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Return list of (image_path, labelme_json_path) with same stem. Exclude *_meta.json."""
    data_dir = data_dir.resolve()
    images: dict[str, Path] = {}
    jsons: dict[str, Path] = {}
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        stem, suffix = p.stem, p.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            images[stem] = p
        elif suffix == ".json" and not p.name.endswith("_meta.json"):
            jsons[stem] = p
    common = sorted(set(images.keys()) & set(jsons.keys()))
    return [(images[s], jsons[s]) for s in common]


def bbox_from_shape(shape: dict) -> Tuple[float, float, float, float] | None:
    """Return (x1, y1, x2, y2) from LabelMe shape."""
    pts = shape.get("points", [])
    stype = shape.get("shape_type", "rectangle")
    if stype == "rectangle" and len(pts) >= 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    elif stype == "polygon" and len(pts) >= 3:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1, y1, x2, y2)


def build_tiled_labelme(
    full_json_path: Path,
    tile_sx: int,
    tile_sy: int,
    tile_w: int,
    tile_h: int,
    tile_image_name: str,
) -> dict:
    """Build LabelMe JSON for one tile: only shapes whose bbox intersects the tile, in tile-local coords."""
    with open(full_json_path, "r") as f:
        data = json.load(f)
    shapes_out = []
    for shape in data.get("shapes", []):
        box = bbox_from_shape(shape)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Intersect with tile
        ix1 = max(x1, tile_sx)
        iy1 = max(y1, tile_sy)
        ix2 = min(x2, tile_sx + tile_w)
        iy2 = min(y2, tile_sy + tile_h)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        # Convert to tile-local
        lx1 = ix1 - tile_sx
        ly1 = iy1 - tile_sy
        lx2 = ix2 - tile_sx
        ly2 = iy2 - tile_sy
        shapes_out.append({
            "label": shape.get("label", "defect"),
            "points": [[lx1, ly1], [lx2, ly2]],
            "shape_type": "rectangle",
            "flags": {},
        })
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes_out,
        "imagePath": tile_image_name,
        "imageData": None,
        "imageHeight": tile_h,
        "imageWidth": tile_w,
    }




def run_sam2_via_command(
    tile_image_path: Path,
    tile_json_path: Path,
    output_mask_path: Path,
    sam2_command: str,
) -> bool:
    """Run external SAM2 script: image + boxes json -> mask PNG. Command format uses %s %s %s for image, json, output."""
    cmd = sam2_command % (str(tile_image_path), str(tile_json_path), str(output_mask_path))
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            logging.warning("SAM2 command failed: %s\nstderr: %s", cmd, result.stderr)
            return False
        return output_mask_path.exists()
    except Exception as e:
        logging.warning("SAM2 command error: %s", e)
        return False


def _refine_mask(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    refine_method: str | None,
    cv_cfg: dict | None = None,
) -> np.ndarray:
    """
    Refine a binary mask using intensity thresholding or kmeans clustering.

    When cv_cfg is provided (from configs/stage/refinement.yaml cv section),
    we mirror its settings for intensity, thresholding, and kmeans (k, max_iter, attempts).
    """
    if not refine_method or refine_method.lower() not in ("kmeans", "threshold"):
        return mask

    refine_method = refine_method.lower()
    h, w = image_bgr.shape[:2]

    # We apply refinement per class independently
    refined_mask = np.zeros_like(mask)

    # Build intensity image similar to refine_sam2_masks.py
    intensity = _build_intensity_image_from_cfg(image_bgr, cv_cfg)

    # Threshold config (only for threshold mode)
    thr_cfg = None
    if cv_cfg is not None:
        thr_cfg = cv_cfg.get("threshold", {})

    # K-means config (if provided)
    kmeans_k = 2
    kmeans_max_iter = 30
    kmeans_attempts = 3
    if cv_cfg is not None:
        refine_cfg = cv_cfg.get("refine", {})
        kmeans_cfg = refine_cfg.get("kmeans", {})
        kmeans_k = int(kmeans_cfg.get("k", kmeans_k))
        kmeans_max_iter = int(kmeans_cfg.get("max_iter", kmeans_max_iter))
        kmeans_attempts = int(kmeans_cfg.get("attempts", kmeans_attempts))

    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue

        cls_mask = (mask == cls_id).astype(np.uint8)

        if refine_method == "threshold":
            dark_mask = _threshold_dark_regions_from_cfg(intensity, thr_cfg)
            refined = (cls_mask & dark_mask).astype(np.uint8)
        elif refine_method == "kmeans":
            ys, xs = np.where(cls_mask > 0)
            if ys.size == 0:
                refined = cls_mask
            else:
                vals = intensity[ys, xs].astype(np.float32).reshape(-1, 1)
                if vals.shape[0] < max(kmeans_k, 2) or kmeans_k < 2:
                    refined = cls_mask
                else:
                    criteria = (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        kmeans_max_iter,
                        1.0,
                    )
                    try:
                        _compact, labels, centers = cv2.kmeans(
                            vals,
                            kmeans_k,
                            None,
                            criteria,
                            kmeans_attempts,
                            cv2.KMEANS_PP_CENTERS,
                        )
                        darkest_idx = int(np.argmin(centers))
                        labels_flat = labels.reshape(-1)
                        selected = (labels_flat == darkest_idx).astype(np.uint8)

                        refined_candidate = np.zeros((h, w), dtype=np.uint8)
                        refined_candidate[ys, xs] = selected

                        # Only keep if it doesn't wipe out the mask entirely or become too tiny
                        if refined_candidate.sum() > (0.05 * cls_mask.sum()):
                            refined = refined_candidate
                        else:
                            refined = cls_mask
                    except Exception:
                        refined = cls_mask
        else:
            refined = cls_mask

        # If refinement completely removes the mask, fall back to original mask
        if refined.sum() == 0:
            refined = cls_mask

        refined_mask[refined > 0] = cls_id

    return refined_mask


def _ensure_bbox_coverage(
    orig_mask: np.ndarray,
    refined_mask: np.ndarray,
    json_path: Path,
) -> np.ndarray:
    """
    Ensure that for every LabelMe rectangle (bbox) that had some SAM2 foreground in the
    original mask, we still have non-empty mask after refinement.

    If refinement wipes out all foreground inside a bbox, we restore the original mask
    values for that bbox region.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        # If JSON can't be read, just return refined_mask unchanged
        return refined_mask

    shapes = data.get("shapes", [])
    h, w = orig_mask.shape[:2]

    for sh in shapes:
        if sh.get("shape_type") != "rectangle":
            continue
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, x2 = int(round(min(x1, x2))), int(round(max(x1, x2)))
        y1, y2 = int(round(min(y1, y2))), int(round(max(y1, y2)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        orig_region = orig_mask[y1:y2, x1:x2]
        ref_region = refined_mask[y1:y2, x1:x2]

        # Only enforce coverage if original SAM2 had some foreground in this bbox
        if orig_region is not None and orig_region.sum() > 0 and ref_region.sum() == 0:
            refined_mask[y1:y2, x1:x2] = orig_region

    return refined_mask


def run_infer_py_subprocess(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    tile_size: int,
    tile_overlap: int,
    tiled: bool = True,
    cfg=None,
) -> bool:
    """Run inference stage in-process; masks written as stem.png to output_dir. Returns True on success. cfg: optional Config from compare stage."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    try:
        from .infer import run_inference_stage
        if cfg is None:
            cfg = get_default_config()
            _resolve_paths(cfg)
        cfg.tiled_inference = tiled
        cfg.tile_size = tile_size
        cfg.tile_overlap = tile_overlap
        run_inference_stage(cfg, input_dir, output_dir, checkpoint_path, tiled=tiled)
        return True
    except Exception as e:
        logging.error("Inference stage error: %s", e)
        return False


def run_sam2_tiled_with_command(
    img_bgr: np.ndarray,
    full_json_path: Path,
    tile_size: int,
    tile_overlap: int,
    sam2_command: str,
    work_dir: Path,
    stem: str,
    tiled: bool = True,
) -> np.ndarray:
    """Create tiles (same grid as tiled infer), build tiled LabelMe JSONs (bboxes in tile-local coords),
    run SAM2 per tile with bbox prompts, merge tile masks back to full size."""
    orig_h, orig_w = img_bgr.shape[:2]
    if tiled:
        positions = _get_tile_positions(orig_h, orig_w, tile_size, tile_overlap)
    else:
        positions = [(0, 0, orig_w, orig_h)]
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(positions):
        crop = img_bgr[sy : sy + crop_h, sx : sx + crop_w]
        tile_name = f"{stem}_t{tile_idx:04d}.png"
        tile_path = work_dir / tile_name
        cv2.imwrite(str(tile_path), crop)
        tiled_data = build_tiled_labelme(full_json_path, sx, sy, crop_w, crop_h, tile_name)
        tile_json_path = work_dir / f"{stem}_t{tile_idx:04d}.json"
        with open(tile_json_path, "w") as f:
            json.dump(tiled_data, f, indent=2)
        out_mask_path = work_dir / f"{stem}_t{tile_idx:04d}_sam2.png"
        ok = run_sam2_via_command(tile_path, tile_json_path, out_mask_path, sam2_command)
        if ok and out_mask_path.exists():
            tile_mask = cv2.imread(str(out_mask_path), cv2.IMREAD_GRAYSCALE)
            if tile_mask is not None and tile_mask.shape[:2] == (crop_h, crop_w):
                full_mask[sy : sy + crop_h, sx : sx + crop_w] = np.maximum(
                    full_mask[sy : sy + crop_h, sx : sx + crop_w],
                    (tile_mask > 0).astype(np.uint8),
                )
            elif tile_mask is not None:
                # Resize if SAM2 returned different size
                tile_mask = cv2.resize(tile_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                full_mask[sy : sy + crop_h, sx : sx + crop_w] = np.maximum(
                    full_mask[sy : sy + crop_h, sx : sx + crop_w],
                    (tile_mask > 0).astype(np.uint8),
                )
    return full_mask


def run_sam2_tiled_inprocess(
    img_bgr: np.ndarray,
    full_json_path: Path,
    tile_size: int,
    tile_overlap: int,
    work_dir: Path,
    stem: str,
    tiled: bool = True,
    label_to_id: Dict[str, int] | None = None,
) -> np.ndarray | None:
    """Run SAM2 in-process if sam2 package is available (bbox prompts per tile, merge to full mask)."""
    try:
        # Use the same local jnj-sam2-pipeline + SAM2LoRA flow as pseudomask stage.
        pipeline_root = _pipeline_root(PROJECT_ROOT)
        _import_sam2_pipeline(pipeline_root)
        from sam2_pipeline.models.sam2_lora import SAM2LoRA  # type: ignore

        model_repo = os.environ.get("SAM2_MODEL_REPO", "facebook/sam2-hiera-large")
        device = os.environ.get("SAM2_DEVICE", "cuda")
        local_ckpt = os.environ.get("SAM2_CKPT_PATH")
        if not local_ckpt:
            default_ckpt = (
                PROJECT_ROOT
                / "jnj-sam2-pipeline"
                / "checkpoints"
                / "base_models"
                / "sam2_hiera_large.pt"
            )
            if default_ckpt.exists():
                local_ckpt = str(default_ckpt)
        kwargs = {"local_path": local_ckpt} if local_ckpt else {}
        logging.info(
            "Loading SAM2 via SAM2LoRA: repo=%s, device=%s, local_ckpt=%s",
            model_repo,
            device,
            local_ckpt,
        )
        model = SAM2LoRA(model_id=model_repo, device=device, **kwargs)
        predictor = model.predictor
        logging.info("SAM2 model loaded successfully.")
    except Exception as e:
        logging.warning("Failed to initialize local SAM2 pipeline: %s", e)
        return None
    orig_h, orig_w = img_bgr.shape[:2]
    if tiled:
        positions = _get_tile_positions(orig_h, orig_w, tile_size, tile_overlap)
    else:
        positions = [(0, 0, orig_w, orig_h)]
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(positions):
        crop = img_bgr[sy : sy + crop_h, sx : sx + crop_w]
        tiled_data = build_tiled_labelme(
            full_json_path, sx, sy, crop_w, crop_h, f"{stem}_t{tile_idx:04d}.png"
        )
        boxes: List[np.ndarray] = []
        class_ids: List[int] = []
        for sh in tiled_data.get("shapes", []):
            pts = sh.get("points", [])
            if len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                boxes.append(np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]))
                if label_to_id is not None:
                    lbl = str(sh.get("label", "defect"))
                    cid = int(label_to_id.get(lbl, 1))
                else:
                    cid = 1
                class_ids.append(cid)
        if not boxes:
            continue
        boxes = np.array(boxes)
        predictor.set_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # One predict per box, writing class ids into mask (multi-class GT)
        tile_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        for i in range(boxes.shape[0]):
            box = boxes[i : i + 1]  # (1, 4) XYXY
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            if masks is not None and len(masks) > 0:
                m = (masks[0] > 0).astype(np.uint8)
                if m.shape[0] != crop_h or m.shape[1] != crop_w:
                    m = cv2.resize(m, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                cls_id = class_ids[i] if i < len(class_ids) else 1
                cls_mask = (m > 0).astype(np.uint8) * cls_id
                # Overwrite with latest non-zero class pixels where applicable
                overwrite = cls_mask > 0
                tile_mask[overwrite] = cls_mask[overwrite]
        if tile_mask.any():
            overwrite = tile_mask > 0
            full_region = full_mask[sy : sy + crop_h, sx : sx + crop_w]
            full_region[overwrite] = tile_mask[overwrite]
            full_mask[sy : sy + crop_h, sx : sx + crop_w] = full_region
    return full_mask


def compare_masks_to_vis(
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    image_bgr: np.ndarray,
    alpha: float = 0.5,
    title: str | None = None,
    *,
    legend_gt_line: str = "GT = reference, Pred = segmentation",
) -> np.ndarray:
    """Create BGR visualization: TP=orange, FP=red, FN=blue. Overlay on image."""
    h, w = gt_binary.shape[:2]
    if pred_binary.shape[:2] != (h, w):
        pred_binary = cv2.resize(
            pred_binary.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
    gt_binary = (gt_binary > 0).astype(np.uint8)
    pred_binary = (pred_binary > 0).astype(np.uint8)
    tp = (gt_binary > 0) & (pred_binary > 0)
    fp = (pred_binary > 0) & (gt_binary == 0)
    fn = (gt_binary > 0) & (pred_binary == 0)
    out = image_bgr.astype(np.float32)
    for region, color in [(tp, COLORS["TP"]), (fp, COLORS["FP"]), (fn, COLORS["FN"])]:
        if region.any():
            out[region] = out[region] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = 28
    cv2.rectangle(out, (10, 5), (280, y0 + 82), (40, 40, 40), -1)
    cv2.putText(out, "TP (both): orange", (20, y0), font, 0.5, COLORS["TP"], 1, cv2.LINE_AA)
    cv2.putText(out, "FP (pred only): red", (20, y0 + 22), font, 0.5, COLORS["FP"], 1, cv2.LINE_AA)
    cv2.putText(out, "FN (GT only): blue", (20, y0 + 44), font, 0.5, COLORS["FN"], 1, cv2.LINE_AA)
    cv2.putText(
        out,
        legend_gt_line,
        (20, y0 + 66),
        font,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # Optional title with class name or context at top-left
    if title:
        cv2.putText(
            out,
            title,
            (10, 18),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def compute_mask_metrics(gt_binary: np.ndarray, pred_binary: np.ndarray) -> dict:
    """Pixel-level IoU, Dice, precision, recall, F1 for binary GT vs prediction."""
    gt_binary = (gt_binary > 0).astype(np.uint8)
    pred_binary = (pred_binary > 0).astype(np.uint8)
    inter = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    gt_sum = gt_binary.sum()
    pred_sum = pred_binary.sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
    prec = inter / pred_sum if pred_sum > 0 else 0.0
    rec = inter / gt_sum if gt_sum > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f2": float(f2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare segmentation model predictions vs reference masks (SAM2, LabelMe polygons, or PNG GT)."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with images and LabelMe JSONs (default: Consensus_Mask_Reviewer_Test)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compare_seg_vs_gt_out"),
        help="Output directory for masks and visualizations",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint (.pth) for tiled infer (default from config if run via run.py)",
    )
    p.add_argument(
        "--deeplab-masks-dir",
        "--pred-masks-dir",
        type=Path,
        default=None,
        dest="deeplab_masks_dir",
        help="Use existing segmentation prediction masks ({stem}.png). If set, infer is not run.",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size (default: 1024)",
    )
    p.add_argument(
        "--tile-overlap",
        type=int,
        default=256,
        help="Tile overlap (default: 256)",
    )
    p.add_argument(
        "--sam2-command",
        type=str,
        default=None,
        help='Command to run SAM2 per tile: use %%s %%s %%s for tile_image_path, tile_json_path, output_mask_path. E.g. "python src/utils/run_sam2_tile.py --image %%s --boxes %%s --output %%s"',
    )
    p.add_argument(
        "--skip-sam2",
        action="store_true",
        help="Skip SAM2; run segmentation infer only and save prediction masks (no GT comparison unless file GT is set).",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Path to save per-image and summary metrics JSON",
    )
    p.add_argument(
        "--compare-with-predictions",
        type=Path,
        default=None,
        help="If set, compare main prediction mask with this dir's {stem}.png; saves {stem}_deeplab_vs_predictions.png",
    )
    p.add_argument(
        "--refine",
        type=str,
        choices=["kmeans", "threshold"],
        default=None,
        help="Apply refinement (kmeans or threshold) to SAM2 predictions",
    )
    p.add_argument(
        "--refine-config",
        type=Path,
        default=None,
        help="Optional path to refinement config YAML (e.g. configs/stage/refinement.yaml) for cv/intensity/kmeans. "
             "When set, intensity/threshold/kmeans settings are taken from its 'cv' section.",
    )
    p.add_argument(
        "--sam2-masks-dir",
        type=Path,
        default=None,
        help="Directory to save/load SAM2 GT masks. If masks exist here, they are loaded instead of running SAM2.",
    )
    p.add_argument(
        "--gt-masks-dir",
        type=Path,
        default=None,
        dest="gt_masks_dir",
        help="With --skip-sam2 and gt_source=png: directory of {stem}.png GT label masks (grayscale class ids).",
    )
    return p.parse_args()


def run_compare(args, deeplab_config=None) -> None:
    """Run full comparison: segmentation predictions vs SAM2, rasterized LabelMe polygons, or PNG GT."""
    setup_logging()
    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    data_dir = data_dir.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_image_labelme_pairs(data_dir)
    if not pairs:
        logging.error("No (image, labelme json) pairs found in %s", data_dir)
        sys.exit(1)
    logging.info("Found %d image+json pairs in %s", len(pairs), data_dir)

    cfg = deeplab_config if deeplab_config is not None else get_default_config()
    _resolve_paths(cfg)

    # Optional refine config (YAML) shared with refine_sam2_masks.py
    cv_cfg: dict | None = None
    if args.refine and args.refine_config is not None:
        refine_cfg_path = args.refine_config if args.refine_config.is_absolute() else PROJECT_ROOT / args.refine_config
        refine_cfg_path = refine_cfg_path.resolve()
        if not refine_cfg_path.exists():
            logging.error("Refine config not found: %s", refine_cfg_path)
            sys.exit(1)
        try:
            with open(refine_cfg_path, "r") as f:
                full_cfg = yaml.safe_load(f) or {}
            cv_cfg = full_cfg.get("cv", {}) or {}
            logging.info("Loaded refine config from %s", refine_cfg_path)
        except Exception as e:
            logging.error("Failed to load refine config %s: %s", refine_cfg_path, e)
            sys.exit(1)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        logging.error("Checkpoint path is not set. Please set stage.checkpoint to the desired model path.")
        sys.exit(1)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
        
    tiled_inference = getattr(cfg, "tiled_inference", False)

    # Load label mapping (for per-class metrics and SAM2 class ids)
    id_to_label: Dict[int, str] | None = None
    label_to_id: Dict[str, int] | None = None
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        lm = ckpt.get("label_mapping", {})
        if isinstance(lm, dict):
            id_to_label = lm.get("id_to_label") or lm.get("id2label")
            label_to_id = lm.get("label_to_id") or lm.get("label2id")
    except Exception as e:
        logging.warning("Could not load label_mapping from checkpoint: %s", e)

    if (id_to_label is None or label_to_id is None) and getattr(cfg, "label_mapping_path", None):
        lm_path = Path(cfg.label_mapping_path)
        if not lm_path.is_absolute():
            lm_path = PROJECT_ROOT / lm_path
        if lm_path.exists():
            try:
                with lm_path.open("r") as f:
                    lm = json.load(f)
                if isinstance(lm, dict):
                    id_to_label = lm.get("id_to_label") or id_to_label
                    label_to_id = lm.get("label_to_id") or label_to_id
            except Exception as e:
                logging.warning("Failed to load label_mapping from %s: %s", lm_path, e)
    
    # Segmentation prediction masks: existing folder or run infer once (tiled)
    if args.deeplab_masks_dir is not None:
        deeplab_masks_dir = args.deeplab_masks_dir if args.deeplab_masks_dir.is_absolute() else PROJECT_ROOT / args.deeplab_masks_dir
        deeplab_masks_dir = deeplab_masks_dir.resolve()
        if not deeplab_masks_dir.is_dir():
            logging.error("pred/deeplab masks dir is not a directory: %s", deeplab_masks_dir)
            sys.exit(1)
        logging.info("Using existing segmentation prediction masks from %s", deeplab_masks_dir)
    else:
        if not checkpoint_path.exists():
            logging.error("Checkpoint not found: %s", checkpoint_path)
            sys.exit(1)
        crop_suffix = "_cropped" if getattr(cfg, "crop_object_enabled", False) else ""
        deeplab_masks_dir = output_dir / f"deeplab_masks{crop_suffix}"
        deeplab_masks_dir.mkdir(parents=True, exist_ok=True)
        if not run_infer_py_subprocess(
            data_dir, deeplab_masks_dir, checkpoint_path, args.tile_size, args.tile_overlap, tiled=tiled_inference, cfg=cfg
        ):
            sys.exit(1)
        logging.info("Segmentation prediction masks written to %s (infer)", deeplab_masks_dir)

    run_sam2 = not args.skip_sam2
    use_sam2_command = bool(args.sam2_command)

    gt_masks_dir: Path | None = getattr(args, "gt_masks_dir", None)
    if gt_masks_dir is not None:
        gt_masks_dir = gt_masks_dir if gt_masks_dir.is_absolute() else PROJECT_ROOT / gt_masks_dir
        gt_masks_dir = gt_masks_dir.resolve()

    gt_source_raw = getattr(args, "gt_source", None)
    if gt_source_raw is None or str(gt_source_raw).strip() == "":
        gt_source = "png" if (gt_masks_dir is not None and gt_masks_dir.is_dir()) else "labelme"
    else:
        gt_source = str(gt_source_raw).strip().lower()
    if gt_source not in ("labelme", "png"):
        logging.error("gt_source must be 'labelme' or 'png', got %r", gt_source_raw)
        sys.exit(1)

    if not run_sam2:
        if gt_source == "png":
            if gt_masks_dir is None or not gt_masks_dir.is_dir():
                logging.error(
                    "use_sam2=false and gt_source=png requires gt_masks_dir to be an existing directory."
                )
                sys.exit(1)
        else:
            logging.info("GT from LabelMe JSON: polygon shapes rasterized to class masks (paired with images).")

    label_to_id_gt: Dict[str, int] = {}
    if label_to_id:
        for k, v in label_to_id.items():
            try:
                label_to_id_gt[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    gt_aliases = getattr(args, "gt_label_to_class_id", None)
    if gt_aliases is not None:
        from omegaconf import OmegaConf

        try:
            raw = OmegaConf.to_container(gt_aliases, resolve=True)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        label_to_id_gt[str(k)] = int(v)
                    except (TypeError, ValueError):
                        continue
                logging.info("Merged gt_label_to_class_id into GT label lookup (%d keys).", len(raw))
        except Exception as e:
            logging.warning("Could not parse gt_label_to_class_id: %s", e)

    # Set up SAM2 cache directory (used when run_sam2; harmless when only file GT)
    sam2_masks_dir = args.sam2_masks_dir
    if sam2_masks_dir is None:
        refine_suffix = f"_{args.refine}" if args.refine else ""
        crop_suffix = "_cropped" if getattr(cfg, "crop_object_enabled", False) else ""
        sam2_masks_dir = output_dir / f"sam2_masks{refine_suffix}{crop_suffix}"
    else:
        sam2_masks_dir = sam2_masks_dir if sam2_masks_dir.is_absolute() else PROJECT_ROOT / sam2_masks_dir
    sam2_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # We will only try to init SAM2 if there are actually missing masks we need to compute
    need_to_run_sam2 = False
    if run_sam2:
        for img_path, _ in pairs:
            sam2_out = sam2_masks_dir / f"{img_path.stem}_sam2.png"
            if not sam2_out.exists():
                need_to_run_sam2 = True
                break
                
    if run_sam2 and need_to_run_sam2 and not use_sam2_command:
        # Try in-process SAM2 on first image
        img0 = cv2.imread(str(pairs[0][0]))
        if img0 is not None:
            sam2_mask = run_sam2_tiled_inprocess(
                img0,
                pairs[0][1],
                args.tile_size,
                args.tile_overlap,
                output_dir,
                pairs[0][0].stem,
                tiled=tiled_inference,
            )
            if sam2_mask is None:
                logging.warning(
                    "SAM2 package not available and --sam2-command not set. Use --skip-sam2 for file/LabelMe GT, or set --sam2-command."
                )
                run_sam2 = False
        else:
            run_sam2 = False

    all_metrics = []
    per_class_metrics: List[dict] = []
    for img_path, json_path in pairs:
        stem = img_path.stem
        orig_labelme_path = json_path
        logging.info("Processing %s ...", stem)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            logging.warning("Skip %s: failed to read image", img_path)
            continue

        if str(PROJECT_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

        # Match prepare/infer: full-image CLAHE on BGR before white-object crop.
        if getattr(cfg, "clahe_enabled", False):
            clip = float(getattr(cfg, "clahe_clip_limit", 2.0))
            tg = getattr(cfg, "clahe_tile_grid", (8, 8))
            grid = (
                (int(tg[0]), int(tg[1]))
                if isinstance(tg, (list, tuple)) and len(tg) >= 2
                else (8, 8)
            )
            img_bgr = apply_clahe_bgr(img_bgr, clip, grid)

        full_h, full_w = img_bgr.shape[:2]
        crop_enabled = bool(getattr(cfg, "crop_object_enabled", False))
        padding = int(getattr(cfg, "crop_object_padding", 0))
        crop_box: Optional[Tuple[int, int, int, int]] = None
        gt_raster_pre_crop: Optional[np.ndarray] = None

        if (not run_sam2) and gt_source == "labelme":
            gt_raster_pre_crop = mask_from_labelme(
                orig_labelme_path, label_to_id_gt, full_h, full_w
            )
            gr, gc = gt_raster_pre_crop.shape[:2]
            if (gr, gc) != (full_h, full_w):
                gt_raster_pre_crop = cv2.resize(
                    gt_raster_pre_crop, (full_w, full_h), interpolation=cv2.INTER_NEAREST
                )

        json_path = orig_labelme_path
        if crop_enabled:
            from utils.image_utils import get_object_crop_bbox, shift_labelme_json

            cx1, cy1, cx2, cy2 = get_object_crop_bbox(img_bgr, padding=padding)
            crop_box = (cx1, cy1, cx2, cy2)
            img_bgr = img_bgr[cy1:cy2, cx1:cx2]
            if gt_raster_pre_crop is not None:
                gt_raster_pre_crop = gt_raster_pre_crop[cy1:cy2, cx1:cx2]
            if run_sam2:
                crop_h, crop_w = img_bgr.shape[0], img_bgr.shape[1]
                temp_json_dir = output_dir / "temp_jsons"
                temp_json_dir.mkdir(parents=True, exist_ok=True)
                new_json_path = temp_json_dir / f"{stem}.json"
                shift_labelme_json(
                    orig_labelme_path,
                    cx1,
                    cy1,
                    new_json_path,
                    image_width=crop_w,
                    image_height=crop_h,
                )
                json_path = new_json_path

        orig_h, orig_w = img_bgr.shape[:2]

        # Prediction mask from infer or --pred-masks-dir / --deeplab-masks-dir
        deeplab_mask_path = deeplab_masks_dir / f"{stem}.png"
        deeplab_mask = cv2.imread(str(deeplab_mask_path), cv2.IMREAD_GRAYSCALE)
        if deeplab_mask is None:
            logging.warning("Segmentation prediction mask not found: %s", deeplab_mask_path)
            deeplab_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        elif deeplab_mask.shape[:2] != (orig_h, orig_w):
            deeplab_mask = cv2.resize(deeplab_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        work_dir = output_dir / "tiles" / stem
        cv2.imwrite(str(output_dir / f"{stem}_deeplab.png"), deeplab_mask)

        # Optional: compare primary prediction mask with a second predictions directory
        if args.compare_with_predictions is not None:
            pred_dir = args.compare_with_predictions if args.compare_with_predictions.is_absolute() else PROJECT_ROOT / args.compare_with_predictions
            pred_dir = pred_dir.resolve()
            pred_path = pred_dir / f"{stem}.png"
            if pred_path.exists():
                pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                if pred_mask is not None:
                    if pred_mask.shape[:2] != (orig_h, orig_w):
                        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    a = (deeplab_mask > 0).astype(np.uint8)
                    b = (pred_mask > 0).astype(np.uint8)
                    same = (a == b).sum()
                    total = a.size
                    agree_pct = 100.0 * same / total if total else 0
                    diff_px = (a != b).sum()
                    logging.info(
                        "  [vs predictions] %s: agreement=%.2f%% (%d / %d pixels), diff_px=%d",
                        stem, agree_pct, int(same), total, int(diff_px),
                    )
                    # Save diff vis: green=both FG, red=script FG / pred BG, blue=script BG / pred FG
                    diff_vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    both_fg = (a > 0) & (b > 0)
                    only_script = (a > 0) & (b == 0)
                    only_pred = (a == 0) & (b > 0)
                    diff_vis[both_fg] = [0, 255, 0]
                    diff_vis[only_script] = [0, 0, 255]
                    diff_vis[only_pred] = [255, 0, 0]
                    cv2.imwrite(str(output_dir / f"{stem}_deeplab_vs_predictions.png"), diff_vis)
                else:
                    logging.warning("  [vs predictions] %s: failed to read %s", stem, pred_path)
            else:
                logging.warning("  [vs predictions] %s: not found %s", stem, pred_path)

        gt_ref_mask: np.ndarray | None = None
        legend_line = "GT = SAM2, Pred = segmentation"
        sam2_out = sam2_masks_dir / f"{stem}_sam2.png"

        if run_sam2:
            work_dir.mkdir(parents=True, exist_ok=True)
            if sam2_out.exists():
                logging.info("  Loading cached SAM2 mask from %s", sam2_out)
                gt_ref_mask = cv2.imread(str(sam2_out), cv2.IMREAD_GRAYSCALE)
            else:
                logging.info("  Generating SAM2 mask...")
                if use_sam2_command and args.sam2_command:
                    gt_ref_mask = run_sam2_tiled_with_command(
                        img_bgr,
                        json_path,
                        args.tile_size,
                        args.tile_overlap,
                        args.sam2_command,
                        work_dir,
                        stem,
                        tiled=tiled_inference,
                    )
                else:
                    gt_ref_mask = run_sam2_tiled_inprocess(
                        img_bgr,
                        json_path,
                        args.tile_size,
                        args.tile_overlap,
                        work_dir,
                        stem,
                        tiled=tiled_inference,
                        label_to_id=label_to_id,
                    )
                if gt_ref_mask is not None:
                    if args.refine:
                        orig_sam2_mask = gt_ref_mask.copy()
                        gt_ref_mask = _refine_mask(gt_ref_mask, img_bgr, args.refine, cv_cfg=cv_cfg)
                        gt_ref_mask = _ensure_bbox_coverage(orig_sam2_mask, gt_ref_mask, json_path)
                    sam2_save = np.clip(gt_ref_mask.astype(np.int32), 0, 255).astype(np.uint8)
                    cv2.imwrite(str(sam2_out), sam2_save)
        elif not run_sam2 and gt_source == "labelme":
            legend_line = "GT = LabelMe polygons, Pred = segmentation"
            gt_ref_mask = gt_raster_pre_crop
            if gt_ref_mask is None:
                gt_ref_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if gt_ref_mask is not None and (gt_ref_mask > 0).any():
                logging.info("  Rasterized GT mask from LabelMe %s (full raster + infer-style crop)", orig_labelme_path.name)
            else:
                logging.warning(
                    "  LabelMe GT mask empty for %s (check shape labels vs label_mapping / gt_label_to_class_id)",
                    stem,
                )
            cv2.imwrite(str(output_dir / f"{stem}_gt_mask.png"), gt_ref_mask)
        elif not run_sam2 and gt_source == "png" and gt_masks_dir is not None:
            gt_path = gt_masks_dir / f"{stem}.png"
            legend_line = "GT = PNG mask, Pred = segmentation"
            if gt_path.exists():
                gt_ref_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt_ref_mask is not None:
                    gh, gw = gt_ref_mask.shape[:2]
                    if (gh, gw) == (orig_h, orig_w):
                        pass
                    elif crop_box is not None and (gh, gw) == (full_h, full_w):
                        cx1, cy1, cx2, cy2 = crop_box
                        gt_ref_mask = gt_ref_mask[cy1:cy2, cx1:cx2]
                    else:
                        gt_ref_mask = cv2.resize(
                            gt_ref_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                        )
                    logging.info("  Loaded GT mask from %s", gt_path)
                    cv2.imwrite(str(output_dir / f"{stem}_gt_mask.png"), gt_ref_mask)
                else:
                    logging.warning("  Failed to read GT mask: %s", gt_path)
            else:
                logging.warning("  GT mask not found: %s", gt_path)

        if gt_ref_mask is not None:
            # Pixel compare: GT reference vs segmentation prediction
            metrics = compute_mask_metrics(
                (gt_ref_mask > 0).astype(np.uint8),
                (deeplab_mask > 0).astype(np.uint8),
            )
            metrics["image"] = stem
            all_metrics.append(metrics)
            deeplab_fg = (deeplab_mask > 0).sum()
            gt_fg = (gt_ref_mask > 0).sum()
            vis = compare_masks_to_vis(
                gt_ref_mask,
                deeplab_mask,
                img_bgr,
                title="All defects (binary)",
                legend_gt_line=legend_line,
            )
            vis_path = output_dir / f"{stem}_compare.jpg"
            cv2.imwrite(str(vis_path), vis)
            logging.info(
                "  %s pred_fg=%d GT_fg=%d | IoU=%.4f Dice=%.4f Prec=%.4f Rec=%.4f F1=%.4f",
                stem, int(deeplab_fg), int(gt_fg),
                metrics["iou"], metrics["dice"], metrics["precision"], metrics["recall"], metrics["f1"],
            )
            if getattr(args, "per_class_metrics", False) and id_to_label:
                for cls_id, cls_name in id_to_label.items():
                    try:
                        cid = int(cls_id)
                    except Exception:
                        continue
                    if cid == 0:
                        continue
                    gt_c = (gt_ref_mask == cid).astype(np.uint8)
                    pred_c = (deeplab_mask == cid).astype(np.uint8)
                    if gt_c.sum() == 0 and pred_c.sum() == 0:
                        continue
                    m_c = compute_mask_metrics(gt_c, pred_c)
                    m_c.update(
                        {
                            "image": stem,
                            "class_id": int(cid),
                            "class_name": str(cls_name),
                        }
                    )
                    per_class_metrics.append(m_c)
                    logging.info(
                        "  [class %s] IoU=%.4f Dice=%.4f Prec=%.4f Rec=%.4f F1=%.4f",
                        cls_name,
                        m_c["iou"],
                        m_c["dice"],
                        m_c["precision"],
                        m_c["recall"],
                        m_c["f1"],
                    )
                    cls_vis = compare_masks_to_vis(
                        gt_c,
                        pred_c,
                        img_bgr,
                        title=f"Class: {cls_name}",
                        legend_gt_line=legend_line,
                    )
                    safe_label = _sanitize_label_for_filename(str(cls_name))
                    cls_vis_path = output_dir / f"{stem}_class_{safe_label}_compare.jpg"
                    cv2.imwrite(str(cls_vis_path), cls_vis)

    if args.save_json and all_metrics:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "iou_mean": float(np.mean([m["iou"] for m in all_metrics])),
            "dice_mean": float(np.mean([m["dice"] for m in all_metrics])),
            "precision_mean": float(np.mean([m["precision"] for m in all_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in all_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in all_metrics])),
            "f2_mean": float(np.mean([m["f2"] for m in all_metrics])),
        }
        payload = {"summary": summary, "per_image": all_metrics}
        # Optional per-class metrics section
        if per_class_metrics:
            payload["per_class"] = per_class_metrics
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        logging.info("Saved metrics JSON to %s", json_path)

        # Also write CSV with per-image metrics and a final summary row
        csv_path = json_path.with_suffix(".csv")
        fieldnames = ["image", "iou", "dice", "precision", "recall", "f1", "f2"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                row = {k: m.get(k) for k in fieldnames}
                writer.writerow(row)
            # Summary row
            writer.writerow({
                "image": "__mean__",
                "iou": summary["iou_mean"],
                "dice": summary["dice_mean"],
                "precision": summary["precision_mean"],
                "recall": summary["recall_mean"],
                "f1": summary["f1_mean"],
                "f2": summary["f2_mean"],
            })
        logging.info("Saved metrics CSV to %s", csv_path)

        # Optional per-class CSV
        if per_class_metrics:
            csv_cls_path = json_path.with_name(json_path.stem + "_per_class.csv")
            cls_fields = ["image", "class_id", "class_name", "iou", "dice", "precision", "recall", "f1", "f2"]
            with csv_cls_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cls_fields)
                writer.writeheader()
                for m in per_class_metrics:
                    row = {k: m.get(k) for k in cls_fields}
                    writer.writerow(row)
            logging.info("Saved per-class metrics CSV to %s", csv_cls_path)
    logging.info("Done. Outputs in %s", output_dir)



def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p

def run(cfg) -> None:
    from argparse import Namespace
    from omegaconf import DictConfig
    root = _resolve_path(cfg.paths.root, Path.cwd())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    stage = cfg.stage
    data_dir = _resolve_path(stage.get("data_dir", "dataset/Consensus_Mask_Reviewer_Test"), root)
    output_dir = _resolve_path(stage.get("output_dir", "outputs/stage6_compare"), root)
    checkpoint = _resolve_path(stage.get("checkpoint", "checkpoints/best_model.pth"), root)
    sam2_masks_dir = stage.get("sam2_masks_dir")
    if sam2_masks_dir is not None:
        sam2_masks_dir = _resolve_path(sam2_masks_dir, root)
    use_sam2 = bool(stage.get("use_sam2", True))
    gt_masks_dir = stage.get("gt_masks_dir")
    if gt_masks_dir is not None:
        gt_masks_dir = _resolve_path(gt_masks_dir, root)
    refine = stage.get("refine")
    refine_config = stage.get("refine_config")
    if refine_config is not None:
        refine_config = _resolve_path(refine_config, root)
    save_json = stage.get("save_json")
    if save_json is not None:
        save_json = _resolve_path(save_json, root)
    tile_size = int(stage.get("tile_size", 1024))
    tile_overlap = int(stage.get("tile_overlap", 256))
    per_class_metrics = bool(stage.get("per_class_metrics", False))
    args = Namespace(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint=checkpoint,
        sam2_masks_dir=sam2_masks_dir,
        refine=refine,
        refine_config=refine_config,
        save_json=save_json,
        skip_sam2=not use_sam2,
        sam2_command=None,
        deeplab_masks_dir=None,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        compare_with_predictions=None,
        per_class_metrics=per_class_metrics,
        gt_masks_dir=gt_masks_dir,
        gt_source=stage.get("gt_source"),
        gt_label_to_class_id=stage.get("gt_label_to_class_id"),
    )
    from config import get_config_from_stage
    deeplab_config = get_config_from_stage(cfg.stage)
    deeplab_config.dataset_root = _resolve_path(deeplab_config.dataset_root, root)
    run_compare(args, deeplab_config=deeplab_config)
