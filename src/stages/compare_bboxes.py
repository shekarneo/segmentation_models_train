#!/usr/bin/env python3
"""
Stage 7: DeepLabV3+ mask predictions -> class-labeled bboxes.

Workflow (bbox-level):
  1. Discover (image, LabelMe JSON) pairs in data_dir.
  2. Get DeepLab masks for each image (either from deeplab_masks_dir or by running infer.py logic).
  3. Convert each class mask to instance bboxes (connected components).
     - Optional mask dilation before extracting bboxes.
  4. Extract GT bboxes from LabelMe JSON (rectangle/polygon); optionally merge overlapping GT boxes
     of the same class into one union box, then compare with predictions.
  5. Optionally export predicted bboxes back to LabelMe-compatible JSONs.

Notes:
  - This stage does NOT use SAM2.
  - If crop_object_enabled is true, comparisons are done in cropped coordinates, but exports are in
    original-image coordinates (so LabelMe viewers can use them with the original images).
  - When clahe_enabled matches prepare/infer, full-image CLAHE on BGR runs before the white-object crop.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


_STAGES_DIR = Path(__file__).resolve().parent
_SRC_DIR = _STAGES_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config, get_default_config, get_config_from_stage
from utils.image_utils import apply_clahe_bgr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p


def discover_image_labelme_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Return list of (image_path, labelme_json_path) with same stem (excluding *_meta.json)."""
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


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def _pred_fully_inside_gt(pred: Box, gt: Box) -> bool:
    """True if the prediction rectangle lies entirely inside the GT rectangle."""
    return (
        gt.x1 <= pred.x1
        and gt.y1 <= pred.y1
        and pred.x2 <= gt.x2
        and pred.y2 <= gt.y2
    )


def match_boxes(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    iou_thresh: float,
) -> Tuple[int, int, int, set[int], set[int], Dict[int, Tuple[int, float]], Dict[int, float]]:
    """
    Greedy 1-to-1 matching:
      - Each GT picks the unmatched predicted box with the best IoU.
      - A match is TP if best IoU >= threshold.
    """
    tp = 0
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    gt_to_pred: Dict[int, Tuple[int, float]] = {}
    pred_to_best_iou: Dict[int, float] = {}

    for pi in range(len(pred_boxes)):
        pred_to_best_iou[pi] = max((iou(gt_boxes[gi], pred_boxes[pi]) for gi in range(len(gt_boxes))), default=0.0)

    for gi, g in enumerate(gt_boxes):
        best_iou_val = 0.0
        best_pi = -1
        for pi, p in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            v = iou(g, p)
            if v > best_iou_val:
                best_iou_val = v
                best_pi = pi
        if best_pi >= 0 and best_iou_val >= iou_thresh:
            tp += 1
            matched_pred.add(best_pi)
            matched_gt.add(gi)
            gt_to_pred[gi] = (best_pi, best_iou_val)

    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_gt, matched_pred, gt_to_pred, pred_to_best_iou


def _bbox_from_labelme_shape(shape: dict, *, gt_format: str) -> Optional[Box]:
    """Convert LabelMe shape to axis-aligned bbox (x1,y1,x2,y2) based on gt_format.

    gt_format values:
      - "bbox"    : only rectangle shapes
      - "polygon" : only polygon shapes
      - "all"     : both rectangle and polygon shapes
    """
    pts = shape.get("points", [])
    stype = shape.get("shape_type", "rectangle")

    if gt_format == "all":
        if stype == "rectangle" and len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
        elif stype == "polygon" and len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            return None
    elif gt_format == "bbox":
        if stype != "rectangle" or len(pts) < 2:
            return None
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    elif gt_format == "polygon":
        if stype != "polygon" or len(pts) < 3:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    x1, x2 = min(float(x1), float(x2)), max(float(x1), float(x2))
    y1, y2 = min(float(y1), float(y2)), max(float(y1), float(y2))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return Box(x1, y1, x2, y2)


def _clip_box_to_image(b: Box, w: int, h: int) -> Box:
    x1 = max(0.0, min(float(b.x1), float(w)))
    y1 = max(0.0, min(float(b.y1), float(h)))
    x2 = max(0.0, min(float(b.x2), float(w)))
    y2 = max(0.0, min(float(b.y2), float(h)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return Box(0.0, 0.0, 0.0, 0.0)
    return Box(x1, y1, x2, y2)


def _load_label_mapping(
    checkpoint_path: Path,
    dataset_root: Optional[Path],
    label_mapping_path: Optional[Path],
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Return (id_to_label, label_to_id).
    Tries checkpoint label_mapping first, then explicit label_mapping_path,
    then dataset_root/label_mapping.json.
    """
    id_to_label: Dict[int, str] = {}
    label_to_id: Dict[str, int] = {}

    # 1) checkpoint
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        lm = ckpt.get("label_mapping", None)
        if isinstance(lm, dict):
            raw_id_to_label = lm.get("id_to_label") or lm.get("id2label")
            raw_label_to_id = lm.get("label_to_id") or lm.get("label2id")
            if isinstance(raw_id_to_label, dict) and isinstance(raw_label_to_id, dict):
                for k, v in raw_id_to_label.items():
                    try:
                        id_to_label[int(k)] = str(v)
                    except Exception:
                        continue
                for k, v in raw_label_to_id.items():
                    try:
                        label_to_id[str(k)] = int(v)
                    except Exception:
                        continue
    except Exception as e:
        logging.warning("Failed to load label_mapping from checkpoint: %s", e)

    # 2) explicit label_mapping_path
    if not id_to_label and label_mapping_path is not None:
        try:
            lm_path = label_mapping_path.resolve()
            with lm_path.open("r") as f:
                lm = json.load(f)
            raw_id_to_label = lm.get("id_to_label") or lm.get("id2label")
            raw_label_to_id = lm.get("label_to_id") or lm.get("label2id")
            if isinstance(raw_id_to_label, dict) and isinstance(raw_label_to_id, dict):
                for k, v in raw_id_to_label.items():
                    try:
                        id_to_label[int(k)] = str(v)
                    except Exception:
                        continue
                for k, v in raw_label_to_id.items():
                    try:
                        label_to_id[str(k)] = int(v)
                    except Exception:
                        continue
        except Exception as e:
            logging.warning("Failed to load label_mapping from %s: %s", label_mapping_path, e)

    # 3) dataset_root fallback
    if not id_to_label and dataset_root is not None:
        try:
            lm_path = dataset_root / "label_mapping.json"
            if lm_path.exists():
                with lm_path.open("r") as f:
                    lm = json.load(f)
                raw_id_to_label = lm.get("id_to_label") or lm.get("id2label")
                raw_label_to_id = lm.get("label_to_id") or lm.get("label2id")
                if isinstance(raw_id_to_label, dict) and isinstance(raw_label_to_id, dict):
                    for k, v in raw_id_to_label.items():
                        try:
                            id_to_label[int(k)] = str(v)
                        except Exception:
                            continue
                    for k, v in raw_label_to_id.items():
                        try:
                            label_to_id[str(k)] = int(v)
                        except Exception:
                            continue
        except Exception as e:
            logging.warning("Failed to load label_mapping from dataset_root: %s", e)

    if not id_to_label or not label_to_id:
        logging.warning("No label mapping found; will fall back to generic labels.")

    return id_to_label, label_to_id


def _get_crop_bbox(cfg: Config, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    crop_enabled = bool(getattr(cfg, "crop_object_enabled", False))
    if not crop_enabled:
        return None
    from utils.image_utils import get_object_crop_bbox

    padding = int(getattr(cfg, "crop_object_padding", 0))
    cx1, cy1, cx2, cy2 = get_object_crop_bbox(img_bgr, padding=padding)
    return int(cx1), int(cy1), int(cx2), int(cy2)


def _convert_mask_to_bboxes(
    mask: np.ndarray,
    class_id: int,
    *,
    dilate_iterations: int,
    dilate_kernel_size: int,
    min_component_area: int,
) -> List[Box]:
    """
    Convert a (H,W) class mask to instance bboxes via connected components.
    """
    bin_mask = (mask == class_id).astype(np.uint8)
    if dilate_iterations > 0:
        k = max(1, int(dilate_kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin_mask = cv2.dilate(bin_mask, kernel, iterations=int(dilate_iterations))

    num_labels, labels = cv2.connectedComponents(bin_mask)
    out: List[Box] = []
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        if min_component_area > 0 and int(ys.size) < int(min_component_area):
            continue
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append(Box(x1, y1, x2, y2))
    return out


def _bbox_area(box: Box) -> float:
    w = max(0.0, box.x2 - box.x1)
    h = max(0.0, box.y2 - box.y1)
    return w * h


def _filter_boxes_by_size(
    boxes: Sequence[Box],
    *,
    min_area: Optional[float] = None,
    min_width: Optional[float] = None,
    min_height: Optional[float] = None,
) -> List[Box]:
    """Filter out boxes smaller than given thresholds."""
    if min_area is None and min_width is None and min_height is None:
        return list(boxes)

    kept: List[Box] = []
    for b in boxes:
        w = b.x2 - b.x1
        h = b.y2 - b.y1
        area = _bbox_area(b)
        if min_area is not None and area < float(min_area):
            continue
        if min_width is not None and w < float(min_width):
            continue
        if min_height is not None and h < float(min_height):
            continue
        kept.append(b)
    return kept


def _box_union(boxes: Sequence[Box]) -> Box:
    """Return union box for a non-empty list of boxes."""
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _merge_overlapping_gt_boxes(boxes: List[Box], iou_threshold: float) -> List[Box]:
    """
    Cluster GT boxes of the same class whose pairwise IoU is strictly above iou_threshold,
    then replace each cluster with the axis-aligned union box.

    Use iou_threshold=0.0 to merge any pair with positive overlap (including one box inside another).
    """
    if len(boxes) <= 1:
        return list(boxes)
    n = len(boxes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if iou(boxes[i], boxes[j]) > iou_threshold:
                union(i, j)

    clusters: Dict[int, List[Box]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(boxes[i])
    return [_box_union(group) for group in clusters.values()]


def _merge_gt_by_proximity(
    boxes: List[Box],
    max_distance: float,
) -> List[Box]:
    """Merge GT boxes whose edges are within max_distance pixels of each other.

    Unlike IoU-based merging, this catches nearby small boxes separated by a gap
    (common with clusters of small scratches that the model predicts as one blob).
    """
    if len(boxes) <= 1 or max_distance < 0:
        return list(boxes)
    n = len(boxes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            # Edge-to-edge distance: expand box_i by max_distance, check overlap with box_j.
            a = boxes[i]
            b = boxes[j]
            if (
                a.x1 - max_distance <= b.x2
                and a.x2 + max_distance >= b.x1
                and a.y1 - max_distance <= b.y2
                and a.y2 + max_distance >= b.y1
            ):
                union(i, j)

    clusters: Dict[int, List[Box]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(boxes[i])
    return [_box_union(group) for group in clusters.values()]


def _merge_predictions_inside_gt(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    *,
    allow_pred_multi_gt: bool = False,
) -> Tuple[List[Optional[Box]], List[Box]]:
    """
    For each GT, merge (union) all predictions assigned to that GT into one box.

    Assignment (so many small preds inside one *large* GT count together):
      1) If the pred is fully inside one or more GT boxes, assign to the
         **smallest-area** GT (or ALL containing GTs when allow_pred_multi_gt).
      2) Otherwise assign to the GT with maximum IoU (or ALL GTs with IoU > 0
         when allow_pred_multi_gt).
      3) Otherwise the pred is an orphan (FP).

    When allow_pred_multi_gt=True a single large prediction can contribute to
    matching several GTs (solves "one big pred covers many small GTs" scenario).
    """
    if not gt_boxes:
        return [], list(pred_boxes)
    if not pred_boxes:
        return [None] * len(gt_boxes), []

    gt_to_preds: Dict[int, List[Box]] = {i: [] for i in range(len(gt_boxes))}
    orphan_preds: List[Box] = []

    for p_box in pred_boxes:
        assigned = False
        # Step 1: check containment
        inside: List[Tuple[int, float]] = []
        for g_idx, g_box in enumerate(gt_boxes):
            if _pred_fully_inside_gt(p_box, g_box):
                inside.append((g_idx, g_box.area()))
        if inside:
            if allow_pred_multi_gt:
                for g_idx, _ in inside:
                    gt_to_preds[g_idx].append(p_box)
            else:
                best_gt_idx = min(inside, key=lambda t: t[1])[0]
                gt_to_preds[best_gt_idx].append(p_box)
            assigned = True

        if not assigned:
            # Step 2: IoU-based assignment
            iou_hits: List[Tuple[int, float]] = []
            for g_idx, g_box in enumerate(gt_boxes):
                v = iou(g_box, p_box)
                if v > 0:
                    iou_hits.append((g_idx, v))
            if iou_hits:
                if allow_pred_multi_gt:
                    for g_idx, _ in iou_hits:
                        gt_to_preds[g_idx].append(p_box)
                else:
                    best_gt_idx = max(iou_hits, key=lambda t: t[1])[0]
                    gt_to_preds[best_gt_idx].append(p_box)
                assigned = True

        if not assigned:
            # Step 3: check if pred *contains* any GT (large pred over small GT)
            if allow_pred_multi_gt:
                for g_idx, g_box in enumerate(gt_boxes):
                    if _pred_fully_inside_gt(g_box, p_box):
                        gt_to_preds[g_idx].append(p_box)
                        assigned = True

        if not assigned:
            orphan_preds.append(p_box)

    merged_for_gt: List[Optional[Box]] = []
    for g_idx in range(len(gt_boxes)):
        preds_inside = gt_to_preds.get(g_idx, [])
        if preds_inside:
            merged_for_gt.append(_box_union(preds_inside))
        else:
            merged_for_gt.append(None)
    return merged_for_gt, orphan_preds


def _run_infer_if_needed(
    *,
    data_dir: Path,
    deeplab_masks_dir: Path,
    checkpoint_path: Path,
    cfg_for_infer: Config,
    tiled_inference: bool,
    tile_size: int,
    tile_overlap: int,
) -> None:
    """
    Run the existing infer.py logic to generate {stem}.png masks into deeplab_masks_dir.
    """
    if deeplab_masks_dir.is_dir() and any(deeplab_masks_dir.glob("*.png")):
        logging.info("Using existing DeepLab masks from %s", deeplab_masks_dir)
        return

    from src.stages.infer import run_inference_stage

    deeplab_masks_dir.mkdir(parents=True, exist_ok=True)

    # Ensure inference uses the same tiling/crop configuration.
    cfg_for_infer.tiled_inference = tiled_inference
    cfg_for_infer.tile_size = tile_size
    cfg_for_infer.tile_overlap = tile_overlap

    run_inference_stage(
        cfg_for_infer,
        input_dir=data_dir,
        output_dir=deeplab_masks_dir,
        checkpoint_path=checkpoint_path,
        tiled=tiled_inference,
    )


def _load_pred_mask(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read DeepLab mask: {mask_path}")
    return m


def _load_gt_boxes_by_class(
    json_path: Path,
    *,
    gt_format: str,
    label_to_id: Dict[str, int],
    crop_box: Optional[Tuple[int, int, int, int]],
    mask_w: int,
    mask_h: int,
) -> Dict[int, List[Box]]:
    with json_path.open("r") as f:
        data = json.load(f)

    cx1 = cy1 = 0
    if crop_box is not None:
        cx1, cy1, _, _ = crop_box

    out: Dict[int, List[Box]] = {}
    for shape in data.get("shapes", []):
        lbl = shape.get("label", "")
        cid = label_to_id.get(str(lbl))
        if cid is None:
            # Unknown label: skip
            continue
        b = _bbox_from_labelme_shape(shape, gt_format=gt_format)
        if b is None:
            continue
        # Shift into crop/mask coordinates if needed
        b2 = Box(b.x1 - float(cx1), b.y1 - float(cy1), b.x2 - float(cx1), b.y2 - float(cy1))
        b2 = _clip_box_to_image(b2, mask_w, mask_h)
        if b2.x2 - b2.x1 <= 0 or b2.y2 - b2.y1 <= 0:
            continue
        out.setdefault(int(cid), []).append(b2)
    return out


def _save_predicted_labelme_jsons(
    *,
    json_path: Path,
    out_json_path: Path,
    predicted_boxes_by_class: Dict[int, List[Box]],
    id_to_label: Dict[int, str],
    crop_box: Optional[Tuple[int, int, int, int]],
) -> None:
    with json_path.open("r") as f:
        data = json.load(f)

    cx1 = cy1 = 0
    if crop_box is not None:
        cx1, cy1, _, _ = crop_box

    predicted_shapes: List[dict] = []
    for cid, boxes in predicted_boxes_by_class.items():
        label = id_to_label.get(int(cid), str(cid))
        for b in boxes:
            # Convert to LabelMe rectangle format: two corner points.
            x1 = float(b.x1 + float(cx1))
            y1 = float(b.y1 + float(cy1))
            x2 = float(b.x2 + float(cx1))
            y2 = float(b.y2 + float(cy1))
            predicted_shapes.append(
                {
                    "label": label,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            )

    data["shapes"] = predicted_shapes
    with out_json_path.open("w") as f:
        json.dump(data, f, indent=2)


def _safe_mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]
    return float(np.mean(vals)) if vals else 0.0


def run(cfg) -> None:
    """
    Hydra entrypoint.
    Expected cfg fields: cfg.paths.root and cfg.stage.{data_dir, output_dir, checkpoint, ...}
    """
    from omegaconf import DictConfig, OmegaConf

    assert isinstance(cfg, DictConfig)

    root = _resolve_path(cfg.paths.root, Path.cwd())
    _setup_logging()

    stage = cfg.stage
    data_dir = _resolve_path(stage.get("data_dir", "dataset/Consensus_Mask_Reviewer_Test"), root)
    output_dir = _resolve_path(stage.get("output_dir", "outputs/stage7_compare_bboxes"), root)
    checkpoint_path = _resolve_path(stage.get("checkpoint"), root)
    deeplab_masks_dir_cfg = stage.get("deeplab_masks_dir")
    deeplab_masks_dir: Optional[Path] = None
    if deeplab_masks_dir_cfg is not None:
        deeplab_masks_dir = _resolve_path(deeplab_masks_dir_cfg, root)

    compare_with_bboxes = bool(stage.get("compare_with_bboxes", True))
    iou_threshold = float(stage.get("iou_threshold", 0.1))
    gt_format_raw = str(stage.get("gt_format", "bbox")).strip().lower()
    if gt_format_raw == "polygone":
        gt_format = "polygon"
    elif gt_format_raw in {"bbox", "polygon", "all", "both"}:
        gt_format = "all" if gt_format_raw == "both" else gt_format_raw
    else:
        raise ValueError(
            f"Unsupported gt_format='{gt_format_raw}'. Use 'bbox', 'polygon', or 'all'."
        )

    min_component_area = int(stage.get("min_component_area", 0))

    merge_predictions_inside_gt = bool(stage.get("merge_predictions_inside_gt", False))
    allow_pred_multi_gt = bool(stage.get("allow_pred_multi_gt", False))
    merge_overlapping_gt_boxes = bool(stage.get("merge_overlapping_gt_boxes", False))
    merge_gt_boxes_iou_threshold = float(stage.get("merge_gt_boxes_iou_threshold", 0.0))
    merge_gt_proximity_px = float(stage.get("merge_gt_proximity_px", -1))

    # Optional bbox size filters (in pixel space) for predicted rectangles.
    min_bbox_area_raw = stage.get("min_bbox_area", None)
    min_bbox_width_raw = stage.get("min_bbox_width", None)
    min_bbox_height_raw = stage.get("min_bbox_height", None)
    min_bbox_area: Optional[float] = float(min_bbox_area_raw) if min_bbox_area_raw is not None else None
    min_bbox_width: Optional[float] = float(min_bbox_width_raw) if min_bbox_width_raw is not None else None
    min_bbox_height: Optional[float] = float(min_bbox_height_raw) if min_bbox_height_raw is not None else None

    # Optional per-class min area filter, keyed by LabelMe label name (e.g. Scratch/Stain).
    min_bbox_area_per_class_cfg = stage.get("min_bbox_area_per_class", None)
    min_bbox_area_per_class: Optional[Dict[str, Optional[float]]] = None
    if min_bbox_area_per_class_cfg is not None:
        try:
            # OmegaConf -> plain dict
            min_bbox_area_per_class = dict(min_bbox_area_per_class_cfg)
        except Exception:
            min_bbox_area_per_class = None

        if min_bbox_area_per_class is not None:
            # Normalize values to Optional[float]
            for k in list(min_bbox_area_per_class.keys()):
                v = min_bbox_area_per_class[k]
                if v is None:
                    min_bbox_area_per_class[k] = None
                else:
                    try:
                        min_bbox_area_per_class[k] = float(v)
                    except Exception:
                        min_bbox_area_per_class[k] = None

    dilate_iterations = int(stage.get("dilate_iterations", 0))
    dilate_kernel_size = int(stage.get("dilate_kernel_size", 15))

    save_labelme_jsons = bool(stage.get("save_labelme_jsons", False))
    labelme_pred_dir = stage.get("labelme_pred_dir")
    labelme_pred_dir_path: Optional[Path] = None
    if save_labelme_jsons:
        labelme_pred_dir_path = _resolve_path(labelme_pred_dir, root)
        labelme_pred_dir_path.mkdir(parents=True, exist_ok=True)

    save_bbox_visualizations = bool(stage.get("save_bbox_visualizations", False))
    vis_dir = stage.get("vis_dir")
    vis_dir_path: Optional[Path] = None
    if save_bbox_visualizations:
        vis_dir_path = _resolve_path(vis_dir, root)
        vis_dir_path.mkdir(parents=True, exist_ok=True)

    # Build DeepLab config for inference from stage config.
    deeplab_cfg = get_config_from_stage(cfg.stage)
    deeplab_cfg.dataset_root = _resolve_path(deeplab_cfg.dataset_root, root)
    deeplab_cfg.checkpoints_dir = _resolve_path(deeplab_cfg.checkpoints_dir, root)

    # Optional explicit label_mapping_path (not required in stage config).
    label_mapping_path = getattr(deeplab_cfg, "label_mapping_path", None)
    if label_mapping_path is not None:
        label_mapping_path = _resolve_path(label_mapping_path, root)

    id_to_label, label_to_id = _load_label_mapping(
        checkpoint_path=checkpoint_path,
        dataset_root=deeplab_cfg.dataset_root,
        label_mapping_path=label_mapping_path,
    )

    # LabelMe JSONs may use names not present in label_mapping (e.g. Scratch/Stain vs binary "defect").
    label_to_id_for_gt = dict(label_to_id)
    gt_map_cfg = stage.get("gt_label_to_class_id", None)
    if gt_map_cfg is not None:
        try:
            raw = OmegaConf.to_container(gt_map_cfg, resolve=True)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        label_to_id_for_gt[str(k)] = int(v)
                    except (TypeError, ValueError):
                        continue
                logging.info(
                    "GT label overrides (gt_label_to_class_id): added %d entries for LabelMe -> class id",
                    len(raw),
                )
        except Exception as e:
            logging.warning("Could not parse gt_label_to_class_id: %s", e)

    tiled_inference = bool(getattr(deeplab_cfg, "tiled_inference", True))
    tile_size = int(getattr(deeplab_cfg, "tile_size", 1024))
    tile_overlap = int(getattr(deeplab_cfg, "tile_overlap", 256))

    pairs = discover_image_labelme_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No (image, LabelMe JSON) pairs found under {data_dir}")
    logging.info("GT format for comparison: %s", gt_format)

    if deeplab_masks_dir is None:
        deeplab_masks_dir = output_dir / "deeplab_masks"

    # Generate masks if needed.
    _run_infer_if_needed(
        data_dir=data_dir,
        deeplab_masks_dir=deeplab_masks_dir,
        checkpoint_path=checkpoint_path,
        cfg_for_infer=deeplab_cfg,
        tiled_inference=tiled_inference,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    all_per_image: List[dict] = []
    totals = {"TP": 0, "FP": 0, "FN": 0}
    per_class_totals: Dict[int, Dict[str, int]] = {}

    def _ensure_class_totals(cid: int) -> None:
        per_class_totals.setdefault(int(cid), {"TP": 0, "FP": 0, "FN": 0})

    for img_path, json_path in pairs:
        stem = img_path.stem
        logging.info("Processing %s ...", stem)
        mask_path = deeplab_masks_dir / f"{stem}.png"
        if not mask_path.exists():
            logging.warning("Missing DeepLab mask: %s (skipping)", mask_path)
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            logging.warning("Failed to read image: %s (skipping)", img_path)
            continue

        # Match prepare/infer: full-image CLAHE on BGR before white-object crop bbox.
        if getattr(deeplab_cfg, "clahe_enabled", False):
            clip = float(getattr(deeplab_cfg, "clahe_clip_limit", 2.0))
            tg = getattr(deeplab_cfg, "clahe_tile_grid", (8, 8))
            grid = (
                (int(tg[0]), int(tg[1]))
                if isinstance(tg, (list, tuple)) and len(tg) >= 2
                else (8, 8)
            )
            img_bgr = apply_clahe_bgr(img_bgr, clip, grid)

        gt_crop_box = _get_crop_bbox(deeplab_cfg, img_bgr)

        pred_mask = _load_pred_mask(mask_path)
        mask_h, mask_w = pred_mask.shape[:2]

        # Make GT boxes in the same coordinate system as pred_mask (cropped if crop enabled).
        gt_boxes_by_class = _load_gt_boxes_by_class(
            json_path,
            gt_format=gt_format,
            label_to_id=label_to_id_for_gt,
            crop_box=gt_crop_box,
            mask_w=mask_w,
            mask_h=mask_h,
        )

        if merge_overlapping_gt_boxes:
            for _cid, _glist in list(gt_boxes_by_class.items()):
                if len(_glist) <= 1:
                    continue
                merged = _merge_overlapping_gt_boxes(list(_glist), merge_gt_boxes_iou_threshold)
                if len(merged) != len(_glist):
                    logging.info(
                        "  Merged overlapping GT class_id=%s: %d -> %d boxes (IoU > %.6f)",
                        _cid,
                        len(_glist),
                        len(merged),
                        merge_gt_boxes_iou_threshold,
                    )
                gt_boxes_by_class[_cid] = merged

        if merge_gt_proximity_px >= 0:
            for _cid, _glist in list(gt_boxes_by_class.items()):
                if len(_glist) <= 1:
                    continue
                merged = _merge_gt_by_proximity(list(_glist), merge_gt_proximity_px)
                if len(merged) != len(_glist):
                    logging.info(
                        "  Proximity-merged GT class_id=%s: %d -> %d boxes (dist <= %.0f px)",
                        _cid,
                        len(_glist),
                        len(merged),
                        merge_gt_proximity_px,
                    )
                gt_boxes_by_class[_cid] = merged

        predicted_boxes_by_class: Dict[int, List[Box]] = {}
        # Only extract from class ids actually present in the mask.
        class_ids = [int(c) for c in np.unique(pred_mask) if int(c) != 0]
        for cid in class_ids:
            predicted_boxes_by_class[cid] = _convert_mask_to_bboxes(
                pred_mask,
                cid,
                dilate_iterations=dilate_iterations,
                dilate_kernel_size=dilate_kernel_size,
                min_component_area=min_component_area,
            )
            logging.info(
                "  Pred class=%s: bboxes=%d (mask_fg_pixels=%d)",
                cid,
                len(predicted_boxes_by_class[cid]),
                int((pred_mask == cid).sum()),
            )
        # Apply optional merge and bbox-size filters and (optionally) keep
        # intermediate results for metrics/visualization.
        class_union = set(gt_boxes_by_class.keys()) | set(predicted_boxes_by_class.keys())
        processed_pred_boxes_by_class: Dict[int, List[Box]] = {}
        merged_for_gt_by_class: Dict[int, List[Optional[Box]]] = {}
        orphan_preds_by_class: Dict[int, List[Box]] = {}

        def _min_area_for_class(cid: int) -> Optional[float]:
            # Reference behavior: if per-class dict is provided, it overrides and
            # classes without an entry get "no min-area filtering" (None).
            if min_bbox_area_per_class is not None:
                label_name = id_to_label.get(int(cid))
                if label_name is not None and label_name in min_bbox_area_per_class:
                    return min_bbox_area_per_class[label_name]
                sid = str(cid)
                if sid in min_bbox_area_per_class:
                    return min_bbox_area_per_class[sid]
                return None
            return min_bbox_area

        for cid in sorted(class_union):
            gt_boxes = gt_boxes_by_class.get(cid, [])
            pred_boxes = predicted_boxes_by_class.get(cid, [])

            min_area_class = _min_area_for_class(cid)

            if merge_predictions_inside_gt:
                merged_for_gt, orphan_preds = _merge_predictions_inside_gt(
                    gt_boxes, pred_boxes, allow_pred_multi_gt=allow_pred_multi_gt,
                )

                merged_for_gt_filtered: List[Optional[Box]] = []
                for m in merged_for_gt:
                    if m is None:
                        merged_for_gt_filtered.append(None)
                    else:
                        filtered = _filter_boxes_by_size(
                            [m],
                            min_area=min_area_class,
                            min_width=min_bbox_width,
                            min_height=min_bbox_height,
                        )
                        merged_for_gt_filtered.append(filtered[0] if filtered else None)

                orphan_preds_filtered = _filter_boxes_by_size(
                    orphan_preds,
                    min_area=min_area_class,
                    min_width=min_bbox_width,
                    min_height=min_bbox_height,
                )

                merged_for_gt_by_class[cid] = merged_for_gt_filtered
                orphan_preds_by_class[cid] = orphan_preds_filtered
                processed_pred_boxes_by_class[cid] = [
                    b for b in merged_for_gt_filtered if b is not None
                ] + orphan_preds_filtered
            else:
                processed_pred_boxes_by_class[cid] = _filter_boxes_by_size(
                    pred_boxes,
                    min_area=min_area_class,
                    min_width=min_bbox_width,
                    min_height=min_bbox_height,
                )

        # Optional export (LabelMe JSON in original-image coordinates).
        if save_labelme_jsons and labelme_pred_dir_path is not None:
            out_json = labelme_pred_dir_path / f"{stem}.json"
            _save_predicted_labelme_jsons(
                json_path=json_path,
                out_json_path=out_json,
                predicted_boxes_by_class=processed_pred_boxes_by_class,
                id_to_label=id_to_label,
                crop_box=gt_crop_box,
            )

        # Compare GT vs predicted bboxes by class.
        if not compare_with_bboxes:
            continue

        per_image_metrics: Dict[str, object] = {
            "image": stem,
            "gt_count": {},
            "pred_count": {},
            "per_class": {},
        }

        for cid in sorted(class_union):
            _ensure_class_totals(cid)
            gt_boxes = gt_boxes_by_class.get(cid, [])
            pred_boxes = processed_pred_boxes_by_class.get(cid, [])

            if merge_predictions_inside_gt:
                merged_for_gt = merged_for_gt_by_class.get(cid, [None] * len(gt_boxes))
                orphan_preds = orphan_preds_by_class.get(cid, [])

                tp = 0
                fp = 0
                fn = 0
                for gi, g_box in enumerate(gt_boxes):
                    merged_box = merged_for_gt[gi] if gi < len(merged_for_gt) else None
                    if merged_box is None:
                        fn += 1
                    else:
                        val = iou(merged_box, g_box)
                        if val >= iou_threshold:
                            tp += 1
                        else:
                            fp += 1
                fp += len(orphan_preds)
            else:
                tp, fp, fn, _mgt, _mp, _gt_to_pred, _pred_to_best_iou = match_boxes(
                    gt_boxes, pred_boxes, iou_thresh=iou_threshold
                )

            per_class_totals[cid]["TP"] += tp
            per_class_totals[cid]["FP"] += fp
            per_class_totals[cid]["FN"] += fn

            totals["TP"] += tp
            totals["FP"] += fp
            totals["FN"] += fn

            label = id_to_label.get(int(cid), str(cid))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_image_metrics["per_class"][str(label)] = {
                "class_id": int(cid),
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": round(precision, 6),
                "Recall": round(recall, 6),
                "F1": round(f1, 6),
            }
            per_image_metrics["gt_count"][str(label)] = len(gt_boxes)
            per_image_metrics["pred_count"][str(label)] = len(pred_boxes)

        all_per_image.append(per_image_metrics)

        if save_bbox_visualizations and vis_dir_path is not None:
            # Visualize by drawing TP/FP/FN boxes (class-agnostic color scheme).
            # Colors (BGR): GT=green dashed, TP=orange, FP=red, FN=blue.
            COLORS = {
                "GT": (0, 255, 0),
                "TP": (0, 165, 255),
                "FP": (0, 0, 255),
                "FN": (255, 0, 0),
            }

            vis = img_bgr.copy()
            # Helper for dashed rectangle
            def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], width: int) -> None:
                dash = 8
                for x in range(int(x1), int(x2), dash * 2):
                    end_x = min(x + dash, x2)
                    cv2.line(img, (x, y1), (end_x, y1), color, width)
                    cv2.line(img, (x, y2), (end_x, y2), color, width)
                for y in range(int(y1), int(y2), dash * 2):
                    end_y = min(y + dash, y2)
                    cv2.line(img, (x1, y), (x1, end_y), color, width)
                    cv2.line(img, (x2, y), (x2, end_y), color, width)

            # Draw per class
            if gt_crop_box is not None:
                ox1, oy1, _, _ = gt_crop_box
            else:
                ox1, oy1 = 0, 0

            for cid in sorted(class_union):
                gt_boxes = gt_boxes_by_class.get(cid, [])
                pred_boxes = processed_pred_boxes_by_class.get(cid, [])

                if merge_predictions_inside_gt:
                    merged_for_gt = merged_for_gt_by_class.get(cid, [None] * len(gt_boxes))
                    orphan_preds = orphan_preds_by_class.get(cid, [])

                    # Evaluate per GT using the merged box (GT-reference merge).
                    for gi, g in enumerate(gt_boxes):
                        merged_box = merged_for_gt[gi] if gi < len(merged_for_gt) else None
                        x1 = int(round(g.x1 + ox1))
                        y1 = int(round(g.y1 + oy1))
                        x2 = int(round(g.x2 + ox1))
                        y2 = int(round(g.y2 + oy1))

                        if merged_box is None:
                            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["FN"], 2)
                            continue

                        val = iou(merged_box, g)
                        mx1 = int(round(merged_box.x1 + ox1))
                        my1 = int(round(merged_box.y1 + oy1))
                        mx2 = int(round(merged_box.x2 + ox1))
                        my2 = int(round(merged_box.y2 + oy1))

                        if val >= iou_threshold:
                            _draw_dashed_rect(vis, x1, y1, x2, y2, COLORS["GT"], 2)
                            cv2.rectangle(vis, (mx1, my1), (mx2, my2), COLORS["TP"], 2)
                        else:
                            cv2.rectangle(vis, (mx1, my1), (mx2, my2), COLORS["FP"], 2)

                    # Orphan predictions are FP
                    for p in orphan_preds:
                        x1 = int(round(p.x1 + ox1))
                        y1 = int(round(p.y1 + oy1))
                        x2 = int(round(p.x2 + ox1))
                        y2 = int(round(p.y2 + oy1))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["FP"], 2)
                else:
                    tp, fp, fn, matched_gt, matched_pred, gt_to_pred, pred_to_best_iou = match_boxes(
                        gt_boxes, pred_boxes, iou_thresh=iou_threshold
                    )

                    # Unmatched GT => FN
                    for gi, g in enumerate(gt_boxes):
                        if gi in matched_gt:
                            continue
                        x1 = int(round(g.x1 + ox1))
                        y1 = int(round(g.y1 + oy1))
                        x2 = int(round(g.x2 + ox1))
                        y2 = int(round(g.y2 + oy1))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["FN"], 2)

                    # Matched GT => green dashed
                    for gi, g in enumerate(gt_boxes):
                        if gi not in matched_gt:
                            continue
                        x1 = int(round(g.x1 + ox1))
                        y1 = int(round(g.y1 + oy1))
                        x2 = int(round(g.x2 + ox1))
                        y2 = int(round(g.y2 + oy1))
                        _draw_dashed_rect(vis, x1, y1, x2, y2, COLORS["GT"], 2)

                    # Matched pred => orange
                    for pi, p in enumerate(pred_boxes):
                        if pi not in matched_pred:
                            continue
                        x1 = int(round(p.x1 + ox1))
                        y1 = int(round(p.y1 + oy1))
                        x2 = int(round(p.x2 + ox1))
                        y2 = int(round(p.y2 + oy1))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["TP"], 2)

                    # Unmatched pred => red
                    for pi, p in enumerate(pred_boxes):
                        if pi in matched_pred:
                            continue
                        x1 = int(round(p.x1 + ox1))
                        y1 = int(round(p.y1 + oy1))
                        x2 = int(round(p.x2 + ox1))
                        y2 = int(round(p.y2 + oy1))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["FP"], 2)

            vis_path = vis_dir_path / f"{stem}_bboxes_compare.jpg"
            cv2.imwrite(str(vis_path), vis)

    if not compare_with_bboxes:
        logging.info("Skipping bbox comparison metrics (compare_with_bboxes=false).")
        return

    tp_total = totals["TP"]
    fp_total = totals["FP"]
    fn_total = totals["FN"]
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class_summary: Dict[str, dict] = {}
    for cid, counts in sorted(per_class_totals.items(), key=lambda kv: kv[0]):
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        label = id_to_label.get(int(cid), str(cid))
        per_class_summary[label] = {
            "class_id": int(cid),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1_c, 6),
        }

    metrics_payload = {
        "iou_threshold": iou_threshold,
        "gt_format": gt_format,
        "compare_with_bboxes": compare_with_bboxes,
        "mask_to_bbox": {
            "min_component_area": min_component_area,
            "dilate_iterations": dilate_iterations,
            "dilate_kernel_size": dilate_kernel_size,
            "merge_overlapping_gt_boxes": merge_overlapping_gt_boxes,
            "merge_gt_boxes_iou_threshold": merge_gt_boxes_iou_threshold,
            "merge_predictions_inside_gt": merge_predictions_inside_gt,
            "allow_pred_multi_gt": allow_pred_multi_gt,
            "merge_gt_proximity_px": merge_gt_proximity_px,
            "min_bbox_area": min_bbox_area,
            "min_bbox_width": min_bbox_width,
            "min_bbox_height": min_bbox_height,
            "min_bbox_area_per_class": min_bbox_area_per_class,
        },
        "summary": {
            "TP": int(tp_total),
            "FP": int(fp_total),
            "FN": int(fn_total),
            "Precision": round(precision, 6),
            "Recall": round(recall, 6),
            "F1": round(f1, 6),
        },
        "per_class": per_class_summary,
        "per_image": all_per_image,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_path = _resolve_path(stage.get("save_json", output_dir / "metrics.json"), root)
    with save_json_path.open("w") as f:
        json.dump(metrics_payload, f, indent=2)
    logging.info("Saved bbox comparison metrics JSON to %s", save_json_path)


def parse_args() -> argparse.Namespace:
    # The stage is typically run via Hydra (run.py). This exists mostly for debugging.
    ap = argparse.ArgumentParser(description="DeepLab mask -> bboxes comparison stage (no SAM2).")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    # Support standalone execution only for quick smoke tests.
    args = parse_args()
    if args.data_dir is None or args.output_dir is None or args.checkpoint is None:
        raise SystemExit("When running standalone, provide --data-dir, --output-dir, --checkpoint (Hydra is preferred).")
    cfg = get_default_config()
    cfg_for_infer = cfg
    stage_cfg = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "checkpoint": args.checkpoint,
    }
    # Minimal standalone fallback: no Hydra config, so we can't reliably run inference with proper model config.
    # Use `python run.py stage=compare_bboxes ...` instead.
    raise SystemExit("Standalone execution is not supported for full inference. Use Hydra: `python run.py stage=compare_bboxes ...`.")

