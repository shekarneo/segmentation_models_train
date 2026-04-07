#!/usr/bin/env python3

"""
Evaluate detection performance by comparing LabelMe GT bboxes to predicted masks.

For each image:
  - GT: read LabelMe JSON, extract axis-aligned bounding boxes per shape.
  - Pred: read semantic mask PNG, treat any non-zero pixel as foreground.
  - Predicted instances: connected components in the binary mask -> per-component bbox.
  - Matching: greedy one-to-one by IoU >= threshold (default 0.1).

Metrics:
  - TP: GT boxes matched to a predicted component
  - FP: predicted components not matched to any GT
  - FN: GT boxes not matched by any prediction
  - Precision, Recall, F1 (global, over all images)

Usage (from project root):
  python scripts/eval_detections_boxes_vs_masks.py
  python scripts/eval_detections_boxes_vs_masks.py --gt-dir dataset/Consensus_Mask_Reviewer_Test --pred-dir predictions --iou-threshold 0.1
  python scripts/eval_detections_boxes_vs_masks.py   --gt-dir dataset/Consensus_Mask_Reviewer_Test  \
     --pred-dir predictions   --images-dir dataset/Consensus_Mask_Reviewer_Test  \
     --output-vis eval_vis   --save-json eval_metrics.json   --iou-threshold 0.1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = PROJECT_ROOT / "dataset" / "Consensus_Mask_Reviewer_Test"
DEFAULT_PRED_DIR = PROJECT_ROOT / "predictions"


@dataclass
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


def _bbox_from_shape(shape: dict) -> Box | None:
    """Convert LabelMe shape to axis-aligned bbox (x1,y1,x2,y2)."""
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
    return Box(x1, y1, x2, y2)


def load_gt_boxes(labelme_json: Path) -> List[Box]:
    """Load GT boxes from LabelMe JSON (all shapes -> bboxes)."""
    with open(labelme_json, "r") as f:
        data = json.load(f)
    boxes: List[Box] = []
    for shape in data.get("shapes", []):
        b = _bbox_from_shape(shape)
        if b is not None:
            boxes.append(b)
    return boxes


def load_pred_boxes(mask_path: Path) -> List[Box]:
    """Connected components in predicted mask -> predicted boxes (any nonzero pixel is foreground)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    fg = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(fg)
    boxes: List[Box] = []
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        boxes.append(Box(x1, y1, x2, y2))
    return boxes


def match_boxes(
    gt_boxes: List[Box], pred_boxes: List[Box], iou_thresh: float
) -> Tuple[int, int, int, Set[int], Set[int], Dict[int, Tuple[int, float]], Dict[int, float]]:
    """
    Greedy one-to-one matching.
    Returns:
      tp, fp, fn,
      matched_gt_indices, matched_pred_indices,
      gt_to_pred: gt_idx -> (pred_idx, iou),
      pred_to_best_iou: pred_idx -> best iou with any GT (for FP label).
    """
    tp = 0
    matched_pred: Set[int] = set()
    matched_gt: Set[int] = set()
    gt_to_pred: Dict[int, Tuple[int, float]] = {}
    pred_to_best_iou: Dict[int, float] = {}
    for pi in range(len(pred_boxes)):
        pred_to_best_iou[pi] = max(
            (iou(gt_boxes[gi], pred_boxes[pi]) for gi in range(len(gt_boxes))),
            default=0.0,
        )
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
        if best_iou_val >= iou_thresh and best_pi >= 0:
            tp += 1
            matched_pred.add(best_pi)
            matched_gt.add(gi)
            gt_to_pred[gi] = (best_pi, best_iou_val)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_gt, matched_pred, gt_to_pred, pred_to_best_iou


# Color coding (BGR) chosen so that saved images visually match
# evaluate_model_performance_save_json / visualize_bboxes (which use RGB):
#   GT:  (0, 255, 0)       -> green
#   TP:  (255, 165, 0) RGB -> (0, 165, 255) BGR (orange)
#   FP:  (255, 0, 0) RGB   -> (0, 0, 255)  BGR (red)
#   FN:  (0, 0, 255) RGB   -> (255, 0, 0)  BGR (blue)
COLORS = {
    "GT": (0, 255, 0),
    "TP": (0, 165, 255),
    "FP": (0, 0, 255),
    "FN": (255, 0, 0),
}


def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], width: int) -> None:
    """Draw dashed rectangle (BGR)."""
    dash = 8
    for x in range(int(x1), int(x2), dash * 2):
        end_x = min(x + dash, x2)
        cv2.line(img, (x, y1), (end_x, y1), color, width)
        cv2.line(img, (x, y2), (end_x, y2), color, width)
    for y in range(int(y1), int(y2), dash * 2):
        end_y = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, end_y), color, width)
        cv2.line(img, (x2, y), (x2, end_y), color, width)


def draw_bboxes_visualization(
    image_path: Path,
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    matched_gt: Set[int],
    matched_pred: Set[int],
    gt_to_pred: Dict[int, Tuple[int, float]],
    pred_to_best_iou: Dict[int, float],
    iou_thresh: float,
    output_path: Path,
) -> None:
    """
    Draw GT/pred boxes with color coding: GT=green dashed, TP=orange, FP=red, FN=blue.
    Same scheme as JnJ_Scratch_Detection_Results/visualize_bboxes.py.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.0, w / 1200.0))
    thickness = max(1, int(round(font_scale * 2)))

    # 1) FN: unmatched GT -> blue
    for gi, g in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        x1, y1, x2, y2 = int(g.x1), int(g.y1), int(g.x2), int(g.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["FN"], 2)
        cv2.putText(img, "FN", (x1, y1 - 4), font, font_scale, COLORS["FN"], thickness, cv2.LINE_AA)

    # 2) GT matched -> green dashed
    for gi, g in enumerate(gt_boxes):
        if gi not in matched_gt:
            continue
        x1, y1, x2, y2 = int(g.x1), int(g.y1), int(g.x2), int(g.y2)
        _draw_dashed_rect(img, x1, y1, x2, y2, COLORS["GT"], 3)
        iou_val = gt_to_pred.get(gi, (None, 0.0))[1]
        cv2.putText(img, f"GT IoU={iou_val:.2f}", (x1, y1 - 4), font, font_scale, COLORS["GT"], thickness, cv2.LINE_AA)

    # 3) TP: matched pred -> orange
    for pi, p in enumerate(pred_boxes):
        if pi not in matched_pred:
            continue
        x1, y1, x2, y2 = int(p.x1), int(p.y1), int(p.x2), int(p.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["TP"], 2)
        for gi, (pred_i, iou_val) in gt_to_pred.items():
            if pred_i == pi:
                cv2.putText(img, f"TP IoU={iou_val:.2f}", (x1, y2 + 16), font, font_scale, COLORS["TP"], thickness, cv2.LINE_AA)
                break

    # 4) FP: unmatched pred -> red
    for pi, p in enumerate(pred_boxes):
        if pi in matched_pred:
            continue
        x1, y1, x2, y2 = int(p.x1), int(p.y1), int(p.x2), int(p.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["FP"], 2)
        best_iou = pred_to_best_iou.get(pi, 0.0)
        cv2.putText(img, f"FP IoU={best_iou:.2f}", (x1, y2 + 16), font, font_scale, COLORS["FP"], thickness, cv2.LINE_AA)

    # Legend (top-left area)
    legend_y = 30
    x_off = 500
    cv2.rectangle(img, (x_off, legend_y - 20), (x_off + 250, legend_y + 95), (40, 40, 40), -1)
    cv2.putText(img, "GT (Green, dashed)", (x_off + 30, legend_y), font, 0.5, COLORS["GT"], 1, cv2.LINE_AA)
    legend_y += 22
    cv2.putText(img, "TP (Orange, solid)", (x_off + 30, legend_y), font, 0.5, COLORS["TP"], 1, cv2.LINE_AA)
    legend_y += 22
    cv2.putText(img, "FP (Red, solid)", (x_off + 30, legend_y), font, 0.5, COLORS["FP"], 1, cv2.LINE_AA)
    legend_y += 22
    cv2.putText(img, "FN (Blue, solid)", (x_off + 30, legend_y), font, 0.5, COLORS["FN"], 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare LabelMe GT bboxes vs predicted masks and report detection metrics."
    )
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help=f"Directory with LabelMe JSONs (and images) [default: {DEFAULT_GT_DIR}]",
    )
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help=f"Directory with predicted mask PNGs [default: {DEFAULT_PRED_DIR}]",
    )
    ap.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="IoU threshold for a detection to match a GT box (default: 0.1).",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing input images (default: same as --gt-dir).",
    )
    ap.add_argument(
        "--output-vis",
        type=Path,
        default=None,
        help="Output directory for visualization images (same color coding as evaluate_model_performance_save_json). If not set, no visualizations are saved.",
    )
    ap.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Path to save summary and per-image metrics JSON. If not set, JSON is not saved.",
    )
    return ap.parse_args()


def _find_image_path(stem: str, gt_dir: Path, images_dir: Path | None) -> Path | None:
    """Return path to image for stem; prefer images_dir, else gt_dir. Try .jpg, .png, .jpeg."""
    search_dir = images_dir if images_dir is not None else gt_dir
    for ext in (".jpg", ".png", ".jpeg"):
        p = search_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    args = parse_args()
    gt_dir: Path = args.gt_dir
    pred_dir: Path = args.pred_dir
    iou_thresh: float = args.iou_threshold
    images_dir: Path | None = args.images_dir
    output_vis: Path | None = args.output_vis
    save_json_path: Path | None = args.save_json

    if not gt_dir.is_dir():
        raise SystemExit(f"GT directory '{gt_dir}' does not exist.")
    if not pred_dir.is_dir():
        raise SystemExit(f"Prediction directory '{pred_dir}' does not exist.")

    json_files = sorted(gt_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No LabelMe JSON files found under '{gt_dir}'.")

    if output_vis is not None:
        output_vis.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {output_vis}")

    total_tp = total_fp = total_fn = 0
    per_image: List[dict] = []

    print(f"GT dir:   {gt_dir}")
    print(f"Pred dir: {pred_dir}")
    print(f"IoU threshold: {iou_thresh}\n")

    for jpath in json_files:
        stem = jpath.stem
        gt_boxes = load_gt_boxes(jpath)
        pred_mask_path = pred_dir / f"{stem}.png"
        pred_boxes = load_pred_boxes(pred_mask_path) if pred_mask_path.exists() else []

        tp, fp, fn, matched_gt, matched_pred, gt_to_pred, pred_to_best_iou = match_boxes(
            gt_boxes, pred_boxes, iou_thresh
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if save_json_path is not None:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_img = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_image.append({
                "image": stem,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1_img, 4),
            })

        if output_vis is not None:
            image_path = _find_image_path(stem, gt_dir, images_dir)
            if image_path is not None:
                out_img = output_vis / f"{stem}_visualized.jpg"
                draw_bboxes_visualization(
                    image_path,
                    gt_boxes,
                    pred_boxes,
                    matched_gt,
                    matched_pred,
                    gt_to_pred,
                    pred_to_best_iou,
                    iou_thresh,
                    out_img,
                )
                print(f"Saved: {out_img}")
            else:
                print(f"WARNING: No image found for {stem}, skipping visualization.")

        print(
            f"{stem}: "
            f"GT={len(gt_boxes)}, Pred={len(pred_boxes)}, "
            f"TP={tp}, FP={fp}, FN={fn}"
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n=== Summary over all images ===")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    if save_json_path is not None:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "iou_threshold": iou_thresh,
            "summary": {
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
            },
            "per_image": per_image,
        }
        with open(save_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetrics JSON saved to: {save_json_path}")

    if output_vis is not None:
        print(f"Visualizations saved to: {output_vis}")


if __name__ == "__main__":
    main()

