"""
Run SAM2 on a single tile with bbox prompts from a LabelMe JSON.
Used by the compare stage when using --sam2-command.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Project root = parent of src (when this file is src/utils/sam2_infer_tile.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "jnj-sam2-pipeline" / "checkpoints" / "base_models"


def _default_checkpoint_path() -> Path:
    env = os.environ.get("SAM2_CHECKPOINT")
    if env:
        return Path(env)
    candidate = DEFAULT_CHECKPOINT_DIR / "sam2_hiera_large.pt"
    if candidate.exists():
        return candidate
    return Path("sam2_hiera_large.pt")


def bboxes_from_labelme(json_path: Path) -> list[list[float]]:
    """Return list of [x1, y1, x2, y2] from LabelMe shapes (rectangle or polygon bbox)."""
    with open(json_path) as f:
        data = json.load(f)
    out = []
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        if shape.get("shape_type") == "rectangle" and len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


CKPT_TO_CONFIG = {
    "sam2_hiera_tiny.pt": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small.pt": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus.pt": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def _load_predictor_from_local_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        return None
    ckpt_path = str(checkpoint_path.resolve())
    config_name = CKPT_TO_CONFIG.get(checkpoint_path.name)
    if not config_name:
        for name, config in CKPT_TO_CONFIG.items():
            if name in checkpoint_path.name or checkpoint_path.name in name:
                config_name = config
                break
        if not config_name:
            config_name = "configs/sam2/sam2_hiera_l.yaml"
    try:
        sam_model = build_sam2(config_file=config_name, ckpt_path=ckpt_path, device=device)
        return SAM2ImagePredictor(sam_model)
    except Exception:
        return None


def run_sam2_predictor(image_bgr: np.ndarray, boxes_xyxy: np.ndarray, checkpoint_path: Path) -> np.ndarray | None:
    """Run SAM2 with bbox prompts (one predict per box, then union). Return binary mask (H,W) or None."""
    predictor = _load_predictor_from_local_checkpoint(checkpoint_path)
    if predictor is None:
        return None
    h, w = image_bgr.shape[:2]
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(boxes_xyxy.shape[0]):
        box = boxes_xyxy[i : i + 1]
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        if masks is not None and len(masks) > 0:
            m = (masks[0] > 0).astype(np.uint8)
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out = np.maximum(out, m)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path, help="Tile image path")
    ap.add_argument("--boxes", required=True, type=Path, help="LabelMe JSON with bboxes (tile coords)")
    ap.add_argument("--output", required=True, type=Path, help="Output mask path (PNG)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Path to SAM2 checkpoint .pt file")
    args = ap.parse_args()
    ckpt = args.checkpoint if args.checkpoint is not None else _default_checkpoint_path()
    img = cv2.imread(str(args.image))
    if img is None:
        print("Failed to read image", args.image, file=sys.stderr)
        return 1
    boxes = bboxes_from_labelme(args.boxes)
    if not boxes:
        cv2.imwrite(str(args.output), np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8))
        return 0
    boxes_np = np.array(boxes, dtype=np.float64)
    mask = run_sam2_predictor(img, boxes_np, ckpt)
    if mask is None:
        print("SAM2 not available or checkpoint failed. Checkpoint:", ckpt, file=sys.stderr)
        return 1
    cv2.imwrite(str(args.output), (mask * 255).astype(np.uint8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
