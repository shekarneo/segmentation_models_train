"""Stage 1: SAM2 + CLAHE Pseudo-mask Generation.

Generates high-quality pixel-level masks from bounding box annotations.
"""

import os
import json
import logging
import shutil
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig

# Clear Hydra before importing SAM2 (it uses Hydra internally)
from hydra.core.global_hydra import GlobalHydra

from ..utils.clahe import apply_clahe
from ..models.sam2_lora import build_sam2_with_isolated_hydra

log = logging.getLogger(__name__)

def load_sam2_predictor(model_repo: str, device: str, local_path: str = None):
    """Load SAM2 predictor with Hydra context cleared.
    
    Args:
        model_repo: HuggingFace repo ID (e.g., "facebook/sam2-hiera-large")
        device: CUDA device string
        local_path: Optional absolute path to local checkpoint file. If provided and exists,
                   bypasses HuggingFace Hub download.
    """
    GlobalHydra.instance().clear()
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    # Map model_repo to config file name
    model_configs = {
        "facebook/sam2-hiera-large":      ("sam2_hiera_l.yaml",  "sam2_hiera_large.pt"),
        "facebook/sam2-hiera-base-plus":  ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
        "facebook/sam2-hiera-small":      ("sam2_hiera_s.yaml",  "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-tiny":       ("sam2_hiera_t.yaml",  "sam2_hiera_tiny.pt"),
    }
    config_file, ckpt_filename = model_configs.get(model_repo, ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"))
    
    # Use local_path if provided and exists, otherwise use from_pretrained
    if local_path and Path(local_path).exists():
        log.info(f"Loading SAM2 from local path: {local_path}")
        sam2_model = build_sam2_with_isolated_hydra(config_file, local_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
    else:
        if local_path:
            log.warning(f"SAM2 local_path not found: {local_path} — falling back to HuggingFace download.")
        predictor = SAM2ImagePredictor.from_pretrained(model_repo, device=device)
    
    return predictor


def load_predictor(cfg: DictConfig):
    """Load SAM2 or SAM3 predictor based on model.family."""
    family = str(cfg.model.get("family", "sam2"))
    device = str(cfg.device)
    if family == "sam3":
        from ..utils.sam3_utils import load_sam3_predictor
        sam3_local = cfg.model.checkpoint.get("local_path", None)
        _proc, _model, predictor = load_sam3_predictor(device, local_path=sam3_local)
        return predictor
    GlobalHydra.instance().clear()
    model_repo = str(cfg.model.checkpoint.repo)
    sam2_local = cfg.model.checkpoint.get("local_path", None)
    return load_sam2_predictor(model_repo, device, local_path=sam2_local)


def _predict_sam2_batch(predictor, bboxes: np.ndarray, multimask_output: bool):
    """Run SAM2 predictor on a batch of boxes (one forward pass). Returns list of (mask, score) per box."""
    if bboxes.size == 0:
        return []
    # predictor._prep_prompts(point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1)
    _, _, _, unnorm_box = predictor._prep_prompts(
        None, None, bboxes, None, True, img_idx=-1
    )
    # unnorm_box is (N, 2, 2); _predict expects boxes that reshape to (N, 2, 2)
    masks, iou_predictions, _ = predictor._predict(
        None, None, unnorm_box, None, multimask_output=multimask_output, img_idx=-1
    )
    # masks (B, C, H, W), iou_predictions (B, C)
    results = []
    for i in range(masks.shape[0]):
        iou_np = iou_predictions[i].float().cpu().numpy()
        best_c = int(np.argmax(iou_np))
        m = masks[i, best_c].float().cpu().numpy()
        s = float(iou_np[best_c])
        if m.ndim == 3:
            m = m[0]
        results.append((m, s))
    return results


def _predict_sam3_batch(predictor, bboxes: np.ndarray, multimask_output: bool):
    """Run SAM3 on a batch of boxes (one decoder forward). Uses predictor._state from set_image.
    Returns list of (mask_2d, score) per box; matches model outputs to input boxes by IoU."""
    if bboxes.size == 0:
        return []
    state = predictor._state
    h, w = state["original_height"], state["original_width"]
    device = predictor.device

    # Ensure text/visual features exist (same as add_geometric_prompt)
    if "language_features" not in state["backbone_out"]:
        dummy_text = predictor.model.backbone.forward_text(["visual"], device=device)
        state["backbone_out"].update(dummy_text)
    state["geometric_prompt"] = predictor.model._get_dummy_prompt()
    # Append all boxes without forwarding (add_geometric_prompt does append then forward; we only append)
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        box_t = torch.tensor([cx, cy, bw, bh], device=device, dtype=torch.float32).view(1, 1, 4)
        label_t = torch.tensor([True], device=device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(box_t, label_t)
    # Single forward for all boxes
    predictor.processor._forward_grounding(state)

    out_masks = state["masks"]   # (M, 1, H, W) or (M, H, W)
    out_scores = state["scores"]  # (M,)
    out_boxes = state["boxes"]    # (M, 4) xyxy

    M = out_masks.shape[0]
    if M == 0:
        return [(np.zeros((h, w), dtype=bool), 0.0) for _ in range(bboxes.shape[0])]

    mask_list = []
    score_list = []
    box_list = []
    for i in range(M):
        m = out_masks[i].float().cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        mask_list.append(m)
        score_list.append(float(out_scores[i].cpu().numpy()))
        b = out_boxes[i].cpu().numpy()
        box_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    # 1:1 assignment: match each input box to best output by box IoU (each output used at most once)
    def _box_iou(a, b):
        xi1 = max(a[0], b[0])
        yi1 = max(a[1], b[1])
        xi2 = min(a[2], b[2])
        yi2 = min(a[3], b[3])
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0.0

    used_out = set()
    results = []
    for j in range(bboxes.shape[0]):
        in_box = [bboxes[j, 0], bboxes[j, 1], bboxes[j, 2], bboxes[j, 3]]
        best_idx = -1
        best_iou = -1.0
        for i in range(M):
            if i in used_out:
                continue
            iou = _box_iou(in_box, box_list[i])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0:
            used_out.add(best_idx)
            m = _clip_mask_to_bbox(mask_list[best_idx], in_box, h, w)
            results.append((m, score_list[best_idx]))
        else:
            results.append((np.zeros((h, w), dtype=bool), 0.0))
    return results


def _bbox_from_mask(mask: np.ndarray):
    """Compute tight [x1,y1,x2,y2] bbox from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2, y2]

def _erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink mask by eroding border. px=0 means no change."""
    if px <= 0:
        return mask
    if mask is None or mask.size == 0 or mask.ndim != 2:
        return mask
    k = 2 * px + 1
    kernel = np.ones((k, k), np.uint8)
    m = np.ascontiguousarray((mask > 0).astype(np.uint8))
    return cv2.erode(m, kernel).astype(bool)


def _pad_and_clip_bbox(bbox, pad: int, w: int, h: int):
    """Pad bbox and clip to image bounds."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _clip_mask_to_bbox(mask: np.ndarray, bbox, h: int, w: int) -> np.ndarray:
    """Zero out mask pixels outside the bbox so output is strictly inside the box."""
    if mask is None or mask.size == 0 or mask.ndim != 2:
        return mask
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    out = np.zeros_like(mask, dtype=mask.dtype)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def _mask_iou_with_bbox(mask: np.ndarray, bbox, h: int, w: int) -> float:
    """IoU of binary mask with the bbox region (bbox as a binary mask)."""
    if mask.size == 0:
        return 0.0
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True
    inter = (mask > 0) & bbox_mask
    union = (mask > 0) | bbox_mask
    if union.sum() == 0:
        return 0.0
    return float(inter.sum()) / float(union.sum())


def _bbox_to_binary_mask(bbox, h: int, w: int) -> np.ndarray:
    """Create a binary mask from a bbox [x1, y1, x2, y2] (one mask per bbox fallback)."""
    x1, y1, x2, y2 = [int(round(x)) for x in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    out = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 1
    return out


def _is_valid_mask(mask: np.ndarray) -> bool:
    """Check if mask is valid (not None, correct shape, and has pixels)."""
    if mask is None:
        return False
    if mask.ndim != 2:
        return False
    if mask.size == 0:
        return False
    # Check if mask has any positive pixels
    return bool((mask > 0).any())


def _bbox_to_polygon(bbox) -> list:
    """Bbox [x1,y1,x2,y2] as LabelMe polygon points (for fallback when mask is empty)."""
    if len(bbox) < 4:
        return []
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _is_valid_bbox(bbox, w: int, h: int, min_side: int = 2) -> bool:
    """Check bbox is valid and within image bounds (avoids OpenCV locateROI errors)."""
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    if x2 <= x1 or y2 <= y1:
        return False
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return False
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    return True

def mask_to_polygon(mask: np.ndarray) -> list:
    """Convert binary mask to polygon points for LabelMe format."""
    if mask is None or mask.size == 0 or mask.ndim != 2:
        return []
    mask_uint8 = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return []
    # Simplify polygon (optional, reduces points)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    points = [[float(p[0][0]), float(p[0][1])] for p in contour]
    return points


def build_labelme_json(
    image_path: str,
    image_height: int,
    image_width: int,
    masks: list,
    labels: list,
    bboxes: list | None = None,
) -> dict:
    """Build LabelMe JSON structure with polygon shapes (one shape per mask/bbox)."""
    shapes = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        points = mask_to_polygon(mask)
        if not points and bboxes is not None and i < len(bboxes):
            points = _bbox_to_polygon(bboxes[i])
        if not points:
            continue
        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })
    return {
        "version": "5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def _merge_overlapping_masks(
    masks: list,
    labels: list,
    scores: list,
    iou_thresh: float,
) -> tuple[list, list]:
    """Merge masks that overlap (IoU > iou_thresh) into clusters; keep one per cluster (highest score).
    Returns (reduced_masks, reduced_labels) for LabelMe export to avoid duplicate shapes."""
    n = len(masks)
    if n <= 1 or iou_thresh <= 0:
        return masks, labels

    # Union-find parent array
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Binary masks for IoU (same shape assumed)
    def _to_bool(m):
        a = np.asarray(m)
        if a.ndim != 2:
            return None
        return (a > 0).astype(np.uint8)

    binaries = [_to_bool(m) for m in masks]
    for i in range(n):
        if binaries[i] is None:
            continue
        for j in range(i + 1, n):
            if binaries[j] is None:
                continue
            a, b = binaries[i], binaries[j]
            if a.shape != b.shape:
                continue
            inter = np.logical_and(a, b).sum()
            union_ = np.logical_or(a, b).sum()
            if union_ <= 0:
                continue
            iou = float(inter) / float(union_)
            if iou >= iou_thresh:
                union(i, j)

    # For each root, pick index with highest score in that component
    by_root: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        by_root.setdefault(r, []).append(i)
    merged_masks = []
    merged_labels = []
    for indices in by_root.values():
        best = max(indices, key=lambda i: scores[i])
        merged_masks.append(masks[best])
        merged_labels.append(labels[best])
    return merged_masks, merged_labels


def load_labelme_annotation(json_path: Path) -> dict:
    """Load LabelMe annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    bboxes = []
    labels = []
    
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            pts = shape['points']
            if len(pts) < 2:
                continue
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            label = shape['label'].replace('_unique', '').rstrip('0123456789')
            labels.append(label)
    
    return {
        'image_size': (data.get('imageWidth', 0), data.get('imageHeight', 0)),
        'bboxes': bboxes,
        'labels': labels
    }

def run(cfg: DictConfig) -> dict:
    """Run pseudomask generation stage."""
    log.info("=" * 60)
    log.info("Stage 1: Pseudo-mask Generation")
    log.info("=" * 60)
    
    # Extract config values BEFORE clearing Hydra
    device = cfg.device
    family = str(cfg.model.get("family", "sam2"))
    base_output = Path(str(cfg.stage.output_dir))
    # Model-specific output dir: stage1_pseudomasks_sam2 or stage1_pseudomasks_sam3
    output_dir = base_output.parent / (base_output.name + "_" + family)
    clahe_clip = float(cfg.augmentation.clahe.clip_limit)
    clahe_tile = tuple(cfg.augmentation.clahe.tile_grid_size)
    use_clahe = bool(cfg.stage.preprocessing.clahe)
    use_part_isolation = bool(cfg.stage.part_isolation.enabled)
    # Use stage-level SAM2 settings (more intuitive than model.inference here)
    multimask = bool(cfg.stage.sam2.multimask_output)
    iou_thresh = float(cfg.stage.sam2.iou_threshold)
    use_bbox_fallback = bool(cfg.stage.sam2.get("use_bbox_fallback", True))
    refine_cfg = cfg.stage.sam2.get("refine", {})
    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_padding = int(refine_cfg.get("padding_px", 8))
    refine_min_area = int(refine_cfg.get("min_mask_area_px", 25))
    erode_px = int(cfg.stage.sam2.get("erode_px", 0))
    erode_px_sam3 = int(cfg.stage.get("sam3", {}).get("erode_px", erode_px))
    max_bbox_batch = int(cfg.stage.sam2.get("max_bbox_batch_size", 0))  # 0 = no limit
    save_labelme = bool(cfg.stage.save.get("labelme", False))
    merge_labelme_iou = float(cfg.stage.save.get("merge_labelme_iou", 0))
    keep_all_tiles = bool(cfg.stage.get("keep_all_tiles", False))

    # Extract data paths.
    # Prefer stage.input_dir (Stage 0 tiling output) so SAM2 sees tiles at native scale.
    # If that directory doesn't exist yet (tiling not run), fall back to raw dataset
    # and warn the user — raw full-res images produce poor masks for thin features.
    # Can disable tiling explicitly with use_tiled_input: false
    use_tiled_input = cfg.stage.get('use_tiled_input', True)
    input_dir_override = cfg.stage.get('input_dir', None)
    
    if use_tiled_input and input_dir_override:
        _inp = Path(str(input_dir_override))
        if _inp.exists():
            # Check if it's a tiled directory structure (has train/test subdirs)
            if (_inp / 'train').exists() or (_inp / 'test').exists():
                data_paths = {
                    'train': str(_inp / 'train'),
                    'val':   str(_inp / 'val') if (_inp / 'val').exists() else None,
                    'test':  str(_inp / 'test'),
                }
                log.info(f"Using pre-tiled input: {_inp}")
            else:
                # Single directory (not tiled structure) — treat as test set only
                log.info(f"Using single directory input: {_inp} (not tiled structure)")
                data_paths = {
                    'train': None,  # Skip train
                    'test':  str(_inp),
                }
        else:
            log.warning(
                f"Tiling output not found: {_inp}\n"
                f"  Run 'python run.py stage=tiling' first for better mask quality.\n"
                f"  Falling back to raw dataset (thin scratches may produce poor masks)."
            )
            data_paths = {
                'train': str(cfg.data.paths.train),
                'test':  str(cfg.data.paths.test),
            }
    else:
        if not use_tiled_input:
            log.info("Tiling disabled (use_tiled_input=false) — using raw full-size images")
            # If input_dir is explicitly set when tiling is disabled, use it directly
            if input_dir_override:
                _inp = Path(str(input_dir_override))
                if _inp.exists():
                    log.info(f"Using specified input directory: {_inp}")
                    data_paths = {
                        'train': None,  # Skip train when input_dir points to specific directory
                        'test':  str(_inp),
                    }
                else:
                    log.warning(f"Input directory not found: {_inp}, falling back to data.paths")
                    data_paths = {
                        'train': str(cfg.data.paths.train),
                        'test':  str(cfg.data.paths.test),
                    }
            else:
                data_paths = {
                    'train': str(cfg.data.paths.train),
                    'test':  str(cfg.data.paths.test),
                }
        else:
            data_paths = {
                'train': str(cfg.data.paths.train),
                'test':  str(cfg.data.paths.test),
            }
    class_map = dict(cfg.data.class_map)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor (SAM2 or SAM3 based on model.family)
    log.info(f"Loading {family.upper()} on {device}...")
    predictor = load_predictor(cfg)
    log.info(f"{family.upper()} loaded.")
    log.info(f"Bbox fallback: {'enabled' if use_bbox_fallback else 'disabled'} "
             f"(masks without SAM detection will {'use bbox' if use_bbox_fallback else 'be skipped'})")

    # Build list of splits to process.
    # Always process train; add val if its directory exists;
    # add test if it differs from train (avoids duplicate processing).
    # Skip train if it's None (when input_dir points to specific directory with tiling disabled)
    stats = {}
    splits_to_process = []
    if data_paths['train'] is not None:
        splits_to_process.append(('train', data_paths['train']))
    val_path = data_paths.get('val')
    if val_path and Path(val_path).exists():
        splits_to_process.append(('val', val_path))
    if data_paths['test'] is not None:
        if data_paths['train'] != data_paths['test']:
            splits_to_process.append(('test', data_paths['test']))
        elif data_paths['train'] is not None:
            log.info("Train and test paths are the same; processing once for both.")
    
    for split, split_path in splits_to_process:
        split_dir = Path(split_path)
        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)
        
        log.info(f"\n--- Processing {split} set ---")
        log.info(f"Input: {split_dir}")
        
        split_stats = {'processed': 0, 'masks': 0, 'filtered': 0}
        
        # Find all annotations
        json_files = list(split_dir.glob('*.json'))
        log.info(f"Found {len(json_files)} annotation files")
        
        for json_path in tqdm(json_files, desc=split):
            img_path = json_path.with_suffix('.jpg')
            if not img_path.exists():
                # Try other extensions
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_path = json_path.with_suffix(ext)
                    if alt_path.exists():
                        img_path = alt_path
                        break
                else:
                    continue
            
            # Load image and annotation
            image = cv2.imread(str(img_path))
            if image is None:
                log.warning(f"Failed to read: {img_path}")
                continue
            ann = load_labelme_annotation(json_path)
            
            if not ann['bboxes']:
                if not keep_all_tiles:
                    continue  # Skip tiles with no bboxes (unless keep_all_tiles is true)
                # keep_all_tiles=true: process tile but generate empty mask
            
            # Apply CLAHE
            if use_clahe:
                image = apply_clahe(image, clahe_clip, clahe_tile)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image_rgb.shape[:2]

            masks = []
            valid_labels = []
            valid_bboxes = []
            scores = []

            predictor.set_image(image_rgb)

            if family == "sam2":
                # Batch path: one forward for all valid bboxes
                valid_indices = []
                valid_boxes_list = []
                valid_labels_pre = []
                for idx, (bbox, label) in enumerate(zip(ann['bboxes'], ann['labels'])):
                    if not _is_valid_bbox(bbox, img_w, img_h):
                        split_stats['filtered'] += 1
                        continue
                    valid_indices.append(idx)
                    valid_boxes_list.append(bbox)
                    valid_labels_pre.append(label)
                if valid_boxes_list:
                    chunk_size = max_bbox_batch if max_bbox_batch > 0 else len(valid_boxes_list)
                    all_batch_results = []
                    try:
                        for start in range(0, len(valid_boxes_list), chunk_size):
                            chunk_boxes = valid_boxes_list[start:start + chunk_size]
                            try:
                                bboxes_np = np.array(chunk_boxes, dtype=np.float64)
                                chunk_results = _predict_sam2_batch(predictor, bboxes_np, multimask)
                                all_batch_results.extend(chunk_results)
                            except RuntimeError as e:
                                if ("out of memory" in str(e).lower() or "CUDA" in str(e)) and chunk_size > 1:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    for b in chunk_boxes:
                                        single = _predict_sam2_batch(predictor, np.array([b], dtype=np.float64), multimask)
                                        all_batch_results.extend(single)
                                else:
                                    raise
                        for (bbox, label), (mask, score_used) in zip(
                            zip(valid_boxes_list, valid_labels_pre), all_batch_results
                        ):
                            if not _is_valid_mask(mask):
                                if use_bbox_fallback:
                                    mask = _bbox_to_binary_mask(bbox, img_h, img_w)
                                    score_used = 0.0
                                else:
                                    split_stats['filtered'] += 1
                                    continue
                            bbox_used = bbox
                            if _is_valid_mask(mask) and score_used >= iou_thresh and refine_enabled and int((mask > 0).sum()) >= refine_min_area:
                                tight = _bbox_from_mask(mask)
                                if tight is not None:
                                    tight = _pad_and_clip_bbox(
                                        tight, pad=refine_padding, w=img_w, h=img_h
                                    )
                                if tight is not None:
                                    mask2, iou2, _ = predictor.predict(
                                        box=np.array(tight), multimask_output=multimask
                                    )
                                    best2 = iou2.argmax()
                                    score2 = float(iou2[best2])
                                    if score2 >= score_used:
                                        mask = np.asarray(mask2[best2])
                                        if mask.ndim == 3:
                                            mask = mask[0]
                                        bbox_used = tight
                                        score_used = score2
                            if _is_valid_mask(mask) and erode_px > 0:
                                mask = _erode_mask(mask, erode_px)
                            if not _is_valid_mask(mask):
                                if use_bbox_fallback:
                                    mask = _bbox_to_binary_mask(bbox, img_h, img_w)
                                    score_used = 0.0
                                else:
                                    split_stats['filtered'] += 1
                                    continue
                            masks.append(mask)
                            valid_labels.append(label)
                            valid_bboxes.append(bbox_used)
                            scores.append(score_used)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "CUDA" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            log.warning(f"Failed on {img_path.name} (batch): CUDA OOM")
                        else:
                            log.warning(f"Failed on {img_path.name} (batch): {e}")
                        split_stats['filtered'] += len(valid_boxes_list)
                    except Exception as e:
                        log.warning(f"Failed on {img_path.name} (batch): {e}")
                        split_stats['filtered'] += len(valid_boxes_list)
            else:
                # SAM3: batch path — one forward for all valid bboxes (same pattern as SAM2)
                valid_indices = []
                valid_boxes_list = []
                valid_labels_pre = []
                for idx, (bbox, label) in enumerate(zip(ann['bboxes'], ann['labels'])):
                    if not _is_valid_bbox(bbox, img_w, img_h):
                        split_stats['filtered'] += 1
                        continue
                    valid_indices.append(idx)
                    valid_boxes_list.append(bbox)
                    valid_labels_pre.append(label)
                if valid_boxes_list:
                    chunk_size = max_bbox_batch if max_bbox_batch > 0 else len(valid_boxes_list)
                    all_batch_results = []
                    try:
                        for start in range(0, len(valid_boxes_list), chunk_size):
                            chunk_boxes = valid_boxes_list[start:start + chunk_size]
                            try:
                                bboxes_np = np.array(chunk_boxes, dtype=np.float64)
                                chunk_results = _predict_sam3_batch(predictor, bboxes_np, multimask)
                                all_batch_results.extend(chunk_results)
                            except RuntimeError as e:
                                if ("out of memory" in str(e).lower() or "CUDA" in str(e)) and chunk_size > 1:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    for b in chunk_boxes:
                                        single = _predict_sam3_batch(predictor, np.array([b], dtype=np.float64), multimask)
                                        all_batch_results.extend(single)
                                else:
                                    raise
                        for (bbox, label), (mask, score_used) in zip(
                            zip(valid_boxes_list, valid_labels_pre), all_batch_results
                        ):
                            if not _is_valid_mask(mask):
                                if use_bbox_fallback:
                                    mask = _bbox_to_binary_mask(bbox, img_h, img_w)
                                    score_used = 0.0
                                else:
                                    split_stats['filtered'] += 1
                                    continue
                            bbox_used = bbox
                            if _is_valid_mask(mask) and score_used >= iou_thresh and refine_enabled and int((mask > 0).sum()) >= refine_min_area:
                                tight = _bbox_from_mask(mask)
                                if tight is not None:
                                    tight = _pad_and_clip_bbox(
                                        tight, pad=refine_padding, w=img_w, h=img_h
                                    )
                                if tight is not None:
                                    mask_preds, iou_preds, _ = predictor.predict(
                                        box=np.array(tight), multimask_output=multimask
                                    )
                                    best_idx = iou_preds.argmax()
                                    score2 = float(iou_preds[best_idx])
                                    if score2 >= score_used:
                                        mask = np.asarray(mask_preds[best_idx])
                                        if mask.ndim == 3:
                                            mask = mask[0]
                                        bbox_used = tight
                                        score_used = score2
                            if _is_valid_mask(mask) and erode_px_sam3 > 0:
                                mask = _erode_mask(mask, erode_px_sam3)
                            if not _is_valid_mask(mask):
                                if use_bbox_fallback:
                                    mask = _bbox_to_binary_mask(bbox, img_h, img_w)
                                    score_used = 0.0
                                else:
                                    split_stats['filtered'] += 1
                                    continue
                            masks.append(mask)
                            valid_labels.append(label)
                            valid_bboxes.append(bbox_used)
                            scores.append(score_used)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "CUDA" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            log.warning(f"Failed on {img_path.name} (SAM3 batch): CUDA OOM")
                        else:
                            log.warning(f"Failed on {img_path.name} (SAM3 batch): {e}")
                        split_stats['filtered'] += len(valid_boxes_list)
                    except Exception as e:
                        log.warning(f"Failed on {img_path.name} (SAM3 batch): {e}")
                        split_stats['filtered'] += len(valid_boxes_list)

            split_stats['processed'] += 1
            split_stats['masks'] += len(masks)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save if we have masks, or if keep_all_tiles=true (save empty mask for background tiles)
            if masks or keep_all_tiles:
                # Save
                stem = json_path.stem
                h, w = image.shape[:2]

                # Instance mask: assign each pixel to the mask with highest score that contains it
                # (avoids "last overwrites" so more instances keep pixels → more valid detections)
                instance_mask = np.zeros((h, w), dtype=np.uint16)
                score_map = np.full((h, w), -1.0, dtype=np.float32)
                for i, m in enumerate(masks, 1):
                    mask_binary = np.asarray(m).astype(bool) if m.ndim == 2 else (np.asarray(m).squeeze() > 0)
                    if mask_binary.shape != (h, w):
                        continue
                    s = float(scores[i - 1])
                    better = mask_binary & (s > score_map)
                    instance_mask[better] = i
                    score_map[better] = s
                # Save mask, image and metadata
                meta = {
                    'instances': [
                        {
                            'id': i + 1,
                            'label': label,
                            'class_id': class_map.get(label, 0),
                            'bbox': bbox,
                            'sam2_score': score
                        }
                        for i, (label, bbox, score) in enumerate(
                            zip(valid_labels, valid_bboxes, scores)
                        )
                    ]
                }

                np.save(split_output / f'{stem}_masks.npy', instance_mask)
                cv2.imwrite(str(split_output / f'{stem}.jpg'), image)
                with open(split_output / f'{stem}_meta.json', 'w') as f:
                    json.dump(meta, f, indent=2)

                # LabelMe JSON with polygon shapes
                if save_labelme:
                    if masks:
                        lm_masks, lm_labels = (masks, valid_labels)
                        lm_bboxes = valid_bboxes
                        if merge_labelme_iou > 0:
                            lm_masks, lm_labels = _merge_overlapping_masks(
                                masks, valid_labels, scores, merge_labelme_iou
                            )
                            lm_bboxes = None  # bbox fallback not aligned after merge
                    else:
                        # Empty masks for background tiles (keep_all_tiles=true)
                        lm_masks, lm_labels = ([], [])
                        lm_bboxes = None
                    labelme_data = build_labelme_json(
                        image_path=f'{stem}.jpg',
                        image_height=h,
                        image_width=w,
                        masks=lm_masks,
                        labels=lm_labels,
                        bboxes=lm_bboxes,
                    )
                    with open(split_output / f'{stem}.json', 'w') as f:
                        json.dump(labelme_data, f, indent=2)
        
        stats[split] = split_stats
        if data_paths['train'] == data_paths['test']:
            stats['test'] = split_stats
        log.info(f"{split}: {split_stats['masks']} masks from {split_stats['processed']} images")
        filter_reason = "low IoU/invalid bbox" if use_bbox_fallback else "low IoU/invalid bbox/no mask (fallback disabled)"
        log.info(f"  Filtered ({filter_reason}): {split_stats['filtered']}")

    # When train and test input paths are the same, we only processed once (train).
    # Copy train outputs to test/ so both splits exist for downstream stages.
    if data_paths['train'] == data_paths['test']:
        train_out = output_dir / 'train'
        test_out = output_dir / 'test'
        test_out.mkdir(parents=True, exist_ok=True)
        for f in train_out.iterdir():
            if f.is_file():
                shutil.copy2(f, test_out / f.name)
        log.info("Train and test paths are the same; copied train outputs to test/.")

    log.info("\n" + "=" * 60)
    log.info("Pseudo-mask generation complete!")
    log.info(f"Output: {output_dir}")

    return stats
