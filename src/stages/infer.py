"""Stage 4: DeepLabV3+ inference. Run: python run.py stage=infer"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
import json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_STAGES_DIR = Path(__file__).resolve().parent
_SRC_DIR = _STAGES_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import torch

from config import Config, get_default_config, IMAGENET_MEAN, IMAGENET_STD
from utils.label_mapping import get_class_labels_for_wandb, get_overlay_colors, load_label_mapping
from model import build_model


def _resolve_paths(cfg: Config) -> None:
    if not cfg.checkpoints_dir.is_absolute():
        cfg.checkpoints_dir = _PROJECT_ROOT / cfg.checkpoints_dir
    if not cfg.dataset_root.is_absolute():
        cfg.dataset_root = _PROJECT_ROOT / cfg.dataset_root


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained DeepLabV3+ model.")
    parser.add_argument("--image", type=Path, default=None, help="Path to a single input image.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder of images to process (mutually exclusive with --image).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for a single output mask (used with --image).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        help="Output folder for masks (default: predictions). Used with --input-dir or as base when --output not set.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resize size (overrides config.image_size if provided).",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        default=None,
        help="Use tiled inference (tile full image, merge to full res). Overrides config.",
    )
    parser.add_argument(
        "--no-tiled",
        action="store_true",
        help="Disable tiled inference; resize full image to image_size instead.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for tiled inference (default: from config, e.g. 1024).",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Overlap between tiles (default: from config, e.g. 256).",
    )
    args = parser.parse_args()
    if args.image is None and args.input_dir is None:
        parser.error("Provide either --image or --input-dir.")
    if args.image is not None and args.input_dir is not None:
        parser.error("Use either --image or --input-dir, not both.")
    return args


def load_model_and_mapping(
    cfg: Config, checkpoint_path: Path
) -> tuple[torch.nn.Module, dict[int, str], dict[int, tuple[int, int, int]]]:
    """Load model from checkpoint; return model, class_names (id->name), overlay_colors (id->BGR). Class names/colors from checkpoint label_mapping or dataset label_mapping.json."""
    logging.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Align model build/runtime fields with training config from checkpoint.
    ckpt_cfg = checkpoint.get("config", {})
    for key in (
        "architecture",
        "encoder_name",
        "encoder_weights",
        "activation",
        "in_channels",
        "encoder_output_stride",
        "decoder_channels",
        "decoder_atrous_rates",
        "image_size",
        "num_classes",
    ):
        if key in ckpt_cfg and getattr(cfg, key, None) is not None:
            setattr(cfg, key, ckpt_cfg[key])
    label_mapping = checkpoint.get("label_mapping")
    if label_mapping:
        cfg.num_classes = label_mapping.get("num_classes", cfg.num_classes)
    # Fallback 1: explicit label_mapping_path if provided
    if not label_mapping and getattr(cfg, "label_mapping_path", None):
        mapping_path = Path(cfg.label_mapping_path)
        if not mapping_path.is_absolute():
            mapping_path = _PROJECT_ROOT / mapping_path
        mapping = None
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
        except Exception:
            mapping = None
        if mapping and mapping.get("id_to_label"):
            # Normalize id_to_label keys to int and compute num_classes
            id_to_label_raw = mapping["id_to_label"]
            id_to_label = {}
            for k, v in id_to_label_raw.items():
                try:
                    id_to_label[int(k)] = str(v)
                except (ValueError, TypeError):
                    continue
            if id_to_label:
                num_classes = max(id_to_label.keys(), default=0) + 1
                cfg.num_classes = num_classes
            label_mapping = {
                "num_classes": cfg.num_classes,
                "id_to_label": id_to_label,
                "label_to_id": mapping.get("label_to_id", {}),
            }
    # Fallback 2: dataset_root/label_mapping.json (legacy behavior)
    if not label_mapping and getattr(cfg, "dataset_root", None):
        label_mapping = load_label_mapping(cfg.dataset_root)
        if label_mapping:
            cfg.num_classes = label_mapping["num_classes"]
    class_names = get_class_labels_for_wandb(label_mapping, cfg.num_classes)
    overlay_colors = get_overlay_colors(cfg.num_classes)
    try:
        model = build_model(cfg, use_pretrained_encoder=False)
    except ValueError as e:
        logging.error("Model build failed: %s", e)
        raise SystemExit(1) from e
        
    # Strip "module." from state dict keys if checkpoint was saved with DataParallel
    state_dict = checkpoint["model_state_dict"]
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            unwrapped_state_dict[k[7:]] = v
        else:
            unwrapped_state_dict[k] = v
            
    model.load_state_dict(unwrapped_state_dict)
    model.eval()
    return model, class_names, overlay_colors


def _apply_clahe_rgb(
    image_rgb: np.ndarray,
    clip_limit: float,
    tile_grid: tuple[int, int],
) -> np.ndarray:
    """Apply CLAHE to an RGB image (on L channel in LAB)."""
    img_8u = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_8u, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_8u = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_8u.astype(np.float32) / 255.0


def _cfg_clahe_grid(cfg: Config) -> tuple[int, int]:
    tg = getattr(cfg, "clahe_tile_grid", (8, 8))
    if isinstance(tg, (list, tuple)) and len(tg) >= 2:
        return int(tg[0]), int(tg[1])
    return 8, 8


def preprocess_image(image_bgr: np.ndarray, image_size: int, cfg: Config | None = None) -> torch.Tensor:
    """Preprocess a BGR image array for inference (simple square resize + optional CLAHE + normalize).
    Matches dataset.py __getitem__ preprocessing used during training."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if cfg is not None and getattr(cfg, "clahe_enabled", False):
        clip_limit = float(getattr(cfg, "clahe_clip_limit", 2.0))
        grid = getattr(cfg, "clahe_tile_grid", (8, 8))
        img = _apply_clahe_rgb(img, clip_limit, grid)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def _preprocess_tile(
    crop_bgr: np.ndarray,
    tile_size: int,
    cfg: Config | None = None,
    *,
    skip_clahe: bool = False,
) -> torch.Tensor:
    """Normalize a BGR crop (possibly smaller than tile_size) and pad to (tile_size, tile_size)."""
    h, w = crop_bgr.shape[:2]
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if (
        cfg is not None
        and getattr(cfg, "clahe_enabled", False)
        and not skip_clahe
    ):
        clip_limit = float(getattr(cfg, "clahe_clip_limit", 2.0))
        grid = _cfg_clahe_grid(cfg)
        img = _apply_clahe_rgb(img, clip_limit, grid)
    if h < tile_size or w < tile_size:
        # Use reflect padding (matches prepare_dataset._pad_tile which uses BORDER_REFLECT_101)
        img = np.pad(
            img,
            ((0, max(0, tile_size - h)), (0, max(0, tile_size - w)), (0, 0)),
            mode="reflect",
        )
    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()


def _get_tile_positions(
    height: int, width: int, tile_size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """
    Return list of (sx, sy, crop_w, crop_h) for each tile.

    Logic mirrors the original SAM2 tiling in stages/tiling.py:
    - stride = tile_size - overlap
    - for each step, compute x2 = min(step + tile_size, width)
      and x1 = max(0, x2 - tile_size) so tiles at the right/bottom
      edges are still tile_size wide/tall and are shifted left/up.
    """
    stride = max(1, tile_size - overlap)
    positions: list[tuple[int, int, int, int]] = []
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


def postprocess_mask(
    mask_logits: torch.Tensor,
    orig_shape: tuple[int, int],
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    probs = torch.softmax(mask_logits, dim=1)
    max_probs, pred = torch.max(probs, dim=1)
    pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    max_probs = max_probs.squeeze(0).cpu().numpy()
    if confidence_threshold > 0:
        pred[max_probs < confidence_threshold] = 0
    h, w = orig_shape
    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_resized

def _collect_images(path: Path) -> list[Path]:
    """Return list of image paths under path (file or directory)."""
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _make_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    overlay_colors: dict[int, tuple[int, int, int]],
    class_names: dict[int, str] | None = None,
    alpha: float = 0.55,
    draw_bbox: bool = True,
) -> np.ndarray:
    """Overlay colored mask on original image; optionally draw label names and bboxes per class."""
    h, w = image_bgr.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)
    overlay = image_bgr.astype(np.float32)
    colored = np.zeros_like(overlay)
    for class_id, bgr in overlay_colors.items():
        if class_id == 0:
            continue
        use_bgr = bgr if bgr != (0, 0, 0) else (0, 255, 0)
        colored[mask == class_id] = use_bgr
    mask_visible = (mask > 0).astype(np.float32)[:, :, np.newaxis]
    blended = overlay * (1.0 - alpha * mask_visible) + colored * (alpha * mask_visible)
    out = np.clip(blended, 0, 255).astype(np.uint8)

    if not draw_bbox and not class_names:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(1.2, w / 1200.0))
    thickness = max(1, int(round(scale * 2)))
    for class_id in sorted(overlay_colors.keys()):
        if class_id == 0:
            continue
        ys, xs = np.where(mask == class_id)
        if ys.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bgr = overlay_colors.get(class_id, (0, 255, 0))
        if bgr == (0, 0, 0):
            bgr = (0, 255, 0)
        if draw_bbox:
            cv2.rectangle(out, (x_min, y_min), (x_max, y_max), bgr, thickness)
        label = (class_names or {}).get(class_id, f"class_{class_id}")
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx, ty = x_min, y_min - 4
        if ty - th < 0:
            ty = y_min + th + 4
        cv2.rectangle(out, (tx, ty - th), (tx + tw + 4, ty + 2), bgr, -1)
        cv2.putText(out, label, (tx + 2, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _count_classes(mask: np.ndarray) -> dict[int, int]:
    """Return dict of class_id -> pixel count for classes present in mask."""
    unique, counts = np.unique(mask, return_counts=True)
    return dict(zip((int(u) for u in unique), (int(c) for c in counts)))


def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    cfg: Config,
    device: torch.device,
    overlay_colors: dict[int, tuple[int, int, int]],
    class_names: dict[int, str] | None = None,
    tiled_save_dir: Path | None = None,
    vis_dir: Path | None = None,
    save_masks: bool = True,
    save_visualizations: bool = True,
) -> tuple[float, np.ndarray]:
    """Run model on one image; save raw mask and overlay at full res. Uses tiling when tiled_inference=True and image is large."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image '{image_path}'.")

    # Match prepare: full-image CLAHE on BGR before white-object crop; then skip per-tile CLAHE.
    clahe_full_applied = False
    if getattr(cfg, "clahe_enabled", False):
        import sys
        _PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(_PROJECT_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT / "src"))
        from utils.image_utils import apply_clahe_bgr

        clip = float(getattr(cfg, "clahe_clip_limit", 2.0))
        img_bgr = apply_clahe_bgr(img_bgr, clip, _cfg_clahe_grid(cfg))
        clahe_full_applied = True

    crop_enabled = getattr(cfg, "crop_object_enabled", False)
    if crop_enabled:
        import sys
        _PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(_PROJECT_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT / "src"))
        from utils.image_utils import get_object_crop_bbox
        padding = getattr(cfg, "crop_object_padding", 0)
        cx1, cy1, cx2, cy2 = get_object_crop_bbox(img_bgr, padding=padding)
        img_bgr = img_bgr[cy1:cy2, cx1:cx2]

    orig_h, orig_w = img_bgr.shape[:2]
    tile_size = getattr(cfg, "tile_size", 1024)
    tile_overlap = getattr(cfg, "tile_overlap", 256)
    # When tiled_inference is True, always use tiled path (same 1024 tiles + merge for any size)
    use_tiled = getattr(cfg, "tiled_inference", False)
    tile_vis_dir: Path | None = None
    tile_positions: list[tuple[int, int, int, int]] | None = None

    t0 = time.perf_counter()
    if use_tiled:
        if tiled_save_dir is not None:
            tile_vis_dir = tiled_save_dir / image_path.stem
            tile_vis_dir.mkdir(parents=True, exist_ok=True)
        positions = _get_tile_positions(orig_h, orig_w, tile_size, tile_overlap)
        tile_positions = positions
        num_classes = cfg.num_classes
        prob_sum = np.zeros((orig_h, orig_w, num_classes), dtype=np.float32)
        count_map = np.zeros((orig_h, orig_w, 1), dtype=np.float32)
        for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(positions):
            crop = img_bgr[sy : sy + crop_h, sx : sx + crop_w]

            # Optionally save 1024x1024 tile image for inspection
            if tile_vis_dir is not None:
                tile_img = crop.copy()
                pad_h = tile_size - tile_img.shape[0]
                pad_w = tile_size - tile_img.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile_img = cv2.copyMakeBorder(
                        tile_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )
                tile_img_path = tile_vis_dir / f"{image_path.stem}_t{tile_idx:04d}.png"
                cv2.imwrite(str(tile_img_path), tile_img)
            tile_tensor = _preprocess_tile(
                crop, tile_size, cfg, skip_clahe=clahe_full_applied
            ).to(device)
            tile_tensor = tile_tensor.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                out = model(tile_tensor)
                if isinstance(out, dict) and "out" in out:
                    out = out["out"]
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()  # (C, Ht, Wt)
            probs = np.transpose(probs, (1, 2, 0))  # (Ht, Wt, C)
            ph, pw = probs.shape[:2]
            if (ph, pw) != (crop_h, crop_w):
                # Slice off the padded region (padding was added at bottom/right)
                probs = probs[:crop_h, :crop_w, :]
            prob_sum[sy : sy + crop_h, sx : sx + crop_w, :] += probs
            count_map[sy : sy + crop_h, sx : sx + crop_w, :] += 1
        prob_avg = prob_sum / np.maximum(count_map, 1e-6)
        mask = np.argmax(prob_avg, axis=-1).astype(np.uint8)
        conf_thresh = float(getattr(cfg, "infer_confidence", 0.0))
        if conf_thresh > 0:
            max_prob = np.max(prob_avg, axis=-1)
            mask[max_prob < conf_thresh] = 0

        if positions:
            logging.info(
                "Tiled inference: %dx%d image -> %d tiles (size=%d, overlap=%d)",
                orig_w, orig_h, len(positions), tile_size, tile_overlap,
            )
    else:
        # Cropped img_bgr; CLAHE already on full image before crop when clahe_enabled (prepare order).
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if getattr(cfg, "clahe_enabled", False) and not clahe_full_applied:
            clip_limit = float(getattr(cfg, "clahe_clip_limit", 2.0))
            img_rgb = _apply_clahe_rgb(img_rgb, clip_limit, _cfg_clahe_grid(cfg))
        # Simple square resize — matches dataset.py __getitem__ used during training
        img_rgb = cv2.resize(img_rgb, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            outputs = model(input_tensor)
            if isinstance(outputs, dict) and "out" in outputs:
                outputs = outputs["out"]
        conf_thresh = float(getattr(cfg, "infer_confidence", 0.0))
        mask = postprocess_mask(outputs, (orig_h, orig_w), confidence_threshold=conf_thresh)
    # Per-tile debug masks (same tiles as tile images)
    if tile_vis_dir is not None and tile_positions:
        for tile_idx, (sx, sy, crop_w, crop_h) in enumerate(tile_positions):
            tile_mask = mask[sy : sy + crop_h, sx : sx + crop_w]
            pad_h = tile_size - tile_mask.shape[0]
            pad_w = tile_size - tile_mask.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile_mask = cv2.copyMakeBorder(
                    tile_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
                )
            tile_mask_path = tile_vis_dir / f"{image_path.stem}_t{tile_idx:04d}_mask.png"
            cv2.imwrite(str(tile_mask_path), tile_mask)

    elapsed = time.perf_counter() - t0

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        if not cv2.imwrite(str(output_path), mask):
            raise RuntimeError(f"Failed to write mask to '{output_path}'.")
    if save_visualizations:
        vis_root = vis_dir if vis_dir is not None else out_dir
        vis_root.mkdir(parents=True, exist_ok=True)
        # Keep overlay name aligned with input image name when possible.
        overlay_path = vis_root / image_path.name
        # Safety: if this would overwrite the raw mask path, fall back to suffix form.
        if save_masks and overlay_path.resolve() == output_path.resolve():
            overlay_path = vis_root / f"{output_path.stem}_overlay{image_path.suffix or '.png'}"
        show_labels = bool(getattr(cfg, "show_overlay_labels", False))
        overlay_img = _make_overlay(
            img_bgr,
            mask,
            overlay_colors,
            class_names=class_names if show_labels else None,
            draw_bbox=False,
        )
        if not cv2.imwrite(str(overlay_path), overlay_img):
            raise RuntimeError(f"Failed to write overlay to '{overlay_path}'.")
    return elapsed, mask


def run_inference_stage(
    cfg: Config,
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    tiled: bool = True,
    image_paths: list | None = None,
) -> None:
    """Run inference on images. If image_paths is given use it; else collect from input_dir. Saves masks to output_dir."""
    setup_logging()
    cfg.tiled_inference = tiled
    _resolve_paths(cfg)

    if not checkpoint_path.is_absolute():
        checkpoint_path = _PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model, class_names, overlay_colors = load_model_and_mapping(cfg, checkpoint_path)
    logging.info(
        "Inference config: num_classes=%d, image_size=%s, tiled_inference=%s, tile_size=%s, tile_overlap=%s",
        cfg.num_classes,
        getattr(cfg, "image_size", "?"),
        getattr(cfg, "tiled_inference", False),
        getattr(cfg, "tile_size", "?"),
        getattr(cfg, "tile_overlap", "?"),
    )
    model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_absolute():
        input_dir = _PROJECT_ROOT / input_dir
    input_dir = input_dir.resolve()

    if image_paths is None:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: '{input_dir}'.")
        image_paths = _collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{input_dir}'.")
    logging.info("Running inference on %d images -> %s", len(image_paths), output_dir)

    save_visualizations = bool(getattr(cfg, "save_visualizations", True))
    save_tiled_outputs = bool(getattr(cfg, "save_tiled_outputs", False))
    save_masks = bool(getattr(cfg, "save_masks", True))
    if not save_visualizations and save_tiled_outputs:
        logging.info(
            "save_visualizations=false: skipping tiled visualization outputs "
            "(save_tiled_outputs=%s).",
            save_tiled_outputs,
        )

    tiled_save_dir = None
    if (
        save_visualizations
        and getattr(cfg, "tiled_inference", False)
        and save_tiled_outputs
    ):
        tiled_root = getattr(cfg, "tiled_output_dir", None)
        if tiled_root is not None:
            tiled_save_dir = tiled_root if tiled_root.is_absolute() else _PROJECT_ROOT / tiled_root

    vis_dir: Path | None = None
    vis_root = getattr(cfg, "infer_visualizations_dir", None)
    if save_visualizations and vis_root is not None:
        vis_dir = vis_root if vis_root.is_absolute() else _PROJECT_ROOT / vis_root
        vis_dir = vis_dir.resolve()
        vis_dir.mkdir(parents=True, exist_ok=True)

    def _log_inference(
        img_name: str, out_path: Path, elapsed: float, mask: np.ndarray, names: dict[int, str]
    ) -> None:
        counts = _count_classes(mask)
        classes_present = sorted(counts.keys())
        name_list = [names.get(c, f"class_{c}") for c in classes_present]
        n_classes = len(classes_present)
        overlay_root = vis_dir if vis_dir is not None else out_path.parent
        overlay_path = overlay_root / f"{out_path.stem}_overlay.png"
        mask_log = str(out_path) if save_masks else "(disabled)"
        logging.info(
            "Saved %s -> mask %s%s | inference %.3fs | %d class(es): %s",
            img_name,
            mask_log,
            f", overlay {overlay_path}" if save_visualizations else "",
            elapsed,
            n_classes,
            ", ".join(f"{name_list[i]}={counts[c]}px" for i, c in enumerate(classes_present)),
        )
        fg_pixels = sum(counts.get(c, 0) for c in counts if c != 0)
        if fg_pixels == 0 and n_classes == 1 and 0 in counts:
            if not getattr(_log_inference, "_warned_bg", False):
                _log_inference._warned_bg = True
                logging.warning(
                    "Only background predicted for %s (and possibly others). Val uses single 1024 resize per image; "
                    "inference may use tiling (full-res crops) so inputs differ. Try --no-tiled to match val.",
                    img_name,
                )

    total_time = 0.0
    n_ok = 0
    n_with_fg = 0
    for img_path in image_paths:
        out_path = output_dir / f"{img_path.stem}.png"
        try:
            elapsed, mask = run_inference(
                model,
                img_path,
                out_path,
                cfg,
                device,
                overlay_colors,
                class_names,
                tiled_save_dir=tiled_save_dir,
                vis_dir=vis_dir,
                save_masks=save_masks,
                save_visualizations=save_visualizations,
            )
            total_time += elapsed
            n_ok += 1
            if (mask > 0).any():
                n_with_fg += 1
            _log_inference(img_path.name, out_path, elapsed, mask, class_names)
        except Exception as e:
            logging.error("Failed %s: %s", img_path, e)
    if n_ok:
        logging.info(
            "Total inference time: %.3fs for %d images (avg %.3fs/image); %d images had foreground predicted",
            total_time, n_ok, total_time / n_ok, n_with_fg,
        )


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
    from config import get_config_from_stage
    train_cfg = get_config_from_stage(cfg.stage)
    input_dir = _resolve_path(getattr(cfg.stage, "input_dir", "dataset/defect_data/images/val"), root)
    output_dir = _resolve_path(getattr(cfg.stage, "output_dir", "outputs/stage4_infer"), root)
    checkpoint = _resolve_path(getattr(cfg.stage, "checkpoint", "checkpoints/best_model.pth"), root)
    tiled = bool(getattr(cfg.stage, "tiled", True))
    run_inference_stage(train_cfg, input_dir, output_dir, checkpoint, tiled=tiled)
