"""Stage 5: Validation/evaluation. Run: python run.py stage=evaluate"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_STAGES_DIR = Path(__file__).resolve().parent
_SRC_DIR = _STAGES_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, get_default_config
from dataset import DefectSegmentationDataset
from utils.label_mapping import get_class_labels_for_wandb, load_label_mapping
from model import build_model


def _resolve_paths(cfg: Config) -> None:
    if not cfg.dataset_root.is_absolute():
        cfg.dataset_root = _PROJECT_ROOT / cfg.dataset_root
    if not cfg.checkpoints_dir.is_absolute():
        cfg.checkpoints_dir = _PROJECT_ROOT / cfg.checkpoints_dir


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeepLabV3+ on defect dataset.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to dataset root.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for validation.")
    parser.add_argument("--image-size", type=int, default=None, help="Resize image size.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pth).",
    )
    return parser.parse_args()


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random as _random
    _random.seed(worker_seed)


def create_val_loader(cfg: Config) -> DataLoader:
    val_ds = DefectSegmentationDataset(
        dataset_root=cfg.dataset_root,
        split="val",
        image_size=cfg.image_size,
    )
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, **loader_kwargs)
    return val_loader


def _compute_metrics_per_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> tuple[list[float], list[float], list[int], list[int], list[int]]:
    """Return per-class (ious, dices, intersections, pred_counts, target_counts) for one batch."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    ious = []
    dices = []
    intersections = []
    pred_counts = []
    target_counts = []
    for cls in range(num_classes):
        pred_inds = pred_flat == cls
        target_inds = target_flat == cls
        inter = (pred_inds & target_inds).sum().item()
        pred_c = pred_inds.sum().item()
        tgt_c = target_inds.sum().item()
        union = (pred_inds | target_inds).sum().item()
        intersections.append(inter)
        pred_counts.append(pred_c)
        target_counts.append(tgt_c)
        ious.append(inter / union if union > 0 else float("nan"))
        total = pred_c + tgt_c
        dices.append(2 * inter / total if total > 0 else float("nan"))
    return ious, dices, intersections, pred_counts, target_counts


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict]:
    """Run validation; return (mean_loss, metrics_dict with per-class and aggregated)."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    # Accumulate per-class: intersection, pred_pixels, target_pixels (then IoU/Dice at end)
    inter_sum = [0] * num_classes
    pred_sum = [0] * num_classes
    tgt_sum = [0] * num_classes

    pbar = tqdm(dataloader, desc="Val", leave=False)
    with torch.inference_mode():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            if isinstance(outputs, dict) and "out" in outputs:
                outputs = outputs["out"]

            loss = criterion(outputs, masks)
            running_loss += loss.item()
            num_batches += 1
            pred = torch.argmax(outputs, dim=1)
            _, _, inter, pred_c, tgt_c = _compute_metrics_per_batch(
                pred, masks, num_classes
            )
            for c in range(num_classes):
                inter_sum[c] += inter[c]
                pred_sum[c] += pred_c[c]
                tgt_sum[c] += tgt_c[c]
            pbar.set_postfix(loss=loss.item())

    mean_loss = running_loss / max(num_batches, 1)
    # Per-class IoU and Dice
    iou_per_class = []
    dice_per_class = []
    for c in range(num_classes):
        union_c = pred_sum[c] + tgt_sum[c] - inter_sum[c]
        iou = inter_sum[c] / union_c if union_c > 0 else float("nan")
        total_c = pred_sum[c] + tgt_sum[c]
        dice = 2 * inter_sum[c] / total_c if total_c > 0 else float("nan")
        iou_per_class.append(iou)
        dice_per_class.append(dice)
    # Aggregate: mIoU and mDice over classes that have any gt (ignore nan)
    valid_ious = [x for x in iou_per_class if not (x != x)]
    valid_dices = [x for x in dice_per_class if not (x != x)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")
    mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else float("nan")
    # Overall pixel accuracy
    total_inter = sum(inter_sum)
    total_pixels = sum(pred_sum)
    overall_acc = total_inter / total_pixels if total_pixels > 0 else float("nan")
    metrics = {
        "loss": mean_loss,
        "iou_per_class": iou_per_class,
        "dice_per_class": dice_per_class,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "overall_pixel_acc": overall_acc,
        "inter_sum": inter_sum,
        "pred_sum": pred_sum,
        "tgt_sum": tgt_sum,
    }
    return mean_loss, metrics


def run_validation(cfg) -> None:
    """Run validation loop. cfg must have dataset_root, checkpoints_dir; optional checkpoint path via getattr(cfg, 'checkpoint', None)."""
    _resolve_paths(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    checkpoint_path = getattr(cfg, "checkpoint", None) or (cfg.checkpoints_dir / "best_model.pth")
    if not checkpoint_path.is_absolute():
        checkpoint_path = _PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    logging.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", {})
    # Do not overwrite local paths from checkpoint (often /workspace/... from Docker training).
    _ckpt_skip_paths = frozenset(
        {
            "dataset_root",
            "checkpoints_dir",
            "checkpoint",
            "label_mapping_path",
            "tiled_output_dir",
            "infer_visualizations_dir",
        }
    )
    for k, v in ckpt_cfg.items():
        if k in _ckpt_skip_paths:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if "label_mapping" in checkpoint:
        cfg.num_classes = checkpoint["label_mapping"].get("num_classes", cfg.num_classes)
    else:
        # Fallback to dataset label_mapping.json when checkpoint does not carry mapping metadata.
        dataset_lm = load_label_mapping(cfg.dataset_root)
        if dataset_lm is not None:
            cfg.num_classes = dataset_lm.get("num_classes", cfg.num_classes)
    try:
        model = build_model(cfg, use_pretrained_encoder=False)
    except ValueError as e:
        logging.error("Model build failed: %s", e)
        raise SystemExit(1) from e
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    val_loader = create_val_loader(cfg)
    criterion = nn.CrossEntropyLoss()

    val_loss, metrics = validate(
        model, val_loader, criterion, device, cfg.num_classes
    )
    class_names = get_class_labels_for_wandb(
        load_label_mapping(cfg.dataset_root), cfg.num_classes
    )

    logging.info("Validation loss: %.6f", val_loss)
    logging.info("Mean IoU: %.4f  |  Mean Dice: %.4f  |  Overall pixel acc: %.4f",
                 metrics["mean_iou"], metrics["mean_dice"], metrics["overall_pixel_acc"])
    logging.info("Per-class results:")
    for c in range(cfg.num_classes):
        name = class_names.get(c, f"class_{c}")
        iou = metrics["iou_per_class"][c]
        dice = metrics["dice_per_class"][c]
        iou_str = f"{iou:.4f}" if iou == iou else "n/a"
        dice_str = f"{dice:.4f}" if dice == dice else "n/a"
        logging.info(
            "  %s (id=%d): IoU=%s, Dice=%s  (pred_px=%d, gt_px=%d)",
            name, c, iou_str, dice_str,
            metrics["pred_sum"][c], metrics["tgt_sum"][c],
        )



def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p

def run(cfg) -> None:
    from omegaconf import DictConfig
    # Anchor relative paths to repo root (directory containing run.py), not Hydra's cwd under outputs/.
    root = _resolve_path(cfg.paths.root, _PROJECT_ROOT)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from config import get_config_from_stage
    train_cfg = get_config_from_stage(cfg.stage)
    train_cfg.dataset_root = _resolve_path(train_cfg.dataset_root, root)
    train_cfg.checkpoints_dir = _resolve_path(train_cfg.checkpoints_dir, root)
    if train_cfg.checkpoint is not None:
        train_cfg.checkpoint = _resolve_path(train_cfg.checkpoint, root)
    else:
        train_cfg.checkpoint = _resolve_path(Path("checkpoints/best_model.pth"), root)
    logging.info("evaluate: paths.root=%s dataset_root=%s", root, train_cfg.dataset_root)
    run_validation(train_cfg)
