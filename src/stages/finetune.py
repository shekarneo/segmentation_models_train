"""Stage 3: DeepLabV3+ training. Run: python run.py stage=finetune"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

_STAGES_DIR = Path(__file__).resolve().parent
_SRC_DIR = _STAGES_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import segmentation_models_pytorch as smp
import cv2

from config import Config, get_default_config, IMAGENET_MEAN, IMAGENET_STD, apply_model_profile
from dataset import DefectSegmentationDataset
from utils.label_mapping import get_class_labels_for_wandb, load_label_mapping
from model import build_model

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on defect dataset.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to dataset root.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size", "--batch_size",
        type=int, default=None, dest="batch_size", help="Batch size (sweep uses --batch_size).",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", "--weight_decay",
        type=float, default=None, dest="weight_decay", help="Weight decay (sweep uses --weight_decay).",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Image size for resizing.")
    parser.add_argument(
        "--encoder-name", "--encoder_name",
        type=str,
        default=None,
        dest="encoder_name",
        help="Encoder backbone (sweep uses --encoder_name). E.g. resnet50, resnet101, mit_b5, efficientnet-b7, convnext_base.",
    )
    parser.add_argument(
        "--encoder-weights", "--encoder_weights",
        type=str,
        default=None,
        dest="encoder_weights",
        help="Pretrained encoder weights (e.g. imagenet). Sweep uses --encoder_weights.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained encoder weights (train encoder from scratch).",
    )
    return parser.parse_args()


def _get_model_core(model: nn.Module) -> nn.Module:
    """Return the underlying model (strip DataParallel or DDP wrapper)."""
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def _resolve_paths(cfg: Config) -> None:
    """Resolve relative paths against project root."""
    if not cfg.dataset_root.is_absolute():
        cfg.dataset_root = _PROJECT_ROOT / cfg.dataset_root
    if not cfg.checkpoints_dir.is_absolute():
        cfg.checkpoints_dir = _PROJECT_ROOT / cfg.checkpoints_dir


def _apply_wandb_config_to_cfg(cfg: Config) -> None:
    """Apply wandb.config to cfg only when running under a WandB sweep."""
    if wandb is None or not wandb.run:
        return
    # Avoid accidental config mutation for normal runs.
    # We only want this behavior for actual sweep agents.
    sweep_id = getattr(wandb.run, "sweep_id", None) or os.environ.get("WANDB_SWEEP_ID")
    if not sweep_id:
        return
    # Keys that may be set by a sweep; map wandb key -> cfg attribute
    sweep_map = [
        ("model_profile", "model_profile"),
        ("lr", "learning_rate"),
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("image_size", "image_size"),
        ("weight_decay", "weight_decay"),
        ("encoder_name", "encoder_name"),
        ("encoder_weights", "encoder_weights"),
        ("use_early_stopping", "use_early_stopping"),
        ("early_stop_patience", "early_stop_patience"),
        ("early_stop_min_delta", "early_stop_min_delta"),
        ("early_stop_metric", "early_stop_metric"),
        ("limit_background_samples", "limit_background_samples"),
        ("max_background_to_foreground_ratio", "max_background_to_foreground_ratio"),
        ("use_class_weights", "use_class_weights"),
        ("use_focal_loss", "use_focal_loss"),
        ("use_dice_loss", "use_dice_loss"),
        ("ce_loss_weight", "ce_loss_weight"),
        ("focal_loss_weight", "focal_loss_weight"),
        ("dice_loss_weight", "dice_loss_weight"),
        ("focal_gamma", "focal_gamma"),
        # Optimizer / scheduler
        ("optimizer_name", "optimizer_name"),
        ("encoder_lr_scale", "encoder_lr_scale"),
        ("scheduler_name", "scheduler_name"),
        ("warmup_pct", "warmup_pct"),
        ("grad_clip", "grad_clip"),
        # Augmentations
        ("augmentations_enabled", "augmentations_enabled"),
        ("aug_hflip_prob", "aug_hflip_prob"),
        ("aug_vflip_prob", "aug_vflip_prob"),
        ("aug_rotate90_prob", "aug_rotate90_prob"),
        ("aug_brightness", "aug_brightness"),
        ("aug_contrast", "aug_contrast"),
        ("aug_noise_std", "aug_noise_std"),
        ("aug_brightness_contrast_prob", "aug_brightness_contrast_prob"),
        ("aug_shift_scale_rotate_prob", "aug_shift_scale_rotate_prob"),
        ("aug_shift_limit", "aug_shift_limit"),
        ("aug_scale_limit", "aug_scale_limit"),
        ("aug_rotate_limit", "aug_rotate_limit"),
        ("aug_clahe_prob", "aug_clahe_prob"),
        ("aug_clahe_clip", "aug_clahe_clip"),
        ("aug_gauss_noise_prob", "aug_gauss_noise_prob"),
        ("aug_gaussian_blur_prob", "aug_gaussian_blur_prob"),
        ("aug_gaussian_blur_kernel", "aug_gaussian_blur_kernel"),
        ("aug_random_crop_defect_prob", "aug_random_crop_defect_prob"),
        ("aug_random_crop_defect_size", "aug_random_crop_defect_size"),
    ]
    for wb_key, cfg_key in sweep_map:
        if not hasattr(cfg, cfg_key):
            continue
        val = getattr(wandb.config, wb_key, None)
        if val is not None:
            setattr(cfg, cfg_key, val)


def _wandb_training_config_dict(cfg: Config) -> dict:
    """Hyperparameters logged to WandB; must reflect cfg after model_profile / sweep apply."""
    return {
        "model_profile": getattr(cfg, "model_profile", None),
        "architecture": getattr(cfg, "architecture", None),
        "batch_size": cfg.batch_size,
        "limit_background_samples": getattr(cfg, "limit_background_samples", False),
        "max_background_to_foreground_ratio": getattr(
            cfg, "max_background_to_foreground_ratio", None
        ),
        "use_class_weights": getattr(cfg, "use_class_weights", True),
        "use_focal_loss": getattr(cfg, "use_focal_loss", True),
        "use_dice_loss": getattr(cfg, "use_dice_loss", True),
        "ce_loss_weight": getattr(cfg, "ce_loss_weight", 1.0),
        "focal_loss_weight": getattr(cfg, "focal_loss_weight", 1.0),
        "dice_loss_weight": getattr(cfg, "dice_loss_weight", 1.0),
        "focal_gamma": getattr(cfg, "focal_gamma", 2.0),
        "optimizer_name": getattr(cfg, "optimizer_name", "adamw"),
        "encoder_lr_scale": getattr(cfg, "encoder_lr_scale", 0.1),
        "scheduler_name": getattr(cfg, "scheduler_name", "onecycle"),
        "warmup_pct": getattr(cfg, "warmup_pct", 0.1),
        "grad_clip": getattr(cfg, "grad_clip", 1.0),
        "lr": cfg.learning_rate,
        "epochs": cfg.epochs,
        "image_size": cfg.image_size,
        "num_classes": cfg.num_classes,
        "encoder_name": cfg.encoder_name,
        "encoder_weights": cfg.encoder_weights,
        "weight_decay": cfg.weight_decay,
        "use_early_stopping": getattr(cfg, "use_early_stopping", False),
        "early_stop_patience": getattr(cfg, "early_stop_patience", 10),
        "early_stop_min_delta": getattr(cfg, "early_stop_min_delta", 0.0),
        "early_stop_metric": getattr(cfg, "early_stop_metric", "val_loss"),
    }


def _seed_worker(worker_id: int) -> None:
    """Seed numpy and random in each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _compute_class_stats(
    train_ds: DefectSegmentationDataset, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (class_pixel_counts, class_weights) for severe class imbalance handling."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, mask_path in getattr(train_ds, "samples", []):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        vals, cts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, cts):
            v_int = int(v)
            if 0 <= v_int < num_classes:
                counts[v_int] += int(c)

    total_pixels = int(counts.sum())
    weights = np.ones(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any() and total_pixels > 0:
        weights[nonzero] = total_pixels / (
            float(num_classes) * counts[nonzero].astype(np.float32)
        )
        # Clip weights to avoid extreme values dominating the loss
        weights = np.clip(weights, a_min=0.1, a_max=50.0)
        mean_w = float(weights[nonzero].mean())
        if mean_w > 0:
            weights /= mean_w
    return counts, weights


def create_dataloaders(
    cfg: Config,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    # Build on-the-fly augmentation for train split
    train_transform = None
    if getattr(cfg, "augmentations_enabled", False):
        hflip_p = float(getattr(cfg, "aug_hflip_prob", 0.5))
        vflip_p = float(getattr(cfg, "aug_vflip_prob", 0.0))
        rot90_p = float(getattr(cfg, "aug_rotate90_prob", 0.0))
        bright_delta = float(getattr(cfg, "aug_brightness", 0.0))
        contrast_delta = float(getattr(cfg, "aug_contrast", 0.0))
        noise_std = float(getattr(cfg, "aug_noise_std", 0.0))
        bc_prob = float(getattr(cfg, "aug_brightness_contrast_prob", 0.5))
        ssr_p = float(getattr(cfg, "aug_shift_scale_rotate_prob", 0.5))
        shift_lim = float(getattr(cfg, "aug_shift_limit", 0.05))
        scale_lim = float(getattr(cfg, "aug_scale_limit", 0.1))
        rot_lim = float(getattr(cfg, "aug_rotate_limit", 15.0))
        clahe_p = float(getattr(cfg, "aug_clahe_prob", 0.0))
        clahe_clip = float(getattr(cfg, "aug_clahe_clip", 2.0))
        gauss_noise_p = float(getattr(cfg, "aug_gauss_noise_prob", 0.0))
        gauss_blur_p = float(getattr(cfg, "aug_gaussian_blur_prob", 0.0))
        gauss_blur_k = int(getattr(cfg, "aug_gaussian_blur_kernel", 3))
        crop_defect_p = float(getattr(cfg, "aug_random_crop_defect_prob", 0.0))
        crop_defect_size = int(getattr(cfg, "aug_random_crop_defect_size", 512))

        def _train_transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # image: (C,H,W), mask: (H,W), currently normalized with IMAGENET_MEAN/STD.
            # Denormalize to [0,1] for spatial/color augmentations.
            mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
            img = image * std + mean  # [0,1] approximately

            # Convert to numpy for cv2-based ops
            img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # H,W,C
            mask_np = mask.cpu().numpy().astype(np.uint8)  # H,W
            h, w = mask_np.shape[:2]

            # Random Crop Around Defect
            if crop_defect_p > 0.0 and np.random.rand() < crop_defect_p:
                ys, xs = np.where(mask_np > 0)
                if ys.size > 0:
                    idx = np.random.randint(0, ys.size)
                    cy, cx = ys[idx], xs[idx]
                    cw = min(crop_defect_size, w)
                    ch = min(crop_defect_size, h)
                    x1 = max(0, cx - cw // 2)
                    y1 = max(0, cy - ch // 2)
                    x2 = x1 + cw
                    y2 = y1 + ch
                    if x2 > w:
                        x2 = w
                        x1 = w - cw
                    if y2 > h:
                        y2 = h
                        y1 = h - ch
                    img_np = img_np[y1:y2, x1:x2]
                    mask_np = mask_np[y1:y2, x1:x2]
                    img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

            # Flips
            if hflip_p > 0.0 and np.random.rand() < hflip_p:
                img_np = np.flip(img_np, axis=1)
                mask_np = np.flip(mask_np, axis=1)
            if vflip_p > 0.0 and np.random.rand() < vflip_p:
                img_np = np.flip(img_np, axis=0)
                mask_np = np.flip(mask_np, axis=0)

            # RandomRotate90
            if rot90_p > 0.0 and np.random.rand() < rot90_p:
                k = np.random.randint(0, 4)
                if k > 0:
                    img_np = np.rot90(img_np, k, axes=(0, 1))
                    mask_np = np.rot90(mask_np, k, axes=(0, 1))
                    h, w = mask_np.shape[:2]

            # ShiftScaleRotate (approx Albumentations)
            if ssr_p > 0.0 and np.random.rand() < ssr_p:
                shift_x = (np.random.rand() * 2.0 - 1.0) * shift_lim * w
                shift_y = (np.random.rand() * 2.0 - 1.0) * shift_lim * h
                scale = 1.0 + (np.random.rand() * 2.0 - 1.0) * scale_lim
                angle = (np.random.rand() * 2.0 - 1.0) * rot_lim
                center = (w / 2.0, h / 2.0)
                M = cv2.getRotationMatrix2D(center, angle, scale)
                M[0, 2] += shift_x
                M[1, 2] += shift_y
                img_np = cv2.warpAffine(
                    img_np,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                mask_np = cv2.warpAffine(
                    mask_np,
                    M,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

            # RandomBrightnessContrast (global jitter)
            if (bright_delta > 0.0 or contrast_delta > 0.0) and bc_prob > 0.0 and np.random.rand() < bc_prob:
                if contrast_delta > 0.0:
                    alpha = 1.0 + (np.random.rand() * 2.0 - 1.0) * contrast_delta
                else:
                    alpha = 1.0
                if bright_delta > 0.0:
                    beta = (np.random.rand() * 2.0 - 1.0) * bright_delta
                else:
                    beta = 0.0
                img_np = img_np * alpha + beta

            # CLAHE (approx Albumentations.CLAHE)
            if clahe_p > 0.0 and np.random.rand() < clahe_p:
                img_8u = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                lab = cv2.cvtColor(img_8u, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                img_8u = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                img_np = img_8u.astype(np.float32) / 255.0

            # Gaussian blur
            if gauss_blur_p > 0.0 and gauss_blur_k > 1 and gauss_blur_k % 2 == 1 and np.random.rand() < gauss_blur_p:
                img_np = cv2.GaussianBlur(img_np, (gauss_blur_k, gauss_blur_k), 0)

            # Gauss noise in [0,1] space
            if noise_std > 0.0 and gauss_noise_p > 0.0 and np.random.rand() < gauss_noise_p:
                img_np = img_np + np.random.normal(0.0, noise_std, img_np.shape).astype(np.float32)

            # Clamp to [0,1] after appearance aug
            img_np = np.clip(img_np, 0.0, 1.0)

            # Back to tensor and re-normalize
            img = torch.from_numpy(img_np).permute(2, 0, 1).to(image.device, dtype=image.dtype)
            mask_t = torch.from_numpy(mask_np.astype(np.int64)).to(mask.device)
            img = (img - mean) / std

            return img, mask_t

        train_transform = _train_transform

    train_ds = DefectSegmentationDataset(
        dataset_root=cfg.dataset_root,
        split="train",
        image_size=cfg.image_size,
        transform=train_transform,
        limit_background_samples=getattr(cfg, "limit_background_samples", False),
        max_background_to_foreground_ratio=getattr(cfg, "max_background_to_foreground_ratio", None),
        background_keep_fraction=getattr(cfg, "background_keep_fraction", None),
    )
    val_ds = DefectSegmentationDataset(
        dataset_root=cfg.dataset_root,
        split="val",
        image_size=cfg.image_size,
    )

    # When using DDP, DistributedSampler partitions data across ranks; set_epoch(epoch) is called each epoch for proper shuffle.
    is_distributed = world_size > 1
    train_sampler: Optional[DistributedSampler] = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_distributed
        else None
    )
    train_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
        drop_last=True,  # avoid BatchNorm "expected >1 value per channel" when last batch has size 1
    )
    if train_sampler is not None:
        train_loader_kwargs["sampler"] = train_sampler
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True
    val_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=_seed_worker if cfg.num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)
    return train_loader, val_loader, train_sampler


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    cfg: Config,
    label_mapping: Optional[Dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_for_save = _get_model_core(model)
    state: Dict = {
        "epoch": epoch,
        "model_state_dict": model_for_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": cfg.__dict__,
    }
    if label_mapping is not None:
        state["label_mapping"] = label_mapping
    torch.save(state, path)
    logging.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: Config,
) -> Tuple[int, float]:
    logging.info("Loading checkpoint from %s", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target = _get_model_core(model)
    target.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore config fields if they exist in checkpoint
    ckpt_cfg = checkpoint.get("config", {})
    for k, v in ckpt_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    logging.info("Resuming from epoch %d, best_val_loss=%.6f", start_epoch, best_val_loss)
    return start_epoch, best_val_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    cfg: Config,
    scheduler=None,
    show_progress: bool = True,
) -> Tuple[float, float, float, float, float]:
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_correct = 0
    running_total = 0
    running_fg_correct = 0
    running_fg_total = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Train", leave=False, disable=not show_progress)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and scaler.is_enabled()

        with autocast("cuda", enabled=use_amp):
            outputs = model(images)

            if isinstance(outputs, dict) and "out" in outputs:
                outputs = outputs["out"]

            loss = getattr(cfg, "ce_loss_weight", 1.0) * ce_loss(outputs, masks)
            if getattr(cfg, "use_focal_loss", True):
                loss = loss + getattr(cfg, "focal_loss_weight", 1.0) * focal_loss(outputs, masks)
            if getattr(cfg, "use_dice_loss", True):
                loss = loss + getattr(cfg, "dice_loss_weight", 1.0) * dice_loss(outputs, masks)
            pred = torch.argmax(outputs, dim=1)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _grad_clip = float(getattr(cfg, "grad_clip", 1.0))
            if _grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _grad_clip = float(getattr(cfg, "grad_clip", 1.0))
            if _grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
            optimizer.step()
        # OneCycleLR: step after optimizer.step() (per batch). PyTorch may warn falsely with AMP.
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Detected call of `lr_scheduler\.step\(\)` before",
                    category=UserWarning,
                )
                scheduler.step()
        running_loss += loss.item()
        running_iou += compute_iou(pred, masks, cfg.num_classes)
        running_dice += compute_dice(pred, masks, cfg.num_classes)
        running_correct += int((pred == masks).sum().item())
        running_total += int(masks.numel())
        fg_mask = masks > 0
        fg_total = int(fg_mask.sum().item())
        if fg_total > 0:
            running_fg_correct += int((pred[fg_mask] == masks[fg_mask]).sum().item())
            running_fg_total += fg_total
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(num_batches, 1)
    avg_iou = running_iou / max(num_batches, 1)
    avg_dice = running_dice / max(num_batches, 1)
    avg_acc = (running_correct / running_total) if running_total > 0 else 0.0
    avg_fg_acc = (running_fg_correct / running_fg_total) if running_fg_total > 0 else 0.0
    return avg_loss, avg_iou, avg_dice, avg_acc, avg_fg_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    cfg: Config,
    class_labels: Dict[int, str],
    log_step: Optional[int] = None,
    show_progress: bool = True,
    per_class_metrics: bool = False,
) -> Tuple[float, float, float, float, float, Optional[list[dict]]]:
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_correct = 0
    running_total = 0
    running_fg_correct = 0
    running_fg_total = 0
    num_batches = 0

    per_class_tp = None
    per_class_fp = None
    per_class_fn = None
    if per_class_metrics:
        import numpy as _np

        per_class_tp = _np.zeros(num_classes, dtype=_np.int64)
        per_class_fp = _np.zeros(num_classes, dtype=_np.int64)
        per_class_fn = _np.zeros(num_classes, dtype=_np.int64)

    pbar = tqdm(dataloader, desc="Val", leave=False, disable=not show_progress)
    with torch.inference_mode():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                if isinstance(outputs, dict) and "out" in outputs:
                    outputs = outputs["out"]

                loss = getattr(cfg, "ce_loss_weight", 1.0) * ce_loss(outputs, masks)
                if getattr(cfg, "use_focal_loss", True):
                    loss = loss + getattr(cfg, "focal_loss_weight", 1.0) * focal_loss(outputs, masks)
                if getattr(cfg, "use_dice_loss", True):
                    loss = loss + getattr(cfg, "dice_loss_weight", 1.0) * dice_loss(outputs, masks)
                pred = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            running_iou += compute_iou(pred, masks, num_classes)
            running_dice += compute_dice(pred, masks, num_classes)
            running_correct += int((pred == masks).sum().item())
            running_total += int(masks.numel())
            fg_mask = masks > 0
            fg_total = int(fg_mask.sum().item())
            if fg_total > 0:
                running_fg_correct += int((pred[fg_mask] == masks[fg_mask]).sum().item())
                running_fg_total += fg_total
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

            # Optional per-class accumulators (for global val metrics)
            if per_class_metrics and per_class_tp is not None:
                pred_flat = pred.view(-1)
                target_flat = masks.view(-1)
                for cls in range(num_classes):
                    pred_inds = pred_flat == cls
                    target_inds = target_flat == cls
                    tp = (pred_inds & target_inds).sum().item()
                    fp = (pred_inds & ~target_inds).sum().item()
                    fn = (~pred_inds & target_inds).sum().item()
                    per_class_tp[cls] += tp
                    per_class_fp[cls] += fp
                    per_class_fn[cls] += fn

            # Log visualization for first validation batch only (rank 0 only when DDP)
            if show_progress and cfg.use_wandb and wandb is not None and batch_idx == 0:
                image_vis = images[0].detach().cpu()
                mean = torch.tensor(IMAGENET_MEAN, device=image_vis.device).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=image_vis.device).view(3, 1, 1)
                image_vis = image_vis * std + mean
                image_vis = image_vis.clamp(0, 1)
                gt = masks[0].detach().cpu().numpy()
                pred_mask = pred[0].detach().cpu().numpy()
                try:
                    log_dict = {
                        "predictions": wandb.Image(
                            image_vis.permute(1, 2, 0).numpy(),
                            masks={
                                "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
                                "ground_truth": {"mask_data": gt, "class_labels": class_labels},
                            },
                        )
                    }
                    wandb.log(log_dict, step=log_step)
                except Exception:
                    pass

    avg_loss = running_loss / max(num_batches, 1)
    avg_iou = running_iou / max(num_batches, 1)
    avg_dice = running_dice / max(num_batches, 1)
    avg_acc = (running_correct / running_total) if running_total > 0 else 0.0
    avg_fg_acc = (running_fg_correct / running_fg_total) if running_fg_total > 0 else 0.0

    per_class_summary: Optional[list[dict]] = None
    if per_class_metrics and per_class_tp is not None:
        per_class_summary = []
        for cls in range(num_classes):
            tp = int(per_class_tp[cls])
            fp = int(per_class_fp[cls])
            fn = int(per_class_fn[cls])
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = (
                2 * tp / (2 * tp + fp + fn)
                if (2 * tp + fp + fn) > 0
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            name = class_labels.get(cls, f"class_{cls}")
            metrics = {
                "class_id": cls,
                "class_name": name,
                "iou": float(iou),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            per_class_summary.append(metrics)
    return avg_loss, avg_iou, avg_dice, avg_acc, avg_fg_acc, per_class_summary


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    ious = []
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred_flat == cls
        target_inds = target_flat == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0.0


def compute_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    dices = []
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred_flat == cls
        target_inds = target_flat == cls

        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()

        if total == 0:
            continue

        dices.append(2 * intersection / total)

    return sum(dices) / len(dices) if dices else 0.0


def _log_sample_images_to_wandb(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_labels: Dict[int, str],
    num_samples: int = 4,
) -> None:
    """Log sample training and validation images + ground truth masks to WandB (step=0)."""
    if wandb is None:
        return

    def _denorm_images(images: torch.Tensor, masks: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        n = min(num_samples, images.size(0))
        for i in range(n):
            img = images[i].detach().cpu()
            mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=img.device).view(3, 1, 1)
            img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            gt = masks[i].detach().cpu().numpy()
            out.append((img, gt))
        return out

    log_dict = {}
    try:
        train_batch = next(iter(train_loader))
        images_t, masks_t = train_batch[0], train_batch[1]
        for i, (img, gt) in enumerate(_denorm_images(images_t, masks_t)):
            log_dict[f"dataset/train_image_gt_{i}"] = wandb.Image(
                img,
                masks={"ground_truth": {"mask_data": gt, "class_labels": class_labels}},
            )
    except (StopIteration, Exception):
        pass
    try:
        val_batch = next(iter(val_loader))
        images_v, masks_v = val_batch[0], val_batch[1]
        for i, (img, gt) in enumerate(_denorm_images(images_v, masks_v)):
            log_dict[f"dataset/val_image_gt_{i}"] = wandb.Image(
                img,
                masks={"ground_truth": {"mask_data": gt, "class_labels": class_labels}},
            )
    except (StopIteration, Exception):
        pass
    if log_dict:
        try:
            wandb.log(log_dict, step=0)
        except Exception:
            pass


def run_training(cfg: Config) -> None:
    """Run full training loop. cfg must have paths resolved (or absolute). Resume via cfg.resume (Path or None)."""
    # DistributedDataParallel: init when launched via torchrun (RANK, LOCAL_RANK, WORLD_SIZE set)
    rank = 0
    world_size = 1
    local_rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    is_main = rank == 0

    _resolve_paths(cfg)

    # Override num_classes from dataset label_mapping.json if present (no hardcoded classes)
    label_mapping = load_label_mapping(cfg.dataset_root)
    if label_mapping:
        cfg.num_classes = label_mapping["num_classes"]
        if is_main:
            logging.info("Using num_classes=%d from %s", cfg.num_classes, cfg.dataset_root / "label_mapping.json")
    class_labels = get_class_labels_for_wandb(label_mapping, cfg.num_classes)

    # WandB init early (rank 0 only) so sweep config (if any) is available before building dataloaders/model
    if cfg.use_wandb and is_main:
        if wandb is None:
            logging.warning("wandb is not installed; disabling wandb logging.")
            cfg.use_wandb = False
        else:
            is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
            logging.info(
                "Model config before wandb (stage-loaded): model_profile=%s architecture=%s encoder_name=%s encoder_weights=%s",
                getattr(cfg, "model_profile", None),
                getattr(cfg, "architecture", None),
                getattr(cfg, "encoder_name", None),
                getattr(cfg, "encoder_weights", None),
            )
            if is_sweep:
                # Init first so wandb.config holds sweep params; then sync cfg, apply profile, then log
                # final hyperparameters (so encoder_name matches model_profile, not stale defaults).
                wandb.init(project=cfg.wandb_project, resume="allow")
                _apply_wandb_config_to_cfg(cfg)
                profile = getattr(cfg, "model_profile", None)
                if isinstance(profile, str) and profile.strip():
                    try:
                        apply_model_profile(cfg, profile)
                    except Exception as e:
                        logging.error("Failed to apply model_profile '%s': %s", profile, e)
                        sys.exit(1)
                # Run name in the UI matches the profile (YAML wandb_run_name is ignored for sweeps).
                if isinstance(profile, str) and profile.strip():
                    try:
                        wandb.run.name = f"{profile}-{wandb.run.id[:8]}"
                    except Exception as e:
                        logging.warning("Could not set wandb run name: %s", e)
                wandb.config.update(_wandb_training_config_dict(cfg), allow_val_change=True)
            else:
                wandb.init(
                    project=cfg.wandb_project,
                    name=getattr(cfg, "wandb_run_name", None)
                    or getattr(cfg, "model_profile", "finetune"),
                    config=_wandb_training_config_dict(cfg),
                    resume="allow",
                )
            logging.info(
                "Resolved model config after wandb sync: model_profile=%s architecture=%s encoder_name=%s encoder_weights=%s",
                getattr(cfg, "model_profile", None),
                getattr(cfg, "architecture", None),
                getattr(cfg, "encoder_name", None),
                getattr(cfg, "encoder_weights", None),
            )
            # Project tables default to "last" logged step; max/min makes sort/filter match best val performance.
            try:
                wandb.define_metric("val_dice", summary="max")
                wandb.define_metric("val_iou", summary="max")
                wandb.define_metric("val_fg_accuracy", summary="max")
                wandb.define_metric("val_loss", summary="min")
                _esm = getattr(cfg, "early_stop_metric", "val_loss")
                _esm_max = _esm in {"val_iou", "val_dice"}
                wandb.define_metric(
                    "best_early_stop_metric",
                    summary="max" if _esm_max else "min",
                )
            except Exception as e:
                logging.warning("wandb.define_metric failed: %s", e)

    # Reproducibility (per-rank seed so workers differ across ranks)
    seed = 42
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.cuda.set_device(device)
    if is_main:
        logging.info("Using device: %s (rank %d / %d)", device, rank, world_size)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_loader, val_loader, train_sampler = create_dataloaders(cfg, rank=rank, world_size=world_size)

    # Compute class-wise pixel statistics and dynamic class weights from training masks
    class_counts, class_weights_np = _compute_class_stats(
        train_loader.dataset, cfg.num_classes
    )
    if not getattr(cfg, "use_class_weights", True):
        class_weights_np = np.ones_like(class_weights_np, dtype=np.float32)
    total_pixels = int(class_counts.sum())
    bg_pixels = int(class_counts[0]) if cfg.num_classes > 0 else 0
    fg_pixels = int(class_counts[1:].sum()) if cfg.num_classes > 1 else 0
    fg_bg_ratio = (fg_pixels / max(bg_pixels, 1)) if bg_pixels > 0 else float("inf")
    if is_main:
        logging.info(
            "Training class pixel counts: %s",
            {i: int(c) for i, c in enumerate(class_counts)},
        )
        logging.info(
            "Training class weights (mean=1): %s",
            {i: float(w) for i, w in enumerate(class_weights_np)},
        )
        logging.info(
            "Total pixels=%d, fg_pixels=%d, bg_pixels=%d, fg:bg ratio=%.6f",
            total_pixels,
            fg_pixels,
            bg_pixels,
            fg_bg_ratio,
        )
        logging.info(
            "Train DataLoader: %d samples, %d batches, batch_size=%d, num_workers=%d",
            len(train_loader.dataset),
            len(train_loader),
            cfg.batch_size,
            cfg.num_workers,
        )
        logging.info(
            "Val DataLoader:   %d samples, %d batches, batch_size=%d, num_workers=%d",
            len(val_loader.dataset),
            len(val_loader),
            cfg.batch_size,
            cfg.num_workers,
        )
        if cfg.use_wandb and wandb is not None:
            _log_sample_images_to_wandb(train_loader, val_loader, class_labels, num_samples=4)

    try:
        model = build_model(cfg)
    except ValueError as e:
        logging.error("Model build failed: %s", e)
        sys.exit(1)
    model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            logging.info("Using DistributedDataParallel on %d GPUs", world_size)
    elif n_gpus > 1:
        model = nn.DataParallel(model)
        if is_main:
            logging.info("Using DataParallel on %d GPUs", n_gpus)

    global ce_loss, focal_loss, dice_loss, class_weights
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    use_weights = getattr(cfg, "use_class_weights", True)
    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights if use_weights else None
    )
    focal_loss = smp.losses.FocalLoss(
        mode="multiclass",
        gamma=getattr(cfg, "focal_gamma", 2.0),
    )
    dice_loss = smp.losses.DiceLoss(
        mode="multiclass",
    )
    # Get the underlying model (unwrap DDP or DataParallel)
    _model_core = _get_model_core(model)
    encoder_lr = cfg.learning_rate * getattr(cfg, "encoder_lr_scale", 0.1)
    optimizer_name = getattr(cfg, "optimizer_name", "adamw").lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        [
            {"params": _model_core.encoder.parameters(), "lr": encoder_lr},
            {"params": _model_core.decoder.parameters(), "lr": cfg.learning_rate},
            {"params": _model_core.segmentation_head.parameters(), "lr": cfg.learning_rate},
        ],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    if is_main:
        logging.info(
            "Optimizer: %s, lr=%.6f (encoder=%.6f), weight_decay=%.6f",
            optimizer_name, cfg.learning_rate, encoder_lr, cfg.weight_decay,
        )

    scheduler_name = getattr(cfg, "scheduler_name", "onecycle").lower()
    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    if scheduler_name == "onecycle":
        warmup_pct = float(getattr(cfg, "warmup_pct", 0.1))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[encoder_lr, cfg.learning_rate, cfg.learning_rate],
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        if is_main:
            logging.info("Scheduler: OneCycleLR, warmup_pct=%.2f, total_steps=%d", warmup_pct, total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
        )
        if is_main:
            logging.info("Scheduler: CosineAnnealingLR, T_max=%d", cfg.epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    resume_path = getattr(cfg, "resume", None)
    if resume_path is not None:
        resume_path = Path(resume_path)
        if resume_path.exists():
            resume_path = resume_path if resume_path.is_absolute() else _PROJECT_ROOT / resume_path
            start_epoch, best_val_loss = load_checkpoint(resume_path, model, optimizer, scheduler, cfg)

    checkpoints_dir = cfg.checkpoints_dir
    # Per-run checkpoint dir when using WandB (e.g. sweeps) so runs don't overwrite each other.
    # Use wandb.run.name so the folder matches what you see in the UI.
    if cfg.use_wandb and wandb is not None and wandb.run is not None:
        def _sanitize_dir_component(s: str) -> str:
            return s.replace("/", "_").replace("\\", "_").replace(" ", "-")

        run_name = str(getattr(wandb.run, "name", "") or "").strip()
        run_id = str(getattr(wandb.run, "id", "") or "").strip()
        # Ensure checkpoint folder always carries WandB run id for unambiguous traceability.
        # If name already contains id suffix (common in sweeps), keep it unchanged.
        if run_name and run_id and run_id not in run_name:
            run_folder = f"{run_name}-{run_id}"
        else:
            run_folder = run_name or run_id or "run"
        checkpoints_dir = checkpoints_dir / _sanitize_dir_component(run_folder)
        if is_main:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Saving checkpoints to run dir: %s", checkpoints_dir)
    # Save resolved finetune stage config next to checkpoints for traceability.
    # If WandB is enabled, this is inside the per-run checkpoint dir.
    try:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        stage_cfg_yaml = getattr(cfg, "_stage_cfg_yaml", None)
        if isinstance(stage_cfg_yaml, str) and stage_cfg_yaml.strip():
            config_path = checkpoints_dir / "finetune_config.yaml"
            with open(config_path, "w") as f:
                f.write(stage_cfg_yaml)
            logging.info("Saved finetune config to %s", config_path)
    except Exception as e:
        logging.warning("Failed to save finetune config next to checkpoints: %s", e)

    best_ckpt_path = checkpoints_dir / "best_model.pth"
    last_ckpt_path = checkpoints_dir / "last_model.pth"

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    patience_counter = 0
    use_early_stopping = getattr(cfg, "use_early_stopping", False)
    early_stop_patience = getattr(cfg, "early_stop_patience", 10)
    early_stop_min_delta = getattr(cfg, "early_stop_min_delta", 0.0)
    early_stop_metric = getattr(cfg, "early_stop_metric", "val_loss")
    maximize_metrics = {"val_iou", "val_dice"}
    maximize = early_stop_metric in maximize_metrics
    best_metric = float("-inf") if maximize else float("inf")
    unknown_metric_warned = False
    if is_main:
        logging.info(
            "Early stopping: metric=%s, mode=%s, patience=%d, min_delta=%.6f",
            early_stop_metric,
            "max" if maximize else "min",
            early_stop_patience,
            early_stop_min_delta,
        )

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # DDP: sync before training so both ranks enter train_one_epoch together (avoids allreduce deadlock)
        if world_size > 1:
            dist.barrier()
        if is_main:
            logging.info("Epoch %d / %d", epoch + 1, cfg.epochs)

        train_loss, train_iou, train_dice, train_acc, train_fg_acc = train_one_epoch(
            model, train_loader, optimizer, device, scaler, cfg,
            scheduler=scheduler, show_progress=is_main,
        )
        # Validation: all ranks run validation so no rank blocks at broadcast (keeps DDP in sync)
        val_loss, val_iou, val_dice, val_acc, val_fg_acc, per_class = validate_one_epoch(
            model,
            val_loader,
            device,
            cfg.num_classes,
            cfg,
            class_labels,
            log_step=epoch + 1,
            show_progress=is_main,
            per_class_metrics=getattr(cfg, "per_class_metrics", False),
        )

        # DDP: sync after validation so all ranks are done before rank 0 does I/O
        if world_size > 1:
            dist.barrier()

        if is_main:
            logging.info(
                "Epoch %d: train_loss=%.6f, train_iou=%.4f, train_dice=%.4f, train_accuracy=%.4f, train_fg_accuracy=%.4f, val_loss=%.6f, val_accuracy=%.4f, val_fg_accuracy=%.4f, val_iou=%.4f, val_dice=%.4f",
                epoch + 1,
                train_loss,
                train_iou,
                train_dice,
                train_acc,
                train_fg_acc,
                val_loss,
                val_acc,
                val_fg_acc,
                val_iou,
                val_dice,
            )
            # Optional per-class metrics logging
            if per_class:
                for m in per_class:
                    logging.info(
                        "  [class %s] IoU=%.4f Dice=%.4f Prec=%.4f Rec=%.4f F1=%.4f",
                        m["class_name"],
                        m["iou"],
                        m["dice"],
                        m["precision"],
                        m["recall"],
                        m["f1"],
                    )

        # Select current metric for early stopping
        if early_stop_metric == "val_loss":
            current_metric = val_loss
        elif early_stop_metric == "val_iou":
            current_metric = val_iou
        elif early_stop_metric == "val_dice":
            current_metric = val_dice
        elif early_stop_metric == "train_loss":
            current_metric = train_loss
        else:
            if not unknown_metric_warned:
                logging.warning(
                    "Unknown early_stop_metric '%s'; falling back to val_loss (minimize).",
                    early_stop_metric,
                )
                unknown_metric_warned = True
            current_metric = val_loss

        if maximize:
            improved = current_metric > (best_metric + early_stop_min_delta)
        else:
            improved = current_metric < (best_metric - early_stop_min_delta)
        if improved:
            best_metric = current_metric
            best_val_loss = val_loss
            if is_main:
                save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, cfg, label_mapping)
            patience_counter = 0
        else:
            patience_counter += 1

        if is_main:
            save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, cfg, label_mapping)

        if cfg.use_wandb and is_main:
            assert wandb is not None
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_iou": train_iou,
                    "train_dice": train_dice,
                    "train_accuracy": train_acc,
                    "train_fg_accuracy": train_fg_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_fg_accuracy": val_fg_acc,
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "best_val_loss": best_val_loss,
                    "best_early_stop_metric": best_metric,
                    "early_stop_metric": early_stop_metric,
                },
                step=epoch + 1,
            )

        # Only step epoch-level schedulers here; OneCycleLR steps per batch in train_one_epoch
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if use_early_stopping and patience_counter >= early_stop_patience:
            if is_main:
                logging.info(
                    "Early stopping at epoch %d (no improvement for %d epochs). Best %s=%.6f (best_val_loss=%.6f)",
                    epoch + 1,
                    early_stop_patience,
                    early_stop_metric,
                    best_metric,
                    best_val_loss,
                )
                if cfg.use_wandb and wandb is not None:
                    wandb.log({"early_stopped": True, "stopped_epoch": epoch + 1}, step=epoch + 1)
            break

        # DDP: sync all ranks before next epoch so rank 0 can finish saving/logging before others start training
        if world_size > 1:
            dist.barrier()

    if is_main:
        logging.info(
            "Training complete. Best %s=%.6f (best_val_loss=%.6f)",
            early_stop_metric,
            best_metric,
            best_val_loss,
        )
        if cfg.use_wandb and wandb is not None:
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p

def run(cfg) -> None:
    from omegaconf import OmegaConf
    root = _resolve_path(cfg.paths.root, Path.cwd())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from config import get_config_from_stage

    # Build training config from stage and resolve dataset/checkpoints paths
    train_cfg = get_config_from_stage(cfg.stage)
    train_cfg.dataset_root = _resolve_path(train_cfg.dataset_root, root)
    base_ckpt_dir = _resolve_path(train_cfg.checkpoints_dir, root)

    # When WandB is disabled, use a per-run subfolder so different runs don't overwrite each other.
    # Include model_profile so the folder is self-describing.
    if not getattr(train_cfg, "use_wandb", True):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_suffix = f"{ts}_{random.randint(0, 9999):04d}"
        profile = getattr(train_cfg, "model_profile", None) or "run"
        profile = str(profile).strip().replace("/", "_").replace("\\", "_").replace(" ", "-")
        ckpt_dir = base_ckpt_dir / f"{profile}_{run_suffix}"
    else:
        ckpt_dir = base_ckpt_dir
    train_cfg.checkpoints_dir = ckpt_dir

    # Keep resolved stage YAML on cfg; run_training will write it after final
    # checkpoint dir is known (e.g. wandb run-id subfolder).
    try:
        stage_cfg = OmegaConf.to_container(cfg.stage, resolve=True)
        setattr(train_cfg, "_stage_cfg_yaml", OmegaConf.to_yaml(stage_cfg))
    except Exception as e:
        logging.warning("Failed to prepare finetune config yaml for saving: %s", e)

    # Resolve resume path (if provided) relative to project root
    resume = getattr(train_cfg, "resume", None)
    if resume is not None:
        setattr(train_cfg, "resume", _resolve_path(resume, root))
    else:
        setattr(train_cfg, "resume", None)
    run_training(train_cfg)
