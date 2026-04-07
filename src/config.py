#!/usr/bin/env python3

"""
Config dataclass and loaders. Stage configs (configs/stage/*.yaml) hold defaults
per stage; get_config_from_stage(stage_cfg) builds Config from Hydra stage config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

# ImageNet normalization (single source of truth for dataset, infer, and viz)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Optional: if default.yaml exists, load_config() uses it; else Config uses dataclass defaults. Prefer stage configs (configs/stage/*.yaml).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
MODELS_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"


@dataclass
class Config:
    dataset_root: Path = Path("dataset/defect_data")
    checkpoints_dir: Path = Path("checkpoints")
    checkpoint: Path | None = None  # for evaluate/infer/compare (e.g. best_model.pth)
    label_mapping_path: Path | None = None  # optional explicit path to label_mapping.json (fallback when checkpoint lacks labels)
    image_size: int = 512
    num_classes: int = 3
    batch_size: int = 8
    num_workers: int = 4
    # Data sampling / background control
    limit_background_samples: bool = False
    max_background_to_foreground_ratio: float = 1.0
    # Fraction of pure background-only images to keep in train split (0.3 -> keep 30% of bg images).
    # Negative value means "disabled" so ratio-based limiting is used instead.
    background_keep_fraction: float = -1.0
    # Loss / imbalance handling
    use_class_weights: bool = True
    use_focal_loss: bool = True
    use_dice_loss: bool = True
    ce_loss_weight: float = 1.0
    focal_loss_weight: float = 1.0
    dice_loss_weight: float = 1.0
    focal_gamma: float = 2.0
    learning_rate: float = 1e-4
    epochs: int = 50
    weight_decay: float = 1e-4
    # Optimizer and scheduler
    optimizer_name: str = "adamw"          # "adam" or "adamw"
    encoder_lr_scale: float = 0.1          # encoder LR = learning_rate * encoder_lr_scale
    scheduler_name: str = "onecycle"       # "onecycle" or "cosine"
    warmup_pct: float = 0.1                # fraction of steps used for warmup (onecycle only)
    grad_clip: float = 1.0                 # max gradient norm (0 = disabled)
    # Model (SMP)
    architecture: str = "DeepLabV3Plus"
    encoder_name: str = "resnet50"
    encoder_weights: str | None = "imagenet"
    activation: str | None = None
    in_channels: int = 3
    encoder_output_stride: int = 16
    decoder_channels: int = 256
    decoder_atrous_rates: Tuple[int, ...] = (12, 24, 36)
    # Opt-in: some SMP versions allow UnetPlusPlus + tu-convnext_*; default False blocks known bad combos.
    allow_unetplusplus_convnext: bool = False
    # Optional model profile name from configs/models/<profile>.yaml
    model_profile: str | None = None
    use_wandb: bool = True
    wandb_project: str = "defect-segmentation"
    wandb_run_name: str | None = None
    # Early stopping
    use_early_stopping: bool = True
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    early_stop_metric: str = "val_loss"  # one of: "val_loss", "val_iou", "val_dice", "train_loss"
    # On-the-fly train-time augmentation
    augmentations_enabled: bool = True
    aug_hflip_prob: float = 0.5
    aug_vflip_prob: float = 0.0
    aug_rotate90_prob: float = 0.0
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_noise_std: float = 0.02
    aug_brightness_contrast_prob: float = 0.5
    aug_shift_scale_rotate_prob: float = 0.5
    aug_shift_limit: float = 0.05
    aug_scale_limit: float = 0.1
    aug_rotate_limit: float = 15.0
    aug_clahe_prob: float = 0.3
    aug_clahe_clip: float = 2.0
    aug_gauss_noise_prob: float = 0.3
    aug_gaussian_blur_prob: float = 0.2
    aug_gaussian_blur_kernel: int = 3
    aug_random_crop_defect_prob: float = 0.0
    aug_random_crop_defect_size: int = 512
    # Optional CLAHE preprocessing (prepare_dataset + infer)
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    
    crop_object_enabled: bool = False
    crop_object_padding: int = 0
    # Dataset preparation: split by ratio (prepare_dataset.py)
    split_by_ratio: bool = False
    splits: Tuple[str, ...] = ("train", "val", "test")
    split_ratios: Tuple[float, ...] = (0.8, 0.1, 0.1)
    split_seed: int = 42
    # Dataset tiling for prepare_dataset.py (optional)
    prepare_tile_enabled: bool = False
    prepare_tile_size: int = 1024
    prepare_tile_overlap: int = 256
    # Inference: tiling for full-res output (match training: 1024 tiles, 256 overlap)
    tiled_inference: bool = False
    tile_size: int = 1024
    tile_overlap: int = 256
    save_tiled_outputs: bool = False
    tiled_output_dir: Path | None = None
    save_masks: bool = True
    save_visualizations: bool = True
    show_overlay_labels: bool = False
    infer_visualizations_dir: Path | None = None
    # Pixels with max class probability below this are set to background (0). 0 = disabled.
    infer_confidence: float = 0.0


def _apply_yaml_to_config(cfg: Config, data: dict[str, Any]) -> None:
    path_keys = (
        "dataset_root",
        "checkpoints_dir",
        "checkpoint",
        "tiled_output_dir",
        "infer_visualizations_dir",
        "label_mapping_path",
    )
    # YAML list -> tuple for Config
    list_keys = ("split_ratios", "decoder_atrous_rates", "splits", "clahe_tile_grid")
    for key, value in data.items():
        if not hasattr(cfg, key):
            continue
        if key in path_keys and value is not None:
            value = Path(value) if isinstance(value, str) else value
        if key in list_keys and value is not None and isinstance(value, list):
            value = tuple(value)
        if value is not None:
            setattr(cfg, key, value)


def _apply_model_profile_to_config(cfg: Config, profile_name: str) -> None:
    """
    Apply model profile YAML from configs/models/<profile_name>.yaml.
    Stage-level keys still override profile keys afterward.
    """
    profile = (profile_name or "").strip()
    if not profile:
        return
    profile_path = MODELS_CONFIG_DIR / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Model profile '{profile}' not found at {profile_path}. "
            f"Create configs/models/{profile}.yaml or remove model_profile."
        )
    import yaml

    raw = yaml.safe_load(profile_path.read_text())
    if isinstance(raw, dict):
        _apply_yaml_to_config(cfg, raw)


def apply_model_profile(cfg: Config, profile_name: str) -> None:
    """Public helper to apply a model profile onto an existing Config."""
    _apply_model_profile_to_config(cfg, profile_name)


def load_config(path: Path | None = None) -> Config:
    """Load Config from YAML file. Uses defaults for missing or invalid file."""
    cfg = Config()
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return cfg
    try:
        import yaml
        raw = yaml.safe_load(config_path.read_text())
        if isinstance(raw, dict):
            _apply_yaml_to_config(cfg, raw)
    except Exception as e:
        import logging as _logging
        _logging.warning("Failed to load config from %s: %s — using dataclass defaults.", config_path, e)
    return cfg


def get_default_config() -> Config:
    """Return Config from configs/default.yaml if present, else dataclass defaults. Prefer get_config_from_stage(cfg.stage) when running via run.py."""
    return load_config()


def get_config_from_stage(stage_cfg) -> Config:
    """Build Config from Hydra stage DictConfig (e.g. cfg.stage). Resolves interpolations and applies to Config. Missing keys use Config dataclass defaults."""
    cfg = Config()
    try:
        from omegaconf import OmegaConf
        data = OmegaConf.to_container(stage_cfg, resolve=True)
        if isinstance(data, dict):
            profile = data.get("model_profile")
            if isinstance(profile, str) and profile.strip():
                _apply_model_profile_to_config(cfg, profile)
            _apply_yaml_to_config(cfg, data)
    except Exception as e:
        import logging as _logging
        _logging.warning("get_config_from_stage failed: %s — using Config defaults.", e)
    return cfg
