"""SAM2 with LoRA adapters for fine-tuning."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import os

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not installed. LoRA will not be available.")

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from sam2.sam2_image_predictor import SAM2ImagePredictor
from huggingface_hub import hf_hub_download
import sam2

log = logging.getLogger(__name__)


def _load_checkpoint(model, ckpt_path):
    """Load checkpoint into model."""
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(sd.get("model", sd), strict=False)
        if missing_keys:
            log.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys: {unexpected_keys}")


def build_sam2_with_isolated_hydra(config_file, ckpt_path, device="cuda"):
    """Build SAM2 model using isolated Hydra initialization."""
    # Get sam2 package config directory
    sam2_config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs/sam2")
    
    # Clear any existing Hydra instance
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()
    
    try:
        # Initialize Hydra with SAM2's config directory
        initialize_config_dir(config_dir=sam2_config_dir, version_base=None)
        
        # Compose the config
        hydra_overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        
        # Instantiate model
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path)
        model = model.to(device)
        model.eval()
        
        return model
    finally:
        # Clear Hydra after we are done
        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()


class SAM2LoRA(nn.Module):
    """SAM2 model with LoRA adapters for efficient fine-tuning."""
    
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        lora_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        
        # Map model_id to SAM2 config file and checkpoint filename
        model_configs = {
            "facebook/sam2-hiera-large":      ("sam2_hiera_l.yaml",  "sam2_hiera_large.pt"),
            "facebook/sam2-hiera-base-plus":  ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
            "facebook/sam2-hiera-small":      ("sam2_hiera_s.yaml",  "sam2_hiera_small.pt"),
            "facebook/sam2-hiera-tiny":       ("sam2_hiera_t.yaml",  "sam2_hiera_tiny.pt"),
        }
        config_file, ckpt_filename = model_configs.get(model_id, ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"))

        # Resolve checkpoint path using this priority order:
        #   1. local_path kwarg (absolute path, bypasses HF Hub entirely)
        #   2. Local HF cache (try without network access first)
        #   3. Download from HuggingFace Hub (requires internet)
        local_path = kwargs.get('local_path', None)
        if local_path and Path(local_path).exists():
            ckpt_path = local_path
            log.info(f"Using local checkpoint: {ckpt_path}")
        else:
            cache_dir = os.environ.get('HF_HUB_CACHE', os.environ.get('HF_HOME', None))
            if cache_dir and not cache_dir.endswith('/hub'):
                cache_dir = os.path.join(cache_dir, 'hub')

            # Try local cache first (no network), then fall back to download
            try:
                ckpt_path = hf_hub_download(
                    model_id, ckpt_filename,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                log.info(f"Using cached checkpoint: {ckpt_path}")
            except Exception:
                log.info(f"Checkpoint not in local cache — downloading from HuggingFace: {model_id}")
                ckpt_path = hf_hub_download(model_id, ckpt_filename, cache_dir=cache_dir)
                log.info(f"Checkpoint downloaded/cached at: {ckpt_path}")
        
        # Build model with isolated Hydra
        log.info(f"Building SAM2 model with config: {config_file}")
        self.sam2 = build_sam2_with_isolated_hydra(config_file, ckpt_path, device=device)
        
        # Apply LoRA if config provided
        if lora_config and PEFT_AVAILABLE:
            self._apply_lora(lora_config)
        elif lora_config and not PEFT_AVAILABLE:
            log.warning("LoRA config provided but PEFT not installed. Skipping.")
        
        self.predictor = SAM2ImagePredictor(self.sam2)
        self.sam2.image_encoder.trunk.gradient_checkpointing = True

    def _apply_lora(self, lora_config: Dict[str, Any]):
        """Apply LoRA adapters to SAM2 image encoder and mask decoder."""
        rank = lora_config.get("rank", 16)
        train_image_encoder = lora_config.get("train_image_encoder", True)
        train_mask_decoder = lora_config.get("train_mask_decoder", True)
        
        log.info(f"Applying LoRA with rank={rank}")
        log.info(f"  Image encoder LoRA: {'enabled' if train_image_encoder else 'disabled'}")
        log.info(f"  Mask decoder LoRA: {'enabled' if train_mask_decoder else 'disabled'}")

        # Image encoder (Hiera): uses target_modules from config (e.g. qkv, proj)
        if train_image_encoder:
            config_encoder = LoraConfig(
                r=rank,
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=list(lora_config.get("target_modules", ["qkv", "proj"])),
                bias="none",
            )
            self.sam2.image_encoder = get_peft_model(self.sam2.image_encoder, config_encoder)
            log.info(f"Image encoder LoRA applied. Trainable parameters: {self.sam2.image_encoder.print_trainable_parameters()}")
        else:
            # Freeze entire image encoder
            for param in self.sam2.image_encoder.parameters():
                param.requires_grad = False
            log.info("Image encoder frozen (no LoRA)")

        # Mask decoder (TwoWayTransformer): uses q_proj, k_proj, v_proj, out_proj
        if train_mask_decoder:
            target_decoder = lora_config.get(
                "target_modules_decoder",
                ["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            config_decoder = LoraConfig(
                r=rank,
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=list(target_decoder),
                bias="none",
            )
            self.sam2.sam_mask_decoder = get_peft_model(self.sam2.sam_mask_decoder, config_decoder)
            log.info("Mask decoder LoRA applied")
        else:
            # Freeze entire mask decoder
            for param in self.sam2.sam_mask_decoder.parameters():
                param.requires_grad = False
            log.info("Mask decoder frozen (no LoRA)")
    
    def forward(self, images, prompts=None):
        """Forward pass through SAM2."""
        return self.sam2(images, prompts)
    
    def get_image_embedding(self, image):
        """Get image embedding from SAM2 encoder."""
        self.predictor.set_image(image)
        return self.predictor._features
    
    def predict(self, image, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        """Run prediction on an image."""
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )
        return masks, scores, logits
    
    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        # Only train the LoRA parameters, keep base model frozen
        if hasattr(self.sam2.image_encoder, "base_model"):
            for name, param in self.sam2.image_encoder.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False
        return self

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Return param groups for optimizer: 'lora' (LoRA params) and 'decoder' (mask decoder / other trainable)."""
        lora_params = []
        decoder_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            else:
                # Mask decoder, prompt encoder, or other trainable (when PEFT not used)
                decoder_params.append(param)
        return {"lora": lora_params, "decoder": decoder_params}


def load_sam2_with_lora(cfg, device: str = "cuda") -> SAM2LoRA:
    """Load SAM2 with LoRA from config."""
    # Build LoRA config if enabled
    lora_config = None
    if cfg.model.lora.enabled:
        lora_config = {
            "rank": cfg.model.lora.rank,
            "alpha": cfg.model.lora.alpha,
            "dropout": cfg.model.lora.dropout,
            "target_modules": list(cfg.model.lora.target_modules),
            "train_image_encoder": cfg.model.lora.get("train_image_encoder", True),
            "train_mask_decoder": cfg.model.lora.get("train_mask_decoder", True),
        }
    
    # Use HuggingFace model_id
    model_id = cfg.model.checkpoint.repo

    # Optional: absolute path to a local .pt file — bypasses HF Hub entirely.
    # Set in model config: checkpoint.local_path: /path/to/sam2_hiera_large.pt
    local_path = cfg.model.checkpoint.get("local_path", None)

    return SAM2LoRA(
        model_id=model_id,
        lora_config=lora_config,
        device=device,
        local_path=local_path,
    )
