#!/usr/bin/env python3

"""
Shared model builder for SMP architectures.

Used by train, val, and infer tools so model selection and config stay in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

import segmentation_models_pytorch as smp

if TYPE_CHECKING:
    from config import Config

# DeepLabV3+ uses dilated encoder; SMP encoders that use pooling for downsampling
# (e.g. Xception) do not support it.
DEEPLABV3PLUS_UNSUPPORTED_ENCODERS = {"xception"}

# ConvNeXt and other timm-only encoders require the "tu-" (timm universal) prefix in SMP.
CONVNEXT_ALIASES = {"convnext_tiny", "convnext_small", "convnext_base", "convnext_large"}

# Supported architectures for YAML-configurable selection.
_ARCH_ALIASES = {
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplabv3+": "DeepLabV3Plus",
    "deeplabv3": "DeepLabV3",
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unet++": "UnetPlusPlus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "manet": "MAnet",
    "linknet": "Linknet",
}


def _normalize_encoder_name(name: str) -> str:
    """Use tu- prefix for ConvNeXt so SMP uses TimmUniversalEncoder."""
    n = (name or "").strip().lower()
    if n in CONVNEXT_ALIASES and not n.startswith("tu-"):
        return f"tu-{n}"
    return name


def _normalize_architecture_name(name: str | None) -> str:
    raw = (name or "DeepLabV3Plus").strip()
    key = raw.lower().replace("_", "").replace("-", "")
    return _ARCH_ALIASES.get(key, raw)


def build_model(
    cfg: "Config",
    *,
    use_pretrained_encoder: bool | None = None,
) -> nn.Module:
    """
    Build an SMP segmentation model from config.

    Args:
        cfg: Config with architecture, encoder_name, encoder_weights, num_classes, activation.
        use_pretrained_encoder: If False, encoder_weights=None (for val/infer).
            If True or None, use cfg.encoder_weights (for training with pretrained).

    Returns:
        SMP model instance (not moved to device).
    """
    architecture = _normalize_architecture_name(getattr(cfg, "architecture", "DeepLabV3Plus"))
    encoder_name = getattr(cfg, "encoder_name", "resnet50") or "resnet50"
    encoder_name_normalized = _normalize_encoder_name(encoder_name)
    encoder_name_lower = encoder_name_normalized.lower()

    if use_pretrained_encoder is False:
        encoder_weights = None
    else:
        encoder_weights = getattr(cfg, "encoder_weights", "imagenet")

    common_kwargs = {
        "encoder_name": encoder_name_normalized,
        "encoder_weights": encoder_weights,
        "classes": cfg.num_classes,
        "activation": cfg.activation,
        "in_channels": getattr(cfg, "in_channels", 3),
    }

    if architecture == "DeepLabV3Plus":
        if encoder_name_lower in DEEPLABV3PLUS_UNSUPPORTED_ENCODERS:
            raise ValueError(
                f"Encoder '{encoder_name}' does not support dilated (atrous) mode required by DeepLabV3+. "
                "Use an encoder that supports encoder_output_stride, e.g. resnet50, resnet101, efficientnet-b0."
            )
        model = smp.DeepLabV3Plus(
            **common_kwargs,
            encoder_output_stride=getattr(cfg, "encoder_output_stride", 16),
            decoder_channels=getattr(cfg, "decoder_channels", 256),
            decoder_atrous_rates=tuple(getattr(cfg, "decoder_atrous_rates", (12, 24, 36))),
        )
        return model

    if architecture == "DeepLabV3":
        model = smp.DeepLabV3(
            **common_kwargs,
            encoder_output_stride=getattr(cfg, "encoder_output_stride", 16),
            decoder_channels=getattr(cfg, "decoder_channels", 256),
            decoder_atrous_rates=tuple(getattr(cfg, "decoder_atrous_rates", (12, 24, 36))),
        )
        return model

    if architecture == "Unet":
        return smp.Unet(**common_kwargs)
    if architecture == "UnetPlusPlus":
        # Some SMP builds break Unet++ + ConvNeXt (zero-channel decoder). Opt in via cfg.allow_unetplusplus_convnext.
        if encoder_name_lower.startswith("tu-convnext") and not getattr(
            cfg, "allow_unetplusplus_convnext", False
        ):
            raise ValueError(
                f"Unsupported combination: architecture='{architecture}' with encoder='{encoder_name}'. "
                "UnetPlusPlus + ConvNeXt can fail on some segmentation_models_pytorch versions. "
                "Use architecture=Unet/FPN/PSPNet/DeepLabV3Plus with ConvNeXt, UnetPlusPlus with resnet50/101, "
                "or set allow_unetplusplus_convnext=true in configs/models/<profile>.yaml to try anyway."
            )
        return smp.UnetPlusPlus(**common_kwargs)
    if architecture == "FPN":
        return smp.FPN(**common_kwargs)
    if architecture == "PSPNet":
        return smp.PSPNet(**common_kwargs)
    if architecture == "PAN":
        return smp.PAN(**common_kwargs)
    if architecture == "MAnet":
        return smp.MAnet(**common_kwargs)
    if architecture == "Linknet":
        return smp.Linknet(**common_kwargs)

    raise ValueError(
        f"Unsupported architecture '{architecture}'. "
        "Use one of: DeepLabV3Plus, DeepLabV3, Unet, UnetPlusPlus, FPN, PSPNet, PAN, MAnet, Linknet."
    )
