#!/usr/bin/env python3
"""Entry point for WandB sweeps: runs the finetune stage so wandb agent invokes training with sweep params."""
import sys
from pathlib import Path

# Compose config with stage=finetune and run the stage
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

if __name__ == "__main__":
    # Match run.py behavior: register ${project_root:} resolver before compose/resolve.
    OmegaConf.register_new_resolver("project_root", lambda: str(_ROOT), replace=True)
    with initialize_config_dir(config_dir=str(_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["stage=finetune"])
    OmegaConf.resolve(cfg)
    from src.stages.finetune import run
    run(cfg)
