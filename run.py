#!/usr/bin/env python3
"""
DeepLabV3+ Pipeline - Single entry point (jnj-sam2-pipeline style).

    Stages: pseudomask → refinement → prepare → finetune → infer → evaluate → compare → compare_bboxes

Usage:
  python run.py stage=pseudomask
  python run.py stage=refinement
  python run.py stage=prepare
  python run.py stage=finetune
  python run.py stage=infer
  python run.py stage=evaluate
  python run.py stage=compare

  # Override stage config
  python run.py stage=pseudomask stage.input_dir=/path/to/images stage.output_dir=/path/to/out
  python run.py stage=finetune stage.epochs=100 stage.encoder_name=resnet101
"""

from pathlib import Path

# Resolver: project root = directory containing run.py (stable regardless of cwd)
_RUN_DIR = Path(__file__).resolve().parent

import hydra
from omegaconf import DictConfig, OmegaConf

# Register before @hydra.main so config can use ${project_root:}
OmegaConf.register_new_resolver("project_root", lambda: str(_RUN_DIR), replace=True)

import logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Dispatch to the selected stage."""
    stage_name = cfg.stage.name
    log.info("Running stage: %s", stage_name)
    OmegaConf.resolve(cfg)

    if stage_name == "pseudomask":
        from src.stages.pseudomask import run
    elif stage_name == "refinement":
        from src.stages.refinement import run
    elif stage_name == "prepare":
        from src.stages.prepare import run
    elif stage_name == "finetune":
        from src.stages.finetune import run
    elif stage_name == "infer":
        from src.stages.infer import run
    elif stage_name == "evaluate":
        from src.stages.evaluate import run
    elif stage_name == "compare":
        from src.stages.compare import run
    elif stage_name == "compare_bboxes":
        from src.stages.compare_bboxes import run
    else:
        raise ValueError(
            f"Unknown stage: {stage_name}. "
            "Use one of: pseudomask, refinement, prepare, finetune, infer, evaluate, compare, compare_bboxes"
        )

    run(cfg)
    log.info("Stage %s completed.", stage_name)


if __name__ == "__main__":
    main()
