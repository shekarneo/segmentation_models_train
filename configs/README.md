# Configs (matches sam2-pipeline layout)

- **`config.yaml`** – Hydra main config for `run.py` (pipeline entry point); defines `paths` and default stage.
- **`sweep.yaml`** – WandB hyperparameter sweep; use `wandb sweep configs/sweep.yaml`.
- **`stage/*.yaml`** – Per-stage config (training/inference/prepare defaults live here; no separate default.yaml). `pseudomask`, `refinement`, `prepare`, `finetune`, `infer`, `evaluate`, `compare`. Override with `python run.py stage=X stage.key=value`.
