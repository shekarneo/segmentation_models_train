## segmentation_models_train

End‑to‑end **semantic segmentation** pipeline for industrial defect detection (e.g. scratch, stain) built around **segmentation_models_pytorch (SMP)** and **SAM2**.

The pipeline is fully staged (prepare → pseudomask → refinement → finetune → infer → evaluate → compare → compare_bboxes), uses **Hydra** for configuration, and integrates **local SAM2 checkpoints** via a bundled `sam2-pipeline`.

---

## Features

- **SMP architecture + backbone training**
  - Configurable architecture + backbone from YAML (e.g. DeepLabV3+, U-Net family, FPN, PSPNet)
  - Multiple backbones (e.g. `resnet50`, `resnet101`, `convnext_base`, MiT, EfficientNet)
  - Class‑balanced losses: CE with class weights, Focal loss, Dice loss
  - Rich augmentations (random crop around defect, flips, shift/scale/rotate, CLAHE, noise, blur)
  - **Per‑class validation metrics** (IoU, Dice, Precision, Recall, F1) with logging to terminal
  - Multi‑GPU training via **DistributedDataParallel (DDP)** using `torchrun`
- **SAM2 integration**
  - Uses only the **local** `sam2-pipeline` and checkpoints (no external paths required)
  - Pseudomask generation with tiling and bbox prompts
  - Optional mask refinement (`kmeans` / threshold) with safe fallbacks
- **Data preparation**
  - Converts raw LabelMe JSONs / masks to `dataset/defect_data`
  - Builds a single source of truth `label_mapping.json` (class ↔ id)
  - Optional object cropping and dataset tiling
- **Inference & evaluation**
  - Tiled inference for large images
  - Overlays with per‑class pixel statistics
- **DeepLab vs SAM2 comparison**
  - Runs SAM2 on test images (or reuses cached SAM2 masks)
  - Global and per‑class metrics + CSV exports
  - Binary and per‑class visualizations with **class names** and TP/FP/FN legend
- **Weights & Biases**
  - Optional logging for training
  - Sweep support via `configs/sweep.yaml` and `run_finetune_wandb.py`

---

## Architectures and Backbones (SMP)

This pipeline uses `segmentation_models_pytorch` and supports selecting **both**:

- `architecture` (decoder/head family)
- `encoder_name` (backbone/feature extractor)

### Supported architectures

Set `architecture` to one of:

- `DeepLabV3Plus` (default, backward compatible with existing configs/checkpoints)
- `DeepLabV3`
- `Unet`
- `UnetPlusPlus`
- `FPN`
- `PSPNet`
- `PAN`
- `MAnet`
- `Linknet`

Also accepted aliases:

- `deeplabv3+` -> `DeepLabV3Plus`
- `unet++` -> `UnetPlusPlus`

### Backbone (`encoder_name`) examples

Commonly used backbones in SMP:

- ResNet family: `resnet34`, `resnet50`, `resnet101`
- EfficientNet family: `efficientnet-b0` ... `efficientnet-b7`
- SE-ResNet / SENet family: `se_resnet50`, `se_resnet101`
- MiT family (SegFormer encoders): `mit_b0` ... `mit_b5`
- ConvNeXt family: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

> Note: ConvNeXt names are normalized automatically by the model builder to SMP timm-universal form (e.g. `convnext_base` -> `tu-convnext_base`).

### Model profiles (`configs/models`)

Architecture/backbone settings are centralized in `configs/models/*.yaml` and selected from stage config via `model_profile`.

Example profiles:

- `deeplabv3plus_resnet50.yaml`
- `deeplabv3plus_resnet101.yaml`
- `unet_convnext_base.yaml`
- `unet_resnet50.yaml`
- `fpn_convnext_base.yaml`
- `unetplusplus_resnet50.yaml`

### YAML configuration example

```yaml
# configs/stage/finetune.yaml
model_profile: unet_convnext_base
```

Or override manually from CLI:

```bash
python run.py stage=finetune stage.architecture=FPN stage.encoder_name=efficientnet-b3
```

### Architecture-specific notes

- `DeepLabV3Plus` / `DeepLabV3` use DeepLab-specific params:
  - `encoder_output_stride`
  - `decoder_channels`
  - `decoder_atrous_rates`
- Non-DeepLab architectures ignore those DeepLab-only fields safely.
- Existing checkpoints/configs continue to work because default is `architecture: DeepLabV3Plus`.

---

## Repository layout

- `run.py`  
  Main Hydra entrypoint. All stages are run as:

  ```bash
  python run.py stage=<name>
  ```

- `configs/`
  - `config.yaml` – global paths and Hydra defaults
  - `sweep.yaml` – WandB sweep configuration
  - `stage/`
    - `prepare.yaml` – dataset preparation
    - `pseudomask.yaml` – SAM2 pseudomask generation
    - `refinement.yaml` – SAM2 mask refinement
    - `finetune.yaml` – DeepLabV3+ training
    - `infer.yaml` – inference
    - `evaluate.yaml` – evaluation
    - `compare.yaml` – DeepLab vs SAM2 comparison

- `src/`
  - `config.py` – shared `Config` dataclass + helpers to convert Hydra configs into runtime configs
  - `dataset.py` – `DefectSegmentationDataset`
  - `model.py` – DeepLabV3+ model construction
  - `stages/`
    - `prepare.py` – Stage 1: dataset preparation
    - `pseudomask.py` – Stage 2: SAM2 pseudomask generation
    - `refinement.py` – Stage 2.5: SAM2 mask refinement
    - `finetune.py` – Stage 3: model training
    - `infer.py` – Stage 4: inference
    - `evaluate.py` – Stage 5: evaluation on dataset
    - `compare.py` – Stage 6: DeepLab vs SAM2 comparison
    - `compare_bboxes.py` – Stage 7: DeepLab mask -> bboxes comparison (no SAM2)
  - `utils/`
    - `image_utils.py` – object cropping, tiling helpers
    - `label_mapping.py` – loading and handling `label_mapping.json`
    - `sam2_infer_tile.py` / `run_sam2_tile.py` – helpers for running SAM2 on tiles

- `dataset/`
  - `defect_data/` – prepared dataset (images, masks, `label_mapping.json`)
  - `Consensus_Mask_Reviewer_Test/` – example test data for comparison (images + LabelMe JSON)

- `sam2-pipeline/`  
  Local SAM2 pipeline (cloned or copied into this repo). Used by `pseudomask.py` and `compare.py`.

- `checkpoints/`  
  Trained DeepLabV3+ model checkpoints (`best_model.pth`, `last_model.pth`, etc.).

- `Dockerfile`, `docker-compose.yml`  
  Containerized environment for reproducible training/inference (optional).

---

## Installation

### 1. Clone and set up environment

```bash
git clone <REPO_URL> segmentation_models_train
cd segmentation_models_train

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Make sure you install a CUDA‑enabled PyTorch version compatible with your GPU if you plan to train on GPU.

### 2. Local SAM2 pipeline

This project expects `sam2-pipeline` to be present **inside this repo**:

```bash
ls sam2-pipeline/src
ls sam2-pipeline/checkpoints/base_models/sam2_hiera_large.pt
```

You can optionally override paths via environment variables:

```bash
export SAM2_PIPELINE_ROOT=/absolute/path/to/sam2-pipeline
export SAM2_CKPT_PATH=/absolute/path/to/sam2_hiera_large.pt
```

If these are not set, defaults under `sam2-pipeline/` in this repo are used.

---

## Stages and usage

All stages are run via `run.py` with Hydra:

```bash
cd segmentation_models_train
python run.py stage=<name>
```

You can override any config field with `stage.<field>=<value>`.

### Stage 1 – Pseudomask generation (`stage=pseudomask`)

Runs SAM2 on full‑size images with LabelMe bbox prompts and outputs LabelMe polygons as pseudomasks.

```bash
python run.py stage=pseudomask
```

Key config (`configs/stage/pseudomask.yaml`):

- `input_dir`, `output_dir`
- `model_repo` (e.g. `facebook/sam2-hiera-large`)
- `local_ckpt` (optional; if `null`, defaults to `sam2-pipeline/checkpoints/base_models/sam2_hiera_large.pt`)
- `tile_size`, `overlap`
- `label_for_masks`, `polygon_label_from_bbox`, `keep_rectangles`

### Stage 1.5 – SAM2 refinement (`stage=refinement`)

Optionally refines SAM2 masks using intensity thresholding or k‑means, controlled via `configs/stage/refinement.yaml`.

```bash
python run.py stage=refinement
```

The compare stage shares this same `refinement.yaml` (via `stage.refine_config`) so you can apply identical refinement settings when treating SAM2 as GT.

### Stage 2 – Prepare dataset (`stage=prepare`)

Converts raw LabelMe JSONs / masks into a training‑ready dataset at `dataset/defect_data/` and writes `label_mapping.json`.

```bash
python run.py stage=prepare
```

Important options (`configs/stage/prepare.yaml`):

- `raw_dir` – source of LabelMe JSONs / masks
- `out_dir` – output dataset root (default: `dataset/defect_data`)
- `splits`, `split_by_ratio`, `split_ratios`, `split_seed`
- `prepare_tile_enabled`, `prepare_tile_size`, `prepare_tile_overlap`
- `clahe_enabled`, `clahe_clip_limit`, `clahe_tile_grid`
- `crop_object_enabled`, `crop_object_padding`

The `label_mapping.json` created here is the **single source of truth** for classes. Training, inference, and comparison all use this file (or the mapping saved into checkpoints).

### Stage 3 – Finetune DeepLabV3+ (`stage=finetune`)

Trains DeepLabV3+ on `dataset/defect_data`.

```bash
python run.py stage=finetune
```

Key training config (`configs/stage/finetune.yaml`):

- **Data & paths**
  - `dataset_root: ${paths.root}/dataset/defect_data`
  - `checkpoints_dir: ${paths.root}/checkpoints`
- **Model**
  - `encoder_name`, `encoder_weights`
  - `image_size`, `num_classes` (overridden from `label_mapping.json`)
- **Training hyperparameters**
  - `batch_size`, `num_workers`, `epochs`, `learning_rate`, `weight_decay`
  - `optimizer_name` (`adamw` / `adam`), `scheduler_name` (`onecycle` / `cosine`)
  - `use_class_weights`, `use_focal_loss`, `use_dice_loss`, `focal_gamma`
- **Augmentations**
  - `augmentations_enabled`, `aug_random_crop_defect_prob`, `aug_random_crop_defect_size`
  - Flips, CLAHE, noise, blur, shift/scale/rotate
- **Early stopping & logging**
  - `use_early_stopping`, `early_stop_patience`, `early_stop_min_delta`, `early_stop_metric`
  - `use_wandb`, `wandb_project`, `wandb_run_name`
- **Per‑class validation metrics**
  - `per_class_metrics: true` logs, per epoch:

    ```text
    Epoch N: train_loss=..., val_loss=..., val_iou=..., val_dice=...
      [class Scratch] IoU=..., Dice=..., Prec=..., Rec=..., F1=...
      [class Stain]   ...
    ```

> **Multi‑GPU (DDP):**
>
> ```bash
> torchrun --nproc_per_node=2 run.py stage=finetune
> ```

Checkpoints embed both the model and `label_mapping` so classes travel with the checkpoint.

### Stage 4 – Inference (`stage=infer`)

Runs DeepLabV3+ inference (optionally tiled) and writes masks + overlays.

```bash
python run.py stage=infer
```

Config (`configs/stage/infer.yaml`):

- `input_dir`, `output_dir`, `checkpoint`
- `image_size`, `num_classes`, `tiled_inference`, `tile_size`, `tile_overlap`
- `clahe_enabled`, `crop_object_enabled`, `crop_object_padding`
- `label_mapping_path` fallback if mapping is not in the checkpoint

### Stage 5 – Evaluation (`stage=evaluate`)

Evaluates on a dataset split (e.g. `val` / `test`) using ground‑truth masks from `dataset_root`.

```bash
python run.py stage=evaluate
```

Config (`configs/stage/evaluate.yaml`) is similar to `infer.yaml` but uses `dataset_root` + `split`.

### Stage 6 – DeepLab vs SAM2 comparison (`stage=compare`)

Treats SAM2 masks as GT and DeepLab predictions as predictions, computes metrics, and generates visualizations.

```bash
python run.py stage=compare
```

Important options (`configs/stage/compare.yaml`):

- `data_dir` – test images + LabelMe JSONs (e.g. `Consensus_Mask_Reviewer_Test`)
- `output_dir` – comparison outputs (masks, jpgs, metrics)
- `checkpoint` – DeepLab checkpoint
- `label_mapping_path` – points directly to `dataset/defect_data/label_mapping.json`
- `sam2_masks_dir` – optional cache for SAM2 GT masks
- Refinement:
  - `refine: kmeans|threshold|null`
  - `refine_config: ${paths.config}/stage/refinement.yaml`
- Crop/tiling:
  - `crop_object_enabled`, `crop_object_padding`
  - `tiled_inference`, `tile_size`, `tile_overlap`
- **Metrics output:**
  - `save_json: ${paths.outputs}/stage6_compare/metrics.json`
  - `per_class_metrics: true`

Outputs:

- `metrics.json` – per‑image and mean IoU, Dice, Precision, Recall, F1, F2
- `metrics.csv` – per‑image rows + summary row
- `metrics_per_class.csv` – per‑image, per‑class metrics
- Visualizations:
  - `<stem>_compare.jpg` – binary (any defect vs background), with legend
  - `<stem>_class_<ClassName>_compare.jpg` – per‑class, titled `Class: <ClassName>`

### Stage 7 – Bbox comparison (`stage=compare_bboxes`)
Converts DeepLabV3+ semantic masks to class-labeled rectangle bboxes (via connected components, with optional dilation), compares them with LabelMe rectangle/polygon bboxes from the test JSONs, and optionally exports the predicted bboxes back to LabelMe-compatible JSONs (so you can skip SAM2-based stages).

```bash
python run.py stage=compare_bboxes
```

Key options (`configs/stage/compare_bboxes.yaml`):
- `compare_with_bboxes` – enable IoU matching metrics
- `dilate_iterations`, `dilate_kernel_size` – extend mask before bbox extraction
- `save_labelme_jsons`, `labelme_pred_dir` – export predicted bboxes as LabelMe JSONs
- `iou_threshold` – IoU threshold for greedy box matching

---

## WandB sweeps

Sweeps use a dedicated entrypoint and config:

- `configs/sweep.yaml` – sweep definition
- `run_finetune_wandb.py` – program used by WandB agents

Example:

```bash
cd segmentation_models_train

wandb sweep configs/sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

`finetune.py` applies `wandb.config` keys (lr, batch_size, epochs, encoder_name, etc.) to the runtime config.

---

## Docker (optional)

You can run everything inside Docker with GPU support:

```bash
cd segmentation_models_train

cp .env.example .env
# Edit .env to set UID and GID (or export on the CLI)
UID=$(id -u) GID=$(id -g) docker compose build
UID=$(id -u) GID=$(id -g) docker compose run --rm segmentation_models_train
```

Inside the container:

```bash
python run.py stage=prepare
python run.py stage=finetune
python run.py stage=compare
```

The container installs both Python and local SAM2 dependencies and uses the same project layout as the host.

---

## Notes and tips

- **Class definitions**  
  All stages use `label_mapping.json` from `dataset/defect_data` (or the mapping saved in the checkpoint) so classes are consistent across training, inference, and comparison.

- **Reproducibility**  
  Training seeds NumPy, PyTorch, and workers. Checkpoints save both the model state and the config (plus `label_mapping`), making runs easier to reproduce.

- **Extensibility**  
  To add new stages or variants, follow the existing pattern:
  - Add a `src/stages/<new_stage>.py`
  - Add `configs/stage/<new_stage>.yaml`
  - Wire it in `run.py` via Hydra.

