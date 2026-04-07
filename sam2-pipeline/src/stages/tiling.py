"""Stage 0: Tile full-resolution images and clip GT bboxes to tile coordinates.

Splits each full-resolution image (e.g. 3032×5320) into overlapping 1024×1024 tiles
and clips GT bounding box annotations to tile-local coordinates.  Only tiles that
contain at least one valid GT annotation are saved.

WHY THIS STAGE EXISTS
---------------------
SAM2 always processes images at its native 1024×1024 resolution.  A 5 px wide scratch
on a 3032×5320 image becomes ≈1 px wide after SAM2's internal resize — too small for
the ViT patch grid (16×16) to see.  By tiling first, the same 5 px scratch is presented
to SAM2 at its original pixel scale, giving it a chance to produce a tight mask.

PIPELINE FLOW (when using this stage)
--------------------------------------
  Stage 0  tiling      → outputs/stage0_tiled_<family>/{train,test}/
  Stage 1  pseudomask  ← reads stage0_tiled_<family>  (set stage.input_dir in pseudomask config)
  Stage 2  augment     → outputs/stage2_augmented_<family>/
  Stage 3  finetune    → use_tiling: false  (images already 1024×1024)
  Stage 4  inference   → unchanged (always tiles at runtime)
"""

import json
import logging
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_from_shape(shape: dict):
    """Return (x1, y1, x2, y2) from a LabelMe shape, or None if invalid.

    Handles both ``rectangle`` (two-point) and ``polygon`` shapes.
    For polygons the axis-aligned bounding box of the polygon is used.
    """
    pts = shape.get('points', [])
    stype = shape.get('shape_type', 'rectangle')

    if stype == 'rectangle' and len(pts) >= 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    elif stype == 'polygon' and len(pts) >= 3:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2


def _load_annotations(json_path: Path) -> list:
    """Load all valid annotations from a LabelMe JSON file.

    Returns:
        List of dicts: ``[{'label': str, 'bbox': [x1, y1, x2, y2]}, ...]``
    """
    with open(json_path) as f:
        data = json.load(f)

    annotations = []
    for shape in data.get('shapes', []):
        result = _bbox_from_shape(shape)
        if result is None:
            continue
        x1, y1, x2, y2 = result
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue  # degenerate box
        label = shape.get('label', 'defect')
        label = label.replace('_unique', '').rstrip('0123456789')
        annotations.append({'label': label, 'bbox': [x1, y1, x2, y2]})

    return annotations


def _generate_tiles(image, annotations: list, tile_size: int,
                    overlap: int, min_bbox_dim: int = 5, keep_all_tiles: bool = False) -> list:
    """Tile an image and assign GT bboxes to tiles.

    Bbox assignment rule:
        A bbox belongs to the tile whose region contains the bbox's **centre
        point**.  This means each annotation is assigned to exactly one tile —
        no duplicate labels across overlapping tiles.

    The bbox is then clipped to the tile region and converted to tile-local
    (pixel) coordinates.  Bboxes whose clipped size is smaller than
    ``min_bbox_dim`` in either dimension are discarded.

    Returns:
        List of tile dicts — one per tile that has at least one annotation::

            {
                'tile':        np.ndarray (tile_size, tile_size, 3),
                'origin':      (x1, y1) in full-image coordinates,
                'annotations': [{'label': str, 'bbox': [lx1, ly1, lx2, ly2]}, ...],
            }
    """
    img_h, img_w = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    for y_step in range(0, img_h, stride):
        for x_step in range(0, img_w, stride):
            # Compute tile bounds — adjust start so the tile is always tile_size wide/tall
            # (same strategy as sliding_window_tiles in tiled_inference.py)
            x2t = min(x_step + tile_size, img_w)
            y2t = min(y_step + tile_size, img_h)
            x1t = max(0, x2t - tile_size)
            y1t = max(0, y2t - tile_size)

            # Assign and clip bboxes whose centre falls inside this tile
            tile_anns = []
            for ann in annotations:
                gx1, gy1, gx2, gy2 = ann['bbox']
                cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0

                if not (x1t <= cx < x2t and y1t <= cy < y2t):
                    continue  # Centre outside this tile → belongs to another tile

                # Clip to tile region, convert to tile-local coordinates
                lx1 = max(gx1, x1t) - x1t
                ly1 = max(gy1, y1t) - y1t
                lx2 = min(gx2, x2t) - x1t
                ly2 = min(gy2, y2t) - y1t

                if lx2 - lx1 < min_bbox_dim or ly2 - ly1 < min_bbox_dim:
                    continue  # Too small after clipping

                tile_anns.append({'label': ann['label'],
                                  'bbox': [lx1, ly1, lx2, ly2]})

            if not tile_anns and not keep_all_tiles:
                continue  # No GT in this tile — skip (unless keep_all_tiles is true)

            # Extract tile image and pad to tile_size if at image boundary
            tile_img = image[y1t:y2t, x1t:x2t].copy()
            pad_h = tile_size - tile_img.shape[0]
            pad_w = tile_size - tile_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )

            tiles.append({
                'tile': tile_img,
                'origin': (x1t, y1t),
                'annotations': tile_anns,
            })

    return tiles


def _build_labelme_json(tile_stem: str, annotations: list,
                        tile_h: int, tile_w: int) -> dict:
    """Build a LabelMe-format JSON dict with rectangle shapes.

    The output JSON is compatible with pseudomask.py's ``load_labelme_annotation``
    which expects ``shape_type: rectangle`` with two corner points.
    """
    shapes = [
        {
            'label': ann['label'],
            'points': [
                [ann['bbox'][0], ann['bbox'][1]],
                [ann['bbox'][2], ann['bbox'][3]],
            ],
            'group_id': None,
            'description': '',
            'shape_type': 'rectangle',
            'flags': {},
            'mask': None,
        }
        for ann in annotations
    ]
    return {
        'version': '5.0',
        'flags': {},
        'shapes': shapes,
        'imagePath': f'{tile_stem}.jpg',
        'imageData': None,
        'imageHeight': tile_h,
        'imageWidth': tile_w,
    }


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def _split_files(json_files: list, val_ratio: float,
                 test_ratio: float, seed: int) -> dict:
    """Randomly split annotation files into train / val / test.

    Returns an ordered dict ``{'train': [...], 'val': [...], 'test': [...]}``.
    'val' and 'test' keys are omitted when their ratio is 0.
    Files are sorted before shuffling so the split is reproducible across
    different filesystems.
    """
    import random
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1.0"
        )

    files = sorted(json_files)          # deterministic order before shuffle
    random.Random(seed).shuffle(files)  # reproducible shuffle

    n = len(files)
    n_test = round(n * test_ratio) if test_ratio > 0 else 0
    n_val  = round(n * val_ratio)  if val_ratio  > 0 else 0
    n_train = n - n_val - n_test

    result = {'train': files[:n_train]}
    if n_val > 0:
        result['val']  = files[n_train:n_train + n_val]
    if n_test > 0:
        result['test'] = files[n_train + n_val:]

    return result


# ---------------------------------------------------------------------------
# Per-split tiling worker (shared by both split and no-split modes)
# ---------------------------------------------------------------------------

def _tile_split(json_files: list, split_output: Path, tile_size: int,
                overlap: int, min_bbox_dim: int, split_name: str, keep_all_tiles: bool = False) -> dict:
    """Tile all images in ``json_files`` and write outputs to ``split_output``."""
    split_output.mkdir(parents=True, exist_ok=True)
    n_images = n_tiles_total = n_skipped = 0

    for json_path in tqdm(json_files, desc=split_name):
        img_path = json_path.with_suffix('.jpg')
        if not img_path.exists():
            for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                alt = json_path.with_suffix(ext)
                if alt.exists():
                    img_path = alt
                    break
            else:
                log.debug(f"No image for {json_path.name}, skipping")
                continue

        image = cv2.imread(str(img_path))
        if image is None:
            log.warning(f"Failed to read: {img_path}")
            continue

        annotations = _load_annotations(json_path)
        if not annotations:
            n_skipped += 1
            log.debug(f"  {json_path.name}: no valid annotations, skipping")
            continue

        stem = json_path.stem
        tile_results = _generate_tiles(image, annotations, tile_size,
                                       overlap, min_bbox_dim, keep_all_tiles=keep_all_tiles)
        if not tile_results:
            n_skipped += 1
            if keep_all_tiles:
                log.debug(f"  {stem}: 0 tiles produced (unexpected)")
            else:
                log.debug(f"  {stem}: 0 tiles with annotations produced")
            continue

        for idx, tr in enumerate(tile_results):
            tile_stem = f'{stem}_t{idx:04d}'
            th, tw = tr['tile'].shape[:2]
            cv2.imwrite(str(split_output / f'{tile_stem}.jpg'), tr['tile'])
            lm = _build_labelme_json(tile_stem, tr['annotations'], th, tw)
            with open(split_output / f'{tile_stem}.json', 'w') as f:
                json.dump(lm, f, indent=2)

        n_images += 1
        n_tiles_total += len(tile_results)

    avg = n_tiles_total / max(n_images, 1)
    log.info(f"  {split_name}: {n_images} images → {n_tiles_total} tiles "
             f"({avg:.1f} avg/image, {n_skipped} skipped — no annotations)")
    return {'images': n_images, 'tiles': n_tiles_total, 'skipped': n_skipped}


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> dict:
    """Stage 0: Tile full-resolution images and clip GT bboxes to each tile."""
    log.info("=" * 60)
    log.info("Stage 0: Image Tiling")
    log.info("=" * 60)

    family = str(cfg.model.get('family', 'sam2'))
    base_output = Path(str(cfg.stage.output_dir))
    output_dir = base_output.parent / (base_output.name + '_' + family)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size    = int(cfg.stage.tile_size)
    overlap      = int(cfg.stage.overlap)
    min_bbox_dim = int(cfg.stage.get('min_bbox_dim', 5))
    keep_all_tiles = bool(cfg.stage.get('keep_all_tiles', False))

    log.info(f"Tile size : {tile_size}×{tile_size} px")
    log.info(f"Overlap   : {overlap} px")
    log.info(f"Min bbox  : {min_bbox_dim} px (after clipping)")
    log.info(f"Keep all tiles: {keep_all_tiles} (include background tiles)")
    log.info(f"Output    : {output_dir}")

    stats = {}
    split_cfg     = cfg.stage.get('split', {})
    split_enabled = bool(split_cfg.get('enabled', False))

    if split_enabled:
        # ── SPLIT MODE ────────────────────────────────────────────────────
        # Read ALL images from a single source dir, split at image level,
        # then tile each split independently.  This prevents data leakage
        # that would occur if you tiled first and then split by tile.
        source_dir = Path(str(cfg.data.paths.train))
        val_ratio  = float(split_cfg.get('val_ratio',  0.0))
        test_ratio = float(split_cfg.get('test_ratio', 0.0))
        seed       = int(split_cfg.get('seed', 42))

        all_json = sorted(source_dir.glob('*.json'))
        if not all_json:
            raise FileNotFoundError(f"No annotation JSONs found in: {source_dir}")

        split_files = _split_files(all_json, val_ratio, test_ratio, seed)

        # Log the split
        log.info(f"\nData split (seed={seed}):")
        for name, files in split_files.items():
            log.info(f"  {name:5s}: {len(files)} images  "
                     f"({100 * len(files) / len(all_json):.0f}%)")

        for split_name, files in split_files.items():
            log.info(f"\n--- {split_name} ---")
            split_output = output_dir / split_name
            stats[split_name] = _tile_split(
                files, split_output, tile_size, overlap, min_bbox_dim, split_name, keep_all_tiles
            )

    else:
        # ── NO-SPLIT MODE (default) ───────────────────────────────────────
        # Use cfg.data.paths.train and cfg.data.paths.test as-is.
        train_path = str(cfg.data.paths.train)
        test_path  = str(cfg.data.paths.test)

        splits_to_process = [('train', train_path)]
        if train_path != test_path:
            splits_to_process.append(('test', test_path))
        else:
            log.info("Train and test paths are the same — processing once.")

        for split_name, split_path in splits_to_process:
            split_dir = Path(split_path)
            log.info(f"\n--- {split_name} ---")
            log.info(f"Input : {split_dir}")
            json_files = sorted(split_dir.glob('*.json'))
            log.info(f"Found {len(json_files)} annotation files")
            split_output = output_dir / split_name
            stats[split_name] = _tile_split(
                json_files, split_output, tile_size, overlap, min_bbox_dim, split_name, keep_all_tiles
            )

        # If train == test, copy train tiles to test/ as well
        if train_path == test_path:
            stats['test'] = stats.get('train', {})
            train_out = output_dir / 'train'
            test_out  = output_dir / 'test'
            test_out.mkdir(parents=True, exist_ok=True)
            for f in train_out.iterdir():
                if f.is_file():
                    shutil.copy2(f, test_out / f.name)
            log.info("Copied train tiles to test/ (same source path).")

    log.info("\n" + "=" * 60)
    log.info(f"Tiling complete!  Output: {output_dir}")

    # Summary table
    log.info("\nSummary:")
    for name, s in stats.items():
        log.info(f"  {name:5s}: {s['images']} images → {s['tiles']} tiles "
                 f"({s['skipped']} skipped)")

    return stats
