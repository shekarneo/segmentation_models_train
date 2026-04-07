"""
Stage 1: SAM2 pseudomask generation.

Full logic: load SAM2 via sam2-pipeline, tile full-size images + bboxes,
run SAM2 per tile, merge polygons to full-image LabelMe JSONs.
"""

from __future__ import annotations

import json
import os
import sys
import importlib.util
from pathlib import Path
from typing import List

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import label as scipy_label


def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p


def _pipeline_root(root: Path) -> Path:
    pipeline_root = root / "sam2-pipeline"
    pipeline_root = Path(os.environ.get("SAM2_PIPELINE_ROOT", str(pipeline_root)))
    pipeline_root = pipeline_root.resolve()
    pipeline_src = pipeline_root / "src"
    if not (pipeline_src / "__init__.py").exists():
        raise FileNotFoundError(
            f"SAM2 pipeline not found at '{pipeline_src}'. "
            "Expected a local copy at sam2-pipeline/src or set SAM2_PIPELINE_ROOT."
        )
    return pipeline_root


def _import_sam2_pipeline(pipeline_root: Path):
    """
    Import the local sam2-pipeline 'src/' package under an isolated alias so:
    - its internal relative imports work
    - it does not collide with this repo's 'src' package
    """
    pkg = "sam2_pipeline"
    if pkg in sys.modules:
        return sys.modules[pkg]
    src_dir = pipeline_root / "src"
    init_py = src_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg,
        init_py,
        submodule_search_locations=[str(src_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for SAM2 pipeline at '{src_dir}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    return module


def _find_image_for_json(json_path: Path) -> Path | None:
    stem = json_path.stem
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        candidate = json_path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return None


def _load_labelme_json(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def _save_labelme_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def _append_pseudomask_shapes(
    data: dict,
    polys: List[dict],
    label_for_masks: str,
    polygon_label_from_bbox: bool,
    keep_rectangles: bool,
) -> dict:
    sam2_shapes = []
    for entry in polys:
        pts = entry.get("points", [])
        if not pts:
            continue
        poly_label = entry.get("label", label_for_masks) if polygon_label_from_bbox else label_for_masks
        sam2_shapes.append({
            "label": poly_label,
            "points": pts,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })
    original_shapes = data.get("shapes", [])
    kept = original_shapes if keep_rectangles else [sh for sh in original_shapes if sh.get("shape_type") != "rectangle"]
    data["shapes"] = kept + sam2_shapes
    return data


def run(cfg: DictConfig) -> None:
    root = _resolve_path(cfg.paths.root, Path.cwd())
    input_dir = _resolve_path(cfg.stage.input_dir, root)
    output_dir = _resolve_path(cfg.stage.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_repo = str(cfg.stage.get("model_repo", "facebook/sam2-hiera-large"))
    device = str(cfg.get("device", "cuda"))
    local_ckpt = cfg.stage.get("local_ckpt") or os.environ.get("SAM2_CKPT_PATH")
    if not local_ckpt:
        default_ckpt = root / "sam2-pipeline" / "checkpoints" / "base_models" / "sam2_hiera_large.pt"
        if default_ckpt.exists():
            local_ckpt = str(default_ckpt)
    tile_size = int(cfg.stage.get("tile_size", 1024))
    overlap = int(cfg.stage.get("overlap", 256))
    label_for_masks = str(cfg.stage.get("label_for_masks", "sam2_pseudomask"))
    polygon_label_from_bbox = bool(cfg.stage.get("polygon_label_from_bbox", False))
    keep_rectangles = bool(cfg.stage.get("keep_rectangles", True))
    merge_multi_mask = bool(cfg.stage.get("merge_multi_mask", True))

    pipeline_root = _pipeline_root(root)
    _import_sam2_pipeline(pipeline_root)
    from sam2_pipeline.models.sam2_lora import SAM2LoRA  # type: ignore
    from sam2_pipeline.stages.pseudomask import (  # type: ignore  # sam2-pipeline
        mask_to_polygon,
        _predict_sam2_batch,
        _bbox_to_binary_mask,
        _is_valid_mask,
        _bbox_to_polygon,
    )
    from sam2_pipeline.stages.tiling import _load_annotations, _generate_tiles  # type: ignore

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Loading SAM2 model: {model_repo} on {device}")
    kwargs = {"local_path": local_ckpt} if local_ckpt else {}
    model = SAM2LoRA(model_id=model_repo, device=device, **kwargs)
    print("SAM2 model loaded.")

    for json_path in json_files:
        img_path = _find_image_for_json(json_path)
        if img_path is None:
            print(f"[WARN] No image found for {json_path.name}, skipping.")
            continue
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Failed to read image {img_path}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        anns = _load_annotations(json_path)
        if not anns:
            print(f"[INFO] No annotations in {json_path.name}, skipping.")
            continue
        tiles = _generate_tiles(
            image_rgb, anns,
            tile_size=tile_size,
            overlap=overlap,
            min_bbox_dim=5,
            keep_all_tiles=False,
        )
        if not tiles:
            print(f"[INFO] No tiles produced for {json_path.name}, skipping.")
            continue

        print(f"{json_path.name}: {len(tiles)} tiles (tile_size={tile_size}, overlap={overlap})")
        predictor = model.predictor
        # Raw polygons collected per (global) bbox; merged later if requested
        polys: List[dict] = []

        for tr in tiles:
            tile_img = tr["tile"]
            origin_x, origin_y = tr["origin"]
            tile_anns = tr["annotations"]
            if not tile_anns:
                continue
            local_boxes: List[List[int]] = [a["bbox"] for a in tile_anns]
            tile_h, tile_w = tile_img.shape[:2]
            predictor.set_image(tile_img)

            chunk_size = 8
            tile_results: List[tuple] = []
            try:
                for start in range(0, len(local_boxes), chunk_size):
                    chunk = local_boxes[start : start + chunk_size]
                    b_np = np.array(chunk, dtype=np.float64)
                    try:
                        chunk_res = _predict_sam2_batch(predictor, b_np, multimask_output=False)
                        tile_results.extend(chunk_res)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and len(chunk) > 1:
                            for single_box in chunk:
                                b_np_single = np.array([single_box], dtype=np.float64)
                                single_res = _predict_sam2_batch(predictor, b_np_single, multimask_output=False)
                                tile_results.extend(single_res)
                        else:
                            raise
            except RuntimeError as e:
                print(f"[WARN] SAM2 tile failed {img_path.name} ({origin_x},{origin_y}): {e}")
                continue

            for ann, (mask, _score) in zip(tile_anns, tile_results):
                bbox_local = ann["bbox"]
                if not _is_valid_mask(mask):
                    mask = _bbox_to_binary_mask(bbox_local, tile_h, tile_w)
                if not _is_valid_mask(mask):
                    continue
                mask = np.asarray(mask)
                mask_crop = mask[:tile_h, :tile_w]
                x1, y1, x2, y2 = [int(round(v)) for v in bbox_local]
                x1 = max(0, min(x1, tile_w - 1))
                x2 = max(0, min(x2, tile_w))
                y1 = max(0, min(y1, tile_h - 1))
                y2 = max(0, min(y2, tile_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                sub = (mask_crop > 0).astype(np.uint8)
                sub_bbox = sub[y1:y2, x1:x2]
                if sub_bbox.size == 0:
                    continue
                labeled, n = scipy_label(sub_bbox)
                if n > 1 and not merge_multi_mask:
                    # Keep only the largest connected region within this bbox
                    areas = [(labeled == i).sum() for i in range(1, n + 1)]
                    best_idx = int(np.argmax(areas)) + 1
                    keep = (labeled == best_idx).astype(np.uint8)
                    sub_bbox[...] = 0
                    sub_bbox[keep > 0] = 1
                    sub[y1:y2, x1:x2] = sub_bbox
                    mask_crop = sub
                pts_tile = mask_to_polygon(mask_crop)
                if not pts_tile:
                    pts_tile = _bbox_to_polygon(bbox_local)
                if not pts_tile:
                    continue
                pts_full = [[float(x + origin_x), float(y + origin_y)] for (x, y) in pts_tile]
                # Compute global bbox key (in full-image coords) so we can merge polygons
                gx1 = int(round(x1 + origin_x))
                gy1 = int(round(y1 + origin_y))
                gx2 = int(round(x2 + origin_x))
                gy2 = int(round(y2 + origin_y))
                bbox_key = (gx1, gy1, gx2, gy2)
                polys.append(
                    {
                        "label": ann.get("label"),
                        "points": pts_full,
                        "bbox_key": bbox_key,
                    }
                )

        # Optionally merge polygons that belong to the same original bbox across tiles
        if merge_multi_mask and polys:
            grouped: dict[tuple[int, int, int, int, str], List[List[List[float]]]] = {}
            for p in polys:
                label = str(p.get("label"))
                k = p.get("bbox_key")
                if k is None:
                    continue
                key = (k[0], k[1], k[2], k[3], label)
                grouped.setdefault(key, []).append(p["points"])

            merged_polys: List[dict] = []
            for (gx1, gy1, gx2, gy2, label), poly_list in grouped.items():
                if not poly_list:
                    continue
                # Rasterize all polygons for this bbox onto a full-image mask, then extract polygons once
                mask_full = np.zeros((h, w), dtype=np.uint8)
                for pts in poly_list:
                    pts_np = np.array(pts, dtype=np.float32)
                    pts_int = np.round(pts_np).astype(np.int32)
                    if pts_int.ndim == 2:
                        pts_int = pts_int.reshape((-1, 1, 2))
                    cv2.fillPoly(mask_full, [pts_int], 1)
                # Clamp bbox to image bounds
                gx1c, gy1c = max(0, gx1), max(0, gy1)
                gx2c, gy2c = min(w, gx2), min(h, gy2)
                if gx2c <= gx1c or gy2c <= gy1c:
                    continue
                # Optionally restrict to bbox region only
                mask_crop = np.zeros_like(mask_full)
                mask_crop[gy1c:gy2c, gx1c:gx2c] = mask_full[gy1c:gy2c, gx1c:gx2c]
                merged_pts = mask_to_polygon(mask_crop)
                if not merged_pts:
                    # Fallback to bbox polygon if SAM2 polygons disappeared
                    merged_pts = _bbox_to_polygon([gx1c, gy1c, gx2c, gy2c])
                # mask_to_polygon returns a list of [x, y] pairs; use it directly
                if merged_pts:
                    merged_polys.append({"label": label, "points": merged_pts})
        else:
            # Strip bbox_key before saving
            merged_polys = [{"label": p.get("label"), "points": p.get("points")} for p in polys]

        data = _load_labelme_json(json_path)
        data["imageWidth"] = int(w)
        data["imageHeight"] = int(h)
        if not data.get("imagePath"):
            data["imagePath"] = img_path.name
        data = _append_pseudomask_shapes(
            data, merged_polys,
            label_for_masks=label_for_masks,
            polygon_label_from_bbox=polygon_label_from_bbox,
            keep_rectangles=keep_rectangles,
        )
        out_img = output_dir / img_path.name
        out_json = output_dir / json_path.name
        cv2.imwrite(str(out_img), image_bgr)
        _save_labelme_json(data, out_json)
        print(f"Processed {img_path.name}: {len(data.get('shapes', []))} shapes ({len(polys)} SAM2 polygons).")

    print(f"Done. Outputs in {output_dir}")
