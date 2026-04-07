"""
Stage 2: Refine SAM2 masks.

Full logic: load refine_config (intensity, threshold/kmeans, morphology),
iterate LabelMe JSONs, refine each polygon mask, save refined masks and JSONs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from omegaconf import DictConfig


def _resolve_path(p, root: Path) -> Path:
    if isinstance(p, (str, Path)):
        p = Path(p)
    return p if p.is_absolute() else root / p


def _find_image_for_json(json_path: Path, exts: List[str]) -> Path | None:
    stem = json_path.stem
    for ext in exts:
        candidate = json_path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return None


def _build_intensity_image(image_bgr: np.ndarray, cv_cfg: Dict[str, Any]) -> np.ndarray:
    mode_cfg = cv_cfg.get("intensity", {})
    mode = str(mode_cfg.get("mode", "gray")).lower()
    if mode == "hsv_v":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        intensity = hsv[:, :, 2]
    else:
        intensity = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe_cfg = mode_cfg.get("clahe", {})
    if bool(clahe_cfg.get("enabled", False)):
        clip = float(clahe_cfg.get("clip_limit", 2.0))
        tile = int(clahe_cfg.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        intensity = clahe.apply(intensity)
    return intensity


def _threshold_dark_regions(intensity: np.ndarray, thr_cfg: Dict[str, Any]) -> np.ndarray:
    method = str(thr_cfg.get("method", "otsu")).lower()
    value = int(thr_cfg.get("value", 40))
    invert = bool(thr_cfg.get("invert", True))
    block_size = int(thr_cfg.get("block_size", 31))
    C = int(thr_cfg.get("C", 5))
    if method == "global":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, bin_mask = cv2.threshold(intensity, value, 255, thresh_type)
    elif method == "otsu":
        _, bin_mask = cv2.threshold(intensity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            bin_mask = cv2.bitwise_not(bin_mask)
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        bin_mask = cv2.adaptiveThreshold(intensity, 255, adaptive_method, cv2.THRESH_BINARY, block_size, C)
        if invert:
            bin_mask = cv2.bitwise_not(bin_mask)
    else:
        raise ValueError(f"Unsupported threshold method: {method}")
    return (bin_mask > 0).astype(np.uint8)


def _apply_morphology(mask: np.ndarray, op: str, kernel_size: int, iterations: int) -> np.ndarray:
    if op == "none":
        return mask
    k = max(1, int(kernel_size))
    kernel = np.ones((k, k), np.uint8)
    it = max(1, int(iterations))
    if op == "open":
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=it)
    elif op == "close":
        out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)
    elif op == "erode":
        out = cv2.erode(mask, kernel, iterations=it)
    elif op == "dilate":
        out = cv2.dilate(mask, kernel, iterations=it)
    else:
        raise ValueError(f"Unsupported morph operation: {op}")
    return out


def _mask_to_polygon(mask: np.ndarray) -> list:
    if mask is None or mask.size == 0 or mask.ndim != 2:
        return []
    mask_uint8 = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return []
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    return [[float(p[0][0]), float(p[0][1])] for p in contour]


def _refine_single_image(
    img_path: Path,
    json_path: Path,
    refine_cfg: Dict[str, Any],
) -> None:
    io_cfg = refine_cfg.get("io", {})
    cv_cfg = refine_cfg.get("cv", {})
    save_cfg = refine_cfg.get("save", {})
    mask_label = str(io_cfg.get("mask_label", "sam2_pseudomask"))

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"[WARN] Failed to read image {img_path}")
        return
    h, w = image_bgr.shape[:2]
    intensity = _build_intensity_image(image_bgr, cv_cfg)

    with open(json_path, "r") as f:
        data = json.load(f)
    shapes = data.get("shapes", [])

    refined_union = np.zeros((h, w), dtype=np.uint8)
    refined_polys: List[Dict[str, Any]] = []
    thr_cfg = cv_cfg.get("threshold", {})
    morph_cfg = cv_cfg.get("morph", {})
    morph_op = str(morph_cfg.get("op", "open")).lower()
    morph_ksize = int(morph_cfg.get("kernel", 3))
    morph_iter = int(morph_cfg.get("iterations", 1))
    refine_method = str(cv_cfg.get("refine", {}).get("method", "threshold")).lower()
    kmeans_cfg = cv_cfg.get("refine", {}).get("kmeans", {})
    kmeans_k = int(kmeans_cfg.get("k", 2))
    kmeans_max_iter = int(kmeans_cfg.get("max_iter", 30))
    kmeans_attempts = int(kmeans_cfg.get("attempts", 3))

    dark_mask = _threshold_dark_regions(intensity, thr_cfg) if refine_method == "threshold" else None

    rect_infos: List[Dict[str, Any]] = []
    for idx, sh in enumerate(shapes):
        if sh.get("shape_type") != "rectangle":
            continue
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, x2 = int(round(min(x1, x2))), int(round(max(x1, x2)))
        y1, y2 = int(round(min(y1, y2))), int(round(max(y1, y2)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rect_infos.append({"index": idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    rect_masks: Dict[int, np.ndarray] = {info["index"]: np.zeros((h, w), dtype=np.uint8) for info in rect_infos}

    for sh in shapes:
        if sh.get("shape_type") != "polygon":
            continue
        pts = sh.get("points", [])
        if not pts:
            continue
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        pts_arr = np.array(pts, dtype=np.float32)
        cv2.fillPoly(poly_mask, [np.round(pts_arr).astype(np.int32)], 1)

        if refine_method == "threshold":
            if dark_mask is None:
                dark_mask = _threshold_dark_regions(intensity, thr_cfg)
            refined = (poly_mask & dark_mask).astype(np.uint8)
        elif refine_method == "kmeans":
            ys, xs = np.where(poly_mask > 0)
            if ys.size == 0:
                continue
            vals = intensity[ys, xs].astype(np.float32).reshape(-1, 1)
            if vals.shape[0] < max(kmeans_k, 2) or kmeans_k < 2:
                refined = poly_mask.copy()
            else:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_max_iter, 1.0)
                try:
                    _, labels, centers = cv2.kmeans(
                        vals, kmeans_k, None, criteria, kmeans_attempts, cv2.KMEANS_PP_CENTERS
                    )
                    darkest_idx = int(np.argmin(centers))
                    selected = (labels.reshape(-1) == darkest_idx).astype(np.uint8)
                    refined = np.zeros((h, w), dtype=np.uint8)
                    refined[ys, xs] = selected
                except Exception:
                    refined = poly_mask.copy()
        else:
            raise ValueError(f"Unsupported refine method: {refine_method}")

        if morph_op != "none":
            refined = _apply_morphology(refined * 255, morph_op, morph_ksize, morph_iter)
            refined = (refined > 0).astype(np.uint8)
        if refined.sum() == 0:
            refined = poly_mask.copy()

        best_rect_index = None
        best_area = 0
        for rect in rect_infos:
            x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
            sub = poly_mask[y1:y2, x1:x2]
            if sub.size == 0:
                continue
            area = int(sub.sum())
            if area > best_area:
                best_area = area
                best_rect_index = rect["index"]

        if best_rect_index is None or best_area == 0:
            refined_union[refined > 0] = 1
            if bool(save_cfg.get("polygons_json", True)):
                pts_refined = _mask_to_polygon(refined)
                if pts_refined:
                    refined_polys.append({
                        "label": sh.get("label", mask_label),
                        "points": pts_refined,
                        "group_id": sh.get("group_id"),
                        "description": sh.get("description", ""),
                        "shape_type": "polygon",
                        "flags": sh.get("flags", {}),
                        "mask": None,
                    })
            continue

        rect_masks[best_rect_index][refined > 0] = 1

    for rect in rect_infos:
        idx = rect["index"]
        rect_mask = rect_masks[idx]
        x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
        sub = rect_mask[y1:y2, x1:x2]
        if sub.sum() == 0:
            sub = np.ones_like(sub, dtype=np.uint8)
        full = np.zeros_like(rect_mask, dtype=np.uint8)
        full[y1:y2, x1:x2] = sub
        refined_union[full > 0] = 1
        if bool(save_cfg.get("polygons_json", True)):
            pts_refined = _mask_to_polygon(full)
            if pts_refined:
                rect_shape = shapes[idx]
                use_bbox_label = bool(save_cfg.get("polygon_label_from_bbox", True))
                poly_label = rect_shape.get("label", mask_label) if use_bbox_label else mask_label
                refined_polys.append({
                    "label": poly_label,
                    "points": pts_refined,
                    "group_id": rect_shape.get("group_id"),
                    "description": rect_shape.get("description", ""),
                    "shape_type": "polygon",
                    "flags": rect_shape.get("flags", {}),
                    "mask": None,
                })

    base_out_root = Path(io_cfg.get("output_dir", "./outputs_sam2_refined"))
    method_suffix = refine_method or "default"
    out_root = base_out_root.parent / f"{base_out_root.name}_{method_suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    out_img = out_root / img_path.name
    cv2.imwrite(str(out_img), image_bgr)
    stem = json_path.stem

    if bool(save_cfg.get("masks_npy", True)):
        out_mask = out_root / f"{stem}_refined_mask.npy"
        np.save(out_mask, refined_union.astype(np.uint8))

    if bool(save_cfg.get("polygons_json", True)):
        overwrite_shapes = bool(save_cfg.get("overwrite_json_shapes", False))
        keep_rectangles = bool(save_cfg.get("keep_rectangles", True))
        if overwrite_shapes:
            kept = [
                sh for sh in shapes
                if sh.get("shape_type") != "polygon"
                and (keep_rectangles or sh.get("shape_type") != "rectangle")
            ]
            data["shapes"] = kept + refined_polys
        else:
            data["shapes"] = shapes + refined_polys
        data["imageWidth"] = int(w)
        data["imageHeight"] = int(h)
        if not data.get("imagePath"):
            data["imagePath"] = img_path.name
        out_json = out_root / json_path.name
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)

    if bool(save_cfg.get("visualizations", True)):
        vis = image_bgr.copy()
        if refined_union.any():
            mask_color = np.zeros_like(image_bgr)
            mask_color[refined_union > 0] = (0, 0, 255)
            vis = cv2.addWeighted(vis, 1.0, mask_color, 0.4, 0)
        for sh in shapes:
            if sh.get("shape_type") != "rectangle":
                continue
            pts = sh.get("points", [])
            if len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, x2 = int(round(min(x1, x2))), int(round(max(x1, x2)))
            y1, y2 = int(round(min(y1, y2))), int(round(max(y1, y2)))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out_vis = out_root / f"{stem}_refined_overlay.jpg"
        cv2.imwrite(str(out_vis), vis)

    print(f"Refined {json_path.name}: {int(refined_union.sum())} px, {len(refined_polys)} polygons.")


def run(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    root = _resolve_path(cfg.paths.root, Path.cwd())
    # Refinement config is inlined in configs/stage/refinement.yaml (io, cv, save)
    refine_cfg = OmegaConf.to_container(cfg.stage, resolve=True) or {}

    io_cfg = refine_cfg.setdefault("io", {})
    if cfg.stage.get("input_dir") is not None:
        io_cfg["input_dir"] = str(_resolve_path(cfg.stage.input_dir, root))
    if cfg.stage.get("output_dir") is not None:
        io_cfg["output_dir"] = str(_resolve_path(cfg.stage.output_dir, root))

    input_dir = Path(io_cfg.get("input_dir", "./outputs_sam2_fullsize"))
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    exts = io_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".JPG", ".PNG"])
    exts = [str(e) for e in exts]

    json_files = sorted(input_dir.glob("*.json"))
    single_json = cfg.stage.get("single_json")
    if single_json:
        candidate = input_dir / (single_json if single_json.endswith(".json") else f"{single_json}.json")
        json_files = [candidate] if candidate.exists() else []
    if not json_files:
        print(f"No JSON files in {input_dir}" + (f" matching {single_json}" if single_json else ""))
        return
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    for json_path in json_files:
        img_path = _find_image_for_json(json_path, exts)
        if img_path is None:
            print(f"[WARN] No image for {json_path.name}, skipping.")
            continue
        _refine_single_image(img_path, json_path, refine_cfg)
