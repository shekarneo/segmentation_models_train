import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


def apply_clahe_bgr(
    image_bgr: np.ndarray,
    clip_limit: float,
    tile_grid: Tuple[int, int],
) -> np.ndarray:
    """CLAHE on L channel in LAB (same as prepare stage)."""
    img_8u = np.clip(image_bgr, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_8u, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_object_crop_bbox(image_bgr: np.ndarray, padding: int = 0) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of the main white object in the image using HSV color filtering
    and connected components (adapted from foreground mask logic).
    Returns: (x_min, y_min, x_max, y_max)
    """
    h, w = image_bgr.shape[:2]
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    
    # Background is green/blue (hue 60-180)
    # Foreground is white/silver: low saturation (< 50) OR high brightness (> 200) with low saturation
    is_background = (hue >= 60) & (hue <= 180)
    is_foreground_color = (~is_background) & ((saturation < 50) | ((value > 200) & (saturation < 80)))
    
    binary = is_foreground_color.astype(np.uint8) * 255
    
    # Morphological operations to clean up mask
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:
        # Fallback to no crop
        return 0, 0, w, h
        
    # Find largest component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    largest_area = areas[largest_label - 1]
    
    # Require it to be at least 5% of the image to be considered the main object
    if largest_area < (h * w * 0.05):
        # Fall back to simple brightness if color filter misses the object
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        threshold = mean_brightness * 1.2
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            largest_area = areas[largest_label - 1]
        else:
            return 0, 0, w, h
    
    foreground_mask = (labels == largest_label).astype(np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    foreground_mask = cv2.dilate(foreground_mask, dilate_kernel, iterations=1)
    
    ys, xs = np.where(foreground_mask > 0)
    if len(ys) == 0:
        return 0, 0, w, h
        
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    return x_min, y_min, x_max, y_max

def shift_labelme_json(
    json_path: Path,
    cx1: int,
    cy1: int,
    out_path: Path,
    *,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> None:
    """
    Read a LabelMe JSON, shift all shape coordinates by (-cx1, -cy1), and save.

    After a crop, pass image_width/image_height of the cropped image so imageWidth/imageHeight
    in JSON match the coordinate system; otherwise rasterizers may rescale points incorrectly.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        for pt in shape.get("points", []):
            pt[0] -= cx1
            pt[1] -= cy1
    if image_width is not None and image_width > 0:
        data["imageWidth"] = int(image_width)
    if image_height is not None and image_height > 0:
        data["imageHeight"] = int(image_height)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
