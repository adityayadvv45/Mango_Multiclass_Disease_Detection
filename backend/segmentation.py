"""
Mango Leaf Segmentation & Background / Non-Leaf Rejection Engine.
Accurately segments mango leaf blades and rigorously excludes:
- White paper / notebook pages
- Human hands / fingers / skin
- Table / wooden desk surfaces
- Soil / floor / background artifacts
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

def detect_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects human skin tones (hands, fingers) using combined YCrCb and HSV color spaces.
    Returns binary mask (255 for skin, 0 otherwise).
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # YCrCb skin tone range
    skin_ycrcb = cv2.inRange(
        ycrcb, 
        np.array([0, 133, 77], dtype=np.uint8), 
        np.array([255, 173, 127], dtype=np.uint8)
    )
    
    # HSV skin tone range
    skin_hsv1 = cv2.inRange(
        hsv, 
        np.array([0, 40, 50], dtype=np.uint8), 
        np.array([25, 220, 255], dtype=np.uint8)
    )
    skin_hsv2 = cv2.inRange(
        hsv, 
        np.array([170, 40, 50], dtype=np.uint8), 
        np.array([180, 220, 255], dtype=np.uint8)
    )
    skin_hsv = cv2.bitwise_or(skin_hsv1, skin_hsv2)
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    
    # Clean up small noise in skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    return skin_mask

def detect_paper_background(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects white paper sheets or high-brightness neutral backgrounds.
    Returns binary mask (255 for paper/bright background, 0 otherwise).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    
    # Paper has extremely low saturation and high brightness
    paper_mask1 = (hsv[:, :, 1] < 18) & (hsv[:, :, 2] > 195)
    paper_bright = (r > 225) & (g > 225) & (b > 225)
    paper_bluish = (b > r + 20) & (b > g + 15) & (hsv[:, :, 2] > 180)
    
    paper_mask = (paper_mask1 | paper_bright | paper_bluish).astype(np.uint8) * 255
    return paper_mask

def detect_soil_table_background(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects brown soil, wood table, or dark non-foliar earth backgrounds.
    Returns binary mask (255 for soil/table, 0 otherwise).
    """
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    
    # Soil/wood has R > G + 18 and ExG < -25
    soil_condition = (r > (g + 18)) & (exg < -25) & (r > b + 15)
    soil_mask = (soil_condition.astype(np.uint8)) * 255
    return soil_mask

def segment_mango_leaf(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool, np.ndarray, Dict[str, Any]]:
    """
    Performs robust multi-cue leaf blade segmentation and background rejection.
    
    Returns:
    - leaf_mask: np.ndarray (uint8, 0 or 255)
    - leaf_detected: bool (True if a valid mango leaf is present)
    - masked_image_bgr: np.ndarray (leaf isolated, background zeroed out)
    - segmentation_metadata: dict with diagnostic info
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w
    
    # 1. Color and vegetation cues
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    
    # Rejection masks
    skin_mask = detect_skin_mask(image_bgr)
    paper_mask = detect_paper_background(image_bgr)
    soil_mask = detect_soil_table_background(image_bgr)
    
    # Foliar color ranges:
    # 1. Healthy / olive / green vegetation
    is_green = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 22)
    is_exg = (exg > -12) & (hsv[:, :, 1] >= 18)
    # 2. Necrotic / chlorotic / brown foliar tissue
    is_necrotic = (hsv[:, :, 0] >= 6) & (hsv[:, :, 0] <= 28) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] > 20) & (hsv[:, :, 2] < 210)
    # 3. Dark fungal soot / anthracnose spots on leaf
    is_dark = (hsv[:, :, 2] < 100) & (hsv[:, :, 1] > 15) & (g >= b - 12)
    # 4. Pale powdery mildew on leaf blade
    is_powdery = (hsv[:, :, 2] > 140) & (hsv[:, :, 1] < 60) & (g > 60) & (r > 60) & (b > 60)
    
    candidate = (is_green | is_exg | is_necrotic | is_dark | is_powdery) & (paper_mask == 0) & (skin_mask == 0) & (soil_mask == 0)
    candidate_mask = candidate.astype(np.uint8) * 255
    
    # Morphological connecting
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback using Otsu on inverted saturation + luminance contrast
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_cand = cv2.bitwise_and(otsu, cv2.bitwise_not(paper_mask))
        contours, _ = cv2.findContours(otsu_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), False, image_bgr, {"reason": "No leaf contour found"}
        
    # Get largest contour corresponding to the main mango leaf blade
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 0.005 * img_area:
        return np.zeros((h, w), dtype=np.uint8), False, image_bgr, {"reason": "Contour area below threshold"}
        
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(leaf_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    
    # Fill internal holes (e.g. inner lesions or veins) to form a solid leaf blade mask
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k_fill, iterations=3)
    
    # Exclude skin mask borders if any
    leaf_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(skin_mask))
    
    leaf_pixel_count = int(cv2.countNonZero(leaf_mask))
    leaf_area_ratio = leaf_pixel_count / float(img_area)
    
    if leaf_area_ratio < 0.005:
        return np.zeros((h, w), dtype=np.uint8), False, image_bgr, {
            "reason": "Leaf area ratio too low",
            "leaf_area_ratio": leaf_area_ratio
        }
        
    masked_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)
    
    metadata = {
        "leaf_detected": True,
        "leaf_area_ratio": float(leaf_area_ratio),
        "leaf_pixel_count": leaf_pixel_count,
        "skin_pixels_rejected": int(cv2.countNonZero(skin_mask)),
        "paper_pixels_rejected": int(cv2.countNonZero(paper_mask)),
        "soil_pixels_rejected": int(cv2.countNonZero(soil_mask))
    }
    
    return leaf_mask, True, masked_bgr, metadata

def is_bbox_inside_leaf(
    bbox: Tuple[int, int, int, int], 
    leaf_mask: np.ndarray, 
    min_overlap_ratio: float = 0.15
) -> bool:
    """
    Checks if a detected bounding box [ymin, xmin, ymax, xmax] overlaps genuinely with the leaf.
    Supports slender, curved, or diagonally oriented leaves.
    """
    ymin, xmin, ymax, xmax = bbox
    h, w = leaf_mask.shape[:2]
    
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h))
    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w))
    
    box_w = xmax - xmin
    box_h = ymax - ymin
    box_area = box_w * box_h
    
    if box_area <= 0:
        return False
        
    box_region = leaf_mask[ymin:ymax, xmin:xmax]
    leaf_overlap_pixels = cv2.countNonZero(box_region)
    overlap_ratio = leaf_overlap_pixels / float(box_area)
    
    return (overlap_ratio >= min_overlap_ratio) and (leaf_overlap_pixels >= 30)

