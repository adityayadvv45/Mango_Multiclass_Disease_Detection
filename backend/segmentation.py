"""
Mango Leaf Segmentation & Background / Non-Leaf Rejection Engine.
Accurately segments mango leaf blades and rigorously excludes:
- White paper / notebook pages
- Human hands / fingers / skin
- Table / wooden desk surfaces
- Soil / floor / grass background
- Border shadows and non-foliar artifacts
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
        np.array([255, 175, 127], dtype=np.uint8)
    )
    
    # HSV skin tone range (hue in reddish/orange range, moderate saturation)
    skin_hsv1 = cv2.inRange(
        hsv, 
        np.array([0, 40, 60], dtype=np.uint8), 
        np.array([25, 200, 255], dtype=np.uint8)
    )
    skin_hsv2 = cv2.inRange(
        hsv, 
        np.array([170, 40, 60], dtype=np.uint8), 
        np.array([180, 200, 255], dtype=np.uint8)
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
    Detects white paper, notebook sheets, or high-brightness neutral backgrounds.
    Returns binary mask (255 for paper/bright background, 0 otherwise).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Paper typically has low saturation and high value
    paper_mask1 = cv2.inRange(
        hsv,
        np.array([0, 0, 175], dtype=np.uint8),
        np.array([180, 45, 255], dtype=np.uint8)
    )
    
    # Neutral grey-white where R ~ G ~ B > 165
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    rg_diff = np.abs(r - g)
    gb_diff = np.abs(g - b)
    rb_diff = np.abs(r - b)
    mean_val = (r + g + b) / 3.0
    
    neutral_white = (rg_diff < 30) & (gb_diff < 30) & (rb_diff < 30) & (mean_val > 165)
    paper_mask2 = (neutral_white.astype(np.uint8)) * 255
    
    paper_mask = cv2.bitwise_or(paper_mask1, paper_mask2)
    return paper_mask

def detect_soil_table_background(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects brown soil, wood table, or dark non-foliar earth backgrounds.
    Returns binary mask (255 for soil/table, 0 otherwise).
    """
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    # Excess green index: 2G - R - B. Plants have positive ExG; soil/wood have negative ExG.
    exg = 2.0 * g - r - b
    
    # Soil/wood has R > G + 10 and ExG < -15
    soil_condition = (r > (g + 10)) & (exg < -15) & (r > b)
    soil_mask = (soil_condition.astype(np.uint8)) * 255
    return soil_mask

def segment_mango_leaf(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool, np.ndarray, Dict[str, Any]]:
    """
    Performs multi-cue leaf segmentation and background rejection.
    
    Returns:
    - leaf_mask: np.ndarray (uint8, 0 or 255)
    - leaf_detected: bool (True if a valid mango leaf is present)
    - masked_image_bgr: np.ndarray (leaf isolated, background zeroed out)
    - segmentation_metadata: dict with diagnostic info
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w
    
    # Compute rejection masks
    skin_mask = detect_skin_mask(image_bgr)
    paper_mask = detect_paper_background(image_bgr)
    soil_mask = detect_soil_table_background(image_bgr)
    
    rejection_mask = cv2.bitwise_or(skin_mask, paper_mask)
    rejection_mask = cv2.bitwise_or(rejection_mask, soil_mask)
    
    # Color spaces
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    
    # 1. Foliage Green & Olive Mask (covers healthy and mild chlorotic tissue)
    # Hue in OpenCV HSV is [0..180]. Green is ~ 25 to 95.
    foliage_green = cv2.inRange(
        hsv,
        np.array([20, 20, 20], dtype=np.uint8),
        np.array([98, 255, 255], dtype=np.uint8)
    )
    
    # 2. ExG plant threshold
    exg_mask = (exg > -5).astype(np.uint8) * 255
    
    # 3. Diseased Necrotic / Halo / Lesion Tissue on Leaf
    # Yellow halos, anthracnose brown necrosis, die-back brown leaf margins
    necrosis_hsv = cv2.inRange(
        hsv,
        np.array([5, 30, 20], dtype=np.uint8),
        np.array([25, 255, 240], dtype=np.uint8)
    )
    
    # 4. Powdery mildew / pale spots
    powdery_hsv = cv2.inRange(
        hsv,
        np.array([15, 10, 100], dtype=np.uint8),
        np.array([105, 100, 255], dtype=np.uint8)
    )
    
    # 5. Sooty Mold (dark fungal coat on leaf surface)
    sooty_hsv = cv2.inRange(
        hsv,
        np.array([0, 0, 10], dtype=np.uint8),
        np.array([180, 255, 70], dtype=np.uint8)
    )
    
    # Primary candidate leaf pixels
    candidate_leaf = cv2.bitwise_or(foliage_green, necrosis_hsv)
    candidate_leaf = cv2.bitwise_or(candidate_leaf, exg_mask)
    candidate_leaf = cv2.bitwise_or(candidate_leaf, powdery_hsv)
    
    # If the background is bright/paper (as in MangoLeafBD dataset),
    # Otsu thresholding on the grayscale/green-channel also separates the leaf cleanly
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # If image has white background, otsu_dark isolates the darker leaf
    corner_pixels = np.vstack([
        image_bgr[:15, :15].reshape(-1, 3),
        image_bgr[:15, -15:].reshape(-1, 3),
        image_bgr[-15:, :15].reshape(-1, 3),
        image_bgr[-15:, -15:].reshape(-1, 3)
    ])
    corner_mean_brightness = float(np.mean(corner_pixels))
    
    if corner_mean_brightness > 140:
        # White/light paper background: Otsu inverse + foliage color gives robust leaf silhouette
        candidate_leaf = cv2.bitwise_or(candidate_leaf, otsu_dark)
        # On white paper, reject paper and skin only
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(paper_mask))
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(skin_mask))
    else:
        # Outdoor/complex background: subtract paper, skin, and soil rejections
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(paper_mask))
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(skin_mask))
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(soil_mask))
    
    # Morphological cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_leaf = cv2.morphologyEx(candidate_leaf, cv2.MORPH_OPEN, kernel_small, iterations=2)
    candidate_leaf = cv2.morphologyEx(candidate_leaf, cv2.MORPH_CLOSE, kernel_small, iterations=3)
    
    # Find connected components / contours
    contours, _ = cv2.findContours(candidate_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), False, image_bgr, {"reason": "No contours found"}
        
    # Filter contours by minimum area (at least 1.2% of the total image)
    min_area = 0.012 * img_area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        # Check if the largest contour is reasonable
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.005 * img_area:
            return np.zeros((h, w), dtype=np.uint8), False, image_bgr, {"reason": "Contour area below threshold"}
        valid_contours = [largest]
        
    # Build the preliminary leaf mask from the valid contours
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    for c in valid_contours:
        cv2.drawContours(leaf_mask, [c], -1, 255, thickness=cv2.FILLED)
        
    # Fill internal holes (e.g. inner dark lesions or sooty mold patches)
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=4)
    
    # Re-apply strict hand/skin rejection to prevent hand intrusion
    leaf_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(skin_mask))
    leaf_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(paper_mask))
    
    # Final check on leaf mask size
    leaf_pixel_count = int(cv2.countNonZero(leaf_mask))
    leaf_area_ratio = leaf_pixel_count / float(img_area)
    
    if leaf_area_ratio < 0.007:
        return np.zeros((h, w), dtype=np.uint8), False, image_bgr, {
            "reason": "Leaf area ratio too low",
            "leaf_area_ratio": leaf_area_ratio
        }
        
    # Apply mask to image (zero out background)
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
    min_overlap_ratio: float = 0.40
) -> bool:
    """
    Checks if a detected bounding box [ymin, xmin, ymax, xmax] is genuinely on the leaf.
    Strictly rejects any detection on paper, hand, table, soil, or background.
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
    
    # Check center point
    cy = (ymin + ymax) // 2
    cx = (xmin + xmax) // 2
    center_is_leaf = (leaf_mask[cy, cx] > 0)
    
    return (overlap_ratio >= min_overlap_ratio) and center_is_leaf
