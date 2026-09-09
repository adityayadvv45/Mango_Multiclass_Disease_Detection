import os
import glob
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class YOLOv8Detector:
    """
    Object Detection and Spatial Localization Engine for Mango Leaf Pathologies.
    Locates diseased regions, computes bounding box coordinates [x1, y1, x2, y2],
    and yields cropped regions for downstream CNN classification.
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(base_dir, "models")
        else:
            self.models_dir = models_dir

        self.yolo_model = None
        self.model_version = "YOLOv8-Localization-Engine"
        self._load_yolo_model()

    def _load_yolo_model(self):
        """Discovers and initializes YOLOv8 object detection weights if present."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return

        pt_files = glob.glob(os.path.join(self.models_dir, "*.pt"))
        if pt_files and HAS_ULTRALYTICS:
            try:
                self.yolo_model = YOLO(pt_files[0])
                self.model_version = f"YOLOv8-{os.path.basename(pt_files[0])}"
                print(f"[YOLO Detector] Loaded YOLOv8 detection weights: {pt_files[0]}")
            except Exception as e:
                print(f"[YOLO Detector] Error loading YOLO model: {e}")
        elif pt_files and not HAS_ULTRALYTICS:
            print("[YOLO Detector] Note: '.pt' weights found but 'ultralytics' package not installed.")

    def detect_regions(self, np_rgb):
        """
        Detects pathological regions and outputs bounding boxes.
        Returns: list of dicts with keys:
          - 'bbox': [x1, y1, x2, y2]
          - 'relative_bbox': [rx1, ry1, rx2, ry2]
          - 'yolo_confidence': float (percentage)
          - 'area': int
        """
        h_img, w_img = np_rgb.shape[:2]

        if self.yolo_model is not None:
            return self._detect_with_yolo(np_rgb, w_img, h_img)
        else:
            return self._detect_with_cv_heuristics(np_rgb, w_img, h_img)

    def _detect_with_yolo(self, np_rgb, w_img, h_img):
        """Runs YOLOv8 bounding box regression and extracts coordinates."""
        detections = []
        try:
            results = self.yolo_model.predict(np_rgb, conf=0.25, verbose=False)
            for res in results:
                boxes = res.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy()) * 100.0
                        x1 = max(0, min(w_img - 1, int(xyxy[0])))
                        y1 = max(0, min(h_img - 1, int(xyxy[1])))
                        x2 = max(x1 + 1, min(w_img, int(xyxy[2])))
                        y2 = max(y1 + 1, min(h_img, int(xyxy[3])))
                        area = (x2 - x1) * (y2 - y1)

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "relative_bbox": [
                                round(x1 / w_img, 4),
                                round(y1 / h_img, 4),
                                round(x2 / w_img, 4),
                                round(y2 / h_img, 4)
                            ],
                            "yolo_confidence": round(conf, 2),
                            "area": int(area)
                        })
        except Exception as e:
            print(f"[YOLO Detector] Detection exception: {e}, falling back to CV localization.")
            return self._detect_with_cv_heuristics(np_rgb, w_img, h_img)

        return self.apply_nms(detections)

    def _detect_with_cv_heuristics(self, np_rgb, w_img, h_img):
        """
        Computer vision morphological localization engine.
        Segments diseased foliage patches using adaptive color-space decomposition,
        specular glare filtering, and contour bounding boxes.
        """
        total_pixels = h_img * w_img
        r = np_rgb[:, :, 0].astype(np.float32)
        g = np_rgb[:, :, 1].astype(np.float32)
        b = np_rgb[:, :, 2].astype(np.float32)

        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        # Compute HSV representation for robust illumination invariance
        if HAS_CV2:
            img_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h_channel = img_hsv[:, :, 0].astype(np.float32)
            s_channel = img_hsv[:, :, 1].astype(np.float32)
            v_channel = img_hsv[:, :, 2].astype(np.float32)
        else:
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            delta = max_c - min_c + 1e-5
            s_channel = (delta / (max_c + 1e-5)) * 255.0
            v_channel = max_c
            h_channel = np.zeros_like(r)

        # -----------------------------------------------------------------
        # 1. SPECULAR GLARE, SKY BACKGROUND & AMBIENT SUNLIGHT REJECTION MASK
        # Sky/outdoor glare has cyan/blue/sky hue (H 95-155, S 12-70) or specular overexposure.
        # This prevents outdoor sky/sunlight from being falsely marked as disease.
        # -----------------------------------------------------------------
        sky_glare = (
            ((h_channel >= 95) & (h_channel <= 155) & (s_channel >= 12) & (s_channel < 70) & (brightness > 130)) |
            ((v_channel > 248) & (s_channel < 10) & (brightness > 242))
        )

        # -----------------------------------------------------------------
        # 2. VEGETATION & LEAF BLADE RECOGNITION
        # -----------------------------------------------------------------
        is_leaf_tissue = (
            ((h_channel >= 15) & (h_channel <= 90) & (s_channel >= 12)) |
            ((r > b + 6) & (brightness > 30) & (brightness < 210)) |
            ((brightness < 120) & (brightness > 15))
        ) & ~sky_glare

        # -----------------------------------------------------------------
        # 3. PATHOLOGY-SPECIFIC LESION SEGMENTATION
        # -----------------------------------------------------------------
        # A) Necrotic & Desiccated Lesions (Bacterial Canker, Anthracnose, Die Back)
        # Warm brown/tan necrotic tissue: R is dominant over B and comparable/higher than G
        brown_necrosis = (
            (r > b + 6) &
            (r > g - 12) &
            (brightness > 30) &
            (brightness < 215) &
            (s_channel > 15) &
            ~sky_glare
        )

        # Dark necrotic spots / canker cores (must be inside leaf blade with necrotic chromaticity):
        dark_spots = (
            (((brightness < 75) & (brightness > 5) & (r >= b) & (r > 15)) |
             ((brightness < 120) & (brightness > 15) & (r > b + 6) & (r > g - 10))) &
            is_leaf_tissue &
            ~sky_glare
        )

        # Chlorotic halos & Yellow margins:
        yellow_halos = (
            (r > 120) &
            (g > 100) &
            (b < 110) &
            (r > b + 15) &
            (brightness > 35) &
            ~sky_glare
        )

        # B) Powdery Mildew (Superficial white/grey fungal mycelium on leaf lamina)
        powdery_patches = (
            (brightness > 165) &
            (brightness <= 250) &
            (np.abs(r - g) < 22) &
            (np.abs(g - b) < 22) &
            (s_channel < 45) &
            (g > 25) &
            ~sky_glare
        )

        # C) Sooty Mold (Dark fungal coat masking green blade)
        sooty_patches = (
            (brightness < 50) &
            (brightness > 8) &
            (s_channel < 60) &
            (is_leaf_tissue | (g > 15)) &
            ~sky_glare
        )

        # D) Combined Pathological Mask
        combined_lesion_mask = (
            brown_necrosis |
            dark_spots |
            yellow_halos |
            powdery_patches |
            sooty_patches
        )

        min_area = max(200, int(total_pixels * 0.0015))
        max_area = int(total_pixels * 0.45)
        detections = []

        if HAS_CV2:
            mask_u8 = (combined_lesion_mask.astype(np.uint8) * 255)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y_box, w_box, h_box = cv2.boundingRect(cnt)
                    # Exclude whole-frame or degenerate background wrappers
                    if (w_box >= w_img * 0.80 and h_box >= h_img * 0.80) or (w_box * h_box > total_pixels * 0.45):
                        continue

                    pad_x = int(w_box * 0.08)
                    pad_y = int(h_box * 0.08)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y_box - pad_y)
                    x2 = min(w_img, x + w_box + pad_x)
                    y2 = min(h_img, y_box + h_box + pad_y)
                    conf = round(min(98.5, max(75.0, 80.0 + (area / total_pixels) * 50.0)), 2)

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "relative_bbox": [
                            round(x1 / w_img, 4),
                            round(y1 / h_img, 4),
                            round(x2 / w_img, 4),
                            round(y2 / h_img, 4)
                        ],
                        "yolo_confidence": conf,
                        "area": int(area)
                    })
        else:
            boxes = self._extract_connected_boxes(combined_lesion_mask, min_area, max_area)
            for bx1, by1, bx2, by2, area in boxes:
                conf = round(min(98.5, max(75.0, 80.0 + (area / total_pixels) * 50.0)), 2)
                detections.append({
                    "bbox": [bx1, by1, bx2, by2],
                    "relative_bbox": [
                        round(bx1 / w_img, 4),
                        round(by1 / h_img, 4),
                        round(bx2 / w_img, 4),
                        round(by2 / h_img, 4)
                    ],
                    "yolo_confidence": conf,
                    "area": int(area)
                })

        return self.apply_nms(detections)

    def _extract_connected_boxes(self, mask, min_area=250, max_area=None):
        """Pure NumPy/Python flood-fill bounding box finder."""
        h, w = mask.shape
        if max_area is None:
            max_area = int(h * w * 0.50)

        visited = np.zeros_like(mask, dtype=bool)
        boxes = []

        step = 3
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x] and not visited[y, x]:
                    stack = [(y, x)]
                    visited[y, x] = True
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    count = 0

                    while stack:
                        cy, cx = stack.pop()
                        count += 1
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)

                        for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))

                    approx_area = count * step * step
                    if min_area <= approx_area <= max_area:
                        pad_x = int((max_x - min_x + 1) * 0.10)
                        pad_y = int((max_y - min_y + 1) * 0.10)
                        bx1 = max(0, min_x - pad_x)
                        by1 = max(0, min_y - pad_y)
                        bx2 = min(w, max_x + pad_x)
                        by2 = min(h, max_y + pad_y)
                        boxes.append((bx1, by1, bx2, by2, approx_area))

        return boxes

    def apply_nms(self, detections, iou_threshold=0.35):
        """Non-Maximum Suppression (NMS) merging overlapping candidate boxes."""
        if not detections:
            return []

        dets = sorted(detections, key=lambda d: d["yolo_confidence"], reverse=True)
        keep = []

        while dets:
            current = dets.pop(0)
            keep.append(current)

            x1_a, y1_a, x2_a, y2_a = current["bbox"]
            area_a = max(1, (x2_a - x1_a) * (y2_a - y1_a))

            remaining = []
            for other in dets:
                x1_b, y1_b, x2_b, y2_b = other["bbox"]
                area_b = max(1, (x2_b - x1_b) * (y2_b - y1_b))

                inter_x1 = max(x1_a, x1_b)
                inter_y1 = max(y1_a, y1_b)
                inter_x2 = min(x2_a, x2_b)
                inter_y2 = min(y2_a, y2_b)

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                union = area_a + area_b - inter_area
                iou = inter_area / max(1, union)

                if iou < iou_threshold:
                    remaining.append(other)
            dets = remaining

        return keep[:6]
