import io
import time
import os
import numpy as np
from PIL import Image

try:
    from detector import YOLOv8Detector
    from classifier import (
        EfficientNetMangoClassifier,
        DISEASE_CLASSES,
        CLASS_MAP,
        CLASS_NAMES,
        NAME_TO_META
    )
except ImportError:
    from backend.detector import YOLOv8Detector
    from backend.classifier import (
        EfficientNetMangoClassifier,
        DISEASE_CLASSES,
        CLASS_MAP,
        CLASS_NAMES,
        NAME_TO_META
    )


class MangoLeafInferenceEngine:
    """
    Two-Stage Multi-Region Deep Learning Inference Pipeline:
    1. YOLOv8 Object Detection -> Identifies all candidate disease bounding boxes [x1, y1, x2, y2].
    2. Region Cropping -> Crops EACH bounding box independently from the original high-resolution image.
    3. EfficientNet-B0 CNN -> Executes independent neural classification on EACH crop patch.
    4. Multi-Disease Aggregation -> Returns every localized region with its own CNN disease label & confidence.
    """
    def __init__(self, models_dir=None, cnn_weights_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if models_dir is None:
            models_dir = os.path.join(base_dir, "models")
        if cnn_weights_path is None:
            cnn_weights_path = os.path.join(base_dir, "models", "mango_cnn_efficientnet.pth")

        self.models_dir = models_dir
        self.cnn_weights_path = cnn_weights_path

        # Stage 1: Spatial Object Detector
        self.detector = YOLOv8Detector(models_dir=self.models_dir)

        # Stage 2: Independent Crop CNN Classifier
        self.classifier = EfficientNetMangoClassifier(weights_path=self.cnn_weights_path)

        self.model_version = f"{self.detector.model_version} + {self.classifier.model_name}"

    def preprocess_image(self, file_bytes):
        """Decode and validate image bytes into RGB numpy array."""
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            np_rgb = np.array(pil_img)
            return np_rgb, pil_img.size
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {str(e)}")

    def predict(self, file_bytes):
        """
        Executes end-to-end multi-region inference:
          Image -> YOLOv8 Detection -> Crop EACH Region -> CNN Classifies EACH Crop -> Combined Result
        """
        start_time = time.time()
        np_rgb, (w_orig, h_orig) = self.preprocess_image(file_bytes)

        # ---------------------------------------------------------------------
        # STAGE 1: Spatial Object Detection & Bounding Box Localization (YOLO)
        # ---------------------------------------------------------------------
        raw_detections = self.detector.detect_regions(np_rgb)
        num_regions = len(raw_detections)
        print(f"[Inference] Stage 1 (YOLOv8): Located {num_regions} candidate pathological region(s).")

        # ---------------------------------------------------------------------
        # STAGE 2: Crop EACH Detected Bounding Box & Classify Independently (CNN)
        # ---------------------------------------------------------------------
        enriched_detections = []
        for idx, det in enumerate(raw_detections):
            x1, y1, x2, y2 = det["bbox"]

            # Boundary-safe padding
            pad = 2
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(w_orig, int(x2) + pad)
            cy2 = min(h_orig, int(y2) + pad)

            # Crop the exact localized patch from original image
            crop_patch = np_rgb[cy1:cy2, cx1:cx2]
            if crop_patch.shape[0] < 4 or crop_patch.shape[1] < 4:
                crop_patch = np_rgb  # Fallback if degenerate

            # Run CNN forward pass specifically on THIS crop
            cnn_result = self.classifier.classify_crop(crop_patch)

            crop_disease = cnn_result["disease"]
            crop_cnn_conf = cnn_result["cnn_confidence"]
            yolo_conf = det["yolo_confidence"]

            print(f"[Inference] Stage 2 (CNN): Crop {idx + 1}/{num_regions} at bbox [{x1}, {y1}, {x2}, {y2}] -> '{crop_disease}' (CNN: {crop_cnn_conf}%, YOLO: {yolo_conf}%)")

            enriched_detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "relative_bbox": det["relative_bbox"],
                "area": int(det["area"]),
                "disease": crop_disease,
                "disease_id": cnn_result["disease_id"],
                "scientific_name": cnn_result["scientific_name"],
                "category": cnn_result["category"],
                "risk": cnn_result["risk"],
                "confidence": crop_cnn_conf,
                "cnn_confidence": crop_cnn_conf,
                "yolo_confidence": yolo_conf,
                "distribution": cnn_result["distribution"]
            })

        # ---------------------------------------------------------------------
        # STAGE 3: Multi-Disease Resolution & Diagnostic Payload Formation
        # ---------------------------------------------------------------------
        # Filter out non-pathological (healthy) candidate crops
        pathological_detections = [d for d in enriched_detections if d["disease_id"] != "healthy"]

        if not pathological_detections:
            # 0 localized lesions found or all candidates confirmed healthy by CNN -> Evaluate whole leaf
            whole_leaf_cnn = self.classifier.classify_crop(np_rgb)
            is_healthy = (whole_leaf_cnn["disease_id"] == "healthy")
            is_multiple = False
            primary_disease = whole_leaf_cnn["disease"]
            primary_id = whole_leaf_cnn["disease_id"]
            primary_conf = whole_leaf_cnn["cnn_confidence"]
            risk = whole_leaf_cnn["risk"]

            if is_healthy:
                status = "Healthy Specimen"
                summary = "The leaf exhibits healthy, uniform pigmentation and laminar structure with no active pathological lesions detected."
                predicted_diseases = [{
                    "name": "Healthy",
                    "disease": "Healthy",
                    "confidence": primary_conf,
                    "disease_id": "healthy",
                    "cnn_confidence": primary_conf,
                    "yolo_confidence": 0.0
                }]
            else:
                status = "Disease Detected"
                summary = f"Diffuse foliage pathology detected: The CNN model predicts {primary_disease} with {primary_conf}% estimated confidence across the leaf blade."
                predicted_diseases = [{
                    "name": primary_disease,
                    "disease": primary_disease,
                    "confidence": primary_conf,
                    "disease_id": primary_id,
                    "cnn_confidence": primary_conf,
                    "yolo_confidence": 0.0
                }]

            scores = whole_leaf_cnn["distribution"]
            final_detections = []
        else:
            # Aggregate all unique disease classes identified across individual crops
            distinct_diseases = {}
            for d in pathological_detections:
                d_name = d["disease"]
                if d_name not in distinct_diseases or d["cnn_confidence"] > distinct_diseases[d_name]["confidence"]:
                    distinct_diseases[d_name] = {
                        "name": d_name,
                        "disease": d_name,
                        "disease_id": d["disease_id"],
                        "confidence": d["cnn_confidence"],
                        "cnn_confidence": d["cnn_confidence"],
                        "yolo_confidence": d["yolo_confidence"],
                        "risk": d["risk"],
                        "bbox": d["bbox"]
                    }

            predicted_diseases = sorted(distinct_diseases.values(), key=lambda x: x["confidence"], reverse=True)
            is_multiple = len(predicted_diseases) > 1

            if is_multiple:
                primary_disease = "Multiple Diseases Detected"
                primary_id = "multiple-diseases"
                primary_conf = round(float(np.mean([d["confidence"] for d in predicted_diseases])), 1)
                status = "Multiple Diseases Detected"
                dis_summary_list = ", ".join([f"{d['name']} ({d['confidence']}%)" for d in predicted_diseases])
                summary = f"Multiple distinct foliar pathologies detected across the leaf: {dis_summary_list}. Each affected region was independently localized by YOLOv8 and classified by EfficientNet-B0."
                risk = "High" if any(d.get("risk") == "High" for d in predicted_diseases) else "Moderate"
                is_healthy = False
            else:
                top_d = predicted_diseases[0]
                primary_disease = top_d["name"]
                primary_id = top_d["disease_id"]
                primary_conf = top_d["confidence"]
                is_healthy = (primary_id == "healthy")
                status = "Healthy Specimen" if is_healthy else "Disease Detected"
                summary = f"YOLOv8 localized {len(pathological_detections)} lesion region(s) and EfficientNet-B0 independently classified {primary_disease} with {primary_conf}% confidence."
                disease_meta = CLASS_MAP.get(primary_id, CLASS_MAP.get("anthracnose", {}))
                risk = disease_meta.get("risk", "Moderate")

            scores = pathological_detections[0]["distribution"]
            final_detections = pathological_detections

        all_predictions = sorted(
            [{"name": name, "confidence": conf, "isTop": (name == primary_disease)} for name, conf in scores.items()],
            key=lambda x: x["confidence"],
            reverse=True
        )

        exec_time = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "disease": primary_disease,
            "disease_id": primary_id,
            "confidence": primary_conf,
            "status": status,
            "risk": risk,
            "is_healthy": is_healthy,
            "is_multiple_diseases": is_multiple,
            "predicted_diseases": predicted_diseases,
            "all_predictions": all_predictions,
            "predictions": all_predictions,
            "detections": final_detections,
            "summary": summary,
            "execution_time_ms": exec_time,
            "executionTimeMs": exec_time,
            "model_version": self.model_version,
            "modelVersion": self.model_version
        }


engine = MangoLeafInferenceEngine()
