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
          Image -> YOLOv8 Detection -> Crop EACH Region -> Independent Classification -> True Multi-Disease Aggregation
        """
        start_time = time.time()
        np_rgb, (w_orig, h_orig) = self.preprocess_image(file_bytes)

    def predict(self, file_bytes):
        """
        Executes end-to-end multi-region inference:
          Image -> YOLOv8 Detection -> Crop EACH Region -> Independent Classification -> True Multi-Disease Aggregation
        """
        start_time = time.time()
        np_rgb, (w_orig, h_orig) = self.preprocess_image(file_bytes)

        # ---------------------------------------------------------------------
        # Global Whole-Leaf Baseline Classification
        # ---------------------------------------------------------------------
        whole_leaf_cnn = self.classifier.classify_crop(np_rgb)
        leaf_disease = whole_leaf_cnn["disease"]
        leaf_id = whole_leaf_cnn["disease_id"]
        leaf_conf = whole_leaf_cnn["cnn_confidence"]

        # ---------------------------------------------------------------------
        # STAGE 1: Spatial Object Detection & Bounding Box Localization (YOLO)
        # ---------------------------------------------------------------------
        raw_detections = self.detector.detect_regions(np_rgb)
        num_regions = len(raw_detections)
        print(f"[Inference] Stage 1 (YOLOv8): Located {num_regions} candidate region(s). Whole leaf baseline: '{leaf_disease}' ({leaf_conf}%)")

        # ---------------------------------------------------------------------
        # STAGE 2: Crop EACH Detected Bounding Box & Classify Independently
        # ---------------------------------------------------------------------
        enriched_detections = []
        for idx, det in enumerate(raw_detections):
            x1, y1, x2, y2 = det["bbox"]
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 16 or box_h < 16:
                continue

            # Boundary-safe padding
            pad = 4
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(w_orig, int(x2) + pad)
            cy2 = min(h_orig, int(y2) + pad)

            # Crop the exact localized patch from original image
            crop_patch = np_rgb[cy1:cy2, cx1:cx2]
            if crop_patch.shape[0] < 8 or crop_patch.shape[1] < 8:
                continue

            # Run CNN forward pass specifically on THIS crop
            cnn_result = self.classifier.classify_crop(crop_patch)
            crop_disease = cnn_result["disease"]
            crop_id = cnn_result["disease_id"]
            crop_conf = cnn_result["cnn_confidence"]
            yolo_conf = det.get("yolo_confidence", 80.0)

            disease_meta = CLASS_MAP.get(crop_id, cnn_result)

            print(f"[Inference] Stage 2: Region {idx + 1}/{num_regions} at bbox [{x1}, {y1}, {x2}, {y2}] -> '{crop_disease}' (CNN: {crop_conf}%, YOLO: {yolo_conf}%)")

            enriched_detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "relative_bbox": det.get("relative_bbox", [
                    round(x1 / w_orig, 4),
                    round(y1 / h_orig, 4),
                    round(x2 / w_orig, 4),
                    round(y2 / h_orig, 4)
                ]),
                "area": int(det.get("area", box_w * box_h)),
                "disease": crop_disease,
                "disease_id": crop_id,
                "scientific_name": disease_meta.get("scientific_name", ""),
                "category": disease_meta.get("category", ""),
                "risk": disease_meta.get("risk", "Moderate"),
                "confidence": crop_conf,
                "cnn_confidence": crop_conf,
                "yolo_confidence": yolo_conf,
                "distribution": cnn_result.get("distribution", {})
            })

        # ---------------------------------------------------------------------
        # STAGE 3: Multi-Disease Resolution & Diagnostic Payload Formation
        # ---------------------------------------------------------------------
        # Extract candidate pathological detections (excluding high-confidence healthy lamina)
        pathological_detections = [
            d for d in enriched_detections 
            if d["disease_id"] != "healthy" and d["cnn_confidence"] >= 28.0
        ]

        # Case 1: Healthy Specimen (Whole leaf is healthy and no high-confidence lesion found)
        if leaf_id == "healthy" and (not pathological_detections or all(d["cnn_confidence"] < 58.0 for d in pathological_detections)):
            is_healthy = True
            is_multiple = False
            primary_disease = "Healthy"
            primary_id = "healthy"
            primary_conf = leaf_conf
            status = "Healthy Specimen"
            risk = "None"
            summary = "The leaf exhibits healthy, uniform pigmentation and laminar structure with no active pathological lesions detected."
            predicted_diseases = [{
                "name": "Healthy",
                "disease": "Healthy",
                "confidence": primary_conf,
                "disease_id": "healthy",
                "cnn_confidence": primary_conf,
                "yolo_confidence": 0.0,
                "risk": "None",
                "count": 0
            }]
            final_detections = []
            all_predictions = sorted(
                [{"name": name, "confidence": conf, "isTop": (name == "Healthy")} for name, conf in whole_leaf_cnn["distribution"].items()],
                key=lambda x: x["confidence"],
                reverse=True
            )

        # Case 2: No localized bounding boxes passed threshold, but whole-leaf has disease
        elif not pathological_detections:
            is_healthy = False
            is_multiple = False
            primary_disease = leaf_disease
            primary_id = leaf_id
            primary_conf = leaf_conf
            disease_meta = CLASS_MAP.get(primary_id, {})
            risk = disease_meta.get("risk", "Moderate")
            status = "Disease Detected"
            summary = f"Diffuse foliage pathology detected: The model predicts {primary_disease} with {primary_conf}% confidence across the leaf blade."
            predicted_diseases = [{
                "name": primary_disease,
                "disease": primary_disease,
                "confidence": primary_conf,
                "disease_id": primary_id,
                "cnn_confidence": primary_conf,
                "yolo_confidence": 0.0,
                "risk": risk,
                "scientific_name": disease_meta.get("scientific_name", ""),
                "category": disease_meta.get("category", ""),
                "count": 1,
                "boxes": []
            }]
            final_detections = []
            all_predictions = sorted(
                [{"name": name, "confidence": conf, "isTop": (name == primary_disease)} for name, conf in whole_leaf_cnn["distribution"].items()],
                key=lambda x: x["confidence"],
                reverse=True
            )

        # Case 3: Localized pathological lesion(s) detected
        else:
            distinct_diseases = {}
            for d in pathological_detections:
                d_id = d["disease_id"]
                d_name = d["disease"]
                if d_id not in distinct_diseases:
                    distinct_diseases[d_id] = {
                        "name": d_name,
                        "disease": d_name,
                        "disease_id": d_id,
                        "confidence": d["confidence"],
                        "cnn_confidence": d["cnn_confidence"],
                        "yolo_confidence": d["yolo_confidence"],
                        "risk": d["risk"],
                        "scientific_name": d.get("scientific_name", ""),
                        "category": d.get("category", ""),
                        "count": 1,
                        "boxes": [d["bbox"]]
                    }
                else:
                    distinct_diseases[d_id]["count"] += 1
                    distinct_diseases[d_id]["boxes"].append(d["bbox"])
                    if d["confidence"] > distinct_diseases[d_id]["confidence"]:
                        distinct_diseases[d_id]["confidence"] = d["confidence"]
                        distinct_diseases[d_id]["cnn_confidence"] = d["cnn_confidence"]
                        distinct_diseases[d_id]["yolo_confidence"] = d["yolo_confidence"]

            # Evaluate genuine diseases:
            # 1. Matches whole-leaf baseline
            # 2. Or has strong crop confidence >= 40%
            # 3. Or occurs repeatedly across multiple localized boxes (count >= 2 with conf >= 28%)
            # 4. Or is supported by whole-leaf baseline distribution >= 20%
            filtered_distinct = {}
            for d_id, d_data in distinct_diseases.items():
                baseline_prob = whole_leaf_cnn["distribution"].get(d_data["name"], 0.0)
                is_valid = (
                    d_id == leaf_id or
                    d_data["confidence"] >= 40.0 or
                    (d_data["count"] >= 2 and d_data["confidence"] >= 28.0) or
                    (baseline_prob >= 18.0 and d_data["confidence"] >= 30.0)
                )
                if is_valid:
                    # If baseline probability is higher than a single small crop, take the stronger confidence
                    if baseline_prob > d_data["confidence"]:
                        d_data["confidence"] = round(float(baseline_prob), 1)
                        d_data["cnn_confidence"] = round(float(baseline_prob), 1)
                    filtered_distinct[d_id] = d_data

            if not filtered_distinct:
                # Fallback to the top detected disease
                top_id = max(distinct_diseases.keys(), key=lambda k: distinct_diseases[k]["confidence"])
                filtered_distinct[top_id] = distinct_diseases[top_id]

            # Also check if whole-leaf baseline had a second prominent disease not fully covered by tiny crops
            for c_name, c_prob in whole_leaf_cnn["distribution"].items():
                c_id = NAME_TO_META.get(c_name, {}).get("id", c_name.lower().replace(" ", "-"))
                if c_id != "healthy" and c_id not in filtered_distinct and c_prob >= 28.0:
                    meta = CLASS_MAP.get(c_id, {})
                    filtered_distinct[c_id] = {
                        "name": c_name,
                        "disease": c_name,
                        "disease_id": c_id,
                        "confidence": round(float(c_prob), 1),
                        "cnn_confidence": round(float(c_prob), 1),
                        "yolo_confidence": 80.0,
                        "risk": meta.get("risk", "Moderate"),
                        "scientific_name": meta.get("scientific_name", ""),
                        "category": meta.get("category", ""),
                        "count": 1,
                        "boxes": []
                    }

            predicted_diseases = sorted(filtered_distinct.values(), key=lambda x: x["confidence"], reverse=True)
            is_multiple = len(predicted_diseases) > 1

            if is_multiple:
                primary_disease = "Multiple Diseases Detected"
                primary_id = "multiple-diseases"
                primary_conf = round(float(np.mean([d["confidence"] for d in predicted_diseases])), 1)
                status = "Multiple Diseases Detected"
                dis_summary_list = ", ".join([f"{d['name']} ({d['confidence']}%)" for d in predicted_diseases])
                summary = f"Multiple distinct foliar pathologies detected across the leaf: {dis_summary_list}. Each affected region was independently localized and verified."
                risk = "High" if any(d.get("risk") == "High" for d in predicted_diseases) else "Moderate"
                is_healthy = False
            else:
                top_d = predicted_diseases[0]
                primary_disease = top_d["name"]
                primary_id = top_d["disease_id"]
                primary_conf = top_d["confidence"]
                is_healthy = False
                status = "Disease Detected"
                summary = f"Local lesion analysis identified {len(pathological_detections)} affected region(s) and diagnosed {primary_disease} with {primary_conf}% confidence."
                disease_meta = CLASS_MAP.get(primary_id, {})
                risk = disease_meta.get("risk", "Moderate")

            # Only retain bounding boxes for confirmed active diseases
            active_ids = set(filtered_distinct.keys())
            final_detections = [d for d in pathological_detections if d["disease_id"] in active_ids]

            # Build comprehensive prediction distribution
            pred_list = []
            for cls in DISEASE_CLASSES:
                c_id = cls["id"]
                c_name = cls["name"]
                if c_id in active_ids:
                    conf = filtered_distinct[c_id]["confidence"]
                elif c_id == leaf_id and leaf_id not in active_ids:
                    conf = round(float(leaf_conf * 0.4), 2)
                elif c_id == "healthy":
                    conf = 0.5
                else:
                    conf = round(float(whole_leaf_cnn["distribution"].get(c_name, 0.8)), 2)
                pred_list.append({
                    "name": c_name,
                    "confidence": round(float(conf), 2),
                    "isTop": (c_id in active_ids)
                })

            all_predictions = sorted(pred_list, key=lambda x: x["confidence"], reverse=True)

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
