"""
Evaluation & Benchmarking Script for Roboflow Mango Leaf Dataset
Tests the full CNN + YOLOv8 inference pipeline on the uploaded Roboflow dataset
and prints structured results comparing:
1. Actual Labels
2. CNN Prediction
3. YOLOv8 Detections
4. Final Combined Prediction
5. Misclassified Images
"""

import os
import sys
import glob
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import get_inference_engine
from backend.models import CANONICAL_CLASSES, normalize_class_name

ROBOFLOW_CLASSES = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Powdery Mildew",
    3: "Die Back",
    4: "Gall Midge"
}

def parse_yolo_labels(label_file: str) -> list:
    """Extracts ground-truth class names from a Roboflow YOLO annotation file."""
    if not os.path.exists(label_file):
        return []
    classes_found = set()
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
                if cls_id in ROBOFLOW_CLASSES:
                    classes_found.add(ROBOFLOW_CLASSES[cls_id])
                else:
                    classes_found.add(f"Unknown_{cls_id}")
            except ValueError:
                continue
    return sorted(list(classes_found))

def evaluate_roboflow_dataset(split="test", max_samples=34):
    engine = get_inference_engine()
    
    dataset_dir = os.path.join(PROJECT_ROOT, "backend", "data", "Multiclass_leaf.v5i.yolov8-data", split)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist.")
        return
        
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")))
    if max_samples:
        image_files = image_files[:max_samples]
        
    print("=" * 110)
    print(f"🧪 ROBOFLOW DATASET VALIDATION BENCHMARK (Split: {split.upper()}, Samples: {len(image_files)})")
    print("=" * 110)
    
    results_summary = []
    correct_count = 0
    misclassified = []
    
    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(labels_dir, f"{base_name}.txt")
        
        actual_labels = parse_yolo_labels(label_file)
        if not actual_labels:
            actual_labels = ["Unlabeled / Healthy"]
            
        with open(img_path, "rb") as f:
            img_bytes = f.read()
            
        res = engine.predict(img_bytes, filename=img_name)
        
        # Extract CNN predictions
        top_cnn = res.get("primaryDiseaseName", "N/A")
        top_conf = res.get("confidence", 0.0)
        cnn_str = f"{top_cnn} ({top_conf}%)"
        
        # Extract YOLO detections
        regions = res.get("regions", [])
        yolo_box_diseases = [r.get("disease", "N/A") for r in regions]
        yolo_counts = defaultdict(int)
        for d in yolo_box_diseases:
            yolo_counts[d] += 1
        yolo_str = ", ".join([f"{k} ({v} boxes)" for k, v in yolo_counts.items()]) if yolo_counts else "None"
        
        # Final combined prediction
        final_disease = res.get("disease", "N/A")
        is_multi = res.get("isMultiPathology", False)
        detected_diseases = [d.get("name") for d in res.get("detectedDiseases", [])]
        
        # Verification: Check if primary or multi-disease matches ground truth
        matched = False
        for actual in actual_labels:
            if actual in detected_diseases or actual == top_cnn or actual == final_disease:
                matched = True
                break
                
        if matched:
            correct_count += 1
            status = "✅ MATCH"
        else:
            status = "❌ MISMATCH"
            misclassified.append({
                "image": img_name[:35] + "...",
                "actual": actual_labels,
                "cnn": cnn_str,
                "yolo": yolo_str,
                "final": final_disease
            })
            
        print(f"[{idx+1:02d}/{len(image_files):02d}] {img_name[:32]}...")
        print(f"     Actual Labels    : {', '.join(actual_labels)}")
        print(f"     CNN Top Pred     : {cnn_str}")
        print(f"     YOLO Detections  : {yolo_str}")
        print(f"     Final Combined   : {final_disease} (Multi: {is_multi}) -> {status}")
        print("-" * 110)
        
    accuracy = (correct_count / len(image_files)) * 100.0 if image_files else 0.0
    print("\n" + "=" * 110)
    print(f"📈 VALIDATION SUMMARY: {correct_count}/{len(image_files)} correct ({accuracy:.1f}%)")
    print(f"   Total Images Tested: {len(image_files)}")
    print(f"   Misclassified Count: {len(misclassified)}")
    if misclassified:
        print("\n⚠️ MISCLASSIFIED IMAGES DETAILS:")
        for m in misclassified:
            print(f"   • File: {m['image']}")
            print(f"     Actual: {m['actual']} | CNN: {m['cnn']} | YOLO: {m['yolo']} | Final: {m['final']}")
    print("=" * 110)

if __name__ == "__main__":
    evaluate_roboflow_dataset(split="test")
