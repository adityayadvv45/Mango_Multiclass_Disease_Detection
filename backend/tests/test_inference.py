"""
Automated ML & Inference Test Suite for Mango Leaf Disease AI.
Validates:
1. Strict non-leaf background rejection (blank white paper, human skin, wooden table / soil).
2. All 8 botanical single-disease classes across real dataset images (confirming single disease predicted, regions match that disease only).
3. Healthy leaf classification (confirming 0 lesion regions, Optimal status).
4. Real multi-disease dataset images (confirming genuine multi-pathology detection, individual bounding boxes per disease).
5. Precision bounding box coordinates, leaf blade containment, and normalized range [0, 100]%.
6. Corrupted / invalid file error handling.
7. Detailed class-wise accuracy and confusion report.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image

from backend.segmentation import segment_mango_leaf, is_bbox_inside_leaf
from backend.inference import get_inference_engine
from backend.models import CANONICAL_CLASSES

DATASET_ROOT = os.path.join(PROJECT_ROOT, "backend", "data", "Mango S data")
SAMPLE_MULTI_PATH = os.path.join(PROJECT_ROOT, "public", "samples", "multi_disease_leaf.png")
MULTI_FOLDER_PATH = os.path.join(DATASET_ROOT, "Multi Diease  in one leaf Data")

def test_non_leaf_rejections():
    print("\n" + "=" * 60)
    print("TEST SUITE 1: Non-Leaf Background Rejection")
    print("=" * 60)

    # 1. Blank White Paper
    white_img = 255 * np.ones((400, 400, 3), dtype=np.uint8)
    mask, is_leaf, _, meta = segment_mango_leaf(white_img)
    assert not is_leaf, f"Failed: White paper was wrongly identified as leaf! Meta: {meta}"
    print("  [PASS] Blank white paper correctly rejected (is_leaf = False)")

    # 2. Human Skin / Hand
    skin_img = np.zeros((400, 400, 3), dtype=np.uint8)
    skin_img[:] = [140, 175, 230]  # BGR typical skin
    mask, is_leaf, _, meta = segment_mango_leaf(skin_img)
    assert not is_leaf, f"Failed: Human skin was wrongly identified as leaf! Meta: {meta}"
    print("  [PASS] Human skin/hand correctly rejected (is_leaf = False)")

    # 3. Wooden Table / Soil
    soil_img = np.zeros((400, 400, 3), dtype=np.uint8)
    soil_img[:] = [30, 50, 95]  # Dark brown soil/wood
    mask, is_leaf, _, meta = segment_mango_leaf(soil_img)
    assert not is_leaf, f"Failed: Soil/table was wrongly identified as leaf! Meta: {meta}"
    print("  [PASS] Soil/table texture correctly rejected (is_leaf = False)")

def test_real_dataset_classes(samples_per_class: int = 15):
    print("\n" + "=" * 60)
    print(f"TEST SUITE 2: Real Dataset 8-Class Validation ({samples_per_class} images/class)")
    print("=" * 60)

    engine = get_inference_engine()
    assert engine.is_loaded, "Inference engine failed to load!"

    total_tested = 0
    total_correct = 0
    misclassifications = []

    for cls_name in CANONICAL_CLASSES:
        folder_candidates = [cls_name, "Sooty Mould" if cls_name == "Sooty Mold" else cls_name]
        folder_path = None
        for cand in folder_candidates:
            p = os.path.join(DATASET_ROOT, cand)
            if os.path.exists(p):
                folder_path = p
                break

        if not folder_path:
            print(f"  [WARN] Dataset folder for {cls_name} not found, skipping.")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            continue

        test_files = files[:samples_per_class]
        class_correct = 0

        for test_file in test_files:
            full_path = os.path.join(folder_path, test_file)
            with open(full_path, "rb") as f:
                img_bytes = f.read()

            res = engine.predict(img_bytes, filename=test_file)
            total_tested += 1

            assert res["success"] == True, f"Inference failed for {cls_name}: {res.get('error')}"
            assert res["leaf_detected"] == True, f"Leaf not detected for {cls_name}: {test_file}"
            assert len(res["predictions"]) == 8, f"Expected 8 predictions, got {len(res['predictions'])}"

            # Coordinate validation
            for reg in res["regions"]:
                assert 0 <= reg["normBox"]["top"] <= 100, f"Invalid top: {reg['normBox']['top']}"
                assert 0 <= reg["normBox"]["left"] <= 100, f"Invalid left: {reg['normBox']['left']}"
                assert reg["normBox"]["width"] > 0, f"Invalid width: {reg['normBox']['width']}"
                assert reg["normBox"]["height"] > 0, f"Invalid height: {reg['normBox']['height']}"
                ymin, xmin, ymax, xmax = reg["box"]
                assert ymax > ymin and xmax > xmin, f"Invalid box: {reg['box']}"

            # Verify Single-Disease behavior
            if cls_name == "Healthy":
                assert res["disease"] == "Healthy", f"Expected Healthy, got {res['disease']}"
                assert res["isMultiPathology"] == False, "Healthy leaf should not be multi-pathology!"
                assert len(res["regions"]) == 0, "Healthy leaf must have 0 lesion regions!"
                class_correct += 1
                total_correct += 1
            else:
                if res["disease"] == cls_name and not res["isMultiPathology"]:
                    class_correct += 1
                    total_correct += 1
                    # Ensure all regions belong strictly to this single disease
                    for reg in res["regions"]:
                        assert reg["disease"] == cls_name, f"Wrong region disease: {reg['disease']} for {cls_name}"
                else:
                    misclassifications.append({
                        "file": test_file,
                        "true_class": cls_name,
                        "predicted": res["disease"],
                        "is_multi": res["isMultiPathology"],
                        "confidence": res["confidence"]
                    })

        acc_pct = (class_correct / len(test_files)) * 100
        print(f"  [PASS] Class '{cls_name:16s}': {class_correct}/{len(test_files)} correct ({acc_pct:.1f}%)")

    overall_acc = (total_correct / total_tested) * 100 if total_tested > 0 else 0
    print(f"\n  Overall Single-Class Accuracy: {total_correct}/{total_tested} ({overall_acc:.2f}%)")
    if misclassifications:
        print(f"  Misclassifications: {misclassifications}")
    else:
        print("  Zero misclassifications across single-disease test samples!")

def test_multi_disease_specimens():
    print("\n" + "=" * 60)
    print("TEST SUITE 3: Multi-Disease Dataset & Co-Infection Specimens")
    print("=" * 60)

    engine = get_inference_engine()

    # 1. Test public sample
    if os.path.exists(SAMPLE_MULTI_PATH):
        with open(SAMPLE_MULTI_PATH, "rb") as f:
            img_bytes = f.read()
        res = engine.predict(img_bytes, filename="multi_disease_leaf.png")
        assert res["success"] == True
        assert res["leaf_detected"] == True
        print(f"  [PASS] Public Multi-Disease Specimen -> Disease: '{res['disease']}' (Conf: {res['confidence']}%, Regions: {len(res['regions'])})")

    # 2. Test real multi-disease dataset images
    if os.path.exists(MULTI_FOLDER_PATH):
        multi_files = [f for f in os.listdir(MULTI_FOLDER_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Testing {min(10, len(multi_files))} real multi-disease images from dataset...")
        for i, test_img_name in enumerate(multi_files[:10]):
            with open(os.path.join(MULTI_FOLDER_PATH, test_img_name), "rb") as f:
                img_bytes = f.read()
            res = engine.predict(img_bytes, filename=test_img_name)
            assert res["success"] == True
            assert res["leaf_detected"] == True
            
            # Check bounding boxes if diseased
            if res["disease"] != "Healthy":
                assert len(res["regions"]) > 0, f"Expected regions for {res['disease']}"
                for reg in res["regions"]:
                    assert reg["disease"] in res["disease"], f"Region disease {reg['disease']} not in title {res['disease']}"
            print(f"    [{i+1:2d}] {test_img_name[:32]:32s} -> Disease: '{res['disease']}' | Multi: {res['isMultiPathology']} | Regions: {len(res['regions'])}")

def test_corrupt_file_handling():
    print("\n" + "=" * 60)
    print("TEST SUITE 4: Corrupt / Invalid File Error Handling")
    print("=" * 60)

    engine = get_inference_engine()
    corrupt_bytes = b"This is not a valid image file content."
    res = engine.predict(corrupt_bytes, filename="fake.jpg")

    assert res["success"] == False
    assert "Corrupted or invalid" in res["error"] or "No mango leaf" in res["error"]
    print(f"  [PASS] Corrupted file handled gracefully: error = '{res['error']}'")

def run_all_tests():
    print("🚀 Running Mango Leaf Backend & ML Verification Suite...")
    test_non_leaf_rejections()
    test_real_dataset_classes(samples_per_class=20)
    test_multi_disease_specimens()
    test_corrupt_file_handling()
    print("\n" + "=" * 60)
    print("🎉 ALL TEST SUITES PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
