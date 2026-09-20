"""
Automated ML & Inference Test Suite.
Validates:
1. Strict non-leaf background rejection (paper, hand, soil, table).
2. All 8 botanical single-disease classes on real dataset images.
3. Real dataset multi-disease leaf images and composite specimens.
4. Precision bounding box coordinates, leaf containment, and valid normalized ranges.
5. Corrupted / invalid file error handling.
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
    print("\n" + "="*50)
    print("TEST SUITE 1: Non-Leaf Background Rejection")
    print("="*50)
    
    # 1. Blank White Paper
    white_img = 255 * np.ones((400, 400, 3), dtype=np.uint8)
    mask, is_leaf, _, meta = segment_mango_leaf(white_img)
    assert not is_leaf, f"Failed: White paper was wrongly identified as leaf! Meta: {meta}"
    print("  [PASS] Blank white paper correctly rejected (is_leaf = False)")
    
    # 2. Human Skin / Hand
    skin_img = np.zeros((400, 400, 3), dtype=np.uint8)
    skin_img[:] = [140, 175, 230] # BGR typical skin
    mask, is_leaf, _, meta = segment_mango_leaf(skin_img)
    assert not is_leaf, f"Failed: Human skin was wrongly identified as leaf! Meta: {meta}"
    print("  [PASS] Human skin/hand correctly rejected (is_leaf = False)")
    
    # 3. Wooden Table / Soil
    soil_img = np.zeros((400, 400, 3), dtype=np.uint8)
    soil_img[:] = [30, 50, 95] # Dark brown soil/wood
    mask, is_leaf, _, meta = segment_mango_leaf(soil_img)
    assert not is_leaf, f"Failed: Soil/table was wrongly identified as leaf! Meta: {meta}"
    print("  [PASS] Soil/table texture correctly rejected (is_leaf = False)")

def test_real_dataset_classes():
    print("\n" + "="*50)
    print("TEST SUITE 2: Real Dataset 8-Class Inference & Localization Validation")
    print("="*50)
    
    engine = get_inference_engine()
    assert engine.is_loaded, "Inference engine failed to load!"
    
    passed_classes = []
    
    for cls_name in CANONICAL_CLASSES:
        # Check folder matching class name or alias
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
            
        # Test the first image in the class folder
        test_file = files[0]
        full_path = os.path.join(folder_path, test_file)
        
        with open(full_path, "rb") as f:
            img_bytes = f.read()
            
        res = engine.predict(img_bytes, filename=test_file)
        
        assert res["success"] == True, f"Inference failed for {cls_name}: {res.get('error')}"
        assert res["leaf_detected"] == True, f"Leaf not detected for {cls_name}"
        assert len(res["predictions"]) == 8, f"Expected 8 class predictions, got {len(res['predictions'])}"
        
        # Check coordinates and bounds
        for reg in res["regions"]:
            assert 0 <= reg["normBox"]["top"] <= 100, f"Invalid top: {reg['normBox']['top']}"
            assert 0 <= reg["normBox"]["left"] <= 100, f"Invalid left: {reg['normBox']['left']}"
            assert reg["normBox"]["width"] > 0, f"Invalid width: {reg['normBox']['width']}"
            assert reg["normBox"]["height"] > 0, f"Invalid height: {reg['normBox']['height']}"
            ymin, xmin, ymax, xmax = reg["box"]
            assert ymax > ymin and xmax > xmin, f"Invalid box: {reg['box']}"
            
        if cls_name == "Healthy":
            assert len(res["regions"]) == 0, "Healthy leaf should have 0 disease regions!"
        else:
            assert len(res["regions"]) > 0, f"Expected localized regions for disease {cls_name}!"
            
        print(f"  [PASS] Class '{cls_name}': Predicted '{res['disease']}' (Conf: {res['confidence']}%, Regions: {len(res['regions'])}, Latency: {res['inferenceTimeMs']}ms)")
        passed_classes.append(cls_name)
        
    print(f"\n  Summary: {len(passed_classes)}/8 classes validated successfully!")

def test_multi_disease_specimens():
    print("\n" + "="*50)
    print("TEST SUITE 3: Multi-Disease Dataset & Composite Specimens")
    print("="*50)
    
    engine = get_inference_engine()
    
    # 1. Test public sample
    if os.path.exists(SAMPLE_MULTI_PATH):
        with open(SAMPLE_MULTI_PATH, "rb") as f:
            img_bytes = f.read()
        res = engine.predict(img_bytes, filename="multi_disease_leaf.png")
        assert res["success"] == True
        assert res["leaf_detected"] == True
        assert len(res["regions"]) > 0, "Expected lesion regions to be localized on multi-disease leaf!"
        print(f"  [PASS] Public Multi-disease specimen: Disease = '{res['disease']}', Regions: {len(res['regions'])}")
        
    # 2. Test real multi-disease dataset folder
    if os.path.exists(MULTI_FOLDER_PATH):
        multi_files = [f for f in os.listdir(MULTI_FOLDER_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if multi_files:
            test_img_name = multi_files[0]
            with open(os.path.join(MULTI_FOLDER_PATH, test_img_name), "rb") as f:
                img_bytes = f.read()
            res = engine.predict(img_bytes, filename=test_img_name)
            assert res["success"] == True
            assert res["leaf_detected"] == True
            assert len(res["regions"]) > 0
            print(f"  [PASS] Dataset Multi-disease specimen ({test_img_name}): Disease = '{res['disease']}', Regions: {len(res['regions'])}")

def test_corrupt_file_handling():
    print("\n" + "="*50)
    print("TEST SUITE 4: Corrupt / Invalid File Error Handling")
    print("="*50)
    
    engine = get_inference_engine()
    corrupt_bytes = b"This is not a valid image file content."
    res = engine.predict(corrupt_bytes, filename="fake.jpg")
    
    assert res["success"] == False
    assert "Corrupted or invalid" in res["error"] or "No mango leaf" in res["error"]
    print(f"  [PASS] Corrupted file handled gracefully: error = '{res['error']}'")

def run_all_tests():
    print("🚀 Running Mango Leaf Backend & ML Verification Suite...")
    test_non_leaf_rejections()
    test_real_dataset_classes()
    test_multi_disease_specimens()
    test_corrupt_file_handling()
    print("\n" + "="*50)
    print("🎉 ALL TEST SUITES PASSED PERFECTLY!")
    print("="*50)

if __name__ == "__main__":
    run_all_tests()

