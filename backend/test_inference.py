import os
import io
import glob
import numpy as np
from PIL import Image, ImageEnhance
from inference import engine, DISEASE_CLASSES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "Mango S data")

def get_sample_image(class_folder):
    """Retrieve first sample image for a given class from the Kaggle dataset."""
    folder_path = os.path.join(DATA_DIR, class_folder)
    images = glob.glob(os.path.join(folder_path, "*.*"))
    if not images:
        raise FileNotFoundError(f"No sample images found in {folder_path}")
    with open(images[0], "rb") as f:
        return f.read()

def create_sunlit_healthy_specimen():
    """Create a sunlit/high-brightness variant of a real healthy leaf."""
    folder_path = os.path.join(DATA_DIR, "Healthy")
    images = glob.glob(os.path.join(folder_path, "*.*"))
    img = Image.open(images[0]).convert("RGB")
    # Increase brightness & contrast by 30% to simulate harsh outdoor sunlight
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def create_multi_disease_composite():
    """
    Synthesize a realistic multi-disease leaf specimen by compositing a real
    Anthracnose diseased patch onto the left side and a real Powdery Mildew
    diseased patch onto the right side of a mango leaf canvas.
    """
    anthracnose_imgs = glob.glob(os.path.join(DATA_DIR, "Anthracnose", "*.*"))
    powdery_imgs = glob.glob(os.path.join(DATA_DIR, "Powdery Mildew", "*.*"))
    healthy_imgs = glob.glob(os.path.join(DATA_DIR, "Healthy", "*.*"))

    canvas = Image.open(healthy_imgs[0]).convert("RGB").resize((600, 450))
    anth_patch = Image.open(anthracnose_imgs[0]).convert("RGB").resize((180, 180))
    pwd_patch = Image.open(powdery_imgs[0]).convert("RGB").resize((180, 180))

    # Paste patches onto canvas
    canvas.paste(anth_patch, (50, 120))
    canvas.paste(pwd_patch, (350, 120))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_diagnostic_tests():
    print("=" * 75)
    print("MANGO LEAF YOLOv8 + CNN DUAL-STAGE INFERENCE DIAGNOSTIC SUITE")
    print(f"Model Engine: {engine.model_version}")
    print(f"Dataset Path: {DATA_DIR}")
    print("=" * 75)

    test_cases = [
        ("Healthy Specimen", lambda: get_sample_image("Healthy"), "Healthy", True),
        ("Harsh Sunlit Healthy Leaf (Glare Invariance)", create_sunlit_healthy_specimen, "Healthy", True),
        ("Anthracnose Pathological Specimen", lambda: get_sample_image("Anthracnose"), "Anthracnose", False),
        ("Bacterial Canker Pathological Specimen", lambda: get_sample_image("Bacterial Canker"), "Bacterial Canker", False),
        ("Cutting Weevil Specimen", lambda: get_sample_image("Cutting Weevil"), "Cutting Weevil", False),
        ("Die Back Specimen", lambda: get_sample_image("Die Back"), "Die Back", False),
        ("Gall Midge Specimen", lambda: get_sample_image("Gall Midge"), "Gall Midge", False),
        ("Powdery Mildew Specimen", lambda: get_sample_image("Powdery Mildew"), "Powdery Mildew", False),
        ("Sooty Mold Specimen", lambda: get_sample_image("Sooty Mould"), "Sooty Mold", False),
        ("Composite Multi-Disease Leaf (Anthracnose + Powdery Mildew)", create_multi_disease_composite, "multi_disease", False),
    ]

    for name, img_fn, expected_disease, expected_healthy in test_cases:
        print(f"\n[TEST CASE] {name}")
        img_bytes = img_fn()
        result = engine.predict(img_bytes)

        print(f"  * Primary Disease Detected: {result['disease']} ({result['confidence']}%)")
        print(f"  * Status: {result['status']}")
        print(f"  * Risk Level: {result['risk']}")
        print(f"  * Is Multiple Diseases: {result['is_multiple_diseases']}")
        
        det_list = [f"{d['name']} (CNN: {d.get('cnn_confidence', d['confidence'])}%, YOLO: {d.get('yolo_confidence', 'N/A')}%)" for d in result['predicted_diseases']]
        print(f"  * Detected Disease List ({len(result['predicted_diseases'])}): {det_list}")
        print(f"  * Localized Bounding Box Count: {len(result['detections'])}")
        for idx, det in enumerate(result['detections']):
            print(f"     -> Box {idx+1}: [{det['disease']}] CNN={det.get('cnn_confidence', det['confidence'])}%, YOLO={det.get('yolo_confidence', 'N/A')}% @ bbox={det['bbox']}")
        
        top3 = [p['name'] + ': ' + str(p['confidence']) + '%' for p in result['all_predictions'][:3]]
        print(f"  * Top-3 Distribution: {top3}")

        # Assertions
        if expected_healthy:
            assert result["is_healthy"] is True, f"{name} should be identified as healthy"
            assert result["disease"] == "Healthy", f"{name} should have primary disease 'Healthy'"
            assert len(result["detections"]) == 0, f"{name} should have 0 disease detections"
        elif expected_disease == "multi_disease":
            assert len(result["detections"]) >= 2, "Multi-disease leaf must localize at least 2 distinct regions"
            assert result["is_multiple_diseases"] is True, "Multi-disease specimen must set is_multiple_diseases=True"
            assert len(result["predicted_diseases"]) >= 2, "Multi-disease specimen must report at least 2 distinct diseases"
        else:
            assert result["is_healthy"] is False, f"{expected_disease} should not be marked healthy"
            assert result["disease"].lower() == expected_disease.lower() or any(d["name"].lower() == expected_disease.lower() for d in result["predicted_diseases"]), f"Expected {expected_disease}, got {result['disease']}"

    print("\n" + "=" * 75)
    print("ALL 10 DIAGNOSTIC REAL-IMAGE INFERENCE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    run_diagnostic_tests()
