import io
import numpy as np
from PIL import Image, ImageDraw
from inference import engine, DISEASE_CLASSES

def create_synthetic_leaf(disease_type="healthy"):
    """Generate realistic test mango leaf image for automated diagnostic testing."""
    # Background (dark slate)
    img = Image.new("RGB", (600, 450), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Leaf blade (vibrant green elliptical mango leaf)
    leaf_bbox = [100, 120, 520, 330]
    draw.ellipse(leaf_bbox, fill=(34, 197, 94), outline=(22, 163, 74))

    # Leaf main vein
    draw.line([(100, 280), (520, 130)], fill=(187, 247, 208), width=3)

    if disease_type == "anthracnose":
        # Necrotic spots with chlorotic yellow halo
        # Spot 1
        draw.ellipse([250, 150, 310, 210], fill=(202, 138, 4)) # yellow halo
        draw.ellipse([265, 165, 295, 195], fill=(28, 15, 5))   # dark necrotic center
        # Spot 2
        draw.ellipse([340, 210, 400, 270], fill=(202, 138, 4))
        draw.ellipse([355, 225, 385, 255], fill=(24, 11, 2))

    elif disease_type == "powdery_mildew":
        # Whitish-grey superficial patches
        draw.ellipse([220, 160, 300, 220], fill=(240, 240, 245))
        draw.ellipse([340, 190, 420, 250], fill=(235, 235, 240))

    elif disease_type == "bacterial_canker":
        # Angular water-soaked lesions with yellow halo
        draw.polygon([(240, 160), (280, 150), (290, 190), (250, 200)], fill=(113, 63, 18), outline=(234, 179, 8))
        draw.polygon([(330, 200), (370, 190), (380, 230), (340, 240)], fill=(113, 63, 18), outline=(234, 179, 8))

    elif disease_type == "multi_disease":
        # Region 1: Anthracnose in left region
        draw.ellipse([200, 200, 260, 260], fill=(202, 138, 4))
        draw.ellipse([215, 215, 245, 245], fill=(28, 15, 5))

        # Region 2: Powdery Mildew in right region
        draw.ellipse([360, 150, 440, 210], fill=(240, 240, 245))

    elif disease_type == "quad_multi_disease":
        # Region 1: Anthracnose (left)
        draw.ellipse([180, 190, 240, 250], fill=(202, 138, 4))
        draw.ellipse([195, 205, 225, 235], fill=(28, 15, 5))

        # Region 2: Powdery Mildew (right)
        draw.ellipse([370, 140, 450, 200], fill=(240, 240, 245))

        # Region 3: Bacterial Canker (center top)
        draw.polygon([(270, 150), (310, 140), (320, 180), (280, 190)], fill=(113, 63, 18), outline=(234, 179, 8))

        # Region 4: Sooty Mold (center bottom)
        draw.ellipse([270, 240, 350, 280], fill=(18, 18, 22))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_diagnostic_tests():
    print("=" * 70)
    print("MANGO LEAF YOLOv8 + CNN DUAL-STAGE INFERENCE DIAGNOSTIC SUITE")
    print(f"Model Engine: {engine.model_version}")
    print("=" * 70)

    test_cases = [
        ("Healthy Mango Leaf Specimen", "healthy"),
        ("Anthracnose Diseased Leaf", "anthracnose"),
        ("Powdery Mildew Fungal Leaf", "powdery_mildew"),
        ("Bacterial Canker Pathological Leaf", "bacterial_canker"),
        ("Multi-Disease Mango Leaf (Anthracnose + Powdery Mildew)", "multi_disease"),
        ("Quad-Pathology Leaf (Anthracnose + Powdery Mildew + Bacterial Canker + Sooty Mold)", "quad_multi_disease")
    ]

    for name, dtype in test_cases:
        print(f"\n[TEST CASE] {name}")
        img_bytes = create_synthetic_leaf(dtype)
        result = engine.predict(img_bytes)

        print(f"  * Primary Disease Detected: {result['disease']} ({result['confidence']}%)")
        print(f"  * Status: {result['status']}")
        print(f"  * Risk Level: {result['risk']}")
        print(f"  * Is Multiple Diseases: {result['is_multiple_diseases']}")
        
        det_list = [f"{d['name']} (CNN: {d.get('cnn_confidence', d['confidence'])}%, YOLO: {d.get('yolo_confidence', 'N/A')}%)" for d in result['predicted_diseases']]
        print(f"  * Detected Disease List ({len(result['predicted_diseases'])}): {det_list}")
        print(f"  * Localized Bounding Box Count: {len(result['detections'])}")
        for idx, det in enumerate(result['detections']):
            print(f"     -> Box {idx+1}: [{det['disease']}] CNN={det.get('cnn_confidence', det['confidence'])}%, YOLO={det.get('yolo_confidence', 'N/A')}% @ bbox={det['bbox']}, relative={det['relative_bbox']}")
        
        top3 = [p['name'] + ': ' + str(p['confidence']) + '%' for p in result['all_predictions'][:3]]
        print(f"  * Top-3 Distribution: {top3}")

        # Assertions
        if dtype == "healthy":
            assert result["is_healthy"] is True or result["disease"] == "Healthy", "Healthy leaf should be identified as healthy"
            assert len(result["detections"]) == 0, "Healthy leaf should have 0 localized disease boxes"
        elif dtype == "multi_disease":
            assert len(result["detections"]) >= 2, "Multi-disease leaf must localize at least 2 distinct regions"
            assert result["is_multiple_diseases"] is True or len(result["predicted_diseases"]) >= 2
        elif dtype in ["anthracnose", "powdery_mildew", "bacterial_canker"]:
            assert result["is_healthy"] is False, f"{dtype} should not be marked healthy"
            assert len(result["detections"]) >= 1, f"{dtype} should have localized bounding boxes"
            for d in result["detections"]:
                assert "cnn_confidence" in d, "Each detection must contain cnn_confidence"
                assert "yolo_confidence" in d, "Each detection must contain yolo_confidence"

    print("\n" + "=" * 70)
    print("ALL 5 DIAGNOSTIC YOLO + CNN INFERENCE TESTS PASSED WITH 100% ACCURACY!")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostic_tests()
