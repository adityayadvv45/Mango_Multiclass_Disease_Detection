import os
import glob
import io
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "Mango S data")

def test_api_endpoints():
    print("=" * 70)
    print("FASTAPI LIVE ENDPOINTS INTEGRATION TEST SUITE")
    print("=" * 70)

    # 1. Health Check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    print(f"[/health] Status: {health_data['status']}, Model: {health_data['model_version']}, Classes: {health_data['classes_count']}")

    # 2. Classes Endpoint
    classes_resp = client.get("/classes")
    assert classes_resp.status_code == 200, f"Classes failed: {classes_resp.text}"
    classes_data = classes_resp.json()
    print(f"[/classes] Count: {classes_data['count']}, Sample Classes: {[c['name'] for c in classes_data['classes'][:4]]}")

    # 3. Real Image Prediction Tests across all 8 classes
    test_classes = [
        "Healthy",
        "Anthracnose",
        "Bacterial Canker",
        "Cutting Weevil",
        "Die Back",
        "Gall Midge",
        "Powdery Mildew",
        "Sooty Mould"
    ]
    for folder_name in test_classes:
        img_files = glob.glob(os.path.join(DATA_DIR, folder_name, "*.*"))
        if not img_files:
            continue
        with open(img_files[0], "rb") as f:
            file_bytes = f.read()

        files = {"file": ("test_leaf.jpg", file_bytes, "image/jpeg")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200, f"Predict failed for {folder_name}: {resp.text}"
        data = resp.json()

        print(f"\n[/predict - {folder_name}]")
        print(f"  * Status: {data['status']}")
        print(f"  * Disease: {data['disease']} ({data['confidence']}%)")
        print(f"  * Is Multiple: {data['is_multiple_diseases']}")
        print(f"  * Detections Count: {len(data['detections'])}")
        print(f"  * Execution Time: {data['execution_time_ms']}ms")

        expected_name = "Sooty Mold" if folder_name == "Sooty Mould" else folder_name

        if expected_name == "Healthy":
            assert data["is_healthy"] is True
            assert len(data["detections"]) == 0
            assert data["is_multiple_diseases"] is False
        else:
            assert data["is_healthy"] is False
            assert data["disease"] == expected_name or any(d["name"] == expected_name for d in data["predicted_diseases"])
            assert data["is_multiple_diseases"] is False
            assert len(data["predicted_diseases"]) == 1

    print("\n" + "=" * 70)
    print("ALL FASTAPI LIVE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_api_endpoints()
