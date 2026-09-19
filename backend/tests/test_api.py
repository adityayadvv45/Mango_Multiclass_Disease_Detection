"""
Automated FastAPI Endpoint Integration Test Suite.
Tests:
- GET /
- GET /health
- GET /classes
- POST /predict with real images and invalid data
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_root_and_health():
    print("\n" + "="*50)
    print("API TEST SUITE 1: Endpoints & Health Check")
    print("="*50)
    
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "online"
    print("  [PASS] GET / -> 200 OK")
    
    r_health = client.get("/health")
    assert r_health.status_code == 200
    health_data = r_health.json()
    assert health_data["status"] == "healthy"
    assert health_data["classes_count"] == 8
    print(f"  [PASS] GET /health -> 200 OK (Loaded models: {health_data['loaded_models']})")
    
    r_classes = client.get("/classes")
    assert r_classes.status_code == 200
    assert len(r_classes.json()["classes"]) == 8
    print("  [PASS] GET /classes -> 200 OK (8 classes)")

def test_api_predict():
    print("\n" + "="*50)
    print("API TEST SUITE 2: POST /predict with Real Image")
    print("="*50)
    
    img_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "Mango S data", "Anthracnose", "20211008_124249 (Custom).jpg"
    )
    if not os.path.exists(img_path):
        print(f"  [WARN] Sample image not found at {img_path}, skipping POST /predict test.")
        return
        
    with open(img_path, "rb") as f:
        files = {"file": ("anthracnose_test.jpg", f, "image/jpeg")}
        res = client.post("/predict", files=files)
        
    assert res.status_code == 200
    data = res.json()
    assert data["success"] == True
    assert data["leaf_detected"] == True
    assert len(data["predictions"]) == 8
    print(f"  [PASS] POST /predict -> 200 OK (Disease: {data['disease']}, Conf: {data['confidence']}%, Regions: {len(data['regions'])})")

def test_api_invalid_file():
    print("\n" + "="*50)
    print("API TEST SUITE 3: POST /predict with Corrupted Image")
    print("="*50)
    
    files = {"file": ("corrupted.jpg", b"bad image data", "image/jpeg")}
    res = client.post("/predict", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] == False
    assert "Corrupted or invalid" in data["error"] or "No mango leaf" in data["error"]
    print("  [PASS] POST /predict with corrupt image -> Handled gracefully with clean error JSON")

def run_all_api_tests():
    print("🚀 Running Mango Leaf API Integration Tests...")
    test_api_root_and_health()
    test_api_predict()
    test_api_invalid_file()
    print("\n" + "="*50)
    print("🎉 ALL API INTEGRATION TESTS PASSED!")
    print("="*50)

if __name__ == "__main__":
    run_all_api_tests()
