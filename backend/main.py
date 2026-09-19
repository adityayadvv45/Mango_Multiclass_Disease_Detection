"""
FastAPI Server for Real-Time Mango Leaf Multi-Disease Detection & Diagnosis.
Provides REST API endpoints for leaf disease inference, model health, and class metadata.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any

from backend.inference import get_inference_engine
from backend.models import CANONICAL_CLASSES, DISEASE_METADATA

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models once into memory
    print("[SERVER] Starting MangoLeaf AI Diagnostic Engine...")
    engine = get_inference_engine()
    print(f"[SERVER] Engine initialized with models: {list(engine.models.keys())}")
    yield
    # Shutdown
    print("[SERVER] Shutting down MangoLeaf server...")

app = FastAPI(
    title="Mango Guard AI - Foliar Diagnostic API",
    description="Deep Learning API for Real Mango Leaf Disease Detection, Leaf Segmentation, and Multi-Pathology Lesion Localization.",
    version="2.4.0",
    lifespan=lifespan
)

# Configure CORS for local development and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Mango Leaf Multiple Disease Detection AI",
        "version": "2.4.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "classes": "GET /classes"
        }
    }

@app.get("/health")
def health_check():
    engine = get_inference_engine()
    return {
        "status": "healthy" if engine.is_loaded else "degraded",
        "loaded_models": list(engine.models.keys()),
        "classes_count": len(engine.class_names),
        "classes": engine.class_names,
        "metrics": engine.metrics
    }

@app.get("/classes")
def get_classes():
    return {
        "classes": CANONICAL_CLASSES,
        "metadata": DISEASE_METADATA
    }

@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):
    """
    Analyzes an uploaded leaf image:
    1. Validates and segments the mango leaf blade.
    2. Rejects paper, hand, fingers, soil, table, or background artifacts.
    3. Runs deep PyTorch CNN consensus inference across all 8 classes.
    4. Localizes genuine lesion bounding boxes on the leaf blade.
    5. Returns multi-pathology detection result and actionable guidance.
    """
    if not file:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No image file provided."}
        )
        
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Uploaded image file is empty."}
            )
            
        engine = get_inference_engine()
        result = engine.predict(image_bytes, filename=file.filename or "leaf.jpg")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An error occurred during leaf disease analysis.",
                "details": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
