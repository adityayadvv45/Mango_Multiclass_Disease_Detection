import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import engine, DISEASE_CLASSES

app = FastAPI(
    title="Mango Leaf Disease Detection API",
    description="Multi-disease detection & bounding box localization for mango foliage",
    version="3.0.0"
)

# Configure CORS for Vite frontend (localhost:5173 / localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "MangoLeafInferenceEngine",
        "model_version": engine.model_version,
        "classes_count": len(DISEASE_CLASSES)
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": DISEASE_CLASSES,
        "count": len(DISEASE_CLASSES)
    }


@app.post("/predict")
async def predict_mango_leaf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided.")

    valid_content_types = ["image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"]
    if file.content_type and file.content_type not in valid_content_types:
        # Check filename extension as fallback
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{file.content_type}'. Please upload a JPG, PNG, or WEBP image.")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = engine.predict(contents)
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"[Error] Prediction exception: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error processing image.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
