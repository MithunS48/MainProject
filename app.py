"""
app.py
-------
FastAPI inference server for the Fish Disease Classifier.

Final model: MobileNetV2 + ConvNeXt features → Polynomial SVM
Accuracy: 98.29%  |  AUC: 0.9989

Endpoints:
    GET  /              — health check
    GET  /model-info    — model details
    POST /predict       — classify a fish image
    GET  /classes       — list all classes

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict

# Import the inference pipeline
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict, CLASS_NAMES, CLASS_LABELS, CLASS_DESCRIPTIONS

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Fish Disease Classifier API",
    description=(
        "Classifies fish images into 4 disease categories using "
        "MobileNetV2 + ConvNeXt feature fusion with Polynomial SVM. "
        "Accuracy: 98.29%"
    ),
    version="1.0.0"
)

# CORS — allow all origins for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RESPONSE MODELS
# ============================================================

class PredictionResponse(BaseModel):
    predicted_class  : str
    predicted_label  : str
    description      : str
    confidence       : float
    confidence_pct   : float
    all_scores       : Dict[str, float]


class ModelInfoResponse(BaseModel):
    model            : str
    feature_models   : list
    feature_dim      : int
    classifier       : str
    kernel           : str
    accuracy         : float
    f1_score         : float
    auc              : float
    classes          : list


class ClassInfo(BaseModel):
    name        : str
    label       : str
    description : str


# ============================================================
# ROUTES
# ============================================================

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status"  : "ok",
        "service" : "Fish Disease Classifier API",
        "version" : "1.0.0"
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    return ModelInfoResponse(
        model          = "MobileNetV2 + ConvNeXt + Polynomial SVM",
        feature_models = ["MobileNetV2", "ConvNeXtTiny"],
        feature_dim    = 2048,
        classifier     = "SVM",
        kernel         = "Polynomial (degree=3)",
        accuracy       = 0.9829,
        f1_score       = 0.9829,
        auc            = 0.9989,
        classes        = CLASS_NAMES
    )


@app.get("/classes", tags=["Model"])
def get_classes():
    return [
        ClassInfo(
            name        = cls,
            label       = CLASS_LABELS[cls],
            description = CLASS_DESCRIPTIONS[cls]
        )
        for cls in CLASS_NAMES
    ]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a fish image (JPG or PNG) and get a disease prediction.

    Returns the predicted class, confidence score, and scores for
    all 4 classes.
    """

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPG or PNG."
        )

    # Read image bytes
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    # Decode image from bytes
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Save to temp file for predict pipeline
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)

    try:
        result = predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return PredictionResponse(**result)


# ============================================================
# STARTUP — preload models
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Preload all models at startup so first request is fast."""
    print("\nPreloading models...")
    try:
        from predict import load_mobilenet_extractor, load_convnext_extractor, load_svm
        load_mobilenet_extractor()
        load_convnext_extractor()
        load_svm()
        print("All models loaded successfully.\n")
    except Exception as e:
        print(f"Warning: Could not preload models: {e}")
        print("Models will be loaded on first request.\n")
