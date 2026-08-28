"""
research.py (router)
---------------------
Public, read-only endpoints that serve the REAL research results
already produced by the existing ML pipeline (results/ folder),
compiled into static/research_data/research_data.json by
extract_research_data.py.

GET /api/model-info          — headline model info (used across the app)
GET /api/research/all        — full research data bundle
GET /api/research/confusion-matrix
GET /api/research/classification-report
GET /api/research/roc-auc
GET /api/research/cnn-comparison
GET /api/research/fusion-results
GET /api/research/pca-comparison
GET /api/research/kernel-comparison
GET /api/research/error-analysis
GET /api/research/dataset-stats
"""

import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.core.config import RESEARCH_DATA_PATH, FINAL_MODEL_ACCURACY, FINAL_MODEL_F1, FINAL_MODEL_AUC
from app.core.inference import CLASS_NAMES, CLASS_LABELS, CLASS_DESCRIPTIONS

router = APIRouter(prefix="/api", tags=["Research & Model Info"])


@lru_cache(maxsize=1)
def _load_research_data() -> dict:
    if not RESEARCH_DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "research_data.json not found. Run "
                "webapp/backend/extract_research_data.py first."
            ),
        )
    with open(RESEARCH_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@router.get("/model-info")
def model_info():
    data = _load_research_data()
    return {
        "model": "MobileNetV2 + ConvNeXt + Polynomial SVM",
        "pipeline": [
            {"step": "Input Fish Image", "detail": "224 x 224 x 3 RGB"},
            {"step": "MobileNetV2", "detail": "1280-dimensional feature vector"},
            {"step": "ConvNeXt", "detail": "768-dimensional feature vector"},
            {"step": "Feature Fusion", "detail": "2048-dimensional concatenated vector"},
            {"step": "Polynomial SVM", "detail": "C = 1, degree = 3, gamma = scale"},
            {"step": "Disease Prediction", "detail": "One of 4 classes"},
        ],
        "feature_dimensions": {"MobileNetV2": 1280, "ConvNeXt": 768, "Fused": 2048},
        "classifier": "Polynomial SVM (C=1, degree=3, gamma=scale, coef0=0.0)",
        "accuracy": FINAL_MODEL_ACCURACY,
        "f1_score": FINAL_MODEL_F1,
        "auc": FINAL_MODEL_AUC,
        "correct_predictions": data["final_model"]["metrics"]["correct_predictions"],
        "incorrect_predictions": data["final_model"]["metrics"]["incorrect_predictions"],
        "test_samples": data["final_model"]["metrics"]["test_samples"],
        "classes": [
            {"name": c, "label": CLASS_LABELS[c], "description": CLASS_DESCRIPTIONS[c]}
            for c in CLASS_NAMES
        ],
    }


@router.get("/research/all")
def research_all():
    return _load_research_data()


@router.get("/research/confusion-matrix")
def confusion_matrix():
    return _load_research_data()["final_model"]["confusion_matrix"]


@router.get("/research/classification-report")
def classification_report():
    d = _load_research_data()["final_model"]
    return {
        "report": d["classification_report"],
        "report_text": d["classification_report_text"],
        "per_class_metrics": d["per_class_metrics"],
    }


@router.get("/research/roc-auc")
def roc_auc():
    return _load_research_data()["final_model"]["auc"]


@router.get("/research/cnn-comparison")
def cnn_comparison():
    return _load_research_data()["cnn_comparison"]


@router.get("/research/fusion-results")
def fusion_results():
    return _load_research_data()["fusion_results"]


@router.get("/research/pca-comparison")
def pca_comparison():
    return _load_research_data()["pca_comparison"]


@router.get("/research/kernel-comparison")
def kernel_comparison():
    return _load_research_data()["kernel_comparison"]


@router.get("/research/error-analysis")
def error_analysis():
    d = _load_research_data()["final_model"]
    return {
        "correct_predictions": d["metrics"]["correct_predictions"],
        "incorrect_predictions": d["metrics"]["incorrect_predictions"],
        "test_samples": d["metrics"]["test_samples"],
        "error_breakdown": d["error_analysis"],
        "sample_errors": _load_research_data()["incorrect_prediction_samples"],
    }


@router.get("/research/dataset-stats")
def dataset_stats():
    return _load_research_data()["dataset_statistics"]


@router.get("/classes")
def get_classes():
    return [
        {"name": c, "label": CLASS_LABELS[c], "description": CLASS_DESCRIPTIONS[c]}
        for c in CLASS_NAMES
    ]
