"""
inference.py
-------------
Thin wrapper around the EXISTING ML pipeline in ../../../src/predict.py.

IMPORTANT — why this wrapper exists (and does not just call
`src/predict.predict()` directly):

  1. The trained ConvNeXt model file
     (results/ConvNeXt/model/convnext_best.keras) is intentionally
     excluded from the git repository (see .gitignore) because of its
     size. It is hosted on Google Drive — see README.md, section
     "Pre-trained Models & Features". Until that file (and the
     MobileNetV2 / VGG16 .keras files) are downloaded and placed under
     results/<Model>/model/, real inference cannot run. This wrapper
     detects that case and raises a clear, actionable error instead of
     a generic file-not-found traceback.

  2. The saved SVM artifact
     (results/final/final_model/mobilenetv2_convnext_polynomial_svm.joblib)
     is a **dictionary** — {"svm": <sklearn SVC>, "models": [...],
     "kernel": ..., "class_names": [...], ...} — not a bare SVM object.
     This wrapper unwraps it correctly (`bundle["svm"]`) before calling
     `.predict()` / `.decision_function()`.

This file does NOT modify src/predict.py or any other original project
file. It only imports the already-existing, working helper functions
(preprocessing + feature extractors) from src/predict.py and adds a
corrected SVM-loading/prediction step on top, plus model-availability
checks used by the API layer.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

from app.core.config import RESULTS_DIR, SRC_DIR

# Make the existing src/ package importable, exactly like app.py already does.
sys.path.insert(0, str(SRC_DIR))

import predict as _predict_module  # existing, untouched src/predict.py

CLASS_NAMES = _predict_module.CLASS_NAMES
CLASS_LABELS = _predict_module.CLASS_LABELS
CLASS_DESCRIPTIONS = _predict_module.CLASS_DESCRIPTIONS

MOBILENET_MODEL_PATH = _predict_module.MOBILENET_MODEL_PATH
CONVNEXT_MODEL_PATH = _predict_module.CONVNEXT_MODEL_PATH
SVM_MODEL_PATH = _predict_module.SVM_MODEL_PATH

_svm_lock = threading.Lock()
_svm_estimator = None  # the unwrapped sklearn SVC


class ModelNotAvailableError(RuntimeError):
    """Raised when a required trained-model artifact is missing on disk."""
    pass


def check_model_availability() -> dict:
    """
    Returns a dict describing which required model artifacts exist on
    disk right now, so the API/frontend can show an accurate status
    instead of failing opaquely mid-request.
    """
    missing = []
    if not MOBILENET_MODEL_PATH.exists():
        missing.append({
            "file": str(MOBILENET_MODEL_PATH.relative_to(RESULTS_DIR.parent)),
            "description": "MobileNetV2 feature-extractor model",
        })
    if not CONVNEXT_MODEL_PATH.exists():
        missing.append({
            "file": str(CONVNEXT_MODEL_PATH.relative_to(RESULTS_DIR.parent)),
            "description": "ConvNeXt feature-extractor model",
        })
    if not SVM_MODEL_PATH.exists():
        missing.append({
            "file": str(SVM_MODEL_PATH.relative_to(RESULTS_DIR.parent)),
            "description": "Final Polynomial-SVM classifier",
        })

    return {
        "ready": len(missing) == 0,
        "missing_files": missing,
        "download_instructions": (
            "Trained model files are hosted on Google Drive due to GitHub "
            "size limits (see README.md). Download them and place inside "
            "the corresponding results/<Model>/model/ folder, then restart "
            "the backend."
        ),
    }


def _load_svm_estimator():
    """
    Load results/final/final_model/mobilenetv2_convnext_polynomial_svm.joblib
    and return the underlying sklearn SVC, correctly unwrapping the
    dictionary bundle it is actually stored as.
    """
    global _svm_estimator
    if _svm_estimator is not None:
        return _svm_estimator

    with _svm_lock:
        if _svm_estimator is not None:
            return _svm_estimator

        if not SVM_MODEL_PATH.exists():
            raise ModelNotAvailableError(
                f"SVM model file not found: {SVM_MODEL_PATH}"
            )

        bundle = joblib.load(str(SVM_MODEL_PATH))

        if isinstance(bundle, dict):
            # Real saved format: {"svm": SVC, "models": [...], "kernel": ..., ...}
            estimator = bundle.get("svm")
            if estimator is None:
                raise ModelNotAvailableError(
                    "SVM bundle file did not contain a 'svm' key."
                )
        else:
            # Fallback in case a bare estimator is ever saved directly.
            estimator = bundle

        _svm_estimator = estimator
        return _svm_estimator


def preload_models():
    """Warm up all three model components at server startup."""
    availability = check_model_availability()
    if not availability["ready"]:
        print("\n[inference] WARNING — required model files are missing:")
        for m in availability["missing_files"]:
            print(f"    - {m['file']}  ({m['description']})")
        print(f"[inference] {availability['download_instructions']}\n")
        return availability

    _predict_module.load_mobilenet_extractor()
    _predict_module.load_convnext_extractor()
    _load_svm_estimator()
    print("[inference] All models loaded successfully.")
    return availability


def run_prediction(image_path: str) -> dict:
    """
    Run the real end-to-end pipeline:
       image -> MobileNetV2 (1280-d) + ConvNeXt (768-d)
             -> concat (2048-d) -> Polynomial SVM -> class + probabilities

    Raises ModelNotAvailableError with a clear message if any required
    trained-model file is missing.
    """
    availability = check_model_availability()
    if not availability["ready"]:
        names = ", ".join(m["description"] for m in availability["missing_files"])
        raise ModelNotAvailableError(
            f"Cannot run prediction — missing trained model file(s): {names}. "
            f"{availability['download_instructions']}"
        )

    t0 = time.time()
    image_array = _predict_module.preprocess_image(image_path)
    t_preprocess = time.time() - t0

    t0 = time.time()
    mob_ext = _predict_module.load_mobilenet_extractor()
    mob_feats = mob_ext(image_array, training=False).numpy()
    t_mobilenet = time.time() - t0

    t0 = time.time()
    cx_ext = _predict_module.load_convnext_extractor()
    cx_feats = cx_ext(image_array, training=False).numpy()
    t_convnext = time.time() - t0

    t0 = time.time()
    fused = np.concatenate([mob_feats, cx_feats], axis=1)  # (1, 2048)
    t_fusion = time.time() - t0

    t0 = time.time()
    svm = _load_svm_estimator()
    predicted_idx = int(svm.predict(fused)[0])
    decision_scores = svm.decision_function(fused)[0]
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    t_svm = time.time() - t0

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_LABELS[predicted_class],
        "description": CLASS_DESCRIPTIONS[predicted_class],
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 2),
        "all_scores": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "timing_ms": {
            "preprocess": round(t_preprocess * 1000, 1),
            "mobilenet_extraction": round(t_mobilenet * 1000, 1),
            "convnext_extraction": round(t_convnext * 1000, 1),
            "feature_fusion": round(t_fusion * 1000, 1),
            "svm_inference": round(t_svm * 1000, 1),
        },
    }
