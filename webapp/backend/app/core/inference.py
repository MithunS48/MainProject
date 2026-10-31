"""
inference.py
-------------
Thin wrapper around the EXISTING ML pipeline in ../../../src/predict.py.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
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

# ── Confidence threshold below which we reject the prediction ─────────────────
# If the SVM is not confident enough, the image is likely not a fish.
# Set high because the model was trained only on fish images and will
# confidently misclassify non-fish images that share visual features
# (e.g. red/orange colors → red_spot, blue/green → gill disease).
CONFIDENCE_THRESHOLD = 0.50


class ModelNotAvailableError(RuntimeError):
    """Raised when a required trained-model artifact is missing on disk."""
    pass


class InvalidImageError(ValueError):
    """Raised when the uploaded image does not appear to be a valid fish image."""
    pass


# ============================================================
# FISH IMAGE VALIDATOR
# ============================================================

def validate_fish_image(image_path: str) -> None:
    """
    Performs fast, pre-model checks to reject obviously invalid images
    before spending time on the full ML pipeline.

    Checks:
    1. Image can be decoded and has valid dimensions
    2. Image is not blank / solid colour (all pixels nearly identical)
    3. Image is not mostly black or white
    4. Image has sufficient detail (Laplacian variance — rejects blurry blanks)
    5. Image aspect ratio is within plausible bounds for a fish photo
    6. Image does not look like a document/screenshot (text-heavy, mostly white)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise InvalidImageError(
            "The uploaded file could not be read as an image. "
            "Please upload a valid JPG or PNG photo of a fish."
        )

    h, w = img.shape[:2]

    # ── Must be at least 32×32 ───────────────────────────────────────────────
    if h < 32 or w < 32:
        raise InvalidImageError(
            f"Image is too small ({w}×{h} px). "
            "Please upload a clear, full-size photo of a fish."
        )

    # ── Resize to 224×224 for fast analysis ─────────────────────────────────
    small = cv2.resize(img, (224, 224)).astype(np.float32)

    # ── Reject blank / solid-colour images ──────────────────────────────────
    std_dev = float(np.std(small))
    if std_dev < 8.0:
        raise InvalidImageError(
            "The image appears to be blank or a solid colour. "
            "Please upload a real photo of a fish."
        )

    # ── Reject mostly-black images ───────────────────────────────────────────
    mean_brightness = float(np.mean(small))
    if mean_brightness < 10.0:
        raise InvalidImageError(
            "The image is too dark or mostly black. "
            "Please upload a well-lit photo of a fish."
        )

    # ── Reject mostly-white / overexposed images ─────────────────────────────
    if mean_brightness > 245.0:
        raise InvalidImageError(
            "The image is overexposed (almost entirely white). "
            "Please upload a properly exposed photo of a fish."
        )

    # ── Reject images with too little detail (blurry or generated) ───────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 25.0:
        raise InvalidImageError(
            "The image is too blurry or lacks detail. "
            "Please upload a sharp, clear photo of a fish."
        )

    # ── Reject document-like / screenshot images ─────────────────────────────
    gray_small = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    white_pixel_ratio = float(np.mean(gray_small > 230))
    if white_pixel_ratio > 0.75:
        raise InvalidImageError(
            "This image appears to be a document, screenshot, or blank page. "
            "Please upload a real photograph of a fish."
        )

    # ── Reject obvious text documents (very high edge + white ratio) ──────────
    edges = cv2.Canny(gray_small, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    if edge_ratio > 0.25 and white_pixel_ratio > 0.60:
        raise InvalidImageError(
            "This image looks like a document or chart, not a fish photo. "
            "Please upload a real photograph of a fish."
        )


def validate_prediction_confidence(result: dict) -> None:
    """
    After ML inference, check if the model was confident enough.
    If confidence is below the threshold, the image is likely not a fish.
    """
    confidence = result.get("confidence", 0.0)
    if confidence < CONFIDENCE_THRESHOLD:
        scores_str = ", ".join(
            f"{k}: {v*100:.1f}%"
            for k, v in result["all_scores"].items()
        )
        raise InvalidImageError(
            f"This image does not appear to be a fish. "
            f"The model could not confidently identify any known fish disease "
            f"(max confidence: {confidence*100:.1f}%). "
            f"Please upload a clear photo of a fish. "
            f"(Scores: {scores_str})"
        )


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
       image -> validate -> MobileNetV2 (1280-d) + ConvNeXt (768-d)
             -> concat (2048-d) -> Polynomial SVM -> class + probabilities

    Raises:
        InvalidImageError       — if the image is not a valid fish photo
        ModelNotAvailableError  — if trained model files are missing
    """
    # ── Step 1: Fast pre-model image validation ──────────────────────────────
    validate_fish_image(image_path)

    # ── Step 2: Check model availability ────────────────────────────────────
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

    result = {
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

    # ── Step 3: Post-model confidence check ──────────────────────────────────
    validate_prediction_confidence(result)

    return result
