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


# ============================================================
# GRAD-CAM
# ============================================================

import base64
import cv2 as _cv2
import numpy as _np_gc
import matplotlib.cm as _cm
import tensorflow as _tf_gc


def run_gradcam(image_path: str) -> dict:
    """
    Compute a Grad-CAM heatmap using MobileNetV2's last conv layer.

    Returns:
        heatmap_b64    : base64 PNG of pure heatmap (jet colormap)
        overlay_b64    : base64 PNG of heatmap blended on original image
        predicted_class: class predicted by MobileNetV2
    """
    if not MOBILENET_MODEL_PATH.exists():
        raise ModelNotAvailableError(
            "MobileNetV2 model not available for Grad-CAM."
        )

    import predict as _pm

    mob_full = _tf_gc.keras.models.load_model(
        str(MOBILENET_MODEL_PATH), safe_mode=False
    )
    mob_base = mob_full.get_layer("mobilenetv2_1.00_224")

    conv_model = _tf_gc.keras.Model(
        inputs=mob_base.inputs,
        outputs=mob_base.get_layer("out_relu").output
    )

    image_array = _pm.preprocess_image(image_path)

    aug_out = mob_full.get_layer("data_augmentation")(
        _tf_gc.cast(image_array, _tf_gc.float32), training=False
    )
    preprocessed = _tf_gc.keras.applications.mobilenet_v2.preprocess_input(
        _tf_gc.identity(aug_out)
    )

    preds      = mob_full(_tf_gc.cast(image_array, _tf_gc.float32), training=False)
    pred_idx   = int(_tf_gc.argmax(preds[0]).numpy())
    pred_class = CLASS_NAMES[pred_idx]

    with _tf_gc.GradientTape() as tape:
        conv_out = conv_model(preprocessed, training=False)
        tape.watch(conv_out)
        preds2 = mob_full(_tf_gc.cast(image_array, _tf_gc.float32), training=False)
        loss   = preds2[:, pred_idx]

    grads = tape.gradient(loss, conv_out)

    if grads is None:
        with _tf_gc.GradientTape() as tape2:
            inp = _tf_gc.cast(image_array, _tf_gc.float32)
            tape2.watch(inp)
            p = mob_full(inp, training=False)
            l = p[:, pred_idx]
        g = tape2.gradient(l, inp)
        heatmap = _tf_gc.reduce_mean(_tf_gc.abs(g[0]), axis=-1).numpy()
    else:
        pooled  = _tf_gc.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = (conv_out[0] @ pooled[..., _tf_gc.newaxis]).numpy().squeeze()

    heatmap = _np_gc.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    orig = _cv2.imread(str(image_path))
    orig = _cv2.cvtColor(orig, _cv2.COLOR_BGR2RGB)
    orig = _cv2.resize(orig, (224, 224), interpolation=_cv2.INTER_AREA)

    hm_resized = _cv2.resize(heatmap, (224, 224))
    hm_colored = (_cm.jet(hm_resized)[:, :, :3] * 255).astype(_np_gc.uint8)
    overlay    = _cv2.addWeighted(orig, 0.55, hm_colored, 0.45, 0)

    def to_b64(arr_rgb):
        bgr = _cv2.cvtColor(arr_rgb, _cv2.COLOR_RGB2BGR)
        _, buf = _cv2.imencode(".png", bgr)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "heatmap_b64"     : to_b64(hm_colored),
        "overlay_b64"     : to_b64(overlay),
        "predicted_class" : pred_class,
    }
