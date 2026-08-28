"""
predict.py
-----------
Inference pipeline for a single fish image.

Final model: MobileNetV2 + ConvNeXt features → Polynomial SVM
Accuracy: 98.29%  |  AUC: 0.9989

Usage:
    python src/predict.py --image path/to/fish.jpg
"""

import argparse
import zipfile
import numpy as np
import joblib
import cv2
import tensorflow as tf
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

MOBILENET_MODEL_PATH = RESULTS_DIR / "MobileNetV2" / "model" / "mobilenetv2_best.keras"
CONVNEXT_MODEL_PATH  = RESULTS_DIR / "ConvNeXt"    / "model" / "convnext_best.keras"
CONVNEXT_WEIGHTS     = RESULTS_DIR / "ConvNeXt"    / "model" / "extracted" / "model.weights.h5"
SVM_MODEL_PATH       = RESULTS_DIR / "final" / "final_model" / "mobilenetv2_convnext_polynomial_svm.joblib"

CLASS_NAMES = ["EUS", "gill", "healthy", "red_spot"]
CLASS_LABELS = {
    "EUS"      : "EUS (Epizootic Ulcerative Syndrome)",
    "gill"     : "Gill Disease",
    "healthy"  : "Healthy",
    "red_spot" : "Red Spot Disease",
}
CLASS_DESCRIPTIONS = {
    "EUS"      : "Epizootic Ulcerative Syndrome — a serious fungal disease causing deep ulcers.",
    "gill"     : "Gill Disease — affects the gills causing respiratory distress.",
    "healthy"  : "No disease detected. The fish appears healthy.",
    "red_spot" : "Red Spot Disease — bacterial infection causing hemorrhagic spots.",
}

IMG_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# LOAD MODELS (cached after first call)
# ============================================================

_mobilenet_extractor = None
_convnext_extractor  = None
_svm_model           = None


def load_mobilenet_extractor():
    global _mobilenet_extractor
    if _mobilenet_extractor is None:
        print("Loading MobileNetV2 feature extractor...")
        full_model = tf.keras.models.load_model(
            str(MOBILENET_MODEL_PATH), safe_mode=False
        )
        _mobilenet_extractor = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer("global_average_pooling").output,
            name="MobileNetV2_Extractor"
        )
        print(f"  MobileNetV2 loaded — feature dim: {_mobilenet_extractor.output.shape[-1]}")
    return _mobilenet_extractor


def load_convnext_extractor():
    global _convnext_extractor
    if _convnext_extractor is None:
        print("Loading ConvNeXt feature extractor...")

        # Extract weights if not already done
        if not CONVNEXT_WEIGHTS.exists():
            print("  Extracting ConvNeXt weights...")
            CONVNEXT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(CONVNEXT_MODEL_PATH), "r") as z:
                z.extract("model.weights.h5", str(CONVNEXT_WEIGHTS.parent))

        # Rebuild architecture
        data_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ], name="data_augmentation")

        base = tf.keras.applications.ConvNeXtTiny(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )

        inputs = tf.keras.Input(shape=(224, 224, 3), name="fish_image")
        x = data_aug(inputs)
        x = tf.keras.layers.Lambda(
            lambda t: tf.cast(t, tf.float32), name="cast_float32"
        )(x)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
        x = tf.keras.layers.LayerNormalization(name="layer_norm")(x)
        x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
        out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

        full_model = tf.keras.Model(inputs=inputs, outputs=out)
        full_model.load_weights(str(CONVNEXT_WEIGHTS))

        _convnext_extractor = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer("global_average_pooling").output,
            name="ConvNeXt_Extractor"
        )
        print(f"  ConvNeXt loaded — feature dim: {_convnext_extractor.output.shape[-1]}")
    return _convnext_extractor


def load_svm():
    global _svm_model
    if _svm_model is None:
        print("Loading SVM classifier...")
        _svm_model = joblib.load(str(SVM_MODEL_PATH))
        print("  SVM loaded")
    return _svm_model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image.
    Returns float32 array of shape (1, 224, 224, 3).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_array: np.ndarray) -> np.ndarray:
    """
    Extract and fuse features from MobileNetV2 + ConvNeXt.
    Returns fused feature vector of shape (1, 2048).
    """
    mob_ext = load_mobilenet_extractor()
    cx_ext  = load_convnext_extractor()

    mob_feats = mob_ext(image_array, training=False).numpy()  # (1, 1280)
    cx_feats  = cx_ext(image_array,  training=False).numpy()  # (1, 768)

    fused = np.concatenate([mob_feats, cx_feats], axis=1)     # (1, 2048)
    return fused


# ============================================================
# PREDICT
# ============================================================

def predict(image_path: str) -> dict:
    """
    Run full inference pipeline on a single image.

    Returns:
        dict with predicted_class, label, description,
        confidence_scores per class
    """
    # Load and preprocess
    image_array = preprocess_image(image_path)

    # Extract features
    features = extract_features(image_array)

    # SVM prediction
    svm = load_svm()
    predicted_idx    = svm.predict(features)[0]
    decision_scores  = svm.decision_function(features)[0]

    # Convert decision scores to pseudo-probabilities via softmax
    exp_scores   = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(probabilities[predicted_idx])

    return {
        "predicted_class"  : predicted_class,
        "predicted_label"  : CLASS_LABELS[predicted_class],
        "description"      : CLASS_DESCRIPTIONS[predicted_class],
        "confidence"       : round(confidence, 4),
        "confidence_pct"   : round(confidence * 100, 2),
        "all_scores"       : {
            CLASS_NAMES[i]: round(float(probabilities[i]), 4)
            for i in range(NUM_CLASSES)
        }
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fish Disease Classifier — Predict from image"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to fish image (jpg/png)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("FISH DISEASE CLASSIFIER")
    print("=" * 55)
    print(f"Image: {args.image}")

    result = predict(args.image)

    print("\n" + "=" * 55)
    print("PREDICTION RESULT")
    print("=" * 55)
    print(f"Class       : {result['predicted_class']}")
    print(f"Label       : {result['predicted_label']}")
    print(f"Confidence  : {result['confidence_pct']}%")
    print(f"Description : {result['description']}")
    print("\nAll class scores:")
    for cls, score in result["all_scores"].items():
        bar = "█" * int(score * 30)
        print(f"  {cls:<12} {score:.4f}  {bar}")
