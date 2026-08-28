"""
evaluate_convnext.py
---------------------
Evaluates the saved convnext_best.keras on the test set
and generates all missing outputs:
  reports/classification_report.txt
  reports/per_class_metrics.csv
  reports/confusion_matrix.csv
  reports/metrics.json
  plots/confusion_matrix.png
"""

import json
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR    = PROJECT_ROOT / "split_dataset" / "test"
RESULTS_DIR = PROJECT_ROOT / "results" / "ConvNeXt"
MODEL_PATH  = RESULTS_DIR / "model" / "convnext_best.keras"
PLOT_DIR    = RESULTS_DIR / "plots"
REPORT_DIR  = RESULTS_DIR / "reports"

for folder in [PLOT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASS_NAMES = ["EUS", "gill", "healthy", "red_spot"]
NUM_CLASSES = len(CLASS_NAMES)
SEED        = 42

tf.keras.utils.set_random_seed(SEED)

print("=" * 70)
print("ConvNeXtTiny — EVALUATION")
print("=" * 70)


# ============================================================
# REBUILD MODEL AND LOAD WEIGHTS
# ConvNeXt .keras may have version mismatch — use weights extraction
# ============================================================

print("\nLoading ConvNeXt model...")

try:
    model = tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False)
    print("  Loaded directly from .keras file")

except Exception as e:
    print(f"  Direct load failed: {str(e)[:100]}")
    print("  Extracting weights from .keras archive...")

    weights_path = RESULTS_DIR / "model" / "extracted" / "model.weights.h5"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(MODEL_PATH), "r") as z:
        z.extract("model.weights.h5", str(weights_path.parent))

    print("  Rebuilding ConvNeXt architecture...")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    base_model = tf.keras.applications.ConvNeXtTiny(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    inputs = tf.keras.Input(shape=(224, 224, 3), name="fish_image")
    x = data_augmentation(inputs)
    x = tf.keras.layers.Lambda(
        lambda t: tf.cast(t, tf.float32),
        name="cast_float32"
    )(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.LayerNormalization(name="layer_norm")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ConvNeXtTiny_Fish_Disease")
    model.load_weights(str(weights_path))
    print("  Weights loaded successfully via extraction")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
).prefetch(tf.data.AUTOTUNE)


# ============================================================
# EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 70)
print("FINAL TEST METRICS")
print("=" * 70)
print(f"Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report)

with open(REPORT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

report_dict = classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# ============================================================
# PER-CLASS METRICS CSV
# ============================================================

rows = []
for cls in CLASS_NAMES:
    rows.append({
        "Class"    : cls,
        "Precision": report_dict[cls]["precision"],
        "Recall"   : report_dict[cls]["recall"],
        "F1"       : report_dict[cls]["f1-score"],
        "Support"  : report_dict[cls]["support"]
    })

per_class_df = pd.DataFrame(rows)
per_class_df.to_csv(REPORT_DIR / "per_class_metrics.csv", index=False)
print("Per-class performance:")
print(per_class_df.to_string(index=False))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)

plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation="nearest")
plt.title(f"ConvNeXtTiny Confusion Matrix\nTest Accuracy: {accuracy*100:.2f}%")
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
plt.yticks(tick_marks, CLASS_NAMES)
threshold = cm.max() / 2
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 color="white" if cm[i, j] > threshold else "black")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=300)
plt.close()
print(f"Saved: {PLOT_DIR / 'confusion_matrix.png'}")


# ============================================================
# SAVE METRICS JSON
# ============================================================

summary = {
    "model"          : "ConvNeXtTiny",
    "classes"        : CLASS_NAMES,
    "image_size"     : [224, 224],
    "batch_size"     : BATCH_SIZE,
    "note"           : "Evaluated from best checkpoint (training stopped early)",
    "test_loss"      : float(test_loss),
    "test_accuracy"  : float(accuracy),
    "test_precision" : float(precision),
    "test_recall"    : float(recall),
    "test_f1"        : float(f1)
}

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump(summary, f, indent=4)
print(f"Saved: {REPORT_DIR / 'metrics.json'}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ConvNeXtTiny EVALUATION COMPLETE")
print("=" * 70)
print(f"Test accuracy  : {accuracy*100:.2f}%")
print(f"Test F1-score  : {f1:.4f}")
print(f"Results saved  : {RESULTS_DIR}")
