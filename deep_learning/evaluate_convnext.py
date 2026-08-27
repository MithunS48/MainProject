import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "split_dataset",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "deep_learning",
    "models",
    "convnext_248_v2_best.keras"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "deep_learning",
    "results",
    "convnext_248_v2"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (248, 248)
BATCH_SIZE = 16


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n================================")
print("CONVNEXTTINY TEST EVALUATION")
print("================================")

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

CLASS_NAMES = list(
    test_generator.class_indices.keys()
)

print("\nClasses:", CLASS_NAMES)
print("Test images:", test_generator.samples)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# TEST ACCURACY
# ============================================================

print("\n================================")
print("TEST EVALUATION")
print("================================")

test_generator.reset()

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_generator.classes


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================================")
print("OVERALL RESULTS")
print("================================")

print(
    "Validation Accuracy : 91.89%"
)

print(
    "Test Accuracy       :",
    f"{test_accuracy * 100:.2f}%"
)

print(
    "Precision           :",
    f"{precision * 100:.2f}%"
)

print(
    "Recall              :",
    f"{recall * 100:.2f}%"
)

print(
    "F1 Score            :",
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\n================================")
print("PER-CLASS PERFORMANCE")
print("================================")

print(report)


report_path = os.path.join(
    RESULT_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

cm_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    "ConvNeXtTiny 248x248 Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "test_results.txt"
)

with open(summary_path, "w") as f:

    f.write("ConvNeXtTiny 248x248 Results\n")
    f.write("============================\n\n")

    f.write("Validation Accuracy: 91.89%\n")
    f.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )
    f.write(
        f"Precision: {precision * 100:.2f}%\n"
    )
    f.write(
        f"Recall: {recall * 100:.2f}%\n"
    )
    f.write(
        f"F1 Score: {f1 * 100:.2f}%\n"
    )


# ============================================================
# DONE
# ============================================================

print("\n================================")
print("EVALUATION COMPLETED")
print("================================")

print("\nFiles created:")

print(report_path)
print(cm_path)
print(summary_path)

print("\nYou can now share these results with your teammates.")