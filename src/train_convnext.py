import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "split_dataset" / "train"
VAL_DIR   = PROJECT_ROOT / "split_dataset" / "validation"
TEST_DIR  = PROJECT_ROOT / "split_dataset" / "test"

RESULTS_DIR = PROJECT_ROOT / "results" / "ConvNeXt"

MODEL_DIR  = RESULTS_DIR / "model"
PLOT_DIR   = RESULTS_DIR / "plots"
REPORT_DIR = RESULTS_DIR / "reports"

for folder in [MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
MAX_EPOCHS    = 20
LEARNING_RATE = 0.0001
SEED          = 42

CLASS_NAMES = ["EUS", "gill", "healthy", "red_spot"]
NUM_CLASSES = len(CLASS_NAMES)

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# INFO
# ============================================================

print("=" * 70)
print("ConvNeXtTiny INDIVIDUAL CNN EXPERIMENT")
print("=" * 70)
print(f"TensorFlow : {tf.__version__}")
print(f"Image size : {IMG_SIZE}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Max epochs : {MAX_EPOCHS}")
print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION LAYER
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")


# ============================================================
# ConvNeXtTiny BASE
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRETRAINED ConvNeXtTiny")
print("=" * 70)

base_model = tf.keras.applications.ConvNeXtTiny(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base — train head only first
base_model.trainable = False


# ============================================================
# MODEL
# NOTE: tf.cast cannot be used directly on a Keras tensor
# inside a Functional API model. Use Lambda layer instead.
# ConvNeXt expects float32 pixels in [0, 255].
# ============================================================

inputs = tf.keras.Input(shape=(224, 224, 3), name="fish_image")

x = data_augmentation(inputs)

x = tf.keras.layers.Lambda(
    lambda t: tf.cast(t, tf.float32),
    name="cast_float32"
)(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = tf.keras.layers.LayerNormalization(name="layer_norm")(x)

x = tf.keras.layers.Dropout(0.5, name="dropout")(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classification"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="ConvNeXtTiny_Fish_Disease"
)

model.summary()


# ============================================================
# PARAMETERS
# ============================================================

total_params         = model.count_params()
trainable_params     = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
non_trainable_params = total_params - trainable_params

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print(f"Total parameters      : {total_params:,}")
print(f"Trainable parameters  : {trainable_params:,}")
print(f"Non-trainable params  : {non_trainable_params:,}")


# ============================================================
# PHASE 1 — TRAIN HEAD ONLY
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

best_model_path = MODEL_DIR / "convnext_best.keras"

callbacks_phase1 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n" + "=" * 70)
print("PHASE 1 — TRAINING HEAD (base frozen)")
print("=" * 70)

start_time = time.time()

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callbacks_phase1
)

phase1_seconds = time.time() - start_time
print(f"\nPhase 1 completed in {phase1_seconds/60:.2f} minutes")


# ============================================================
# PHASE 2 — FINE-TUNING (unfreeze last 40 layers)
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 — FINE-TUNING (last 40 layers unfrozen)")
print("=" * 70)

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE / 10,
        weight_decay=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_phase2
)

phase2_seconds   = time.time() - start_time - phase1_seconds
training_seconds = time.time() - start_time
training_minutes = training_seconds / 60

print(f"\nPhase 2 completed in {phase2_seconds/60:.2f} minutes")
print(f"Total training time   : {training_minutes:.2f} minutes")


# ============================================================
# MERGE HISTORY
# ============================================================

merged_history = {}
for key in history1.history:
    merged_history[key] = (
        history1.history[key] + history2.history[key]
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "convnext_final.keras"
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_dict = {
    key: [float(x) for x in values]
    for key, values in merged_history.items()
}

with open(REPORT_DIR / "training_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)


# ============================================================
# ACCURACY CURVE
# ============================================================

phase1_epochs = len(history1.history["accuracy"])

plt.figure(figsize=(10, 6))
plt.plot(merged_history["accuracy"],     label="Training Accuracy")
plt.plot(merged_history["val_accuracy"], label="Validation Accuracy")
plt.axvline(
    x=phase1_epochs - 1,
    color="gray", linestyle="--",
    label="Fine-tuning start"
)
plt.title("ConvNeXtTiny Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "accuracy_curve.png", dpi=300)
plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(merged_history["loss"],     label="Training Loss")
plt.plot(merged_history["val_loss"], label="Validation Loss")
plt.axvline(
    x=phase1_epochs - 1,
    color="gray", linestyle="--",
    label="Fine-tuning start"
)
plt.title("ConvNeXtTiny Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "loss_curve.png", dpi=300)
plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


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
print(f"Accuracy  : {accuracy:.4f}")
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


# ============================================================
# PER-CLASS METRICS CSV
# ============================================================

report_dict = classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

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
print("\nPer-class performance:")
print(per_class_df.to_string(index=False))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

pd.DataFrame(
    cm, index=CLASS_NAMES, columns=CLASS_NAMES
).to_csv(REPORT_DIR / "confusion_matrix.csv")

plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("ConvNeXtTiny Confusion Matrix")
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


# ============================================================
# SAVE ALL METRICS
# ============================================================

best_val_accuracy = max(merged_history["val_accuracy"])
best_val_loss     = min(merged_history["val_loss"])

summary = {
    "model"                    : "ConvNeXtTiny",
    "classes"                  : CLASS_NAMES,
    "image_size"               : [224, 224],
    "batch_size"               : BATCH_SIZE,
    "maximum_epochs"           : MAX_EPOCHS,
    "epochs_completed"         : len(merged_history["loss"]),
    "learning_rate"            : LEARNING_RATE,
    "fine_tuning"              : True,
    "fine_tuning_layers"       : 40,
    "total_parameters"         : int(total_params),
    "trainable_parameters"     : int(trainable_params),
    "non_trainable_parameters" : int(non_trainable_params),
    "training_time_seconds"    : float(training_seconds),
    "training_time_minutes"    : float(training_minutes),
    "best_validation_accuracy" : float(best_val_accuracy),
    "best_validation_loss"     : float(best_val_loss),
    "test_loss"                : float(test_loss),
    "test_accuracy"            : float(accuracy),
    "test_precision"           : float(precision),
    "test_recall"              : float(recall),
    "test_f1"                  : float(f1)
}

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump(summary, f, indent=4)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ConvNeXtTiny EXPERIMENT COMPLETE")
print("=" * 70)
print(f"Best validation accuracy : {best_val_accuracy * 100:.2f}%")
print(f"Test accuracy            : {accuracy * 100:.2f}%")
print(f"Test precision           : {precision:.4f}")
print(f"Test recall              : {recall:.4f}")
print(f"Test F1                  : {f1:.4f}")
print(f"Training time            : {training_minutes:.2f} minutes")
print(f"\nResults saved at:\n{RESULTS_DIR}")
print("\nDone.")
