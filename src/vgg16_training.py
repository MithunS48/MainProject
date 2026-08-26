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
VAL_DIR = PROJECT_ROOT / "split_dataset" / "validation"
TEST_DIR = PROJECT_ROOT / "split_dataset" / "test"

RESULTS_DIR = PROJECT_ROOT / "results" / "VGG16"

MODEL_DIR = RESULTS_DIR / "model"
PLOT_DIR = RESULTS_DIR / "plots"
REPORT_DIR = RESULTS_DIR / "reports"

for folder in [MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MAX_EPOCHS = 10

LEARNING_RATE = 0.0001

SEED = 42

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]

NUM_CLASSES = len(CLASS_NAMES)

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("VGG16 INDIVIDUAL CNN EXPERIMENT")
print("=" * 70)

print(f"TensorFlow: {tf.__version__}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {MAX_EPOCHS}")

print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")


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


# ============================================================
# PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# VGG16 BASE
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRETRAINED VGG16")
print("=" * 70)

base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pretrained convolutional layers.
# Only the new classification head will be trained.
base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="fish_image"
)

# VGG16 ImageNet preprocessing
x = tf.keras.applications.vgg16.preprocess_input(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = tf.keras.layers.Dropout(
    0.5,
    name="dropout"
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classification"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="VGG16_Fish_Disease"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# PARAMETERS
# ============================================================

total_params = model.count_params()

trainable_params = np.sum([
    np.prod(variable.shape)
    for variable in model.trainable_variables
])

non_trainable_params = (
    total_params - trainable_params
)

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)

print(f"Total parameters:      {total_params:,}")
print(f"Trainable parameters:  {int(trainable_params):,}")
print(f"Non-trainable params:  {int(non_trainable_params):,}")


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR / "vgg16_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
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


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING VGG16 TRAINING")
print("=" * 70)

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callbacks
)

training_seconds = time.time() - start_time
training_minutes = training_seconds / 60


print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Training time: "
    f"{training_minutes:.2f} minutes"
)


# ============================================================
# SAVE MODEL
# ============================================================

final_model_path = (
    MODEL_DIR / "vgg16_final.keras"
)

model.save(final_model_path)

print(
    f"\nModel saved to:\n"
    f"{final_model_path}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_dict = {
    key: [float(x) for x in values]
    for key, values in history.history.items()
}

with open(
    REPORT_DIR / "training_history.json",
    "w"
) as file:

    json.dump(
        history_dict,
        file,
        indent=4
    )


# ============================================================
# TRAINING ACCURACY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "VGG16 Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "accuracy_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# TRAINING LOSS PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "VGG16 Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "loss_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    true_labels = np.argmax(
        labels.numpy(),
        axis=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(true_labels)
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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


print("\n" + "=" * 70)
print("FINAL TEST METRICS")
print("=" * 70)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

with open(
    REPORT_DIR / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# PER-CLASS METRICS
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

rows = []

for class_name in CLASS_NAMES:

    rows.append({

        "Class": class_name,

        "Precision":
            report_dict[class_name]["precision"],

        "Recall":
            report_dict[class_name]["recall"],

        "F1":
            report_dict[class_name]["f1-score"],

        "Support":
            report_dict[class_name]["support"]
    })


per_class_df = pd.DataFrame(rows)

per_class_df.to_csv(
    REPORT_DIR / "per_class_metrics.csv",
    index=False
)


print("\nPer-class performance:")
print(
    per_class_df.to_string(
        index=False
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "VGG16 Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    NUM_CLASSES
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

threshold = cm.max() / 2

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white"
            if cm[i, j] > threshold
            else "black"
        )

plt.ylabel("True Class")
plt.xlabel("Predicted Class")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# BEST VALIDATION RESULTS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)


# ============================================================
# SAVE ALL METRICS
# ============================================================

summary = {

    "model": "VGG16",

    "classes": CLASS_NAMES,

    "image_size": [224, 224],

    "batch_size": BATCH_SIZE,

    "maximum_epochs": MAX_EPOCHS,

    "epochs_completed":
        len(history.history["loss"]),

    "learning_rate": LEARNING_RATE,

    "total_parameters":
        int(total_params),

    "trainable_parameters":
        int(trainable_params),

    "non_trainable_parameters":
        int(non_trainable_params),

    "training_time_seconds":
        float(training_seconds),

    "training_time_minutes":
        float(training_minutes),

    "best_validation_accuracy":
        float(best_val_accuracy),

    "best_validation_loss":
        float(best_val_loss),

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(accuracy),

    "test_precision":
        float(precision),

    "test_recall":
        float(recall),

    "test_f1":
        float(f1)
}


with open(
    REPORT_DIR / "metrics.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VGG16 EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Test precision: "
    f"{precision:.4f}"
)

print(
    f"Test recall: "
    f"{recall:.4f}"
)

print(
    f"Test F1: "
    f"{f1:.4f}"
)

print(
    f"\nResults saved at:\n"
    f"{RESULTS_DIR}"
)

print("\nDone.")