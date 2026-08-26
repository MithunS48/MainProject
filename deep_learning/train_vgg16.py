import os
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "split_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "split_dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "split_dataset", "test")

MODEL_DIR = os.path.join(BASE_DIR, "deep_learning", "models")
RESULT_DIR = os.path.join(BASE_DIR, "deep_learning", "results", "vgg16")
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42


# ============================================================
# DATA
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

validation_generator = test_val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())

print("\nClasses:", CLASS_NAMES)
print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Test images:", test_generator.samples)


# ============================================================
# BUILD VGG16
# ============================================================

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# PARAMETERS
# ============================================================

total_params = model.count_params()

trainable_params = int(
    np.sum([
        np.prod(v.shape)
        for v in model.trainable_weights
    ])
)

print("\nTotal parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "vgg16_epoch_{epoch:02d}.keras"
)

best_model_path = os.path.join(
    MODEL_DIR,
    "vgg16_fish_disease_best.keras"
)

callbacks = [

    ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_freq="epoch",
        verbose=1
    ),

    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting VGG16 training...")

start_time = time.time()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

training_time = time.time() - start_time

print("\nTraining completed.")
print("Training time:", training_time, "seconds")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "vgg16_fish_disease.keras"
)

model.save(final_model_path)

print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("VGG16 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("VGG16 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating VGG16 on test set...")

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


# ============================================================
# PREDICTIONS
# ============================================================

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

print("\n================================")
print("VGG16 OVERALL RESULTS")
print("================================")

print("Accuracy :", test_accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print(
    "Training time:",
    training_time,
    "seconds"
)

print(
    "Total parameters:",
    total_params
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

print("\nClassification Report")
print(report)

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:
    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
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

plt.title("VGG16 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

per_class_results = {}

for class_name in CLASS_NAMES:

    per_class_results[class_name] = {

        "precision":
            report_dict[class_name]["precision"],

        "recall":
            report_dict[class_name]["recall"],

        "f1_score":
            report_dict[class_name]["f1-score"],

        "support":
            report_dict[class_name]["support"]
    }


with open(
    os.path.join(
        RESULT_DIR,
        "per_class_results.json"
    ),
    "w"
) as file:

    json.dump(
        per_class_results,
        file,
        indent=4
    )


# ============================================================
# FINAL RESULTS JSON
# ============================================================

results = {

    "model": "VGG16",

    "image_size": "128x128",

    "test_accuracy":
        float(test_accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "training_time_seconds":
        float(training_time),

    "total_parameters":
        int(total_params),

    "trainable_parameters":
        int(trainable_params),

    "classes":
        CLASS_NAMES,

    "training_images":
        train_generator.samples,

    "validation_images":
        validation_generator.samples,

    "test_images":
        test_generator.samples
}


with open(
    os.path.join(
        RESULT_DIR,
        "vgg16_results.json"
    ),
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\n================================")
print("VGG16 EXPERIMENT COMPLETED")
print("================================")

print(
    "\nResults saved in:",
    RESULT_DIR
)