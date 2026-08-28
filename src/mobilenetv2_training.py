import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\MainProject"

TRAIN_DIR = os.path.join(
    PROJECT_DIR, "split_dataset", "train"
)

VAL_DIR = os.path.join(
    PROJECT_DIR, "split_dataset", "validation"
)

TEST_DIR = os.path.join(
    PROJECT_DIR, "split_dataset", "test"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR, "results", "MobileNetV2"
)

MODEL_DIR = os.path.join(
    RESULT_DIR, "model"
)

PLOT_DIR = os.path.join(
    RESULT_DIR, "plots"
)

REPORT_DIR = os.path.join(
    RESULT_DIR, "reports"
)


# Create result directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_CLASSES = 4

# Maximum number of epochs
EPOCHS = 20

# 1e-4 = 0.0001
LEARNING_RATE = 1e-4

# Stop if validation accuracy does not improve
# for 3 consecutive epochs
PATIENCE = 3

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("MOBILENETV2 TRAINING")
print("=" * 70)

print("Project directory :", PROJECT_DIR)
print("Train directory   :", TRAIN_DIR)
print("Validation dir    :", VAL_DIR)
print("Test directory    :", TEST_DIR)

print()
print("Image size        :", IMG_SIZE)
print("Batch size        :", BATCH_SIZE)
print("Maximum epochs    :", EPOCHS)
print("Learning rate     :", LEARNING_RATE)
print("Early stopping    :", PATIENCE)

print()
print("Classes:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"{i}: {class_name}")

print("=" * 70)


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"\nDataset directory not found:\n{directory}"
        )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nApplying pixel normalization...")
print("Original range : [0, 255]")
print("Normalized     : [0, 1]")


normalization_layer = layers.Rescaling(
    1.0 / 255.0
)


train_ds = train_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


val_ds = val_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


test_ds = test_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# PREFETCH
# ============================================================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MOBILENETV2
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING MOBILENETV2")
print("=" * 70)


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze pretrained ImageNet layers
base_model.trainable = False


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)


x = base_model(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.Dropout(
    0.5
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:")

model.summary()


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = model.count_params()

trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.trainable_variables
])


print("\n")
print("=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)

print(
    f"Total parameters     : {total_params:,}"
)

print(
    f"Trainable parameters : {trainable_params:,}"
)

print("=" * 70)


# ============================================================
# MODEL PATHS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_final.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


csv_logger = CSVLogger(
    os.path.join(
        RESULT_DIR,
        "training_log.csv"
    )
)


callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
    csv_logger
]


# ============================================================
# START TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING MOBILENETV2 TRAINING")
print("=" * 70)

print()
print("Maximum epochs :", EPOCHS)
print("Learning rate  :", LEARNING_RATE)
print("Batch size     :", BATCH_SIZE)
print("Early stopping :", PATIENCE)

print()
print("Training started...")
print("=" * 70)


start_time = time.time()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


end_time = time.time()


# ============================================================
# TRAINING TIME
# ============================================================

training_seconds = end_time - start_time

training_minutes = training_seconds / 60

training_hours = training_minutes / 60


print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Training time: "
    f"{training_minutes:.2f} minutes"
)

print(
    f"Training time: "
    f"{training_hours:.2f} hours"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)


print("\nFinal model saved to:")

print(final_model_path)


# ============================================================
# BEST VALIDATION ACCURACY
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)


best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)


print("\n")
print("=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(
    f"Best epoch              : {best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# TRAINING ACCURACY PLOT
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

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "accuracy_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# TRAINING LOSS PLOT
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

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "MobileNetV2 Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "loss_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best MobileNetV2 model...")

best_model = tf.keras.models.load_model(
    best_model_path
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)


test_loss, keras_test_accuracy = (
    best_model.evaluate(
        test_ds,
        verbose=1
    )
)


print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(
    f"{keras_test_accuracy * 100:.2f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n")
print("Generating predictions...")


y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = best_model.predict(
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


    y_true.extend(
        true_labels
    )

    y_pred.extend(
        predicted_labels
    )


y_true = np.array(y_true)

y_pred = np.array(y_pred)


# ============================================================
# FINAL METRICS
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


print("\n")
print("=" * 70)
print("FINAL TEST METRICS")
print("=" * 70)

print(
    f"Accuracy :  {accuracy:.4f}"
)

print(
    f"Precision:  {precision:.4f}"
)

print(
    f"Recall   :  {recall:.4f}"
)

print(
    f"F1-score :  {f1:.4f}"
)


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


print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


report_path = os.path.join(
    REPORT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(report)


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


print("\n")
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)


for class_name in CLASS_NAMES:

    class_precision = (
        report_dict[class_name]["precision"]
    )

    class_recall = (
        report_dict[class_name]["recall"]
    )

    class_f1 = (
        report_dict[class_name]["f1-score"]
    )

    support = (
        report_dict[class_name]["support"]
    )

    print(
        f"{class_name:10s} "
        f"Precision: {class_precision:.4f}   "
        f"Recall: {class_recall:.4f}   "
        f"F1: {class_f1:.4f}   "
        f"Support: {int(support)}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "MobileNetV2 Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    len(CLASS_NAMES)
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


threshold = cm.max() / 2.0


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.ylabel(
    "True Label"
)


plt.xlabel(
    "Predicted Label"
)


plt.tight_layout()


confusion_matrix_path = os.path.join(
    PLOT_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=300
)


plt.close()


# ============================================================
# SAVE RESULTS AS JSON
# ============================================================

results = {

    "model": "MobileNetV2",

    "classes": CLASS_NAMES,

    "image_size": [
        IMG_SIZE[0],
        IMG_SIZE[1]
    ],

    "batch_size": BATCH_SIZE,

    "maximum_epochs": EPOCHS,

    "best_epoch": int(best_epoch),

    "learning_rate": LEARNING_RATE,

    "patience": PATIENCE,

    "best_validation_accuracy":
        float(best_val_accuracy),

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(accuracy),

    "test_precision":
        float(precision),

    "test_recall":
        float(recall),

    "test_f1":
        float(f1),

    "training_time_minutes":
        float(training_minutes),

    "training_time_hours":
        float(training_hours),

    "total_parameters":
        int(total_params),

    "trainable_parameters":
        int(trainable_params)
}


results_path = os.path.join(
    RESULT_DIR,
    "mobilenetv2_results.json"
)


with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MOBILENETV2 EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy            : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Test precision           : "
    f"{precision:.4f}"
)

print(
    f"Test recall              : "
    f"{recall:.4f}"
)

print(
    f"Test F1                  : "
    f"{f1:.4f}"
)

print(
    f"Training time            : "
    f"{training_minutes:.2f} minutes"
)

print(
    f"Total parameters         : "
    f"{total_params:,}"
)

print("\nResults directory:")
print(RESULT_DIR)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\nPlots:")
print(PLOT_DIR)

print("\nReports:")
print(REPORT_DIR)

print("=" * 70)