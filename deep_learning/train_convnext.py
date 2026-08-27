import os
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

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

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(BASE_DIR, "split_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "split_dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "split_dataset", "test")

MODEL_DIR = os.path.join(
    BASE_DIR,
    "deep_learning",
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "deep_learning",
    "results",
    "convnext_248_v2"
)

CHECKPOINT_DIR = os.path.join(
    RESULT_DIR,
    "checkpoints"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# PROJECT SETTINGS
# ============================================================

IMG_SIZE = (248, 248)
BATCH_SIZE = 16
TOTAL_EPOCHS = 10

# Fine-tuning starts after Epoch 5
FINE_TUNE_START_EPOCH = 5

SEED = 42


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\n================================")
print("CONVNEXTTINY 248x248 V3")
print("================================")

print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Total epochs:", TOTAL_EPOCHS)
print("Fine-tuning starts at Epoch 6")


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.08,

    height_shift_range=0.08,

    zoom_range=0.10,

    horizontal_flip=True

)

val_test_datagen = ImageDataGenerator()


# ============================================================
# DATA GENERATORS
# ============================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=SEED

)


validation_generator = val_test_datagen.flow_from_directory(

    VAL_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)


test_generator = val_test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)


# ============================================================
# DATASET INFORMATION
# ============================================================

NUM_CLASSES = train_generator.num_classes

CLASS_NAMES = list(
    train_generator.class_indices.keys()
)

print("\nClasses:", CLASS_NAMES)

print(
    "Training images:",
    train_generator.samples
)

print(
    "Validation images:",
    validation_generator.samples
)

print(
    "Test images:",
    test_generator.samples
)


# ============================================================
# MODEL PATHS
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "convnext_248_v2_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "convnext_248_v2_fish_disease.keras"
)


# ============================================================
# FIND LATEST CHECKPOINT
# ============================================================

def find_latest_checkpoint():

    checkpoints = []

    for filename in os.listdir(CHECKPOINT_DIR):

        if (
            filename.startswith("convnext_248_v2_epoch_")
            and filename.endswith(".keras")
        ):

            checkpoints.append(filename)

    if not checkpoints:

        return None, 0

    checkpoints.sort()

    latest = checkpoints[-1]

    path = os.path.join(
        CHECKPOINT_DIR,
        latest
    )

    epoch_string = latest.replace(
        "convnext_248_v2_epoch_",
        ""
    ).replace(
        ".keras",
        ""
    )

    completed_epoch = int(epoch_string)

    return path, completed_epoch


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nLoading ConvNeXtTiny ImageNet weights...")

    base_model = ConvNeXtTiny(

        weights="imagenet",

        include_top=False,

        include_preprocessing=True,

        input_shape=(248, 248, 3)

    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(248, 248, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    return model


# ============================================================
# COMPILE FROZEN MODEL
# ============================================================

def compile_frozen_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )


# ============================================================
# ENABLE FINE-TUNING
# ============================================================

def enable_fine_tuning(model):

    print("\n================================")
    print("ENABLING FINE-TUNING")
    print("================================")

    base_model = model.layers[1]

    base_model.trainable = True

    # Fine-tune approximately the last 30%
    fine_tune_from = int(
        len(base_model.layers) * 0.70
    )

    for layer in base_model.layers[:fine_tune_from]:

        layer.trainable = False

    for layer in base_model.layers[fine_tune_from:]:

        layer.trainable = True

    print(
        "Fine-tuning from layer:",
        fine_tune_from,
        "of",
        len(base_model.layers)
    )

    trainable_count = sum(
        np.prod(v.shape)
        for v in model.trainable_weights
    )

    print(
        "Fine-tuning trainable parameters:",
        trainable_count
    )

    # IMPORTANT:
    # Recompile AFTER changing trainable layers.
    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ============================================================
# EPOCH CHECKPOINT CALLBACK
# ============================================================

class EpochCheckpoint(
    tf.keras.callbacks.Callback
):

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        epoch_number = epoch + 1

        checkpoint_path = os.path.join(

            CHECKPOINT_DIR,

            f"convnext_248_v2_epoch_{epoch_number:02d}.keras"

        )

        self.model.save(
            checkpoint_path
        )

        print(
            f"\nCheckpoint saved: {checkpoint_path}"
        )


# ============================================================
# CREATE / LOAD MODEL
# ============================================================

latest_checkpoint, completed_epoch = (
    find_latest_checkpoint()
)


if latest_checkpoint is not None:

    print("\n================================")
    print("CHECKPOINT FOUND")
    print("================================")

    print(
        "Latest checkpoint:",
        latest_checkpoint
    )

    print(
        "Completed epochs:",
        completed_epoch
    )

    print(
        "Remaining epochs:",
        TOTAL_EPOCHS - completed_epoch
    )

    print(
        "\nLoading checkpoint..."
    )

    model = tf.keras.models.load_model(
        latest_checkpoint
    )

else:

    print("\nNo previous checkpoint found.")

    print(
        "Starting from Epoch 1."
    )

    model = build_model()

    compile_frozen_model(model)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_params = model.count_params()

trainable_params = sum(

    np.prod(v.shape)

    for v in model.trainable_weights

)

print("\n================================")
print("MODEL INFORMATION")
print("================================")

print(
    "Total parameters:",
    total_params
)

print(
    "Trainable parameters:",
    trainable_params
)


# ============================================================
# TRAINING
# ============================================================

start_time = time.time()


if completed_epoch < FINE_TUNE_START_EPOCH:

    # ========================================================
    # STAGE 1
    # FROZEN CONVNEXT
    # ========================================================

    stage1_end = FINE_TUNE_START_EPOCH

    print("\n================================")
    print("FROZEN TRAINING")
    print("================================")

    print(
        "Training until Epoch:",
        stage1_end
    )

    checkpoint_callback = EpochCheckpoint()

    best_checkpoint = ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

    )

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=1,

        min_lr=1e-7,

        verbose=1

    )

    model.fit(

        train_generator,

        validation_data=validation_generator,

        initial_epoch=completed_epoch,

        epochs=stage1_end,

        callbacks=[

            checkpoint_callback,

            best_checkpoint,

            reduce_lr

        ]

    )

    completed_epoch = stage1_end


# ============================================================
# STAGE 2
# FINE-TUNING
# ============================================================

if completed_epoch < TOTAL_EPOCHS:

    print("\n================================")
    print("STARTING FINE-TUNING")
    print("================================")

    print(
        "Fine-tuning begins at Epoch 6"
    )

    # Enable fine-tuning and RECOMPILE
    model = enable_fine_tuning(model)

    # Save best model during fine-tuning
    best_checkpoint = ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

    )

    checkpoint_callback = EpochCheckpoint()

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=1,

        min_lr=1e-7,

        verbose=1

    )

    print("\n================================")
    print("FINE-TUNING TRAINING")
    print("================================")

    print(
        "Starting from Epoch:",
        completed_epoch + 1
    )

    print(
        "Ending at Epoch:",
        TOTAL_EPOCHS
    )

    model.fit(

        train_generator,

        validation_data=validation_generator,

        initial_epoch=completed_epoch,

        epochs=TOTAL_EPOCHS,

        callbacks=[

            checkpoint_callback,

            best_checkpoint,

            reduce_lr

        ]

    )

else:

    print(
        "\nAll 10 epochs are already completed."
    )


training_time = time.time() - start_time


# ============================================================
# LOAD BEST MODEL
# ============================================================

if os.path.exists(BEST_MODEL_PATH):

    print(
        "\nLoading best validation model..."
    )

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_PATH
)

print(
    "\nFinal model saved:"
)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# MODEL SIZE
# ============================================================

model_size_mb = (

    os.path.getsize(
        FINAL_MODEL_PATH
    )
    /
    (1024 * 1024)

)

print(
    "\nFinal model size:",
    round(model_size_mb, 2),
    "MB"
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n================================")
print("TEST EVALUATION")
print("================================")

test_loss, test_accuracy = model.evaluate(

    test_generator,

    verbose=1

)

print(
    "\nTest Accuracy:",
    test_accuracy
)

print(
    "Test Loss:",
    test_loss
)


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
print("CONVNEXTTINY 248x248 RESULTS")
print("================================")

print(
    "Accuracy :",
    test_accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall   :",
    recall
)

print(
    "F1 Score :",
    f1
)

print(
    "Training time:",
    training_time,
    "seconds"
)

print(
    "Total parameters:",
    model.count_params()
)

print(
    "Model size:",
    round(model_size_mb, 2),
    "MB"
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

print(
    "\nClassification Report"
)

print(
    report
)

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

plt.figure(
    figsize=(8, 6)
)

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

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

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
# PER-CLASS RESULTS
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

    "model":
        "ConvNeXtTiny",

    "image_size":
        "248x248",

    "epochs":
        TOTAL_EPOCHS,

    "batch_size":
        BATCH_SIZE,

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
        int(model.count_params()),

    "model_size_mb":
        float(model_size_mb),

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
        "convnext_results.json"
    ),

    "w"

) as file:

    json.dump(

        results,

        file,

        indent=4

    )


# ============================================================
# COMPLETE
# ============================================================

print("\n================================")
print("CONVNEXTTINY 248x248 V3")
print("EXPERIMENT COMPLETED")
print("================================")

print(
    "\nResults saved in:",
    RESULT_DIR
)