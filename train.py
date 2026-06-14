"""
Fish Disease Detection — Improved Training Pipeline
Two-phase training: frozen base → fine-tune top layers

Usage:
    python train.py [--epochs 30] [--fine-tune-epochs 20]
                    [--output model/fish_disease_model.keras]
                    [--batch-size 32]
"""

import argparse
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2-based fish disease classifier."
    )
    parser.add_argument("--epochs",           type=int,   default=30,
                        help="Phase-1 training epochs (default: 30)")
    parser.add_argument("--fine-tune-epochs", type=int,   default=20,
                        help="Phase-2 fine-tuning epochs (default: 20)")
    parser.add_argument("--output",           type=str,
                        default="model/fish_disease_model.keras",
                        help="Output path for saved model")
    parser.add_argument("--batch-size",       type=int,   default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--img-size",         type=int,   default=224,
                        help="Input image size (default: 224)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Data generators
# ──────────────────────────────────────────────
def build_generators(batch_size: int, img_size: int):
    train_dir = os.path.join("split_dataset", "train")
    val_dir   = os.path.join("split_dataset", "validation")
    test_dir  = os.path.join("split_dataset", "test")

    if not os.path.isdir(train_dir):
        print(f"Error: Training directory '{train_dir}' not found.")
        sys.exit(1)

    import tensorflow as tf

    # Richer augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        zoom_range=0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    size = (img_size, img_size)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=size, batch_size=batch_size, class_mode="categorical"
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir, target_size=size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    test_gen = eval_datagen.flow_from_directory(
        test_dir, target_size=size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen, test_gen


# ──────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────
def compute_class_weights(train_gen):
    from sklearn.utils.class_weight import compute_class_weight
    labels  = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def build_model(num_classes: int, img_size: int):
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False          # Phase 1: frozen base

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ──────────────────────────────────────────────
# Phase 1 — train head only
# ──────────────────────────────────────────────
def phase1_train(model, train_gen, val_gen, epochs, class_weights):
    import tensorflow as tf
    print("\n" + "="*50)
    print("PHASE 1 — Training classification head (base frozen)")
    print("="*50)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
    ]
    return model.fit(
        train_gen, epochs=epochs, validation_data=val_gen,
        class_weight=class_weights, callbacks=callbacks, verbose=1
    )


# ──────────────────────────────────────────────
# Phase 2 — fine-tune top layers of base
# ──────────────────────────────────────────────
def phase2_finetune(model, base, train_gen, val_gen, epochs, class_weights):
    import tensorflow as tf
    print("\n" + "="*50)
    print("PHASE 2 — Fine-tuning top layers of MobileNetV2")
    print("="*50)

    # Unfreeze the top 40 layers of the base model
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  Unfrozen base layers: {trainable_count} / {len(base.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4,
            min_lr=1e-7, verbose=1
        ),
    ]
    return model.fit(
        train_gen, epochs=epochs, validation_data=val_gen,
        class_weight=class_weights, callbacks=callbacks, verbose=1
    )


# ──────────────────────────────────────────────
# Evaluation & save
# ──────────────────────────────────────────────
def evaluate_and_save(model, test_gen, output_path: str, class_indices: dict):
    from sklearn.metrics import classification_report, confusion_matrix

    # Class labels in correct order
    idx_to_class = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nRunning inference on test set…")
    test_gen.reset()
    preds         = model.predict(test_gen, verbose=1)
    pred_classes  = np.argmax(preds, axis=1)
    true_classes  = test_gen.classes

    accuracy = np.mean(pred_classes == true_classes)
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(true_classes, pred_classes, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(true_classes, pred_classes)
    print(cm)

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    model.save(output_path)
    print(f"\nModel saved → {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 50)
    print("AquaScan — Fish Disease Detection Training")
    print("=" * 50)
    print(f"  Phase-1 epochs    : {args.epochs}")
    print(f"  Phase-2 epochs    : {args.fine_tune_epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Image size        : {args.img_size}x{args.img_size}")
    print(f"  Output            : {args.output}")
    print()

    # Data
    print("Loading datasets…")
    train_gen, val_gen, test_gen = build_generators(args.batch_size, args.img_size)
    print(f"  Train   : {train_gen.samples} images | {len(train_gen.class_indices)} classes")
    print(f"  Val     : {val_gen.samples} images")
    print(f"  Test    : {test_gen.samples} images")
    print(f"  Classes : {train_gen.class_indices}")

    # Class weights
    print("\nComputing class weights…")
    class_weights = compute_class_weights(train_gen)
    print(f"  Weights: {class_weights}")

    # Build
    print("\nBuilding model…")
    model, base = build_model(
        num_classes=len(train_gen.class_indices),
        img_size=args.img_size
    )
    print(f"  Total params: {model.count_params():,}")

    # Phase 1
    phase1_train(model, train_gen, val_gen, args.epochs, class_weights)

    # Phase 2
    phase2_finetune(model, base, train_gen, val_gen, args.fine_tune_epochs, class_weights)

    # Evaluate & save
    evaluate_and_save(model, test_gen, args.output, train_gen.class_indices)


if __name__ == "__main__":
    main()
