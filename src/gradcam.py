"""
gradcam.py
-----------
Generates Grad-CAM heatmap visualizations for VGG16 and MobileNetV2.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
the regions of the image that the CNN focused on when making
its prediction.

For each model:
  - Takes 1 sample image per class (4 classes = 4 images)
  - Generates original + heatmap + overlay side by side
  - Saves a grid showing all 4 classes

Outputs saved to results/gradcam/:
  vgg16_gradcam.png
  mobilenetv2_gradcam.png
"""

import zipfile
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SPLIT_DIR     = PROJECT_ROOT / "split_dataset" / "test"
RESULTS_DIR   = PROJECT_ROOT / "results"
GRADCAM_DIR   = RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

MOBILENET_PATH  = RESULTS_DIR / "MobileNetV2" / "model" / "mobilenetv2_best.keras"
VGG16_PATH      = RESULTS_DIR / "VGG16"       / "model" / "vgg16_best.keras"
VGG16_WEIGHTS   = RESULTS_DIR / "VGG16"       / "model" / "extracted" / "model.weights.h5"

CLASS_NAMES  = ["EUS", "gill", "healthy", "red_spot"]
CLASS_LABELS = {
    "EUS"      : "EUS (Epizootic Ulcerative Syndrome)",
    "gill"     : "Gill Disease",
    "healthy"  : "Healthy",
    "red_spot" : "Red Spot Disease",
}
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

IMG_SIZE    = (224, 224)
NUM_CLASSES = 4
SEED        = 42

random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("=" * 65)
print("GRAD-CAM VISUALIZATION")
print("=" * 65)


# ============================================================
# LOAD SAMPLE IMAGES — 1 per class from test set
# ============================================================

def get_sample_images():
    """Returns dict: {class_name: (image_rgb, image_path)}"""
    samples = {}
    for cls in CLASS_NAMES:
        cls_dir = SPLIT_DIR / cls
        files   = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        chosen  = random.choice(files)
        img     = cv2.imread(str(chosen))
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img     = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        samples[cls] = (img, chosen)
    return samples


# ============================================================
# GRAD-CAM CORE
# ============================================================

def compute_gradcam(model, image_array, layer_name, class_idx):
    """
    Computes the Grad-CAM heatmap for a given image and class.

    Args:
        model       : Keras model
        image_array : float32 array (1, H, W, 3)
        layer_name  : name of the last conv layer to hook into
        class_idx   : class index to visualize

    Returns:
        heatmap : float32 array (H, W) normalized to [0, 1]
    """
    # Build a sub-model that outputs (feature_maps, predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]

    # Gradients of the class score w.r.t. feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image_rgb, heatmap, alpha=0.45):
    """
    Overlays the Grad-CAM heatmap on the original image.

    Returns:
        overlay : uint8 RGB image
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))

    # Apply colormap (jet)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_colored


# ============================================================
# GENERATE GRAD-CAM GRID FOR ONE MODEL
# ============================================================

def generate_gradcam_grid(model, layer_name, samples, model_name, output_path):
    """
    Creates a 4×3 grid (4 classes × 3 columns: original, heatmap, overlay).
    """
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, 3, figsize=(12, n_classes * 3.5))

    fig.suptitle(
        f"{model_name} — Grad-CAM Visualization\n"
        f"Columns: Original Image | Heatmap | Overlay",
        fontsize=14, fontweight="bold", y=1.01
    )

    col_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row, cls in enumerate(CLASS_NAMES):
        image_rgb, image_path = samples[cls]

        # Prepare input
        image_array = image_rgb.astype(np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # Get prediction
        preds      = model.predict(image_array, verbose=0)
        pred_idx   = int(np.argmax(preds[0]))
        pred_cls   = CLASS_NAMES[pred_idx]
        confidence = float(preds[0][pred_idx]) * 100

        # Grad-CAM for the TRUE class
        true_idx = CLASS_NAMES.index(cls)
        heatmap  = compute_gradcam(model, image_array, layer_name, true_idx)
        overlay, heatmap_colored = overlay_heatmap(image_rgb, heatmap)

        color = CLASS_COLORS[row]

        # Col 0: Original
        axes[row][0].imshow(image_rgb)
        axes[row][0].set_ylabel(
            CLASS_LABELS[cls],
            fontsize=9, fontweight="bold",
            color=color, labelpad=6
        )
        axes[row][0].set_xlabel(
            f"Predicted: {pred_cls} ({confidence:.1f}%)",
            fontsize=8,
            color="green" if pred_cls == cls else "red"
        )

        # Col 1: Heatmap
        axes[row][1].imshow(heatmap_colored)
        axes[row][1].set_xlabel("Red = high attention", fontsize=8)

        # Col 2: Overlay
        axes[row][2].imshow(overlay)
        axes[row][2].set_xlabel("Heatmap overlaid on image", fontsize=8)

        # Borders
        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            for spine in axes[row][col].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading sample images...")
samples = get_sample_images()
for cls, (img, path) in samples.items():
    print(f"  {cls}: {path.name}")


# ============================================================
# VGG16 GRAD-CAM
# ============================================================

print("\n" + "=" * 65)
print("VGG16 GRAD-CAM")
print("=" * 65)

print("Loading VGG16 model...")

# Extract weights if needed
if not VGG16_WEIGHTS.exists():
    VGG16_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(VGG16_PATH), "r") as z:
        z.extract("model.weights.h5", str(VGG16_WEIGHTS.parent))

# Rebuild VGG16 architecture
vgg16_base = tf.keras.applications.VGG16(
    weights=None, include_top=False, input_shape=(224, 224, 3)
)
inputs = tf.keras.Input(shape=(224, 224, 3), name="fish_image")
x      = tf.keras.applications.vgg16.preprocess_input(inputs)
x      = vgg16_base(x, training=False)
x      = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
x      = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
x      = tf.keras.layers.Dropout(0.5, name="dropout")(x)
out    = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

vgg16_model = tf.keras.Model(inputs=inputs, outputs=out, name="VGG16_Fish_Disease")
vgg16_model.load_weights(str(VGG16_WEIGHTS))
vgg16_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("  VGG16 loaded")

# Last conv layer inside the VGG16 base = block5_conv3
# We target the VGG16 base sub-model output before GAP
VGG16_CONV_LAYER = "vgg16"  # use the base model output (7x7x512)

# For VGG16 we need to build a model that exposes the last conv layer directly
vgg16_inner_last = None
for layer in vgg16_base.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        vgg16_inner_last = layer.name

print(f"  Last conv layer in VGG16 base: {vgg16_inner_last}")

def compute_gradcam_vgg16(image_array, class_idx):
    """
    Grad-CAM for VGG16 using GradientTape directly on the base model.
    Watches the conv output inside the base and computes gradients.
    """
    image_tensor = tf.cast(image_array, tf.float32)
    preprocessed = tf.keras.applications.vgg16.preprocess_input(
        tf.identity(image_tensor)
    )

    # Get last conv layer output as a persistent tensor
    last_conv_layer = vgg16_base.get_layer(vgg16_inner_last)

    # Build sub-model: vgg16_base input → last conv output + base output
    conv_model = tf.keras.Model(
        inputs=vgg16_base.inputs,
        outputs=[last_conv_layer.output, vgg16_base.output]
    )

    with tf.GradientTape() as tape:
        conv_out, base_out = conv_model(preprocessed, training=False)
        tape.watch(conv_out)
        # Forward through head layers
        gap    = tf.reduce_mean(conv_out, axis=[1, 2])
        dense  = tf.keras.activations.relu(
            vgg16_model.get_layer("dense_256")(gap)
        )
        preds  = vgg16_model.get_layer("classification")(dense)
        loss   = preds[:, class_idx]

    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


print("Generating VGG16 Grad-CAM grid...")
n_classes = len(CLASS_NAMES)
fig, axes = plt.subplots(n_classes, 3, figsize=(12, n_classes * 3.5))
fig.suptitle(
    "VGG16 — Grad-CAM Visualization\n"
    "Columns: Original Image | Heatmap | Overlay",
    fontsize=14, fontweight="bold", y=1.01
)

col_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight="bold", pad=8)

for row, cls in enumerate(CLASS_NAMES):
    image_rgb, _ = samples[cls]
    image_array  = np.expand_dims(image_rgb.astype(np.float32), axis=0)

    # Prediction
    preds      = vgg16_model.predict(image_array, verbose=0)
    pred_idx   = int(np.argmax(preds[0]))
    pred_cls   = CLASS_NAMES[pred_idx]
    confidence = float(preds[0][pred_idx]) * 100

    true_idx         = CLASS_NAMES.index(cls)
    heatmap          = compute_gradcam_vgg16(image_array, true_idx)
    overlay, hmap_c  = overlay_heatmap(image_rgb, heatmap)
    color            = CLASS_COLORS[row]

    axes[row][0].imshow(image_rgb)
    axes[row][0].set_ylabel(CLASS_LABELS[cls], fontsize=9, fontweight="bold", color=color, labelpad=6)
    axes[row][0].set_xlabel(f"Predicted: {pred_cls} ({confidence:.1f}%)", fontsize=8,
                             color="green" if pred_cls == cls else "red")
    axes[row][1].imshow(hmap_c)
    axes[row][1].set_xlabel("Red = high attention", fontsize=8)
    axes[row][2].imshow(overlay)
    axes[row][2].set_xlabel("Heatmap overlaid on image", fontsize=8)

    for col in range(3):
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        for spine in axes[row][col].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

plt.tight_layout()
vgg16_out = GRADCAM_DIR / "vgg16_gradcam.png"
plt.savefig(str(vgg16_out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {vgg16_out}")


# ============================================================
# MobileNetV2 GRAD-CAM
# ============================================================

print("\n" + "=" * 65)
print("MobileNetV2 GRAD-CAM")
print("=" * 65)

print("Loading MobileNetV2 model...")
mobilenet_model = tf.keras.models.load_model(
    str(MOBILENET_PATH), safe_mode=False
)
mobilenet_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
print("  MobileNetV2 loaded")

# Get the inner MobileNetV2 base
mob_base = mobilenet_model.get_layer("mobilenetv2_1.00_224")

# Last conv layer = out_relu (output of Conv_1_bn activation)
MOB_LAST_CONV = "out_relu"
print(f"  Using Grad-CAM layer: {MOB_LAST_CONV}")


def compute_gradcam_mobilenet(image_array, class_idx):
    """
    Grad-CAM for MobileNetV2 — watches conv_out inside the tape.
    """
    conv_model = tf.keras.Model(
        inputs=mob_base.inputs,
        outputs=mob_base.get_layer(MOB_LAST_CONV).output
    )

    image_tensor = tf.cast(image_array, tf.float32)
    aug_out      = mobilenet_model.get_layer("data_augmentation")(
        image_tensor, training=False
    )
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.identity(aug_out)
    )

    with tf.GradientTape() as tape:
        conv_out = conv_model(preprocessed, training=False)
        tape.watch(conv_out)
        # Use full model prediction for loss
        preds = mobilenet_model(image_tensor, training=False)
        loss  = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)

    # Fallback: gradient saliency if grads are None
    if grads is None:
        with tf.GradientTape() as tape2:
            inp = tf.cast(image_array, tf.float32)
            tape2.watch(inp)
            p = mobilenet_model(inp, training=False)
            l = p[:, class_idx]
        g = tape2.gradient(l, inp)
        heatmap = tf.reduce_mean(tf.abs(g[0]), axis=-1)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


print("Generating MobileNetV2 Grad-CAM grid...")
fig, axes = plt.subplots(n_classes, 3, figsize=(12, n_classes * 3.5))
fig.suptitle(
    "MobileNetV2 — Grad-CAM Visualization\n"
    "Columns: Original Image | Heatmap | Overlay",
    fontsize=14, fontweight="bold", y=1.01
)

for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight="bold", pad=8)

for row, cls in enumerate(CLASS_NAMES):
    image_rgb, _ = samples[cls]
    image_array  = np.expand_dims(image_rgb.astype(np.float32), axis=0)

    preds      = mobilenet_model.predict(image_array, verbose=0)
    pred_idx   = int(np.argmax(preds[0]))
    pred_cls   = CLASS_NAMES[pred_idx]
    confidence = float(preds[0][pred_idx]) * 100

    true_idx        = CLASS_NAMES.index(cls)
    heatmap         = compute_gradcam_mobilenet(image_array, true_idx)
    overlay, hmap_c = overlay_heatmap(image_rgb, heatmap)
    color           = CLASS_COLORS[row]

    axes[row][0].imshow(image_rgb)
    axes[row][0].set_ylabel(CLASS_LABELS[cls], fontsize=9, fontweight="bold", color=color, labelpad=6)
    axes[row][0].set_xlabel(f"Predicted: {pred_cls} ({confidence:.1f}%)", fontsize=8,
                             color="green" if pred_cls == cls else "red")
    axes[row][1].imshow(hmap_c)
    axes[row][1].set_xlabel("Red = high attention", fontsize=8)
    axes[row][2].imshow(overlay)
    axes[row][2].set_xlabel("Heatmap overlaid on image", fontsize=8)

    for col in range(3):
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        for spine in axes[row][col].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

plt.tight_layout()
mob_out = GRADCAM_DIR / "mobilenetv2_gradcam.png"
plt.savefig(str(mob_out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {mob_out}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 65)
print("GRAD-CAM COMPLETE")
print("=" * 65)
print(f"\nOutputs saved to: {GRADCAM_DIR}")
print("  vgg16_gradcam.png       — VGG16 attention maps (4 classes × 3 views)")
print("  mobilenetv2_gradcam.png — MobileNetV2 attention maps (4 classes × 3 views)")
print("\nEach row = one disease class")
print("Col 1 = original image")
print("Col 2 = Grad-CAM heatmap (red = high attention)")
print("Col 3 = overlay (heatmap on image)")
