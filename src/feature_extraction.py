"""
feature_extraction.py
----------------------
Extracts deep features from VGG16 and MobileNetV2 for all three
splits (train / validation / test) and saves them as numpy arrays.

Feature dimensions:
  VGG16       : 512  (GlobalAveragePooling2D after conv base)
  MobileNetV2 : 1280 (GlobalAveragePooling2D after conv base)
  Fused       : 1792 (concatenation of both)

Outputs saved to  results/features/:
  vgg16_train.npz        vgg16_val.npz        vgg16_test.npz
  mobilenet_train.npz    mobilenet_val.npz    mobilenet_test.npz
  fused_train.npz        fused_val.npz        fused_test.npz

Each .npz contains:
  features : float32 array  (N, D)
  labels   : int32   array  (N,)   0=EUS 1=gill 2=healthy 3=red_spot
"""

import os
import zipfile
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR   = PROJECT_ROOT / "split_dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_DIR = RESULTS_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

VGG16_KERAS    = RESULTS_DIR / "VGG16"    / "model" / "vgg16_best.keras"
MOBILENET_KERAS = RESULTS_DIR / "MobileNetV2" / "model" / "mobilenetv2_best.keras"
VGG16_WEIGHTS  = RESULTS_DIR / "VGG16"    / "model" / "extracted" / "model.weights.h5"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

CLASS_NAMES = ["EUS", "gill", "healthy", "red_spot"]
NUM_CLASSES = len(CLASS_NAMES)

tf.keras.utils.set_random_seed(SEED)

print("=" * 65)
print("PHASE 3 — FEATURE EXTRACTION")
print("=" * 65)
print(f"TensorFlow : {tf.__version__}")
print(f"Image size : {IMG_SIZE}")
print(f"Batch size : {BATCH_SIZE}")


# ============================================================
# STEP 1 — BUILD VGG16 FEATURE EXTRACTOR
# VGG16 .keras file has a Keras version mismatch so we rebuild
# the architecture and load from the extracted weights file.
# ============================================================

print("\n" + "=" * 65)
print("BUILDING VGG16 FEATURE EXTRACTOR")
print("=" * 65)

# Extract weights from .keras zip if not already done
if not VGG16_WEIGHTS.exists():
    print("  Extracting VGG16 weights from .keras archive...")
    VGG16_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(VGG16_KERAS), "r") as z:
        z.extract("model.weights.h5", str(VGG16_WEIGHTS.parent))
    print(f"  Extracted to: {VGG16_WEIGHTS}")
else:
    print(f"  Weights already extracted: {VGG16_WEIGHTS}")

# Rebuild exact same architecture as vgg16_training.py
vgg16_base = tf.keras.applications.VGG16(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

vgg16_inputs = tf.keras.Input(shape=(224, 224, 3), name="fish_image")
x = tf.keras.applications.vgg16.preprocess_input(vgg16_inputs)
x = vgg16_base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
vgg16_out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

vgg16_full = tf.keras.Model(
    inputs=vgg16_inputs,
    outputs=vgg16_out,
    name="VGG16_Fish_Disease"
)
vgg16_full.load_weights(str(VGG16_WEIGHTS))
print("  VGG16 weights loaded successfully")

# Feature extractor = output of GAP layer (512-dim)
vgg16_extractor = tf.keras.Model(
    inputs=vgg16_full.input,
    outputs=vgg16_full.get_layer("global_average_pooling").output,
    name="VGG16_FeatureExtractor"
)
print(f"  VGG16 feature dim: {vgg16_extractor.output.shape[-1]}")


# ============================================================
# STEP 2 — BUILD MobileNetV2 FEATURE EXTRACTOR
# MobileNetV2 .keras loads fine with current Keras version.
# ============================================================

print("\n" + "=" * 65)
print("BUILDING MobileNetV2 FEATURE EXTRACTOR")
print("=" * 65)

mobilenet_full = tf.keras.models.load_model(
    str(MOBILENET_KERAS),
    safe_mode=False
)
print("  MobileNetV2 loaded successfully")

# Feature extractor = output of GAP layer (1280-dim)
mobilenet_extractor = tf.keras.Model(
    inputs=mobilenet_full.input,
    outputs=mobilenet_full.get_layer("global_average_pooling").output,
    name="MobileNetV2_FeatureExtractor"
)
print(f"  MobileNetV2 feature dim: {mobilenet_extractor.output.shape[-1]}")


# ============================================================
# STEP 3 — DATA LOADING HELPER
# ============================================================

def load_split(split_name):
    """
    Returns a tf.data.Dataset and the ordered list of labels.
    shuffle=False ensures label order is preserved.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / split_name,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CLASS_NAMES,
        shuffle=False
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# STEP 4 — EXTRACT FEATURES FUNCTION
# ============================================================

def extract_features(extractor, dataset, split_name, model_name):
    """
    Runs extractor on all batches and returns (features, labels).
    """
    print(f"\n  Extracting {model_name} features for [{split_name}]...")

    all_features = []
    all_labels   = []

    for batch_idx, (images, labels) in enumerate(dataset):
        feats = extractor(images, training=False)
        all_features.append(feats.numpy())
        all_labels.append(labels.numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1} done...")

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels,   axis=0).astype(np.int32)

    print(f"    Features shape : {features.shape}")
    print(f"    Labels shape   : {labels.shape}")
    print(f"    Label counts   : { {CLASS_NAMES[i]: int((labels==i).sum()) for i in range(NUM_CLASSES)} }")

    return features, labels


# ============================================================
# STEP 5 — EXTRACT AND SAVE FOR ALL SPLITS
# ============================================================

SPLITS = ["train", "validation", "test"]

vgg16_features    = {}
mobilenet_features = {}
all_labels        = {}

for split in SPLITS:
    print(f"\n{'='*65}")
    print(f"PROCESSING SPLIT: {split.upper()}")
    print(f"{'='*65}")

    ds = load_split(split)

    # --- VGG16 ---
    vgg_feats, labels = extract_features(
        vgg16_extractor, ds, split, "VGG16"
    )
    vgg16_features[split] = vgg_feats
    all_labels[split]     = labels

    np.savez_compressed(
        str(FEATURE_DIR / f"vgg16_{split}.npz"),
        features=vgg_feats,
        labels=labels
    )
    print(f"    Saved: features/vgg16_{split}.npz")

    # --- MobileNetV2 ---
    # Reload dataset (iterator exhausted)
    ds = load_split(split)
    mob_feats, _ = extract_features(
        mobilenet_extractor, ds, split, "MobileNetV2"
    )
    mobilenet_features[split] = mob_feats

    np.savez_compressed(
        str(FEATURE_DIR / f"mobilenet_{split}.npz"),
        features=mob_feats,
        labels=labels
    )
    print(f"    Saved: features/mobilenet_{split}.npz")

    # --- Fused (concatenation) ---
    fused = np.concatenate([vgg_feats, mob_feats], axis=1)
    np.savez_compressed(
        str(FEATURE_DIR / f"fused_{split}.npz"),
        features=fused,
        labels=labels
    )
    print(f"    Fused shape    : {fused.shape}")
    print(f"    Saved: features/fused_{split}.npz")


# ============================================================
# STEP 6 — SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 65)

print(f"\n{'Split':<12} {'VGG16':>10} {'MobileNet':>12} {'Fused':>10} {'Samples':>10}")
print("-" * 56)

for split in SPLITS:
    v = vgg16_features[split].shape
    m = mobilenet_features[split].shape
    f_dim = v[1] + m[1]
    print(f"{split:<12} {str(v):>10} {str(m):>12} {f'({v[0]},{f_dim})':>10} {v[0]:>10}")

print(f"\nVGG16 feature dim       : {vgg16_features['train'].shape[1]}")
print(f"MobileNetV2 feature dim : {mobilenet_features['train'].shape[1]}")
print(f"Fused feature dim       : {vgg16_features['train'].shape[1] + mobilenet_features['train'].shape[1]}")

print(f"\nAll features saved to: {FEATURE_DIR}")

files = list(FEATURE_DIR.glob("*.npz"))
print(f"\nFiles created ({len(files)}):")
for f in sorted(files):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name:<30} {size_mb:.1f} MB")

print("\nNext step: python src/fusion_experiments.py")
