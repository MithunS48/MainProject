"""
feature_extraction.py
----------------------
Extracts deep features from VGG16, MobileNetV2 and ConvNeXtTiny
for all three splits (train / validation / test).

Feature dimensions:
  VGG16       : 512
  MobileNetV2 : 1280
  ConvNeXt    : 768
  Fused (all) : 2560

Outputs saved to results/features/:
  vgg16_train/val/test.npz
  mobilenet_train/val/test.npz
  convnext_train/val/test.npz
  fused_train/val/test.npz          (VGG16+MobileNetV2, existing)
  fused_all_train/val/test.npz      (VGG16+MobileNetV2+ConvNeXt)
"""

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

VGG16_KERAS      = RESULTS_DIR / "VGG16"       / "model" / "vgg16_best.keras"
MOBILENET_KERAS  = RESULTS_DIR / "MobileNetV2" / "model" / "mobilenetv2_best.keras"
CONVNEXT_KERAS   = RESULTS_DIR / "ConvNeXt"    / "model" / "convnext_best.keras"
VGG16_WEIGHTS    = RESULTS_DIR / "VGG16"       / "model" / "extracted" / "model.weights.h5"
CONVNEXT_WEIGHTS = RESULTS_DIR / "ConvNeXt"    / "model" / "extracted" / "model.weights.h5"

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
# STEP 3 — BUILD ConvNeXtTiny FEATURE EXTRACTOR
# Uses weights extraction same as VGG16 approach
# ============================================================

print("\n" + "=" * 65)
print("BUILDING ConvNeXtTiny FEATURE EXTRACTOR")
print("=" * 65)

if not CONVNEXT_WEIGHTS.exists():
    print("  Extracting ConvNeXt weights from .keras archive...")
    CONVNEXT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(CONVNEXT_KERAS), "r") as z:
        z.extract("model.weights.h5", str(CONVNEXT_WEIGHTS.parent))
    print(f"  Extracted to: {CONVNEXT_WEIGHTS}")
else:
    print(f"  Weights already extracted: {CONVNEXT_WEIGHTS}")

data_augmentation_cx = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

convnext_base = tf.keras.applications.ConvNeXtTiny(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

cx_inputs = tf.keras.Input(shape=(224, 224, 3), name="fish_image")
cx = data_augmentation_cx(cx_inputs)
cx = tf.keras.layers.Lambda(
    lambda t: tf.cast(t, tf.float32), name="cast_float32"
)(cx)
cx = convnext_base(cx, training=False)
cx = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(cx)
cx = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(cx)
cx = tf.keras.layers.LayerNormalization(name="layer_norm")(cx)
cx = tf.keras.layers.Dropout(0.5, name="dropout")(cx)
cx_out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(cx)

convnext_full = tf.keras.Model(inputs=cx_inputs, outputs=cx_out, name="ConvNeXtTiny_Fish_Disease")
convnext_full.load_weights(str(CONVNEXT_WEIGHTS))
print("  ConvNeXt weights loaded successfully")

convnext_extractor = tf.keras.Model(
    inputs=convnext_full.input,
    outputs=convnext_full.get_layer("global_average_pooling").output,
    name="ConvNeXt_FeatureExtractor"
)
print(f"  ConvNeXt feature dim: {convnext_extractor.output.shape[-1]}")
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

vgg16_features     = {}
mobilenet_features = {}
convnext_features  = {}
all_labels         = {}

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
        features=vgg_feats, labels=labels
    )
    print(f"    Saved: features/vgg16_{split}.npz")

    # --- MobileNetV2 ---
    ds = load_split(split)
    mob_feats, _ = extract_features(
        mobilenet_extractor, ds, split, "MobileNetV2"
    )
    mobilenet_features[split] = mob_feats
    np.savez_compressed(
        str(FEATURE_DIR / f"mobilenet_{split}.npz"),
        features=mob_feats, labels=labels
    )
    print(f"    Saved: features/mobilenet_{split}.npz")

    # --- ConvNeXt ---
    ds = load_split(split)
    cx_feats, _ = extract_features(
        convnext_extractor, ds, split, "ConvNeXt"
    )
    convnext_features[split] = cx_feats
    np.savez_compressed(
        str(FEATURE_DIR / f"convnext_{split}.npz"),
        features=cx_feats, labels=labels
    )
    print(f"    Saved: features/convnext_{split}.npz")

    # --- Fused VGG16+MobileNetV2 ---
    fused = np.concatenate([vgg_feats, mob_feats], axis=1)
    np.savez_compressed(
        str(FEATURE_DIR / f"fused_{split}.npz"),
        features=fused, labels=labels
    )
    print(f"    Fused (VGG16+MobileNetV2) shape : {fused.shape}")
    print(f"    Saved: features/fused_{split}.npz")

    # --- Fused ALL (VGG16+MobileNetV2+ConvNeXt) ---
    fused_all = np.concatenate([vgg_feats, mob_feats, cx_feats], axis=1)
    np.savez_compressed(
        str(FEATURE_DIR / f"fused_all_{split}.npz"),
        features=fused_all, labels=labels
    )
    print(f"    Fused ALL shape                 : {fused_all.shape}")
    print(f"    Saved: features/fused_all_{split}.npz")


# ============================================================
# STEP 6 — SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 65)

print(f"\n{'Split':<12} {'VGG16':>8} {'MobileNet':>10} {'ConvNeXt':>10} {'Fused':>8} {'FusedAll':>10}")
print("-" * 62)

for split in SPLITS:
    v  = vgg16_features[split].shape[1]
    m  = mobilenet_features[split].shape[1]
    c  = convnext_features[split].shape[1]
    n  = vgg16_features[split].shape[0]
    print(f"{split:<12} {v:>8} {m:>10} {c:>10} {v+m:>8} {v+m+c:>10}  (N={n})")

print(f"\nVGG16 feature dim         : {vgg16_features['train'].shape[1]}")
print(f"MobileNetV2 feature dim   : {mobilenet_features['train'].shape[1]}")
print(f"ConvNeXt feature dim      : {convnext_features['train'].shape[1]}")
print(f"Fused (V+M) dim           : {vgg16_features['train'].shape[1] + mobilenet_features['train'].shape[1]}")
print(f"Fused ALL (V+M+C) dim     : {vgg16_features['train'].shape[1] + mobilenet_features['train'].shape[1] + convnext_features['train'].shape[1]}")

print(f"\nAll features saved to: {FEATURE_DIR}")

files = list(FEATURE_DIR.glob("*.npz"))
print(f"\nFiles ({len(files)}):")
for f in sorted(files):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name:<35} {size_mb:.1f} MB")

print("\nNext step: python src/fusion_experiments.py")
