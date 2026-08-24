import tensorflow as tf
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = str(PROJECT_ROOT / "augmented_dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# LOAD AUGMENTED DATASET
# ==========================================

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

# ==========================================
# NORMALIZATION
# Paper: scaling factor = 1.0 / 255
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255.0
)

dataset = dataset.map(
    lambda images, labels:
        (normalization_layer(images), labels)
)

# ==========================================
# VERIFY
# ==========================================

for images, labels in dataset.take(1):
    print("Image shape  :", images.shape)
    print("Label shape  :", labels.shape)
    print("Min pixel    :", tf.reduce_min(images).numpy())
    print("Max pixel    :", tf.reduce_max(images).numpy())