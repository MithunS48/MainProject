import tensorflow as tf

DATASET_PATH = "../dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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

for images, labels in dataset.take(1):
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())