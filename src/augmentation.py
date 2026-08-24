import cv2
import numpy as np
from pathlib import Path
import random


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "augmented_dataset"

IMAGE_SIZE = (224, 224)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]


# ============================================================
# TARGET DATASET SIZE
# ============================================================

ORIGINAL_TOTAL = 2450
TARGET_TOTAL = 10500

AUGMENTED_IMAGES_REQUIRED = (
    TARGET_TOTAL - ORIGINAL_TOTAL
)

print("Original images:", ORIGINAL_TOTAL)
print("Target images:", TARGET_TOTAL)
print(
    "New augmented images required:",
    AUGMENTED_IMAGES_REQUIRED
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for class_name in CLASS_NAMES:

    output_class_dir = OUTPUT_DIR / class_name

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Could not read: {image_path}"
        )

        return None

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to 224 × 224
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return image


# ============================================================
# HORIZONTAL FLIP
# ============================================================

def horizontal_flip(image):

    return cv2.flip(
        image,
        1
    )


# ============================================================
# VERTICAL FLIP
# ============================================================

def vertical_flip(image):

    return cv2.flip(
        image,
        0
    )


# ============================================================
# ZOOM
# ============================================================

def zoom_image(image, zoom_factor):
    """
    zoom_factor > 1.0 zooms in (crops centre and resizes back)
    zoom_factor < 1.0 zooms out (adds border padding)
    """

    height, width = image.shape[:2]

    new_h = int(height / zoom_factor)
    new_w = int(width / zoom_factor)

    # Clamp so we never exceed original dimensions
    new_h = max(1, min(new_h, height))
    new_w = max(1, min(new_w, width))

    y1 = (height - new_h) // 2
    x1 = (width - new_w) // 2

    cropped = image[y1:y1 + new_h, x1:x1 + new_w]

    zoomed = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return zoomed


# ============================================================
# WIDTH / HEIGHT SHIFT
# ============================================================

def shift_image(
    image,
    width_shift,
    height_shift
):

    height, width = image.shape[:2]

    tx = width_shift * width
    ty = height_shift * height

    matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    shifted = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )

    return shifted


# ============================================================
# ROTATION
# ============================================================

def rotate_image(
    image,
    angle
):

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# BRIGHTNESS
# ============================================================

def change_brightness(
    image,
    factor
):

    image_float = image.astype(
        np.float32
    )

    image_float *= factor

    image_float = np.clip(
        image_float,
        0,
        255
    )

    return image_float.astype(
        np.uint8
    )


# ============================================================
# RANDOM AUGMENTATION
# ============================================================

def generate_augmented_image(
    image,
    augmentation_type
):

    if augmentation_type == "flip":

        return horizontal_flip(
            image
        )

    elif augmentation_type == "vflip":

        return vertical_flip(
            image
        )

    elif augmentation_type == "zoom_in":

        return zoom_image(
            image,
            1.3
        )

    elif augmentation_type == "zoom_out":

        return zoom_image(
            image,
            0.75
        )

    elif augmentation_type == "shift":

        width_shift = random.uniform(
            -0.2,
            0.2
        )

        height_shift = random.uniform(
            -0.2,
            0.2
        )

        return shift_image(
            image,
            width_shift,
            height_shift
        )

    elif augmentation_type == "rotate20":

        return rotate_image(
            image,
            20
        )

    elif augmentation_type == "rotate40":

        return rotate_image(
            image,
            40
        )

    elif augmentation_type == "rotate60":

        return rotate_image(
            image,
            60
        )

    elif augmentation_type == "brightness":

        factor = random.uniform(
            0.7,
            1.3
        )

        return change_brightness(
            image,
            factor
        )

    elif augmentation_type == "flip_rotate20":

        image = horizontal_flip(
            image
        )

        return rotate_image(
            image,
            20
        )

    elif augmentation_type == "flip_rotate40":

        image = horizontal_flip(
            image
        )

        return rotate_image(
            image,
            40
        )

    elif augmentation_type == "flip_rotate60":

        image = horizontal_flip(
            image
        )

        return rotate_image(
            image,
            60
        )

    elif augmentation_type == "shift_rotate20":

        image = shift_image(
            image,
            random.uniform(-0.2, 0.2),
            random.uniform(-0.2, 0.2)
        )

        return rotate_image(
            image,
            20
        )

    elif augmentation_type == "shift_rotate40":

        image = shift_image(
            image,
            random.uniform(-0.2, 0.2),
            random.uniform(-0.2, 0.2)
        )

        return rotate_image(
            image,
            40
        )

    elif augmentation_type == "shift_rotate60":

        image = shift_image(
            image,
            random.uniform(-0.2, 0.2),
            random.uniform(-0.2, 0.2)
        )

        return rotate_image(
            image,
            60
        )

    else:

        return image


# ============================================================
# AUGMENTATION TYPES
# ============================================================

AUGMENTATION_TYPES = [

    "flip",

    "vflip",

    "zoom_in",

    "zoom_out",

    "shift",

    "rotate20",

    "rotate40",

    "rotate60",

    "brightness",

    "flip_rotate20",

    "flip_rotate40",

    "flip_rotate60",

    "shift_rotate20",

    "shift_rotate40",

    "shift_rotate60"

]


# ============================================================
# GET ALL ORIGINAL IMAGES
# ============================================================

all_images = []

for class_name in CLASS_NAMES:

    class_dir = INPUT_DIR / class_name

    if not class_dir.exists():

        print(
            f"ERROR: Folder not found: "
            f"{class_dir}"
        )

        continue

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:

        image_files.extend(
            class_dir.glob(extension)
        )

    for image_path in image_files:

        all_images.append(
            (
                class_name,
                image_path
            )
        )


print()
print(
    "Total original images found:",
    len(all_images)
)


# ============================================================
# COPY ORIGINAL IMAGES
# ============================================================

print()
print(
    "Copying original images..."
)

original_count = 0

for class_name, image_path in all_images:

    image = load_image(
        image_path
    )

    if image is None:
        continue

    output_path = (
        OUTPUT_DIR /
        class_name /
        f"original_{original_count:05d}.jpg"
    )

    # Convert RGB → BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        image_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    original_count += 1


print(
    "Original images copied:",
    original_count
)


# ============================================================
# GENERATE AUGMENTED IMAGES
# ============================================================

images_needed = (
    TARGET_TOTAL - original_count
)

print()
print(
    "Generating augmented images:",
    images_needed
)

augmented_count = 0

image_index = 0

while augmented_count < images_needed:

    class_name, image_path = all_images[
        image_index % len(all_images)
    ]

    image_index += 1

    image = load_image(
        image_path
    )

    if image is None:
        continue

    augmentation_type = random.choice(
        AUGMENTATION_TYPES
    )

    augmented = generate_augmented_image(
        image,
        augmentation_type
    )

    output_path = (
        OUTPUT_DIR /
        class_name /
        f"augmented_{augmented_count:05d}.jpg"
    )

    # RGB → BGR
    augmented_bgr = cv2.cvtColor(
        augmented,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        augmented_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    augmented_count += 1

    if augmented_count % 500 == 0:

        print(
            f"Generated "
            f"{augmented_count} / "
            f"{images_needed}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

final_total = (
    original_count +
    augmented_count
)

print()
print("=" * 50)
print("AUGMENTATION COMPLETED")
print("=" * 50)

print(
    "Original images:",
    original_count
)

print(
    "Augmented images:",
    augmented_count
)

print(
    "Final dataset:",
    final_total
)

print(
    "Expected dataset:",
    TARGET_TOTAL
)

print()
print(
    "Saved to:",
    OUTPUT_DIR
)