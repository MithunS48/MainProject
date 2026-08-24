import shutil
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "augmented_dataset"
OUTPUT_DIR = PROJECT_ROOT / "split_dataset"

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

SEED = 42

random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:

    for class_name in CLASS_NAMES:

        folder = (
            OUTPUT_DIR /
            split /
            class_name
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_images(folder):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    files = []

    for extension in extensions:
        files.extend(folder.glob(extension))

    # Remove duplicates
    return list(set(files))


# ============================================================
# COPY FILES
# ============================================================

def copy_files(files, destination):

    copied = 0

    for image_path in files:

        destination_file = (
            destination /
            image_path.name
        )

        shutil.copy2(
            image_path,
            destination_file
        )

        copied += 1

    return copied


# ============================================================
# MAIN SPLITTING
# ============================================================

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)

total_train = 0
total_validation = 0
total_test = 0


for class_name in CLASS_NAMES:

    source_class = (
        SOURCE_DIR /
        class_name
    )

    print(f"\nProcessing: {class_name}")

    if not source_class.exists():

        print(
            f"ERROR: Folder not found: "
            f"{source_class}"
        )

        continue

    # Get images
    images = get_images(
        source_class
    )

    # Shuffle
    random.shuffle(images)

    total_images = len(images)

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    train_count = int(
        total_images *
        TRAIN_RATIO
    )

    validation_count = int(
        total_images *
        VALIDATION_RATIO
    )

    test_count = (
        total_images
        - train_count
        - validation_count
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_images = images[
        :train_count
    ]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # --------------------------------------------------------
    # Destination folders
    # --------------------------------------------------------

    train_dir = (
        OUTPUT_DIR /
        "train" /
        class_name
    )

    validation_dir = (
        OUTPUT_DIR /
        "validation" /
        class_name
    )

    test_dir = (
        OUTPUT_DIR /
        "test" /
        class_name
    )

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    print(
        f"  Total: {total_images}"
    )

    print(
        f"  Copying training: "
        f"{train_count}"
    )

    copy_files(
        train_images,
        train_dir
    )

    print(
        f"  Copying validation: "
        f"{validation_count}"
    )

    copy_files(
        validation_images,
        validation_dir
    )

    print(
        f"  Copying testing: "
        f"{test_count}"
    )

    copy_files(
        test_images,
        test_dir
    )

    # --------------------------------------------------------
    # Update totals
    # --------------------------------------------------------

    total_train += train_count
    total_validation += validation_count
    total_test += test_count


# ============================================================
# FINAL SUMMARY
# ============================================================

total = (
    total_train +
    total_validation +
    total_test
)

print("\n")
print("=" * 60)
print("SPLIT COMPLETED")
print("=" * 60)

print(
    f"Training:    {total_train}"
)

print(
    f"Validation:  {total_validation}"
)

print(
    f"Testing:     {total_test}"
)

print(
    f"Total:       {total}"
)

print("\nDataset saved at:")

print(
    OUTPUT_DIR
)

print("=" * 60)