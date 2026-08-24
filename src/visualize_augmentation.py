import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import random

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use one sample image to demonstrate all steps
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "dataset" / "EUS" / "EUS_5.jpg"

IMAGE_SIZE = (224, 224)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_PATH = PROJECT_ROOT / "src" / "augmentation_visualization.png"


# ============================================================
# IMAGE LOADING  (same logic as augmentation.py)
# ============================================================

def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image


# ============================================================
# AUGMENTATION FUNCTIONS  (same logic as augmentation.py)
# ============================================================

def horizontal_flip(image):
    return cv2.flip(image, 1)


def vertical_flip(image):
    return cv2.flip(image, 0)


def zoom_image(image, zoom_factor):
    """
    zoom_factor > 1.0  →  zoom in  (crop centre, resize back)
    zoom_factor < 1.0  →  zoom out (shrink, replicate border)
    """
    height, width = image.shape[:2]
    new_h = max(1, min(int(height / zoom_factor), height))
    new_w = max(1, min(int(width  / zoom_factor), width))
    y1 = (height - new_h) // 2
    x1 = (width  - new_w) // 2
    cropped = image[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)


def shift_image(image, width_shift, height_shift):
    height, width = image.shape[:2]
    tx = width_shift * width
    ty = height_shift * height
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        image, matrix, (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )


def change_brightness(image, factor):
    image_float = image.astype(np.float32)
    image_float *= factor
    image_float = np.clip(image_float, 0, 255)
    return image_float.astype(np.uint8)


# ============================================================
# BUILD LIST OF ALL STEPS WITH LABELS & DESCRIPTIONS
# ============================================================

def build_steps(image):
    """
    Returns a list of (title, description, result_image) tuples,
    one entry per augmentation step plus the original.
    """
    steps = []

    # ----------------------------------------------------------
    # Step 0: Original
    # ----------------------------------------------------------
    steps.append((
        "Original",
        "Resized to 224×224\nNo changes applied",
        image.copy()
    ))

    # ----------------------------------------------------------
    # Step 1: Horizontal Flip
    # ----------------------------------------------------------
    flipped = horizontal_flip(image)
    steps.append((
        "Step 1 — Horizontal Flip",
        "Mirror image left ↔ right\ncv2.flip(image, 1)",
        flipped
    ))

    # ----------------------------------------------------------
    # Step 2: Vertical Flip
    # ----------------------------------------------------------
    vflipped = vertical_flip(image)
    steps.append((
        "Step 2 — Vertical Flip",
        "Mirror image top ↕ bottom\ncv2.flip(image, 0)",
        vflipped
    ))

    # ----------------------------------------------------------
    # Step 3: Zoom In
    # ----------------------------------------------------------
    zoomed_in = zoom_image(image, 1.3)
    steps.append((
        "Step 3 — Zoom In (×1.3)",
        "Crop centre to 1/1.3 size\nthen resize back to 224×224",
        zoomed_in
    ))

    # ----------------------------------------------------------
    # Step 4: Zoom Out
    # ----------------------------------------------------------
    zoomed_out = zoom_image(image, 0.75)
    steps.append((
        "Step 4 — Zoom Out (×0.75)",
        "Crop centre to 1/0.75 size\nthen resize back to 224×224",
        zoomed_out
    ))

    # ----------------------------------------------------------
    # Step 5: Width / Height Shift
    # ----------------------------------------------------------
    shifted = shift_image(image, 0.15, 0.10)
    steps.append((
        "Step 5 — Shift",
        "Translate: +15% width, +10% height\nBorders filled by replication",
        shifted
    ))

    # ----------------------------------------------------------
    # Step 6: Rotate 20°
    # ----------------------------------------------------------
    rot20 = rotate_image(image, 20)
    steps.append((
        "Step 6 — Rotate 20°",
        "Rotate around center by 20°\nBorders filled by replication",
        rot20
    ))

    # ----------------------------------------------------------
    # Step 7: Rotate 40°
    # ----------------------------------------------------------
    rot40 = rotate_image(image, 40)
    steps.append((
        "Step 7 — Rotate 40°",
        "Rotate around center by 40°\nBorders filled by replication",
        rot40
    ))

    # ----------------------------------------------------------
    # Step 8: Rotate 60°
    # ----------------------------------------------------------
    rot60 = rotate_image(image, 60)
    steps.append((
        "Step 8 — Rotate 60°",
        "Rotate around center by 60°\nBorders filled by replication",
        rot60
    ))

    # ----------------------------------------------------------
    # Step 9: Brightness (darken)
    # ----------------------------------------------------------
    dark = change_brightness(image, 0.7)
    steps.append((
        "Step 9a — Brightness (dark)",
        "Multiply pixels by 0.7\nDarkens the image",
        dark
    ))

    # ----------------------------------------------------------
    # Step 10: Brightness (brighten)
    # ----------------------------------------------------------
    bright = change_brightness(image, 1.3)
    steps.append((
        "Step 9b — Brightness (bright)",
        "Multiply pixels by 1.3\nBrightens the image",
        bright
    ))

    # ----------------------------------------------------------
    # Step 11: Flip + Rotate 20°
    # ----------------------------------------------------------
    flip_rot20 = rotate_image(horizontal_flip(image), 20)
    steps.append((
        "Step 10 — Flip + Rotate 20°",
        "Horizontal flip, then rotate 20°\nCombined transform",
        flip_rot20
    ))

    # ----------------------------------------------------------
    # Step 12: Flip + Rotate 40°
    # ----------------------------------------------------------
    flip_rot40 = rotate_image(horizontal_flip(image), 40)
    steps.append((
        "Step 11 — Flip + Rotate 40°",
        "Horizontal flip, then rotate 40°\nCombined transform",
        flip_rot40
    ))

    # ----------------------------------------------------------
    # Step 13: Flip + Rotate 60°
    # ----------------------------------------------------------
    flip_rot60 = rotate_image(horizontal_flip(image), 60)
    steps.append((
        "Step 12 — Flip + Rotate 60°",
        "Horizontal flip, then rotate 60°\nCombined transform",
        flip_rot60
    ))

    # ----------------------------------------------------------
    # Step 14: Shift + Rotate 20°
    # ----------------------------------------------------------
    shift_rot20 = rotate_image(shift_image(image, 0.15, 0.10), 20)
    steps.append((
        "Step 13 — Shift + Rotate 20°",
        "Shift (+15%, +10%), then rotate 20°\nCombined transform",
        shift_rot20
    ))

    # ----------------------------------------------------------
    # Step 15: Shift + Rotate 40°
    # ----------------------------------------------------------
    shift_rot40 = rotate_image(shift_image(image, 0.15, 0.10), 40)
    steps.append((
        "Step 14 — Shift + Rotate 40°",
        "Shift (+15%, +10%), then rotate 40°\nCombined transform",
        shift_rot40
    ))

    # ----------------------------------------------------------
    # Step 16: Shift + Rotate 60°
    # ----------------------------------------------------------
    shift_rot60 = rotate_image(shift_image(image, 0.15, 0.10), 60)
    steps.append((
        "Step 15 — Shift + Rotate 60°",
        "Shift (+15%, +10%), then rotate 60°\nCombined transform",
        shift_rot60
    ))

    return steps


# ============================================================
# PLOT
# ============================================================

def plot_steps(steps):
    n = len(steps)          # 14 panels (original + 13 augmentations)
    cols = 4
    rows = (n + cols - 1) // cols  # ceiling division → 4 rows

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4, rows * 4.5)
    )

    fig.suptitle(
        "Data Augmentation — All Steps (15 techniques)\n"
        "Sample image: EUS class (Fish Disease Dataset)",
        fontsize=16,
        fontweight="bold",
        y=1.01
    )

    axes_flat = axes.flatten()

    for i, (title, description, img) in enumerate(steps):
        ax = axes_flat[i]
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(description, fontsize=8, labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        # Highlight the original with a green border
        if i == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ecc71")
                spine.set_linewidth(3)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#3498db")
                spine.set_linewidth(1.5)

    # Hide any unused axes
    for j in range(len(steps), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend
    original_patch = mpatches.Patch(
        color="#2ecc71", label="Original image"
    )
    aug_patch = mpatches.Patch(
        color="#3498db", label="Augmented image"
    )
    fig.legend(
        handles=[original_patch, aug_patch],
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout()
    plt.savefig(
        str(OUTPUT_PATH),
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Visualization saved to: {OUTPUT_PATH}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading sample image...")
    image = load_image(SAMPLE_IMAGE_PATH)
    print(f"  Image shape: {image.shape}")
    print(f"  Pixel range: [{image.min()}, {image.max()}]")

    print("\nApplying all augmentation steps...")
    steps = build_steps(image)
    print(f"  Total panels: {len(steps)} (1 original + {len(steps)-1} augmented)")

    print("\nGenerating visualization...")
    plot_steps(steps)
