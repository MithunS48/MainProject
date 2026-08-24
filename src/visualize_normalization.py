import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_IMAGE_PATH = (
    PROJECT_ROOT / "augmented_dataset" / "EUS" / "augmented_00000.jpg"
)

IMAGE_SIZE = (224, 224)

OUTPUT_PATH = (
    PROJECT_ROOT / "src" / "normalization_visualization.png"
)


# ============================================================
# LOAD IMAGE  (returns uint8, range 0–255)
# ============================================================

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image  # uint8, [0, 255]


# ============================================================
# NORMALIZE  (returns float32, range 0.0–1.0)
# ============================================================

def normalize(image):
    return image.astype(np.float32) / 255.0


# ============================================================
# PIXEL HISTOGRAM
# ============================================================

def plot_histogram(ax, image_flat, color, title, x_label, value_range):
    ax.hist(
        image_flat,
        bins=64,
        color=color,
        alpha=0.85,
        edgecolor="none"
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Pixel count", fontsize=9)
    ax.set_xlim(value_range)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def visualize(image_raw, image_norm):

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Normalization — Before vs After\n"
        "Scaling: pixel / 255  →  [0, 255]  becomes  [0.0, 1.0]",
        fontsize=15,
        fontweight="bold",
        y=1.01
    )

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        hspace=0.5,
        wspace=0.35
    )

    # ----------------------------------------------------------
    # Row 0: Images side by side
    # ----------------------------------------------------------

    ax_orig = fig.add_subplot(gs[0, 0:2])
    ax_orig.imshow(image_raw)
    ax_orig.set_title(
        "BEFORE Normalization",
        fontsize=11, fontweight="bold", color="#e74c3c"
    )
    ax_orig.set_xlabel(
        f"Pixel range: [{image_raw.min()},  {image_raw.max()}]\n"
        f"dtype: {image_raw.dtype}",
        fontsize=9
    )
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    for spine in ax_orig.spines.values():
        spine.set_edgecolor("#e74c3c")
        spine.set_linewidth(2.5)

    ax_norm = fig.add_subplot(gs[0, 2:4])
    ax_norm.imshow(image_norm)
    ax_norm.set_title(
        "AFTER Normalization",
        fontsize=11, fontweight="bold", color="#2ecc71"
    )
    ax_norm.set_xlabel(
        f"Pixel range: [{image_norm.min():.4f},  {image_norm.max():.4f}]\n"
        f"dtype: {image_norm.dtype}",
        fontsize=9
    )
    ax_norm.set_xticks([])
    ax_norm.set_yticks([])
    for spine in ax_norm.spines.values():
        spine.set_edgecolor("#2ecc71")
        spine.set_linewidth(2.5)

    # ----------------------------------------------------------
    # Row 1, col 0-1: Histograms (before)
    # ----------------------------------------------------------

    ax_hist_orig = fig.add_subplot(gs[1, 0:2])
    plot_histogram(
        ax_hist_orig,
        image_raw.flatten(),
        color="#e74c3c",
        title="Pixel Distribution — BEFORE",
        x_label="Pixel value",
        value_range=(0, 255)
    )

    # ----------------------------------------------------------
    # Row 1, col 2-3: Histograms (after)
    # ----------------------------------------------------------

    ax_hist_norm = fig.add_subplot(gs[1, 2:4])
    plot_histogram(
        ax_hist_norm,
        image_norm.flatten(),
        color="#2ecc71",
        title="Pixel Distribution — AFTER",
        x_label="Pixel value",
        value_range=(0.0, 1.0)
    )

    # ----------------------------------------------------------
    # Stats box
    # ----------------------------------------------------------

    stats_text = (
        f"{'Metric':<18} {'Before':>12}  {'After':>12}\n"
        f"{'-'*44}\n"
        f"{'Min':<18} {image_raw.min():>12}  {image_norm.min():>12.4f}\n"
        f"{'Max':<18} {image_raw.max():>12}  {image_norm.max():>12.4f}\n"
        f"{'Mean':<18} {image_raw.mean():>12.2f}  {image_norm.mean():>12.4f}\n"
        f"{'Std Dev':<18} {image_raw.std():>12.2f}  {image_norm.std():>12.4f}\n"
        f"{'dtype':<18} {str(image_raw.dtype):>12}  {str(image_norm.dtype):>12}\n"
        f"{'Formula':<18} {'pixel':>12}  {'pixel / 255':>12}"
    )

    fig.text(
        0.5, -0.04,
        stats_text,
        ha="center",
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f0f0f0",
            edgecolor="#cccccc",
            linewidth=1.5
        )
    )

    plt.savefig(
        str(OUTPUT_PATH),
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Saved to: {OUTPUT_PATH}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading image...")
    image_raw = load_image(SAMPLE_IMAGE_PATH)
    print(f"  Before — shape: {image_raw.shape}  "
          f"dtype: {image_raw.dtype}  "
          f"range: [{image_raw.min()}, {image_raw.max()}]")

    image_norm = normalize(image_raw)
    print(f"  After  — shape: {image_norm.shape}  "
          f"dtype: {image_norm.dtype}  "
          f"range: [{image_norm.min():.4f}, {image_norm.max():.4f}]")

    print("\nGenerating visualization...")
    visualize(image_raw, image_norm)
