"""
dataset_analysis.py
--------------------
Generates a full statistical analysis of the fish disease dataset.

Outputs saved to results/dataset/:
  - dataset_statistics.csv
  - class_distribution.png
  - split_distribution.png
  - sample_images.png
  - image_dimensions.png
  - analysis_report.txt
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
import random

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_DIR   = PROJECT_ROOT / "dataset"
AUGMENTED_DIR  = PROJECT_ROOT / "augmented_dataset"
SPLIT_DIR      = PROJECT_ROOT / "split_dataset"
RESULTS_DIR    = PROJECT_ROOT / "results" / "dataset"

CLASS_NAMES = ["EUS", "gill", "healthy", "red_spot"]
CLASS_LABELS = {
    "EUS"      : "EUS (Epizootic Ulcerative Syndrome)",
    "gill"     : "Gill Disease",
    "healthy"  : "Healthy",
    "red_spot" : "Red Spot Disease",
}
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

SPLITS = ["train", "validation", "test"]

SEED = 42
random.seed(SEED)

# Create output directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    total = 0
    for ext in ["*.jpg", "*.jpeg", "*.png",
                "*.JPG", "*.JPEG", "*.PNG"]:
        total += len(list(folder.glob(ext)))
    return total


def get_image_files(folder: Path) -> list:
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png",
                "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(folder.glob(ext))
    return files


def load_image_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# 1. COUNT IMAGES
# ============================================================

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

# --- Original ---
original_counts = {c: count_images(ORIGINAL_DIR / c) for c in CLASS_NAMES}
original_total  = sum(original_counts.values())

# --- Augmented ---
augmented_counts = {c: count_images(AUGMENTED_DIR / c) for c in CLASS_NAMES}
augmented_total  = sum(augmented_counts.values())

# --- Split ---
split_counts = {}
for split in SPLITS:
    split_counts[split] = {
        c: count_images(SPLIT_DIR / split / c) for c in CLASS_NAMES
    }

split_totals = {s: sum(split_counts[s].values()) for s in SPLITS}
grand_total  = sum(split_totals.values())

print(f"\nOriginal dataset total  : {original_total}")
print(f"Augmented dataset total : {augmented_total}")
print(f"Split dataset total     : {grand_total}")
print(f"  Train      : {split_totals['train']}")
print(f"  Validation : {split_totals['validation']}")
print(f"  Test       : {split_totals['test']}")


# ============================================================
# 2. IMAGE DIMENSION STATISTICS (sample 50 per class)
# ============================================================

print("\nSampling image dimensions...")

dim_stats = {}

for cls in CLASS_NAMES:
    files = get_image_files(AUGMENTED_DIR / cls)
    sample = random.sample(files, min(50, len(files)))
    heights, widths, channels_list = [], [], []
    for f in sample:
        img = cv2.imread(str(f))
        if img is not None:
            h, w, c = img.shape
            heights.append(h)
            widths.append(w)
            channels_list.append(c)
    dim_stats[cls] = {
        "height_mean" : round(float(np.mean(heights)),  2),
        "height_std"  : round(float(np.std(heights)),   2),
        "width_mean"  : round(float(np.mean(widths)),   2),
        "width_std"   : round(float(np.std(widths)),    2),
        "channels"    : int(np.median(channels_list)),
    }
    print(f"  {cls}: H={dim_stats[cls]['height_mean']} "
          f"W={dim_stats[cls]['width_mean']}")


# ============================================================
# 3. BUILD CSV
# ============================================================

print("\nBuilding statistics CSV...")

rows = []
for cls in CLASS_NAMES:
    train_n = split_counts["train"][cls]
    val_n   = split_counts["validation"][cls]
    test_n  = split_counts["test"][cls]
    total_n = train_n + val_n + test_n

    rows.append({
        "class"              : cls,
        "class_label"        : CLASS_LABELS[cls],
        "original_images"    : original_counts[cls],
        "augmented_images"   : augmented_counts[cls],
        "train_images"       : train_n,
        "validation_images"  : val_n,
        "test_images"        : test_n,
        "total_split_images" : total_n,
        "train_pct"          : round(train_n / total_n * 100, 1),
        "validation_pct"     : round(val_n   / total_n * 100, 1),
        "test_pct"           : round(test_n  / total_n * 100, 1),
        "class_pct_of_total" : round(total_n / grand_total * 100, 1),
        "height_mean"        : dim_stats[cls]["height_mean"],
        "width_mean"         : dim_stats[cls]["width_mean"],
        "channels"           : dim_stats[cls]["channels"],
    })

df = pd.DataFrame(rows)
csv_path = RESULTS_DIR / "dataset_statistics.csv"
df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")


# ============================================================
# 4. CLASS DISTRIBUTION GRAPH
# ============================================================

print("\nGenerating class distribution graph...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Class Distribution — Fish Disease Dataset",
    fontsize=14, fontweight="bold"
)

# -- Bar chart: original vs augmented --
ax = axes[0]
x      = np.arange(len(CLASS_NAMES))
width  = 0.35
orig_vals = [original_counts[c]  for c in CLASS_NAMES]
aug_vals  = [augmented_counts[c] for c in CLASS_NAMES]

bars1 = ax.bar(x - width/2, orig_vals, width,
               label="Original",  color="#95a5a6", edgecolor="white")
bars2 = ax.bar(x + width/2, aug_vals,  width,
               label="Augmented", color=CLASS_COLORS, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(
    [CLASS_LABELS[c].replace(" ", "\n") for c in CLASS_NAMES],
    fontsize=8
)
ax.set_ylabel("Number of images")
ax.set_title("Original vs Augmented per Class")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 15,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 15,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=7)

# -- Pie chart: augmented distribution --
ax2 = axes[1]
aug_vals_pie = [augmented_counts[c] for c in CLASS_NAMES]
wedges, texts, autotexts = ax2.pie(
    aug_vals_pie,
    labels=[CLASS_LABELS[c] for c in CLASS_NAMES],
    colors=CLASS_COLORS,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in texts:
    t.set_fontsize(8)
for at in autotexts:
    at.set_fontsize(8)
ax2.set_title("Augmented Class Share (%)")

plt.tight_layout()
dist_path = RESULTS_DIR / "class_distribution.png"
plt.savefig(str(dist_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {dist_path}")


# ============================================================
# 5. SPLIT DISTRIBUTION GRAPH
# ============================================================

print("\nGenerating split distribution graph...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Train / Validation / Test Split Distribution",
    fontsize=14, fontweight="bold"
)

# -- Grouped bar per class --
ax = axes[0]
x      = np.arange(len(CLASS_NAMES))
width  = 0.25
split_colors = ["#2980b9", "#27ae60", "#e67e22"]

for i, split in enumerate(SPLITS):
    vals = [split_counts[split][c] for c in CLASS_NAMES]
    bars = ax.bar(x + (i - 1)*width, vals, width,
                  label=split.capitalize(),
                  color=split_colors[i],
                  edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 10,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(
    [CLASS_LABELS[c].replace(" ", "\n") for c in CLASS_NAMES],
    fontsize=8
)
ax.set_ylabel("Number of images")
ax.set_title("Images per Class per Split")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -- Pie: overall split proportions --
ax2 = axes[1]
split_total_vals = [split_totals[s] for s in SPLITS]
wedges, texts, autotexts = ax2.pie(
    split_total_vals,
    labels=[f"{s.capitalize()}\n({split_totals[s]})" for s in SPLITS],
    colors=split_colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in texts:
    t.set_fontsize(9)
for at in autotexts:
    at.set_fontsize(9)
ax2.set_title(f"Overall Split Proportions\n(Total: {grand_total})")

plt.tight_layout()
split_path = RESULTS_DIR / "split_distribution.png"
plt.savefig(str(split_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {split_path}")


# ============================================================
# 6. SAMPLE IMAGES (4 per class = 16 panels)
# ============================================================

print("\nGenerating sample image grid...")

fig, axes = plt.subplots(
    len(CLASS_NAMES), 4,
    figsize=(14, len(CLASS_NAMES) * 3.5)
)
fig.suptitle(
    "Sample Images — One Row per Class",
    fontsize=14, fontweight="bold", y=1.01
)

for row, cls in enumerate(CLASS_NAMES):
    files = get_image_files(AUGMENTED_DIR / cls)
    samples = random.sample(files, min(4, len(files)))

    for col in range(4):
        ax = axes[row][col]
        if col < len(samples):
            img = load_image_rgb(samples[col])
            if img is not None:
                ax.imshow(img)
        if col == 0:
            ax.set_ylabel(
                CLASS_LABELS[cls],
                fontsize=9,
                fontweight="bold",
                color=CLASS_COLORS[row],
                rotation=90,
                labelpad=6
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CLASS_COLORS[row])
            spine.set_linewidth(1.8)

plt.tight_layout()
samples_path = RESULTS_DIR / "sample_images.png"
plt.savefig(str(samples_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {samples_path}")


# ============================================================
# 7. IMAGE DIMENSION CHART
# ============================================================

print("\nGenerating image dimension chart...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Image Dimension Statistics (sampled 50 images/class)",
    fontsize=13, fontweight="bold"
)

x = np.arange(len(CLASS_NAMES))
width = 0.35
heights_mean = [dim_stats[c]["height_mean"] for c in CLASS_NAMES]
widths_mean  = [dim_stats[c]["width_mean"]  for c in CLASS_NAMES]
heights_std  = [dim_stats[c]["height_std"]  for c in CLASS_NAMES]
widths_std   = [dim_stats[c]["width_std"]   for c in CLASS_NAMES]

ax = axes[0]
ax.bar(x - width/2, heights_mean, width,
       yerr=heights_std, capsize=4,
       label="Height", color="#8e44ad", edgecolor="white", alpha=0.85)
ax.bar(x + width/2, widths_mean,  width,
       yerr=widths_std,  capsize=4,
       label="Width",  color="#16a085", edgecolor="white", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=9)
ax.set_ylabel("Pixels")
ax.set_title("Mean Height & Width per Class")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Scatter of sampled dims
ax2 = axes[1]
for i, cls in enumerate(CLASS_NAMES):
    files  = get_image_files(AUGMENTED_DIR / cls)
    sample = random.sample(files, min(50, len(files)))
    ws, hs = [], []
    for f in sample:
        img = cv2.imread(str(f))
        if img is not None:
            hs.append(img.shape[0])
            ws.append(img.shape[1])
    ax2.scatter(ws, hs, label=CLASS_LABELS[cls],
                color=CLASS_COLORS[i], alpha=0.6, s=30)

ax2.set_xlabel("Width (px)")
ax2.set_ylabel("Height (px)")
ax2.set_title("Width vs Height Scatter (50 samples/class)")
ax2.legend(fontsize=7)
ax2.grid(linestyle="--", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
dims_path = RESULTS_DIR / "image_dimensions.png"
plt.savefig(str(dims_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {dims_path}")


# ============================================================
# 8. TEXT REPORT
# ============================================================

print("\nWriting analysis report...")

report_lines = [
    "=" * 60,
    "FISH DISEASE DATASET — ANALYSIS REPORT",
    "=" * 60,
    "",
    "PROJECT: Fish Disease Classification (4-class)",
    "MODELS : VGG16, MobileNetV2, ConvNeXt",
    "",
    "CLASSES:",
    *[f"  {i+1}. {CLASS_LABELS[c]} ({c})" for i, c in enumerate(CLASS_NAMES)],
    "",
    "-" * 60,
    "ORIGINAL DATASET",
    "-" * 60,
    *[f"  {CLASS_LABELS[c]:<40} {original_counts[c]:>5} images"
      for c in CLASS_NAMES],
    f"  {'TOTAL':<40} {original_total:>5} images",
    "",
    "-" * 60,
    "AUGMENTED DATASET",
    "-" * 60,
    *[f"  {CLASS_LABELS[c]:<40} {augmented_counts[c]:>5} images"
      for c in CLASS_NAMES],
    f"  {'TOTAL':<40} {augmented_total:>5} images",
    "",
    "-" * 60,
    "SPLIT DATASET  (Train 60% / Validation 20% / Test 20%)",
    "-" * 60,
]

header = f"  {'Class':<25} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}"
report_lines.append(header)
report_lines.append("  " + "-" * 50)

for cls in CLASS_NAMES:
    t  = split_counts["train"][cls]
    v  = split_counts["validation"][cls]
    te = split_counts["test"][cls]
    tot = t + v + te
    report_lines.append(
        f"  {CLASS_LABELS[cls]:<25} {t:>7} {v:>7} {te:>7} {tot:>7}"
    )

report_lines += [
    "  " + "-" * 50,
    f"  {'TOTAL':<25} {split_totals['train']:>7} "
    f"{split_totals['validation']:>7} {split_totals['test']:>7} "
    f"{grand_total:>7}",
    f"  {'PERCENTAGE':<25} "
    f"{split_totals['train']/grand_total*100:>6.1f}% "
    f"{split_totals['validation']/grand_total*100:>6.1f}% "
    f"{split_totals['test']/grand_total*100:>6.1f}%",
    "",
    "-" * 60,
    "IMAGE DIMENSIONS (mean over 50 samples per class)",
    "-" * 60,
    f"  {'Class':<25} {'Height':>8} {'Width':>8} {'Channels':>10}",
    "  " + "-" * 54,
    *[f"  {CLASS_LABELS[c]:<25} {dim_stats[c]['height_mean']:>8} "
      f"{dim_stats[c]['width_mean']:>8} {dim_stats[c]['channels']:>10}"
      for c in CLASS_NAMES],
    "",
    "-" * 60,
    "OUTPUT FILES",
    "-" * 60,
    "  dataset_statistics.csv    — full per-class statistics",
    "  class_distribution.png    — original vs augmented bar + pie",
    "  split_distribution.png    — train/val/test breakdown",
    "  sample_images.png         — 4 sample images per class",
    "  image_dimensions.png      — height/width stats + scatter",
    "  analysis_report.txt       — this report",
    "",
    "=" * 60,
    "NEXT STAGE: Model building",
    "  python src/train.py",
    "=" * 60,
]

report_text = "\n".join(report_lines)
report_path = RESULTS_DIR / "analysis_report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(f"  Saved: {report_path}")

# Also print to console
print()
print(report_text)
