"""
generate_analysis.py
---------------------
Generates enhanced analysis graphs and reports for both
VGG16 and MobileNetV2 using already-saved results.

New outputs per model (saved alongside existing results):
  plots/
    per_class_bar.png       — per-class precision/recall/F1 bar chart
    learning_rate_curve.png — learning rate schedule over epochs
    training_summary.png    — combined 4-panel dashboard
  reports/
    summary_card.txt        — one-page model summary card

Also generates:
  results/comparison/
    model_comparison.png    — side-by-side accuracy/F1/precision/recall
    comparison_table.csv    — comparison table
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

MODELS = {
    "VGG16": {
        "results_dir" : RESULTS_DIR / "VGG16",
        "color"       : "#e74c3c",
        "marker"      : "o",
    },
    "MobileNetV2": {
        "results_dir" : RESULTS_DIR / "MobileNetV2",
        "color"       : "#2980b9",
        "marker"      : "s",
    },
}

CLASS_NAMES  = ["EUS", "gill", "healthy", "red_spot"]
CLASS_LABELS = ["EUS", "Gill\nDisease", "Healthy", "Red Spot\nDisease"]

COMPARISON_DIR = RESULTS_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_csv(path):
    return pd.read_csv(path)


# ============================================================
# LOAD DATA FOR BOTH MODELS
# ============================================================

model_data = {}

for model_name, cfg in MODELS.items():
    rd = cfg["results_dir"]
    model_data[model_name] = {
        "history"     : load_json(rd / "reports" / "training_history.json"),
        "metrics"     : load_json(rd / "reports" / "metrics.json"),
        "per_class"   : load_csv(rd  / "reports" / "per_class_metrics.csv"),
        "plot_dir"    : rd / "plots",
        "report_dir"  : rd / "reports",
        "color"       : cfg["color"],
        "marker"      : cfg["marker"],
    }
    print(f"Loaded {model_name} data")


# ============================================================
# 1. PER-CLASS BAR CHART
# ============================================================

print("\nGenerating per-class bar charts...")

for model_name, data in model_data.items():
    df     = data["per_class"]
    color  = data["color"]

    x      = np.arange(len(CLASS_NAMES))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_p = ax.bar(x - width, df["Precision"], width,
                    label="Precision", color=color,     alpha=0.85, edgecolor="white")
    bars_r = ax.bar(x,          df["Recall"],   width,
                    label="Recall",    color=color,     alpha=0.55, edgecolor="white")
    bars_f = ax.bar(x + width,  df["F1"],       width,
                    label="F1-Score",  color="#2ecc71", alpha=0.85, edgecolor="white")

    # Value labels
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=10)
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Per-Class Precision / Recall / F1", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = data["plot_dir"] / "per_class_bar.png"
    plt.savefig(str(out), dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# 2. LEARNING RATE CURVE
# ============================================================

print("\nGenerating learning rate curves...")

for model_name, data in model_data.items():
    lr_vals = data["history"].get("learning_rate", [])
    if not lr_vals:
        print(f"  {model_name}: no LR data, skipping")
        continue

    epochs = range(1, len(lr_vals) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lr_vals, color=data["color"],
            marker=data["marker"], linewidth=2, markersize=5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title(f"{model_name} — Learning Rate Schedule", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = data["plot_dir"] / "learning_rate_curve.png"
    plt.savefig(str(out), dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# 3. TRAINING SUMMARY DASHBOARD (4 panels)
# ============================================================

print("\nGenerating training summary dashboards...")

for model_name, data in model_data.items():
    hist    = data["history"]
    metrics = data["metrics"]
    df      = data["per_class"]
    color   = data["color"]

    epochs      = range(1, len(hist["accuracy"]) + 1)
    phase_split = None

    # Detect fine-tuning phase split (LR drop)
    lr_vals = hist.get("learning_rate", [])
    if lr_vals:
        for i in range(1, len(lr_vals)):
            if lr_vals[i] < lr_vals[i-1] * 0.5:
                phase_split = i
                break

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"{model_name} — Training Summary Dashboard\n"
        f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%  |  "
        f"F1: {metrics['test_f1']:.4f}  |  "
        f"224×224 images",
        fontsize=14, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # --- Panel 1: Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, hist["accuracy"],     label="Train",      color=color,     linewidth=2)
    ax1.plot(epochs, hist["val_accuracy"], label="Validation", color=color,     linewidth=2, linestyle="--")
    if phase_split:
        ax1.axvline(phase_split, color="gray", linestyle=":", label="Fine-tune start")
    ax1.set_title("Accuracy Curve", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel 2: Loss ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, hist["loss"],     label="Train",      color="#e67e22", linewidth=2)
    ax2.plot(epochs, hist["val_loss"], label="Validation", color="#e67e22", linewidth=2, linestyle="--")
    if phase_split:
        ax2.axvline(phase_split, color="gray", linestyle=":", label="Fine-tune start")
    ax2.set_title("Loss Curve", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --- Panel 3: Per-class F1 bar ---
    ax3 = fig.add_subplot(gs[1, 0])
    bar_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax3.bar(CLASS_LABELS, df["F1"], color=bar_colors, edgecolor="white", alpha=0.85)
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax3.set_ylim(0.80, 1.02)
    ax3.set_ylabel("F1-Score")
    ax3.set_title("Per-Class F1-Score", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # --- Panel 4: Overall metrics bar ---
    ax4 = fig.add_subplot(gs[1, 1])
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_vals  = [
        metrics["test_accuracy"],
        metrics["test_precision"],
        metrics["test_recall"],
        metrics["test_f1"]
    ]
    bar_h = ax4.barh(metric_names, metric_vals,
                     color=color, alpha=0.85, edgecolor="white")
    for bar in bar_h:
        ax4.text(bar.get_width() - 0.01,
                 bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.4f}",
                 ha="right", va="center", color="white", fontsize=10, fontweight="bold")
    ax4.set_xlim(0.85, 1.01)
    ax4.set_title("Overall Test Metrics", fontweight="bold")
    ax4.grid(axis="x", linestyle="--", alpha=0.4)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    plt.tight_layout()
    out = data["plot_dir"] / "training_summary.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# 4. SUMMARY CARD (text)
# ============================================================

print("\nGenerating summary cards...")

for model_name, data in model_data.items():
    metrics  = data["metrics"]
    df       = data["per_class"]
    hist     = data["history"]
    epochs_n = len(hist["accuracy"])

    lines = [
        "=" * 55,
        f"  {model_name} — MODEL SUMMARY CARD",
        "=" * 55,
        "",
        f"  Image size       : 224 × 224 × 3",
        f"  Batch size       : {metrics.get('batch_size', 32)}",
        f"  Epochs completed : {epochs_n}",
        f"  Learning rate    : {metrics.get('learning_rate', 0.0001)}",
        f"  Total parameters : {metrics.get('total_parameters', 'N/A'):,}" if isinstance(metrics.get('total_parameters'), int) else f"  Total parameters : N/A",
        f"  Training time    : {metrics.get('training_time_minutes', 0):.1f} minutes",
        "",
        "-" * 55,
        "  TEST PERFORMANCE",
        "-" * 55,
        f"  Accuracy         : {metrics['test_accuracy']*100:.2f}%",
        f"  Precision        : {metrics['test_precision']:.4f}",
        f"  Recall           : {metrics['test_recall']:.4f}",
        f"  F1-Score         : {metrics['test_f1']:.4f}",
        f"  Best val acc     : {metrics.get('best_validation_accuracy', 0)*100:.2f}%",
        "",
        "-" * 55,
        "  PER-CLASS F1-SCORE",
        "-" * 55,
    ]

    for _, row in df.iterrows():
        lines.append(
            f"  {row['Class']:<12} P={row['Precision']:.4f}  "
            f"R={row['Recall']:.4f}  F1={row['F1']:.4f}"
        )

    lines += [
        "",
        "-" * 55,
        "  OUTPUT FILES",
        "-" * 55,
        "  plots/accuracy_curve.png",
        "  plots/loss_curve.png",
        "  plots/confusion_matrix.png",
        "  plots/per_class_bar.png",
        "  plots/learning_rate_curve.png",
        "  plots/training_summary.png",
        "  reports/classification_report.txt",
        "  reports/per_class_metrics.csv",
        "  reports/confusion_matrix.csv",
        "  reports/metrics.json",
        "  reports/training_history.json",
        "  reports/summary_card.txt",
        "=" * 55,
    ]

    card_text = "\n".join(lines)
    out = data["report_dir"] / "summary_card.txt"
    out.write_text(card_text, encoding="utf-8")
    print(f"  Saved: {out}")
    print()
    print(card_text)


# ============================================================
# 5. MODEL COMPARISON CHART
# ============================================================

print("\nGenerating model comparison chart...")

vgg_m    = model_data["VGG16"]["metrics"]
mob_m    = model_data["MobileNetV2"]["metrics"]

metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
vgg_vals  = [vgg_m["test_accuracy"], vgg_m["test_precision"],
             vgg_m["test_recall"],   vgg_m["test_f1"]]
mob_vals  = [mob_m["test_accuracy"], mob_m["test_precision"],
             mob_m["test_recall"],   mob_m["test_f1"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "VGG16 vs MobileNetV2 — Model Comparison\nFish Disease Classification (4 classes, 224×224)",
    fontsize=14, fontweight="bold"
)

# --- Bar comparison ---
ax = axes[0]
x     = np.arange(len(metric_names))
width = 0.35

bars1 = ax.bar(x - width/2, vgg_vals, width,
               label="VGG16",       color="#e74c3c", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, mob_vals, width,
               label="MobileNetV2", color="#2980b9", alpha=0.85, edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Score")
ax.set_title("Overall Metrics Comparison", fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Per-class F1 comparison ---
ax2 = axes[1]
vgg_f1 = model_data["VGG16"]["per_class"]["F1"].values
mob_f1 = model_data["MobileNetV2"]["per_class"]["F1"].values
x2 = np.arange(len(CLASS_NAMES))

bars3 = ax2.bar(x2 - width/2, vgg_f1, width,
                label="VGG16",       color="#e74c3c", alpha=0.85, edgecolor="white")
bars4 = ax2.bar(x2 + width/2, mob_f1, width,
                label="MobileNetV2", color="#2980b9", alpha=0.85, edgecolor="white")

for bars in [bars3, bars4]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x2)
ax2.set_xticklabels(CLASS_LABELS, fontsize=9)
ax2.set_ylim(0.80, 1.02)
ax2.set_ylabel("F1-Score")
ax2.set_title("Per-Class F1-Score Comparison", fontweight="bold")
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
out = COMPARISON_DIR / "model_comparison.png"
plt.savefig(str(out), dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")


# ============================================================
# 6. COMPARISON TABLE CSV
# ============================================================

rows = []
for model_name, data in model_data.items():
    m  = data["metrics"]
    df = data["per_class"]
    rows.append({
        "Model"           : model_name,
        "Test_Accuracy"   : round(m["test_accuracy"],  4),
        "Test_Precision"  : round(m["test_precision"], 4),
        "Test_Recall"     : round(m["test_recall"],    4),
        "Test_F1"         : round(m["test_f1"],        4),
        "Best_Val_Acc"    : round(m.get("best_validation_accuracy", 0), 4),
        "Epochs"          : len(data["history"]["accuracy"]),
        "Training_Min"    : round(m.get("training_time_minutes", 0), 1),
        "Total_Params"    : m.get("total_parameters", "N/A"),
        "EUS_F1"          : round(df[df["Class"]=="EUS"]["F1"].values[0],    4),
        "Gill_F1"         : round(df[df["Class"]=="gill"]["F1"].values[0],   4),
        "Healthy_F1"      : round(df[df["Class"]=="healthy"]["F1"].values[0],4),
        "RedSpot_F1"      : round(df[df["Class"]=="red_spot"]["F1"].values[0],4),
    })

comp_df = pd.DataFrame(rows)
out_csv = COMPARISON_DIR / "comparison_table.csv"
comp_df.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")

print("\n" + comp_df.to_string(index=False))


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nNew files per model:")
print("  plots/per_class_bar.png")
print("  plots/learning_rate_curve.png")
print("  plots/training_summary.png")
print("  reports/summary_card.txt")
print("\nComparison files:")
print("  results/comparison/model_comparison.png")
print("  results/comparison/comparison_table.csv")
