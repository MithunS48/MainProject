"""
fusion_experiments.py
----------------------
<<<<<<< HEAD
Runs all 3 fusion combinations using extracted features:
  1. VGG16 alone
  2. MobileNetV2 alone
  3. VGG16 + MobileNetV2 (concatenation)

Each combination is evaluated using a simple Linear SVM classifier
to confirm the feature quality before Phase 4 (PCA) and Phase 5
(full SVM experiments).

Results saved to results/fusion/:
  fusion_results.csv
  fusion_comparison.png
  <combination>/confusion_matrix.png
  <combination>/classification_report.txt
=======
Runs all 7 fusion combinations using extracted features:
  1. VGG16 alone
  2. MobileNetV2 alone
  3. ConvNeXt alone
  4. VGG16 + MobileNetV2
  5. VGG16 + ConvNeXt
  6. MobileNetV2 + ConvNeXt
  7. VGG16 + MobileNetV2 + ConvNeXt

Each combination uses LinearSVC with StandardScaler.
Results saved to results/fusion/.
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
import time

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR  = PROJECT_ROOT / "results" / "features"
FUSION_DIR   = PROJECT_ROOT / "results" / "fusion"
FUSION_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ["EUS", "gill", "healthy", "red_spot"]
CLASS_LABELS = ["EUS", "Gill\nDisease", "Healthy", "Red Spot\nDisease"]
NUM_CLASSES  = len(CLASS_NAMES)

print("=" * 65)
<<<<<<< HEAD
print("PHASE 3 — FUSION EXPERIMENTS")
=======
print("PHASE 3 — FUSION EXPERIMENTS (7 combinations)")
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
print("=" * 65)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading extracted features...")

def load_npz(path):
    data = np.load(str(path))
    return data["features"], data["labels"]

vgg16_train,    y_train = load_npz(FEATURE_DIR / "vgg16_train.npz")
vgg16_val,      y_val   = load_npz(FEATURE_DIR / "vgg16_validation.npz")
vgg16_test,     y_test  = load_npz(FEATURE_DIR / "vgg16_test.npz")

mob_train,  _ = load_npz(FEATURE_DIR / "mobilenet_train.npz")
mob_val,    _ = load_npz(FEATURE_DIR / "mobilenet_validation.npz")
mob_test,   _ = load_npz(FEATURE_DIR / "mobilenet_test.npz")

<<<<<<< HEAD
=======
cx_train,   _ = load_npz(FEATURE_DIR / "convnext_train.npz")
cx_val,     _ = load_npz(FEATURE_DIR / "convnext_validation.npz")
cx_test,    _ = load_npz(FEATURE_DIR / "convnext_test.npz")

>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
fused_train, _ = load_npz(FEATURE_DIR / "fused_train.npz")
fused_val,   _ = load_npz(FEATURE_DIR / "fused_validation.npz")
fused_test,  _ = load_npz(FEATURE_DIR / "fused_test.npz")

<<<<<<< HEAD
# Combine train+val for fitting the classifier
# (test stays completely separate)
vgg16_trainval  = np.concatenate([vgg16_train, vgg16_val], axis=0)
mob_trainval    = np.concatenate([mob_train,   mob_val],   axis=0)
fused_trainval  = np.concatenate([fused_train, fused_val], axis=0)
y_trainval      = np.concatenate([y_train,     y_val],     axis=0)

print(f"  Train+Val samples : {len(y_trainval)}")
print(f"  Test samples      : {len(y_test)}")

print("\nFeature dimensions:")
print(f"  VGG16       : {vgg16_train.shape[1]}")
print(f"  MobileNetV2 : {mob_train.shape[1]}")
print(f"  Fused       : {fused_train.shape[1]}")


# ============================================================
# DEFINE COMBINATIONS
=======
fused_all_train, _ = load_npz(FEATURE_DIR / "fused_all_train.npz")
fused_all_val,   _ = load_npz(FEATURE_DIR / "fused_all_validation.npz")
fused_all_test,  _ = load_npz(FEATURE_DIR / "fused_all_test.npz")

# Combine train+val for SVM fitting
y_trainval = np.concatenate([y_train, y_val], axis=0)

def tv(a, b):
    return np.concatenate([a, b], axis=0)

print(f"  Train+Val : {len(y_trainval)}  |  Test : {len(y_test)}")
print(f"  VGG16 dim : {vgg16_train.shape[1]}")
print(f"  Mobile dim: {mob_train.shape[1]}")
print(f"  ConvNeXt  : {cx_train.shape[1]}")


# ============================================================
# DEFINE ALL 7 COMBINATIONS
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
# ============================================================

COMBINATIONS = {
    "VGG16": {
<<<<<<< HEAD
        "X_trainval" : vgg16_trainval,
        "X_test"     : vgg16_test,
        "feature_dim": vgg16_train.shape[1],
        "color"      : "#e74c3c",
    },
    "MobileNetV2": {
        "X_trainval" : mob_trainval,
        "X_test"     : mob_test,
        "feature_dim": mob_train.shape[1],
        "color"      : "#2980b9",
    },
    "VGG16+MobileNetV2": {
        "X_trainval" : fused_trainval,
        "X_test"     : fused_test,
        "feature_dim": fused_train.shape[1],
        "color"      : "#8e44ad",
=======
        "X_tv"  : tv(vgg16_train, vgg16_val),
        "X_test": vgg16_test,
        "dim"   : vgg16_train.shape[1],
        "color" : "#e74c3c",
    },
    "MobileNetV2": {
        "X_tv"  : tv(mob_train, mob_val),
        "X_test": mob_test,
        "dim"   : mob_train.shape[1],
        "color" : "#2980b9",
    },
    "ConvNeXt": {
        "X_tv"  : tv(cx_train, cx_val),
        "X_test": cx_test,
        "dim"   : cx_train.shape[1],
        "color" : "#27ae60",
    },
    "VGG16+MobileNetV2": {
        "X_tv"  : tv(fused_train, fused_val),
        "X_test": fused_test,
        "dim"   : fused_train.shape[1],
        "color" : "#8e44ad",
    },
    "VGG16+ConvNeXt": {
        "X_tv"  : tv(np.concatenate([vgg16_train, cx_train], axis=1),
                     np.concatenate([vgg16_val,   cx_val],   axis=1)),
        "X_test": np.concatenate([vgg16_test, cx_test], axis=1),
        "dim"   : vgg16_train.shape[1] + cx_train.shape[1],
        "color" : "#e67e22",
    },
    "MobileNetV2+ConvNeXt": {
        "X_tv"  : tv(np.concatenate([mob_train, cx_train], axis=1),
                     np.concatenate([mob_val,   cx_val],   axis=1)),
        "X_test": np.concatenate([mob_test, cx_test], axis=1),
        "dim"   : mob_train.shape[1] + cx_train.shape[1],
        "color" : "#16a085",
    },
    "VGG16+MobileNetV2+ConvNeXt": {
        "X_tv"  : tv(fused_all_train, fused_all_val),
        "X_test": fused_all_test,
        "dim"   : fused_all_train.shape[1],
        "color" : "#2c3e50",
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
    },
}


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

for combo_name, cfg in COMBINATIONS.items():

    print(f"\n{'='*65}")
    print(f"COMBINATION: {combo_name}")
    print(f"{'='*65}")
<<<<<<< HEAD
    print(f"  Feature dim : {cfg['feature_dim']}")
=======
    print(f"  Feature dim : {cfg['dim']}")
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c

    combo_dir = FUSION_DIR / combo_name.replace("+", "_plus_")
    combo_dir.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
    # Pipeline: StandardScaler + LinearSVC
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        ))
    ])

    # Train
    print(f"  Training LinearSVC...")
    t0 = time.time()
    clf.fit(cfg["X_trainval"], y_trainval)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.2f}s")

    # Predict
=======
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(C=1.0, max_iter=5000, random_state=42, dual="auto"))
    ])

    print(f"  Training LinearSVC...")
    t0 = time.time()
    clf.fit(cfg["X_tv"], y_trainval)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.2f}s")

>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
    y_pred = clf.predict(cfg["X_test"])

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy  : {accuracy*100:.2f}%")
    print(f"  Test Precision : {precision:.4f}")
    print(f"  Test Recall    : {recall:.4f}")
    print(f"  Test F1        : {f1:.4f}")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print(f"\n{report}")

    report_path = combo_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Per-class metrics
    report_dict = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title(
        f"Confusion Matrix — {combo_name}\n"
        f"Accuracy: {accuracy*100:.2f}%",
        fontweight="bold"
    )
    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES)
    threshold = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=11)
    ax.set_ylabel("True Class")
    ax.set_xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(str(combo_dir / "confusion_matrix.png"), dpi=300)
    plt.close()

    # Store results
    results.append({
        "Combination"    : combo_name,
<<<<<<< HEAD
        "Feature_Dim"    : cfg["feature_dim"],
=======
        "Feature_Dim"    : cfg["dim"],
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
        "Accuracy"       : round(accuracy,  4),
        "Precision"      : round(precision, 4),
        "Recall"         : round(recall,    4),
        "F1_Score"       : round(f1,        4),
        "Train_Time_s"   : round(train_time, 2),
        "EUS_F1"         : round(report_dict["EUS"]["f1-score"],      4),
        "Gill_F1"        : round(report_dict["gill"]["f1-score"],     4),
        "Healthy_F1"     : round(report_dict["healthy"]["f1-score"],  4),
        "RedSpot_F1"     : round(report_dict["red_spot"]["f1-score"], 4),
<<<<<<< HEAD
        "y_pred"         : y_pred,  # kept for plotting, dropped before CSV
=======
        "y_pred"         : y_pred,
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
        "color"          : cfg["color"],
        "cm"             : cm,
    })


# ============================================================
# SAVE RESULTS CSV
# ============================================================

df_rows = [{k: v for k, v in r.items()
            if k not in ("y_pred", "color", "cm")}
           for r in results]

results_df = pd.DataFrame(df_rows)
csv_path = FUSION_DIR / "fusion_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved: {csv_path}")
print("\n" + results_df.to_string(index=False))


# ============================================================
# COMPARISON CHART
# ============================================================

print("\nGenerating comparison chart...")

combo_names = [r["Combination"] for r in results]
colors      = [r["color"]       for r in results]

<<<<<<< HEAD
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "Phase 3 — Fusion Experiments: Feature Combination Comparison\n"
=======
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Phase 3 — All 7 Fusion Combinations\n"
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
    "Classifier: LinearSVC  |  Features: GAP after pretrained CNN",
    fontsize=14, fontweight="bold"
)

<<<<<<< HEAD
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel 1: Overall metrics bar ---
ax1 = fig.add_subplot(gs[0, 0:2])
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score"]
x = np.arange(len(combo_names))
width = 0.2
metric_colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

for i, (metric, mc) in enumerate(zip(metrics_to_plot, metric_colors)):
    vals = [r[metric] for r in results]
    bars = ax1.bar(x + (i - 1.5) * width, vals, width,
                   label=metric.replace("_", " "),
                   color=mc, alpha=0.85, edgecolor="white")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=7)

ax1.set_xticks(x)
ax1.set_xticklabels(combo_names, fontsize=9)
ax1.set_ylim(0.80, 1.02)
ax1.set_ylabel("Score")
ax1.set_title("Overall Metrics by Combination", fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Panel 2: Accuracy only (big, easy to read) ---
ax2 = fig.add_subplot(gs[0, 2])
acc_vals = [r["Accuracy"] * 100 for r in results]
bars = ax2.barh(combo_names, acc_vals, color=colors, alpha=0.85, edgecolor="white")
for bar in bars:
    ax2.text(bar.get_width() - 0.5,
             bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.2f}%",
             ha="right", va="center",
             color="white", fontsize=11, fontweight="bold")
ax2.set_xlim(80, 102)
ax2.set_title("Test Accuracy (%)", fontweight="bold")
=======
# --- Panel 1: Accuracy bar ---
ax1 = axes[0]
acc_vals = [r["Accuracy"] * 100 for r in results]
bars = ax1.barh(combo_names, acc_vals, color=colors, alpha=0.85, edgecolor="white")
for bar in bars:
    ax1.text(bar.get_width() - 0.3,
             bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.2f}%",
             ha="right", va="center",
             color="white", fontsize=10, fontweight="bold")
ax1.set_xlim(80, 103)
ax1.set_title("Test Accuracy (%)", fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Panel 2: F1 bar ---
ax2 = axes[1]
f1_vals = [r["F1_Score"] for r in results]
bars2 = ax2.barh(combo_names, f1_vals, color=colors, alpha=0.85, edgecolor="white")
for bar in bars2:
    ax2.text(bar.get_width() - 0.003,
             bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.4f}",
             ha="right", va="center",
             color="white", fontsize=10, fontweight="bold")
ax2.set_xlim(0.80, 1.03)
ax2.set_title("F1-Score (weighted)", fontweight="bold")
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
ax2.grid(axis="x", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

<<<<<<< HEAD
# --- Panels 3,4,5: Per-class F1 per combination ---
per_class_keys = ["EUS_F1", "Gill_F1", "Healthy_F1", "RedSpot_F1"]
per_class_labels = ["EUS", "Gill", "Healthy", "Red Spot"]
class_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

for idx, result in enumerate(results):
    ax = fig.add_subplot(gs[1, idx])
    f1_vals = [result[k] for k in per_class_keys]
    bars = ax.bar(per_class_labels, f1_vals,
                  color=class_colors, alpha=0.85, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("F1-Score")
    ax.set_title(f"{result['Combination']}\nPer-Class F1",
                 fontweight="bold", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

=======
>>>>>>> bea697917103835bf2ea40c5c3574d2a34c6e35c
plt.tight_layout()
chart_path = FUSION_DIR / "fusion_comparison.png"
plt.savefig(str(chart_path), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {chart_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

best = max(results, key=lambda r: r["Accuracy"])

print("\n" + "=" * 65)
print("PHASE 3 COMPLETE")
print("=" * 65)

print(f"\n{'Combination':<25} {'Accuracy':>10} {'F1':>10} {'Dim':>8}")
print("-" * 55)
for r in results:
    marker = " ← BEST" if r["Combination"] == best["Combination"] else ""
    print(f"{r['Combination']:<25} {r['Accuracy']*100:>9.2f}% {r['F1_Score']:>10.4f} {r['Feature_Dim']:>8}{marker}")

print(f"\nBest combination : {best['Combination']}")
print(f"Best accuracy    : {best['Accuracy']*100:.2f}%")
print(f"Best F1-score    : {best['F1_Score']:.4f}")

print(f"\nResults saved to : {FUSION_DIR}")
print("\nNext step: python src/pca_svm_experiments.py")
