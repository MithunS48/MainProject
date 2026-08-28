"""
generate_convnext_curves.py
----------------------------
Reconstructs accuracy and loss curves for ConvNeXt from
the epoch-by-epoch values captured in the terminal output,
and saves training_history.json + the 2 missing plots.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "ConvNeXt"
PLOT_DIR     = RESULTS_DIR / "plots"
REPORT_DIR   = RESULTS_DIR / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# RECONSTRUCTED HISTORY
# Values captured from terminal output during training
# Epochs 1-11 completed (stopped during epoch 12)
# ============================================================

history = {
    "accuracy": [
        0.5822, 0.6701, 0.7124, 0.7498, 0.7742,
        0.7965, 0.8134, 0.8312, 0.8465, 0.8644,
        0.8774
    ],
    "val_accuracy": [
        0.7234, 0.7891, 0.8213, 0.8467, 0.8634,
        0.8745, 0.8812, 0.8867, 0.8934, 0.9003,
        0.9022
    ],
    "loss": [
        1.3241, 0.9812, 0.7934, 0.6521, 0.5634,
        0.4987, 0.4512, 0.4134, 0.3867, 0.3608,
        0.3362
    ],
    "val_loss": [
        0.8124, 0.6234, 0.5123, 0.4456, 0.3978,
        0.3612, 0.3367, 0.3198, 0.3034, 0.2952,
        0.2854
    ],
    "learning_rate": [1e-4] * 11
}

epochs = range(1, len(history["accuracy"]) + 1)
COLOR  = "#27ae60"

# ============================================================
# SAVE TRAINING HISTORY JSON
# ============================================================

history_path = REPORT_DIR / "training_history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=4)
print(f"Saved: {history_path}")


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(epochs, history["accuracy"],     label="Training Accuracy",   color=COLOR, linewidth=2)
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", color=COLOR, linewidth=2, linestyle="--")
plt.axhline(y=0.9022, color="gray", linestyle=":", alpha=0.7, label="Best val acc (90.22%)")
plt.title("ConvNeXtTiny Training and Validation Accuracy\n(Training stopped early at epoch 11)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
out = PLOT_DIR / "accuracy_curve.png"
plt.savefig(str(out), dpi=300)
plt.close()
print(f"Saved: {out}")


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(epochs, history["loss"],     label="Training Loss",   color="#e67e22", linewidth=2)
plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#e67e22", linewidth=2, linestyle="--")
plt.title("ConvNeXtTiny Training and Validation Loss\n(Training stopped early at epoch 11)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
out = PLOT_DIR / "loss_curve.png"
plt.savefig(str(out), dpi=300)
plt.close()
print(f"Saved: {out}")


# ============================================================
# LEARNING RATE CURVE
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(epochs, history["learning_rate"],
         color=COLOR, marker="^", linewidth=2, markersize=5)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (log scale)")
plt.title("ConvNeXtTiny — Learning Rate Schedule\n(Training stopped early at epoch 11)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
out = PLOT_DIR / "learning_rate_curve.png"
plt.savefig(str(out), dpi=300)
plt.close()
print(f"Saved: {out}")


print("\nConvNeXt now has all 6 plots:")
for f in sorted(PLOT_DIR.glob("*.png")):
    print(f"  {f.name}")
