"""
extract_research_data.py
-------------------------
One-time / re-runnable script that reads the EXISTING research artifacts
under /home/user/webapp/results/ (produced by the original ML pipeline
in src/) and compiles them into a single static JSON file that the
FastAPI backend serves to the frontend.

This script does NOT modify anything inside results/ or src/. It only
reads those files and writes a new JSON file into
webapp/backend/static/research_data/research_data.json.

Every number in the output JSON is copied verbatim from the project's
own generated reports/CSVs — nothing is invented.
"""

import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_PATH = Path(__file__).resolve().parent / "static" / "research_data" / "research_data.json"


def read_csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_classification_report(text: str):
    """Parse sklearn classification_report plain text into structured data."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    rows = []
    accuracy = None
    macro = None
    weighted = None
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "accuracy":
            accuracy = float(parts[-2]) if len(parts) >= 2 else float(parts[-1])
            continue
        if parts[0] == "macro" and parts[1] == "avg":
            macro = {
                "precision": float(parts[2]),
                "recall": float(parts[3]),
                "f1_score": float(parts[4]),
                "support": int(parts[5]),
            }
            continue
        if parts[0] == "weighted" and parts[1] == "avg":
            weighted = {
                "precision": float(parts[2]),
                "recall": float(parts[3]),
                "f1_score": float(parts[4]),
                "support": int(parts[5]),
            }
            continue
        # class row: name precision recall f1 support
        if len(parts) == 5:
            name, p, r, f1, sup = parts
            rows.append({
                "class": name,
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f1),
                "support": int(sup),
            })
    return {
        "per_class": rows,
        "accuracy": accuracy,
        "macro_avg": macro,
        "weighted_avg": weighted,
    }


def main():
    data = {}

    # ---------------------------------------------------------
    # 1. Final model metrics (MobileNetV2 + ConvNeXt + Poly SVM)
    # ---------------------------------------------------------
    final_metrics = read_json(RESULTS_DIR / "final" / "reports" / "final_metrics.json")
    final_cm_rows = read_csv_rows(RESULTS_DIR / "final" / "reports" / "final_confusion_matrix.csv")
    final_per_class = read_csv_rows(RESULTS_DIR / "final" / "reports" / "final_per_class_metrics.csv")
    final_auc_rows = read_csv_rows(RESULTS_DIR / "final" / "reports" / "final_auc_results.csv")
    final_cls_report_text = read_text(RESULTS_DIR / "final" / "reports" / "final_classification_report.txt")
    final_config = read_json(RESULTS_DIR / "final" / "final_model" / "final_model_configuration.json")

    class_order = ["EUS", "gill", "healthy", "red_spot"]
    cm_matrix = []
    for row in final_cm_rows:
        cm_matrix.append([int(row[c]) for c in class_order])

    error_analysis_rows = read_csv_rows(RESULTS_DIR / "final" / "analysis" / "error_analysis.csv")

    data["final_model"] = {
        "name": "MobileNetV2 + ConvNeXt + Polynomial SVM",
        "config": final_config,
        "metrics": final_metrics,
        "confusion_matrix": {
            "labels": class_order,
            "matrix": cm_matrix,
        },
        "per_class_metrics": [
            {
                "class": r["Class"],
                "precision": float(r["Precision"]),
                "recall": float(r["Recall"]),
                "f1_score": float(r["F1_Score"]),
                "support": int(r["Support"]),
            }
            for r in final_per_class
        ],
        "classification_report": parse_classification_report(final_cls_report_text),
        "classification_report_text": final_cls_report_text,
        "auc": {
            r["Class"]: float(r["AUC"]) for r in final_auc_rows
        },
        "error_analysis": [
            {
                "true_class": r["True_Class"],
                "predicted_class": r["Predicted_Class"],
                "count": int(r["Count"]),
            }
            for r in error_analysis_rows
        ],
    }

    # ---------------------------------------------------------
    # 2. Individual CNN comparison (VGG16 / MobileNetV2 / ConvNeXt)
    # ---------------------------------------------------------
    comparison_rows = read_csv_rows(RESULTS_DIR / "comparison" / "comparison_table.csv")
    data["cnn_comparison"] = [
        {
            "model": r["Model"],
            "test_accuracy": float(r["Test_Accuracy"]),
            "test_precision": float(r["Test_Precision"]),
            "test_recall": float(r["Test_Recall"]),
            "test_f1": float(r["Test_F1"]),
            "best_val_accuracy": float(r["Best_Val_Acc"]),
            "epochs": int(r["Epochs"]),
            "training_minutes": float(r["Training_Min"]),
        }
        for r in comparison_rows
    ]
    # append the final fused model for a complete "CNN Comparison" table as requested
    data["cnn_comparison"].append({
        "model": "MobileNetV2 + ConvNeXt + Polynomial SVM",
        "test_accuracy": final_metrics["accuracy"],
        "test_precision": final_metrics["weighted_precision"],
        "test_recall": final_metrics["weighted_recall"],
        "test_f1": final_metrics["weighted_f1"],
        "best_val_accuracy": None,
        "epochs": None,
        "training_minutes": None,
        "is_final": True,
    })

    # ---------------------------------------------------------
    # 3. Fusion experiment results (7 combinations)
    # ---------------------------------------------------------
    fusion_rows = read_csv_rows(RESULTS_DIR / "fusion" / "fusion_results.csv")
    data["fusion_results"] = [
        {
            "combination": r["Combination"],
            "feature_dim": int(r["Feature_Dim"]),
            "accuracy": float(r["Accuracy"]),
            "precision": float(r["Precision"]),
            "recall": float(r["Recall"]),
            "f1_score": float(r["F1_Score"]),
            "train_time_s": float(r["Train_Time_s"]),
            "is_highlighted": r["Combination"] == "MobileNetV2+ConvNeXt",
        }
        for r in fusion_rows
    ]

    # ---------------------------------------------------------
    # 4. PCA vs No-PCA comparison (phase 4)
    # ---------------------------------------------------------
    pca_rows = read_csv_rows(RESULTS_DIR / "pca_svm" / "phase4_pca_svm_results.csv")
    data["pca_comparison"] = [
        {
            "combination": r["Combination"],
            "experiment": r["Experiment"],
            "original_dim": int(r["Original_Feature_Dim"]),
            "pca_dim": int(r["PCA_Feature_Dim"]),
            "variance_retained_pct": float(r["Variance_Retained_%"]),
            "accuracy": float(r["Accuracy"]),
            "precision": float(r["Precision"]),
            "recall": float(r["Recall"]),
            "f1_score": float(r["F1_Score"]),
        }
        for r in pca_rows
    ]

    # ---------------------------------------------------------
    # 5. SVM Kernel comparison (phase 5)
    # ---------------------------------------------------------
    kernel_rows = read_csv_rows(RESULTS_DIR / "svm_kernel" / "phase5_svm_kernel_results.csv")
    data["kernel_comparison"] = [
        {
            "combination": r["Combination"],
            "kernel": r["Kernel"],
            "feature_dim": int(r["Feature_Dimension"]),
            "accuracy": float(r["Accuracy"]),
            "precision": float(r["Precision"]),
            "recall": float(r["Recall"]),
            "f1_score": float(r["F1_Score"]),
        }
        for r in kernel_rows
    ]

    # ---------------------------------------------------------
    # 6. Dataset statistics
    # ---------------------------------------------------------
    dataset_rows = read_csv_rows(RESULTS_DIR / "dataset" / "dataset_statistics.csv")
    data["dataset_statistics"] = [
        {
            "class": r["class"],
            "class_label": r["class_label"],
            "original_images": int(r["original_images"]),
            "augmented_images": int(r["augmented_images"]),
            "train_images": int(r["train_images"]),
            "validation_images": int(r["validation_images"]),
            "test_images": int(r["test_images"]),
        }
        for r in dataset_rows
    ]

    # ---------------------------------------------------------
    # 7. Individual model metrics.json (VGG16, MobileNetV2, ConvNeXt)
    # ---------------------------------------------------------
    data["individual_models"] = {
        "VGG16": read_json(RESULTS_DIR / "VGG16" / "reports" / "metrics.json"),
        "MobileNetV2": read_json(RESULTS_DIR / "MobileNetV2" / "reports" / "metrics.json"),
        "ConvNeXt": read_json(RESULTS_DIR / "ConvNeXt" / "reports" / "metrics.json"),
    }

    # ---------------------------------------------------------
    # 8. Sample error examples with probability scores
    # ---------------------------------------------------------
    incorrect_rows = read_csv_rows(RESULTS_DIR / "final" / "analysis" / "incorrect_predictions.csv")
    data["incorrect_prediction_samples"] = [
        {
            "sample_index": int(r["sample_index"]),
            "true_class": r["true_class"],
            "predicted_class": r["predicted_class"],
            "confidence": round(float(r["confidence"]), 4),
            "prob_EUS": round(float(r["prob_EUS"]), 4),
            "prob_gill": round(float(r["prob_gill"]), 4),
            "prob_healthy": round(float(r["prob_healthy"]), 4),
            "prob_red_spot": round(float(r["prob_red_spot"]), 4),
        }
        for r in incorrect_rows[:30]
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Research data written to: {OUT_PATH}")
    print(f"Top-level keys: {list(data.keys())}")


if __name__ == "__main__":
    main()
