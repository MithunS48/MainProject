"""
======================================================================
PHASE 5 — SVM KERNEL COMPARISON — RESUMABLE VERSION
======================================================================

Uses Phase 3 CNN features.

Feature configurations:
    1. VGG16
    2. MobileNetV2
    3. ConvNeXt
    4. VGG16 + MobileNetV2
    5. VGG16 + ConvNeXt
    6. MobileNetV2 + ConvNeXt
    7. VGG16 + MobileNetV2 + ConvNeXt

SVM kernels:
    - Linear
    - RBF
    - Polynomial

PCA:
    NOT USED.
    Phase 4 already compared PCA vs No PCA.

IMPORTANT:
    This script RESUMES existing experiments.
    Existing prediction files are not repeated.

Linear SVM:
    LinearSVC is used because it is much faster for linear
    classification than SVC(kernel="linear").

RBF / Polynomial:
    SVC is used.

======================================================================
"""

import os
import time
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_DIR = r"D:\MainProject"

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "features"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "svm_kernel"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

RANDOM_STATE = 42

SVM_C = 1.0

RBF_GAMMA = "scale"

POLY_DEGREE = 3

POLY_GAMMA = "scale"

POLY_COEF0 = 0.0

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]


# =====================================================================
# EXACT PHASE 3 FEATURE FILES
# =====================================================================

FEATURE_FILES = {

    "VGG16": {
        "train": "vgg16_train.npz",
        "validation": "vgg16_validation.npz",
        "test": "vgg16_test.npz"
    },

    "MobileNetV2": {
        "train": "mobilenet_train.npz",
        "validation": "mobilenet_validation.npz",
        "test": "mobilenet_test.npz"
    },

    "ConvNeXt": {
        "train": "convnext_train.npz",
        "validation": "convnext_validation.npz",
        "test": "convnext_test.npz"
    }
}


# =====================================================================
# FEATURE CONFIGURATIONS
# =====================================================================

FEATURE_CONFIGS = {

    "VGG16": [
        "VGG16"
    ],

    "MobileNetV2": [
        "MobileNetV2"
    ],

    "ConvNeXt": [
        "ConvNeXt"
    ],

    "VGG16_plus_MobileNetV2": [
        "VGG16",
        "MobileNetV2"
    ],

    "VGG16_plus_ConvNeXt": [
        "VGG16",
        "ConvNeXt"
    ],

    "MobileNetV2_plus_ConvNeXt": [
        "MobileNetV2",
        "ConvNeXt"
    ],

    "VGG16_plus_MobileNetV2_plus_ConvNeXt": [
        "VGG16",
        "MobileNetV2",
        "ConvNeXt"
    ]
}


# =====================================================================
# SVM CONFIGURATION
# =====================================================================

SVM_CONFIGS = {

    "Linear": {
        "kernel": "linear",
        "C": SVM_C
    },

    "RBF": {
        "kernel": "rbf",
        "C": SVM_C,
        "gamma": RBF_GAMMA
    },

    "Polynomial": {
        "kernel": "poly",
        "C": SVM_C,
        "degree": POLY_DEGREE,
        "gamma": POLY_GAMMA,
        "coef0": POLY_COEF0
    }
}


# =====================================================================
# CREATE DIRECTORIES
# =====================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("PHASE 5 — SVM KERNEL COMPARISON")
print("=" * 70)

print(
    f"Project directory : {PROJECT_DIR}"
)

print(
    f"Feature directory : {FEATURE_DIR}"
)

print(
    f"Output directory  : {OUTPUT_DIR}"
)

print()

print(
    "PCA               : NOT USED"
)

print(
    "Linear SVM        : LinearSVC"
)

print(
    "RBF SVM           : SVC"
)

print(
    "Polynomial SVM    : SVC"
)

print(
    f"SVM C             : {SVM_C}"
)

print(
    f"RBF gamma         : {RBF_GAMMA}"
)

print(
    f"Polynomial degree : {POLY_DEGREE}"
)

print(
    f"Random seed       : {RANDOM_STATE}"
)

print("=" * 70)


# =====================================================================
# CHECK FEATURE DIRECTORY
# =====================================================================

if not os.path.exists(FEATURE_DIR):

    raise FileNotFoundError(
        f"\nFeature directory not found:\n"
        f"{FEATURE_DIR}"
    )


# =====================================================================
# LOAD NPZ
# =====================================================================

def load_npz(path):

    print(
        f"\nLoading: {os.path.basename(path)}"
    )

    data = np.load(
        path,
        allow_pickle=True
    )

    print(
        "Arrays:",
        list(data.keys())
    )

    return data


# =====================================================================
# EXTRACT FEATURES AND LABELS
# =====================================================================

def extract_features_labels(
    data,
    filename
):

    arrays = {}

    for key in data.keys():

        value = np.asarray(
            data[key]
        )

        print(
            f"    {key}: "
            f"shape={value.shape}, "
            f"dtype={value.dtype}"
        )

        arrays[key] = value


    # ---------------------------------------------------------------
    # Find feature array
    # ---------------------------------------------------------------

    feature_candidates = []

    for key, value in arrays.items():

        if value.ndim == 2:

            if np.issubdtype(
                value.dtype,
                np.number
            ):

                # Exclude one-hot labels
                if value.shape[1] != len(
                    CLASS_NAMES
                ):

                    feature_candidates.append(
                        (key, value)
                    )


    if not feature_candidates:

        raise ValueError(
            f"\nFeature array not found in "
            f"{filename}"
        )


    preferred = []

    for key, value in feature_candidates:

        key_lower = key.lower()

        if any(
            word in key_lower
            for word in [
                "feature",
                "features",
                "embedding",
                "embeddings",
                "data",
                "x"
            ]
        ):

            preferred.append(
                (key, value)
            )


    if preferred:

        feature_key, features = (
            preferred[0]
        )

    else:

        feature_key, features = (
            feature_candidates[0]
        )


    # ---------------------------------------------------------------
    # Find labels
    # ---------------------------------------------------------------

    label_candidates = []

    for key, value in arrays.items():

        if value.ndim == 1:

            label_candidates.append(
                (key, value)
            )


    # One-hot label support

    if not label_candidates:

        for key, value in arrays.items():

            if (
                value.ndim == 2
                and value.shape[1]
                == len(CLASS_NAMES)
            ):

                if np.issubdtype(
                    value.dtype,
                    np.number
                ):

                    labels = np.argmax(
                        value,
                        axis=1
                    )

                    label_candidates.append(
                        (key, labels)
                    )


    if not label_candidates:

        raise ValueError(
            f"\nLabel array not found in "
            f"{filename}"
        )


    preferred_labels = []

    for key, value in label_candidates:

        key_lower = key.lower()

        if any(
            word in key_lower
            for word in [
                "label",
                "labels",
                "target",
                "targets",
                "y"
            ]
        ):

            preferred_labels.append(
                (key, value)
            )


    if preferred_labels:

        label_key, labels = (
            preferred_labels[0]
        )

    else:

        label_key, labels = (
            label_candidates[0]
        )


    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels
    ).reshape(-1)


    # ---------------------------------------------------------------
    # String labels
    # ---------------------------------------------------------------

    if labels.dtype.kind in [
        "U",
        "S",
        "O"
    ]:

        mapping = {

            "eus": 0,

            "gill": 1,

            "gill disease": 1,

            "healthy": 2,

            "red_spot": 3,

            "red spot": 3,

            "red spot disease": 3
        }

        converted = []

        for label in labels:

            name = str(
                label
            ).strip().lower()

            if name not in mapping:

                raise ValueError(
                    f"Unknown label: {label}"
                )

            converted.append(
                mapping[name]
            )

        labels = np.array(
            converted,
            dtype=int
        )

    else:

        labels = labels.astype(int)


    # ---------------------------------------------------------------
    # Verify sample count
    # ---------------------------------------------------------------

    if len(features) != len(labels):

        raise ValueError(
            f"\nFeature/label mismatch in "
            f"{filename}\n"
            f"Features = {len(features)}\n"
            f"Labels   = {len(labels)}"
        )


    print(
        f"    Feature array: {feature_key}"
    )

    print(
        f"    Label array  : {label_key}"
    )

    print(
        f"    Feature shape: {features.shape}"
    )

    print(
        f"    Label shape  : {labels.shape}"
    )

    return features, labels


# =====================================================================
# LOAD ALL THREE CNN FEATURES
# =====================================================================

model_data = {}


print()
print("=" * 70)
print("LOADING PHASE 3 FEATURES")
print("=" * 70)


for model_name, files in FEATURE_FILES.items():

    print()
    print("-" * 70)
    print(
        f"MODEL: {model_name}"
    )
    print("-" * 70)

    model_data[model_name] = {}

    for split in [
        "train",
        "validation",
        "test"
    ]:

        filename = files[split]

        path = os.path.join(
            FEATURE_DIR,
            filename
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nMissing feature file:\n"
                f"{path}"
            )

        data = load_npz(
            path
        )

        features, labels = (
            extract_features_labels(
                data,
                filename
            )
        )

        model_data[model_name][split] = {

            "features": features,

            "labels": labels
        }


# =====================================================================
# VERIFY LABEL CONSISTENCY
# =====================================================================

print()
print("=" * 70)
print("VERIFYING LABEL CONSISTENCY")
print("=" * 70)


reference_train_labels = None
reference_test_labels = None


for model_name in FEATURE_FILES:

    train_labels = (
        model_data[model_name]
        ["train"]
        ["labels"]
    )

    validation_labels = (
        model_data[model_name]
        ["validation"]
        ["labels"]
    )

    test_labels = (
        model_data[model_name]
        ["test"]
        ["labels"]
    )


    combined_labels = np.concatenate(
        [
            train_labels,
            validation_labels
        ]
    )


    if reference_train_labels is None:

        reference_train_labels = (
            combined_labels
        )

        reference_test_labels = (
            test_labels
        )

    else:

        if not np.array_equal(
            reference_train_labels,
            combined_labels
        ):

            raise ValueError(
                f"Training/validation labels "
                f"do not match for {model_name}."
            )

        if not np.array_equal(
            reference_test_labels,
            test_labels
        ):

            raise ValueError(
                f"Test labels do not match "
                f"for {model_name}."
            )


y_train = reference_train_labels

y_test = reference_test_labels


print(
    f"Training + validation samples : "
    f"{len(y_train)}"
)

print(
    f"Test samples                  : "
    f"{len(y_test)}"
)


# =====================================================================
# COMBINE TRAIN + VALIDATION FEATURES
# =====================================================================

for model_name in FEATURE_FILES:

    train_features = (
        model_data[model_name]
        ["train"]
        ["features"]
    )

    validation_features = (
        model_data[model_name]
        ["validation"]
        ["features"]
    )

    test_features = (
        model_data[model_name]
        ["test"]
        ["features"]
    )


    model_data[model_name][
        "combined_train"
    ] = np.concatenate(
        [
            train_features,
            validation_features
        ],
        axis=0
    )


    model_data[model_name][
        "test_features"
    ] = test_features


# =====================================================================
# FEATURE MATRIX BUILDER
# =====================================================================

def build_features(
    model_list,
    split
):

    arrays = []

    for model_name in model_list:

        if split == "train":

            arrays.append(
                model_data[model_name][
                    "combined_train"
                ]
            )

        elif split == "test":

            arrays.append(
                model_data[model_name][
                    "test_features"
                ]
            )

        else:

            raise ValueError(
                "Invalid split"
            )


    return np.concatenate(
        arrays,
        axis=1
    )


# =====================================================================
# DISPLAY FEATURE CONFIGURATIONS
# =====================================================================

print()
print("=" * 70)
print("FEATURE CONFIGURATIONS")
print("=" * 70)


for name, models in FEATURE_CONFIGS.items():

    X_train = build_features(
        models,
        "train"
    )

    X_test = build_features(
        models,
        "test"
    )

    print(
        f"{name:<45}"
        f"Train: {X_train.shape}   "
        f"Test: {X_test.shape}"
    )


# =====================================================================
# CHECK WHETHER EXPERIMENT ALREADY EXISTS
# =====================================================================

def experiment_exists(
    combination_name,
    kernel_name
):

    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"{combination_name}_"
        f"{kernel_name}_predictions.csv"
    )

    model_file = os.path.join(
        MODEL_DIR,
        f"{combination_name}_"
        f"{kernel_name}.joblib"
    )

    report_file = os.path.join(
        REPORT_DIR,
        f"{combination_name}_"
        f"{kernel_name}_"
        f"classification_report.txt"
    )

    return (
        os.path.exists(prediction_file)
        and
        os.path.exists(model_file)
        and
        os.path.exists(report_file)
    )


# =====================================================================
# RECOVER EXISTING RESULT FROM FILES
# =====================================================================

def recover_existing_result(
    combination_name,
    kernel_name,
    models
):

    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"{combination_name}_"
        f"{kernel_name}_predictions.csv"
    )


    model_file = os.path.join(
        MODEL_DIR,
        f"{combination_name}_"
        f"{kernel_name}.joblib"
    )


    prediction_df = pd.read_csv(
        prediction_file
    )


    y_true = prediction_df[
        "true_label"
    ].to_numpy()


    y_pred = prediction_df[
        "predicted_label"
    ].to_numpy()


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    X_train = build_features(
        models,
        "train"
    )


    feature_dimension = (
        X_train.shape[1]
    )


    training_time = np.nan


    if os.path.exists(model_file):

        try:

            saved = joblib.load(
                model_file
            )

            if isinstance(
                saved,
                dict
            ):

                if (
                    "training_time"
                    in saved
                ):

                    training_time = (
                        saved[
                            "training_time"
                        ]
                    )

        except Exception:

            pass


    print(
        f"Existing result found: "
        f"{combination_name} — "
        f"{kernel_name}"
    )

    print(
        f"    Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    return {

        "Combination":
            combination_name,

        "Kernel":
            kernel_name,

        "Feature_Dimension":
            feature_dimension,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "Training_Time_s":
            training_time
    }


# =====================================================================
# TRAIN ONE EXPERIMENT
# =====================================================================

def run_experiment(
    combination_name,
    models,
    kernel_name,
    parameters
):

    print()
    print("-" * 70)

    print(
        f"{combination_name} — "
        f"{kernel_name} SVM"
    )

    print("-" * 70)


    # ---------------------------------------------------------------
    # Skip if complete
    # ---------------------------------------------------------------

    if experiment_exists(
        combination_name,
        kernel_name
    ):

        print(
            "Experiment already completed."
        )

        return recover_existing_result(
            combination_name,
            kernel_name,
            models
        )


    # ---------------------------------------------------------------
    # Build feature matrices
    # ---------------------------------------------------------------

    X_train = build_features(
        models,
        "train"
    )

    X_test = build_features(
        models,
        "test"
    )


    feature_dimension = (
        X_train.shape[1]
    )


    print(
        f"Feature dimension: "
        f"{feature_dimension}"
    )

    print(
        f"Training samples: "
        f"{X_train.shape[0]}"
    )

    print(
        f"Test samples: "
        f"{X_test.shape[0]}"
    )


    # ---------------------------------------------------------------
    # Create classifier
    # ---------------------------------------------------------------

    if kernel_name == "Linear":

        svm = LinearSVC(
            C=parameters["C"],
            random_state=RANDOM_STATE,
            max_iter=20000,
            dual="auto"
        )

    else:

        svm = SVC(
            probability=True,
            random_state=RANDOM_STATE,
            **parameters
        )


    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print(
        "Training SVM..."
    )

    start_time = time.time()


    svm.fit(
        X_train,
        y_train
    )


    training_time = (
        time.time() - start_time
    )


    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )


    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    print(
        "Generating predictions..."
    )


    y_pred = svm.predict(
        X_test
    )


    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    print()
    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score  : "
        f"{f1:.4f}"
    )


    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )


    print()
    print(
        "Classification Report:"
    )

    print(report)


    report_filename = (
        f"{combination_name}_"
        f"{kernel_name}_"
        f"classification_report.txt"
    )


    with open(
        os.path.join(
            REPORT_DIR,
            report_filename
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    cm_filename = (
        f"{combination_name}_"
        f"{kernel_name}_"
        f"confusion_matrix.png"
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )


    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )


    ax.set_title(
        f"{combination_name}\n"
        f"{kernel_name} SVM"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            PLOT_DIR,
            cm_filename
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ---------------------------------------------------------------
    # Save model
    # ---------------------------------------------------------------

    model_filename = (
        f"{combination_name}_"
        f"{kernel_name}.joblib"
    )


    joblib.dump(
        {
            "svm": svm,

            "models": models,

            "kernel": kernel_name,

            "parameters": parameters,

            "class_names": CLASS_NAMES,

            "feature_dimension":
                feature_dimension,

            "training_time":
                training_time

        },

        os.path.join(
            MODEL_DIR,
            model_filename
        )
    )


    # ---------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------

    prediction_filename = (
        f"{combination_name}_"
        f"{kernel_name}_predictions.csv"
    )


    prediction_df = pd.DataFrame({

        "true_label": y_test,

        "true_class": [
            CLASS_NAMES[x]
            for x in y_test
        ],

        "predicted_label": y_pred,

        "predicted_class": [
            CLASS_NAMES[x]
            for x in y_pred
        ]

    })


    prediction_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            prediction_filename
        ),
        index=False
    )


    return {

        "Combination":
            combination_name,

        "Kernel":
            kernel_name,

        "Feature_Dimension":
            feature_dimension,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "Training_Time_s":
            training_time
    }


# =====================================================================
# RUN ALL 21 EXPERIMENTS
# =====================================================================

results = []


total_experiments = (
    len(FEATURE_CONFIGS)
    *
    len(SVM_CONFIGS)
)


experiment_number = 0


for combination_name, models in FEATURE_CONFIGS.items():

    for kernel_name, parameters in SVM_CONFIGS.items():

        experiment_number += 1

        print()
        print(
            "=" * 70
        )

        print(
            f"EXPERIMENT "
            f"{experiment_number}/"
            f"{total_experiments}"
        )

        print(
            f"{combination_name} — "
            f"{kernel_name}"
        )

        print(
            "=" * 70
        )


        result = run_experiment(

            combination_name,

            models,

            kernel_name,

            parameters
        )


        results.append(
            result
        )


# =====================================================================
# MASTER RESULTS DATAFRAME
# =====================================================================

results_df = pd.DataFrame(
    results
)


# =====================================================================
# SORT RESULTS
# =====================================================================

kernel_order = {
    "Linear": 0,
    "RBF": 1,
    "Polynomial": 2
}


results_df["_kernel_order"] = (
    results_df["Kernel"].map(
        kernel_order
    )
)


results_df = results_df.sort_values(
    [
        "Combination",
        "_kernel_order"
    ]
)


results_df = results_df.drop(
    columns=["_kernel_order"]
)


# =====================================================================
# SAVE MASTER CSV
# =====================================================================

master_csv = os.path.join(
    OUTPUT_DIR,
    "phase5_svm_kernel_results.csv"
)


results_df.to_csv(
    master_csv,
    index=False
)


# =====================================================================
# PRINT COMPLETE RESULTS
# =====================================================================

print()
print("=" * 70)
print("PHASE 5 RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# =====================================================================
# ACCURACY PIVOT
# =====================================================================

accuracy_pivot = results_df.pivot(
    index="Combination",
    columns="Kernel",
    values="Accuracy"
)


fig, ax = plt.subplots(
    figsize=(14, 7)
)


accuracy_pivot.plot(
    kind="bar",
    ax=ax
)


ax.set_title(
    "SVM Kernel Comparison"
)

ax.set_xlabel(
    "Feature Configuration"
)

ax.set_ylabel(
    "Test Accuracy"
)

ax.set_ylim(
    0,
    1
)

ax.tick_params(
    axis="x",
    rotation=45
)

ax.legend(
    title="SVM Kernel"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "svm_kernel_accuracy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =====================================================================
# F1 PLOT
# =====================================================================

f1_pivot = results_df.pivot(
    index="Combination",
    columns="Kernel",
    values="F1_Score"
)


fig, ax = plt.subplots(
    figsize=(14, 7)
)


f1_pivot.plot(
    kind="bar",
    ax=ax
)


ax.set_title(
    "SVM Kernel F1-score Comparison"
)

ax.set_xlabel(
    "Feature Configuration"
)

ax.set_ylabel(
    "Weighted F1-score"
)

ax.set_ylim(
    0,
    1
)

ax.tick_params(
    axis="x",
    rotation=45
)

ax.legend(
    title="SVM Kernel"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "svm_kernel_f1_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =====================================================================
# TRAINING TIME PLOT
# =====================================================================

time_pivot = results_df.pivot(
    index="Combination",
    columns="Kernel",
    values="Training_Time_s"
)


fig, ax = plt.subplots(
    figsize=(14, 7)
)


time_pivot.plot(
    kind="bar",
    ax=ax
)


ax.set_title(
    "SVM Kernel Training Time Comparison"
)

ax.set_xlabel(
    "Feature Configuration"
)

ax.set_ylabel(
    "Training Time (seconds)"
)

ax.tick_params(
    axis="x",
    rotation=45
)

ax.legend(
    title="SVM Kernel"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "svm_kernel_training_time.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =====================================================================
# FIND BEST RESULT
# =====================================================================

best_index = results_df[
    "Accuracy"
].idxmax()


best_result = results_df.loc[
    best_index
]


print()
print("=" * 70)
print("BEST PHASE 5 RESULT")
print("=" * 70)


print(
    f"Combination : "
    f"{best_result['Combination']}"
)


print(
    f"Kernel      : "
    f"{best_result['Kernel']}"
)


print(
    f"Accuracy    : "
    f"{best_result['Accuracy'] * 100:.2f}%"
)


print(
    f"Precision   : "
    f"{best_result['Precision']:.4f}"
)


print(
    f"Recall      : "
    f"{best_result['Recall']:.4f}"
)


print(
    f"F1-score    : "
    f"{best_result['F1_Score']:.4f}"
)


# =====================================================================
# SAVE BEST RESULT
# =====================================================================

best_result_dict = {}

for key, value in best_result.to_dict().items():

    if isinstance(
        value,
        (np.integer, np.floating)
    ):

        value = value.item()

    best_result_dict[key] = value


with open(
    os.path.join(
        OUTPUT_DIR,
        "best_phase5_result.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        best_result_dict,
        file,
        indent=4
    )


# =====================================================================
# SAVE CONFIGURATION
# =====================================================================

configuration = {

    "phase":
        "Phase 5",

    "PCA":
        False,

    "linear_implementation":
        "LinearSVC",

    "rbf_implementation":
        "SVC",

    "polynomial_implementation":
        "SVC",

    "random_state":
        RANDOM_STATE,

    "class_names":
        CLASS_NAMES,

    "feature_files":
        FEATURE_FILES,

    "feature_configurations":
        FEATURE_CONFIGS,

    "svm_configurations":
        SVM_CONFIGS
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "phase5_configuration.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)

print()
print("Master results:")
print(master_csv)

print()
print("Plots:")
print(PLOT_DIR)

print()
print("Reports:")
print(REPORT_DIR)

print()
print("Models:")
print(MODEL_DIR)

print()
print("Next step:")
print("Phase 6 — Final selected model evaluation")

print("=" * 70)