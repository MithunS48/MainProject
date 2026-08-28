import os
import time
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
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


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_DIR = r"D:\MainProject"

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "features"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "pca_svm"
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

PCA_VARIANCE = 0.95

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]


# ================================================================
# EXACT PHASE 3 FILES
# ================================================================

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


# ================================================================
# FEATURE COMBINATIONS
# ================================================================

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


# ================================================================
# CREATE DIRECTORIES
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PHASE 4 — PCA + SVM EXPERIMENTS")
print("=" * 70)

print(f"Project directory : {PROJECT_DIR}")
print(f"Feature directory : {FEATURE_DIR}")
print(f"Output directory  : {OUTPUT_DIR}")
print()
print("SVM kernel        : Linear")
print(f"SVM C             : {SVM_C}")
print(f"PCA variance      : {PCA_VARIANCE}")
print(f"Random seed       : {RANDOM_STATE}")
print("=" * 70)


# ================================================================
# CHECK FEATURE DIRECTORY
# ================================================================

if not os.path.exists(FEATURE_DIR):

    raise FileNotFoundError(
        f"\nFeature directory not found:\n{FEATURE_DIR}"
    )


# ================================================================
# LOAD NPZ FILE
# ================================================================

def load_npz_file(path):

    print(f"\nLoading: {path}")

    data = np.load(
        path,
        allow_pickle=True
    )

    print(
        "Available arrays:",
        list(data.keys())
    )

    return data


# ================================================================
# IDENTIFY FEATURE ARRAY
# ================================================================

def extract_features_and_labels(npz_data, filename):

    """
    Automatically identifies:

        feature array = 2D numeric array
        label array   = 1D array

    This avoids assuming specific NPZ key names.
    """

    arrays = {}

    for key in npz_data.keys():

        value = np.asarray(
            npz_data[key]
        )

        print(
            f"    {key}: shape={value.shape}, "
            f"dtype={value.dtype}"
        )

        arrays[key] = value


    # ------------------------------------------------------------
    # Find feature arrays
    # ------------------------------------------------------------

    feature_candidates = []

    for key, value in arrays.items():

        if value.ndim == 2:

            if np.issubdtype(
                value.dtype,
                np.number
            ):

                feature_candidates.append(
                    (key, value)
                )


    if len(feature_candidates) == 0:

        raise ValueError(
            f"\nCould not identify feature array in:\n"
            f"{filename}\n"
            f"Available arrays: {list(arrays.keys())}"
        )


    # Prefer arrays containing feature-related names

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
                "x",
                "data"
            ]
        ):

            preferred.append(
                (key, value)
            )


    if len(preferred) > 0:

        feature_key, features = preferred[0]

    else:

        feature_key, features = (
            feature_candidates[0]
        )


    # ------------------------------------------------------------
    # Find labels
    # ------------------------------------------------------------

    label_candidates = []

    for key, value in arrays.items():

        if value.ndim == 1:

            label_candidates.append(
                (key, value)
            )


    # Also support one-hot labels

    if len(label_candidates) == 0:

        for key, value in arrays.items():

            if (
                value.ndim == 2
                and value.shape[1] == len(CLASS_NAMES)
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


    if len(label_candidates) == 0:

        raise ValueError(
            f"\nCould not identify labels in:\n"
            f"{filename}\n"
            f"Available arrays: {list(arrays.keys())}"
        )


    # Prefer label-related names

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


    if len(preferred_labels) > 0:

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
    )


    # ------------------------------------------------------------
    # Convert one-hot labels
    # ------------------------------------------------------------

    if labels.ndim > 1:

        labels = np.argmax(
            labels,
            axis=1
        )


    labels = labels.reshape(-1)


    # ------------------------------------------------------------
    # Convert string labels if necessary
    # ------------------------------------------------------------

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

            label_string = str(
                label
            ).strip().lower()

            if label_string not in mapping:

                raise ValueError(
                    f"Unknown class label: "
                    f"{label}"
                )

            converted.append(
                mapping[label_string]
            )

        labels = np.array(
            converted,
            dtype=int
        )

    else:

        labels = labels.astype(int)


    # ------------------------------------------------------------
    # Verify sample count
    # ------------------------------------------------------------

    if len(features) != len(labels):

        raise ValueError(
            f"\nFeature/label mismatch in "
            f"{filename}\n"
            f"Features: {len(features)}\n"
            f"Labels  : {len(labels)}"
        )


    print(
        f"    Feature array : {feature_key}"
    )

    print(
        f"    Label array   : {label_key}"
    )

    print(
        f"    Features shape: {features.shape}"
    )

    print(
        f"    Labels shape  : {labels.shape}"
    )

    return features, labels


# ================================================================
# LOAD ALL THREE MODELS
# ================================================================

model_data = {}


print()
print("=" * 70)
print("LOADING PHASE 3 FEATURES")
print("=" * 70)


for model_name, files in FEATURE_FILES.items():

    print()
    print("-" * 70)
    print(f"MODEL: {model_name}")
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
                f"\nRequired Phase 3 file not found:\n"
                f"{path}"
            )

        npz_data = load_npz_file(
            path
        )

        features, labels = (
            extract_features_and_labels(
                npz_data,
                filename
            )
        )

        model_data[model_name][split] = {
            "features": features,
            "labels": labels
        }


# ================================================================
# VERIFY LABELS
# ================================================================

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

    val_labels = (
        model_data[model_name]
        ["validation"]
        ["labels"]
    )

    test_labels = (
        model_data[model_name]
        ["test"]
        ["labels"]
    )


    if reference_train_labels is None:

        reference_train_labels = np.concatenate(
            [
                train_labels,
                val_labels
            ]
        )

        reference_test_labels = test_labels

    else:

        current_train_labels = np.concatenate(
            [
                train_labels,
                val_labels
            ]
        )

        if not np.array_equal(
            reference_train_labels,
            current_train_labels
        ):

            raise ValueError(
                f"Training/validation label order "
                f"does not match for {model_name}."
            )

        if not np.array_equal(
            reference_test_labels,
            test_labels
        ):

            raise ValueError(
                f"Test label order "
                f"does not match for {model_name}."
            )


y_train = reference_train_labels

y_test = reference_test_labels


print(
    f"Train + Validation samples : "
    f"{len(y_train)}"
)

print(
    f"Test samples               : "
    f"{len(y_test)}"
)


# ================================================================
# PREPARE COMBINED TRAIN/VALIDATION DATA
# ================================================================

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


    combined_train = np.concatenate(
        [
            train_features,
            validation_features
        ],
        axis=0
    )


    model_data[model_name]["combined_train"] = (
        combined_train
    )

    model_data[model_name]["test_features"] = (
        test_features
    )


# ================================================================
# DISPLAY FEATURE DIMENSIONS
# ================================================================

print()
print("=" * 70)
print("INDIVIDUAL FEATURE DIMENSIONS")
print("=" * 70)


for model_name in FEATURE_FILES:

    X = model_data[model_name][
        "combined_train"
    ]

    X_test = model_data[model_name][
        "test_features"
    ]

    print(
        f"{model_name:<15} "
        f"Train: {X.shape} "
        f"Test: {X_test.shape}"
    )


# ================================================================
# BUILD FUSED FEATURES
# ================================================================

def build_feature_matrix(
    model_list,
    split
):

    arrays = []

    for model_name in model_list:

        if split == "train":

            array = model_data[model_name][
                "combined_train"
            ]

        elif split == "test":

            array = model_data[model_name][
                "test_features"
            ]

        else:

            raise ValueError(
                "split must be train or test"
            )

        arrays.append(array)


    return np.concatenate(
        arrays,
        axis=1
    )


# ================================================================
# DISPLAY ALL CONFIGURATIONS
# ================================================================

print()
print("=" * 70)
print("FEATURE CONFIGURATIONS")
print("=" * 70)


for combination, models in FEATURE_CONFIGS.items():

    X_train = build_feature_matrix(
        models,
        "train"
    )

    X_test = build_feature_matrix(
        models,
        "test"
    )

    print(
        f"{combination:<42}"
        f"Train: {str(X_train.shape):<18}"
        f"Test: {X_test.shape}"
    )


# ================================================================
# RUN ONE EXPERIMENT
# ================================================================

def run_experiment(
    combination_name,
    models,
    use_pca
):

    experiment_name = (
        "WITH_PCA"
        if use_pca
        else "NO_PCA"
    )

    print()
    print("-" * 70)
    print(
        f"{combination_name} — "
        f"{experiment_name}"
    )
    print("-" * 70)


    # ------------------------------------------------------------
    # Build feature matrices
    # ------------------------------------------------------------

    X_train = build_feature_matrix(
        models,
        "train"
    )

    X_test = build_feature_matrix(
        models,
        "test"
    )


    original_dimension = (
        X_train.shape[1]
    )


    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------

    pca = None

    pca_time = 0.0


    if use_pca:

        print("Applying PCA...")

        pca_start = time.time()

        pca = PCA(
            n_components=PCA_VARIANCE,
            svd_solver="full",
            random_state=RANDOM_STATE
        )


        X_train_used = (
            pca.fit_transform(
                X_train
            )
        )


        X_test_used = (
            pca.transform(
                X_test
            )
        )


        pca_time = (
            time.time() - pca_start
        )


        pca_dimension = (
            X_train_used.shape[1]
        )


        variance_retained = (
            np.sum(
                pca.explained_variance_ratio_
            ) * 100
        )


        print(
            f"Original dimensions : "
            f"{original_dimension}"
        )

        print(
            f"PCA dimensions      : "
            f"{pca_dimension}"
        )

        print(
            f"Variance retained   : "
            f"{variance_retained:.2f}%"
        )

    else:

        X_train_used = X_train

        X_test_used = X_test

        pca_dimension = (
            original_dimension
        )

        variance_retained = 100.0


    # ------------------------------------------------------------
    # Linear SVM
    # ------------------------------------------------------------

    print("Training Linear SVM...")

    start_time = time.time()


    svm = LinearSVC(
        C=SVM_C,
        random_state=RANDOM_STATE,
        max_iter=20000
    )


    svm.fit(
        X_train_used,
        y_train
    )


    training_time = (
        time.time() - start_time
    )


    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    print("Generating predictions...")


    y_pred = svm.predict(
        X_test_used
    )


    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )


    print()
    print("Classification Report:")
    print(report)


    report_file = (
        f"{combination_name}_"
        f"{experiment_name}_"
        f"classification_report.txt"
    )


    with open(
        os.path.join(
            REPORT_DIR,
            report_file
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)


    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    cm_file = (
        f"{combination_name}_"
        f"{experiment_name}_"
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
        f"{experiment_name}"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            PLOT_DIR,
            cm_file
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ------------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------------

    model_file = (
        f"{combination_name}_"
        f"{experiment_name}.joblib"
    )


    joblib.dump(
        {
            "svm": svm,
            "pca": pca,
            "models": models,
            "class_names": CLASS_NAMES,
            "original_dimension":
                original_dimension,
            "pca_dimension":
                pca_dimension,
            "variance_retained":
                variance_retained
        },
        os.path.join(
            MODEL_DIR,
            model_file
        )
    )


    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------

    prediction_file = (
        f"{combination_name}_"
        f"{experiment_name}_"
        f"predictions.csv"
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
            prediction_file
        ),
        index=False
    )


    # ------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------

    return {

        "Combination":
            combination_name,

        "Experiment":
            experiment_name,

        "Original_Feature_Dim":
            original_dimension,

        "PCA_Feature_Dim":
            pca_dimension,

        "Variance_Retained_%":
            variance_retained,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "Training_Time_s":
            training_time,

        "PCA_Time_s":
            pca_time,

        "Total_Time_s":
            training_time + pca_time
    }


# ================================================================
# RUN ALL 14 EXPERIMENTS
# ================================================================

results = []


for combination_name, models in FEATURE_CONFIGS.items():

    # ------------------------------------------------------------
    # NO PCA
    # ------------------------------------------------------------

    result_no_pca = run_experiment(
        combination_name,
        models,
        use_pca=False
    )

    results.append(
        result_no_pca
    )


    # ------------------------------------------------------------
    # WITH PCA
    # ------------------------------------------------------------

    result_pca = run_experiment(
        combination_name,
        models,
        use_pca=True
    )

    results.append(
        result_pca
    )


# ================================================================
# RESULTS DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)


# ================================================================
# SAVE MASTER RESULTS
# ================================================================

master_csv = os.path.join(
    OUTPUT_DIR,
    "phase4_pca_svm_results.csv"
)


results_df.to_csv(
    master_csv,
    index=False
)


# ================================================================
# PRINT RESULTS
# ================================================================

print()
print("=" * 70)
print("PHASE 4 RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# PCA VS NO PCA — ACCURACY
# ================================================================

accuracy_pivot = results_df.pivot(
    index="Combination",
    columns="Experiment",
    values="Accuracy"
)


fig, ax = plt.subplots(
    figsize=(13, 7)
)


accuracy_pivot.plot(
    kind="bar",
    ax=ax
)


ax.set_title(
    "PCA vs No-PCA Accuracy Comparison"
)

ax.set_xlabel(
    "Feature Configuration"
)

ax.set_ylabel(
    "Accuracy"
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
    title="Experiment"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "pca_vs_no_pca_accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# PCA VS NO PCA — F1
# ================================================================

f1_pivot = results_df.pivot(
    index="Combination",
    columns="Experiment",
    values="F1_Score"
)


fig, ax = plt.subplots(
    figsize=(13, 7)
)


f1_pivot.plot(
    kind="bar",
    ax=ax
)


ax.set_title(
    "PCA vs No-PCA F1-score Comparison"
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
    title="Experiment"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "pca_vs_no_pca_f1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# PCA DIMENSION REDUCTION
# ================================================================

pca_results = results_df[
    results_df["Experiment"] == "WITH_PCA"
].copy()


fig, ax = plt.subplots(
    figsize=(13, 7)
)


x = np.arange(
    len(pca_results)
)

width = 0.38


ax.bar(
    x - width / 2,
    pca_results[
        "Original_Feature_Dim"
    ],
    width,
    label="Original"
)


ax.bar(
    x + width / 2,
    pca_results[
        "PCA_Feature_Dim"
    ],
    width,
    label="After PCA"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    pca_results[
        "Combination"
    ],
    rotation=45,
    ha="right"
)


ax.set_ylabel(
    "Feature Dimensions"
)

ax.set_xlabel(
    "Feature Configuration"
)

ax.set_title(
    "Feature Dimension Reduction Using PCA"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "pca_dimension_reduction.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# FIND BEST RESULT
# ================================================================

best_index = results_df[
    "Accuracy"
].idxmax()


best_result = results_df.loc[
    best_index
]


print()
print("=" * 70)
print("BEST PHASE 4 RESULT")
print("=" * 70)

print(
    f"Combination : "
    f"{best_result['Combination']}"
)

print(
    f"Experiment  : "
    f"{best_result['Experiment']}"
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


# ================================================================
# SAVE BEST RESULT
# ================================================================

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
        "best_phase4_result.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        best_result_dict,
        f,
        indent=4
    )


# ================================================================
# SAVE CONFIGURATION
# ================================================================

configuration = {

    "phase": "Phase 4",

    "svm_kernel": "Linear",

    "svm_C": SVM_C,

    "pca_variance": PCA_VARIANCE,

    "random_state": RANDOM_STATE,

    "classes": CLASS_NAMES,

    "feature_files": FEATURE_FILES,

    "feature_configurations":
        FEATURE_CONFIGS
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "phase4_configuration.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        configuration,
        f,
        indent=4
    )


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 70)
print("PHASE 4 COMPLETE")
print("=" * 70)

print()
print("Results:")
print(OUTPUT_DIR)

print()
print("Master CSV:")
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
print("Phase 5 — Linear vs RBF vs Polynomial SVM")

print("=" * 70)