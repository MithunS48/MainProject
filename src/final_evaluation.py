"""
======================================================================
PHASE 6 — FINAL MODEL EVALUATION
======================================================================

FINAL MODEL SELECTED FROM PHASE 5
----------------------------------

MobileNetV2 + ConvNeXt
        |
        v
Feature Fusion
        |
        v
2048-dimensional feature vector
        |
        v
No PCA
        |
        v
Polynomial SVM
        |
        v
4 Disease Classes

Classes:
    EUS
    gill
    healthy
    red_spot

Phase 5 best result:
    Accuracy : approximately 98.29%

IMPORTANT:
    This script DOES NOT TRAIN the SVM again.

    It loads the already trained Phase 5 model and evaluates it
    on the original Phase 5 test features.

Outputs:
    results/final/
        |
        +-- final_model/
        |
        +-- reports/
        |
        +-- plots/
        |
        +-- predictions/
        |
        +-- analysis/

======================================================================
"""

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


warnings.filterwarnings("ignore")


# =====================================================================
# 1. PROJECT PATHS
# =====================================================================

PROJECT_DIR = r"D:\MainProject"


FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "features"
)


PHASE5_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "svm_kernel"
)


PHASE5_MODEL_DIR = os.path.join(
    PHASE5_DIR,
    "models"
)


OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "final"
)


REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)


PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)


PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)


ANALYSIS_DIR = os.path.join(
    OUTPUT_DIR,
    "analysis"
)


FINAL_MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "final_model"
)


# =====================================================================
# 2. FINAL MODEL CONFIGURATION
# =====================================================================

FINAL_COMBINATION = (
    "MobileNetV2_plus_ConvNeXt"
)


FINAL_KERNEL = "Polynomial"


FINAL_MODEL_FILE = os.path.join(
    PHASE5_MODEL_DIR,
    "MobileNetV2_plus_ConvNeXt_Polynomial.joblib"
)


# =====================================================================
# 3. FEATURE FILES
# =====================================================================

MOBILENET_TRAIN_FILE = os.path.join(
    FEATURE_DIR,
    "mobilenet_train.npz"
)


MOBILENET_VALIDATION_FILE = os.path.join(
    FEATURE_DIR,
    "mobilenet_validation.npz"
)


MOBILENET_TEST_FILE = os.path.join(
    FEATURE_DIR,
    "mobilenet_test.npz"
)


CONVNEXT_TRAIN_FILE = os.path.join(
    FEATURE_DIR,
    "convnext_train.npz"
)


CONVNEXT_VALIDATION_FILE = os.path.join(
    FEATURE_DIR,
    "convnext_validation.npz"
)


CONVNEXT_TEST_FILE = os.path.join(
    FEATURE_DIR,
    "convnext_test.npz"
)


# =====================================================================
# 4. MODEL SETTINGS
# =====================================================================

CLASS_NAMES = [
    "EUS",
    "gill",
    "healthy",
    "red_spot"
]


NUM_CLASSES = len(
    CLASS_NAMES
)


EXPECTED_FEATURE_DIMENSION = (
    1280 + 768
)


EXPECTED_TRAIN_SAMPLES = 8395


EXPECTED_TEST_SAMPLES = 2105


EXPECTED_C = 1.0


EXPECTED_POLY_DEGREE = 3


EXPECTED_GAMMA = "scale"


EXPECTED_COEF0 = 0.0


RANDOM_STATE = 42


# =====================================================================
# 5. CREATE OUTPUT DIRECTORIES
# =====================================================================

for directory in [
    OUTPUT_DIR,
    REPORT_DIR,
    PLOT_DIR,
    PREDICTION_DIR,
    ANALYSIS_DIR,
    FINAL_MODEL_DIR
]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# =====================================================================
# 6. HEADER
# =====================================================================

print()
print("=" * 70)
print("PHASE 6 — FINAL MODEL EVALUATION")
print("=" * 70)

print(
    f"Project directory : {PROJECT_DIR}"
)

print(
    f"Feature directory : {FEATURE_DIR}"
)

print(
    f"Final model       : {FINAL_COMBINATION}"
)

print(
    f"SVM kernel        : {FINAL_KERNEL}"
)

print(
    "PCA               : No"
)

print(
    f"Expected features : "
    f"{EXPECTED_FEATURE_DIMENSION}"
)

print(
    f"Expected test     : "
    f"{EXPECTED_TEST_SAMPLES}"
)

print(
    f"Classes           : "
    f"{CLASS_NAMES}"
)

print("=" * 70)


# =====================================================================
# 7. CHECK REQUIRED FILES
# =====================================================================

required_files = [

    FINAL_MODEL_FILE,

    MOBILENET_TRAIN_FILE,
    MOBILENET_VALIDATION_FILE,
    MOBILENET_TEST_FILE,

    CONVNEXT_TRAIN_FILE,
    CONVNEXT_VALIDATION_FILE,
    CONVNEXT_TEST_FILE
]


print()
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


for file_path in required_files:

    if os.path.exists(file_path):

        print(
            f"[OK] {file_path}"
        )

    else:

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}"
        )


# =====================================================================
# 8. LOAD NPZ FILE
# =====================================================================

def load_npz(file_path):

    print()
    print(
        f"Loading: "
        f"{os.path.basename(file_path)}"
    )

    data = np.load(
        file_path,
        allow_pickle=True
    )

    print(
        "Available arrays:",
        list(data.keys())
    )

    return data


# =====================================================================
# 9. FIND FEATURE ARRAY
# =====================================================================

def get_features(
    data,
    filename
):

    candidates = []


    for key in data.keys():

        value = np.asarray(
            data[key]
        )


        if (
            value.ndim == 2
            and np.issubdtype(
                value.dtype,
                np.number
            )
            and value.shape[1] != NUM_CLASSES
        ):

            candidates.append(
                (key, value)
            )


    if len(candidates) == 0:

        raise ValueError(
            f"\nCould not find feature array "
            f"in {filename}."
        )


    preferred = []


    for key, value in candidates:

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


    if len(preferred) > 0:

        key, features = preferred[0]

    else:

        key, features = candidates[0]


    features = np.asarray(
        features,
        dtype=np.float32
    )


    print(
        f"Feature array : {key}"
    )

    print(
        f"Feature shape : {features.shape}"
    )


    return features


# =====================================================================
# 10. FIND LABEL ARRAY
# =====================================================================

def get_labels(
    data,
    filename
):

    candidates = []


    # ---------------------------------------------------------------
    # Look for one-dimensional labels
    # ---------------------------------------------------------------

    for key in data.keys():

        value = np.asarray(
            data[key]
        )


        if value.ndim == 1:

            candidates.append(
                (key, value)
            )


    # ---------------------------------------------------------------
    # Check one-hot labels
    # ---------------------------------------------------------------

    if len(candidates) == 0:

        for key in data.keys():

            value = np.asarray(
                data[key]
            )


            if (
                value.ndim == 2
                and value.shape[1]
                == NUM_CLASSES
                and np.issubdtype(
                    value.dtype,
                    np.number
                )
            ):

                labels = np.argmax(
                    value,
                    axis=1
                )


                candidates.append(
                    (key, labels)
                )


    if len(candidates) == 0:

        raise ValueError(
            f"\nCould not find labels "
            f"in {filename}."
        )


    # ---------------------------------------------------------------
    # Prefer label-like names
    # ---------------------------------------------------------------

    preferred = []


    for key, value in candidates:

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

            preferred.append(
                (key, value)
            )


    if len(preferred) > 0:

        key, labels = preferred[0]

    else:

        key, labels = candidates[0]


    labels = np.asarray(
        labels
    ).reshape(-1)


    # ---------------------------------------------------------------
    # Convert string labels to integer labels
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

            label_name = str(
                label
            ).strip().lower()


            if label_name not in mapping:

                raise ValueError(
                    f"Unknown class label: "
                    f"{label}"
                )


            converted.append(
                mapping[label_name]
            )


        labels = np.array(
            converted,
            dtype=np.int64
        )


    else:

        labels = labels.astype(
            np.int64
        )


    print(
        f"Label array   : {key}"
    )

    print(
        f"Label shape   : {labels.shape}"
    )


    return labels


# =====================================================================
# 11. LOAD MOBILE NET FEATURES
# =====================================================================

print()
print("=" * 70)
print("LOADING MOBILENETV2 FEATURES")
print("=" * 70)


mobilenet_train_data = load_npz(
    MOBILENET_TRAIN_FILE
)


mobilenet_validation_data = load_npz(
    MOBILENET_VALIDATION_FILE
)


mobilenet_test_data = load_npz(
    MOBILENET_TEST_FILE
)


mobilenet_train = get_features(
    mobilenet_train_data,
    "mobilenet_train.npz"
)


mobilenet_validation = get_features(
    mobilenet_validation_data,
    "mobilenet_validation.npz"
)


mobilenet_test = get_features(
    mobilenet_test_data,
    "mobilenet_test.npz"
)


mobilenet_train_labels = get_labels(
    mobilenet_train_data,
    "mobilenet_train.npz"
)


mobilenet_validation_labels = get_labels(
    mobilenet_validation_data,
    "mobilenet_validation.npz"
)


mobilenet_test_labels = get_labels(
    mobilenet_test_data,
    "mobilenet_test.npz"
)


# =====================================================================
# 12. LOAD CONVNEXT FEATURES
# =====================================================================

print()
print("=" * 70)
print("LOADING CONVNEXT FEATURES")
print("=" * 70)


convnext_train_data = load_npz(
    CONVNEXT_TRAIN_FILE
)


convnext_validation_data = load_npz(
    CONVNEXT_VALIDATION_FILE
)


convnext_test_data = load_npz(
    CONVNEXT_TEST_FILE
)


convnext_train = get_features(
    convnext_train_data,
    "convnext_train.npz"
)


convnext_validation = get_features(
    convnext_validation_data,
    "convnext_validation.npz"
)


convnext_test = get_features(
    convnext_test_data,
    "convnext_test.npz"
)


convnext_train_labels = get_labels(
    convnext_train_data,
    "convnext_train.npz"
)


convnext_validation_labels = get_labels(
    convnext_validation_data,
    "convnext_validation.npz"
)


convnext_test_labels = get_labels(
    convnext_test_data,
    "convnext_test.npz"
)


# =====================================================================
# 13. CHECK INDIVIDUAL FEATURE DIMENSIONS
# =====================================================================

print()
print("=" * 70)
print("CHECKING FEATURE DIMENSIONS")
print("=" * 70)


if mobilenet_train.shape[1] != 1280:

    raise ValueError(
        "Unexpected MobileNetV2 feature "
        f"dimension: {mobilenet_train.shape[1]}"
    )


if mobilenet_validation.shape[1] != 1280:

    raise ValueError(
        "Unexpected MobileNetV2 validation "
        f"dimension: {mobilenet_validation.shape[1]}"
    )


if mobilenet_test.shape[1] != 1280:

    raise ValueError(
        "Unexpected MobileNetV2 test "
        f"dimension: {mobilenet_test.shape[1]}"
    )


if convnext_train.shape[1] != 768:

    raise ValueError(
        "Unexpected ConvNeXt feature "
        f"dimension: {convnext_train.shape[1]}"
    )


if convnext_validation.shape[1] != 768:

    raise ValueError(
        "Unexpected ConvNeXt validation "
        f"dimension: {convnext_validation.shape[1]}"
    )


if convnext_test.shape[1] != 768:

    raise ValueError(
        "Unexpected ConvNeXt test "
        f"dimension: {convnext_test.shape[1]}"
    )


print(
    "MobileNetV2 : 1280 features"
)

print(
    "ConvNeXt    : 768 features"
)

print(
    "Fused       : 2048 features"
)


# =====================================================================
# 14. CHECK SAMPLE COUNTS
# =====================================================================

print()
print("=" * 70)
print("CHECKING SAMPLE COUNTS")
print("=" * 70)


print(
    f"MobileNet train      : "
    f"{len(mobilenet_train)}"
)

print(
    f"MobileNet validation : "
    f"{len(mobilenet_validation)}"
)

print(
    f"MobileNet test       : "
    f"{len(mobilenet_test)}"
)


print(
    f"ConvNeXt train       : "
    f"{len(convnext_train)}"
)

print(
    f"ConvNeXt validation  : "
    f"{len(convnext_validation)}"
)

print(
    f"ConvNeXt test        : "
    f"{len(convnext_test)}"
)


# =====================================================================
# 15. VERIFY MOBILE NET LABELS
# =====================================================================

if not np.array_equal(
    mobilenet_train_labels,
    convnext_train_labels
):

    raise ValueError(
        "MobileNetV2 and ConvNeXt "
        "training labels are not aligned."
    )


if not np.array_equal(
    mobilenet_validation_labels,
    convnext_validation_labels
):

    raise ValueError(
        "MobileNetV2 and ConvNeXt "
        "validation labels are not aligned."
    )


if not np.array_equal(
    mobilenet_test_labels,
    convnext_test_labels
):

    raise ValueError(
        "MobileNetV2 and ConvNeXt "
        "test labels are not aligned."
    )


print(
    "[OK] MobileNetV2 and ConvNeXt labels "
    "are aligned."
)


# =====================================================================
# 16. FINAL FEATURE FUSION
# =====================================================================

print()
print("=" * 70)
print("CREATING FINAL FUSED FEATURES")
print("=" * 70)


# ---------------------------------------------------------------------
# IMPORTANT
#
# The features must first be concatenated HORIZONTALLY:
#
# MobileNetV2: 1280
# ConvNeXt   :  768
# ------------------
# Fusion     : 2048
#
# This is done separately for train and validation.
# Then train and validation are concatenated VERTICALLY.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# TRAIN FUSION
# ---------------------------------------------------------------------

train_fused = np.concatenate(
    [
        mobilenet_train,
        convnext_train
    ],
    axis=1
)


# ---------------------------------------------------------------------
# VALIDATION FUSION
# ---------------------------------------------------------------------

validation_fused = np.concatenate(
    [
        mobilenet_validation,
        convnext_validation
    ],
    axis=1
)


# ---------------------------------------------------------------------
# TEST FUSION
# ---------------------------------------------------------------------

X_test = np.concatenate(
    [
        mobilenet_test,
        convnext_test
    ],
    axis=1
)


# ---------------------------------------------------------------------
# COMBINE TRAIN + VALIDATION
# ---------------------------------------------------------------------

X_train = np.concatenate(
    [
        train_fused,
        validation_fused
    ],
    axis=0
)


# ---------------------------------------------------------------------
# COMBINE LABELS
# ---------------------------------------------------------------------

y_train = np.concatenate(
    [
        mobilenet_train_labels,
        mobilenet_validation_labels
    ],
    axis=0
)


y_test = mobilenet_test_labels


# =====================================================================
# 17. VERIFY FINAL SHAPES
# =====================================================================

print()
print("=" * 70)
print("FINAL FEATURE SHAPES")
print("=" * 70)


print(
    f"MobileNetV2 train      : "
    f"{mobilenet_train.shape}"
)

print(
    f"ConvNeXt train         : "
    f"{convnext_train.shape}"
)

print(
    f"Fused train            : "
    f"{train_fused.shape}"
)


print(
    f"MobileNetV2 validation : "
    f"{mobilenet_validation.shape}"
)

print(
    f"ConvNeXt validation    : "
    f"{convnext_validation.shape}"
)

print(
    f"Fused validation       : "
    f"{validation_fused.shape}"
)


print(
    f"Final X_train         : "
    f"{X_train.shape}"
)

print(
    f"Final X_test          : "
    f"{X_test.shape}"
)


print(
    f"Final y_train         : "
    f"{y_train.shape}"
)

print(
    f"Final y_test          : "
    f"{y_test.shape}"
)


# =====================================================================
# 18. HARD CHECKS
# =====================================================================

if X_train.shape[1] != EXPECTED_FEATURE_DIMENSION:

    raise ValueError(
        "\nTraining feature dimension "
        "does not match expected 2048.\n"
        f"Expected: {EXPECTED_FEATURE_DIMENSION}\n"
        f"Found   : {X_train.shape[1]}"
    )


if X_test.shape[1] != EXPECTED_FEATURE_DIMENSION:

    raise ValueError(
        "\nTest feature dimension "
        "does not match expected 2048.\n"
        f"Expected: {EXPECTED_FEATURE_DIMENSION}\n"
        f"Found   : {X_test.shape[1]}"
    )


if X_train.shape[0] != EXPECTED_TRAIN_SAMPLES:

    raise ValueError(
        "\nUnexpected training sample count.\n"
        f"Expected: {EXPECTED_TRAIN_SAMPLES}\n"
        f"Found   : {X_train.shape[0]}"
    )


if X_test.shape[0] != EXPECTED_TEST_SAMPLES:

    raise ValueError(
        "\nUnexpected test sample count.\n"
        f"Expected: {EXPECTED_TEST_SAMPLES}\n"
        f"Found   : {X_test.shape[0]}"
    )


print()
print(
    "[OK] Final training matrix = "
    "(8395, 2048)"
)

print(
    "[OK] Final test matrix = "
    "(2105, 2048)"
)


# =====================================================================
# 19. LOAD TRAINED PHASE 5 MODEL
# =====================================================================

print()
print("=" * 70)
print("LOADING PHASE 5 FINAL MODEL")
print("=" * 70)


print(
    f"Model file:\n{FINAL_MODEL_FILE}"
)


saved_model = joblib.load(
    FINAL_MODEL_FILE
)


# ---------------------------------------------------------------------
# Phase 5 saves a dictionary containing the actual SVM.
# ---------------------------------------------------------------------

if isinstance(
    saved_model,
    dict
):

    if "svm" not in saved_model:

        raise ValueError(
            "The saved model dictionary "
            "does not contain the 'svm' object."
        )


    svm = saved_model["svm"]


    print(
        "Saved model format : dictionary"
    )


    print(
        f"Saved kernel       : "
        f"{saved_model.get('kernel')}"
    )


    print(
        f"Saved parameters   : "
        f"{saved_model.get('parameters')}"
    )


else:

    svm = saved_model


    print(
        "Saved model format : direct estimator"
    )


# =====================================================================
# 20. VERIFY MODEL TYPE AND DIMENSION
# =====================================================================

print()
print("=" * 70)
print("VERIFYING FINAL SVM")
print("=" * 70)


print(
    f"SVM type: "
    f"{type(svm).__name__}"
)


if hasattr(
    svm,
    "kernel"
):

    print(
        f"Kernel: "
        f"{svm.kernel}"
    )


if hasattr(
    svm,
    "C"
):

    print(
        f"C: "
        f"{svm.C}"
    )


if hasattr(
    svm,
    "degree"
):

    print(
        f"Degree: "
        f"{svm.degree}"
    )


if hasattr(
    svm,
    "gamma"
):

    print(
        f"Gamma: "
        f"{svm.gamma}"
    )


if hasattr(
    svm,
    "coef0"
):

    print(
        f"Coef0: "
        f"{svm.coef0}"
    )


if hasattr(
    svm,
    "n_features_in_"
):

    model_feature_dimension = (
        svm.n_features_in_
    )


    print(
        f"Model expects : "
        f"{model_feature_dimension}"
    )


    if (
        model_feature_dimension
        != EXPECTED_FEATURE_DIMENSION
    ):

        raise ValueError(
            "\nModel feature dimension mismatch."
            f"\nExpected: "
            f"{EXPECTED_FEATURE_DIMENSION}"
            f"\nModel: "
            f"{model_feature_dimension}"
        )


    print(
        "[OK] Model expects 2048 features."
    )


# =====================================================================
# 21. FINAL PREDICTIONS
# =====================================================================

print()
print("=" * 70)
print("GENERATING FINAL TEST PREDICTIONS")
print("=" * 70)


y_pred = svm.predict(
    X_test
)


print(
    "[OK] Predictions generated."
)


# =====================================================================
# 22. PROBABILITY SCORES
# =====================================================================

probabilities = None


if hasattr(
    svm,
    "predict_proba"
):

    print(
        "Generating probability scores..."
    )


    try:

        probabilities = svm.predict_proba(
            X_test
        )


        print(
            "[OK] Probability scores generated."
        )


    except Exception as error:

        print(
            "Probability generation failed:"
        )

        print(
            error
        )


# =====================================================================
# 23. DECISION SCORES
# =====================================================================

decision_scores = None


if hasattr(
    svm,
    "decision_function"
):

    try:

        decision_scores = svm.decision_function(
            X_test
        )


        print(
            "[OK] Decision scores generated."
        )


    except Exception as error:

        print(
            "Decision score generation failed:"
        )

        print(
            error
        )


# =====================================================================
# 24. FINAL METRICS
# =====================================================================

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


# =====================================================================
# 25. PRINT FINAL RESULTS
# =====================================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)


print(
    f"Accuracy  : "
    f"{accuracy * 100:.4f}%"
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


# =====================================================================
# 26. CLASSIFICATION REPORT
# =====================================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    report
)


report_file = os.path.join(
    REPORT_DIR,
    "final_classification_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )


# =====================================================================
# 27. CONFUSION MATRIX
# =====================================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)


print()
print("=" * 70)
print("FINAL CONFUSION MATRIX")
print("=" * 70)

print(
    cm
)


# ---------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------

confusion_csv = os.path.join(
    REPORT_DIR,
    "final_confusion_matrix.csv"
)


confusion_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


confusion_df.to_csv(
    confusion_csv
)


# ---------------------------------------------------------------------
# Save image
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7, 6)
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
    "Final Model Confusion Matrix"
)


plt.tight_layout()


confusion_png = os.path.join(
    PLOT_DIR,
    "final_confusion_matrix.png"
)


plt.savefig(
    confusion_png,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =====================================================================
# 28. PER-CLASS METRICS
# =====================================================================

print()
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)


report_dict = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


per_class_rows = []


for class_name in CLASS_NAMES:

    per_class_rows.append({

        "Class":
            class_name,

        "Precision":
            report_dict[class_name]["precision"],

        "Recall":
            report_dict[class_name]["recall"],

        "F1_Score":
            report_dict[class_name]["f1-score"],

        "Support":
            int(
                report_dict[class_name]["support"]
            )
    })


per_class_df = pd.DataFrame(
    per_class_rows
)


print(
    per_class_df.to_string(
        index=False
    )
)


per_class_file = os.path.join(
    REPORT_DIR,
    "final_per_class_metrics.csv"
)


per_class_df.to_csv(
    per_class_file,
    index=False
)


# =====================================================================
# 29. PREDICTION DATAFRAME
# =====================================================================

prediction_df = pd.DataFrame({

    "sample_index":
        np.arange(
            len(y_test)
        ),

    "true_label":
        y_test,

    "true_class":
        [
            CLASS_NAMES[index]
            for index in y_test
        ],

    "predicted_label":
        y_pred,

    "predicted_class":
        [
            CLASS_NAMES[index]
            for index in y_pred
        ],

    "correct":
        (
            y_test == y_pred
        )
})


# =====================================================================
# 30. ADD PROBABILITY COLUMNS
# =====================================================================

if probabilities is not None:

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        prediction_df[
            f"prob_{class_name}"
        ] = probabilities[
            :,
            class_index
        ]


# =====================================================================
# 31. ADD CONFIDENCE
# =====================================================================

if probabilities is not None:

    prediction_df[
        "confidence"
    ] = np.max(
        probabilities,
        axis=1
    )


# =====================================================================
# 32. SAVE ALL PREDICTIONS
# =====================================================================

prediction_file = os.path.join(
    PREDICTION_DIR,
    "final_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print()
print(
    f"Predictions saved to:\n"
    f"{prediction_file}"
)


# =====================================================================
# 33. CORRECT PREDICTIONS
# =====================================================================

correct_df = prediction_df[
    prediction_df["correct"] == True
].copy()


correct_file = os.path.join(
    ANALYSIS_DIR,
    "correct_predictions.csv"
)


correct_df.to_csv(
    correct_file,
    index=False
)


# =====================================================================
# 34. INCORRECT PREDICTIONS
# =====================================================================

incorrect_df = prediction_df[
    prediction_df["correct"] == False
].copy()


incorrect_file = os.path.join(
    ANALYSIS_DIR,
    "incorrect_predictions.csv"
)


incorrect_df.to_csv(
    incorrect_file,
    index=False
)


print()
print(
    f"Correct predictions   : "
    f"{len(correct_df)}"
)

print(
    f"Incorrect predictions : "
    f"{len(incorrect_df)}"
)


# =====================================================================
# 35. ERROR ANALYSIS
# =====================================================================

print()
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


error_rows = []


for true_index, true_class in enumerate(
    CLASS_NAMES
):

    for predicted_index, predicted_class in enumerate(
        CLASS_NAMES
    ):

        if true_index == predicted_index:

            continue


        count = np.sum(
            (
                y_test == true_index
            )
            &
            (
                y_pred == predicted_index
            )
        )


        if count > 0:

            error_rows.append({

                "True_Class":
                    true_class,

                "Predicted_Class":
                    predicted_class,

                "Count":
                    int(count)
            })


if len(error_rows) > 0:

    error_df = pd.DataFrame(
        error_rows
    )


    error_df = error_df.sort_values(
        "Count",
        ascending=False
    )


    print(
        error_df.to_string(
            index=False
        )
    )


else:

    error_df = pd.DataFrame(
        columns=[
            "True_Class",
            "Predicted_Class",
            "Count"
        ]
    )


    print(
        "No classification errors."
    )


error_file = os.path.join(
    ANALYSIS_DIR,
    "error_analysis.csv"
)


error_df.to_csv(
    error_file,
    index=False
)


# =====================================================================
# 36. ROC / AUC
# =====================================================================

print()
print("=" * 70)
print("ROC CURVES AND AUC")
print("=" * 70)


y_test_binary = label_binarize(
    y_test,
    classes=np.arange(NUM_CLASSES)
)


# ---------------------------------------------------------------------
# Prefer probability scores
# ---------------------------------------------------------------------

roc_scores = None


if probabilities is not None:

    roc_scores = probabilities

elif (
    decision_scores is not None
    and decision_scores.ndim == 2
):

    roc_scores = decision_scores


roc_rows = []


macro_auc = None


if roc_scores is not None:

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )


    class_auc_values = []


    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        fpr, tpr, _ = roc_curve(
            y_test_binary[
                :,
                class_index
            ],

            roc_scores[
                :,
                class_index
            ]
        )


        class_auc = auc(
            fpr,
            tpr
        )


        class_auc_values.append(
            class_auc
        )


        roc_rows.append({

            "Class":
                class_name,

            "AUC":
                class_auc
        })


        ax.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC = {class_auc:.4f})"
            )
        )


    # ---------------------------------------------------------------
    # Micro-average ROC
    # ---------------------------------------------------------------

    micro_fpr, micro_tpr, _ = roc_curve(
        y_test_binary.ravel(),
        roc_scores.ravel()
    )


    micro_auc = auc(
        micro_fpr,
        micro_tpr
    )


    roc_rows.append({

        "Class":
            "micro-average",

        "AUC":
            micro_auc
    })


    ax.plot(
        micro_fpr,
        micro_tpr,
        linestyle="--",
        label=(
            f"Micro-average "
            f"(AUC = {micro_auc:.4f})"
        )
    )


    # ---------------------------------------------------------------
    # Random classifier reference
    # ---------------------------------------------------------------

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=":"
    )


    ax.set_xlabel(
        "False Positive Rate"
    )


    ax.set_ylabel(
        "True Positive Rate"
    )


    ax.set_title(
        "Final Model ROC Curves"
    )


    ax.legend(
        loc="lower right"
    )


    plt.tight_layout()


    roc_png = os.path.join(
        PLOT_DIR,
        "final_roc_curves.png"
    )


    plt.savefig(
        roc_png,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ---------------------------------------------------------------
    # Macro AUC
    # ---------------------------------------------------------------

    macro_auc = np.mean(
        class_auc_values
    )


    roc_rows.append({

        "Class":
            "macro-average",

        "AUC":
            macro_auc
    })


    roc_df = pd.DataFrame(
        roc_rows
    )


    roc_csv = os.path.join(
        REPORT_DIR,
        "final_auc_results.csv"
    )


    roc_df.to_csv(
        roc_csv,
        index=False
    )


    print(
        roc_df.to_string(
            index=False
        )
    )


else:

    print(
        "ROC/AUC could not be generated "
        "because no suitable scores were available."
    )


# =====================================================================
# 37. SAVE FINAL METRICS JSON
# =====================================================================

final_metrics = {

    "model":
        "MobileNetV2 + ConvNeXt",

    "feature_dimension":
        int(X_test.shape[1]),

    "pca":
        False,

    "classifier":
        "SVM",

    "kernel":
        "Polynomial",

    "C":
        EXPECTED_C,

    "degree":
        EXPECTED_POLY_DEGREE,

    "gamma":
        EXPECTED_GAMMA,

    "coef0":
        EXPECTED_COEF0,

    "training_samples":
        int(X_train.shape[0]),

    "test_samples":
        int(X_test.shape[0]),

    "accuracy":
        float(accuracy),

    "accuracy_percent":
        float(
            accuracy * 100
        ),

    "weighted_precision":
        float(precision),

    "weighted_recall":
        float(recall),

    "weighted_f1":
        float(f1),

    "correct_predictions":
        int(len(correct_df)),

    "incorrect_predictions":
        int(len(incorrect_df)),

    "macro_auc":
        (
            float(macro_auc)
            if macro_auc is not None
            else None
        )
}


metrics_file = os.path.join(
    REPORT_DIR,
    "final_metrics.json"
)


with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_metrics,
        file,
        indent=4
    )


# =====================================================================
# 38. SAVE FINAL MODEL CONFIGURATION
# =====================================================================

final_configuration = {

    "phase":
        "Phase 6",

    "feature_models": [

        "MobileNetV2",

        "ConvNeXt"
    ],

    "feature_dimensions": {

        "MobileNetV2":
            1280,

        "ConvNeXt":
            768,

        "Fused":
            2048
    },

    "fusion":
        True,

    "pca":
        False,

    "classifier":
        "SVM",

    "kernel":
        "Polynomial",

    "C":
        1.0,

    "degree":
        3,

    "gamma":
        "scale",

    "coef0":
        0.0,

    "random_state":
        RANDOM_STATE,

    "classes":
        CLASS_NAMES,

    "source_phase5_model":
        FINAL_MODEL_FILE
}


configuration_file = os.path.join(
    FINAL_MODEL_DIR,
    "final_model_configuration.json"
)


with open(
    configuration_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_configuration,
        file,
        indent=4
    )


# =====================================================================
# 39. COPY FINAL MODEL
# =====================================================================

final_model_copy = os.path.join(
    FINAL_MODEL_DIR,
    "mobilenetv2_convnext_polynomial_svm.joblib"
)


joblib.dump(
    saved_model,
    final_model_copy
)


# =====================================================================
# 40. FINAL SUMMARY
# =====================================================================

print()
print("=" * 70)
print("PHASE 6 — FINAL EVALUATION COMPLETE")
print("=" * 70)


print()
print("FINAL MODEL")
print("-" * 70)


print(
    "MobileNetV2 + ConvNeXt"
)

print(
    "        ↓"
)

print(
    "Feature Fusion"
)

print(
    "        ↓"
)

print(
    "2048 Features"
)

print(
    "        ↓"
)

print(
    "No PCA"
)

print(
    "        ↓"
)

print(
    "Polynomial SVM"
)


print()
print("FINAL PERFORMANCE")
print("-" * 70)


print(
    f"Accuracy  : "
    f"{accuracy * 100:.4f}%"
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


if macro_auc is not None:

    print(
        f"Macro AUC : "
        f"{macro_auc:.4f}"
    )


print()
print("OUTPUT FILES")
print("-" * 70)


print(
    f"Final model:\n"
    f"{final_model_copy}"
)


print(
    f"\nConfusion matrix:\n"
    f"{confusion_png}"
)


print(
    f"\nROC curves:\n"
    f"{os.path.join(PLOT_DIR, 'final_roc_curves.png')}"
)


print(
    f"\nClassification report:\n"
    f"{report_file}"
)


print(
    f"\nPredictions:\n"
    f"{prediction_file}"
)


print(
    f"\nError analysis:\n"
    f"{error_file}"
)


print(
    f"\nFinal metrics:\n"
    f"{metrics_file}"
)


print()
print("=" * 70)
print("FINAL SYSTEM READY FOR PAPER")
print("=" * 70)