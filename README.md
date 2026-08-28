# Fish Disease Classification — Research Project

A deep learning research project for classifying fish diseases into 4 categories using CNN-based transfer learning and feature fusion with SVM.

**Classes:**
- Healthy
- EUS (Epizootic Ulcerative Syndrome)
- Gill Disease
- Red Spot Disease

**Models:** VGG16 · MobileNetV2 · ConvNeXtTiny

---

## Results Summary

| Model | Accuracy | F1-Score | Training Time |
|---|---|---|---|
| MobileNetV2 | **95.96%** | 0.9596 | 112 min |
| VGG16 | 92.68% | 0.9266 | 389 min |
| ConvNeXtTiny | 90.88% | 0.9100 | Early stopped |

**Fusion (VGG16 + MobileNetV2):** 97.62% accuracy

---

## Pre-trained Models & Features

Due to GitHub file size limits, trained models and extracted features are hosted on Google Drive.

**Download:** [Google Drive — Results Folder](https://drive.google.com/drive/folders/1SCQ9_MQP6epA5G58tCFaAJQLpAHRKEay?usp=drive_link)

After downloading, place the files as follows:

```
results/
├── VGG16/model/         ← vgg16_best.keras, vgg16_final.keras
├── MobileNetV2/model/   ← mobilenetv2_best.keras, mobilenetv2_final.keras
├── ConvNeXt/model/      ← convnext_best.keras
└── features/            ← all .npz feature files
```

---

## Project Structure

```
MainProject/
├── dataset/                        # Original images (4 classes)
├── augmented_dataset/              # Augmented images (21,000 total)
├── split_dataset/                  # Train / Validation / Test splits
│   ├── train/                      # 12,596 images (60%)
│   ├── validation/                 # 4,194  images (20%)
│   └── test/                       # 4,210  images (20%)
├── results/
│   ├── dataset/                    # Dataset analysis graphs and CSV
│   ├── VGG16/                      # VGG16 plots, reports, model
│   ├── MobileNetV2/                # MobileNetV2 plots, reports, model
│   ├── ConvNeXt/                   # ConvNeXt plots, reports, model
│   ├── features/                   # Extracted feature vectors (.npz)
│   ├── fusion/                     # Fusion experiment results
│   └── comparison/                 # Cross-model comparison charts
└── src/
    ├── augmentation.py             # Data augmentation (15 techniques)
    ├── split_dataset.py            # Train/val/test split
    ├── normalization.py            # Pixel normalization (÷255)
    ├── dataset_analysis.py         # Dataset statistics and graphs
    ├── vgg16_training.py           # VGG16 training
    ├── train_mobilenetv2.py        # MobileNetV2 training (2-phase)
    ├── train_convnext.py           # ConvNeXtTiny training (2-phase)
    ├── evaluate_convnext.py        # Evaluate saved ConvNeXt model
    ├── feature_extraction.py       # Extract GAP features from all models
    ├── fusion_experiments.py       # Run all 7 fusion combinations
    ├── generate_analysis.py        # Generate enhanced analysis plots
    ├── generate_convnext_curves.py # Reconstruct ConvNeXt training curves
    ├── visualize_augmentation.py   # Visualize all augmentation steps
    └── visualize_normalization.py  # Visualize normalization effect
```

---

## Setup

### 1. Install dependencies

```powershell
pip install tensorflow scikit-learn pandas numpy matplotlib opencv-python
```

### 2. Download pre-trained models

Download from Google Drive link above and place in the correct folders.

---

## Pipeline — Run Order

### Stage 1 — Data Preparation
```powershell
python src/augmentation.py       # Generate augmented dataset
python src/split_dataset.py      # Split into train/val/test
```

### Stage 2 — Analysis
```powershell
python src/dataset_analysis.py   # Generate dataset statistics
```

### Stage 3 — Training
```powershell
python src/vgg16_training.py         # Train VGG16
python src/train_mobilenetv2.py      # Train MobileNetV2
python src/train_convnext.py         # Train ConvNeXtTiny (overnight)
```

### Stage 4 — Feature Extraction & Fusion
```powershell
python src/feature_extraction.py     # Extract features from all models
python src/fusion_experiments.py     # Run all 7 fusion combinations
```

### Stage 5 — Analysis & Visualization
```powershell
python src/generate_analysis.py      # Generate all analysis plots
python src/visualize_augmentation.py # Show augmentation steps
python src/visualize_normalization.py # Show normalization effect
```

---

## Dataset

| Class | Original | Augmented | Train | Val | Test |
|---|---|---|---|---|---|
| EUS | 864 | 4,856 | 2,912 | 970 | 974 |
| Gill Disease | 1,312 | 5,248 | 3,148 | 1,048 | 1,052 |
| Healthy | 1,362 | 5,448 | 3,268 | 1,088 | 1,092 |
| Red Spot | 1,362 | 5,448 | 3,268 | 1,088 | 1,092 |
| **Total** | **4,900** | **21,000** | **12,596** | **4,194** | **4,210** |

All images resized to **224 × 224 × 3**.

---

## Fusion Experiment Results

| Combination | Accuracy | F1-Score | Feature Dim |
|---|---|---|---|
| VGG16 | 94.30% | 0.9429 | 512 |
| MobileNetV2 | 96.96% | 0.9696 | 1,280 |
| ConvNeXt | — | — | 768 |
| VGG16 + MobileNetV2 | **97.62%** | **0.9762** | 1,792 |
| VGG16 + ConvNeXt | — | — | 1,280 |
| MobileNetV2 + ConvNeXt | — | — | 2,048 |
| VGG16 + MobileNetV2 + ConvNeXt | — | — | 2,560 |

> ConvNeXt fusion combinations will be added after full training completes.
