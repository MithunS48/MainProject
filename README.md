# Fish Disease Detector

A full-stack web application that classifies fish images into four categories using a MobileNetV2-based CNN:

- **Healthy** — no disease detected
- **EUS** — Epizootic Ulcerative Syndrome
- **Bacterial Gill Disease**
- **Bacterial Red Spot Disease**

The system consists of a Python training pipeline, a FastAPI inference server, and a static HTML/CSS/JS frontend.

---

## Prerequisites

- Python 3.9 or higher
- `pip`
- A modern web browser (Chrome, Firefox, Edge, Safari)

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 2. Train the model

Run the training pipeline against the included dataset. This produces the model file the API server requires.

```bash
python train.py --epochs 20 --output model/fish_disease_model.keras
```

Training prints per-epoch accuracy and a per-class classification report on completion. The model is saved to `model/fish_disease_model.keras`.

> **Note:** The `split_dataset/` directory must exist before training. If you only have the raw `dataset/` folder, split it into train/validation/test subsets first.

---

### 3. Start the API server

The server reads the model path from the `MODEL_PATH` environment variable (default: `model/fish_disease_model.keras`).

**Linux / macOS:**
```bash
MODEL_PATH=model/fish_disease_model.keras uvicorn app:app --reload
```

**Windows (Command Prompt):**
```cmd
set MODEL_PATH=model/fish_disease_model.keras && uvicorn app:app --reload
```

**Windows (PowerShell):**
```powershell
$env:MODEL_PATH="model/fish_disease_model.keras"; uvicorn app:app --reload
```

Or simply (uses the default path):
```bash
uvicorn app:app --reload
```

The server starts on `http://localhost:8000`. Interactive API docs are available at `http://localhost:8000/docs`.

> **Startup failure:** If the model file is not found at the configured path, the server logs an error and exits with code 1. Train the model first or set `MODEL_PATH` to the correct path.

---

### 4. Open the web app

Open `index.html` directly in your browser — no build step required.

```
file:///path/to/project/index.html
```

Or double-click `index.html` in your file manager.

Upload a fish image (JPG, JPEG, or PNG, max 10 MB), click **Analyze**, and the app will display the predicted disease class, confidence score, and treatment recommendations.

---

## Running tests

### Python tests

```bash
pytest tests/test_api.py -v
```

### JavaScript tests

```bash
npm install
npm test
```

---

## Project structure

```
fish-disease-detection/
├── dataset/                  # Raw labeled images (eus, gill, healthy, red_spot)
├── split_dataset/            # Train/validation/test splits
│   ├── train/
│   ├── validation/
│   └── test/
├── model/
│   └── .gitkeep              # Placeholder; trained model saved here
├── tests/
│   └── test_api.py           # Python unit tests for API helpers
├── train.py                  # Training pipeline (MobileNetV2 transfer learning)
├── app.py                    # FastAPI inference server
├── index.html                # Web app entry point
├── style.css                 # Responsive styles
├── app.js                    # Frontend logic (upload, preview, API call, results)
├── requirements.txt          # Python dependencies
├── package.json              # JS dev dependencies (vitest, fast-check)
└── README.md
```

---

## API reference

### `POST /predict`

Classify a fish image.

**Request:** `multipart/form-data` with a `file` field (JPEG or PNG).

**Response (200):**
```json
{
  "predicted_class": "healthy",
  "confidence": 0.9731
}
```

**Error responses:**
- `400` — no file provided
- `422` — unsupported file format
- `500` — inference error

---

## Disclaimer

This tool is for informational purposes only. Consult a veterinarian for professional advice.
