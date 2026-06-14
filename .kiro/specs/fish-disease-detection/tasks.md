   # Implementation Plan: Fish Disease Detection Web Application

## Overview

Implement a full-stack fish disease detection system consisting of three components: a Python training pipeline (`train.py`) that fine-tunes a MobileNetV2 CNN on the labeled dataset, a FastAPI server (`app.py`) that loads the trained model and exposes a `/predict` endpoint, and a static HTML/CSS/JavaScript frontend (`index.html`, `style.css`, `app.js`) that handles image upload, preview, and result display. Implementation proceeds bottom-up: training pipeline first, then API server, then frontend, with integration wiring at the end.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create the `model/` output directory placeholder (`.gitkeep`)
  - Create `requirements.txt` with pinned versions: `tensorflow==2.16.*`, `fastapi==0.111.*`, `uvicorn[standard]==0.29.*`, `python-multipart==0.0.9`, `Pillow==10.3.*`, `scikit-learn==1.4.*`, `numpy==1.26.*`, `hypothesis==6.100.*`
  - Create `tests/` directory with empty `__init__.py`
  - Create `package.json` in project root with `vitest` and `fast-check` as dev dependencies for JS property tests
  - _Requirements: 8.1, 8.2_

- [x] 2. Implement the training pipeline (`train.py`)
  - [x] 2.1 Implement data loading and augmentation
    - Use `tf.keras.preprocessing.image.ImageDataGenerator` to load from `split_dataset/train`, `split_dataset/validation`, and `split_dataset/test`
    - Apply augmentation to training generator: horizontal flip, rotation range 20°, zoom range 20%
    - Normalize pixel values to [0, 1] for all generators
    - Target image size 224×224, batch size configurable via `--batch-size` (default 32)
    - _Requirements: 4.1, 4.2_

  - [x] 2.2 Implement model architecture and training
    - Build MobileNetV2 transfer-learning model: frozen base + `GlobalAveragePooling2D` + `Dense(128, relu)` + `Dropout(0.3)` + `Dense(4, softmax)`
    - Compile with Adam (lr=1e-4), categorical crossentropy, accuracy metric
    - Compute class weights from training set distribution to handle EUS imbalance
    - Train with early stopping (patience=5 on val accuracy); print per-epoch train/val accuracy
    - Accept `--epochs` (default 20) and `--output` (default `model/fish_disease_model.keras`) CLI args
    - Create output directory automatically if it does not exist
    - _Requirements: 4.3, 4.4_

  - [x] 2.3 Implement model evaluation and saving
    - After training, run inference on `split_dataset/test` generator
    - Print overall accuracy and `sklearn.metrics.classification_report` (per-class precision, recall, F1)
    - Save trained model to the configured output path in `.keras` format
    - If dataset directory is not found, print an error message and exit with non-zero code
    - _Requirements: 4.3, 4.4, 4.5, 5.1, 5.2_

- [x] 3. Checkpoint — training pipeline complete
  - Verify `train.py` runs end-to-end: `python train.py --epochs 1 --output model/fish_disease_model.keras`
  - Confirm console output includes per-epoch accuracy and a classification report
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement API server core (`app.py`)
  - [x] 4.1 Implement startup, model loading, and CORS
    - Read model path from `MODEL_PATH` environment variable (default: `model/fish_disease_model.keras`)
    - Load Keras model at startup; if file not found, log error and call `sys.exit(1)`
    - Enable CORS for all origins using `fastapi.middleware.cors.CORSMiddleware`
    - Define `CLASS_LABELS = {0: "eus", 1: "gill", 2: "healthy", 3: "red_spot"}`
    - _Requirements: 6.1, 8.3, 8.4_

  - [x] 4.2 Implement image preprocessing function
    - Write `preprocess_image(image_bytes: bytes) -> np.ndarray` that:
      1. Decodes bytes to PIL Image
      2. Converts to RGB (handles RGBA and grayscale)
      3. Resizes to 224×224
      4. Normalizes to [0.0, 1.0]
      5. Adds batch dimension → shape `(1, 224, 224, 3)`
    - _Requirements: 2.1_

  - [ ]* 4.3 Write property test for image preprocessing (Property 6)
    - **Property 6: Image preprocessing preserves shape invariant**
    - Use `hypothesis` with `st.integers` for width/height and `st.sampled_from` for image modes (RGB, RGBA, L)
    - Assert output shape is `(1, 224, 224, 3)` and all values are in [0.0, 1.0]
    - **Validates: Requirements 2.1**

  - [x] 4.4 Implement the `/predict` endpoint
    - Accept `POST /predict` with `multipart/form-data` field `file`
    - Return HTTP 400 if no file is provided
    - Validate file extension (case-insensitive) against `{jpg, jpeg, png}`; return HTTP 422 if unsupported
    - Call `preprocess_image`, run model inference, map argmax to `CLASS_LABELS`, return `{"predicted_class": str, "confidence": float}`
    - Catch inference exceptions and return HTTP 500 with descriptive message
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 2.1, 2.2, 2.3, 2.4_

  - [ ]* 4.5 Write property test for confidence and class output (Properties 1 & 2)
    - **Property 1: Confidence score is a valid probability**
    - **Property 2: Predicted class is always a known Disease_Class**
    - Use `hypothesis` to generate synthetic valid images (random RGB arrays saved as JPEG bytes)
    - Mock model inference to return random softmax-like outputs; assert `confidence` ∈ [0.0, 1.0] and `predicted_class` ∈ `{"healthy", "eus", "gill", "red_spot"}`
    - **Validates: Requirements 6.2, 2.1**

  - [ ]* 4.6 Write property test for file format validation (Property 4)
    - **Property 4: File format validation rejects non-supported types**
    - Use `hypothesis` with `st.text` to generate random file extensions
    - Assert the validation function accepts only `jpg`, `jpeg`, `png` (case-insensitive) and rejects all others
    - **Validates: Requirements 1.3, 6.4**

- [x] 5. Write API server unit tests (`tests/test_api.py`)
  - [x] 5.1 Unit test `validate_file_format`
    - Test accepted extensions: `.jpg`, `.jpeg`, `.png` and uppercase variants
    - Test rejected extensions: `.gif`, `.bmp`, `.txt`, `.pdf`, no extension
    - _Requirements: 6.4, 1.3_

  - [ ]* 5.2 Unit test `preprocess_image` output shape and range
    - Test with a small RGB JPEG, RGBA PNG, and grayscale image
    - Assert shape `(1, 224, 224, 3)` and pixel values in [0.0, 1.0]
    - _Requirements: 2.1_

  - [ ]* 5.3 Integration test `/predict` endpoint (using FastAPI `TestClient`)
    - POST a valid JPEG → assert HTTP 200, JSON has `predicted_class` in known set, `confidence` in [0, 1]
    - POST with no file → assert HTTP 400
    - POST with a `.txt` file → assert HTTP 422
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 6. Checkpoint — API server complete
  - Run `pytest tests/test_api.py` and confirm all tests pass
  - Manually start server: `uvicorn app:app --reload` and verify `/docs` loads
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement the Web App frontend
  - [x] 7.1 Create `index.html` with semantic structure
    - File input (accept `.jpg,.jpeg,.png`), drag-and-drop zone, image preview area
    - "Analyze" button (disabled until image selected)
    - Loading spinner element (hidden by default)
    - Result card section (hidden by default): disease label, confidence percentage, image thumbnail, description, treatment
    - Error banner with "Retry" button (hidden by default)
    - All interactive elements have accessible labels (`aria-label` / `<label>`)
    - _Requirements: 1.1, 1.2, 3.1, 3.2, 3.3, 3.4, 7.1, 7.2, 7.3_

  - [x] 7.2 Create `style.css` with responsive layout
    - Mobile-first layout using CSS flexbox/grid
    - Minimum text contrast ratio 4.5:1 (WCAG 2.1 AA) for all text/background pairs
    - Visible focus indicators for keyboard navigation
    - Loading spinner animation
    - Responsive breakpoints for desktop and mobile
    - _Requirements: 7.1, 7.2_

  - [x] 7.3 Implement `app.js` — file validation and preview
    - Define `const API_BASE_URL = "http://localhost:8000"` at top of file
    - Implement `validateFileType(file)`: check MIME type and extension against `{image/jpeg, image/png}` and `{jpg, jpeg, png}`
    - Implement `validateFileSize(file)`: reject files > 10 MB (10 × 1024 × 1024 bytes)
    - On file selection: run both validators; show inline error if invalid; show image preview via `FileReader` if valid; enable "Analyze" button
    - Support drag-and-drop onto the upload zone
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 7.4 Write property test for `validateFileSize` (Property 5)
    - **Property 5: File size validation rejects oversized files**
    - Use `fast-check` to generate random file sizes; assert files > 10 MB are rejected and files ≤ 10 MB are accepted
    - **Validates: Requirements 1.4**

  - [x] 7.5 Implement `app.js` — disease content data and display helpers
    - Define `DISEASE_CONTENT` object mapping each class to `{ label, description, treatment }` as specified in the design
    - Implement `getDiseaseContent(predictedClass)`: return the content object for the given class
    - Implement `formatConfidence(confidence)`: return `(confidence * 100).toFixed(1) + "%"`
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 3.7_

  - [ ]* 7.6 Write property test for confidence display round-trip (Property 3)
    - **Property 3: Confidence display round-trip**
    - Use `fast-check` with `fc.float({ min: 0, max: 1 })` to generate random confidence values
    - Assert `parseFloat(formatConfidence(c)) / 100` is within 0.001 of `c`
    - **Validates: Requirements 3.2**

  - [ ]* 7.7 Write property test for disease content completeness (Property 7)
    - **Property 7: Disease content completeness**
    - For each non-healthy class (`eus`, `gill`, `red_spot`), assert `getDiseaseContent` returns a non-empty `description` and non-empty `treatment` string
    - **Validates: Requirements 3.3, 3.5, 3.6, 3.7**

  - [x] 7.8 Implement `app.js` — API call and result rendering
    - On "Analyze" click: show loading spinner, disable button, send `POST /predict` via `fetch` with `FormData`
    - On success (HTTP 200): hide spinner, populate result card with label, confidence, image, description, treatment; show result card
    - On API error (4xx/5xx): parse error JSON and display message in error banner with "Retry" button
    - On network failure (`fetch` throws): display generic network error in error banner with "Retry" button
    - On unexpected JSON shape: display generic error message
    - "Retry" button resets UI to Preview state
    - _Requirements: 1.5, 2.2, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 7.3, 7.4_

- [ ] 8. Write Web App unit tests (`tests/test_app.js`)
  - [ ]* 8.1 Unit test `validateFileType`
    - Test accepted MIME types: `image/jpeg`, `image/png`
    - Test rejected types: `image/gif`, `application/pdf`, `text/plain`
    - _Requirements: 1.3_

  - [ ]* 8.2 Unit test `validateFileSize`
    - Test file at exactly 10 MB (accepted), 10 MB + 1 byte (rejected), 0 bytes (accepted)
    - _Requirements: 1.4_

  - [ ]* 8.3 Unit test `formatConfidence`
    - Test `0.923` → `"92.3%"`, `1.0` → `"100.0%"`, `0.0` → `"0.0%"`, `0.9999` → `"100.0%"`
    - _Requirements: 3.2_

  - [ ]* 8.4 Unit test `getDiseaseContent`
    - Assert all four classes return objects with non-empty `label` field
    - Assert `eus`, `gill`, `red_spot` return non-empty `description` and `treatment`
    - Assert `healthy` returns empty or absent `treatment`
    - _Requirements: 3.1, 3.3, 3.5, 3.6, 3.7_

- [x] 9. Checkpoint — frontend complete
  - Run `npx vitest --run` and confirm all JS tests pass
  - Open `index.html` in a browser, upload a test image, verify preview and UI states work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Integration wiring and end-to-end validation
  - [x] 10.1 Wire frontend to running API server
    - Confirm `API_BASE_URL` in `app.js` points to `http://localhost:8000`
    - Start API server with a trained model: `MODEL_PATH=model/fish_disease_model.keras uvicorn app:app`
    - Open `index.html`, upload a fish image, verify the full flow: preview → loading → result card with label, confidence, description, and treatment
    - _Requirements: 1.5, 2.2, 3.1, 3.2, 3.3, 3.4_

  - [ ]* 10.2 Write integration tests for `/predict` endpoint
    - POST a real JPEG from `split_dataset/test` → assert HTTP 200, `predicted_class` in known set, `confidence` in [0, 1]
    - POST with no file → assert HTTP 400
    - POST with `.txt` file → assert HTTP 422
    - Start server with missing model path → assert process exits non-zero
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.4_

  - [x] 10.3 Verify startup failure behavior
    - Set `MODEL_PATH` to a non-existent path, start the server, confirm it logs an error and exits with code 1
    - _Requirements: 8.4_

- [x] 11. Final checkpoint — full system verified
  - Run `pytest tests/` and `npx vitest --run` — confirm all tests pass
  - Verify `train.py --help`, `uvicorn app:app --help` work as expected
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use **Hypothesis** (Python) and **fast-check** (JavaScript), each running ≥ 100 iterations
- The training pipeline must be run first to produce `model/fish_disease_model.keras` before the API server can start
- Class weights are required to handle the EUS class imbalance (~388 train images vs ~650–676 for other classes)
- The frontend is a static file — no build step required; open `index.html` directly in a browser
