# Requirements Document

## Introduction

This document defines the requirements for a fish disease detection web application. The system allows users (fish farmers, aquaculture workers, veterinarians) to upload images of fish and receive an automated classification identifying whether the fish is healthy or affected by one of three diseases: Epizootic Ulcerative Syndrome (EUS), Bacterial Gill Disease, or Bacterial Red Spot Disease. The underlying model is trained on a labeled dataset of 2,450 images across four classes. The web app provides a simple, accessible interface for rapid on-site diagnosis support.

## Glossary

- **Classifier**: The trained deep learning model that performs image-based fish disease classification.
- **Web_App**: The browser-based frontend application through which users interact with the system.
- **API_Server**: The backend server that receives image uploads, invokes the Classifier, and returns prediction results.
- **Prediction**: The output of the Classifier, consisting of a disease class label and a confidence score.
- **Disease_Class**: One of four categories — `healthy`, `eus`, `gill`, or `red_spot`.
- **Confidence_Score**: A floating-point value between 0.0 and 1.0 representing the Classifier's certainty for the predicted Disease_Class.
- **Upload**: The act of a user submitting an image file to the Web_App for classification.
- **Supported_Format**: Image file formats accepted by the system: JPEG, JPG, and PNG.
- **EUS**: Epizootic Ulcerative Syndrome — a fungal/bacterial fish disease causing ulcerative lesions.
- **Bacterial_Gill_Disease**: A bacterial infection affecting fish gill tissue.
- **Bacterial_Red_Spot_Disease**: A bacterial infection causing red hemorrhagic spots on fish skin.

---

## Requirements

### Requirement 1: Image Upload

**User Story:** As a fish farmer, I want to upload a photo of my fish, so that I can get an automated disease assessment without needing laboratory equipment.

#### Acceptance Criteria

1. THE Web_App SHALL provide an image upload interface that accepts files in Supported_Format.
2. WHEN a user selects an image file in a Supported_Format, THE Web_App SHALL display a preview of the selected image before submission.
3. IF a user attempts to upload a file that is not in a Supported_Format, THEN THE Web_App SHALL display an error message stating the accepted file types and reject the file.
4. IF a user attempts to upload a file larger than 10 MB, THEN THE Web_App SHALL display an error message indicating the maximum allowed file size and reject the file.
5. WHEN a user submits an uploaded image, THE Web_App SHALL send the image to the API_Server for classification.

---

### Requirement 2: Disease Classification

**User Story:** As a fish farmer, I want the system to classify my fish image into a disease category, so that I can take appropriate action quickly.

#### Acceptance Criteria

1. WHEN the API_Server receives a valid image, THE Classifier SHALL classify the image into exactly one Disease_Class from: `healthy`, `eus`, `gill`, or `red_spot`.
2. WHEN the Classifier produces a Prediction, THE API_Server SHALL return the predicted Disease_Class and its Confidence_Score to the Web_App.
3. THE Classifier SHALL produce a Prediction within 10 seconds of receiving a valid image on standard server hardware.
4. IF the API_Server encounters an internal error during classification, THEN THE API_Server SHALL return an error response with an HTTP 500 status code and a descriptive error message.

---

### Requirement 3: Display of Results

**User Story:** As a fish farmer, I want to see the classification result clearly on screen, so that I can understand the diagnosis at a glance.

#### Acceptance Criteria

1. WHEN the Web_App receives a Prediction from the API_Server, THE Web_App SHALL display the predicted Disease_Class label in human-readable form (e.g., "Healthy", "EUS", "Bacterial Gill Disease", "Bacterial Red Spot Disease").
2. WHEN the Web_App receives a Prediction, THE Web_App SHALL display the Confidence_Score as a percentage rounded to one decimal place (e.g., "92.3%").
3. WHEN the predicted Disease_Class is not `healthy`, THE Web_App SHALL display a brief description of the identified disease alongside the result.
4. WHEN the Web_App displays a result, THE Web_App SHALL also display the uploaded image alongside the Prediction for visual confirmation.
5. WHEN the predicted Disease_Class is `eus`, THE Web_App SHALL display recommended treatment measures for EUS alongside the disease description, including: improving water quality, reducing stocking density, applying antifungal treatments (e.g., potassium permanganate bath), and consulting a veterinarian for antibiotic therapy if secondary bacterial infection is present.
6. WHEN the predicted Disease_Class is `gill`, THE Web_App SHALL display recommended treatment measures for Bacterial Gill Disease alongside the disease description, including: improving water quality and aeration, reducing organic load, applying salt baths or potassium permanganate treatments, and administering antibiotics as prescribed by a veterinarian.
7. WHEN the predicted Disease_Class is `red_spot`, THE Web_App SHALL display recommended treatment measures for Bacterial Red Spot Disease alongside the disease description, including: isolating affected fish, improving water quality, applying antibiotic baths (e.g., oxytetracycline), and consulting a veterinarian for systemic antibiotic treatment if the infection is severe.

---

### Requirement 4: Model Training Pipeline

**User Story:** As a developer, I want a reproducible model training pipeline, so that the Classifier can be retrained when new data becomes available.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL load images from the `split_dataset/train` directory, organized by Disease_Class subdirectory.
2. THE Training_Pipeline SHALL apply data augmentation (horizontal flip, rotation up to 20 degrees, zoom up to 20%) to training images to improve generalization.
3. WHEN training is complete, THE Training_Pipeline SHALL save the trained Classifier model to a designated output path in a format loadable by the API_Server.
4. WHEN training is complete, THE Training_Pipeline SHALL output the final training accuracy and validation accuracy to the console.
5. THE Training_Pipeline SHALL evaluate the trained Classifier on images from the `split_dataset/test` directory and output per-class precision, recall, and F1-score.

---

### Requirement 5: Model Performance

**User Story:** As a developer, I want the Classifier to meet a minimum accuracy threshold, so that the system provides reliable disease assessments.

#### Acceptance Criteria

1. THE Classifier SHALL achieve an overall classification accuracy of at least 80% on the test split of the dataset.
2. THE Classifier SHALL achieve a per-class F1-score of at least 0.75 for each Disease_Class on the test split.

---

### Requirement 6: API Interface

**User Story:** As a developer, I want a well-defined API endpoint for image classification, so that the Web_App and any future clients can integrate reliably.

#### Acceptance Criteria

1. THE API_Server SHALL expose a POST endpoint at `/predict` that accepts a multipart/form-data request containing an image file.
2. WHEN the `/predict` endpoint receives a valid request, THE API_Server SHALL return a JSON response containing the fields `predicted_class` (string) and `confidence` (float between 0.0 and 1.0).
3. IF the `/predict` endpoint receives a request with no image file, THEN THE API_Server SHALL return an HTTP 400 status code with a JSON error message.
4. IF the `/predict` endpoint receives an image file in an unsupported format, THEN THE API_Server SHALL return an HTTP 422 status code with a JSON error message.
5. THE API_Server SHALL respond to valid `/predict` requests within 10 seconds under normal operating conditions.

---

### Requirement 7: User Interface Accessibility and Usability

**User Story:** As a fish farmer with limited technical experience, I want the web interface to be simple and intuitive, so that I can use it without training.

#### Acceptance Criteria

1. THE Web_App SHALL be usable on modern desktop and mobile browsers (Chrome, Firefox, Safari, Edge — latest two major versions).
2. THE Web_App SHALL display all text content with a minimum contrast ratio of 4.5:1 against its background, in compliance with WCAG 2.1 Level AA.
3. THE Web_App SHALL provide clear visual feedback (e.g., a loading indicator) WHILE a classification request is in progress.
4. WHEN a classification request fails due to a network error, THE Web_App SHALL display a user-readable error message and provide an option to retry the Upload.

---

### Requirement 8: Application Deployment

**User Story:** As a developer, I want the application to be runnable locally with minimal setup, so that it can be demonstrated and tested easily.

#### Acceptance Criteria

1. THE Web_App SHALL be servable as a static frontend that communicates with the API_Server over HTTP.
2. THE API_Server SHALL be startable with a single command after dependencies are installed.
3. THE API_Server SHALL load the trained Classifier model from a configurable file path specified via an environment variable or configuration file.
4. IF the Classifier model file is not found at the configured path on startup, THEN THE API_Server SHALL log an error message and exit with a non-zero exit code.
