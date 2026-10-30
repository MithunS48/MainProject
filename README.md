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

**Best fusion (MobileNetV2 + ConvNeXt):** 97.77% accuracy

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

**Web Application:**

```
webapp/
├── backend/                       # FastAPI backend
│   └── app/
│       ├── main.py                 # App entry point + startup
│       ├── routers/iot.py          # IoT water-quality endpoints
│       ├── models/models.py        # SQLAlchemy ORM models
│       └── schemas/schemas.py     # Pydantic request/response schemas
└── frontend/                      # React + Vite frontend
    └── src/
        ├── pages/farmer/
        │   ├── FarmerDashboard.jsx # Dashboard with live IoT widget
        │   └── WaterQuality.jsx    # Full water-quality monitoring page
        └── utils/
            ├── waterQualityAdvisor.js  # pH/temp → fish impact, recommendations, disease risks
            └── diseaseInfo.js           # Disease descriptions and management

iot/
└── firmware/AquaScan_ESP32/        # ESP32 Arduino firmware

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
| VGG16 | 94.39% | 0.9438 | 512 |
| MobileNetV2 | 96.63% | 0.9663 | 1,280 |
| ConvNeXt | 94.01% | 0.9400 | 768 |
| VGG16 + MobileNetV2 | 97.43% | 0.9744 | 1,792 |
| VGG16 + ConvNeXt | 96.39% | 0.9638 | 1,280 |
| **MobileNetV2 + ConvNeXt** | **97.77%** | **0.9777** | **2,048** |
| VGG16 + MobileNetV2 + ConvNeXt | 97.53% | 0.9753 | 2,560 |

---

## IoT Water-Quality Monitoring Module

Integrates an ESP32 + pH sensor + DS18B20 temperature sensor into the
existing AquaScan web application. Live readings appear on the farmer
dashboard under **Water Quality**.

### Hardware Requirements

| Component | Details |
|---|---|
| ESP32 | Any Wi-Fi-capable variant (ESP32-WROOM-32, ESP32-S3, etc.) |
| pH sensor module | Analog output; compatible with 3.3 V ADC |
| DS18B20 | Waterproof probe recommended for pond use |
| 4.7 kΩ resistor | Pull-up for DS18B20 OneWire data line |

### Wiring

```
pH sensor module
  VCC  → ESP32 3.3 V (check your module — some need 5 V)
  GND  → ESP32 GND
  A0   → GPIO 34

DS18B20 (waterproof probe)
  Red   (VCC)  → ESP32 3.3 V
  Black (GND)  → ESP32 GND
  Yellow (DATA) → GPIO 4  +  4.7 kΩ pull-up to 3.3 V
```

### ESP32 Setup

1. Open `iot/firmware/AquaScan_ESP32/AquaScan_ESP32.ino` in Arduino IDE
   (or use PlatformIO with `iot/firmware/AquaScan_ESP32/platformio.ini`).
2. Install required libraries via Library Manager:
   - `OneWire` by Paul Stoffregen
   - `DallasTemperature` by Miles Burton
   - `ArduinoJson` v6 by Benoit Blanchon
3. Fill in the **CONFIGURATION** section:
   ```cpp
   const char* WIFI_SSID      = "your-wifi-ssid";
   const char* WIFI_PASSWORD  = "your-wifi-password";
   const char* BACKEND_URL    = "http://192.168.1.100:8000";
   const char* DEVICE_ID      = "ESP32-001";
   const char* DEVICE_API_KEY = "your-secret-api-key";
   ```
4. Flash to the ESP32.

> **Security**: Never commit real credentials to version control.

### pH Sensor Calibration

1. Measure the voltage output at a pH 7 buffer solution and update
   `PH_VOLTAGE_AT_PH7` in the firmware.
2. Optionally measure at pH 4 or pH 10 and adjust `PH_SLOPE`.

### Device Registration (one-time)

Before the ESP32 can POST readings, register it through the web app:

1. Log in as a farmer.
2. Go to **Dashboard → Water Quality**.
3. Click **Register Device**.
4. Enter the same `DEVICE_ID` and `DEVICE_API_KEY` as in the firmware.
5. Save the API key — it is stored as a bcrypt hash and cannot be retrieved.

### Backend API

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/iot/readings` | Device API key (`X-Device-API-Key` header) | ESP32 posts a reading |
| GET | `/api/iot/readings/latest?device_id=` | JWT Bearer | Latest reading with derived values |
| GET | `/api/iot/readings/history?device_id=` | JWT Bearer | Paginated reading history |
| GET | `/api/iot/devices/{device_id}` | JWT Bearer | Device status and thresholds |
| POST | `/api/iot/devices/register` | JWT Bearer | Register a new device |
| GET | `/api/iot/devices` | JWT Bearer (admin) | List all devices |
| POST | `/api/iot/mock/reading` | JWT Bearer | Inject a single mock reading |
| POST | `/api/iot/mock/burst` | JWT Bearer | Inject multiple readings for charts |
| POST | `/api/iot/mock/clear` | JWT Bearer | Clear all readings for a device |

#### Example ESP32 payload
```json
{
  "deviceId":    "ESP32-001",
  "ph":          6.84,
  "temperature": 28.5,
  "timestamp":   "2026-09-02T14:30:00"
}
```

### Database Schema

Two new tables are added automatically on server startup (SQLAlchemy
`create_all`):

**`iot_readings`**

| Column | Type | Notes |
|---|---|---|
| id | String PK | UUID4 |
| device_id | String, indexed | e.g. `ESP32-001` |
| ph | Float | 0 – 14 |
| temperature | Float | °C |
| raw_payload | Text | full JSON from device |
| received_at | DateTime, indexed | server-side timestamp |
| device_ts | DateTime, nullable | device-reported timestamp |

**`iot_devices`**

| Column | Type | Notes |
|---|---|---|
| id | String PK | UUID4 |
| device_id | String, unique | |
| api_key_hash | String | bcrypt hash |
| label | String, nullable | human-friendly name |
| owner_id | FK → users.id | |
| is_active | Boolean | |
| ph_min/max | Float | alert thresholds |
| temp_min/max | Float | alert thresholds |

### Derived / Calculated Values

Only scientifically defensible values calculable from pH and temperature:

| Value | Formula | Notes |
|---|---|---|
| pH status | acidic / normal / alkaline | based on configurable thresholds |
| [H⁺] concentration | 10^(−pH) mol/L | |
| pOH | 14 − pH | valid at ~25 °C; approximate otherwise |
| Temperature status | too_low / normal / too_high | based on configurable thresholds |
| Alerts | threshold comparison | configurable per device |

> Turbidity, TDS, dissolved oxygen, nitrate etc. are **not** calculated
> from pH/temperature — dedicated sensors are required for those.

### Running the IoT Module

```powershell
# Start backend (from webapp/backend/)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (from webapp/frontend/)
npm run dev
```

Navigate to `http://localhost:5173/dashboard/water-quality` after logging in
as a farmer.

### Running IoT Tests

```powershell
cd webapp/backend
pip install pytest httpx
pytest tests/test_iot.py -v
```

### Complete Data Flow

```
DS18B20 ──┐
           ├─► ESP32 (reads every 60 s)
pH sensor ─┘       │
                    │  POST /api/iot/readings
                    │  Header: X-Device-API-Key
                    ▼
             FastAPI backend
                    │  validates + stores IoTReading in SQLite
                    ▼
             iot_readings table
                    │
                    │  GET /api/iot/devices/{id}      (dashboard polls every 30 s)
                    │  GET /api/iot/readings/history
                    ▼
             React frontend
             /dashboard
                    │
                    ▼
          ┌─────────────────────────────┐
          │  Water Quality Widget         │
          │  • pH gauge + status          │
          │  • Temperature gauge + status │
          │  • Overall condition banner    │
          │  • Fish impact explanation     │
          │  • Recommended actions         │
          │  • Disease risk tags            │
          │  • Auto-refresh every 30 s     │
          └─────────────────────────────┘
```

### Water Quality Advisor

The farmer dashboard includes an intelligent advisor that analyzes live pH
and temperature readings and provides real-time guidance:

| Feature | Description |
|---|---|---|
| **Condition Assessment** | Classifies overall water quality as Optimal / Warning / Critical based on pH (6.5–8.5) and temperature (22–30 °C) |
| **Fish Impact** | Explains what is physically happening to the fish at the current readings (e.g. gill damage, immune suppression, oxygen depletion) |
| **Recommendations** | Step-by-step actionable solutions (lime dosing, water exchange %, aeration, shading nets, feeding adjustments) |
| **Disease Risk Prediction** | Lists diseases likely to appear if conditions persist (EUS, Gill Disease, Red Spot, Columnaris, Saprolegniasis, etc.) |

The advisor covers the full pH range (0–14) and temperature range
(-10 to 40 °C) with specific advice at each severity level.

### Mock / Test Mode

For testing the dashboard without an ESP32 connected, mock endpoints are
available under the **Test Mode** panel on the Water Quality page:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/iot/mock/reading?device_id=` | Inject a single random or custom reading |
| POST | `/api/iot/mock/burst?device_id=&count=20` | Inject 20 readings with drift to populate charts |
| POST | `/api/iot/mock/clear?device_id=` | Delete all readings for a device |

The mock panel lets you specify custom pH and temperature values, or leave
them blank for realistic random values. The **+20 Readings** button generates
a time series with slight drift so the history charts display properly.

### Adding Future Sensors

The schema is designed for extension. To add turbidity or TDS:

1. Add nullable columns to `IoTReading` in `models.py`:
   ```python
   turbidity = Column(Float, nullable=True)  # NTU
   ```
2. Add fields to `IoTReadingIn` / `IoTReadingOut` schemas.
3. Update `_derive()` in `iot.py` for new derived values.
4. Update ESP32 firmware to read and include the new sensor.
5. Add chart panels to `WaterQuality.jsx`.

No existing data or endpoints are affected.
