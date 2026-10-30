/**
 * AquaScan_ESP32.ino
 * ──────────────────
 * ESP32 firmware for the AquaScan IoT Water-Quality Monitor.
 *
 * Hardware
 * --------
 *  - ESP32 (any variant with Wi-Fi)
 *  - Analog pH sensor module  → GPIO 34  (ADC1 channel 6)
 *  - DS18B20 waterproof temperature sensor → GPIO 4  (OneWire bus)
 *
 * Wiring
 * ------
 *  pH sensor module:
 *    VCC  → ESP32 3.3 V (some modules need 5 V — check your module's datasheet)
 *    GND  → ESP32 GND
 *    A0   → GPIO 34  (ADC — use 3.3 V tolerant pin)
 *
 *  DS18B20 (TO-92 or waterproof probe):
 *    VCC  (red)   → ESP32 3.3 V
 *    GND  (black) → ESP32 GND
 *    DATA (yellow/white) → GPIO 4 + 4.7 kΩ pull-up resistor to 3.3 V
 *
 * Required Arduino libraries (install via Library Manager or platformio.ini)
 * --------------------------------------------------------------------------
 *  - OneWire          (Paul Stoffregen)
 *  - DallasTemperature (Miles Burton)
 *  - ArduinoJson      v6 or v7  (Benoit Blanchon)
 *  - WiFi             (built-in ESP32 core)
 *  - HTTPClient       (built-in ESP32 core)
 *
 * Configuration
 * -------------
 * Fill in the CONFIGURATION section below.
 * DO NOT commit credentials to a public repository.
 * Use a secrets file or environment variables instead.
 *
 * How it works
 * ------------
 *  1. Connects to Wi-Fi on boot.
 *  2. Every READ_INTERVAL_MS (default 60 s), reads pH and temperature.
 *  3. Validates both readings (plausibility checks).
 *  4. If valid, POSTs JSON payload to BACKEND_URL/api/iot/readings
 *     with X-Device-API-Key header for authentication.
 *  5. Handles Wi-Fi reconnect and HTTP errors gracefully.
 *  6. Serial monitor output at 115200 baud for debugging.
 *
 * Payload sent to backend
 * -----------------------
 *  {
 *    "deviceId":    "ESP32-001",
 *    "ph":          6.84,
 *    "temperature": 28.5,
 *    "timestamp":   "2026-09-02T14:30:00"
 *  }
 */

// ─────────────────────────────────────────────────────────────────────────────
// DEPENDENCIES
// ─────────────────────────────────────────────────────────────────────────────
#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>
#include <time.h>

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURATION  ← fill in before flashing
// ─────────────────────────────────────────────────────────────────────────────

// Wi-Fi credentials
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// AquaScan backend
const char* BACKEND_URL   = "http://192.168.1.100:8000";   // change to your server IP/domain
const char* DEVICE_ID     = "ESP32-001";                    // must match registration
const char* DEVICE_API_KEY = "YOUR_DEVICE_API_KEY";         // set during device registration

// NTP for timestamps
const char* NTP_SERVER    = "pool.ntp.org";
const long  GMT_OFFSET_SEC   = 0;      // UTC; adjust for local time zone
const int   DAYLIGHT_OFFSET_SEC = 0;

// Reading interval
const unsigned long READ_INTERVAL_MS = 60000UL;  // 60 seconds

// Pin definitions
const int PH_PIN    = 34;   // ADC1 CH6 — must be ADC1 for reliable WiFi use
const int DS18B20_PIN = 4;  // OneWire data pin

// pH calibration
// Measure voltage at known pH buffer solutions and adjust these values.
// Typical values for a 3.3 V system:
const float PH_VOLTAGE_AT_PH7  = 1.65f;  // midpoint voltage at pH 7
const float PH_SLOPE            = -0.18f; // V per pH unit (negative because pH ↑ → V ↓)

// Valid reading ranges (outside these → sensor error / disconnected)
const float PH_MIN_VALID   = 0.0f;
const float PH_MAX_VALID   = 14.0f;
const float TEMP_MIN_VALID = -10.0f;
const float TEMP_MAX_VALID = 100.0f;

// ADC calibration (ESP32 ADC is non-linear — apply correction)
const int   ADC_SAMPLES    = 64;          // average this many samples
const float ADC_VREF       = 3.3f;        // ESP32 supply voltage
const int   ADC_RESOLUTION = 4095;        // 12-bit ADC

// ─────────────────────────────────────────────────────────────────────────────
// GLOBALS
// ─────────────────────────────────────────────────────────────────────────────
OneWire           oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

unsigned long lastReadMs = 0;
bool          ntpSynced  = false;

// ─────────────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n========================================");
  Serial.println("  AquaScan ESP32 Water-Quality Monitor");
  Serial.println("========================================");

  // Initialise DS18B20
  ds18b20.begin();
  Serial.printf("DS18B20 devices found: %d\n", ds18b20.getDeviceCount());

  // Connect to Wi-Fi
  connectWiFi();

  // Sync NTP time
  if (WiFi.status() == WL_CONNECTED) {
    configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER);
    Serial.print("Syncing NTP time");
    for (int i = 0; i < 20; i++) {
      delay(500);
      Serial.print(".");
      time_t now;
      time(&now);
      if (now > 1000000000UL) { ntpSynced = true; break; }
    }
    Serial.println(ntpSynced ? " OK" : " FAILED (will use empty timestamp)");
  }

  Serial.println("\nSetup complete. Starting readings...\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  if (now - lastReadMs >= READ_INTERVAL_MS || lastReadMs == 0) {
    lastReadMs = now;

    // Ensure Wi-Fi is connected
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] Disconnected — reconnecting...");
      connectWiFi();
      if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Reconnect failed. Skipping this cycle.");
        return;
      }
    }

    float ph   = readPH();
    float temp = readTemperature();

    Serial.printf("[Sensors] pH=%.3f  Temp=%.2f °C\n", ph, temp);

    // Validate
    bool phValid   = (ph   >= PH_MIN_VALID   && ph   <= PH_MAX_VALID);
    bool tempValid = (temp >= TEMP_MIN_VALID  && temp <= TEMP_MAX_VALID);

    if (!phValid) {
      Serial.printf("[Error] pH reading out of range (%.3f). Check sensor.\n", ph);
    }
    if (!tempValid) {
      Serial.printf("[Error] Temperature reading out of range (%.2f). Check DS18B20.\n", temp);
    }

    if (phValid && tempValid) {
      sendReading(ph, temp);
    } else {
      Serial.println("[Skip] Invalid reading — not sending to backend.");
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SENSOR READING
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Read pH via averaged ADC sampling and linear calibration.
 * For best accuracy, calibrate using pH 4, 7, and 10 buffer solutions.
 */
float readPH() {
  long sum = 0;
  for (int i = 0; i < ADC_SAMPLES; i++) {
    sum += analogRead(PH_PIN);
    delay(2);
  }
  float raw     = (float)sum / ADC_SAMPLES;
  float voltage = raw * (ADC_VREF / ADC_RESOLUTION);
  float ph      = 7.0f + (voltage - PH_VOLTAGE_AT_PH7) / PH_SLOPE;
  return ph;
}

/**
 * Read temperature from DS18B20.
 * Returns DEVICE_DISCONNECTED_C (-127) on failure — caller must validate.
 */
float readTemperature() {
  ds18b20.requestTemperatures();
  float temp = ds18b20.getTempCByIndex(0);
  return temp;
}

// ─────────────────────────────────────────────────────────────────────────────
// NETWORK
// ─────────────────────────────────────────────────────────────────────────────

void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected. IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Connection failed.");
  }
}

/**
 * Send a reading to the AquaScan backend.
 *
 * POST /api/iot/readings
 * Headers:
 *   Content-Type: application/json
 *   X-Device-API-Key: <DEVICE_API_KEY>
 * Body: { deviceId, ph, temperature, timestamp }
 */
void sendReading(float ph, float temp) {
  HTTPClient http;
  String url = String(BACKEND_URL) + "/api/iot/readings";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-API-Key", DEVICE_API_KEY);
  http.setTimeout(10000);  // 10 s timeout

  // Build JSON payload
  StaticJsonDocument<256> doc;
  doc["deviceId"]    = DEVICE_ID;
  doc["ph"]          = serialized(String(ph, 3));
  doc["temperature"] = serialized(String(temp, 2));
  doc["timestamp"]   = getISO8601Timestamp();

  String body;
  serializeJson(doc, body);

  Serial.printf("[HTTP] POST %s\n", url.c_str());
  Serial.printf("[HTTP] Body: %s\n", body.c_str());

  int httpCode = http.POST(body);

  if (httpCode > 0) {
    Serial.printf("[HTTP] Response code: %d\n", httpCode);
    if (httpCode == 201) {
      Serial.println("[HTTP] Reading stored successfully.");
    } else {
      Serial.printf("[HTTP] Unexpected response: %s\n", http.getString().c_str());
    }
  } else {
    Serial.printf("[HTTP] Request failed: %s\n", http.errorToString(httpCode).c_str());
    Serial.println("[HTTP] Backend may be unreachable. Will retry next cycle.");
  }

  http.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Return current time as ISO-8601 string, e.g. "2026-09-02T14:30:00".
 * Returns empty string if NTP is not synced.
 */
String getISO8601Timestamp() {
  if (!ntpSynced) return "";

  time_t now;
  struct tm timeInfo;
  time(&now);
  localtime_r(&now, &timeInfo);

  char buf[25];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &timeInfo);
  return String(buf);
}
