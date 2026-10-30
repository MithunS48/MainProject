"""
test_iot.py
-----------
Tests for the IoT water-quality monitoring endpoints.

Run:
    cd webapp/backend
    pytest tests/test_iot.py -v

Coverage:
  - Device registration
  - Device API-key authentication
  - Reading submission (valid, invalid pH, invalid temp)
  - Latest-reading retrieval
  - Reading history (pagination)
  - Device status (online / offline)
  - Water-quality status calculation
  - Alert conditions
  - DB persistence
  - Error handling (404, 403, 401)
"""

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import Base, get_db
from app.models.models import User, IoTDevice, IoTReading, UserRole
from app.core.security import hash_password
from app.routers.iot import _derive, ONLINE_THRESHOLD_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# TEST DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///./test_iot.db"
engine      = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def clean_tables():
    """Wipe IoT tables before every test for isolation."""
    db = TestSession()
    db.query(IoTReading).delete()
    db.query(IoTDevice).delete()
    db.commit()
    db.close()
    yield


@pytest.fixture(scope="module")
def farmer_token():
    """Create a farmer user and return a JWT Bearer token."""
    db = TestSession()
    user = db.query(User).filter(User.email == "testfarmer@aquascan-test.com").first()
    if not user:
        user = User(
            full_name="Test Farmer",
            email="testfarmer@aquascan-test.com",
            hashed_password=hash_password("Test@1234"),
            role=UserRole.farmer,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()

    client = TestClient(app)
    resp = client.post("/api/auth/login", json={
        "email": "testfarmer@aquascan-test.com",
        "password": "Test@1234",
    })
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def farmer_id(farmer_token):
    client = TestClient(app)
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {farmer_token}"})
    return resp.json()["id"]


client = TestClient(app)
PLAIN_KEY = "SecureKey1!"


def _register_device(token, farmer_id, device_id="ESP32-TEST"):
    """Helper: register a device owned by the test farmer."""
    db = TestSession()
    existing = db.query(IoTDevice).filter(IoTDevice.device_id == device_id).first()
    if not existing:
        import bcrypt
        hashed = bcrypt.hashpw(PLAIN_KEY.encode(), bcrypt.gensalt()).decode()
        device = IoTDevice(
            device_id=device_id,
            api_key_hash=hashed,
            label="Test Pond",
            owner_id=farmer_id,
            ph_min=6.5, ph_max=8.5,
            temp_min=20.0, temp_max=32.0,
        )
        db.add(device)
        db.commit()
    db.close()


def _post_reading(device_id="ESP32-TEST", ph=7.0, temp=25.0, api_key=None):
    """Helper: POST a reading directly to the endpoint."""
    return client.post(
        "/api/iot/readings",
        json={
            "deviceId": device_id,
            "ph": ph,
            "temperature": temp,
            "timestamp": "2026-09-02T14:30:00",
        },
        headers={"X-Device-API-Key": api_key or PLAIN_KEY},
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestDeviceRegistration:

    def test_register_device_success(self, farmer_token):
        resp = client.post(
            "/api/iot/devices/register",
            json={
                "device_id": "ESP32-REG",
                "api_key": PLAIN_KEY,
                "label": "Pond A",
                "ph_min": 6.5, "ph_max": 8.5,
                "temp_min": 20.0, "temp_max": 32.0,
            },
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 201
        assert resp.json()["device_id"] == "ESP32-REG"

    def test_register_duplicate_device(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id, "ESP32-DUP")
        resp = client.post(
            "/api/iot/devices/register",
            json={"device_id": "ESP32-DUP", "api_key": PLAIN_KEY},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 409

    def test_register_requires_auth(self):
        resp = client.post(
            "/api/iot/devices/register",
            json={"device_id": "ESP32-NOAUTH", "api_key": PLAIN_KEY},
        )
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# READING SUBMISSION  (ESP32 → backend)
# ─────────────────────────────────────────────────────────────────────────────

class TestReadingSubmission:

    def test_valid_reading_stored(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(ph=7.2, temp=27.5)
        assert resp.status_code == 201
        assert resp.json()["status"] == "ok"

    def test_invalid_api_key_rejected(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(api_key="WRONG_KEY")
        assert resp.status_code == 401

    def test_unregistered_device_rejected(self):
        resp = _post_reading(device_id="GHOST-DEVICE")
        assert resp.status_code == 401

    def test_ph_below_zero_rejected(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(ph=-0.1)
        assert resp.status_code == 422

    def test_ph_above_14_rejected(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(ph=14.1)
        assert resp.status_code == 422

    def test_temperature_below_minus_10_rejected(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(temp=-10.1)
        assert resp.status_code == 422

    def test_temperature_above_100_rejected(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(temp=100.1)
        assert resp.status_code == 422

    def test_boundary_ph_zero_accepted(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        assert _post_reading(ph=0.0).status_code == 201

    def test_boundary_ph_14_accepted(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        assert _post_reading(ph=14.0).status_code == 201


# ─────────────────────────────────────────────────────────────────────────────
# DB PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabasePersistence:

    def test_reading_persisted_to_db(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = _post_reading(ph=6.9, temp=26.3)
        assert resp.status_code == 201
        reading_id = resp.json()["id"]

        db = TestSession()
        row = db.query(IoTReading).filter(IoTReading.id == reading_id).first()
        db.close()

        assert row is not None
        assert abs(row.ph - 6.9) < 0.01
        assert abs(row.temperature - 26.3) < 0.01

    def test_device_id_stored_correctly(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        _post_reading(ph=7.0, temp=25.0)

        db = TestSession()
        row = db.query(IoTReading).filter(
            IoTReading.device_id == "ESP32-TEST"
        ).first()
        db.close()
        assert row is not None
        assert row.device_id == "ESP32-TEST"


# ─────────────────────────────────────────────────────────────────────────────
# LATEST READING
# ─────────────────────────────────────────────────────────────────────────────

class TestLatestReading:

    def test_latest_reading_returned(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        _post_reading(ph=7.1, temp=25.0)
        _post_reading(ph=7.5, temp=26.0)   # this should be latest

        resp = client.get(
            "/api/iot/readings/latest",
            params={"device_id": "ESP32-TEST"},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["ph"] - 7.5) < 0.01

    def test_latest_includes_derived_values(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        _post_reading(ph=7.0, temp=25.0)

        resp = client.get(
            "/api/iot/readings/latest",
            params={"device_id": "ESP32-TEST"},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        data = resp.json()
        assert "ph_status" in data
        assert "temp_status" in data
        assert "h_ion_conc" in data
        assert "poh" in data
        assert "alerts" in data

    def test_latest_404_no_readings(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = client.get(
            "/api/iot/readings/latest",
            params={"device_id": "ESP32-TEST"},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 404

    def test_latest_requires_auth(self, farmer_id, farmer_token):
        _register_device(farmer_token, farmer_id)
        resp = client.get(
            "/api/iot/readings/latest",
            params={"device_id": "ESP32-TEST"},
        )
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# READING HISTORY
# ─────────────────────────────────────────────────────────────────────────────

class TestReadingHistory:

    def test_history_returns_all_readings(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        for i in range(5):
            _post_reading(ph=6.5 + i * 0.2, temp=25.0 + i)

        resp = client.get(
            "/api/iot/readings/history",
            params={"device_id": "ESP32-TEST", "page": 1, "page_size": 10},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5

    def test_history_pagination(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        for i in range(10):
            _post_reading(ph=7.0, temp=25.0)

        resp = client.get(
            "/api/iot/readings/history",
            params={"device_id": "ESP32-TEST", "page": 1, "page_size": 3},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["items"]) == 3
        assert data["page"] == 1
        assert data["page_size"] == 3

    def test_history_newest_first(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        _post_reading(ph=6.0, temp=20.0)
        _post_reading(ph=8.0, temp=30.0)  # posted later → should appear first

        resp = client.get(
            "/api/iot/readings/history",
            params={"device_id": "ESP32-TEST"},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        items = resp.json()["items"]
        assert abs(items[0]["ph"] - 8.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE STATUS
# ─────────────────────────────────────────────────────────────────────────────

class TestDeviceStatus:

    def test_device_status_returned(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = client.get(
            "/api/iot/devices/ESP32-TEST",
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == "ESP32-TEST"
        assert "online" in data
        assert "ph_min" in data
        assert "ph_max" in data

    def test_device_offline_when_no_readings(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        resp = client.get(
            "/api/iot/devices/ESP32-TEST",
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.json()["online"] is False

    def test_device_online_after_recent_reading(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        _post_reading(ph=7.0, temp=25.0)

        resp = client.get(
            "/api/iot/devices/ESP32-TEST",
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.json()["online"] is True

    def test_device_404_not_found(self, farmer_token):
        resp = client.get(
            "/api/iot/devices/NONEXISTENT",
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# WATER-QUALITY STATUS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

class TestWaterQualityCalculation:
    """Unit tests for _derive() — no HTTP involved."""

    def _make_reading(self, ph, temp):
        r = IoTReading(
            id="test-id",
            device_id="ESP32-TEST",
            ph=ph,
            temperature=temp,
            received_at=datetime.now(timezone.utc),
        )
        return r

    def test_normal_ph_status(self):
        r = self._make_reading(7.0, 25.0)
        out = _derive(r)
        assert out.ph_status == "normal"

    def test_acidic_ph_status(self):
        r = self._make_reading(5.0, 25.0)
        out = _derive(r)
        assert out.ph_status == "acidic"

    def test_alkaline_ph_status(self):
        r = self._make_reading(9.0, 25.0)
        out = _derive(r)
        assert out.ph_status == "alkaline"

    def test_normal_temp_status(self):
        r = self._make_reading(7.0, 26.0)
        out = _derive(r)
        assert out.temp_status == "normal"

    def test_temp_too_low(self):
        r = self._make_reading(7.0, 10.0)
        out = _derive(r)
        assert out.temp_status == "too_low"

    def test_temp_too_high(self):
        r = self._make_reading(7.0, 40.0)
        out = _derive(r)
        assert out.temp_status == "too_high"

    def test_h_ion_concentration_ph7(self):
        r = self._make_reading(7.0, 25.0)
        out = _derive(r)
        assert abs(out.h_ion_conc - 1e-7) < 1e-10

    def test_poh_at_ph7(self):
        r = self._make_reading(7.0, 25.0)
        out = _derive(r)
        assert abs(out.poh - 7.0) < 0.01

    def test_poh_at_ph4(self):
        r = self._make_reading(4.0, 25.0)
        out = _derive(r)
        assert abs(out.poh - 10.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# ALERT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestAlerts:

    def _make_reading(self, ph, temp):
        return IoTReading(
            id="alert-test",
            device_id="ESP32-TEST",
            ph=ph,
            temperature=temp,
            received_at=datetime.now(timezone.utc),
        )

    def _device(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            ph_min=6.5, ph_max=8.5,
            temp_min=20.0, temp_max=32.0
        )

    def test_no_alerts_in_normal_range(self):
        r = self._make_reading(7.0, 26.0)
        out = _derive(r, self._device())
        assert out.alerts == []

    def test_alert_ph_too_low(self):
        r = self._make_reading(5.0, 26.0)
        out = _derive(r, self._device())
        assert any("too low" in a.lower() for a in out.alerts)

    def test_alert_ph_too_high(self):
        r = self._make_reading(9.5, 26.0)
        out = _derive(r, self._device())
        assert any("too high" in a.lower() for a in out.alerts)

    def test_alert_temp_too_low(self):
        r = self._make_reading(7.0, 15.0)
        out = _derive(r, self._device())
        assert any("too low" in a.lower() for a in out.alerts)

    def test_alert_temp_too_high(self):
        r = self._make_reading(7.0, 38.0)
        out = _derive(r, self._device())
        assert any("too high" in a.lower() for a in out.alerts)

    def test_multiple_alerts_combined(self):
        r = self._make_reading(4.0, 40.0)
        out = _derive(r, self._device())
        assert len(out.alerts) == 2  # pH too low + temp too high

    def test_alert_in_latest_api_response(self, farmer_token, farmer_id):
        _register_device(farmer_token, farmer_id)
        _post_reading(ph=5.0, temp=25.0)  # acidic — should trigger alert

        resp = client.get(
            "/api/iot/readings/latest",
            params={"device_id": "ESP32-TEST"},
            headers={"Authorization": f"Bearer {farmer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) > 0

