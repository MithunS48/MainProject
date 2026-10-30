"""
iot.py
------
IoT water-quality monitoring endpoints.

Device authentication
---------------------
The ESP32 authenticates with an API key sent in the
  X-Device-API-Key: <plain-text key>
header.  The plain-text key is matched against the bcrypt hash stored
in the iot_devices table.  This is intentionally simple so that the
ESP32 firmware only needs to store a single secret string.

Endpoints
---------
POST /api/iot/readings                  -- ESP32 posts a new reading
GET  /api/iot/readings/latest           -- latest reading for caller's device
GET  /api/iot/readings/history          -- paginated reading history
GET  /api/iot/devices/{device_id}       -- device status + thresholds
POST /api/iot/devices/register          -- register a new ESP32 device
GET  /api/iot/devices                   -- list all devices (admin)

Security notes
--------------
- POST /api/iot/readings  uses device API-key auth (no user JWT required)
  so the ESP32 does not need to perform a separate login.
- All read endpoints require a valid user JWT (farmer or admin).
- Admins can see all devices/readings; farmers see only their own devices.
"""

import json
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_admin
from app.models.models import IoTReading, IoTDevice, User
from app.schemas.schemas import (
    IoTReadingIn, IoTReadingOut, IoTDeviceStatus,
    IoTDeviceRegister, PaginatedIoTReadings,
)

router = APIRouter(prefix="/api/iot", tags=["IoT Water Quality"])

# ── configurable ──────────────────────────────────────────────────────────────
ONLINE_THRESHOLD_SECONDS = 300   # device is "online" if reading within 5 min


# ============================================================
# HELPERS
# ============================================================

def _derive(reading: IoTReading, device: Optional[IoTDevice] = None) -> IoTReadingOut:
    """
    Compute scientifically defensible derived values from pH and
    temperature.  Only values that can genuinely be calculated from
    these two measurements are included.
    """
    ph   = reading.ph
    temp = reading.temperature

    # pH status
    if ph < 6.5:
        ph_status = "acidic"
    elif ph > 8.5:
        ph_status = "alkaline"
    else:
        ph_status = "normal"

    # temperature status (optimal tilapia/carp range: 26–30 °C)
    ph_min   = device.ph_min   if device else 6.5
    ph_max   = device.ph_max   if device else 8.5
    temp_min = device.temp_min if device else 20.0
    temp_max = device.temp_max if device else 32.0

    if temp < temp_min:
        temp_status = "too_low"
    elif temp > temp_max:
        temp_status = "too_high"
    else:
        temp_status = "normal"

    # hydrogen-ion concentration  [H⁺] = 10^(-pH)
    h_ion_conc = round(10 ** (-ph), 9)

    # pOH = 14 - pH  (valid for 25 °C; approximate at other temps)
    poh = round(14.0 - ph, 3)

    # alerts
    alerts: list[str] = []
    if ph < ph_min:
        alerts.append(f"pH too low ({ph:.2f} < {ph_min}). Water is acidic.")
    if ph > ph_max:
        alerts.append(f"pH too high ({ph:.2f} > {ph_max}). Water is alkaline.")
    if temp < temp_min:
        alerts.append(f"Temperature too low ({temp:.1f} °C < {temp_min} °C).")
    if temp > temp_max:
        alerts.append(f"Temperature too high ({temp:.1f} °C > {temp_max} °C).")

    return IoTReadingOut(
        id          = reading.id,
        device_id   = reading.device_id,
        ph          = ph,
        temperature = temp,
        received_at = reading.received_at,
        device_ts   = reading.device_ts,
        ph_status   = ph_status,
        temp_status = temp_status,
        h_ion_conc  = h_ion_conc,
        poh         = poh,
        alerts      = alerts,
    )


def _verify_device_key(
    device_id: str,
    plain_key: str,
    db: Session,
) -> IoTDevice:
    """Return the IoTDevice if the API key is valid, else raise 401."""
    device = db.query(IoTDevice).filter(
        IoTDevice.device_id == device_id,
        IoTDevice.is_active == True,
    ).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Device '{device_id}' not registered or inactive.",
        )
    if not bcrypt.checkpw(plain_key.encode(), device.api_key_hash.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid device API key.",
        )
    return device


# ============================================================
# ESP32 → BACKEND: POST A NEW READING
# ============================================================

@router.post(
    "/readings",
    status_code=status.HTTP_201_CREATED,
    summary="ESP32 posts a sensor reading",
)
def post_reading(
    payload: IoTReadingIn,
    x_device_api_key: str = Header(..., description="Plain-text device API key"),
    db: Session = Depends(get_db),
):
    """
    Called by the ESP32 firmware on every reading cycle.

    Authentication: `X-Device-API-Key` header — the plain-text key
    registered via `POST /api/iot/devices/register`.

    The server verifies the key against the stored bcrypt hash.
    """
    device = _verify_device_key(payload.deviceId, x_device_api_key, db)

    # Parse optional device-reported timestamp
    device_ts: Optional[datetime] = None
    if payload.timestamp:
        try:
            device_ts = datetime.fromisoformat(payload.timestamp)
        except ValueError:
            pass  # ignore malformed timestamp, server time will be used

    reading = IoTReading(
        device_id   = payload.deviceId,
        ph          = payload.ph,
        temperature = payload.temperature,
        raw_payload = json.dumps(payload.model_dump()),
        device_ts   = device_ts,
        received_at = datetime.now(timezone.utc),
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    return {
        "status":  "ok",
        "id":      reading.id,
        "message": "Reading stored successfully.",
    }


# ============================================================
# FRONTEND: LATEST READING
# ============================================================

@router.get(
    "/readings/latest",
    response_model=IoTReadingOut,
    summary="Get the most recent reading for a device",
)
def get_latest_reading(
    device_id: str = Query(..., description="ESP32 device ID, e.g. ESP32-001"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the latest reading for the given device.
    Farmers can only query devices they own; admins can query any device.
    """
    device = db.query(IoTDevice).filter(
        IoTDevice.device_id == device_id
    ).first()

    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found.",
        )

    # Farmers may only see their own devices
    if current_user.role != "admin" and device.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this device.",
        )

    reading = (
        db.query(IoTReading)
        .filter(IoTReading.device_id == device_id)
        .order_by(IoTReading.received_at.desc())
        .first()
    )
    if not reading:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No readings found for device '{device_id}'.",
        )
    return _derive(reading, device)


# ============================================================
# FRONTEND: READING HISTORY (PAGINATED)
# ============================================================

@router.get(
    "/readings/history",
    response_model=PaginatedIoTReadings,
    summary="Paginated reading history for a device",
)
def get_reading_history(
    device_id:  str = Query(...),
    page:       int = Query(1, ge=1),
    page_size:  int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns paginated sensor readings, newest first.
    Used by the frontend to render historical trend charts.
    """
    device = db.query(IoTDevice).filter(
        IoTDevice.device_id == device_id
    ).first()
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found.")

    if current_user.role != "admin" and device.owner_id != current_user.id:
        raise HTTPException(403, "You do not have access to this device.")

    base_q = (
        db.query(IoTReading)
        .filter(IoTReading.device_id == device_id)
        .order_by(IoTReading.received_at.desc())
    )
    total  = base_q.count()
    rows   = base_q.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedIoTReadings(
        items     = [_derive(r, device) for r in rows],
        total     = total,
        page      = page,
        page_size = page_size,
    )


# ============================================================
# FRONTEND: DEVICE STATUS
# ============================================================

@router.get(
    "/devices/{device_id}",
    response_model=IoTDeviceStatus,
    summary="Device status and thresholds",
)
def get_device_status(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns device metadata, online/offline status, thresholds,
    and the most recent reading (with derived values and alerts).
    """
    device = db.query(IoTDevice).filter(
        IoTDevice.device_id == device_id
    ).first()
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found.")

    if current_user.role != "admin" and device.owner_id != current_user.id:
        raise HTTPException(403, "You do not have access to this device.")

    latest = (
        db.query(IoTReading)
        .filter(IoTReading.device_id == device_id)
        .order_by(IoTReading.received_at.desc())
        .first()
    )

    now    = datetime.now(timezone.utc)
    online = False
    last_seen = None
    if latest and latest.received_at:
        ts = latest.received_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        last_seen = ts
        online = (now - ts).total_seconds() < ONLINE_THRESHOLD_SECONDS

    return IoTDeviceStatus(
        device_id = device.device_id,
        label     = device.label,
        is_active = device.is_active,
        online    = online,
        last_seen = last_seen,
        ph_min    = device.ph_min,
        ph_max    = device.ph_max,
        temp_min  = device.temp_min,
        temp_max  = device.temp_max,
        latest    = _derive(latest, device) if latest else None,
    )


# ============================================================
# DEVICE REGISTRATION
# ============================================================

@router.post(
    "/devices/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new ESP32 device",
)
def register_device(
    payload: IoTDeviceRegister,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    A farmer (or admin) registers their ESP32 device.
    The plain-text API key is hashed with bcrypt and stored.
    The plain-text key is NEVER stored — it must be saved by the user.
    """
    existing = db.query(IoTDevice).filter(
        IoTDevice.device_id == payload.device_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Device '{payload.device_id}' already registered.",
        )

    hashed = bcrypt.hashpw(
        payload.api_key.encode(), bcrypt.gensalt()
    ).decode()

    device = IoTDevice(
        device_id    = payload.device_id,
        api_key_hash = hashed,
        label        = payload.label,
        owner_id     = current_user.id,
        ph_min       = payload.ph_min,
        ph_max       = payload.ph_max,
        temp_min     = payload.temp_min,
        temp_max     = payload.temp_max,
    )
    db.add(device)
    db.commit()
    db.refresh(device)

    return {
        "status":    "ok",
        "device_id": device.device_id,
        "message":   (
            "Device registered. Save your API key — it cannot be retrieved "
            "after this point. Configure it in your ESP32 firmware as "
            "DEVICE_API_KEY."
        ),
    }


# ============================================================
# MOCK SENSOR ENDPOINTS — for testing without ESP32 hardware
# ============================================================

import random as _random

@router.post(
    "/mock/reading",
    status_code=status.HTTP_201_CREATED,
    summary="Inject a mock sensor reading (no ESP32 required)",
    tags=["Mock"],
)
def inject_mock_reading(
    device_id: str = Query(..., description="Registered device ID"),
    ph: Optional[float] = Query(None, description="Override pH (random if omitted)"),
    temperature: Optional[float] = Query(None, description="Override temperature (random if omitted)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Injects a single fake sensor reading directly into the database.
    Useful for testing the frontend without an ESP32 connected.

    If ph or temperature are omitted, realistic random values are generated
    that occasionally drift outside optimal ranges so you can see alerts.
    """
    device = db.query(IoTDevice).filter(IoTDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found. Register it first.")

    if current_user.role != "admin" and device.owner_id != current_user.id:
        raise HTTPException(403, "You do not have access to this device.")

    # Generate realistic values if not provided
    if ph is None:
        ph = round(_random.uniform(5.5, 9.5), 2)
    if temperature is None:
        temperature = round(_random.uniform(15.0, 35.0), 1)

    reading = IoTReading(
        device_id   = device_id,
        ph          = max(0.0, min(14.0, ph)),
        temperature = temperature,
        received_at = datetime.now(timezone.utc),
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    return {
        "status":     "ok",
        "id":         reading.id,
        "device_id":  device_id,
        "ph":         reading.ph,
        "temperature": reading.temperature,
        "message":    "Mock reading injected. Check the dashboard.",
    }


@router.post(
    "/mock/burst",
    summary="Inject multiple mock readings to populate charts",
    tags=["Mock"],
)
def inject_mock_burst(
    device_id: str = Query(...),
    count: int = Query(20, ge=1, le=100, description="Number of readings to generate"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generates 'count' readings with slight random variations, simulating
    an ESP32 sending data over time. Useful for populating history charts.

    Values slowly drift so you can see trends and occasional alerts.
    """
    device = db.query(IoTDevice).filter(IoTDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found.")

    if current_user.role != "admin" and device.owner_id != current_user.id:
        raise HTTPException(403, "You do not have access to this device.")

    # Start from a realistic baseline and drift
    base_ph   = _random.uniform(6.8, 7.5)
    base_temp = _random.uniform(24.0, 28.0)
    now = datetime.now(timezone.utc)

    for i in range(count):
        # Small random walk with occasional larger drift
        base_ph   += _random.uniform(-0.15, 0.15)
        base_temp += _random.uniform(-0.5, 0.5)

        # Clamp to plausible ranges
        ph_clamped   = max(0.0, min(14.0, round(base_ph, 3)))
        temp_clamped = max(-10.0, min(100.0, round(base_temp, 2)))

        # Spread timestamps over the last 'count' minutes
        ts = now - timedelta(minutes=(count - i))

        reading = IoTReading(
            device_id   = device_id,
            ph          = ph_clamped,
            temperature = temp_clamped,
            received_at = ts,
        )
        db.add(reading)

    db.commit()

    return {
        "status":     "ok",
        "device_id":  device_id,
        "injected":   count,
        "message":    f"{count} mock readings injected. Refresh the dashboard.",
    }


@router.post(
    "/mock/clear",
    summary="Delete all readings for a device (mock data cleanup)",
    tags=["Mock"],
)
def clear_mock_readings(
    device_id: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Removes all sensor readings for the given device.
    The device registration itself is preserved.
    """
    device = db.query(IoTDevice).filter(IoTDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found.")

    if current_user.role != "admin" and device.owner_id != current_user.id:
        raise HTTPException(403, "You do not have access to this device.")

    deleted = db.query(IoTReading).filter(IoTReading.device_id == device_id).delete()
    db.commit()

    return {
        "status":  "ok",
        "deleted": deleted,
        "message": f"Cleared {deleted} readings for '{device_id}'.",
    }


# ============================================================
# ADMIN: LIST ALL DEVICES
# ============================================================

@router.get(
    "/devices",
    summary="List all registered devices (admin only)",
)
def list_devices(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    devices = db.query(IoTDevice).order_by(IoTDevice.created_at.desc()).all()
    now = datetime.now(timezone.utc)
    result = []
    for d in devices:
        latest = (
            db.query(IoTReading)
            .filter(IoTReading.device_id == d.device_id)
            .order_by(IoTReading.received_at.desc())
            .first()
        )
        last_seen = None
        online = False
        if latest and latest.received_at:
            ts = latest.received_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            last_seen = ts
            online = (now - ts).total_seconds() < ONLINE_THRESHOLD_SECONDS

        result.append({
            "device_id":  d.device_id,
            "label":      d.label,
            "is_active":  d.is_active,
            "online":     online,
            "last_seen":  last_seen,
            "owner_id":   d.owner_id,
            "created_at": d.created_at,
            "ph_min":     d.ph_min,
            "ph_max":     d.ph_max,
            "temp_min":   d.temp_min,
            "temp_max":   d.temp_max,
        })
    return result
