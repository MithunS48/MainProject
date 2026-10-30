"""
models.py
---------
SQLAlchemy ORM models for the web application layer:
  - User        (farmer / admin accounts)
  - Prediction  (stored inference results / history)
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, Float, Enum, Integer, Text
)
from sqlalchemy.orm import relationship

from app.core.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


class UserRole(str, enum.Enum):
    farmer = "farmer"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.farmer)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    predictions = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    image_path = Column(String, nullable=False)   # relative static path
    original_filename = Column(String, nullable=True)

    predicted_class = Column(String, nullable=False)     # EUS/gill/healthy/red_spot
    predicted_label = Column(String, nullable=False)     # human readable label
    confidence = Column(Float, nullable=False)           # 0..1
    confidence_pct = Column(Float, nullable=False)       # 0..100

    prob_eus = Column(Float, nullable=True)
    prob_gill = Column(Float, nullable=True)
    prob_healthy = Column(Float, nullable=True)
    prob_red_spot = Column(Float, nullable=True)

    status = Column(String, nullable=False, default="success")  # success / error
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="predictions")


# ============================================================
# IoT — WATER QUALITY MONITORING
# ============================================================

class IoTReading(Base):
    """
    Stores a single sensor reading posted by an ESP32 device.

    Additional sensors (turbidity, TDS/EC, DO) can be added here
    later as nullable columns without breaking existing readings.
    """
    __tablename__ = "iot_readings"

    id           = Column(String,  primary_key=True, default=gen_uuid)
    device_id    = Column(String,  nullable=False, index=True)   # e.g. "ESP32-001"
    api_key_hash = Column(String,  nullable=True)                # hashed device API key
    ph           = Column(Float,   nullable=False)               # 0 – 14
    temperature  = Column(Float,   nullable=False)               # °C
    # ── extensibility placeholders (nullable) ──────────────────
    # turbidity    = Column(Float, nullable=True)   # NTU — add when sensor present
    # tds          = Column(Float, nullable=True)   # ppm — add when sensor present
    # dissolved_o2 = Column(Float, nullable=True)   # mg/L — add when sensor present
    raw_payload  = Column(Text,    nullable=True)                # full JSON from device
    received_at  = Column(DateTime, default=utcnow, index=True)  # server-side timestamp
    device_ts    = Column(DateTime, nullable=True)               # device-reported timestamp


class IoTDevice(Base):
    """
    Registered ESP32 devices.  A device must be registered before it
    can POST readings (checked via hashed API key).

    A single device can be shared across multiple farm users, or each
    farmer can register their own device.
    """
    __tablename__ = "iot_devices"

    id           = Column(String,  primary_key=True, default=gen_uuid)
    device_id    = Column(String,  unique=True, nullable=False, index=True)
    api_key_hash = Column(String,  nullable=False)    # bcrypt hash of the plain API key
    label        = Column(String,  nullable=True)     # human-friendly name
    owner_id     = Column(String,  ForeignKey("users.id"), nullable=True, index=True)
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=utcnow)

    # thresholds (configurable per device)
    ph_min       = Column(Float, default=6.5)
    ph_max       = Column(Float, default=8.5)
    temp_min     = Column(Float, default=20.0)
    temp_max     = Column(Float, default=32.0)

    owner = relationship("User", backref="iot_devices")
