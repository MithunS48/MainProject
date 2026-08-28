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
