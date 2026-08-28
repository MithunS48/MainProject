"""
schemas.py
----------
Pydantic request/response schemas.
Passwords / hashed_password are never included in any response schema.
"""

from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, EmailStr, field_validator


# ============================================================
# AUTH
# ============================================================

class UserRegister(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    role: str = "farmer"  # "farmer" or "admin"
    admin_invite_code: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number.")
        return v

    @field_validator("role")
    @classmethod
    def role_valid(cls, v):
        if v not in ("farmer", "admin"):
            raise ValueError("Role must be 'farmer' or 'admin'.")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserPublic"


class UserPublic(BaseModel):
    id: str
    full_name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    analyses_count: int = 0

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


# ============================================================
# ADMIN — USER MANAGEMENT
# ============================================================

class AdminUserUpdate(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = None


# ============================================================
# PREDICTIONS
# ============================================================

class PredictionOut(BaseModel):
    id: str
    user_id: str
    image_url: str
    original_filename: Optional[str]
    predicted_class: str
    predicted_label: str
    confidence: float
    confidence_pct: float
    all_scores: Dict[str, float]
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionAdminOut(PredictionOut):
    user_name: str
    user_email: str


class PaginatedPredictions(BaseModel):
    items: List[PredictionOut]
    total: int
    page: int
    page_size: int


class PaginatedAdminPredictions(BaseModel):
    items: List[PredictionAdminOut]
    total: int
    page: int
    page_size: int


# ============================================================
# ADMIN — ANALYTICS
# ============================================================

class AdminOverview(BaseModel):
    total_users: int
    total_farmers: int
    total_admins: int
    total_predictions: int
    healthy_predictions: int
    disease_predictions: int
    most_detected_disease: Optional[str]
    model_accuracy: float
    model_f1: float
    model_auc: float


class DiseaseDistributionItem(BaseModel):
    disease: str
    count: int


Token.model_rebuild()
