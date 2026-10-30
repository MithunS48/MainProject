"""
config.py
---------
Central configuration for the Fish Disease Detection backend.

Paths point back into the ORIGINAL, untouched ML project:
  PROJECT_ROOT/src          -> existing training/inference scripts (untouched)
  PROJECT_ROOT/results      -> existing trained models & reports (untouched)

This backend is a NEW layer added on top of the existing project;
none of the original files are modified.
"""

import os
from pathlib import Path

# webapp/backend/app/core/config.py -> go up to project root
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
WEBAPP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent  # /home/user/webapp

SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

STATIC_DIR = BACKEND_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESEARCH_DATA_PATH = STATIC_DIR / "research_data" / "research_data.json"

DB_PATH = BACKEND_DIR / "app.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# JWT settings
SECRET_KEY = os.environ.get("FISHAPP_SECRET_KEY", "dev-secret-key-change-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Admin registration protection
ADMIN_INVITE_CODE = os.environ.get("FISHAPP_ADMIN_INVITE_CODE", "AQUASCAN-ADMIN-2026")

# Upload constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# Real reported final-model metrics
FINAL_MODEL_ACCURACY = 0.9829
FINAL_MODEL_F1       = 0.9829
FINAL_MODEL_AUC      = 0.9989

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Email (SMTP) ──────────────────────────────────────────────────────────────
# Set these environment variables to enable email alerts.
# Example (Gmail):
#   FISHAPP_SMTP_HOST=smtp.gmail.com
#   FISHAPP_SMTP_PORT=587
#   FISHAPP_SMTP_USER=you@gmail.com
#   FISHAPP_SMTP_PASSWORD=your-app-password   (use Gmail App Password, not account password)
#   FISHAPP_SMTP_FROM=AquaScan <you@gmail.com>
SMTP_HOST     = os.environ.get("FISHAPP_SMTP_HOST", "")
SMTP_PORT     = int(os.environ.get("FISHAPP_SMTP_PORT", "587"))
SMTP_USER     = os.environ.get("FISHAPP_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("FISHAPP_SMTP_PASSWORD", "")
SMTP_FROM     = os.environ.get("FISHAPP_SMTP_FROM", SMTP_USER)
SMTP_ENABLED  = bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)

# ── Twilio SMS ────────────────────────────────────────────────────────────────
# Set these to enable SMS alerts:
#   FISHAPP_TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxx
#   FISHAPP_TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxx
#   FISHAPP_TWILIO_FROM_NUMBER=+1234567890   (your Twilio phone number)
TWILIO_ACCOUNT_SID  = os.environ.get("FISHAPP_TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.environ.get("FISHAPP_TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER  = os.environ.get("FISHAPP_TWILIO_FROM_NUMBER", "")
TWILIO_ENABLED      = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER)
