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

# Admin registration protection: a secret invite code required to create
# an Admin account through the public /auth/register endpoint.
# In production, set this via environment variable and share only with
# trusted staff. Default kept here for local/demo use only.
ADMIN_INVITE_CODE = os.environ.get("FISHAPP_ADMIN_INVITE_CODE", "AQUASCAN-ADMIN-2026")

# Upload constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# Real reported final-model metrics (kept in sync with results/final/reports/final_metrics.json)
FINAL_MODEL_ACCURACY = 0.9829
FINAL_MODEL_F1 = 0.9829
FINAL_MODEL_AUC = 0.9989

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
