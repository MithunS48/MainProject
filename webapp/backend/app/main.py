"""
main.py
-------
Fish Disease Detection & Classification — Web Application Backend.

This is a NEW backend layer built on top of the EXISTING, untouched ML
project (src/ + results/). It exposes:

  - JWT authentication with role-based authorization (farmer / admin)
  - Admin-registration protection (invite code required)
  - Real ML inference via the actual trained MobileNetV2 + ConvNeXt +
    Polynomial SVM pipeline (see app/core/inference.py)
  - Prediction history storage (SQLite)
  - Admin analytics / user management / prediction management
  - Read-only endpoints serving the project's real, already-computed
    research results (confusion matrix, ROC/AUC, fusion experiments,
    PCA comparison, SVM kernel comparison, error analysis, etc.)

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    (from webapp/backend/)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.database import Base, engine, SessionLocal
from app.core.config import STATIC_DIR, ADMIN_INVITE_CODE
from app.core.security import hash_password
from app.models.models import User, UserRole
from app.routers import auth, predict, users, admin, research

# ------------------------------------------------------------
# Create DB tables
# ------------------------------------------------------------
Base.metadata.create_all(bind=engine)


def _seed_default_admin():
    """Create a default admin account on first run, for demo purposes.
    Credentials are printed once to server logs. In production, change
    the password immediately and rotate ADMIN_INVITE_CODE."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.role == UserRole.admin).first()
        if existing:
            return
        admin_user = User(
            full_name="System Administrator",
            email="admin@aquascan.ai",
            hashed_password=hash_password("Admin@123"),
            role=UserRole.admin,
        )
        db.add(admin_user)
        db.commit()
        print("\n" + "=" * 60)
        print("  Default ADMIN account created:")
        print("    email    : admin@aquascan.ai")
        print("    password : Admin@123")
        print("  Please change this password after first login.")
        print(f"  Admin self-registration invite code: {ADMIN_INVITE_CODE}")
        print("=" * 60 + "\n")
    finally:
        db.close()


_seed_default_admin()

# ------------------------------------------------------------
# App
# ------------------------------------------------------------

app = FastAPI(
    title="AquaScan — Fish Disease Detection API",
    description=(
        "REST API for the AI-Based Fish Disease Detection and "
        "Classification System. Final model: MobileNetV2 + ConvNeXt "
        "feature fusion -> Polynomial SVM. Test accuracy: 98.29%."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(users.router)
app.include_router(admin.router)
app.include_router(research.router)


@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "AquaScan Fish Disease Detection API",
        "version": "1.0.0",
    }


@app.on_event("startup")
async def startup_event():
    print("\nPreloading ML models (MobileNetV2 + ConvNeXt + SVM)...")
    try:
        from app.core.inference import preload_models
        preload_models()
    except Exception as e:
        print(f"Warning: could not preload models at startup: {e}")
