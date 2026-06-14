"""
AquaScan — Fish Disease Detection API
FastAPI server with:
  - Single & batch image prediction + Grad-CAM
  - API Key authentication (X-API-Key header)
  - Rate limiting (per IP, token-bucket style)
  - User authentication (PIN-based, JWT tokens)

Environment variables:
  MODEL_PATH      — path to .keras model   (default: model/fish_disease_model.keras)
  API_KEYS        — comma-separated valid API keys (default: aquascan-dev-key)
  JWT_SECRET      — secret for signing JWTs (default: change-me-in-production)
  RATE_LIMIT_RPM  — requests per minute per IP (default: 30)
  ADMIN_PIN       — admin PIN for user management (default: 1234)

Start: uvicorn app:app --reload
"""

import io
import logging
import os
import sys
import base64
import time
import secrets
import hashlib
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config from environment
# ──────────────────────────────────────────────
MODEL_PATH     = os.environ.get("MODEL_PATH",     "model/fish_disease_model.keras")
JWT_SECRET     = os.environ.get("JWT_SECRET",     "change-me-in-production")
ADMIN_PIN      = os.environ.get("ADMIN_PIN",      "1234")
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "30"))

# API keys: comma-separated list or env var
_raw_keys = os.environ.get("API_KEYS", "aquascan-dev-key")
VALID_API_KEYS: set = {k.strip() for k in _raw_keys.split(",") if k.strip()}

CLASS_LABELS        = {0: "eus", 1: "gill", 2: "healthy", 3: "red_spot"}
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ──────────────────────────────────────────────
# In-memory stores (replace with DB for production)
# ──────────────────────────────────────────────

# Users: {username: {"pin_hash": str, "role": str, "email": str, ...}}
USERS: Dict[str, dict] = {
    "admin": {
        "pin_hash":  hashlib.sha256(ADMIN_PIN.encode()).hexdigest(),
        "role":      "admin",
        "email":     "",
        "full_name": "Administrator",
        "created_at": time.time(),
    }
}

# Per-user scan history: {username: [{"predicted_class": str, "confidence": float, ...}]}
USER_SCANS: Dict[str, list] = defaultdict(list)

# Active JWT tokens: {token: {"username": str, "expires": float}}
ACTIVE_TOKENS: Dict[str, dict] = {}

# Rate limiter: {ip: {"count": int, "window_start": float}}
RATE_STORE: Dict[str, dict] = defaultdict(lambda: {"count": 0, "window_start": time.time()})

# ──────────────────────────────────────────────
# JWT helpers (simple HMAC-based, no dependency)
# ──────────────────────────────────────────────
import hmac
import json

TOKEN_TTL = 3600  # 1 hour


def _sign(payload: dict) -> str:
    """Create a simple signed token: base64(payload).base64(signature)"""
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig  = hmac.new(JWT_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _verify(token: str) -> Optional[dict]:
    """Verify token signature and expiry. Returns payload or None."""
    try:
        body, sig = token.rsplit(".", 1)
        expected  = hmac.new(JWT_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=="))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def create_token(username: str, role: str) -> str:
    payload = {
        "sub":  username,
        "role": role,
        "iat":  time.time(),
        "exp":  time.time() + TOKEN_TTL,
    }
    return _sign(payload)


# ──────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────
def check_rate_limit(ip: str):
    """Token-bucket rate limiter. Raises 429 if over limit."""
    now    = time.time()
    bucket = RATE_STORE[ip]
    # Reset window every 60 seconds
    if now - bucket["window_start"] >= 60:
        bucket["count"]        = 0
        bucket["window_start"] = now

    bucket["count"] += 1
    if bucket["count"] > RATE_LIMIT_RPM:
        logger.warning("Rate limit exceeded for IP %s (%d req/min)", ip, bucket["count"])
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_RPM} requests per minute.",
            headers={"Retry-After": "60"}
        )


# ──────────────────────────────────────────────
# API Key dependency
# ──────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Require a valid API key in the X-API-Key header.
    Set API_KEYS env var to a comma-separated list of valid keys.
    Default dev key: aquascan-dev-key
    """
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        logger.warning("Invalid or missing API key: %s", x_api_key)
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide a valid X-API-Key header."
        )
    return x_api_key


# ──────────────────────────────────────────────
# JWT Bearer dependency
# ──────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> dict:
    """Extract and validate JWT from Authorization: Bearer <token> header."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    payload = _verify(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")
    return {"username": payload["sub"], "role": payload.get("role", "user")}


def get_admin_user(user: dict = Depends(get_current_user)) -> dict:
    """Require admin role."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return user


# ──────────────────────────────────────────────
# Lifespan — model loading
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        logger.error("TensorFlow not installed: %s", exc); sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at '%s'. Train first or set MODEL_PATH.", MODEL_PATH)
        sys.exit(1)

    logger.info("Loading model from '%s'…", MODEL_PATH)
    try:
        app.state.model = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded. API keys loaded: %d key(s).", len(VALID_API_KEYS))
        logger.info("Rate limit: %d req/min per IP.", RATE_LIMIT_RPM)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc); sys.exit(1)

    yield
    logger.info("Shutting down.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="AquaScan — Fish Disease Detection API",
    description=(
        "Classifies fish images into disease categories using a CNN.\n\n"
        "**Authentication:** All `/predict` endpoints require:\n"
        "1. `X-API-Key` header with a valid API key\n"
        "2. `Authorization: Bearer <token>` (obtain token from `/auth/login`)\n\n"
        "**Rate Limit:** "  + str(RATE_LIMIT_RPM) + " requests per minute per IP."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Middleware — log every request
# ──────────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(
            "%s %s — %d (%.1fms) — IP: %s",
            request.method, request.url.path,
            response.status_code, duration, client_ip
        )
        return response

app.add_middleware(RequestLogMiddleware)


# ──────────────────────────────────────────────
# Core ML helpers
# ──────────────────────────────────────────────
def validate_file_format(filename: str) -> bool:
    if not filename: return False
    parts = filename.rsplit(".", 1)
    return len(parts) == 2 and parts[-1].lower() in SUPPORTED_EXTENSIONS


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def run_inference(image_bytes: bytes, model):
    tensor      = preprocess_image(image_bytes)
    predictions = model.predict(tensor, verbose=0)
    class_index = int(np.argmax(predictions[0]))
    confidence  = float(predictions[0][class_index])
    all_probs   = {CLASS_LABELS[i]: float(predictions[0][i]) for i in range(4)}
    return class_index, confidence, all_probs


def severity_from_confidence(confidence: float, predicted_class: str) -> str:
    if predicted_class == "healthy": return "none"
    if confidence >= 0.85: return "severe"
    elif confidence >= 0.65: return "moderate"
    return "mild"


def compute_gradcam(model, image_bytes: bytes, class_index: int) -> str:
    import tensorflow as tf
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer; break
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv_layer = sub; break
        if last_conv_layer: break
    if last_conv_layer is None: return ""
    try:
        import cv2
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
        img_tensor = tf.cast(preprocess_image(image_bytes), tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_index]
        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()
        heatmap_r    = cv2.resize(heatmap, (224, 224))
        colored      = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
        orig         = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224)))
        overlay      = cv2.addWeighted(orig, 0.6, colored[:, :, ::-1], 0.4, 0)
        buf          = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc); return ""


# ══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS — no API key required
# ══════════════════════════════════════════════════════════════

@app.post(
    "/auth/signup",
    summary="Public self-registration (anyone can sign up)",
    tags=["Authentication"]
)
async def signup(request: Request):
    """
    Public signup — anyone can create a regular user account.
    No authentication required.

    Body: `{"username": "john", "pin": "5678", "email": "john@example.com"}`
    - username: 3–30 chars, alphanumeric + underscore only
    - pin: minimum 4 characters
    - email: optional
    """
    check_rate_limit(request.client.host if request.client else "unknown")

    body     = await request.json()
    username = body.get("username", "").strip().lower()
    pin      = body.get("pin", "").strip()
    email    = body.get("email", "").strip()
    full_name= body.get("full_name", "").strip()

    # Validation
    import re
    if not username or not pin:
        raise HTTPException(status_code=400, detail="username and pin are required.")
    if not re.match(r'^[a-z0-9_]{3,30}$', username):
        raise HTTPException(status_code=400, detail="Username must be 3–30 characters: letters, numbers, underscore only.")
    if len(pin) < 4:
        raise HTTPException(status_code=400, detail="PIN must be at least 4 characters.")
    if username in USERS:
        raise HTTPException(status_code=409, detail=f"Username '{username}' is already taken.")

    USERS[username] = {
        "pin_hash":  hashlib.sha256(pin.encode()).hexdigest(),
        "role":      "user",
        "email":     email,
        "full_name": full_name,
        "created_at": time.time(),
    }

    token = create_token(username, "user")
    logger.info("New user signed up: '%s' (email: %s)", username, email or "—")
    return {
        "message":      f"Account created successfully. Welcome, {username}!",
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   TOKEN_TTL,
        "username":     username,
        "role":         "user"
    }


@app.post(
    "/auth/login",
    summary="Login with username + PIN",
    tags=["Authentication"]
)
async def login(request: Request):
    """
    Authenticate with username and PIN.
    Returns a JWT bearer token valid for 1 hour.

    Body: `{"username": "admin", "pin": "1234"}`
    """
    check_rate_limit(request.client.host if request.client else "unknown")

    body = await request.json()
    username = body.get("username", "").strip()
    pin      = body.get("pin", "").strip()

    if not username or not pin:
        raise HTTPException(status_code=400, detail="username and pin are required.")

    user = USERS.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or PIN.")

    pin_hash = hashlib.sha256(pin.encode()).hexdigest()
    if not hmac.compare_digest(pin_hash, user["pin_hash"]):
        logger.warning("Failed login attempt for user '%s' from %s", username, request.client.host if request.client else "?")
        raise HTTPException(status_code=401, detail="Invalid username or PIN.")

    token = create_token(username, user["role"])
    logger.info("User '%s' logged in successfully.", username)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   TOKEN_TTL,
        "username":     username,
        "role":         user["role"]
    }


@app.post(
    "/auth/register",
    summary="Register a new user (admin only)",
    tags=["Authentication"]
)
async def register(
    request: Request,
    admin: dict = Depends(get_admin_user)
):
    """Register a new user. Requires admin JWT."""
    body     = await request.json()
    username = body.get("username", "").strip()
    pin      = body.get("pin", "").strip()
    role     = body.get("role", "user").strip()

    if not username or not pin:
        raise HTTPException(status_code=400, detail="username and pin are required.")
    if len(pin) < 4:
        raise HTTPException(status_code=400, detail="PIN must be at least 4 characters.")
    if username in USERS:
        raise HTTPException(status_code=409, detail=f"User '{username}' already exists.")
    if role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")

    USERS[username] = {
        "pin_hash": hashlib.sha256(pin.encode()).hexdigest(),
        "role": role
    }
    logger.info("Admin '%s' registered new user '%s' (role: %s).", admin["username"], username, role)
    return {"message": f"User '{username}' registered successfully.", "role": role}


@app.get(
    "/auth/me",
    summary="Get current user info",
    tags=["Authentication"]
)
async def me(user: dict = Depends(get_current_user)):
    """Returns info about the currently authenticated user."""
    return {"username": user["username"], "role": user["role"]}


@app.get(
    "/auth/users",
    summary="List all users (admin only)",
    tags=["Authentication"]
)
async def list_users(admin: dict = Depends(get_admin_user)):
    """Returns list of all registered users. Admin only."""
    return {"users": [{"username": u, "role": v["role"]} for u, v in USERS.items()]}


# ══════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS — require API key + JWT
# ══════════════════════════════════════════════════════════════

@app.post(
    "/predict",
    summary="Single image prediction",
    tags=["Prediction"]
)
async def predict(
    request: Request,
    file: UploadFile = File(None),
    _api_key: str   = Depends(verify_api_key),
    _user:    dict  = Depends(get_current_user),
):
    """
    Classify a single fish image.
    Returns predicted class, confidence, severity, Grad-CAM heatmap,
    and all class probabilities.

    **Requires:** X-API-Key header + Authorization: Bearer <token>
    """
    check_rate_limit(request.client.host if request.client else "unknown")

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")
    if not validate_file_format(file.filename):
        raise HTTPException(status_code=422, detail="Unsupported format. Accepted: JPEG, JPG, PNG.")

    image_bytes = await file.read()
    try:
        class_index, confidence, all_probs = run_inference(image_bytes, app.state.model)
        predicted_class = CLASS_LABELS[class_index]
        severity        = severity_from_confidence(confidence, predicted_class)
        heatmap_b64     = compute_gradcam(app.state.model, image_bytes, class_index)
        logger.info("Predict: user=%s class=%s conf=%.2f", _user["username"], predicted_class, confidence)

        # Save scan to per-user history
        USER_SCANS[_user["username"]].insert(0, {
            "predicted_class":   predicted_class,
            "confidence":        confidence,
            "severity":          severity,
            "timestamp":         time.time(),
            "filename":          file.filename or "",
        })

        return {
            "predicted_class":   predicted_class,
            "confidence":        confidence,
            "all_probabilities": all_probs,
            "severity":          severity,
            "gradcam":           heatmap_b64,
        }
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.post(
    "/predict-batch",
    summary="Batch image prediction (up to 20 images)",
    tags=["Prediction"]
)
async def predict_batch(
    request: Request,
    files:     List[UploadFile] = File(...),
    _api_key:  str  = Depends(verify_api_key),
    _user:     dict = Depends(get_current_user),
):
    """
    Classify up to 20 fish images in one request.

    **Requires:** X-API-Key header + Authorization: Bearer <token>
    """
    check_rate_limit(request.client.host if request.client else "unknown")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch.")

    results = []
    for f in files:
        if not validate_file_format(f.filename or ""):
            results.append({"filename": f.filename, "error": "Unsupported format",
                             "predicted_class": None, "confidence": None, "severity": None})
            continue
        try:
            image_bytes     = await f.read()
            class_index, confidence, all_probs = run_inference(image_bytes, app.state.model)
            predicted_class = CLASS_LABELS[class_index]
            severity        = severity_from_confidence(confidence, predicted_class)
            results.append({"filename": f.filename, "predicted_class": predicted_class,
                             "confidence": confidence, "all_probabilities": all_probs,
                             "severity": severity, "error": None})
        except Exception as exc:
            results.append({"filename": f.filename, "predicted_class": None,
                             "confidence": None, "severity": None, "error": str(exc)})

    healthy_count  = sum(1 for r in results if r["predicted_class"] == "healthy")
    diseased_count = sum(1 for r in results if r["predicted_class"] and r["predicted_class"] != "healthy")
    logger.info("Batch predict: user=%s total=%d healthy=%d diseased=%d",
                _user["username"], len(results), healthy_count, diseased_count)

    # Save batch scans to per-user history
    for r in results:
        if r.get("predicted_class"):
            USER_SCANS[_user["username"]].insert(0, {
                "predicted_class": r["predicted_class"],
                "confidence":      r["confidence"],
                "severity":        r.get("severity", "none"),
                "timestamp":       time.time(),
                "filename":        r.get("filename", ""),
            })

    return {"total": len(results), "healthy_count": healthy_count,
            "diseased_count": diseased_count, "results": results}


# ══════════════════════════════════════════════════════════════
# USER SCAN HISTORY — authenticated users see only their own
# ══════════════════════════════════════════════════════════════

@app.get("/scans/mine", summary="Get my scan history", tags=["Scans"])
async def get_my_scans(
    _api_key: str  = Depends(verify_api_key),
    user:     dict = Depends(get_current_user),
):
    """Returns the authenticated user's own scan history only."""
    scans = USER_SCANS.get(user["username"], [])
    return {
        "username": user["username"],
        "total":    len(scans),
        "scans":    scans
    }


@app.delete("/scans/mine", summary="Clear my scan history", tags=["Scans"])
async def clear_my_scans(
    _api_key: str  = Depends(verify_api_key),
    user:     dict = Depends(get_current_user),
):
    """Clears the authenticated user's scan history."""
    USER_SCANS[user["username"]] = []
    logger.info("User '%s' cleared their scan history.", user["username"])
    return {"message": "Your scan history has been cleared."}


# ══════════════════════════════════════════════════════════════
# ADMIN PANEL — view all users, profiles, and their scans
# ══════════════════════════════════════════════════════════════

@app.get("/admin/users", summary="List all users with stats (admin only)", tags=["Admin"])
async def admin_list_users(
    _api_key: str  = Depends(verify_api_key),
    admin:    dict = Depends(get_admin_user),
):
    """Returns all registered users with their profile and scan statistics. Admin only."""
    result = []
    for username, profile in USERS.items():
        scans = USER_SCANS.get(username, [])
        healthy  = sum(1 for s in scans if s["predicted_class"] == "healthy")
        diseased = len(scans) - healthy
        result.append({
            "username":    username,
            "full_name":   profile.get("full_name", ""),
            "email":       profile.get("email", ""),
            "role":        profile.get("role", "user"),
            "created_at":  profile.get("created_at", 0),
            "total_scans": len(scans),
            "healthy":     healthy,
            "diseased":    diseased,
        })
    return {"total_users": len(result), "users": result}


@app.get("/admin/users/{username}/scans", summary="Get a user's scan history (admin only)", tags=["Admin"])
async def admin_get_user_scans(
    username: str,
    _api_key: str  = Depends(verify_api_key),
    admin:    dict = Depends(get_admin_user),
):
    """Returns full scan history for a specific user. Admin only."""
    if username not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    profile = USERS[username]
    scans   = USER_SCANS.get(username, [])
    return {
        "username":  username,
        "full_name": profile.get("full_name", ""),
        "email":     profile.get("email", ""),
        "role":      profile.get("role", "user"),
        "total_scans": len(scans),
        "scans":     scans,
    }


@app.delete("/admin/users/{username}", summary="Delete a user (admin only)", tags=["Admin"])
async def admin_delete_user(
    username: str,
    _api_key: str  = Depends(verify_api_key),
    admin:    dict = Depends(get_admin_user),
):
    """Delete a user and their scan history. Admin only. Cannot delete self."""
    if username == admin["username"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    if username not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    del USERS[username]
    USER_SCANS.pop(username, None)
    logger.info("Admin '%s' deleted user '%s'.", admin["username"], username)
    return {"message": f"User '{username}' and their data have been deleted."}


@app.get("/admin/stats", summary="Platform-wide statistics (admin only)", tags=["Admin"])
async def admin_stats(
    _api_key: str  = Depends(verify_api_key),
    admin:    dict = Depends(get_admin_user),
):
    """Returns platform-wide aggregated statistics. Admin only."""
    all_scans = [s for scans in USER_SCANS.values() for s in scans]
    from collections import Counter
    class_counts = Counter(s["predicted_class"] for s in all_scans)
    return {
        "total_users":  len(USERS),
        "total_scans":  len(all_scans),
        "healthy":      class_counts.get("healthy", 0),
        "eus":          class_counts.get("eus", 0),
        "gill":         class_counts.get("gill", 0),
        "red_spot":     class_counts.get("red_spot", 0),
    }


# ══════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS — no auth required
# ══════════════════════════════════════════════════════════════

CHATBOT_KB = {
    # ── Disease info ──
    "eus": {
        "name": "EUS (Epizootic Ulcerative Syndrome)",
        "symptoms": "Deep ulcerative lesions on body, reddish-brown patches, fin erosion, lethargy.",
        "causes": "Caused by the oomycete Aphanomyces invadans, often triggered by monsoon conditions, low temperature, and poor water quality.",
        "treatment": [
            "Improve water quality immediately (change 30% water)",
            "Reduce stocking density to lower stress",
            "Apply potassium permanganate bath (2–4 mg/L for 30 min)",
            "Use antifungal treatment like malachite green (if legal in your region)",
            "Consult a veterinarian for oxytetracycline antibiotic therapy",
        ],
        "prevention": "Maintain good water quality, avoid overcrowding, quarantine new fish for 2 weeks before adding to pond.",
    },
    "gill": {
        "name": "Bacterial Gill Disease",
        "symptoms": "Rapid gill movement, gasping at surface, pale/swollen gills, reduced appetite, lethargy.",
        "causes": "Caused by bacteria like Flavobacterium branchiophilum. Worsened by high ammonia, low oxygen, overcrowding.",
        "treatment": [
            "Increase aeration and water flow immediately",
            "Perform 25–30% water change",
            "Apply salt bath (0.5–1% NaCl for 10–15 min)",
            "Apply potassium permanganate (2 mg/L) as a dip",
            "Consult a vet for antibiotic therapy (oxytetracycline or florfenicol)",
        ],
        "prevention": "Maintain dissolved oxygen above 5 mg/L, control ammonia levels, avoid overcrowding.",
    },
    "red_spot": {
        "name": "Bacterial Red Spot Disease (Motile Aeromonad Septicemia)",
        "symptoms": "Hemorrhagic red spots on skin, fin and scale erosion, bloating, bloody discharge from vent.",
        "causes": "Caused by Aeromonas hydrophila bacteria. Common in stressed fish with poor water quality.",
        "treatment": [
            "Isolate affected fish immediately to prevent spread",
            "Improve water quality (30% water change)",
            "Apply antibiotic bath: oxytetracycline (10–20 mg/L for 1 hour)",
            "Feed medicated pellets with oxytetracycline if available",
            "Consult a veterinarian for systemic antibiotic injection in severe cases",
        ],
        "prevention": "Avoid physical injury to fish, maintain clean water, quarantine new fish.",
    },
    "healthy": {
        "name": "Healthy Fish",
        "symptoms": "Active swimming, clear eyes, bright colors, good appetite, normal gill movement.",
        "causes": "No disease detected.",
        "treatment": ["No treatment needed. Continue regular monitoring and good pond management."],
        "prevention": "Maintain water quality, regular feeding schedule, periodic health checks.",
    },
}

CHATBOT_RESPONSES = {
    # greetings
    ("hello", "hi", "hey", "namaste", "vanakkam"): "Hello! 👋 I'm AquaScan AI Assistant. I can help you with fish disease information, treatment advice, and water quality tips. What would you like to know?",
    ("how are you", "how r u"): "I'm doing great, ready to help you keep your fish healthy! 🐟 What's your question?",
    ("bye", "goodbye", "exit", "quit"): "Goodbye! Keep your fish healthy. 🐟 Feel free to chat anytime!",
    ("thank", "thanks", "thank you"): "You're welcome! 😊 Let me know if you have more questions about your fish.",

    # disease questions
    ("what is eus", "eus disease", "about eus", "eus symptoms", "eus treatment"): None,  # handled dynamically
    ("what is gill", "gill disease", "bacterial gill", "gill symptoms", "gill treatment"): None,
    ("what is red spot", "red spot disease", "red spot symptoms", "red spot treatment"): None,

    # water quality
    ("water quality", "water ph", "ph level", "water temperature", "water oxygen"): (
        "🌊 **Ideal Water Quality Parameters for Fish Farming:**\n\n"
        "• **pH:** 7.0 – 8.5 (optimal: 7.5–8.0)\n"
        "• **Temperature:** 25–32°C for most tropical fish\n"
        "• **Dissolved Oxygen:** >5 mg/L (minimum: 4 mg/L)\n"
        "• **Ammonia (NH₃):** <0.02 mg/L\n"
        "• **Nitrite (NO₂):** <0.5 mg/L\n"
        "• **Turbidity:** 20–40 cm Secchi depth\n\n"
        "Poor water quality is the #1 cause of disease outbreaks."
    ),
    ("ammonia", "high ammonia", "ammonia level"): (
        "⚠️ **High Ammonia in Fish Pond:**\n\n"
        "High ammonia stresses fish and makes them vulnerable to disease.\n\n"
        "**Causes:** Overfeeding, dead fish, overcrowding, insufficient aeration.\n\n"
        "**Solutions:**\n"
        "• Reduce feeding by 50% immediately\n"
        "• Do a 25–30% water change\n"
        "• Increase aeration\n"
        "• Remove dead fish and debris\n"
        "• Add beneficial bacteria (probiotics) to break down ammonia"
    ),
    ("feeding", "how to feed", "fish feed", "diet"): (
        "🍽️ **Fish Feeding Guidelines:**\n\n"
        "• Feed 2–3 times per day\n"
        "• Give only what fish can eat in 5–10 minutes\n"
        "• Overfeeding is a major cause of poor water quality\n"
        "• Use quality pellets appropriate for the species\n"
        "• Reduce feeding in cold weather (fish metabolism slows)\n"
        "• Remove uneaten feed after 30 minutes"
    ),
    ("prevention", "prevent disease", "how to prevent"): (
        "🛡️ **Disease Prevention Best Practices:**\n\n"
        "1. Quarantine new fish for 2 weeks before adding to pond\n"
        "2. Maintain good water quality (check pH, oxygen, ammonia weekly)\n"
        "3. Avoid overcrowding — follow recommended stocking density\n"
        "4. Feed high-quality, disease-free feed\n"
        "5. Remove dead or sick fish immediately\n"
        "6. Disinfect equipment between ponds\n"
        "7. Regular health checks — scan fish weekly with AquaScan\n"
        "8. Keep records of disease history"
    ),
    ("salt bath", "salt treatment", "saline"): (
        "🧂 **Salt Bath Treatment:**\n\n"
        "Salt baths are effective for bacterial and parasitic infections.\n\n"
        "**Short bath (stress relief):** 0.5–1% NaCl for 10–15 minutes\n"
        "**Long bath (treatment):** 0.1–0.3% NaCl for 24–48 hours\n\n"
        "**How to prepare:** 1% = 10g salt per 1 litre of water\n\n"
        "⚠️ Watch fish closely. Remove immediately if they show stress (loss of balance, rapid breathing)."
    ),
    ("potassium permanganate", "kmno4", "permanganate"): (
        "🟣 **Potassium Permanganate (KMnO₄) Treatment:**\n\n"
        "Effective against EUS, gill disease, external parasites, and fungi.\n\n"
        "**Bath treatment:** 2–4 mg/L for 30–60 minutes\n"
        "**Pond treatment:** 2 mg/L (whole pond)\n\n"
        "⚠️ **Safety notes:**\n"
        "• Always dissolve crystals fully before adding to pond\n"
        "• Do not exceed 4 mg/L — toxic at high doses\n"
        "• Do not use in hot weather or low oxygen conditions\n"
        "• Water turns pink = correct dose; turns brown = overdose"
    ),
    ("oxytetracycline", "antibiotic", "antibiotic treatment"): (
        "💊 **Oxytetracycline Antibiotic Treatment:**\n\n"
        "Used for bacterial infections: Red Spot, Gill Disease, EUS secondary infections.\n\n"
        "**Bath treatment:** 10–20 mg/L for 1 hour\n"
        "**Medicated feed:** 55–83 mg/kg body weight per day for 10 days\n\n"
        "⚠️ **Important:**\n"
        "• Always consult a veterinarian before use\n"
        "• Complete the full course — don't stop early\n"
        "• Observe withdrawal period before selling fish for food\n"
        "• Overuse causes antibiotic resistance"
    ),
    ("stocking density", "how many fish", "overcrowding"): (
        "🐠 **Recommended Stocking Density:**\n\n"
        "• **Catfish (Catla, Rohu):** 5,000–10,000 fish/hectare\n"
        "• **Tilapia:** 3–5 fish/m² in pond\n"
        "• **Prawns:** 5–20/m² depending on system\n"
        "• **Ornamental fish (tanks):** 1 cm of fish per 1 litre of water\n\n"
        "Overcrowding = stress = disease. When in doubt, stock less."
    ),
    ("help", "what can you do", "options", "menu"): (
        "🤖 **I can help you with:**\n\n"
        "🔬 **Disease Info:** Ask about EUS, Gill Disease, Red Spot, or Healthy fish\n"
        "🌊 **Water Quality:** pH, oxygen, ammonia, temperature guidance\n"
        "💊 **Treatment:** Salt bath, potassium permanganate, antibiotics\n"
        "🛡️ **Prevention:** Best practices to keep fish healthy\n"
        "🍽️ **Feeding:** Feeding schedules and guidelines\n"
        "📊 **Scan Results:** Ask about your latest diagnosis\n\n"
        "Just type your question naturally!"
    ),
}


def get_chatbot_response(message: str, context: dict = None) -> str:
    """Rule-based chatbot with disease knowledge base."""
    msg = message.lower().strip()

    # Check for disease-specific queries
    for disease_key, disease_data in CHATBOT_KB.items():
        keywords = [disease_key, disease_data["name"].lower()]
        if disease_key == "eus":
            keywords += ["ulcer", "ulcerative", "lesion"]
        elif disease_key == "gill":
            keywords += ["gill", "breathing", "gasping", "respiratory"]
        elif disease_key == "red_spot":
            keywords += ["red spot", "hemorrhag", "bleeding", "spots", "aeromonas"]

        if any(kw in msg for kw in keywords):
            if "symptom" in msg or "sign" in msg:
                return f"🔴 **{disease_data['name']} — Symptoms:**\n\n{disease_data['symptoms']}"
            elif "cause" in msg or "why" in msg or "reason" in msg:
                return f"🔍 **{disease_data['name']} — Causes:**\n\n{disease_data['causes']}"
            elif "treat" in msg or "cure" in msg or "medicine" in msg or "how to" in msg:
                steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(disease_data["treatment"]))
                return f"💊 **{disease_data['name']} — Treatment:**\n\n{steps}"
            elif "prevent" in msg or "avoid" in msg:
                return f"🛡️ **{disease_data['name']} — Prevention:**\n\n{disease_data['prevention']}"
            else:
                steps = "\n".join(f"• {s}" for s in disease_data["treatment"])
                return (
                    f"🐟 **{disease_data['name']}**\n\n"
                    f"**Symptoms:** {disease_data['symptoms']}\n\n"
                    f"**Causes:** {disease_data['causes']}\n\n"
                    f"**Treatment:**\n{steps}\n\n"
                    f"**Prevention:** {disease_data['prevention']}"
                )

    # Check pattern responses
    for keywords, response in CHATBOT_RESPONSES.items():
        if any(kw in msg for kw in keywords):
            if response:
                return response

    # Context-aware: if user has recent scan result
    if context and context.get("last_disease"):
        disease = context["last_disease"]
        if "what should i do" in msg or "what now" in msg or "next step" in msg:
            data = CHATBOT_KB.get(disease, {})
            if data.get("treatment"):
                steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(data["treatment"]))
                return f"Based on your recent diagnosis of **{data.get('name', disease)}**, here's what to do:\n\n{steps}"

    # Fallback
    return (
        "🤔 I'm not sure about that. Here are topics I can help with:\n\n"
        "• Type **EUS**, **Gill Disease**, or **Red Spot** for disease info\n"
        "• Type **water quality** for water parameter guidance\n"
        "• Type **treatment** or **prevention** for care tips\n"
        "• Type **help** to see all available topics"
    )


@app.post("/chatbot", summary="Fish health chatbot", tags=["Chatbot"])
async def chatbot(request: Request):
    """
    Rule-based fish health assistant chatbot.
    No authentication required.

    Body: `{"message": "What is EUS?", "context": {"last_disease": "eus"}}`
    """
    check_rate_limit(request.client.host if request.client else "unknown")
    body    = await request.json()
    message = body.get("message", "").strip()
    context = body.get("context", {})

    if not message:
        raise HTTPException(status_code=400, detail="message is required.")
    if len(message) > 500:
        raise HTTPException(status_code=400, detail="Message too long. Max 500 characters.")

    response = get_chatbot_response(message, context)
    return {
        "response": response,
        "message":  message,
    }


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    """Returns API health status. No authentication required."""
    return {
        "status":       "ok",
        "model_loaded": hasattr(app.state, "model"),
        "version":      "2.0.0",
        "rate_limit":   f"{RATE_LIMIT_RPM} req/min"
    }


@app.get("/auth/api-keys-info", summary="API key info", tags=["System"])
async def api_keys_info():
    """Shows how many API keys are configured (not the keys themselves)."""
    return {
        "api_keys_configured": len(VALID_API_KEYS),
        "hint": "Set API_KEYS env var to a comma-separated list of keys. Send key in X-API-Key header."
    }