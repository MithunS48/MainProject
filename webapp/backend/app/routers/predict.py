"""
predict.py (router)
--------------------
POST /api/predict            — run real inference on an uploaded fish image
GET  /api/predictions        — current user's prediction history (paginated)
GET  /api/predictions/{id}   — single prediction detail
GET  /api/model-status       — are the required trained-model files present?
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.config import UPLOADS_DIR, MAX_UPLOAD_SIZE_BYTES, ALLOWED_CONTENT_TYPES
from app.core.inference import run_prediction, ModelNotAvailableError, check_model_availability
from app.models.models import User, Prediction
from app.schemas.schemas import PredictionOut, PaginatedPredictions

router = APIRouter(prefix="/api", tags=["Prediction"])


def _prediction_to_out(p: Prediction) -> PredictionOut:
    return PredictionOut(
        id=p.id,
        user_id=p.user_id,
        image_url=f"/static/uploads/{Path(p.image_path).name}",
        original_filename=p.original_filename,
        predicted_class=p.predicted_class,
        predicted_label=p.predicted_label,
        confidence=p.confidence,
        confidence_pct=p.confidence_pct,
        all_scores={
            "EUS": p.prob_eus or 0.0,
            "gill": p.prob_gill or 0.0,
            "healthy": p.prob_healthy or 0.0,
            "red_spot": p.prob_red_spot or 0.0,
        },
        status=p.status,
        error_message=p.error_message,
        created_at=p.created_at,
    )


@router.get("/model-status")
def model_status():
    return check_model_availability()


@router.post("/predict", response_model=PredictionOut)
async def predict_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPG or PNG.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    ext = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOADS_DIR / saved_name
    with open(saved_path, "wb") as f:
        f.write(contents)

    try:
        result = run_prediction(str(saved_path))
    except ModelNotAvailableError as e:
        prediction = Prediction(
            user_id=current_user.id,
            image_path=str(saved_path),
            original_filename=file.filename,
            predicted_class="unknown",
            predicted_label="Unavailable",
            confidence=0.0,
            confidence_pct=0.0,
            status="error",
            error_message=str(e),
        )
        db.add(prediction)
        db.commit()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        prediction = Prediction(
            user_id=current_user.id,
            image_path=str(saved_path),
            original_filename=file.filename,
            predicted_class="unknown",
            predicted_label="Error",
            confidence=0.0,
            confidence_pct=0.0,
            status="error",
            error_message=f"Inference error: {e}",
        )
        db.add(prediction)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    scores = result["all_scores"]
    prediction = Prediction(
        user_id=current_user.id,
        image_path=str(saved_path),
        original_filename=file.filename,
        predicted_class=result["predicted_class"],
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        confidence_pct=result["confidence_pct"],
        prob_eus=scores.get("EUS"),
        prob_gill=scores.get("gill"),
        prob_healthy=scores.get("healthy"),
        prob_red_spot=scores.get("red_spot"),
        status="success",
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return _prediction_to_out(prediction)


@router.get("/predictions", response_model=PaginatedPredictions)
def get_my_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    disease: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort: str = Query("date_desc", pattern="^(date_desc|date_asc|confidence_desc|confidence_asc)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Prediction).filter(Prediction.user_id == current_user.id)

    if disease and disease != "all":
        q = q.filter(Prediction.predicted_class == disease)
    if search:
        q = q.filter(Prediction.predicted_label.ilike(f"%{search}%"))

    if sort == "date_desc":
        q = q.order_by(desc(Prediction.created_at))
    elif sort == "date_asc":
        q = q.order_by(Prediction.created_at)
    elif sort == "confidence_desc":
        q = q.order_by(desc(Prediction.confidence))
    else:
        q = q.order_by(Prediction.confidence)

    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()

    return PaginatedPredictions(
        items=[_prediction_to_out(p) for p in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction_detail(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    if p.user_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this prediction.")
    return _prediction_to_out(p)
