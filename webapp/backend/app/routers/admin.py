"""
admin.py (router)
------------------
All endpoints here require an authenticated Admin account
(enforced via Depends(require_admin)).

GET    /api/admin/overview             — dashboard summary cards
GET    /api/admin/analytics            — disease distribution + model performance
GET    /api/admin/users                — list/search/filter users
GET    /api/admin/users/{id}           — user detail
PATCH  /api/admin/users/{id}           — activate/deactivate, change role
DELETE /api/admin/users/{id}           — delete user (and their predictions)
GET    /api/admin/predictions          — list/search/filter/sort all predictions
GET    /api/admin/predictions/{id}     — prediction detail (admin view)
"""

from typing import Optional
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.core.database import get_db
from app.core.deps import require_admin
from app.core.config import FINAL_MODEL_ACCURACY, FINAL_MODEL_F1, FINAL_MODEL_AUC
from app.models.models import User, Prediction, UserRole
from app.schemas.schemas import (
    UserPublic, AdminUserUpdate, PredictionAdminOut, PaginatedAdminPredictions,
    AdminOverview,
)
from pathlib import Path

router = APIRouter(prefix="/api/admin", tags=["Admin"], dependencies=[Depends(require_admin)])


def _user_to_public(user: User) -> UserPublic:
    return UserPublic(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=user.role.value if isinstance(user.role, UserRole) else user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        analyses_count=len(user.predictions) if user.predictions is not None else 0,
    )


def _prediction_to_admin_out(p: Prediction) -> PredictionAdminOut:
    return PredictionAdminOut(
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
        user_name=p.user.full_name if p.user else "Unknown",
        user_email=p.user.email if p.user else "unknown",
    )


# ============================================================
# OVERVIEW
# ============================================================

@router.get("/overview", response_model=AdminOverview)
def overview(db: Session = Depends(get_db)):
    total_users = db.query(User).count()
    total_farmers = db.query(User).filter(User.role == UserRole.farmer).count()
    total_admins = db.query(User).filter(User.role == UserRole.admin).count()

    total_predictions = db.query(Prediction).filter(Prediction.status == "success").count()
    healthy_predictions = db.query(Prediction).filter(
        Prediction.status == "success", Prediction.predicted_class == "healthy"
    ).count()
    disease_predictions = total_predictions - healthy_predictions

    rows = (
        db.query(Prediction.predicted_class, func.count(Prediction.id))
        .filter(Prediction.status == "success", Prediction.predicted_class != "healthy")
        .group_by(Prediction.predicted_class)
        .order_by(desc(func.count(Prediction.id)))
        .first()
    )
    most_detected = rows[0] if rows else None

    return AdminOverview(
        total_users=total_users,
        total_farmers=total_farmers,
        total_admins=total_admins,
        total_predictions=total_predictions,
        healthy_predictions=healthy_predictions,
        disease_predictions=disease_predictions,
        most_detected_disease=most_detected,
        model_accuracy=FINAL_MODEL_ACCURACY,
        model_f1=FINAL_MODEL_F1,
        model_auc=FINAL_MODEL_AUC,
    )


@router.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    """Live disease-distribution counts from the app's own stored
    predictions (from real users using the system), plus the static,
    already-computed research metrics for model performance."""
    rows = (
        db.query(Prediction.predicted_class, func.count(Prediction.id))
        .filter(Prediction.status == "success")
        .group_by(Prediction.predicted_class)
        .all()
    )
    live_distribution = {cls: count for cls, count in rows}
    for cls in ["EUS", "gill", "healthy", "red_spot"]:
        live_distribution.setdefault(cls, 0)

    return {
        "live_disease_distribution": live_distribution,
        "model_performance": {
            "accuracy": FINAL_MODEL_ACCURACY,
            "f1_score": FINAL_MODEL_F1,
            "auc": FINAL_MODEL_AUC,
        },
    }


# ============================================================
# USER MANAGEMENT
# ============================================================

@router.get("/users")
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    role: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(User)
    if role and role != "all":
        q = q.filter(User.role == role)
    if search:
        q = q.filter(
            (User.full_name.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%"))
        )
    total = q.count()
    items = q.order_by(desc(User.created_at)).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items": [_user_to_public(u) for u in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/users/{user_id}", response_model=UserPublic)
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return _user_to_public(user)


@router.patch("/users/{user_id}", response_model=UserPublic)
def update_user(
    user_id: str,
    payload: AdminUserUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.id == admin.id and payload.is_active is False:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")

    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.role is not None:
        if payload.role not in ("farmer", "admin"):
            raise HTTPException(status_code=400, detail="Invalid role.")
        user.role = UserRole(payload.role)

    db.commit()
    db.refresh(user)
    return _user_to_public(user)


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully."}


# ============================================================
# PREDICTION MANAGEMENT
# ============================================================

@router.get("/predictions", response_model=PaginatedAdminPredictions)
def list_all_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    disease: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    sort: str = Query("date_desc", pattern="^(date_desc|date_asc|confidence_desc|confidence_asc)$"),
    db: Session = Depends(get_db),
):
    q = db.query(Prediction).join(User, Prediction.user_id == User.id)

    if disease and disease != "all":
        q = q.filter(Prediction.predicted_class == disease)
    if status_filter and status_filter != "all":
        q = q.filter(Prediction.status == status_filter)
    if search:
        q = q.filter(
            (User.full_name.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%"))
        )

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

    return PaginatedAdminPredictions(
        items=[_prediction_to_admin_out(p) for p in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionAdminOut)
def get_prediction_admin(prediction_id: str, db: Session = Depends(get_db)):
    p = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return _prediction_to_admin_out(p)
