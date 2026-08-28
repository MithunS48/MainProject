"""
users.py (router)
------------------
GET  /api/users/me           — profile info
PUT  /api/users/me           — edit own profile (name / email)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, UserRole
from app.schemas.schemas import UserPublic, UserUpdate

router = APIRouter(prefix="/api/users", tags=["Users"])


def _to_public(user: User) -> UserPublic:
    return UserPublic(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=user.role.value if isinstance(user.role, UserRole) else user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        analyses_count=len(user.predictions) if user.predictions is not None else 0,
    )


@router.get("/me", response_model=UserPublic)
def get_my_profile(current_user: User = Depends(get_current_user)):
    return _to_public(current_user)


@router.put("/me", response_model=UserPublic)
def update_my_profile(
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.full_name:
        current_user.full_name = payload.full_name.strip()
    if payload.email and str(payload.email).lower() != current_user.email:
        exists = db.query(User).filter(User.email == str(payload.email).lower()).first()
        if exists:
            raise HTTPException(status_code=409, detail="Email already in use.")
        current_user.email = str(payload.email).lower().strip()

    db.commit()
    db.refresh(current_user)
    return _to_public(current_user)
