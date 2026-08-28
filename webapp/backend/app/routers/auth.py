"""
auth.py
-------
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me

Admin registration protection: creating an account with role="admin"
through this PUBLIC endpoint requires a correct `admin_invite_code`
(see app.core.config.ADMIN_INVITE_CODE). Without it, the request is
rejected — nobody can self-register as an administrator.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.core.deps import get_current_user
from app.core.config import ADMIN_INVITE_CODE
from app.models.models import User, UserRole
from app.schemas.schemas import UserRegister, UserLogin, Token, UserPublic

router = APIRouter(prefix="/api/auth", tags=["Auth"])


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


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    role = UserRole.farmer
    if payload.role == "admin":
        if not payload.admin_invite_code or payload.admin_invite_code != ADMIN_INVITE_CODE:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Administrator accounts cannot be created without a valid "
                    "invite code. Please contact the system owner, or register "
                    "as a Farmer/User."
                ),
            )
        role = UserRole.admin

    user = User(
        full_name=payload.full_name.strip(),
        email=str(payload.email).lower().strip(),
        hashed_password=hash_password(payload.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id, "role": user.role.value})
    return Token(access_token=token, user=_to_public(user))


@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == str(payload.email).lower().strip()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account has been deactivated.")

    token = create_access_token({"sub": user.id, "role": user.role.value})
    return Token(access_token=token, user=_to_public(user))


@router.post("/logout")
def logout():
    # Stateless JWT — logout is handled client-side by discarding the token.
    return {"message": "Logged out successfully."}


@router.get("/me", response_model=UserPublic)
def me(current_user: User = Depends(get_current_user)):
    return _to_public(current_user)
