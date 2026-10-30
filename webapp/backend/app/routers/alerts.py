"""
alerts.py
---------
Alert notification endpoints.

POST /api/alerts/test-email   — send a test email to the current user
POST /api/alerts/test-sms     — send a test SMS
POST /api/alerts/iot          — manually trigger an IoT water-quality alert
POST /api/alerts/disease      — manually trigger a disease detection alert

All endpoints require a valid user JWT.
Sending is non-blocking and returns immediately with a status report.
"""

from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.deps import get_current_user
from app.core.notifications import send_alert, build_iot_alert, build_disease_alert
from app.core.config import SMTP_ENABLED, TWILIO_ENABLED
from app.models.models import User

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


class AlertConfig(BaseModel):
    email:     Optional[str] = None   # recipient email (defaults to user's own email)
    phone:     Optional[str] = None   # E.164 format, e.g. +1234567890


class IoTAlertRequest(AlertConfig):
    device_id: str
    alerts:    list[str]
    ph:        float
    temperature: float


class DiseaseAlertRequest(AlertConfig):
    predicted_class: str
    predicted_label: str
    confidence_pct:  float
    image_url:       str = ""


@router.get("/status", summary="Alert channel status")
def alert_status(current_user: User = Depends(get_current_user)):
    """Returns which alert channels are currently configured."""
    return {
        "email_enabled": SMTP_ENABLED,
        "sms_enabled":   TWILIO_ENABLED,
    }


@router.post("/test-email", summary="Send a test email alert")
def test_email(
    payload: AlertConfig,
    current_user: User = Depends(get_current_user),
):
    to = payload.email or current_user.email
    results = send_alert(
        subject   = "✅ AquaScan — Email Alert Test",
        body_text = (
            f"Hello {current_user.full_name},\n\n"
            "This is a test email from AquaScan.\n"
            "Your email alerts are configured correctly."
        ),
        to_email  = to,
    )
    return {"sent_to": to, "results": results}


@router.post("/test-sms", summary="Send a test SMS alert")
def test_sms(
    payload: AlertConfig,
    current_user: User = Depends(get_current_user),
):
    if not payload.phone:
        return {"error": "Phone number required for SMS test."}
    results = send_alert(
        subject   = "AquaScan SMS Test",
        body_text = f"Hello {current_user.full_name}, your AquaScan SMS alerts are working.",
        to_phone  = payload.phone,
    )
    return {"sent_to": payload.phone, "results": results}


@router.post("/iot", summary="Trigger an IoT water-quality alert")
def trigger_iot_alert(
    payload: IoTAlertRequest,
    current_user: User = Depends(get_current_user),
):
    to_email = payload.email or current_user.email
    subject, body_text, body_html = build_iot_alert(
        payload.device_id, payload.alerts, payload.ph, payload.temperature
    )
    results = send_alert(
        subject   = subject,
        body_text = body_text,
        body_html = body_html,
        to_email  = to_email,
        to_phone  = payload.phone,
    )
    return {"results": results}


@router.post("/disease", summary="Trigger a disease detection alert")
def trigger_disease_alert(
    payload: DiseaseAlertRequest,
    current_user: User = Depends(get_current_user),
):
    to_email = payload.email or current_user.email
    subject, body_text, body_html = build_disease_alert(
        payload.predicted_class,
        payload.predicted_label,
        payload.confidence_pct,
        payload.image_url,
    )
    results = send_alert(
        subject   = subject,
        body_text = body_text,
        body_html = body_html,
        to_email  = to_email,
        to_phone  = payload.phone,
    )
    return {"results": results}
