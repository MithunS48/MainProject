"""
notifications.py
----------------
Sends email and/or SMS alert notifications.

Both channels are optional and disabled by default.
Enable by setting environment variables (see config.py).

Email  — uses Python stdlib smtplib (no extra dependency).
SMS    — uses the Twilio REST API via the `twilio` package
         (pip install twilio).  Gracefully degrades to a warning
         if the package is not installed.

Usage
-----
    from app.core.notifications import send_alert

    send_alert(
        to_email   = "farmer@example.com",
        to_phone   = "+1234567890",   # optional
        subject    = "⚠️ Water Quality Alert — Pond 1",
        body_text  = "pH 5.2 is too low. Immediate action required.",
        body_html  = "<p>...</p>",    # optional
    )
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from app.core.config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_ENABLED,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_ENABLED,
)

logger = logging.getLogger(__name__)


# ─── Email ────────────────────────────────────────────────────────────────────

def send_email(
    to_email:  str,
    subject:   str,
    body_text: str,
    body_html: Optional[str] = None,
) -> bool:
    """
    Send an email via SMTP.
    Returns True on success, False on failure.
    Raises nothing — logs errors internally.
    """
    if not SMTP_ENABLED:
        logger.warning(
            "Email not configured (FISHAPP_SMTP_HOST/USER/PASSWORD unset). "
            "Skipping email to %s.", to_email
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM
        msg["To"]      = to_email

        msg.attach(MIMEText(body_text, "plain"))
        if body_html:
            msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())

        logger.info("Email sent to %s: %s", to_email, subject)
        return True

    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc)
        return False


# ─── SMS ──────────────────────────────────────────────────────────────────────

def send_sms(to_phone: str, body: str) -> bool:
    """
    Send an SMS via Twilio.
    Returns True on success, False on failure.
    Raises nothing — logs errors internally.
    """
    if not TWILIO_ENABLED:
        logger.warning(
            "Twilio not configured (FISHAPP_TWILIO_* unset). "
            "Skipping SMS to %s.", to_phone
        )
        return False

    try:
        from twilio.rest import Client  # type: ignore
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body[:1600],           # Twilio max SMS length
            from_=TWILIO_FROM_NUMBER,
            to=to_phone,
        )
        logger.info("SMS sent to %s (SID: %s)", to_phone, message.sid)
        return True

    except ImportError:
        logger.error(
            "Twilio package not installed. Run: pip install twilio"
        )
        return False
    except Exception as exc:
        logger.error("Failed to send SMS to %s: %s", to_phone, exc)
        return False


# ─── Combined alert ───────────────────────────────────────────────────────────

def send_alert(
    subject:    str,
    body_text:  str,
    to_email:   Optional[str] = None,
    to_phone:   Optional[str] = None,
    body_html:  Optional[str] = None,
) -> dict:
    """
    Send alert via all configured and requested channels.
    Returns a dict of channel → bool indicating success.
    """
    results = {}

    if to_email:
        results["email"] = send_email(to_email, subject, body_text, body_html)

    if to_phone:
        results["sms"] = send_sms(to_phone, f"{subject}\n\n{body_text}")

    return results


# ─── Alert message builders ───────────────────────────────────────────────────

def build_iot_alert(device_id: str, alerts: list[str], ph: float, temp: float) -> tuple[str, str, str]:
    """Returns (subject, body_text, body_html) for an IoT water-quality alert."""
    subject   = f"⚠️ AquaScan Water Quality Alert — {device_id}"
    body_text = (
        f"Water quality alert for device {device_id}.\n\n"
        f"Current readings:\n"
        f"  pH          : {ph:.2f}\n"
        f"  Temperature : {temp:.1f} °C\n\n"
        f"Alerts:\n" + "\n".join(f"  • {a}" for a in alerts) +
        "\n\nLog in to AquaScan to view details."
    )
    body_html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;">
      <h2 style="color:#f97316;">⚠️ Water Quality Alert</h2>
      <p><strong>Device:</strong> {device_id}</p>
      <table style="border-collapse:collapse;width:100%;">
        <tr><td style="padding:6px 12px;background:#f0f9ff;"><strong>pH</strong></td>
            <td style="padding:6px 12px;">{ph:.2f}</td></tr>
        <tr><td style="padding:6px 12px;background:#f0f9ff;"><strong>Temperature</strong></td>
            <td style="padding:6px 12px;">{temp:.1f} °C</td></tr>
      </table>
      <h3 style="color:#dc2626;">Alerts</h3>
      <ul>{"".join(f"<li>{a}</li>" for a in alerts)}</ul>
      <p style="color:#64748b;font-size:12px;">Log in to AquaScan to view full details.</p>
    </div>
    """
    return subject, body_text, body_html


def build_disease_alert(
    predicted_class: str,
    predicted_label: str,
    confidence_pct:  float,
    image_url:       str,
) -> tuple[str, str, str]:
    """Returns (subject, body_text, body_html) for a disease detection alert."""
    subject = f"🐟 AquaScan Disease Alert — {predicted_label}"
    body_text = (
        f"Disease detected in your fish image.\n\n"
        f"  Diagnosis   : {predicted_label}\n"
        f"  Confidence  : {confidence_pct:.1f}%\n\n"
        f"Log in to AquaScan to view the full diagnosis and recommended actions."
    )
    body_html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;">
      <h2 style="color:#dc2626;">🐟 Disease Detection Alert</h2>
      <table style="border-collapse:collapse;width:100%;">
        <tr><td style="padding:6px 12px;background:#fef2f2;"><strong>Diagnosis</strong></td>
            <td style="padding:6px 12px;">{predicted_label}</td></tr>
        <tr><td style="padding:6px 12px;background:#fef2f2;"><strong>Confidence</strong></td>
            <td style="padding:6px 12px;">{confidence_pct:.1f}%</td></tr>
      </table>
      <p style="color:#64748b;font-size:12px;">Log in to AquaScan to view full details and recommended actions.</p>
    </div>
    """
    return subject, body_text, body_html
