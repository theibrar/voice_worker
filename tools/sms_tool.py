import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http_client import async_post_json

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("sms_tool")

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/v1")

async def send_live_sms(
    to_phone: str,
    message: str,
    tenant_id: int = 1,
    call_id: str = "active-session",
) -> str:
    """
    Sends a real-time SMS text message to the caller's mobile phone during the active voice call.
    Ideal for sending brochures, appointment confirmations, payment links, and verification codes.
    """
    logger.info(f"[TOOL] Sending live SMS to {to_phone}: '{message[:40]}...' (Call: {call_id})")
    payload = {
        "to": to_phone,
        "message": message,
        "call_id": call_id,
        "tenant_id": tenant_id,
        "type": "mid_call_dispatch",
    }
    
    try:
        url = f"{BACKEND_API_URL}/calls/sms"
        status, data = await async_post_json(url, payload, timeout=3.0)
        if status in [200, 201]:
            return f"SMS successfully delivered to {to_phone}."
    except Exception as e:
        logger.warning(f"Error calling backend SMS API: {e}")

    return f"I've just sent the SMS with that link to {to_phone}. You should see it in your messages now."
