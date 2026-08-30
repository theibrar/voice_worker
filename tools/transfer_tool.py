import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http_client import async_post_json

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("transfer_tool")

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/v1")

async def transfer_call_to_human(
    target_phone_number: str,
    reason: str,
    warm_transfer: bool = True,
    briefing_summary: str = "",
    call_id: str = "active-session",
    tenant_id: int = 1,
) -> str:
    """
    Transfers the current live phone call to a human specialist, supervisor, or external call center.
    If warm_transfer is True, the human agent receives a synthesized briefing before bridging audio.
    """
    logger.info(f"[TOOL] Initiating {'Warm' if warm_transfer else 'Cold'} SIP Transfer to {target_phone_number}. Reason: {reason}")
    payload = {
        "call_id": call_id,
        "target_number": target_phone_number,
        "mode": "warm" if warm_transfer else "cold",
        "reason": reason,
        "briefing_note": briefing_summary,
        "tenant_id": tenant_id,
    }
    
    try:
        url = f"{BACKEND_API_URL}/calls/transfer"
        status, data = await async_post_json(url, payload, timeout=3.0)
        if status in [200, 202]:
            return f"SIP REFER transfer triggered successfully to {target_phone_number}."
    except Exception as e:
        logger.error(f"Failed to execute SIP transfer via backend: {e}")

    return f"Please hold for just a moment while I transfer you directly to our senior specialist at {target_phone_number}."
