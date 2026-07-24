"""Best-effort media-usage recording for media generation tools.

TTS/ASR/video/music/sound-effect models don't return a normalised usage
payload the way image/chat models do, and their adapters are factory-only, so
the natural metering point is the tool call site where the request params and
result are both in scope. This helper wraps ``add_media_usage`` so a failure in
accounting can never break the underlying media call.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...model.chat.token_context import add_media_usage

logger = logging.getLogger(__name__)


def record_media_usage(
    unit: str,
    quantity: float,
    *,
    model: str = "",
    model_id: str = "",
    call_type: str = "",
) -> None:
    """Record one media model call; swallow any error."""
    try:
        add_media_usage(
            unit=unit,
            quantity=quantity,
            model=model,
            model_id=model_id,
            call_type=call_type,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record %s media usage: %s", call_type, e)


def _coerce_float(value: object) -> Optional[float]:
    """float(value) or None if not a usable number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
