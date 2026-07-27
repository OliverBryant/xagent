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

from ...model.chat.token_context import MediaCallType, MediaUnit, add_media_usage

logger = logging.getLogger(__name__)


def coerce_duration(value: object) -> Optional[float]:
    """A positive duration in seconds, or None when unusable.

    Distinct from ``token_context._coerce_float``, which folds bad input to
    ``0.0``: here the caller must be able to tell "provider reported no
    duration" apart from "provider reported zero", because those take
    different metering branches.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def record_media_usage(
    unit: MediaUnit | str,
    quantity: float,
    *,
    model: str = "",
    model_id: str = "",
    call_type: MediaCallType | str = "",
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


def record_media_seconds(
    seconds: Optional[float],
    *,
    model: str = "",
    model_id: str = "",
    call_type: MediaCallType | str = "",
) -> None:
    """Record a duration-billed media call, keeping the unit stable.

    Duration-billed modalities (video/ASR/music/sound effect) must always
    report ``MediaUnit.SECONDS``: a price table keyed on (model, unit) breaks
    if the same model sometimes reports "requests" just because the provider
    omitted a duration. When the duration is unknown the call is still recorded
    — as ``seconds`` with ``quantity=0`` and a warning — so the event is
    visible to billing as unmeasured rather than silently mis-dimensioned.
    """
    if seconds is None:
        logger.warning(
            "No duration reported for %s call on model %r; recording 0 seconds "
            "(call happened but is unmeasured)",
            call_type,
            model,
        )
    record_media_usage(
        MediaUnit.SECONDS,
        seconds or 0.0,
        model=model,
        model_id=model_id,
        call_type=call_type,
    )
