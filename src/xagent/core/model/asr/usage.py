"""Media-usage recording for ASR providers.

Recorded at the provider layer rather than the tool layer because ASR is
reached from several entry points that do not go through ``audio_tool`` — the
Telegram channel and the ``/speech/transcribe`` API both call ``transcribe``
directly. Metering here makes every caller billable by construction.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from ..chat.token_context import MediaCallType, MediaUnit, add_media_usage
from .base import ASRResult

logger = logging.getLogger(__name__)


def _duration_from_raw(raw_response: Any) -> Optional[float]:
    """Provider-reported total audio duration, if it exposed one."""
    if not isinstance(raw_response, dict):
        return None
    for key in ("duration", "audio_duration", "duration_seconds"):
        value = raw_response.get(key)
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    return None


def _duration_from_segments(result: ASRResult) -> Optional[float]:
    """Fall back to the end of the last timed segment."""
    if not result.segments:
        return None
    last_end = 0.0
    for segment in result.segments:
        end = getattr(segment, "end", None)
        if end is None:
            continue
        try:
            last_end = max(last_end, float(end))
        except (TypeError, ValueError):
            continue
    return last_end if last_end > 0 else None


def record_asr_usage(
    result: Union[str, ASRResult],
    *,
    model_name: str = "",
    model_id: str = "",
) -> None:
    """Record one transcription on the current token context.

    ASR is duration-billed, so the unit is always seconds — a call whose
    duration cannot be determined records 0 seconds rather than switching
    units, which would make a (model, unit) price table unusable. Best-effort:
    any failure is swallowed so accounting never breaks a transcription.
    """
    try:
        seconds: Optional[float] = None
        if isinstance(result, ASRResult):
            seconds = _duration_from_raw(
                result.raw_response
            ) or _duration_from_segments(result)
        if seconds is None:
            logger.warning(
                "No audio duration available for ASR call on model %r; "
                "recording 0 seconds (call happened but is unmeasured)",
                model_name,
            )
        add_media_usage(
            unit=MediaUnit.SECONDS,
            quantity=seconds or 0.0,
            model=model_name,
            model_id=model_id,
            call_type=MediaCallType.ASR,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record ASR usage: %s", e)
