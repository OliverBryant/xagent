"""Unit stability for duration-billed media tools.

The billed unit must depend on the modality, never on how complete the
provider's response happened to be — a price table keyed on (model, unit) is
unusable if the same model sometimes reports "seconds" and sometimes
"requests".
"""

import pytest

from xagent.core.model.chat.token_context import (
    TokenContextManager,
    aggregate_media_usage_by_model,
)
from xagent.core.tools.core.media_usage import (
    coerce_duration,
    record_media_seconds,
    record_media_usage,
    resolve_billing_model,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (12.5, 12.5),
        ("3", 3.0),
        (0, None),  # zero is "no usable duration", not a real measurement
        (-5, None),
        (None, None),
        (True, None),  # bool must not sneak through as 1.0
        ("nonsense", None),
    ],
)
def test_coerce_duration(value, expected) -> None:
    assert coerce_duration(value) == expected


def test_record_media_seconds_keeps_unit_stable_when_duration_missing() -> None:
    # Same model, one call with a duration and one without: both must report
    # seconds so billing sees a single line item, not two different units.
    with TokenContextManager() as manager:
        record_media_seconds(30.0, model="veo", call_type="video")
        record_media_seconds(None, model="veo", call_type="video")
        details = manager.get_usage().details

    assert [entry["unit"] for entry in details] == ["seconds", "seconds"]
    # The unmeasured call records 0 and is dropped from the billable rollup.
    assert [entry["quantity"] for entry in details] == [30.0, 0.0]
    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["unit"] == "seconds"
    assert groups[0]["quantity"] == 30.0


def test_record_media_seconds_warns_when_unmeasured(caplog) -> None:
    with caplog.at_level("WARNING"):
        with TokenContextManager():
            record_media_seconds(None, model="veo", call_type="video")
    assert "unmeasured" in caplog.text


def test_record_media_usage_never_raises() -> None:
    # Accounting must never break the underlying media call.
    with TokenContextManager() as manager:
        record_media_usage("seconds", None, model="m", call_type="video")  # type: ignore[arg-type]
        assert manager.get_usage().media_calls == 1


def test_resolve_billing_model_never_returns_a_placeholder() -> None:
    """`_configured_model_id`-style lookups return Optional[str]; passing that
    through str() records a model literally named "None"."""

    class _Model:
        model_name = "elevenlabs-music-v1"

    # None id -> falls back to the provider's own name, not "None".
    assert resolve_billing_model(None, _Model()) == "elevenlabs-music-v1"
    assert resolve_billing_model("", _Model()) == "elevenlabs-music-v1"
    # A real configured id always wins.
    assert resolve_billing_model("cfg-id", _Model()) == "cfg-id"
    # Nothing identifies the model: an explicit fallback, never None/"None".
    assert resolve_billing_model(None, None) == "default"

    # Placeholder names on the model are not treated as identities.
    class _Placeholder:
        model_name = "None"

    assert resolve_billing_model(None, _Placeholder()) == "default"
