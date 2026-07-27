"""Image providers record media usage into the active token context."""

from xagent.core.model.chat.token_context import (
    TokenContextManager,
    aggregate_media_usage_by_model,
)
from xagent.core.model.image.usage import record_image_usage


def test_record_image_usage_gemini_style_tokens() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"image_url": "x", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            model_name="gemini-image",
            call_type="generate_image",
            resolution="2K",
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["type"] == "media"
    assert entry["unit"] == "images"
    assert entry["quantity"] == 1.0
    assert entry["provider_tokens"] == 8
    # Provider-reported, not a local estimate — billing may price these.
    assert entry["tokens_estimated"] is False
    assert entry["call_type"] == "generate_image"


def test_record_image_usage_honours_n_and_model_id() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {}},
            model_name="dall-e",
            model_id="img-1",
            image_count=4,
            resolution="1024x1024",
        )
        entry = manager.get_usage().details[0]

    # n>1 must bill 4 images, not 1.
    assert entry["quantity"] == 4.0
    assert entry["model_id"] == "img-1"
    assert entry["resolution"] == "1024x1024"


def test_record_image_usage_records_resolution_tier() -> None:
    # Resolution tier is recorded so cloud can price by (model, resolution),
    # while the real image tokens let a token-based price take precedence.
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            model_name="gemini-image",
            call_type="generate_image",
            resolution="2K",
        )
        entry = manager.get_usage().details[0]

    assert entry["resolution"] == "2K"
    assert entry["provider_input_tokens"] == 5
    assert entry["provider_output_tokens"] == 3


def test_record_image_usage_empty_or_missing_usage() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"image_url": "x", "usage": {}},
            model_name="sdxl",
            call_type="edit_image",
        )
        record_image_usage({"image_url": "x"}, model_name="foo")
        usage = manager.get_usage()

    assert usage.media_calls == 2
    assert all(entry["provider_tokens"] == 0 for entry in usage.details)


def test_record_image_usage_never_raises_on_garbage() -> None:
    with TokenContextManager() as manager:
        # Not a dict / no usage / None usage must all be tolerated.
        record_image_usage({}, model_name="m")  # type: ignore[arg-type]
        record_image_usage({"usage": None}, model_name="m")
        usage = manager.get_usage()

    assert usage.media_calls == 2


def test_image_usage_shows_in_media_aggregation() -> None:
    with TokenContextManager() as manager:
        record_image_usage({"usage": {}}, model_name="sd", call_type="generate_image")
        record_image_usage({"usage": {}}, model_name="sd", call_type="generate_image")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["model_name"] == "sd"
    assert groups[0]["unit"] == "images"
    assert groups[0]["quantity"] == 2.0
    assert groups[0]["calls"] == 2
