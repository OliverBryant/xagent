"""The embedding adapter records media usage on the token context."""

from unittest.mock import MagicMock

from xagent.core.model.chat.token_context import (
    TokenContextManager,
    aggregate_media_usage_by_model,
)
from xagent.core.model.embedding.adapter import EmbeddingModelAdapter
from xagent.core.model.model import EmbeddingModelConfig


def _adapter() -> EmbeddingModelAdapter:
    config = EmbeddingModelConfig(
        id="e1",
        model_name="text-embed",
        model_provider="openai",
        api_key="k",
        base_url="http://example",
    )
    # Build without constructing a real provider client.
    adapter = EmbeddingModelAdapter.__new__(EmbeddingModelAdapter)
    adapter.model_config = config
    adapter._embedding_model = MagicMock()
    return adapter


def test_embedding_encode_records_requests_and_estimates_tokens():
    adapter = _adapter()
    adapter._embedding_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    with TokenContextManager() as manager:
        out = adapter.encode(["hello world", "foo bar baz"])
        details = manager.get_usage().details

    assert out == [[0.1, 0.2], [0.3, 0.4]]
    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["call_type"] == "embedding"
    assert groups[0]["unit"] == "requests"
    assert groups[0]["quantity"] == 2.0  # two input texts
    assert groups[0]["model_name"] == "text-embed"


def test_embedding_encode_single_string_counts_one_request():
    adapter = _adapter()
    adapter._embedding_model.encode.return_value = [0.1, 0.2]

    with TokenContextManager() as manager:
        adapter.encode("just one")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert groups[0]["quantity"] == 1.0
