"""Tests for the rerank adapter factory ``create_rerank_adapter``."""

from unittest.mock import patch

from xagent.core.model.model import RerankModelConfig
from xagent.core.model.rerank import BaseRerank, DashscopeRerank, XinferenceRerank
from xagent.core.model.rerank.adapter import RerankModelAdapter, _create_rerank_model


def _config(provider: str) -> RerankModelConfig:
    return RerankModelConfig(
        id=f"test-{provider}",
        model_name="some-model",
        model_provider=provider,
        api_key="k",
        base_url="http://example",
    )


def test_create_rerank_model_routes_to_xinference():
    config = _config("xinference")
    with patch("xinference_client.RESTfulClient") as _stub:
        model = _create_rerank_model(config)
    assert isinstance(model, XinferenceRerank)


def test_create_rerank_model_routes_to_dashscope():
    config = _config("dashscope")
    model = _create_rerank_model(config)
    assert isinstance(model, DashscopeRerank)


def test_create_rerank_model_default_provider_is_dashscope():
    # ``RerankModelConfig`` defaults to "dashscope".
    config = RerankModelConfig(
        id="default",
        model_name="m",
        api_key="k",
    )
    model = _create_rerank_model(config)
    assert isinstance(model, DashscopeRerank)


def test_rerank_model_adapter_wraps_provider_specific_model():
    config = _config("xinference")
    with patch("xinference_client.RESTfulClient"):
        adapter = RerankModelAdapter(config)
    assert isinstance(adapter, BaseRerank)
    assert isinstance(adapter._rerank_model, XinferenceRerank)


def test_rerank_model_adapter_provider_case_insensitive():
    # Uppercase provider should still route to Xinference.
    config = RerankModelConfig(
        id="upper",
        model_name="m",
        model_provider="XINFERENCE",
        api_key="k",
        base_url="http://example",
    )
    with patch("xinference_client.RESTfulClient"):
        model = _create_rerank_model(config)
    assert isinstance(model, XinferenceRerank)


def test_rerank_adapter_records_media_usage():
    """compress() records a type:"media" rerank entry on the token context."""
    from unittest.mock import MagicMock

    from xagent.core.model.chat.token_context import (
        TokenContextManager,
        aggregate_media_usage_by_model,
    )

    config = _config("dashscope")
    adapter = RerankModelAdapter(config)
    adapter._rerank_model = MagicMock()
    adapter._rerank_model.compress.return_value = ["doc-a", "doc-b"]

    with TokenContextManager() as manager:
        out = adapter.compress(["doc-a", "doc-b", "doc-c"], query="q")
        details = manager.get_usage().details

    assert out == ["doc-a", "doc-b"]
    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["call_type"] == "rerank"
    assert groups[0]["unit"] == "requests"
    assert groups[0]["model_name"] == "some-model"
