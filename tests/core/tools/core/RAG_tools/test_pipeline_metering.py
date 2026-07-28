"""Rerank metering must survive the production search path.

Rerank previously shipped completely unbilled: the search pipeline unwrapped
the adapter to reach the raw provider's ``compress_with_scores``, which had no
metering. A unit test that calls the adapter directly cannot catch that — it
passes whether or not the orchestration bypasses the adapter.

So this drives ``_try_unified_rerank``, the function that actually chooses and
calls the rerank object. Re-introducing the unwrap there makes these fail.
"""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from xagent.core.model.chat.token_context import (
    TokenContextManager,
    aggregate_media_usage_by_model,
)
from xagent.core.model.model import RerankModelConfig
from xagent.core.model.rerank.adapter import RerankModelAdapter
from xagent.core.model.rerank.base import BaseRerank
from xagent.core.tools.core.RAG_tools.core.schemas import SearchConfig, SearchResult
from xagent.core.tools.core.RAG_tools.pipelines import document_search


def _metered_adapter() -> RerankModelAdapter:
    """The real adapter (which meters) wrapping a faked provider."""
    adapter = RerankModelAdapter.__new__(RerankModelAdapter)
    adapter.model_config = RerankModelConfig(
        id="rr-1", model_name="rerank-v1", api_key="k"
    )
    adapter._rerank_model = MagicMock()
    adapter._rerank_model.compress.return_value = ["doc-b", "doc-a"]
    adapter._rerank_model.compress_with_scores.return_value = [
        ("doc-b", 0.9),
        ("doc-a", 0.4),
    ]
    return adapter


def _cfg() -> SearchConfig:
    return SearchConfig(embedding_model_id="emb-1", rerank_model_id="rr-1")


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            doc_id="d1",
            chunk_id=f"c{i}",
            text=text,
            score=0.5,
            parse_hash="h",
            model_tag="emb-1",
        )
        for i, text in enumerate(["doc-a", "doc-b"])
    ]


@pytest.fixture
def patched_resolver(monkeypatch: pytest.MonkeyPatch):
    """Point the pipeline's resolver at our metered adapter."""

    adapter = _metered_adapter()

    def _resolve(_cfg: Optional[SearchConfig] = None):
        return adapter

    monkeypatch.setattr(document_search, "_resolve_unified_rerank", _resolve)
    return adapter


def test_search_pipeline_rerank_is_metered(patched_resolver) -> None:
    """The production rerank entry point must record usage.

    Fails if _try_unified_rerank reaches past the adapter to the raw provider,
    which is the exact bug this guards.
    """
    warnings: list[str] = []
    with TokenContextManager() as manager:
        outcome = document_search._try_unified_rerank(
            _results(), "query", _cfg(), warnings
        )
        details = manager.get_usage().details

    assert outcome is not None, "rerank did not run; test is not exercising the path"
    groups = [
        g for g in aggregate_media_usage_by_model(details) if g["call_type"] == "rerank"
    ]
    assert groups, (
        "rerank ran but recorded no usage — the search pipeline is reaching "
        "past the metered adapter"
    )
    assert groups[0]["unit"] == "requests"
    assert groups[0]["quantity"] == 1.0
    assert groups[0]["model_name"] == "rerank-v1"
    assert groups[0]["model_id"] == "rr-1"


def test_rerank_usage_is_recorded_once_per_search(patched_resolver) -> None:
    """Two searches bill two rerank calls — guards against both double-counting
    and a silently skipped record."""
    cfg = _cfg()
    with TokenContextManager() as manager:
        document_search._try_unified_rerank(_results(), "q1", cfg, [])
        document_search._try_unified_rerank(_results(), "q2", cfg, [])
        details = manager.get_usage().details

    groups = [
        g for g in aggregate_media_usage_by_model(details) if g["call_type"] == "rerank"
    ]
    assert len(groups) == 1
    assert groups[0]["calls"] == 2
    assert groups[0]["quantity"] == 2.0


def test_scored_rerank_is_reachable_without_unwrapping() -> None:
    """`compress_with_scores` on the base is what lets the pipeline get scores
    without bypassing the adapter. If it moved back to the providers only, the
    unwrap would return."""
    assert callable(getattr(BaseRerank, "compress_with_scores", None))
    adapter = _metered_adapter()
    assert adapter.compress_with_scores(["doc-a"], "q") == [
        ("doc-b", 0.9),
        ("doc-a", 0.4),
    ]


def test_resolver_returns_the_adapter_not_the_inner_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolver must hand back the metered adapter itself."""
    adapter = _metered_adapter()
    monkeypatch.setattr(
        document_search,
        "resolve_rerank_adapter",
        lambda **_: (None, adapter),
    )

    resolved = document_search._resolve_unified_rerank(_cfg())
    assert resolved is adapter
    assert resolved is not adapter._rerank_model
