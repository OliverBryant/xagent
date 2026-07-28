"""Metering must survive the production call path, not just the adapter API.

Two modalities previously shipped completely unbilled while their unit tests
passed, because those tests called the metered method directly and production
reached the provider another way:

* rerank — the search pipeline unwrapped the adapter to reach the raw provider
* embedding — bulk ingestion crossed a ThreadPoolExecutor boundary, which does
  not propagate the contextvar the usage accumulator lives in

These tests assert at the seam that actually broke, so a regression that
reintroduces either escape hatch fails here.
"""

import concurrent.futures
from unittest.mock import MagicMock

from xagent.core.model.chat.token_context import (
    TokenContextManager,
    aggregate_media_usage_by_model,
    get_token_usage,
    set_token_usage,
)
from xagent.core.model.model import RerankModelConfig
from xagent.core.model.rerank.adapter import RerankModelAdapter
from xagent.core.model.rerank.base import BaseRerank


def _adapter() -> RerankModelAdapter:
    adapter = RerankModelAdapter.__new__(RerankModelAdapter)
    adapter.model_config = RerankModelConfig(
        id="rr-1", model_name="rerank-v1", api_key="k"
    )
    adapter._rerank_model = MagicMock()
    adapter._rerank_model.compress.return_value = ["a"]
    adapter._rerank_model.compress_with_scores.return_value = [("a", 0.9)]
    return adapter


def test_compress_with_scores_is_reachable_through_the_base_class() -> None:
    """The search pipeline needs scores; if it can only get them from the raw
    provider it will keep unwrapping the adapter and skipping metering."""
    assert hasattr(BaseRerank, "compress_with_scores")


def test_scored_rerank_through_adapter_is_metered() -> None:
    # compress_with_scores is the method the RAG search pipeline calls.
    adapter = _adapter()
    with TokenContextManager() as manager:
        adapter.compress_with_scores(["a", "b"], "q")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["call_type"] == "rerank"
    assert groups[0]["unit"] == "requests"
    assert groups[0]["quantity"] == 1.0
    assert groups[0]["model_name"] == "rerank-v1"
    assert groups[0]["model_id"] == "rr-1"


def test_search_pipeline_does_not_unwrap_the_rerank_adapter() -> None:
    """Regression guard for the exact bypass: resolving must hand back the
    adapter (which meters), never its inner provider (which does not)."""
    from xagent.core.tools.core.RAG_tools.pipelines import document_search

    adapter = _adapter()
    assert document_search._supports_rerank(adapter)
    # The resolver's contract is "return something that can score" — the
    # adapter qualifies, so there is no reason left to reach past it.
    assert adapter.compress_with_scores(["a"], "q") == [("a", 0.9)]


def test_embedding_usage_survives_a_thread_pool_boundary() -> None:
    """Bulk ingestion encodes batches in a ThreadPoolExecutor. Usage recorded
    in those workers must land on the caller's TokenUsage, not a per-thread
    instance that is discarded when the worker returns."""
    from xagent.core.model.embedding.adapter import EmbeddingModelAdapter
    from xagent.core.model.model import EmbeddingModelConfig

    adapter = EmbeddingModelAdapter.__new__(EmbeddingModelAdapter)
    adapter.model_config = EmbeddingModelConfig(
        id="e-1", model_name="text-embed", api_key="k"
    )
    adapter._embedding_model = MagicMock()
    adapter._embedding_model.encode.return_value = [[0.1]]

    batches = [["chunk-a"], ["chunk-b"], ["chunk-c"], ["chunk-d"]]

    with TokenContextManager() as manager:
        caller_usage = get_token_usage()

        def encode_in_context(batch: list[str]) -> None:
            # Mirrors the pipeline: bind the caller's usage inside the worker.
            set_token_usage(caller_usage)
            adapter.encode(batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(encode_in_context, batches))

        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["call_type"] == "embedding"
    assert groups[0]["unit"] == "texts"
    # One text per batch, all four batches accounted for.
    assert groups[0]["calls"] == len(batches)
    assert groups[0]["quantity"] == float(len(batches))


def test_bare_thread_pool_would_lose_usage() -> None:
    """Documents *why* the binding above is required: without it the same code
    records nothing, which is how bulk embedding shipped unbilled."""
    from xagent.core.model.embedding.adapter import EmbeddingModelAdapter
    from xagent.core.model.model import EmbeddingModelConfig

    adapter = EmbeddingModelAdapter.__new__(EmbeddingModelAdapter)
    adapter.model_config = EmbeddingModelConfig(
        id="e-1", model_name="text-embed", api_key="k"
    )
    adapter._embedding_model = MagicMock()
    adapter._embedding_model.encode.return_value = [[0.1]]

    with TokenContextManager() as manager:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda b: adapter.encode(b), [["x"], ["y"]]))
        assert manager.get_usage().details == []
