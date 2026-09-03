from unittest.mock import Mock

import pandas as pd
import pytest

from xagent.core.memory.base import MemoryBackendUnavailableError
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import (
    LanceDBMemoryStore,
    _VECTOR_SPACE_METADATA_KEY,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

from .conftest import ConstantEmbedding


class FailingEmbedding(ConstantEmbedding):
    def encode(self, text, dimension=None, instruct=None):
        raise RuntimeError("embedding unavailable")


def _store(tmp_path, embedding, identity):
    return LanceDBMemoryStore(
        db_dir=str(tmp_path),
        collection_name="required",
        embedding_model=embedding,
        vector_space_identity=identity,
    )


def _schema(store):
    table = store._vector_store.get_raw_connection().open_table("required")
    try:
        return table.schema
    finally:
        _safe_close_table(table)


def test_strict_initialization_add_and_search_succeed(tmp_path) -> None:
    store = _store(tmp_path, ConstantEmbedding(8), "provider/model/a")

    store.ensure_required_vector_search()
    added = store.add_required_vector(MemoryNote(content="alpha"))

    assert added.success
    assert [note.content for note in store.search_required_vector("alpha")] == ["alpha"]
    assert _schema(store).metadata[_VECTOR_SPACE_METADATA_KEY] == b"provider/model/a"


def test_strict_embedding_failure_does_not_use_text_fallback(tmp_path) -> None:
    store = _store(tmp_path, FailingEmbedding(8), "provider/model/a")

    with pytest.raises(MemoryBackendUnavailableError):
        store.ensure_required_vector_search()
    with pytest.raises(MemoryBackendUnavailableError):
        store.add_required_vector(MemoryNote(content="alpha"))

    assert store.add(MemoryNote(content="alpha")).success
    assert [note.content for note in store.search("alpha")] == ["alpha"]


def test_strict_search_rejects_missing_vector_column(tmp_path) -> None:
    store = _store(tmp_path, ConstantEmbedding(8), "provider/model/a")
    store.ensure_required_vector_search()
    store.ensure_text_persistence()

    with pytest.raises(MemoryBackendUnavailableError):
        store.search_required_vector("alpha")


def test_strict_ann_failure_raises_while_default_falls_back(tmp_path) -> None:
    store = _store(tmp_path, ConstantEmbedding(8), "provider/model/a")
    store.ensure_required_vector_search()
    metadata = '{"content":"alpha","timestamp":"2026-01-01T00:00:00Z"}'
    table = Mock()
    table.schema = _schema(store)
    vector_query = Mock()
    vector_query.limit.return_value.to_pandas.side_effect = RuntimeError("ANN failed")
    text_query = Mock()
    text_query.to_pandas.return_value = pd.DataFrame(
        [{"id": "a", "text": "alpha", "metadata": metadata}]
    )
    table.search.side_effect = lambda vector=None, **_kwargs: (
        vector_query if vector is not None else text_query
    )
    connection = Mock()
    connection.open_table.return_value = table
    store._vector_store.get_raw_connection = Mock(return_value=connection)

    with pytest.raises(MemoryBackendUnavailableError):
        store.search_required_vector("alpha")
    assert [note.content for note in store.search("alpha")] == ["alpha"]


@pytest.mark.parametrize(
    ("old_dim", "new_dim"),
    [(8, 8), (8, 12)],
    ids=("same-dimension-model-switch", "dimension-change"),
)
def test_vector_identity_change_reembeds_existing_rows_before_write(
    tmp_path, old_dim, new_dim
) -> None:
    first = _store(tmp_path, ConstantEmbedding(old_dim, 0.1), "provider/model/a")
    first.ensure_required_vector_search()
    added = first.add_required_vector(MemoryNote(content="alpha"))
    assert added.success

    restarted = _store(tmp_path, ConstantEmbedding(new_dim, 0.9), "provider/model/b")
    restarted.ensure_required_vector_search()

    table = restarted._vector_store.get_raw_connection().open_table("required")
    try:
        arrow = table.to_arrow()
    finally:
        _safe_close_table(table)
    assert arrow.num_rows == 1
    assert arrow.column("vector").to_pylist() == [[pytest.approx(0.9)] * new_dim]
    assert arrow.schema.metadata[_VECTOR_SPACE_METADATA_KEY] == b"provider/model/b"
    assert restarted.get(added.memory_id).success


def test_failed_identity_migration_preserves_old_rows_and_identity(tmp_path) -> None:
    first = _store(tmp_path, ConstantEmbedding(8), "provider/model/a")
    first.ensure_required_vector_search()
    added = first.add_required_vector(MemoryNote(content="alpha"))

    failed = _store(tmp_path, FailingEmbedding(12), "provider/model/b")
    with pytest.raises(MemoryBackendUnavailableError):
        failed.ensure_required_vector_search()

    assert first.get(added.memory_id).success
    assert _schema(first).metadata[_VECTOR_SPACE_METADATA_KEY] == b"provider/model/a"
