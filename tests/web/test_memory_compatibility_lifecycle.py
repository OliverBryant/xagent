from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.lancedb_maintenance import (
    MaintenanceOutcome,
    MaintenanceStatus,
)
from xagent.core.memory.vector_compatibility import VectorCompatibility
from xagent.core.model import EmbeddingModelConfig
from xagent.web import dynamic_memory_store as memory_module
from xagent.web.dynamic_memory_store import (
    DynamicMemoryStoreManager,
    _embedding_model_config,
)
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


def _model(**overrides):
    values = {
        "id": 7,
        "model_id": "shared-embedding",
        "model_provider": "dashscope",
        "model_name": "text-embedding-v4",
        "api_key": "shared-secret",
        "base_url": "https://embedding.example/v1",
        "dimension": 768,
        "instruct": "retrieval.document",
        "max_retries": 4,
        "updated_at": "2026-09-09T00:00:00Z",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _store(connection, embedding=object()):
    base = LanceDBMemoryStore.__new__(LanceDBMemoryStore)
    base._collection_name = "memories"
    base._embedding_model = embedding
    base._vector_store = SimpleNamespace(get_raw_connection=lambda: connection)
    return UserIsolatedMemoryStore(base), base


def test_shared_config_and_store_are_stable_across_request_users(monkeypatch):
    config = _embedding_model_config(_model())
    assert config == EmbeddingModelConfig(
        id="shared-embedding",
        model_provider="dashscope",
        model_name="text-embedding-v4",
        api_key="shared-secret",
        base_url="https://embedding.example/v1",
        dimension=768,
        instruct="retrieval.document",
        max_retries=4,
    )

    manager = DynamicMemoryStoreManager()
    shared = manager._memory_store
    monkeypatch.setattr(
        manager,
        "_get_embedding_model_from_db",
        lambda: pytest.fail("request acquisition resolved an embedding identity"),
    )
    with UserContext(101):
        first = manager.get_memory_store()
    with UserContext(202):
        second = manager.get_memory_store()
    assert first is second is shared


def test_manager_constructs_dormant_store_with_complete_shared_config(
    monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(memory_module, "get_storage_root", lambda: tmp_path)
    monkeypatch.setattr(
        memory_module,
        "LanceDBMemoryStore",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    DynamicMemoryStoreManager()._create_lancedb_store(_model())

    assert captured["initialize_schema"] is False
    assert captured["include_null_vector_fallback"] is True
    assert captured["embedding_model"] == _embedding_model_config(_model())


def test_startup_lifecycle_unwraps_serializes_and_orders_primitives(monkeypatch):
    events = []

    class Connection:
        def open_table(self, name):
            events.append(("open", name))
            return SimpleNamespace(close=lambda: None)

    wrapper, _base = _store(Connection())
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: _model())
    monkeypatch.setattr(manager, "_create_lancedb_store", lambda _model: wrapper)

    def maintain(connection, name):
        assert manager._lock._is_owned()  # type: ignore[attr-defined]
        events.append(("maintain", connection, name))
        return MaintenanceOutcome(MaintenanceStatus.COMPLETE)

    def admit(connection, name, identity):
        events.append(("admit", connection, name, identity))
        return VectorCompatibility.MATCHING

    monkeypatch.setattr(memory_module, "maintain_lancedb_memory_table", maintain)
    monkeypatch.setattr(memory_module, "create_or_recreate_vector_capable_table", admit)

    manager.run_startup_compatibility_lifecycle()

    assert [event[0] for event in events] == ["open", "maintain", "admit"]
    assert events[1][1] is events[2][1]
    assert events[2][3].model_name == "text-embedding-v4"
    assert manager._memory_store is wrapper


def test_malformed_legacy_data_admits_text_only_without_recreation(monkeypatch):
    connection = SimpleNamespace(
        open_table=lambda _name: SimpleNamespace(close=lambda: None)
    )
    wrapper, base = _store(connection)
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: _model())
    monkeypatch.setattr(manager, "_create_lancedb_store", lambda _model: wrapper)
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: MaintenanceOutcome(MaintenanceStatus.INVALID_LEGACY_DATA),
    )
    recreate = Mock()
    monkeypatch.setattr(
        memory_module, "create_or_recreate_vector_capable_table", recreate
    )

    manager.run_startup_compatibility_lifecycle()

    assert base._embedding_model is None
    assert manager._memory_store is wrapper
    assert manager._is_lancedb is True
    recreate.assert_not_called()


def test_missing_table_is_recreated_then_maintained(monkeypatch):
    events = []

    class Connection:
        def open_table(self, _name):
            raise ValueError("Table 'memories' was not found")

    wrapper, _base = _store(Connection())
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: _model())
    monkeypatch.setattr(manager, "_create_lancedb_store", lambda _model: wrapper)
    monkeypatch.setattr(
        memory_module,
        "create_or_recreate_vector_capable_table",
        lambda *_args: events.append("recreate") or VectorCompatibility.MATCHING,
    )
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: events.append("maintain")
        or MaintenanceOutcome(MaintenanceStatus.COMPLETE),
    )

    manager.run_startup_compatibility_lifecycle()

    assert events == ["recreate", "maintain"]


def test_mismatching_vector_space_admits_text_only(monkeypatch):
    connection = SimpleNamespace(
        open_table=lambda _name: SimpleNamespace(close=lambda: None)
    )
    wrapper, base = _store(connection)
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: _model())
    monkeypatch.setattr(manager, "_create_lancedb_store", lambda _model: wrapper)
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: MaintenanceOutcome(MaintenanceStatus.COMPLETE),
    )
    monkeypatch.setattr(
        memory_module,
        "create_or_recreate_vector_capable_table",
        lambda *_args: VectorCompatibility.MISMATCHING,
    )

    manager.run_startup_compatibility_lifecycle()

    assert base._embedding_model is None


def test_real_maintenance_failure_propagates_and_preserves_manager_state(monkeypatch):
    connection = SimpleNamespace(
        open_table=lambda _name: SimpleNamespace(close=lambda: None)
    )
    wrapper, _base = _store(connection)
    manager = DynamicMemoryStoreManager()
    previous = manager._memory_store
    manager._is_lancedb = True
    manager._last_embedding_model_id = 3
    manager._last_embedding_model_fingerprint = (3, "previous")
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda **_: _model())
    monkeypatch.setattr(manager, "_create_lancedb_store", lambda _model: wrapper)
    monkeypatch.setattr(
        memory_module,
        "maintain_lancedb_memory_table",
        lambda *_args: (_ for _ in ()).throw(OSError("real I/O failure")),
    )

    with pytest.raises(OSError, match="real I/O failure"):
        manager.run_startup_compatibility_lifecycle()

    assert manager._memory_store is previous
    assert manager._is_lancedb is True
    assert manager._last_embedding_model_id == 3
    assert manager._last_embedding_model_fingerprint == (3, "previous")
