"""Tests for DynamicMemoryStoreManager embedding-config change detection."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web import dynamic_memory_store as memory_store_module
from xagent.web.dynamic_memory_store import (
    DynamicMemoryStoreManager,
    MemoryBackendUnavailableError,
)
from xagent.web.user_isolated_memory import UserIsolatedMemoryStore


class FakeLanceStore:
    def __init__(self, model: Any) -> None:
        self.model = model


def _manager_with_fake_db(monkeypatch, model_holder: dict) -> DynamicMemoryStoreManager:
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(
        manager, "_get_embedding_model_from_db", lambda: model_holder["model"]
    )
    monkeypatch.setattr(
        manager,
        "_create_lancedb_store",
        lambda model: FakeLanceStore(model),
    )
    return manager


def _model(model_id: int, updated_at: str, api_key: str) -> Any:
    return SimpleNamespace(
        id=model_id,
        updated_at=updated_at,
        api_key=api_key,
        model_provider="dashscope",
        model_id="configured-embedding",
        model_name="text-embedding-v4",
        base_url="https://embedding.example/v1",
        dimension=1024,
    )


def test_key_rotation_on_same_model_rebuilds_store(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "old-key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    first = manager.get_memory_store()
    assert isinstance(first, FakeLanceStore)
    assert first.model.api_key == "old-key"

    # Same model id, but the row was edited (key rotation bumps updated_at).
    holder["model"] = _model(2, "2026-07-17 11:00:00", "new-key")
    assert manager.check_embedding_model_change() is True
    second = manager.get_memory_store()
    assert isinstance(second, FakeLanceStore)
    assert second.model.api_key == "new-key"
    assert second is not first


def test_unchanged_model_keeps_store_instance(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    first = manager.get_memory_store()
    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is first


def test_embedding_configuration_read_happens_under_lock(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    def get_model_under_lock() -> Any:
        assert manager._lock._is_owned()  # type: ignore[attr-defined]
        return holder["model"]

    monkeypatch.setattr(manager, "_get_embedding_model_from_db", get_model_under_lock)

    assert isinstance(manager.get_memory_store(), FakeLanceStore)


def test_persistence_can_be_required_without_vector_search(monkeypatch) -> None:
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda: None)
    persistent_store = FakeLanceStore(None)
    create_store = Mock(return_value=persistent_store)
    monkeypatch.setattr(manager, "_create_lancedb_store", create_store)

    assert manager.get_memory_store(require_persistence=True) is persistent_store
    create_store.assert_called_once_with(None)
    assert manager.get_store_info()["supports_vector_search"] is False


def test_vector_search_requirement_fails_without_embedding_model(monkeypatch) -> None:
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda: None)

    with pytest.raises(MemoryBackendUnavailableError):
        manager.get_memory_store(require_vector_search=True)

    assert isinstance(manager.get_memory_store(), UserIsolatedMemoryStore)
    assert isinstance(manager.get_memory_store()._base_store, InMemoryMemoryStore)


def test_vector_search_requirement_succeeds_with_embedding_model(monkeypatch) -> None:
    model = _model(2, "2026-07-17 10:00:00", "key")
    manager = _manager_with_fake_db(monkeypatch, {"model": model})

    store = manager.get_memory_store(require_vector_search=True)

    assert isinstance(store, FakeLanceStore)
    assert store.model is model
    assert manager.get_store_info()["supports_vector_search"] is True


def test_strict_creation_failure_is_reported_but_default_falls_back(
    monkeypatch,
) -> None:
    manager = DynamicMemoryStoreManager()
    model = _model(2, "2026-07-17 10:00:00", "key")
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda: model)
    monkeypatch.setattr(
        manager,
        "_create_lancedb_store",
        Mock(side_effect=RuntimeError("backend down")),
    )

    with pytest.raises(MemoryBackendUnavailableError) as exc_info:
        manager.get_memory_store(require_vector_search=True)
    assert isinstance(exc_info.value.__cause__, RuntimeError)

    store = manager.get_memory_store()
    assert isinstance(store, UserIsolatedMemoryStore)
    assert isinstance(store._base_store, InMemoryMemoryStore)


def test_strict_model_lookup_exception_fails_closed(monkeypatch) -> None:
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(
        manager,
        "_get_embedding_model_from_db",
        Mock(side_effect=RuntimeError("model store down")),
    )

    with pytest.raises(MemoryBackendUnavailableError) as exc_info:
        manager.get_memory_store(require_persistence=True)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_swallowed_model_database_failure_still_fails_closed_in_strict_mode(
    monkeypatch,
) -> None:
    manager = DynamicMemoryStoreManager()

    def broken_db():
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(memory_store_module, "get_db", broken_db)

    with pytest.raises(MemoryBackendUnavailableError):
        manager.get_memory_store(require_persistence=True)


def test_database_embedding_configuration_is_propagated(monkeypatch, tmp_path) -> None:
    manager = DynamicMemoryStoreManager()
    model = _model(7, "2026-07-17 10:00:00", "secret")
    adapter = Mock(name="embedding-adapter")
    create_adapter = Mock(return_value=adapter)
    lancedb_store = Mock(name="lancedb-store")
    monkeypatch.setattr(memory_store_module, "create_embedding_adapter", create_adapter)
    monkeypatch.setattr(memory_store_module, "LanceDBMemoryStore", lancedb_store)
    monkeypatch.setattr(memory_store_module, "get_storage_root", lambda: tmp_path)

    wrapped = manager._create_lancedb_store(model)

    config = create_adapter.call_args.args[0]
    assert config.id == "configured-embedding"
    assert config.model_name == "text-embedding-v4"
    assert config.model_provider == "dashscope"
    assert config.api_key == "secret"
    assert config.base_url == "https://embedding.example/v1"
    assert config.dimension == 1024
    assert isinstance(wrapped, UserIsolatedMemoryStore)
    assert wrapped._base_store is lancedb_store.return_value
