"""Tests for DynamicMemoryStoreManager embedding-config change detection."""

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.model import EmbeddingModelConfig
from xagent.web import dynamic_memory_store
from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager


class FakeLanceStore:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.maintenance_calls = 0

    def maintain_schema(self) -> None:
        self.maintenance_calls += 1


def _manager_with_fake_db(monkeypatch, model_holder: dict) -> DynamicMemoryStoreManager:
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(
        manager, "_get_embedding_model_from_db", lambda: model_holder["model"]
    )
    monkeypatch.setattr(
        manager,
        "_create_lancedb_store",
        lambda model, **_kwargs: FakeLanceStore(model),
    )
    return manager


def _model(model_id: int, updated_at: str, api_key: str) -> Any:
    return SimpleNamespace(
        id=model_id,
        updated_at=updated_at,
        api_key=api_key,
        model_id=f"embedding-{model_id}",
        model_provider="dashscope",
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


def test_startup_maintenance_reaches_the_persistent_store(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)
    monkeypatch.setattr(dynamic_memory_store, "LanceDBMemoryStore", FakeLanceStore)

    manager.maintain_schema()

    assert manager._memory_store.maintenance_calls == 1


def test_startup_maintenance_propagates_store_creation_failure(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    def fail(_model, *, fallback_on_error=True):
        assert not fallback_on_error
        raise RuntimeError("memory table unavailable")

    monkeypatch.setattr(manager, "_create_lancedb_store", fail)

    with pytest.raises(RuntimeError, match="memory table unavailable"):
        manager.maintain_schema()


def test_dynamic_store_preserves_complete_embedding_config(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def build_store(**kwargs):
        captured.update(kwargs)
        return FakeLanceStore(kwargs["embedding_model"])

    monkeypatch.setattr(dynamic_memory_store, "get_storage_root", lambda: tmp_path)
    monkeypatch.setattr(dynamic_memory_store, "LanceDBMemoryStore", build_store)
    monkeypatch.setattr(dynamic_memory_store.os.path, "exists", lambda _path: False)
    manager = DynamicMemoryStoreManager()

    manager._create_lancedb_store(_model(7, "2026-07-17 10:00:00", "key"))

    config = captured["embedding_model"]
    assert isinstance(config, EmbeddingModelConfig)
    assert config.model_name == "text-embedding-v4"
    assert config.base_url == "https://embedding.example/v1"
