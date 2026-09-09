"""Tests for DynamicMemoryStoreManager embedding-config change detection."""

from types import SimpleNamespace
from typing import Any

from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager


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
        dimension=1024,
    )


def test_request_acquisition_does_not_rotate_shared_store(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "old-key")}
    manager = _manager_with_fake_db(monkeypatch, holder)
    shared = FakeLanceStore(holder["model"])
    manager._memory_store = shared
    manager._is_lancedb = True
    manager._last_embedding_model_fingerprint = (2, "2026-07-17 10:00:00")

    first = manager.get_memory_store()
    holder["model"] = _model(2, "2026-07-17 11:00:00", "new-key")
    second = manager.get_memory_store()
    assert first is second is shared
    assert shared.model.api_key == "old-key"


def test_unchanged_model_keeps_store_instance(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    manager.check_embedding_model_change()
    first = manager.get_memory_store()
    assert manager.get_memory_store() is first


def test_explicit_configuration_check_reads_under_lock(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    def get_model_under_lock() -> Any:
        assert manager._lock._is_owned()  # type: ignore[attr-defined]
        return holder["model"]

    monkeypatch.setattr(manager, "_get_embedding_model_from_db", get_model_under_lock)

    assert manager.check_embedding_model_change() is True
