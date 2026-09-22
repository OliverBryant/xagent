"""DynamicMemoryStoreManager: what a configuration change must *not* do.

Before the lifecycle redesign this manager rebuilt its store online whenever the
embedding configuration moved. It no longer does: the runtime consumes the
explicit global authority, and every change to it takes effect through
quiescence and an all-worker restart. These tests pin the inversion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from pydantic import SecretStr

from xagent.web import dynamic_memory_store as manager_module
from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager
from xagent.web.memory_lifecycle import (
    AdmissionResult,
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.services.global_memory_embedding_authority import (
    CREDENTIAL_CONFIGURED,
    CredentialSource,
    GlobalMemoryEmbeddingAuthoritySnapshot,
)


class FakeLanceStore:
    def __init__(self, snapshot: Any) -> None:
        self.snapshot = snapshot


def _snapshot(
    *, model_name: str = "text-embedding-3-small", api_key: str = "key"
) -> GlobalMemoryEmbeddingAuthoritySnapshot:
    now = datetime.now(timezone.utc)
    return GlobalMemoryEmbeddingAuthoritySnapshot(
        provider="openai",
        model_name=model_name,
        endpoint="https://api.openai.com/v1/embeddings",
        dimension=1024,
        instruct=None,
        max_retries=3,
        credential_source=CredentialSource.ORGANIZATION_OWNED,
        global_sharing_consent=True,
        consented_by_actor_subject="admin",
        consented_at=now,
        created_at=now,
        updated_at=now,
        credential_status=CREDENTIAL_CONFIGURED,
        api_key=SecretStr(api_key),
        credential_identity=f"identity-for-{api_key}",
    )


def _manager_with_fake_authority(
    monkeypatch, holder: dict
) -> DynamicMemoryStoreManager:
    """A manager whose authority read and admission are both fakes."""

    def read() -> Optional[GlobalMemoryEmbeddingAuthoritySnapshot]:
        return holder["snapshot"]

    def admit(snapshot, **_kwargs) -> AdmissionResult:
        return AdmissionResult(
            MemoryLifecycleStatus(MemoryLifecycleState.READY),
            store=FakeLanceStore(snapshot),
            vector_space_fingerprint=snapshot.vector_space_fingerprint(),
        )

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", read)
    monkeypatch.setattr(manager_module, "admit_authority_storage", admit)
    manager = DynamicMemoryStoreManager()
    # Admission is startup-only; nothing publishes a store lazily any more.
    manager.admit()
    return manager


def test_key_rotation_on_the_same_authority_does_not_rebuild(monkeypatch) -> None:
    """Rotating the credential is not a vector-space change, and never reloads."""
    holder = {"snapshot": _snapshot(api_key="old-key")}
    manager = _manager_with_fake_authority(monkeypatch, holder)

    first = manager.get_memory_store()
    assert isinstance(first, FakeLanceStore)
    assert first.snapshot.api_key.get_secret_value() == "old-key"

    holder["snapshot"] = _snapshot(api_key="new-key")

    assert manager.check_embedding_model_change() is False
    second = manager.get_memory_store()
    assert second is first
    assert second.snapshot.api_key.get_secret_value() == "old-key"


def test_changed_vector_space_revokes_instead_of_rebuilding(monkeypatch) -> None:
    holder = {"snapshot": _snapshot()}
    manager = _manager_with_fake_authority(monkeypatch, holder)
    assert isinstance(manager.get_memory_store(), FakeLanceStore)

    holder["snapshot"] = _snapshot(model_name="text-embedding-3-large")

    assert manager.check_embedding_model_change() is True
    with pytest.raises(MemoryUnavailableError) as raised:
        manager.get_memory_store()
    assert raised.value.status.state is MemoryLifecycleState.RESTART_REQUIRED


def test_unchanged_authority_keeps_store_instance(monkeypatch) -> None:
    holder = {"snapshot": _snapshot()}
    manager = _manager_with_fake_authority(monkeypatch, holder)

    first = manager.get_memory_store()
    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is first


def test_authority_read_happens_under_the_lock(monkeypatch) -> None:
    holder = {"snapshot": _snapshot()}
    manager = _manager_with_fake_authority(monkeypatch, holder)

    def read_under_lock() -> GlobalMemoryEmbeddingAuthoritySnapshot:
        assert manager._lock._is_owned()  # type: ignore[attr-defined]
        return holder["snapshot"]

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", read_under_lock)

    assert isinstance(manager.get_memory_store(), FakeLanceStore)
