from types import SimpleNamespace

import pytest

from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.services import memory_policy as policy_module
from xagent.web.services.memory_policy import (
    TrustedMemoryPolicyDecision,
    TrustedMemoryPolicyRequest,
    resolve_agent_service_memory_policy,
    set_trusted_memory_policy_resolver,
)


@pytest.fixture(autouse=True)
def clear_trusted_resolver():
    set_trusted_memory_policy_resolver(None)
    yield
    set_trusted_memory_policy_resolver(None)


def _task(
    *,
    agent_id: int | None = None,
    agent_config: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=41,
        user_id=7,
        agent_id=agent_id,
        source="internal",
        agent_config=agent_config,
    )


def test_default_preview_policy_stays_disabled_and_in_memory(monkeypatch):
    def unexpected_store_lookup():
        raise AssertionError("preview defaults must not load the persistent store")

    monkeypatch.setattr(policy_module, "get_memory_store", unexpected_store_lookup)

    policy = resolve_agent_service_memory_policy(
        task=_task(agent_config={"is_preview": True}),
    )

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.reason == "preview_memory_disabled"


def test_default_published_agent_policy_stays_disabled(monkeypatch):
    memory_store = object()
    monkeypatch.setattr(policy_module, "get_memory_store", lambda: memory_store)

    policy = resolve_agent_service_memory_policy(task=_task(agent_id=19))

    assert policy.memory is memory_store
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.reason == "published_agent_memory_disabled"


def test_default_unpublished_task_policy_stays_enabled(monkeypatch):
    memory_store = object()
    monkeypatch.setattr(policy_module, "get_memory_store", lambda: memory_store)

    policy = resolve_agent_service_memory_policy(task=_task())

    assert policy.memory is memory_store
    assert policy.memory_enabled is True
    assert policy.memory_available is True
    assert policy.reason is None


def test_trusted_resolver_can_enable_memory_and_receives_primitive_context(
    monkeypatch,
):
    memory_store = object()
    requests: list[TrustedMemoryPolicyRequest] = []
    monkeypatch.setattr(policy_module, "get_memory_store", lambda: memory_store)

    def resolve(request: TrustedMemoryPolicyRequest) -> TrustedMemoryPolicyDecision:
        requests.append(request)
        return TrustedMemoryPolicyDecision(
            enabled=True,
            available=True,
            reason="trusted_request_enabled",
        )

    set_trusted_memory_policy_resolver(resolve)

    policy = resolve_agent_service_memory_policy(task=_task(agent_id=19))

    assert policy.memory is memory_store
    assert policy.memory_enabled is True
    assert policy.memory_available is True
    assert policy.reason == "trusted_request_enabled"
    assert requests == [
        TrustedMemoryPolicyRequest(
            task_id=41,
            user_id=7,
            agent_id=19,
            source="internal",
            is_preview=False,
        )
    ]


def test_trusted_resolver_can_explicitly_disable_default_enabled_memory(monkeypatch):
    memory_store = object()
    monkeypatch.setattr(policy_module, "get_memory_store", lambda: memory_store)
    set_trusted_memory_policy_resolver(
        lambda request: TrustedMemoryPolicyDecision(
            enabled=False,
            available=True,
            reason="disabled_by_host_policy",
        )
    )

    policy = resolve_agent_service_memory_policy(task=_task())

    assert policy.memory is memory_store
    assert policy.memory_enabled is False
    assert policy.memory_available is True
    assert policy.reason == "disabled_by_host_policy"


@pytest.mark.parametrize(
    "decision",
    [
        None,
        TrustedMemoryPolicyDecision(
            enabled=1,  # type: ignore[arg-type]
            available=True,
            reason="wrong_enabled_type",
        ),
        TrustedMemoryPolicyDecision(
            enabled=True,
            available=False,
            reason="contradictory_availability",
        ),
        TrustedMemoryPolicyDecision(
            enabled=False,
            available=False,
            reason="  ",
        ),
    ],
)
def test_invalid_trusted_resolver_decisions_fail_closed(monkeypatch, decision):
    memory_store = object()
    monkeypatch.setattr(policy_module, "get_memory_store", lambda: memory_store)
    set_trusted_memory_policy_resolver(lambda request: decision)  # type: ignore[arg-type]

    policy = resolve_agent_service_memory_policy(task=_task())

    assert policy.memory is memory_store
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.reason == "invalid_trusted_resolver_decision"


def test_trusted_resolver_exception_fails_closed_without_logging_details(
    monkeypatch,
    caplog,
):
    memory_store = object()
    monkeypatch.setattr(policy_module, "get_memory_store", lambda: memory_store)

    class ResolverFailure(RuntimeError):
        pass

    def fail(request: TrustedMemoryPolicyRequest) -> TrustedMemoryPolicyDecision:
        raise ResolverFailure("host-private-detail")

    set_trusted_memory_policy_resolver(fail)

    policy = resolve_agent_service_memory_policy(task=_task())

    assert policy.memory is memory_store
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.reason == "trusted_resolver_failed"
    assert "ResolverFailure" in caplog.text
    assert "host-private-detail" not in caplog.text


def test_resolver_registration_rejects_non_callable():
    with pytest.raises(TypeError, match="resolver must be callable or None"):
        set_trusted_memory_policy_resolver(object())  # type: ignore[arg-type]
