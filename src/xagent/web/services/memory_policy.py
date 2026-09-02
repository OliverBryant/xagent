"""Trusted host overrides for task memory eligibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ...core.memory.base import MemoryStore
from ...core.memory.in_memory import InMemoryMemoryStore
from ..dynamic_memory_store import get_memory_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrustedMemoryPolicyRequest:
    """Primitive task context exposed to an embedding application's resolver."""

    task_id: int | None
    user_id: int | None
    agent_id: int | None
    source: str | None
    is_preview: bool


@dataclass(frozen=True)
class TrustedMemoryPolicyDecision:
    """A trusted host's explicit memory policy decision."""

    enabled: bool
    available: bool
    reason: str


TrustedMemoryPolicyResolver = Callable[
    [TrustedMemoryPolicyRequest], TrustedMemoryPolicyDecision
]


@dataclass(frozen=True)
class AgentServiceMemoryPolicy:
    """Resolved memory store and eligibility for an agent runtime."""

    memory: MemoryStore
    memory_enabled: bool
    memory_available: bool
    reason: str | None = None


_trusted_memory_policy_resolver: TrustedMemoryPolicyResolver | None = None


def set_trusted_memory_policy_resolver(
    resolver: TrustedMemoryPolicyResolver | None,
) -> None:
    """Install or clear the process-wide trusted memory policy resolver.

    The embedding application is responsible for authenticating and validating
    requests before they reach this hook. XAgent only exposes primitive task
    metadata and validates the resolver's decision before applying it.
    """

    global _trusted_memory_policy_resolver
    if resolver is not None and not callable(resolver):
        raise TypeError("resolver must be callable or None")
    _trusted_memory_policy_resolver = resolver


def _optional_exact_int(value: object) -> int | None:
    return value if type(value) is int else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _default_memory_policy(
    *,
    task: Any | None,
    config: Mapping[str, Any],
) -> AgentServiceMemoryPolicy:
    if config.get("is_preview") is True:
        return AgentServiceMemoryPolicy(
            memory=InMemoryMemoryStore(),
            memory_enabled=False,
            memory_available=False,
            reason="preview_memory_disabled",
        )

    if task is not None and getattr(task, "agent_id", None):
        return AgentServiceMemoryPolicy(
            memory=get_memory_store(),
            memory_enabled=False,
            memory_available=False,
            reason="published_agent_memory_disabled",
        )

    return AgentServiceMemoryPolicy(
        memory=get_memory_store(),
        memory_enabled=True,
        memory_available=True,
    )


def _resolver_request(
    *,
    task: Any | None,
    config: Mapping[str, Any],
) -> TrustedMemoryPolicyRequest:
    return TrustedMemoryPolicyRequest(
        task_id=_optional_exact_int(getattr(task, "id", None)),
        user_id=_optional_exact_int(getattr(task, "user_id", None)),
        agent_id=_optional_exact_int(getattr(task, "agent_id", None)),
        source=_optional_str(getattr(task, "source", None)),
        is_preview=config.get("is_preview") is True,
    )


def _fail_closed(
    default: AgentServiceMemoryPolicy,
    *,
    reason: str,
) -> AgentServiceMemoryPolicy:
    return AgentServiceMemoryPolicy(
        memory=default.memory,
        memory_enabled=False,
        memory_available=False,
        reason=reason,
    )


def _validated_decision(
    decision: object,
) -> tuple[bool, bool, str] | None:
    if not isinstance(decision, TrustedMemoryPolicyDecision):
        return None
    if type(decision.enabled) is not bool or type(decision.available) is not bool:
        return None
    if not isinstance(decision.reason, str) or not decision.reason.strip():
        return None
    if decision.enabled and not decision.available:
        return None
    return decision.enabled, decision.available, decision.reason.strip()


def resolve_agent_service_memory_policy(
    *,
    task: Any | None = None,
    agent_config: Mapping[str, Any] | None = None,
) -> AgentServiceMemoryPolicy:
    """Resolve the default policy, then apply a trusted host override if set."""

    config = agent_config
    if config is None:
        task_config = getattr(task, "agent_config", None)
        config = task_config if isinstance(task_config, Mapping) else {}

    default = _default_memory_policy(task=task, config=config)
    resolver = _trusted_memory_policy_resolver
    if resolver is None:
        return default

    try:
        decision = resolver(_resolver_request(task=task, config=config))
    except Exception as exc:
        logger.warning(
            "Trusted memory policy resolver failed (%s)",
            type(exc).__name__,
        )
        return _fail_closed(default, reason="trusted_resolver_failed")

    validated = _validated_decision(decision)
    if validated is None:
        logger.warning("Trusted memory policy resolver returned an invalid decision")
        return _fail_closed(default, reason="invalid_trusted_resolver_decision")

    enabled, available, reason = validated
    memory = default.memory
    if enabled and isinstance(memory, InMemoryMemoryStore):
        memory = get_memory_store()
    return AgentServiceMemoryPolicy(
        memory=memory,
        memory_enabled=enabled,
        memory_available=available,
        reason=reason,
    )
