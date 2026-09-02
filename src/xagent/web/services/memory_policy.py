"""Trusted host overrides for AgentService memory eligibility."""

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrustedMemoryPolicyRequest:
    """Stable request facts exposed to a trusted host policy resolver."""

    task_id: int | None
    user_id: int | None
    agent_id: int | None
    is_preview: bool
    default_enabled: bool


@dataclass(frozen=True)
class TrustedMemoryPolicyDecision:
    """A host's explicit memory eligibility and backend availability decision."""

    enabled: bool
    available: bool
    reason: str | None = None


TrustedMemoryPolicyResolver = Callable[
    [TrustedMemoryPolicyRequest], TrustedMemoryPolicyDecision
]

_trusted_memory_policy_resolver: TrustedMemoryPolicyResolver | None = None


def set_trusted_memory_policy_resolver(
    resolver: TrustedMemoryPolicyResolver | None,
) -> None:
    """Install a process-level resolver owned by the trusted host service.

    Passing ``None`` restores XAgent's default preview and published-agent
    policy. The hook is intended for startup-time registration, before the
    process begins serving requests.
    """

    global _trusted_memory_policy_resolver
    if resolver is not None and not callable(resolver):
        raise TypeError("resolver must be callable or None")
    _trusted_memory_policy_resolver = resolver


def resolve_trusted_memory_policy(
    request: TrustedMemoryPolicyRequest,
) -> TrustedMemoryPolicyDecision | None:
    """Return a validated override, or ``None`` when no hook is installed.

    A registered hook is authoritative. Resolver failures and malformed
    decisions therefore become an unavailable, disabled decision instead of
    falling back to the more permissive default policy.
    """

    resolver = _trusted_memory_policy_resolver
    if resolver is None:
        return None

    try:
        decision = resolver(request)
    except Exception:
        logger.exception("Trusted memory policy resolver failed")
        return _failed_decision("resolver_error")

    if not _is_valid_decision(decision):
        logger.error("Trusted memory policy resolver returned an invalid decision")
        return _failed_decision("invalid_decision")
    return decision


def _is_valid_decision(decision: object) -> bool:
    if not isinstance(decision, TrustedMemoryPolicyDecision):
        return False
    if type(decision.enabled) is not bool or type(decision.available) is not bool:
        return False
    if decision.reason is not None and (
        not isinstance(decision.reason, str) or not decision.reason.strip()
    ):
        return False
    return decision.available or decision.reason is not None


def _failed_decision(reason: str) -> TrustedMemoryPolicyDecision:
    return TrustedMemoryPolicyDecision(enabled=False, available=False, reason=reason)
