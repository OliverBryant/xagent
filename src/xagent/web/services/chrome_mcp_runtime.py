"""Narrow authorization seam for the future Chrome actor runtime.

This module does not build an execution key and does not load Chrome tools.
The connection identity and the exact task/run/turn/lease-attempt identity
must both be supplied by the later integration before session reuse is wired.
"""

from __future__ import annotations

from .mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)


def require_chrome_builtin_stdio_policy(
    policy: MCPActorAuthorizationPolicy | None,
) -> MCPActorAuthorizationPolicy:
    """Require an explicit actor capability before any Chrome stdio work.

    Returning the same immutable policy lets the integration compare its
    resource owner with the actor connection identity.  This is only an
    authorization gate; passing it is insufficient to identify a browser
    session and must never be used as a session key.
    """

    if policy is None or not policy.allow_builtin_stdio:
        raise MCPBuiltinOAuthActorPolicyRequiredError(
            "Chrome actor stdio execution requires an explicit capability"
        )
    return policy
