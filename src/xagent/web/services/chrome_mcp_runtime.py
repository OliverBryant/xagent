"""Authorization, identity, and dedicated-sandbox binding for Chrome MCP."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from ...core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_SANDBOX_LIFECYCLE_TYPE,
    ChromeExecutionScope,
    ChromeExecutionSessionPool,
    ChromeSandboxHandle,
    ChromeSessionContractError,
)
from ..builtin_mcp_registry import get_builtin_stdio_session_scope
from ..sandbox_manager import SandboxCapacityError, get_sandbox_manager
from .actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
)
from .mcp_runtime import (
    CALLER_ID_ENV_VAR,
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
    MCPBuiltinOAuthActorPolicyRequiredError,
)

_ACTOR_STDIO_SESSION_IDENTITY_KEY = "actor_stdio_session_identity"
_CHROME_SCOPE_HASH_DOMAIN = b"xagent.chrome.execution-session.v1\x00"
_chrome_pool_manager: object | None = None
_chrome_pool: ChromeExecutionSessionPool | None = None


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


def _hash_identity_key(key: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(_CHROME_SCOPE_HASH_DOMAIN)
    for value in key:
        if isinstance(value, bool):
            raise ChromeSessionContractError("Chrome session identity is invalid")
        if isinstance(value, int):
            encoded = f"i:{value}".encode()
        elif isinstance(value, str):
            encoded = b"s:" + value.encode("utf-8")
        elif isinstance(value, UUID):
            encoded = b"u:" + value.bytes
        else:
            raise ChromeSessionContractError("Chrome session identity is invalid")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def consume_chrome_execution_scope(
    server_name: str,
    connection: Mapping[str, Any],
) -> tuple[ChromeExecutionScope | None, Mapping[str, Any]]:
    """Consume the host-only identity before returning an executable config."""

    execution_scoped = get_builtin_stdio_session_scope(server_name) == "execution"
    if not execution_scoped:
        if _ACTOR_STDIO_SESSION_IDENTITY_KEY in connection:
            raise ChromeSessionContractError(
                "non-execution MCP connection supplied a session identity"
            )
        return None, connection
    executable_connection = dict(connection)
    identity = executable_connection.pop(_ACTOR_STDIO_SESSION_IDENTITY_KEY, None)
    if type(identity) is not ActorMCPStdioSessionIdentity:
        raise ChromeSessionContractError(
            "execution-scoped Chrome requires an exact session identity"
        )
    if (
        type(identity.execution) is not MCPActorExecutionIdentity
        or type(identity.connection) is not ActorMCPStdioConnectionIdentity
        or identity.connection.app_id != server_name
    ):
        raise ChromeSessionContractError("Chrome session identity does not match")
    env = executable_connection.get("env")
    if not isinstance(env, Mapping) or env.get(CALLER_ID_ENV_VAR) != str(
        identity.connection.user_id
    ):
        raise ChromeSessionContractError(
            "Chrome session caller identity does not match"
        )
    key = identity.key
    return (
        ChromeExecutionScope(key=key, digest=_hash_identity_key(key)),
        executable_connection,
    )


async def _create_chrome_sandbox(scope_digest: str) -> ChromeSandboxHandle:
    """Create and actively pin one sandbox whose name contains only a digest."""

    manager = get_sandbox_manager()
    if manager is None:
        raise ChromeSessionContractError("Chrome sandbox is unavailable")
    try:
        provider = await manager.get_or_create_lease_provider(
            CHROME_SANDBOX_LIFECYCLE_TYPE,
            scope_digest,
        )
        attached = await manager.attach_provider(
            CHROME_SANDBOX_LIFECYCLE_TYPE,
            scope_digest,
            provider,
        )
    except SandboxCapacityError as exc:
        raise ChromeSessionContractError("Chrome sandbox is unavailable") from exc
    except Exception as exc:
        raise ChromeSessionContractError("Chrome sandbox creation failed") from exc
    if not attached:
        raise ChromeSessionContractError("Chrome sandbox attachment failed")

    async def delete() -> None:
        await manager.delete_sandbox(CHROME_SANDBOX_LIFECYCLE_TYPE, scope_digest)

    return ChromeSandboxHandle(sandbox=provider.primary_sandbox, delete=delete)


def get_chrome_execution_session_pool() -> ChromeExecutionSessionPool:
    """Return the process-local coordinator for sandbox-owned Chrome sessions."""

    global _chrome_pool, _chrome_pool_manager
    manager = get_sandbox_manager()
    if manager is None:
        raise ChromeSessionContractError("Chrome sandbox is unavailable")
    if _chrome_pool is None or _chrome_pool_manager is not manager:
        _chrome_pool = ChromeExecutionSessionPool(_create_chrome_sandbox)
        _chrome_pool_manager = manager
    return _chrome_pool
