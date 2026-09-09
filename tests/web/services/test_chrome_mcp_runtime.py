from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    ChromeSessionContractError,
)
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
)
from xagent.web.services.chrome_mcp_runtime import (
    _create_chrome_sandbox,
    consume_chrome_execution_scope,
    require_chrome_builtin_stdio_policy,
)
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
    MCPBuiltinOAuthActorPolicyRequiredError,
)


def _identity() -> ActorMCPStdioSessionIdentity:
    return ActorMCPStdioSessionIdentity(
        execution=MCPActorExecutionIdentity(
            task_id=7,
            run_id="run-one",
            turn_id="turn-one",
            lease_attempt_id="attempt-one",
        ),
        connection=ActorMCPStdioConnectionIdentity(
            user_id=11,
            resource_owner_key="toby:owner",
            app_id="chrome-devtools",
            catalog_app_generation=UUID("11111111-1111-4111-8111-111111111111"),
            lifecycle_generation=UUID("22222222-2222-4222-8222-222222222222"),
        ),
    )


def _connection(identity=None):
    connection = {
        "transport": "stdio",
        "command": "npx",
        "args": [],
        "env": {"XAGENT_MCP_CALLER_ID": "11"},
    }
    if identity is not None:
        connection["actor_stdio_session_identity"] = identity
    return connection


@pytest.mark.parametrize(
    "policy",
    [None, MCPActorAuthorizationPolicy(resource_owner_key="toby:owner")],
)
def test_chrome_stdio_requires_explicit_actor_capability(policy):
    with pytest.raises(MCPBuiltinOAuthActorPolicyRequiredError):
        require_chrome_builtin_stdio_policy(policy)


def test_chrome_stdio_gate_returns_the_same_immutable_policy():
    policy = MCPActorAuthorizationPolicy(
        resource_owner_key="toby:owner", allow_builtin_stdio=True
    )
    assert require_chrome_builtin_stdio_policy(policy) is policy


def test_same_exact_execution_and_connection_has_one_stable_opaque_scope():
    identity = _identity()
    first, first_connection = consume_chrome_execution_scope(
        "chrome-devtools", _connection(identity)
    )
    second, second_connection = consume_chrome_execution_scope(
        "chrome-devtools", _connection(identity)
    )

    assert first == second
    assert first is not None
    assert first.key == identity.key
    assert len(first.digest) == 64
    assert identity.connection.resource_owner_key not in repr(first)
    assert identity.execution.run_id not in repr(first)
    assert "actor_stdio_session_identity" not in first_connection
    assert "actor_stdio_session_identity" not in second_connection


@pytest.mark.parametrize(
    ("field", "value"),
    [("turn_id", "turn-two"), ("lease_attempt_id", "attempt-two")],
)
def test_cross_turn_and_cross_attempt_never_reuse_scope(field, value):
    identity = _identity()
    changed = replace(
        identity,
        execution=replace(identity.execution, **{field: value}),
    )
    original_scope, _ = consume_chrome_execution_scope(
        "chrome-devtools", _connection(identity)
    )
    changed_scope, _ = consume_chrome_execution_scope(
        "chrome-devtools", _connection(changed)
    )

    assert original_scope is not None and changed_scope is not None
    assert original_scope.key != changed_scope.key
    assert original_scope.digest != changed_scope.digest


@pytest.mark.parametrize(
    ("part", "field", "value"),
    [
        ("execution", "task_id", 8),
        ("execution", "run_id", "run-two"),
        ("execution", "turn_id", "turn-two"),
        ("execution", "lease_attempt_id", "attempt-two"),
        ("connection", "user_id", 12),
        ("connection", "resource_owner_key", "toby:other"),
        ("connection", "app_id", "chrome-devtools-other"),
        (
            "connection",
            "catalog_app_generation",
            UUID("33333333-3333-4333-8333-333333333333"),
        ),
        (
            "connection",
            "lifecycle_generation",
            UUID("44444444-4444-4444-8444-444444444444"),
        ),
    ],
)
def test_every_execution_and_connection_dimension_changes_scope(part, field, value):
    identity = _identity()
    changed = replace(
        identity, **{part: replace(getattr(identity, part), **{field: value})}
    )
    original_scope, _ = consume_chrome_execution_scope(
        "chrome-devtools", _connection(identity)
    )
    changed_connection = _connection(changed)
    changed_connection["env"]["XAGENT_MCP_CALLER_ID"] = str(changed.connection.user_id)

    if field == "app_id":
        with pytest.raises(ChromeSessionContractError, match="does not match"):
            consume_chrome_execution_scope("chrome-devtools", changed_connection)
    else:
        changed_scope, _ = consume_chrome_execution_scope(
            "chrome-devtools", changed_connection
        )
        assert original_scope is not None and changed_scope is not None
        assert changed_scope.digest != original_scope.digest


@pytest.mark.parametrize("identity", [None, object()])
def test_execution_scoped_chrome_requires_exact_identity(identity):
    with pytest.raises(ChromeSessionContractError, match="exact session identity"):
        consume_chrome_execution_scope("chrome-devtools", _connection(identity))


def test_execution_scoped_chrome_rejects_identity_subclasses():
    class DerivedIdentity(ActorMCPStdioSessionIdentity):
        pass

    identity = _identity()
    derived = DerivedIdentity(
        execution=identity.execution,
        connection=identity.connection,
    )
    with pytest.raises(ChromeSessionContractError, match="exact session identity"):
        consume_chrome_execution_scope("chrome-devtools", _connection(derived))


def test_execution_scoped_chrome_rejects_wrong_nested_identity_type():
    identity = _identity()
    malformed = ActorMCPStdioSessionIdentity(
        execution=object(),  # type: ignore[arg-type]
        connection=identity.connection,
    )
    with pytest.raises(ChromeSessionContractError, match="does not match"):
        consume_chrome_execution_scope("chrome-devtools", _connection(malformed))


def test_chrome_identity_user_mismatch_fails_closed():
    connection = _connection(_identity())
    connection["env"]["XAGENT_MCP_CALLER_ID"] = "999"
    with pytest.raises(ChromeSessionContractError, match="caller identity"):
        consume_chrome_execution_scope("chrome-devtools", connection)


def test_ordinary_stdio_remains_per_call_without_identity():
    scope, connection = consume_chrome_execution_scope("ordinary-server", _connection())
    assert scope is None
    assert connection == _connection()


@pytest.mark.asyncio
async def test_dedicated_sandbox_uses_only_opaque_scope_and_attaches(monkeypatch):
    manager = SimpleNamespace(
        get_or_create_lease_provider=AsyncMock(),
        attach_provider=AsyncMock(return_value=True),
        delete_sandbox=AsyncMock(),
    )
    provider = SimpleNamespace(primary_sandbox=object())
    manager.get_or_create_lease_provider.return_value = provider
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_sandbox_manager",
        lambda: manager,
    )
    digest = "a" * 64

    handle = await _create_chrome_sandbox(digest)
    await handle.delete()

    manager.get_or_create_lease_provider.assert_awaited_once_with(
        "chrome-execution", digest
    )
    manager.attach_provider.assert_awaited_once_with(
        "chrome-execution", digest, provider
    )
    manager.delete_sandbox.assert_awaited_once_with("chrome-execution", digest)


@pytest.mark.asyncio
async def test_missing_sandbox_manager_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_sandbox_manager", lambda: None
    )
    with pytest.raises(ChromeSessionContractError, match="unavailable"):
        await _create_chrome_sandbox("b" * 64)
