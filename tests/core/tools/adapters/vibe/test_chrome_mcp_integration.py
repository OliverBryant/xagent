from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe import mcp_adapter
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    ChromeExecutionMCPToolAdapter,
    MCPFailurePhase,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_DEVTOOLS_PACKAGE,
    ChromeExecutionSessionPool,
)
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
)
from xagent.web.services.mcp_runtime import MCPActorExecutionIdentity


def _identity(*, turn_id: str = "turn-one", attempt: str = "attempt-one"):
    return ActorMCPStdioSessionIdentity(
        execution=MCPActorExecutionIdentity(
            task_id=7,
            run_id="run-one",
            turn_id=turn_id,
            lease_attempt_id=attempt,
        ),
        connection=ActorMCPStdioConnectionIdentity(
            user_id=11,
            resource_owner_key="toby:owner-secret",
            app_id="chrome-devtools",
            catalog_app_generation=UUID("11111111-1111-4111-8111-111111111111"),
            lifecycle_generation=UUID("22222222-2222-4222-8222-222222222222"),
        ),
    )


def _connection(identity=...):
    connection = {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "--prefer-offline",
            CHROME_DEVTOOLS_PACKAGE,
            "--headless",
            "--isolated",
        ],
        "env": {"XAGENT_MCP_CALLER_ID": "11"},
    }
    if identity is ...:
        identity = _identity()
    if identity is not None:
        connection["actor_stdio_session_identity"] = identity
    return connection


def _tool():
    return MCPTool(
        name="navigate_page",
        description="Navigate",
        inputSchema={"type": "object", "properties": {}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox", [None, object()])
async def test_chrome_consumes_identity_before_any_loader_or_serializer(
    monkeypatch, sandbox
):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.invoke_tool.return_value = {
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
    }
    serialized_connections = []

    async def list_tools(_sandbox, connection):
        serialized_connections.append(connection)
        json.dumps(connection)
        return [_tool()]

    direct = AsyncMock()
    generic_sandbox = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "list_tools_in_sandbox", list_tools)
    monkeypatch.setattr(mcp_adapter, "_load_direct_mcp_tools", direct)
    monkeypatch.setattr(mcp_adapter, "load_sandboxed_mcp_tools", generic_sandbox)
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )

    result = await load_mcp_tools_as_agent_tools(
        {"chrome-devtools": _connection()}, sandbox=sandbox
    )

    assert len(result.tools) == 1
    assert isinstance(result.tools[0], ChromeExecutionMCPToolAdapter)
    assert "actor_stdio_session_identity" not in result.tools[0].connection
    assert all(
        "actor_stdio_session_identity" not in item for item in serialized_connections
    )
    assert "toby:owner-secret" not in repr(serialized_connections)
    direct.assert_not_awaited()
    generic_sandbox.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, object()])
@pytest.mark.parametrize("sandbox", [None, object()])
async def test_missing_or_wrong_chrome_identity_has_no_fallback(
    monkeypatch, identity, sandbox
):
    direct = AsyncMock()
    generic_sandbox = AsyncMock()
    dedicated = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "_load_direct_mcp_tools", direct)
    monkeypatch.setattr(mcp_adapter, "load_sandboxed_mcp_tools", generic_sandbox)
    monkeypatch.setattr(mcp_adapter, "_load_execution_scoped_chrome_tools", dedicated)

    result = await load_mcp_tools_as_agent_tools(
        {"chrome-devtools": _connection(identity)}, sandbox=sandbox
    )

    assert result.tools == ()
    assert result.failures[0].phase is MCPFailurePhase.SESSION_START
    direct.assert_not_awaited()
    generic_sandbox.assert_not_awaited()
    dedicated.assert_not_awaited()


@pytest.mark.asyncio
async def test_chrome_adapter_reuses_scope_and_validates_daemon_result(monkeypatch):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.invoke_tool.return_value = {
        "content": [{"type": "text", "text": "same browser"}],
        "structuredContent": {"page": 2},
        "isError": False,
    }
    monkeypatch.setattr(
        mcp_adapter, "list_tools_in_sandbox", AsyncMock(return_value=[_tool()])
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    loaded = await load_mcp_tools_as_agent_tools({"chrome-devtools": _connection()})
    tool = loaded.tools[0]

    first = await tool._execute_mcp_call(tool.connection, {}, {})
    second = await tool._execute_mcp_call(tool.connection, {}, {})

    assert (
        first
        == second
        == {
            "content": [
                {
                    "type": "text",
                    "text": "same browser",
                    "annotations": None,
                    "meta": None,
                }
            ],
            "structured_content": {"page": 2},
            "is_error": False,
        }
    )
    assert pool.invoke_tool.await_count == 2
    assert (
        pool.invoke_tool.await_args_list[0].args[0]
        == pool.invoke_tool.await_args_list[1].args[0]
    )


@pytest.mark.asyncio
async def test_invalid_daemon_result_fails_without_per_call_fallback(monkeypatch):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.invoke_tool.return_value = {"content": [], "isError": "false"}
    direct_session = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "create_session", direct_session)
    monkeypatch.setattr(
        mcp_adapter, "list_tools_in_sandbox", AsyncMock(return_value=[_tool()])
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    loaded = await load_mcp_tools_as_agent_tools({"chrome-devtools": _connection()})
    tool = loaded.tools[0]

    with pytest.raises(Exception):
        await tool._execute_mcp_call(tool.connection, {}, {})
    direct_session.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_chrome_teardown_does_not_abort_remaining_tool_cleanup(
    monkeypatch,
):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.close_shielded.side_effect = asyncio.CancelledError
    monkeypatch.setattr(
        mcp_adapter, "list_tools_in_sandbox", AsyncMock(return_value=[_tool()])
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    loaded = await load_mcp_tools_as_agent_tools({"chrome-devtools": _connection()})

    await loaded.tools[0].teardown()

    pool.close_shielded.assert_awaited_once()
