from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_DEVTOOLS_PACKAGE,
    ChromeDaemonClient,
    ChromeDaemonLaunchSpec,
    ChromeExecutionSession,
    ChromeExecutionSessionPool,
    ChromeLeaseState,
    ChromeSandboxHandle,
    ChromeSessionContractError,
    ChromeSessionReaper,
    ChromeSessionReaperCandidate,
)
from xagent.sandbox.base import ExecResult, Sandbox


def _connection(*, suffix: list[str] | None = None):
    return {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "--prefer-offline",
            CHROME_DEVTOOLS_PACKAGE,
            "--headless",
            "--isolated",
            *(suffix or []),
        ],
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sandbox() -> AsyncMock:
    sandbox = AsyncMock(spec=Sandbox)
    sandbox.name = "chrome-execution::test"
    return sandbox


class TestChromeDaemonLaunchSpec:
    def test_accepts_only_the_canonical_pinned_npx_shape(self):
        spec = ChromeDaemonLaunchSpec.from_connection(
            _connection(suffix=["--no-usage-statistics"])
        )
        assert spec.server_args == (
            "--headless",
            "--isolated",
            "--no-usage-statistics",
        )

    @pytest.mark.parametrize(
        "connection",
        [
            {"transport": "sse", "command": "npx", "args": []},
            {"transport": "stdio", "command": "node", "args": []},
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "chrome-devtools-mcp@latest"],
            },
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["--prefer-offline", "-y", CHROME_DEVTOOLS_PACKAGE],
            },
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "--prefer-offline", CHROME_DEVTOOLS_PACKAGE],
            },
        ],
    )
    def test_rejects_noncanonical_or_unisolated_launches(self, connection):
        with pytest.raises(ChromeSessionContractError):
            ChromeDaemonLaunchSpec.from_connection(connection)


class TestChromeDaemonClient:
    @pytest.mark.asyncio
    async def test_every_operation_stays_inside_sandbox_and_passes_safe_env(self):
        sandbox = _sandbox()
        sandbox.exec.side_effect = [
            ExecResult(exit_code=0, stdout="", stderr=""),
            ExecResult(exit_code=0, stdout="", stderr=""),
        ]
        sandbox.read_file.return_value = '{"running":true}'
        client = ChromeDaemonClient(sandbox, session_id="a" * 32)

        result = await client.status()

        assert result == {"running": True}
        run = sandbox.exec.await_args_list[0]
        assert run.args[0] == "python"
        assert "chrome_daemon_runner.py" in run.args[1]
        assert run.kwargs["env"] == {
            "NPM_CONFIG_CACHE": "/opt/npm-cache",
            "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
        }
        assert sandbox.exec.await_args_list[1].args[:2] == ("rm", "-f")

    @pytest.mark.asyncio
    async def test_sandbox_failure_has_no_direct_process_fallback(self):
        sandbox = _sandbox()
        sandbox.exec.side_effect = [
            ExecResult(exit_code=17, stdout="secret", stderr="secret"),
            ExecResult(exit_code=0, stdout="", stderr=""),
        ]
        client = ChromeDaemonClient(sandbox, session_id="b" * 32)

        with pytest.raises(ChromeSessionContractError, match="exit code 17"):
            await client.status()

        assert sandbox.exec.await_count == 2

    def test_rejects_untrusted_session_names(self):
        with pytest.raises(ValueError, match="lowercase hex"):
            ChromeDaemonClient(_sandbox(), session_id="../../shared")


class TestChromeExecutionSession:
    @pytest.mark.asyncio
    async def test_reuses_one_healthy_daemon_and_serializes_calls(self):
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock()),
            session_id="c" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        client = AsyncMock()
        client.status.return_value = {"running": True}
        client.invoke_tool.side_effect = [{"value": 1}, {"value": 2}]
        session._client = client

        first, second = await asyncio.gather(
            session.invoke_tool("navigate_page", {"url": "https://one.invalid"}),
            session.invoke_tool("take_snapshot", {}),
        )

        assert (first, second) == ({"value": 1}, {"value": 2})
        client.start.assert_awaited_once()
        client.status.assert_awaited_once()
        assert client.invoke_tool.await_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_deletes_sandbox_even_when_daemon_stop_fails(self):
        delete = AsyncMock()
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=delete),
            session_id="d" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        session._started = True
        client = AsyncMock()
        client.stop.side_effect = RuntimeError("stop failed")
        session._client = client

        await session.close()
        await session.close()

        client.stop.assert_awaited_once()
        delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_continues_after_waiter_cancellation(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        deleted = asyncio.Event()

        async def delete():
            entered.set()
            await release.wait()
            deleted.set()

        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=delete),
            session_id="e" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        waiter = asyncio.create_task(session.close())
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.wait_for(deleted.wait(), timeout=1)


class TestChromeExecutionSessionPool:
    @pytest.mark.asyncio
    async def test_factory_is_once_per_opaque_full_execution_digest(self):
        factory = AsyncMock(
            side_effect=lambda _digest: ChromeSandboxHandle(
                sandbox=_sandbox(), delete=AsyncMock()
            )
        )
        pool = ChromeExecutionSessionPool(factory)
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        one = _digest("exact execution one")
        two = _digest("exact execution two")

        first, duplicate = await asyncio.gather(
            pool.get_or_create(one, launch), pool.get_or_create(one, launch)
        )
        distinct = await pool.get_or_create(two, launch)

        assert first is duplicate
        assert distinct is not first
        assert factory.await_args_list[0].args == (one,)
        assert factory.await_args_list[1].args == (two,)

    @pytest.mark.asyncio
    async def test_factory_failure_propagates_without_fallback(self):
        factory = AsyncMock(side_effect=RuntimeError("sandbox unavailable"))
        pool = ChromeExecutionSessionPool(factory)
        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            await pool.get_or_create(
                _digest("execution"),
                ChromeDaemonLaunchSpec.from_connection(_connection()),
            )

    @pytest.mark.asyncio
    async def test_same_scope_rejects_launch_spec_substitution(self):
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock())
            )
        )
        scope = _digest("execution")
        await pool.get_or_create(
            scope, ChromeDaemonLaunchSpec.from_connection(_connection())
        )

        with pytest.raises(ChromeSessionContractError, match="cannot change"):
            await pool.get_or_create(
                scope,
                ChromeDaemonLaunchSpec.from_connection(
                    _connection(suffix=["--no-usage-statistics"])
                ),
            )

    @pytest.mark.asyncio
    async def test_recreate_waits_until_old_sandbox_is_deleted(self):
        delete_entered = asyncio.Event()
        allow_delete = asyncio.Event()

        async def slow_delete():
            delete_entered.set()
            await allow_delete.wait()

        factory = AsyncMock(
            side_effect=[
                ChromeSandboxHandle(sandbox=_sandbox(), delete=slow_delete),
                ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock()),
            ]
        )
        pool = ChromeExecutionSessionPool(factory)
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        scope = _digest("execution")
        first = await pool.get_or_create(scope, launch)

        closing = asyncio.create_task(pool.close(scope))
        await delete_entered.wait()
        recreating = asyncio.create_task(pool.get_or_create(scope, launch))
        await asyncio.sleep(0)
        assert factory.await_count == 1

        allow_delete.set()
        await closing
        second = await recreating

        assert second is not first
        assert factory.await_count == 2

    def test_rejects_partial_or_non_digest_scope(self):
        pool = ChromeExecutionSessionPool(AsyncMock())
        with pytest.raises(ValueError, match="SHA-256"):
            pool._validate_scope_digest("connection-identity-only")


class TestChromeSessionReaper:
    @pytest.mark.asyncio
    async def test_only_ttl_expired_stale_fences_reach_atomic_reclaim(self):
        now = datetime(2026, 9, 9, tzinfo=timezone.utc)
        fresh = ChromeSessionReaperCandidate("fresh", now, "f1")
        current = ChromeSessionReaperCandidate(
            "current", now - timedelta(hours=1), "f2"
        )
        unknown = ChromeSessionReaperCandidate(
            "unknown", now - timedelta(hours=1), "f3"
        )
        stale = ChromeSessionReaperCandidate("stale", now - timedelta(hours=1), "f4")

        async def classify(candidate):
            return {
                "current": ChromeLeaseState.CURRENT,
                "unknown": ChromeLeaseState.UNKNOWN,
                "stale": ChromeLeaseState.STALE,
            }[candidate.lifecycle_id]

        reclaim = AsyncMock(return_value=True)
        reaper = ChromeSessionReaper(
            list_candidates=AsyncMock(return_value=[fresh, current, unknown, stale]),
            classify_lease=classify,
            reclaim_if_stale=reclaim,
            ttl=timedelta(minutes=30),
        )

        assert await reaper.sweep(now=now) == ("stale",)
        reclaim.assert_awaited_once_with(stale)

    @pytest.mark.asyncio
    async def test_classification_failure_is_fail_closed(self):
        candidate = ChromeSessionReaperCandidate(
            "candidate",
            datetime.now(timezone.utc) - timedelta(hours=1),
            "fence",
        )
        reclaim = AsyncMock()
        reaper = ChromeSessionReaper(
            list_candidates=AsyncMock(return_value=[candidate]),
            classify_lease=AsyncMock(side_effect=RuntimeError("database unavailable")),
            reclaim_if_stale=reclaim,
        )

        assert await reaper.sweep() == ()
        reclaim.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reclaim_failure_does_not_skip_later_orphans(self):
        now = datetime.now(timezone.utc)
        first = ChromeSessionReaperCandidate("first", now - timedelta(hours=1), "f1")
        second = ChromeSessionReaperCandidate("second", now - timedelta(hours=1), "f2")
        reclaim = AsyncMock(side_effect=[RuntimeError("delete failed"), True])
        reaper = ChromeSessionReaper(
            list_candidates=AsyncMock(return_value=[first, second]),
            classify_lease=AsyncMock(return_value=ChromeLeaseState.STALE),
            reclaim_if_stale=reclaim,
        )

        assert await reaper.sweep(now=now) == ("second",)
        assert reclaim.await_count == 2


def test_chrome_session_primitives_are_not_wired_into_generic_mcp_loader():
    import xagent.core.tools.adapters.vibe.mcp_adapter as generic_loader

    source = generic_loader.__loader__.get_source(generic_loader.__name__)
    assert source is not None
    assert "ChromeExecutionSessionPool" not in source
