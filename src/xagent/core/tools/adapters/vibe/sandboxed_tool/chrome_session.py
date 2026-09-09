"""Execution-scoped Chrome DevTools sessions hosted only in sandboxes.

The runtime-specific adapter deliberately sits above this module.  It must
turn its exact execution/actor/catalog identity into a SHA-256 digest and
provide a dedicated sandbox plus a destructive cleanup callback.  This layer
does not guess web task identity and has no host-process fallback.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import posixpath
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncIterator, cast

from ......config import get_chrome_session_ttl_seconds
from ......sandbox.base import Sandbox
from ....core.mcp.sessions import Connection
from .chrome_daemon_runner import CHROME_DEVTOOLS_PACKAGE
from .sandboxed_tool_wrapper import SANDBOX_SRC_ROOT

logger = logging.getLogger(__name__)

CHROME_DEVTOOLS_APP_ID = "chrome-devtools"
CHROME_SANDBOX_LIFECYCLE_TYPE = "chrome-execution"

_CHROME_RUNNER_PATH = (
    f"{SANDBOX_SRC_ROOT}/xagent/core/tools/adapters/vibe/sandboxed_tool/"
    "chrome_daemon_runner.py"
)
_SCOPE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_RESULT_FILE_PREFIX = "/tmp/xagent_chrome_daemon_"
_RUNNER_TIMEOUT_SECONDS = 75.0
_RUNNER_MAX_OUTPUT_BYTES = 64 * 1024
_CHROME_SANDBOX_ENV = {
    "NPM_CONFIG_CACHE": "/opt/npm-cache",
    "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
}


def chrome_metadata_connection(connection: Connection) -> Connection:
    """Return the secret-minimal connection serialized for sandbox discovery."""

    launch = ChromeDaemonLaunchSpec.from_connection(connection)
    env = connection.get("env")
    if not isinstance(env, Mapping):
        raise ChromeSessionContractError("Chrome connection env is invalid")
    return cast(
        Connection,
        {
            "transport": connection.get("transport"),
            "command": connection.get("command"),
            "args": [
                "-y",
                "--prefer-offline",
                CHROME_DEVTOOLS_PACKAGE,
                *launch.server_args,
            ],
            "env": {**dict(env), **_CHROME_SANDBOX_ENV},
        },
    )


class ChromeSessionContractError(RuntimeError):
    """An execution-scoped Chrome session violated its fail-closed contract."""


@dataclass(frozen=True)
class ChromeDaemonLaunchSpec:
    """Validated daemon flags derived from the canonical built-in connection."""

    server_args: tuple[str, ...]

    @classmethod
    def from_connection(cls, connection: Connection) -> "ChromeDaemonLaunchSpec":
        if connection.get("transport") != "stdio":
            raise ChromeSessionContractError("Chrome daemon requires stdio transport")
        command = connection.get("command")
        if not isinstance(command, str) or posixpath.basename(command) != "npx":
            raise ChromeSessionContractError("Chrome daemon requires sandboxed npx")
        raw_args = connection.get("args")
        if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
            raise ChromeSessionContractError("Chrome daemon args must be a sequence")
        args = list(raw_args)
        if not all(isinstance(arg, str) for arg in args):
            raise ChromeSessionContractError("Chrome daemon args must be strings")
        try:
            package_index = args.index(CHROME_DEVTOOLS_PACKAGE)
        except ValueError as exc:
            raise ChromeSessionContractError(
                "Chrome daemon must use the pinned package"
            ) from exc
        if args.count(CHROME_DEVTOOLS_PACKAGE) != 1:
            raise ChromeSessionContractError("Chrome package pin is ambiguous")
        if args[:package_index] != ["-y", "--prefer-offline"]:
            raise ChromeSessionContractError(
                "Chrome npx resolution flags are not canonical"
            )
        server_args = tuple(args[package_index + 1 :])
        if "--headless" not in server_args or "--isolated" not in server_args:
            raise ChromeSessionContractError(
                "Chrome daemon requires canonical headless isolated flags"
            )
        return cls(server_args=server_args)


@dataclass(frozen=True)
class ChromeSandboxHandle:
    """Dedicated sandbox and the callback that irreversibly removes it."""

    sandbox: Sandbox
    delete: Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class ChromeExecutionScope:
    """In-memory identity plus the only value allowed outside this process."""

    key: tuple[Any, ...] = field(repr=False)
    digest: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Chrome execution scope key must be non-empty")
        if _SCOPE_DIGEST_RE.fullmatch(self.digest) is None:
            raise ValueError("scope digest must be a SHA-256 lowercase hex digest")


class ChromeDaemonClient:
    """Run the pinned daemon controller exclusively inside one sandbox."""

    def __init__(self, sandbox: Sandbox, *, session_id: str) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
            raise ValueError("session_id must be 32 lowercase hex characters")
        self._sandbox = sandbox
        self._session_id = session_id

    @staticmethod
    def _encoded_json(value: Any) -> str:
        return base64.b64encode(
            json.dumps(value, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

    async def _run(self, operation: str, *args: str) -> dict[str, Any]:
        result_file = f"{_RESULT_FILE_PREFIX}{uuid.uuid4().hex}.json"
        command_args = [
            _CHROME_RUNNER_PATH,
            operation,
            "--session-id",
            self._session_id,
            "--result-file",
            result_file,
            *args,
        ]
        try:
            try:
                execution = await asyncio.wait_for(
                    self._sandbox.exec(
                        "python",
                        *command_args,
                        env=dict(_CHROME_SANDBOX_ENV),
                        max_output_bytes=_RUNNER_MAX_OUTPUT_BYTES,
                    ),
                    timeout=_RUNNER_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise ChromeSessionContractError(
                    f"sandbox Chrome {operation} timed out"
                ) from exc
            if execution.exit_code != 0:
                raise ChromeSessionContractError(
                    f"sandbox Chrome {operation} failed with exit code "
                    f"{execution.exit_code}"
                )
            try:
                payload = json.loads(await self._sandbox.read_file(result_file))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise ChromeSessionContractError(
                    f"sandbox Chrome {operation} returned no valid result"
                ) from exc
            if not isinstance(payload, dict):
                raise ChromeSessionContractError(
                    f"sandbox Chrome {operation} returned a non-object result"
                )
            return payload
        finally:
            try:
                await self._sandbox.exec("rm", "-f", result_file)
            except Exception:
                logger.debug("Chrome runner result cleanup failed", exc_info=True)

    async def start(self, launch: ChromeDaemonLaunchSpec) -> dict[str, Any]:
        return await self._run(
            "start",
            "--server-args-b64",
            self._encoded_json(launch.server_args),
        )

    async def status(self) -> dict[str, Any]:
        return await self._run("status")

    async def invoke_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not tool_name or tool_name != tool_name.strip():
            raise ValueError("tool_name must be an exact non-blank string")
        return await self._run(
            "tool",
            "--tool",
            tool_name,
            "--arguments-b64",
            self._encoded_json(dict(arguments)),
        )

    async def stop(self) -> None:
        await self._run("stop")


class ChromeExecutionSession:
    """One serialized daemon lifecycle in one dedicated sandbox."""

    def __init__(
        self,
        handle: ChromeSandboxHandle,
        *,
        session_id: str,
        launch: ChromeDaemonLaunchSpec,
    ) -> None:
        self._handle = handle
        self._client = ChromeDaemonClient(handle.sandbox, session_id=session_id)
        self._launch = launch
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def launch(self) -> ChromeDaemonLaunchSpec:
        return self._launch

    @property
    def sandbox(self) -> Sandbox:
        return self._handle.sandbox

    def _status_matches_launch(self, payload: Mapping[str, Any]) -> bool:
        status = payload.get("status", payload)
        if not isinstance(status, Mapping):
            return False
        args = status.get("args")
        expected_args = [
            *self._launch.server_args,
            "--viaCli",
            "--experimentalStructuredContent",
        ]
        return (
            status.get("version") == CHROME_DEVTOOLS_PACKAGE.rsplit("@", 1)[1]
            and args == expected_args
        )

    async def _ensure_started_locked(self) -> None:
        if self._closed:
            raise ChromeSessionContractError("Chrome execution session is closed")
        if self._started:
            status = await self._client.status()
            if status.get("running") is True and self._status_matches_launch(status):
                return
            raise ChromeSessionContractError(
                "Chrome daemon status does not match its execution launch"
            )
        status = await self._client.start(self._launch)
        if not self._status_matches_launch(status):
            raise ChromeSessionContractError(
                "Chrome daemon start does not match its execution launch"
            )
        self._started = True

    async def invoke_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        async with self._lock:
            await self._ensure_started_locked()
            return await self._client.invoke_tool(tool_name, arguments)

    async def _cleanup(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._started:
                    await self._client.stop()
            except BaseException:
                logger.warning("Chrome daemon stop failed; deleting its sandbox")
            finally:
                self._started = False
                await self._handle.delete()

    async def close(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        await asyncio.shield(self._cleanup_task)


ChromeSandboxFactory = Callable[[str], Awaitable[ChromeSandboxHandle]]


@dataclass
class _PoolLockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class ChromeExecutionSessionPool:
    """Process-local handles to sandbox-owned, deterministically named sessions."""

    def __init__(self, sandbox_factory: ChromeSandboxFactory) -> None:
        self._sandbox_factory = sandbox_factory
        self._sessions: dict[str, tuple[tuple[Any, ...], ChromeExecutionSession]] = {}
        self._locks: dict[str, _PoolLockEntry] = {}
        self._lock_guard = asyncio.Lock()

    @asynccontextmanager
    async def _scope_locked(self, scope_digest: str) -> AsyncIterator[None]:
        async with self._lock_guard:
            entry = self._locks.setdefault(scope_digest, _PoolLockEntry())
            entry.users += 1
        try:
            await entry.lock.acquire()
            try:
                yield
            finally:
                entry.lock.release()
        finally:
            async with self._lock_guard:
                entry.users -= 1
                if (
                    entry.users == 0
                    and scope_digest not in self._sessions
                    and self._locks.get(scope_digest) is entry
                ):
                    self._locks.pop(scope_digest, None)

    async def get_or_create(
        self, scope: ChromeExecutionScope, launch: ChromeDaemonLaunchSpec
    ) -> ChromeExecutionSession:
        scope_digest = scope.digest
        async with self._scope_locked(scope_digest):
            entry = self._sessions.get(scope_digest)
            if entry is not None:
                stored_key, session = entry
                if stored_key != scope.key:
                    raise ChromeSessionContractError(
                        "Chrome execution scope digest collision"
                    )
                if session.launch != launch:
                    raise ChromeSessionContractError(
                        "one execution scope cannot change its Chrome launch spec"
                    )
                return session
            handle = await self._sandbox_factory(scope_digest)
            session = ChromeExecutionSession(
                handle,
                session_id=scope_digest[:32],
                launch=launch,
            )
            self._sessions[scope_digest] = (scope.key, session)
            return session

    async def invoke_tool(
        self,
        scope: ChromeExecutionScope,
        launch: ChromeDaemonLaunchSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        session = await self.get_or_create(scope, launch)
        try:
            return await session.invoke_tool(tool_name, arguments)
        except ChromeSessionContractError:
            await self.close_shielded(scope)
            raise

    async def close(self, scope: ChromeExecutionScope) -> None:
        scope_digest = scope.digest
        async with self._scope_locked(scope_digest):
            entry = self._sessions.get(scope_digest)
            if entry is None:
                return
            stored_key, session = entry
            if stored_key != scope.key:
                raise ChromeSessionContractError(
                    "Chrome execution scope digest collision"
                )
            self._sessions.pop(scope_digest, None)
            if session is not None:
                await session.close()

    async def close_shielded(self, scope: ChromeExecutionScope) -> None:
        """Keep cleanup alive even when the execution task is cancelled."""

        cleanup = asyncio.create_task(self.close(scope))

        def consume(task: asyncio.Task[None]) -> None:
            if not task.cancelled():
                task.exception()

        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cleanup.add_done_callback(consume)
            raise

    async def close_all(self) -> None:
        cancelled: asyncio.CancelledError | None = None
        for scope_digest in tuple(self._sessions):
            try:
                key, _session = self._sessions[scope_digest]
                await self.close(ChromeExecutionScope(key=key, digest=scope_digest))
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception:
                logger.warning(
                    "Chrome session cleanup failed for scope %s",
                    scope_digest,
                    exc_info=True,
                )
        if cancelled is not None:
            raise cancelled


class ChromeLeaseState(Enum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChromeSessionReaperCandidate:
    """Opaque durable lifecycle record supplied by the future runtime adapter."""

    lifecycle_id: str
    created_at: datetime
    lease_fence: str

    def __post_init__(self) -> None:
        if (
            _SCOPE_DIGEST_RE.fullmatch(self.lifecycle_id) is None
            or _SCOPE_DIGEST_RE.fullmatch(self.lease_fence) is None
        ):
            raise ValueError("Chrome lifecycle and lease fence must be opaque digests")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


class ChromeSessionReaper:
    """TTL gate plus fail-closed lease fencing for orphan cleanup.

    ``reclaim_if_stale`` must perform the final fence comparison atomically
    with claiming/deleting the lifecycle.  The pre-classification avoids
    unnecessary destructive calls; it is not itself the race-proof delete.
    """

    def __init__(
        self,
        *,
        list_candidates: Callable[
            [], Awaitable[Sequence[ChromeSessionReaperCandidate]]
        ],
        classify_lease: Callable[
            [ChromeSessionReaperCandidate], Awaitable[ChromeLeaseState]
        ],
        reclaim_if_stale: Callable[[ChromeSessionReaperCandidate], Awaitable[bool]],
        ttl: timedelta | None = None,
    ) -> None:
        if ttl is None:
            ttl = timedelta(seconds=get_chrome_session_ttl_seconds())
        if ttl <= timedelta(0):
            raise ValueError("Chrome session TTL must be positive")
        self._list_candidates = list_candidates
        self._classify_lease = classify_lease
        self._reclaim_if_stale = reclaim_if_stale
        self._ttl = ttl

    async def sweep(self, *, now: datetime | None = None) -> tuple[str, ...]:
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        reclaimed: list[str] = []
        for candidate in await self._list_candidates():
            if current_time - candidate.created_at < self._ttl:
                continue
            try:
                state = await self._classify_lease(candidate)
            except Exception:
                logger.warning(
                    "Chrome lease classification failed for one lifecycle",
                    exc_info=True,
                )
                continue
            if state is not ChromeLeaseState.STALE:
                continue
            try:
                if await self._reclaim_if_stale(candidate):
                    reclaimed.append(candidate.lifecycle_id)
            except Exception:
                logger.warning(
                    "Chrome fenced reclaim failed for one lifecycle",
                    exc_info=True,
                )
        return tuple(reclaimed)
