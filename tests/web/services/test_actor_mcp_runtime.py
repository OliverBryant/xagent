from __future__ import annotations

import logging
import json
import uuid
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import Tool as MCPTool
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe import mcp_adapter
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    ChromeExecutionMCPToolAdapter,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    ChromeExecutionSessionPool,
)
from xagent.web.builtin_mcp_registry import (
    get_builtin_execution_fields,
    get_builtin_public_mcp_app,
)
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.services.actor_mcp_connections import (
    ActorMCPConnectionCredentialCorruptionError,
    ActorMCPConnectionMetadata,
    ActorMCPConnectionSnapshot,
    ActorMCPConnectionValidationError,
    create_actor_mcp_connection,
)
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPConnectionServiceAdapter,
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
    production_actor_mcp_stdio_connection_adapter,
    resolve_actor_mcp_stdio_configs,
)
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
)
from xagent.web.tools.config import WebToolConfig

USER_ID = 41
OWNER = "toby:slack:T1:U1"
APP_ID = "posthog"


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _seed_app(db: Session, *, app_id: str = APP_ID) -> PublicMCPApp:
    execution = get_builtin_execution_fields(app_id)
    assert execution is not None
    app = PublicMCPApp(
        app_id=app_id,
        name=execution["name"],
        transport=execution["transport"],
        provider_name=execution["provider_name"],
        oauth_scopes=execution["oauth_scopes"],
        launch_config=execution["launch_config"],
        is_visible_in_connector=True,
    )
    db.add(app)
    db.flush()
    return app


class _FakeAdapter:
    def __init__(
        self,
        identity: ActorMCPStdioConnectionIdentity,
        credentials: Mapping[str, str] | None,
    ) -> None:
        self.identity = identity
        self.credentials = credentials
        self.list_calls: list[dict[str, object]] = []
        self.secret_calls: list[dict[str, object]] = []

    def list_connection_identities(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
    ) -> Sequence[ActorMCPStdioConnectionIdentity]:
        self.list_calls.append(
            {"user_id": user_id, "resource_owner_key": resource_owner_key}
        )
        return (self.identity,)

    def get_connection_credentials(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
        app_id: str,
        catalog_app_generation: uuid.UUID,
        expected_lifecycle_generation: uuid.UUID,
    ) -> Mapping[str, str] | None:
        self.secret_calls.append(
            {
                "user_id": user_id,
                "resource_owner_key": resource_owner_key,
                "app_id": app_id,
                "catalog_app_generation": catalog_app_generation,
                "expected_lifecycle_generation": expected_lifecycle_generation,
            }
        )
        return self.credentials


class _StableIdentityOnlyServer:
    def __init__(self, server_id: int, name: str) -> None:
        self.id = server_id
        self.name = name
        self.forbidden_accesses: list[str] = []

    def __getattr__(self, name: str) -> object:
        self.forbidden_accesses.append(name)
        raise AssertionError(f"actor resolver read forbidden MCPServer field: {name}")


def _identity(app: PublicMCPApp, **overrides: object):
    values = {
        "user_id": USER_ID,
        "resource_owner_key": OWNER,
        "app_id": app.app_id,
        "catalog_app_generation": app.generation,
        "lifecycle_generation": uuid.uuid4(),
    }
    values.update(overrides)
    return ActorMCPStdioConnectionIdentity(**values)


def _credentials() -> dict[str, str]:
    return {
        "POSTHOG_API_KEY": "actor-api-key",
        "POSTHOG_HOST": "https://actor.example.test",
    }


def _policy(*, allow: bool = True) -> MCPActorAuthorizationPolicy:
    return MCPActorAuthorizationPolicy(
        resource_owner_key=OWNER,
        allow_builtin_stdio=allow,
    )


def _execution_identity(
    *,
    task_id: int = 91,
    run_id: str = "run-1",
    turn_id: str = "turn-1",
    lease_attempt_id: str = "attempt-1",
) -> MCPActorExecutionIdentity:
    return MCPActorExecutionIdentity(
        task_id=task_id,
        run_id=run_id,
        turn_id=turn_id,
        lease_attempt_id=lease_attempt_id,
    )


def test_synthetic_config_uses_exact_lifecycle_fenced_adapter_inputs(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    identity = _identity(app)
    adapter = _FakeAdapter(identity, _credentials())

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert adapter.list_calls == [{"user_id": USER_ID, "resource_owner_key": OWNER}]
    assert adapter.secret_calls == [
        {
            "user_id": USER_ID,
            "resource_owner_key": OWNER,
            "app_id": APP_ID,
            "catalog_app_generation": app.generation,
            "expected_lifecycle_generation": identity.lifecycle_generation,
        }
    ]
    assert result.blocked_server_ids == frozenset()
    assert len(result.configs) == 1
    config = result.configs[0]
    execution = get_builtin_execution_fields(APP_ID)
    assert execution is not None
    assert config["config"]["command"] == execution["launch_config"]["command"]
    assert config["config"]["args"] == execution["launch_config"]["args"]
    assert config["config"]["env"] == {
        **_credentials(),
        "XAGENT_MCP_CALLER_ID": str(USER_ID),
    }
    assert "id" not in config


def test_production_adapter_is_a_stateless_exact_service_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_generation = uuid.uuid4()
    lifecycle_generation = uuid.uuid4()
    metadata = ActorMCPConnectionMetadata(
        id=7,
        lifecycle_generation=lifecycle_generation,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        catalog_app_generation=catalog_generation,
        configured_field_names=frozenset(_credentials()),
    )
    snapshot = ActorMCPConnectionSnapshot(
        id=7,
        lifecycle_generation=lifecycle_generation,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        catalog_app_generation=catalog_generation,
        credentials=_credentials(),
    )
    list_calls: list[dict[str, object]] = []
    secret_calls: list[dict[str, object]] = []

    def fake_list(_db: Session, **kwargs: object) -> list[object]:
        list_calls.append(kwargs)
        return [metadata]

    def fake_get(_db: Session, **kwargs: object) -> object:
        secret_calls.append(kwargs)
        return snapshot

    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_runtime.list_actor_mcp_connection_metadata",
        fake_list,
    )
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_runtime.get_actor_mcp_connection_credentials_internal",
        fake_get,
    )
    adapter = ActorMCPConnectionServiceAdapter()

    identities = adapter.list_connection_identities(
        object(),  # type: ignore[arg-type]
        user_id=USER_ID,
        resource_owner_key=OWNER,
    )
    credentials = adapter.get_connection_credentials(
        object(),  # type: ignore[arg-type]
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        catalog_app_generation=catalog_generation,
        expected_lifecycle_generation=lifecycle_generation,
    )

    assert identities == (
        ActorMCPStdioConnectionIdentity(
            user_id=USER_ID,
            resource_owner_key=OWNER,
            app_id=APP_ID,
            catalog_app_generation=catalog_generation,
            lifecycle_generation=lifecycle_generation,
        ),
    )
    assert credentials == _credentials()
    assert list_calls == [{"user_id": USER_ID, "resource_owner_key": OWNER}]
    assert secret_calls == [
        {
            "user_id": USER_ID,
            "resource_owner_key": OWNER,
            "app_id": APP_ID,
            "catalog_app_generation": catalog_generation,
            "expected_lifecycle_generation": lifecycle_generation,
        }
    ]


def test_production_adapter_resolves_credentials_from_storage_service(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    _seed_app(db)
    create_actor_mcp_connection(
        db,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        credentials=_credentials(),
    )

    def forbidden_transaction_boundary() -> None:
        raise AssertionError("runtime adapter must not own commit or rollback")

    monkeypatch.setattr(db, "commit", forbidden_transaction_boundary)
    monkeypatch.setattr(db, "rollback", forbidden_transaction_boundary)

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=production_actor_mcp_stdio_connection_adapter(),
        visible_servers=(),
    )

    assert len(result.configs) == 1
    assert result.configs[0]["config"]["env"] == {
        **_credentials(),
        "XAGENT_MCP_CALLER_ID": str(USER_ID),
    }


@pytest.mark.asyncio
async def test_create_default_tools_registers_production_adapter_only_for_actor_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.api.chat import create_default_tools

    captured: list[dict[str, object]] = []

    class _FakeToolConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

        def set_task_runtime_contribution(self, _contribution: object) -> None:
            pass

    async def create_tools(_config: object) -> list[object]:
        return []

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: object()
    )
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)

    for policy in (None, _policy()):
        await create_default_tools(
            None,
            user=SimpleNamespace(id=USER_ID, is_admin=False),
            task_id="91",
            mcp_runtime_authorization_policy=policy,
        )

    assert captured[0]["mcp_actor_stdio_connection_adapter"] is None
    assert (
        captured[1]["mcp_actor_stdio_connection_adapter"]
        is production_actor_mcp_stdio_connection_adapter()
    )


@pytest.mark.parametrize(
    "failure",
    [
        ActorMCPConnectionCredentialCorruptionError("sensitive-value"),
        RuntimeError("sensitive-value"),
    ],
)
def test_credential_failures_remain_blocked_without_logging_values(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    caplog.set_level(logging.INFO)
    app = _seed_app(db)

    class _FailingAdapter(_FakeAdapter):
        def get_connection_credentials(self, *_args: object, **_kwargs: object):
            raise failure

    collision = _StableIdentityOnlyServer(73, "chrome-devtools")
    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=_FailingAdapter(_identity(app), _credentials()),
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})
    assert type(failure).__name__ in caplog.text
    assert "sensitive-value" not in caplog.text


def test_missing_adapter_remains_blocked_without_custom_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    collision = _StableIdentityOnlyServer(73, APP_ID)

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=None,
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})


@pytest.mark.parametrize(
    "failure",
    [
        ActorMCPConnectionValidationError("sensitive-value"),
        RuntimeError("sensitive-value"),
    ],
)
def test_list_failures_remain_blocked_without_logging_values(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    caplog.set_level(logging.INFO)

    class _FailingListAdapter:
        def list_connection_identities(self, *_args: object, **_kwargs: object):
            raise failure

    collision = _StableIdentityOnlyServer(73, APP_ID)
    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=_FailingListAdapter(),  # type: ignore[arg-type]
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})
    assert type(failure).__name__ in caplog.text
    assert "sensitive-value" not in caplog.text


def test_reserved_collision_reads_only_stable_server_identity(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    collision = _StableIdentityOnlyServer(73, "  PostHog  ")

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})
    assert adapter.secret_calls == []
    assert collision.forbidden_accesses == []


def test_policy_or_feature_off_never_falls_back_to_reserved_server(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", raising=False)
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    collision = _StableIdentityOnlyServer(73, APP_ID)

    feature_off = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(collision,),
    )
    legacy_policy = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(allow=False),
        adapter=adapter,
        visible_servers=(collision,),
    )

    assert feature_off.configs == ()
    assert feature_off.blocked_server_ids == frozenset({73})
    assert legacy_policy.blocked_server_ids == frozenset()
    assert adapter.list_calls == []


@pytest.mark.parametrize(
    "override",
    [
        {"user_id": 42},
        {"resource_owner_key": "toby:slack:T1:U2"},
        {"catalog_app_generation": uuid.uuid4()},
    ],
)
def test_mismatched_connection_identity_fails_closed(
    db: Session, monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app, **override), _credentials())

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()
    assert adapter.secret_calls == []


def test_incomplete_runtime_credentials_fail_closed(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(
        _identity(app),
        {"POSTHOG_API_KEY": "actor-api-key"},
    )

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()


@pytest.mark.asyncio
async def test_web_loader_appends_synthetic_config_without_env_source_queries(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    config = WebToolConfig(
        db=db,
        request=None,
        user_id=USER_ID,
        task_id="execution-id-is-not-derived-here",
        mcp_runtime_authorization_policy=_policy(),
        mcp_actor_stdio_connection_adapter=adapter,
    )
    monkeypatch.setattr(
        config,
        "_visible_mcp_server_query",
        lambda _team_ids: SimpleNamespace(all=lambda: []),
    )

    def forbidden(*_args: object, **_kwargs: object):
        raise AssertionError("ordinary MCP credential source was queried")

    from xagent.web.services import mcp_runtime

    monkeypatch.setattr(mcp_runtime, "load_user_env_overrides", forbidden)
    monkeypatch.setattr(mcp_runtime, "load_shared_env_overrides", forbidden)
    monkeypatch.setattr(mcp_runtime, "load_user_env_sources", forbidden)

    result = await config._load_mcp_server_configs()

    assert [item["name"] for item in result] == [APP_ID]


def test_execution_scoped_chrome_requires_complete_execution_identity(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="chrome-devtools")
    adapter = _FakeAdapter(_identity(app), None)

    missing = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )
    present = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
        execution_identity=_execution_identity(),
    )

    assert missing.configs == ()
    assert len(present.configs) == 1
    session_identity = present.configs[0]["actor_stdio_session_identity"]
    assert isinstance(session_identity, ActorMCPStdioSessionIdentity)
    assert session_identity.key == (
        91,
        "run-1",
        "turn-1",
        "attempt-1",
        USER_ID,
        OWNER,
        "chrome-devtools",
        app.generation,
        adapter.identity.lifecycle_generation,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("task_id", None),
        ("run_id", None),
        ("turn_id", None),
        ("lease_attempt_id", None),
    ],
)
def test_actor_execution_identity_rejects_each_missing_field(
    field_name: str, value: object
) -> None:
    values: dict[str, object] = {
        "task_id": 91,
        "run_id": "run-1",
        "turn_id": "turn-1",
        "lease_attempt_id": "attempt-1",
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        MCPActorExecutionIdentity(**values)  # type: ignore[arg-type]


def test_chrome_session_key_changes_for_retry_and_later_turn() -> None:
    connection = ActorMCPStdioConnectionIdentity(
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        catalog_app_generation=uuid.uuid4(),
        lifecycle_generation=uuid.uuid4(),
    )
    initial = ActorMCPStdioSessionIdentity(_execution_identity(), connection)
    retry = ActorMCPStdioSessionIdentity(
        _execution_identity(lease_attempt_id="attempt-2"), connection
    )
    later_turn = ActorMCPStdioSessionIdentity(
        _execution_identity(turn_id="turn-2"), connection
    )

    assert initial.key != retry.key
    assert initial.key != later_turn.key


@pytest.mark.asyncio
async def test_tool_factory_threads_actor_stdio_session_identity_to_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ActorMCPStdioConnectionIdentity(
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        catalog_app_generation=uuid.uuid4(),
        lifecycle_generation=uuid.uuid4(),
    )
    session_identity = ActorMCPStdioSessionIdentity(_execution_identity(), connection)
    captured: dict[str, object] = {}

    async def fake_load(connections: dict[str, object], **_kwargs: object) -> object:
        captured.update(connections)
        return SimpleNamespace(tools=(), failures=())

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fake_load,
    )

    await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "chrome-devtools",
                "transport": "stdio",
                "config": {"command": "npx", "args": [], "env": {}},
                "actor_stdio_session_identity": session_identity,
            }
        ]
    )

    assert (
        captured["chrome-devtools"]["actor_stdio_session_identity"] is session_identity
    )  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox", [None, SimpleNamespace(name="existing-sandbox")])
async def test_production_storage_resolver_factory_chrome_consumes_identity_before_ipc(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    sandbox: object | None,
) -> None:
    """Exercise the real storage/resolver/factory chain up to sandbox I/O."""

    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="chrome-devtools")
    builtin = get_builtin_public_mcp_app("chrome-devtools")
    assert builtin is not None
    assert builtin["is_visible_in_connector"] is False
    create_actor_mcp_connection(
        db,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        credentials={},
    )
    resolution = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=production_actor_mcp_stdio_connection_adapter(),
        visible_servers=(),
        execution_identity=_execution_identity(),
    )
    assert len(resolution.configs) == 1
    synthetic = resolution.configs[0]
    identity = synthetic["actor_stdio_session_identity"]
    assert type(identity) is ActorMCPStdioSessionIdentity

    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    child_connections: list[dict[str, object]] = []

    async def serializer_spy(_sandbox: object, connection: dict[str, object]):
        child_connections.append(connection)
        json.dumps(connection)
        return [
            MCPTool(
                name="navigate_page",
                description="Navigate",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    direct = AsyncMock()
    generic_sandbox = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "list_tools_in_sandbox", serializer_spy)
    monkeypatch.setattr(mcp_adapter, "_load_direct_mcp_tools", direct)
    monkeypatch.setattr(mcp_adapter, "load_sandboxed_mcp_tools", generic_sandbox)
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        list(resolution.configs),
        sandbox=sandbox,  # type: ignore[arg-type]
    )

    assert len(tools) == 1
    assert isinstance(tools[0], ChromeExecutionMCPToolAdapter)
    assert "actor_stdio_session_identity" not in tools[0].connection
    assert len(child_connections) == 1
    child = child_connections[0]
    assert "actor_stdio_session_identity" not in child
    assert OWNER not in repr(child)
    assert identity.execution.run_id not in repr(child)
    direct.assert_not_awaited()
    generic_sandbox.assert_not_awaited()
    assert app.is_visible_in_connector is True
