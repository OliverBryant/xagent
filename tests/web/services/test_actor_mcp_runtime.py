from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.web.builtin_mcp_registry import get_builtin_execution_fields
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    resolve_actor_mcp_stdio_configs,
)
from xagent.web.services.mcp_runtime import MCPActorAuthorizationPolicy
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


def _seed_app(db: Session) -> PublicMCPApp:
    execution = get_builtin_execution_fields(APP_ID)
    assert execution is not None
    app = PublicMCPApp(
        app_id=APP_ID,
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
        "app_id": APP_ID,
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

    assert adapter.list_calls == [
        {"user_id": USER_ID, "resource_owner_key": OWNER}
    ]
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
