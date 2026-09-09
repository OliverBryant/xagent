from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.core.utils.encryption import _is_encrypted
from xagent.web.models.actor_mcp_connection import ActorMCPServerConnection
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.actor_mcp_connections import (
    ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH,
    ActorMCPConnectionValidationError,
    create_actor_mcp_connection,
    delete_actor_mcp_connection,
    get_actor_mcp_connection_snapshot,
    list_actor_mcp_connection_snapshots,
    update_actor_mcp_connection_credentials,
)

ALICE_OWNER = "toby:slack:T1:alice"
BOB_OWNER = "toby:slack:T1:bob"


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _seed_app(db: Session, *, app_id: str = "posthog") -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=app_id,
        name=app_id,
        transport="stdio",
        launch_config={"command": "persisted-values-are-not-trusted"},
    )
    db.add(app)
    db.flush()
    return app


def _seed_users(db: Session) -> tuple[User, User]:
    users = (
        User(username="actor-owner", password_hash="x"),
        User(username="other-user", password_hash="x"),
    )
    db.add_all(users)
    db.flush()
    return users


def _create(
    db: Session,
    *,
    user_id: int,
    owner: str,
    app_id: str,
    credentials: dict[str, str] | None = None,
):
    return create_actor_mcp_connection(
        db,
        user_id=user_id,
        resource_owner_key=owner,
        app_id=app_id,
        credentials=credentials
        if credentials is not None
        else {"POSTHOG_API_KEY": "secret-value", "POSTHOG_HOST": "https://x.test"},
    )


def test_credentials_are_encrypted_at_rest_and_decrypted_in_snapshot(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)

    snapshot = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )
    stored = db.query(ActorMCPServerConnection).one()

    assert snapshot.credentials == {
        "POSTHOG_API_KEY": "secret-value",
        "POSTHOG_HOST": "https://x.test",
    }
    assert stored.encrypted_env is not None
    assert all(_is_encrypted(value) for value in stored.encrypted_env.values())
    assert "secret-value" not in str(stored.encrypted_env)
    assert "https://x.test" not in str(stored.encrypted_env)
    assert "secret-value" not in repr(stored)
    assert "secret-value" not in repr(snapshot)
    assert "https://x.test" not in repr(snapshot)


def test_reads_and_deletes_require_exact_user_owner_and_app_tuple(db: Session) -> None:
    user, other_user = _seed_users(db)
    app = _seed_app(db)
    created = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )

    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=BOB_OWNER,
            app_id=app.app_id,
        )
        is None
    )
    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=other_user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        is None
    )
    assert (
        list_actor_mcp_connection_snapshots(
            db, user_id=other_user.id, resource_owner_key=ALICE_OWNER
        )
        == []
    )
    assert not delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=BOB_OWNER,
        app_id=app.app_id,
    )
    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        == created
    )


def test_actor_tuple_uniqueness_is_enforced_by_app_and_catalog_generation(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    first_app = _seed_app(db, app_id="posthog")
    second_app = _seed_app(db, app_id="google-maps")
    db.add(
        ActorMCPServerConnection(
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=first_app.app_id,
            catalog_app_generation=first_app.generation,
        )
    )
    db.commit()

    db.add(
        ActorMCPServerConnection(
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=first_app.app_id,
            catalog_app_generation=second_app.generation,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    db.add(
        ActorMCPServerConnection(
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=second_app.app_id,
            catalog_app_generation=first_app.generation,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()


def test_partial_update_merges_omitted_fields_and_preserves_generation(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )

    updated = update_actor_mcp_connection_credentials(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        credentials={"POSTHOG_API_KEY": "rotated"},
    )

    assert updated is not None
    assert updated.lifecycle_generation == original.lifecycle_generation
    assert updated.credentials == {
        "POSTHOG_API_KEY": "rotated",
        "POSTHOG_HOST": "https://x.test",
    }


def test_lifecycle_generation_cannot_be_updated(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )
    row = db.query(ActorMCPServerConnection).one()
    db.commit()
    original_generation = row.lifecycle_generation

    row.lifecycle_generation = uuid.uuid4()
    with pytest.raises(ValueError, match="immutable"):
        db.flush()
    db.rollback()

    assert (
        db.query(ActorMCPServerConnection).one().lifecycle_generation
        == original_generation
    )


def test_hard_delete_and_recreate_gets_a_new_generation(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )

    assert delete_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
    )
    assert db.query(ActorMCPServerConnection).count() == 0
    replacement = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
        credentials={
            "POSTHOG_API_KEY": "new-secret",
            "POSTHOG_HOST": "https://new.test",
        },
    )

    assert isinstance(replacement.lifecycle_generation, uuid.UUID)
    assert replacement.lifecycle_generation != original.lifecycle_generation
    assert replacement.credentials == {
        "POSTHOG_API_KEY": "new-secret",
        "POSTHOG_HOST": "https://new.test",
    }


def test_create_rejects_incomplete_unknown_blank_null_and_oversize_credentials(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    invalid_credentials = [
        {"POSTHOG_API_KEY": "secret-value"},
        {
            "POSTHOG_API_KEY": "secret-value",
            "POSTHOG_HOST": "https://x.test",
            "EXTRA": "x",
        },
        {"POSTHOG_API_KEY": "secret-value", "POSTHOG_HOST": ""},
        {"POSTHOG_API_KEY": "secret-value", "POSTHOG_HOST": None},
        {
            "POSTHOG_API_KEY": "x" * (ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH + 1),
            "POSTHOG_HOST": "https://x.test",
        },
    ]
    for credentials in invalid_credentials:
        with pytest.raises(ActorMCPConnectionValidationError):
            create_actor_mcp_connection(
                db,
                user_id=user.id,
                resource_owner_key=ALICE_OWNER,
                app_id=app.app_id,
                credentials=credentials,
            )
    assert db.query(ActorMCPServerConnection).count() == 0


def test_update_rejects_unknown_blank_null_and_oversize_credentials(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(
        db,
        user_id=user.id,
        owner=ALICE_OWNER,
        app_id=app.app_id,
    )
    invalid_credentials = [
        {"EXTRA": "x"},
        {"POSTHOG_API_KEY": ""},
        {"POSTHOG_API_KEY": None},
        {"POSTHOG_API_KEY": "x" * (ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH + 1)},
    ]
    for credentials in invalid_credentials:
        with pytest.raises(ActorMCPConnectionValidationError):
            update_actor_mcp_connection_credentials(
                db,
                user_id=user.id,
                resource_owner_key=ALICE_OWNER,
                app_id=app.app_id,
                credentials=credentials,
            )
    db.expire_all()
    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
        )
        == original
    )


def test_keyless_connection_persists_null_env(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db, app_id="chrome-devtools")

    snapshot = create_actor_mcp_connection(
        db,
        user_id=user.id,
        resource_owner_key=ALICE_OWNER,
        app_id=app.app_id,
        credentials=None,
    )

    assert snapshot.credentials is None
    assert db.query(ActorMCPServerConnection).one().encrypted_env is None


def test_non_builtin_public_app_is_rejected(db: Session) -> None:
    user, _other = _seed_users(db)
    custom = _seed_app(db, app_id="custom-stdio")
    custom.launch_config = {
        "command": "untrusted-command",
        "required_env": ["POSTHOG_API_KEY", "POSTHOG_HOST"],
    }

    with pytest.raises(ActorMCPConnectionValidationError, match="built-in"):
        _create(
            db,
            user_id=user.id,
            owner=ALICE_OWNER,
            app_id=custom.app_id,
        )


@pytest.mark.parametrize("required_env", [{"KEY": "bad"}, ["KEY", "KEY"], [""]])
def test_invalid_builtin_credential_schema_is_rejected(
    db: Session, monkeypatch, required_env
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_connections.get_builtin_execution_fields",
        lambda _app_id: {
            "transport": "stdio",
            "launch_config": {"command": "trusted", "required_env": required_env},
        },
    )

    with pytest.raises(ActorMCPConnectionValidationError, match="credential schema"):
        _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)


def test_catalog_delete_cascades_and_recreate_does_not_revive_connection(
    db: Session,
) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    original = _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    db.commit()

    db.delete(app)
    db.commit()
    assert db.query(ActorMCPServerConnection).count() == 0

    replacement_app = _seed_app(db)
    assert replacement_app.generation != original.catalog_app_generation
    assert (
        get_actor_mcp_connection_snapshot(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=replacement_app.app_id,
        )
        is None
    )


def test_update_rejects_stale_catalog_generation(db: Session) -> None:
    user, _other = _seed_users(db)
    app = _seed_app(db)
    other_app = _seed_app(db, app_id="google-maps")
    _create(db, user_id=user.id, owner=ALICE_OWNER, app_id=app.app_id)
    row = db.query(ActorMCPServerConnection).one()
    db.execute(
        ActorMCPServerConnection.__table__.update()
        .where(ActorMCPServerConnection.id == row.id)
        .values(catalog_app_generation=other_app.generation)
    )
    db.expire_all()

    with pytest.raises(ActorMCPConnectionValidationError, match="stale"):
        update_actor_mcp_connection_credentials(
            db,
            user_id=user.id,
            resource_owner_key=ALICE_OWNER,
            app_id=app.app_id,
            credentials={"POSTHOG_API_KEY": "rotated"},
        )
