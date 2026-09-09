from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from ...core.utils.encryption import decrypt_env_dict_strict, encrypt_env_dict
from ..builtin_mcp_registry import get_builtin_execution_fields
from ..models.actor_mcp_connection import (
    ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH,
    ActorMCPServerConnection,
)
from ..models.public_mcp import PublicMCPApp
from ..repositories.actor_mcp_connections import (
    add_actor_mcp_connection,
    get_actor_mcp_connection,
    hard_delete_actor_mcp_connection,
    list_actor_mcp_connections,
)

ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH = 4096


class ActorMCPConnectionValidationError(ValueError):
    """The actor connection request violates the internal storage contract."""


@dataclass(frozen=True)
class ActorMCPConnectionSnapshot:
    id: int
    lifecycle_generation: UUID
    user_id: int
    resource_owner_key: str
    app_id: str
    catalog_app_generation: UUID
    credentials: dict[str, str] | None = dataclass_field(repr=False)


def _positive_id(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActorMCPConnectionValidationError(
            f"{field_name} must be a persisted positive integer"
        )
    return value


def _owner_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ActorMCPConnectionValidationError("resource_owner_key must be a string")
    if not value or value != value.strip():
        raise ActorMCPConnectionValidationError(
            "resource_owner_key must be an exact non-blank string"
        )
    if len(value) > ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH:
        raise ActorMCPConnectionValidationError(
            "resource_owner_key exceeds "
            f"{ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH} characters"
        )
    return value


def _exact_app_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ActorMCPConnectionValidationError(
            "app_id must be an exact non-blank canonical app id"
        )
    if len(value) > 100:
        raise ActorMCPConnectionValidationError("app_id exceeds 100 characters")
    return value


def _builtin_stdio_definition(
    db: Session, *, app_id: str
) -> tuple[PublicMCPApp, frozenset[str]]:
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).one_or_none()
    execution = get_builtin_execution_fields(app_id)
    if (
        app is None
        or execution is None
        or str(execution.get("transport") or "").lower() != "stdio"
    ):
        raise ActorMCPConnectionValidationError(
            "app_id must identify a code-defined built-in stdio app"
        )
    launch = execution.get("launch_config")
    if not isinstance(launch, dict) or not launch.get("command"):
        raise ActorMCPConnectionValidationError(
            "built-in stdio app has an invalid execution definition"
        )
    raw_fields = launch.get("required_env")
    if raw_fields is None:
        fields: list[Any] = []
    elif isinstance(raw_fields, list):
        fields = raw_fields
    else:
        raise ActorMCPConnectionValidationError(
            "canonical stdio app has an invalid credential schema"
        )
    if any(not isinstance(field, str) or not field for field in fields) or len(
        fields
    ) != len(set(fields)):
        raise ActorMCPConnectionValidationError(
            "canonical stdio app has an invalid credential schema"
        )
    return app, frozenset(fields)


def _validated_credentials(
    credentials: Mapping[str, Any] | None,
    *,
    allowed_fields: frozenset[str],
    require_complete: bool,
) -> dict[str, str] | None:
    if credentials is None:
        supplied: Mapping[str, Any] = {}
    elif not isinstance(credentials, Mapping):
        raise ActorMCPConnectionValidationError("credentials must be an object")
    else:
        supplied = credentials

    unknown = set(supplied) - allowed_fields
    if unknown:
        raise ActorMCPConnectionValidationError(
            "credentials contain fields outside the canonical app schema"
        )
    if require_complete and set(supplied) != allowed_fields:
        raise ActorMCPConnectionValidationError(
            "connection creation requires every canonical credential field"
        )
    validated: dict[str, str] = {}
    for field, value in supplied.items():
        if not isinstance(field, str):
            raise ActorMCPConnectionValidationError(
                "credential field names must be strings"
            )
        if value is None:
            raise ActorMCPConnectionValidationError(
                "credential values must not be null"
            )
        if not isinstance(value, str):
            raise ActorMCPConnectionValidationError("credential values must be strings")
        if not value.strip():
            raise ActorMCPConnectionValidationError(
                "credential values must not be blank"
            )
        if len(value) > ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH:
            raise ActorMCPConnectionValidationError(
                "credential value exceeds "
                f"{ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH} characters"
            )
        validated[field] = value
    return validated or None


def _snapshot(row: ActorMCPServerConnection) -> ActorMCPConnectionSnapshot:
    decrypted = decrypt_env_dict_strict(row.encrypted_env)
    return ActorMCPConnectionSnapshot(
        id=row.id,
        lifecycle_generation=row.lifecycle_generation,
        user_id=row.user_id,
        resource_owner_key=row.resource_owner_key,
        app_id=row.app_id,
        catalog_app_generation=row.catalog_app_generation,
        credentials=dict(decrypted) if decrypted is not None else None,
    )


def create_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    credentials: Mapping[str, Any] | None,
) -> ActorMCPConnectionSnapshot:
    user_id = _positive_id(user_id, field_name="user_id")
    owner_key = _owner_key(resource_owner_key)
    app_id = _exact_app_id(app_id)
    app, allowed_fields = _builtin_stdio_definition(db, app_id=app_id)
    validated = _validated_credentials(
        credentials, allowed_fields=allowed_fields, require_complete=True
    )
    encrypted = encrypt_env_dict(validated)
    row = add_actor_mcp_connection(
        db,
        ActorMCPServerConnection(
            user_id=user_id,
            resource_owner_key=owner_key,
            app_id=app_id,
            catalog_app_generation=app.generation,
            encrypted_env=encrypted,
        ),
    )
    return _snapshot(row)


def get_actor_mcp_connection_snapshot(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
) -> ActorMCPConnectionSnapshot | None:
    row = get_actor_mcp_connection(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
        app_id=_exact_app_id(app_id),
    )
    return _snapshot(row) if row is not None else None


def list_actor_mcp_connection_snapshots(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
) -> list[ActorMCPConnectionSnapshot]:
    rows = list_actor_mcp_connections(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
    )
    return [_snapshot(row) for row in rows]


def update_actor_mcp_connection_credentials(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    credentials: Mapping[str, Any],
) -> ActorMCPConnectionSnapshot | None:
    user_id = _positive_id(user_id, field_name="user_id")
    owner_key = _owner_key(resource_owner_key)
    app_id = _exact_app_id(app_id)
    row = get_actor_mcp_connection(
        db,
        user_id=user_id,
        resource_owner_key=owner_key,
        app_id=app_id,
        for_update=True,
    )
    if row is None:
        return None
    app, allowed_fields = _builtin_stdio_definition(db, app_id=app_id)
    if app.generation != row.catalog_app_generation:
        raise ActorMCPConnectionValidationError(
            "actor connection belongs to a stale catalog app lifecycle"
        )
    updates = _validated_credentials(
        credentials, allowed_fields=allowed_fields, require_complete=False
    )
    existing = decrypt_env_dict_strict(row.encrypted_env) or {}
    if updates:
        existing.update(updates)
    row.encrypted_env = encrypt_env_dict(existing) or None
    db.flush()
    return _snapshot(row)


def delete_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
) -> bool:
    row = get_actor_mcp_connection(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
        app_id=_exact_app_id(app_id),
        for_update=True,
    )
    if row is None:
        return False
    hard_delete_actor_mcp_connection(db, row)
    return True
