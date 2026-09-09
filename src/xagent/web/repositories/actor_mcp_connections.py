from __future__ import annotations

from sqlalchemy.orm import Query, Session

from ..models.actor_mcp_connection import ActorMCPServerConnection


def scoped_actor_mcp_connection_query(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
) -> Query[ActorMCPServerConnection]:
    """Build the only permitted base query for actor MCP connections."""
    return db.query(ActorMCPServerConnection).filter(
        ActorMCPServerConnection.user_id == user_id,
        ActorMCPServerConnection.resource_owner_key == resource_owner_key,
    )


def get_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    for_update: bool = False,
) -> ActorMCPServerConnection | None:
    query = scoped_actor_mcp_connection_query(
        db,
        user_id=user_id,
        resource_owner_key=resource_owner_key,
    ).filter(ActorMCPServerConnection.app_id == app_id)
    if for_update:
        query = query.with_for_update()
    return query.populate_existing().one_or_none()


def list_actor_mcp_connections(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
) -> list[ActorMCPServerConnection]:
    return (
        scoped_actor_mcp_connection_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .order_by(ActorMCPServerConnection.id)
        .all()
    )


def add_actor_mcp_connection(
    db: Session, connection: ActorMCPServerConnection
) -> ActorMCPServerConnection:
    db.add(connection)
    db.flush()
    return connection


def hard_delete_actor_mcp_connection(
    db: Session, connection: ActorMCPServerConnection
) -> None:
    db.delete(connection)
    db.flush()
