"""add actor-scoped stdio MCP connection storage

Revision ID: 20260909_actor_mcp_connections
Revises: 20260901_seed_zendesk_mcp_app
Create Date: 2026-09-09
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.web.models.generation import RandomUUID

revision: str = "20260909_actor_mcp_connections"
down_revision: Union[str, None] = "20260901_seed_zendesk_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "actor_mcp_server_connections"


def _adopt_current_metadata_table() -> bool:
    """Adopt an exact metadata-created table, rejecting partial schema drift."""
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(TABLE):
        return False

    columns = {column["name"]: column for column in inspector.get_columns(TABLE)}
    expected_columns = {
        "id",
        "lifecycle_generation",
        "user_id",
        "resource_owner_key",
        "app_id",
        "catalog_app_generation",
        "encrypted_env",
        "created_at",
        "updated_at",
    }
    nullable_columns = {name for name, column in columns.items() if column["nullable"]}
    lengths = {
        name: getattr(column["type"], "length", None)
        for name, column in columns.items()
    }
    primary_key = tuple(inspector.get_pk_constraint(TABLE)["constrained_columns"])
    unique_columns = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints(TABLE)
    }
    foreign_keys = {
        tuple(foreign_key["constrained_columns"]): (
            foreign_key["referred_table"],
            tuple(foreign_key["referred_columns"]),
            str((foreign_key.get("options") or {}).get("ondelete")).upper(),
        )
        for foreign_key in inspector.get_foreign_keys(TABLE)
    }
    indexes = {
        (index["name"], tuple(index["column_names"]))
        for index in inspector.get_indexes(TABLE)
    }
    check_constraints = {
        constraint["name"] for constraint in inspector.get_check_constraints(TABLE)
    }
    valid = (
        set(columns) == expected_columns
        and nullable_columns == {"encrypted_env"}
        and lengths["resource_owner_key"] == 512
        and lengths["app_id"] == 100
        and columns["lifecycle_generation"]["default"] is not None
        and columns["created_at"]["default"] is not None
        and columns["updated_at"]["default"] is not None
        and primary_key == ("id",)
        and "ck_actor_mcp_server_connections_generation_nonempty" in check_constraints
        and unique_columns
        >= {
            ("lifecycle_generation",),
            ("user_id", "resource_owner_key", "app_id"),
            ("user_id", "resource_owner_key", "catalog_app_generation"),
        }
        and foreign_keys
        == {
            ("user_id",): ("users", ("id",), "CASCADE"),
            ("catalog_app_generation",): (
                "public_mcp_apps",
                ("generation",),
                "CASCADE",
            ),
        }
        and (op.f("ix_actor_mcp_server_connections_id"), ("id",)) in indexes
    )
    if not valid:
        raise RuntimeError(
            "actor_mcp_server_connections already exists with incompatible schema"
        )
    return True


def upgrade() -> None:
    if _adopt_current_metadata_table():
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "lifecycle_generation",
            sa.Uuid(),
            server_default=RandomUUID(),
            nullable=False,
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("resource_owner_key", sa.String(length=512), nullable=False),
        sa.Column("app_id", sa.String(length=100), nullable=False),
        sa.Column("catalog_app_generation", sa.Uuid(), nullable=False),
        sa.Column("encrypted_env", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "CAST(lifecycle_generation AS VARCHAR) <> ''",
            name="ck_actor_mcp_server_connections_generation_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_app_generation"],
            ["public_mcp_apps.generation"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "lifecycle_generation",
            name="uq_actor_mcp_server_connections_lifecycle_generation",
        ),
        sa.UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "app_id",
            name="uq_actor_mcp_server_connections_actor_app",
        ),
        sa.UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "catalog_app_generation",
            name="uq_actor_mcp_server_connections_actor_catalog_generation",
        ),
    )
    op.create_index(op.f("ix_actor_mcp_server_connections_id"), TABLE, ["id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_actor_mcp_server_connections_id"), table_name=TABLE)
    op.drop_table(TABLE)
