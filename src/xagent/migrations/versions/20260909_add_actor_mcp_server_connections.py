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


def upgrade() -> None:
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
