"""add stable MCP catalog and association generations

Revision ID: 20260902_mcp_generations
Revises: 20260901_taskstatus_waiting_for_user
Create Date: 2026-09-02

The integer primary keys on ``public_mcp_apps`` and ``user_mcpservers`` are
storage identities, not lifecycle identities. In particular, SQLite can reuse
the maximum ROWID after deletion, and the association's logical unique key can
be recreated after its original row is deleted. Random UUID generations let
later lifecycle fences distinguish each replacement from its predecessor.
"""

from __future__ import annotations

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260902_mcp_generations"
down_revision: Union[str, None] = "20260901_taskstatus_waiting_for_user"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CATALOG_TABLE = "public_mcp_apps"
CATALOG_COLUMN = "generation"
CATALOG_UNIQUE = "uq_public_mcp_apps_generation"
CATALOG_NONEMPTY = "ck_public_mcp_apps_generation_nonempty"
ASSOCIATION_TABLE = "user_mcpservers"
ASSOCIATION_COLUMN = "lifecycle_generation"
ASSOCIATION_UNIQUE = "uq_user_mcpservers_lifecycle_generation"
ASSOCIATION_NONEMPTY = "ck_user_mcpservers_lifecycle_generation_nonempty"


def _table_exists(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _add_and_backfill(
    table_name: str,
    column_name: str,
    unique_name: str,
    nonempty_name: str,
) -> None:
    generation_type = sa.Uuid(as_uuid=True)
    op.add_column(
        table_name,
        sa.Column(column_name, generation_type, nullable=True),
    )

    table = sa.table(
        table_name,
        sa.column("id", sa.Integer()),
        sa.column(column_name, generation_type),
    )
    bind = op.get_bind()
    row_ids = list(bind.scalars(sa.select(table.c.id)))
    generated: set[uuid.UUID] = set()
    backfill_rows: list[dict[str, object]] = []
    for row_id in row_ids:
        generation = uuid.uuid4()
        while generation in generated:
            generation = uuid.uuid4()
        generated.add(generation)
        backfill_rows.append({"_row_id": row_id, "_generation": generation})

    if backfill_rows:
        bind.execute(
            sa.update(table)
            .where(table.c.id == sa.bindparam("_row_id"))
            .values({column_name: sa.bindparam("_generation")}),
            backfill_rows,
        )

    with op.batch_alter_table(table_name) as batch_op:
        batch_op.alter_column(
            column_name,
            existing_type=generation_type,
            nullable=False,
        )
        batch_op.create_unique_constraint(unique_name, [column_name])
        batch_op.create_check_constraint(
            nonempty_name,
            f"CAST({column_name} AS VARCHAR) <> ''",
        )


def upgrade() -> None:
    if _table_exists(CATALOG_TABLE):
        _add_and_backfill(
            CATALOG_TABLE,
            CATALOG_COLUMN,
            CATALOG_UNIQUE,
            CATALOG_NONEMPTY,
        )
    if _table_exists(ASSOCIATION_TABLE):
        _add_and_backfill(
            ASSOCIATION_TABLE,
            ASSOCIATION_COLUMN,
            ASSOCIATION_UNIQUE,
            ASSOCIATION_NONEMPTY,
        )


def _drop_generation(
    table_name: str,
    column_name: str,
    unique_name: str,
    nonempty_name: str,
) -> None:
    with op.batch_alter_table(table_name) as batch_op:
        batch_op.drop_constraint(nonempty_name, type_="check")
        batch_op.drop_constraint(unique_name, type_="unique")
        batch_op.drop_column(column_name)


def downgrade() -> None:
    # Older application versions do not read these internal lifecycle tokens.
    # Dropping them is therefore schema-compatible, but a later re-upgrade
    # intentionally generates new UUIDs instead of resurrecting old lifecycles.
    if _table_exists(ASSOCIATION_TABLE):
        _drop_generation(
            ASSOCIATION_TABLE,
            ASSOCIATION_COLUMN,
            ASSOCIATION_UNIQUE,
            ASSOCIATION_NONEMPTY,
        )
    if _table_exists(CATALOG_TABLE):
        _drop_generation(
            CATALOG_TABLE,
            CATALOG_COLUMN,
            CATALOG_UNIQUE,
            CATALOG_NONEMPTY,
        )
