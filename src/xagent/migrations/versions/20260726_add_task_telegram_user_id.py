"""add Telegram sender ownership to tasks

Revision ID: 20260726_add_task_telegram_user_id
Revises: 20260725_add_uploaded_file_recovery_index
Create Date: 2026-07-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260726_add_task_telegram_user_id"
down_revision: Union[str, None] = "20260725_add_uploaded_file_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "telegram_user_id"
INDEX = "ix_tasks_telegram_user_id"


def _online_columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {str(item["name"]) for item in inspector.get_columns(TABLE)}


def _online_indexes() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {
        name
        for item in inspector.get_indexes(TABLE)
        if (name := item.get("name")) is not None
    }


def _online_table_exists() -> bool:
    return TABLE in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"

    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable. Emit the unconditional DDL instead of inspecting.
    if context.as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=32), nullable=True))
        if is_postgresql:
            with context.autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    [COLUMN],
                    postgresql_concurrently=True,
                )
        else:
            op.create_index(INDEX, TABLE, [COLUMN])
        return

    if not _online_table_exists():
        return

    if COLUMN not in _online_columns():
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=32), nullable=True))

    if COLUMN not in _online_columns():
        return

    # A plain CREATE INDEX holds a SHARE lock and blocks writes to the live
    # tasks table for the whole build, so PostgreSQL builds it concurrently.
    if is_postgresql:
        with context.autocommit_block():
            op.create_index(
                INDEX,
                TABLE,
                [COLUMN],
                if_not_exists=True,
                postgresql_concurrently=True,
            )
        return

    if INDEX not in _online_indexes():
        op.create_index(INDEX, TABLE, [COLUMN])


def downgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"

    if context.as_sql:
        if is_postgresql:
            with context.autocommit_block():
                op.drop_index(
                    INDEX,
                    table_name=TABLE,
                    postgresql_concurrently=True,
                )
        else:
            op.drop_index(INDEX, table_name=TABLE)
        op.drop_column(TABLE, COLUMN)
        return

    if not _online_table_exists():
        return

    if is_postgresql:
        with context.autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
    elif INDEX in _online_indexes():
        op.drop_index(INDEX, table_name=TABLE)

    if COLUMN in _online_columns():
        op.drop_column(TABLE, COLUMN)
