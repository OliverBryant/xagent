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


def _existing_columns(inspector: sa.Inspector) -> list[str]:
    return [str(column["name"]) for column in inspector.get_columns(TABLE)]


def _existing_indexes(inspector: sa.Inspector) -> list[str]:
    return [str(index["name"]) for index in inspector.get_indexes(TABLE)]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return

    if COLUMN not in _existing_columns(inspector):
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=32), nullable=True))

    inspector = sa.inspect(bind)
    if INDEX not in _existing_indexes(inspector):
        op.create_index(INDEX, TABLE, [COLUMN])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return

    if INDEX in _existing_indexes(inspector):
        op.drop_index(INDEX, table_name=TABLE)

    inspector = sa.inspect(bind)
    if COLUMN in _existing_columns(inspector):
        op.drop_column(TABLE, COLUMN)
