import importlib.util
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def _migration_module():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260726_add_task_telegram_user_id.py"
    )
    spec = importlib.util.spec_from_file_location("telegram_task_owner_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def test_migration_adds_and_removes_telegram_user_id() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

            inspector = sa.inspect(connection)
            assert "telegram_user_id" in {
                column["name"] for column in inspector.get_columns("tasks")
            }
            assert migration.INDEX in {
                index["name"] for index in inspector.get_indexes("tasks")
            }

            migration.downgrade()
            inspector = sa.inspect(connection)
            assert "telegram_user_id" not in {
                column["name"] for column in inspector.get_columns("tasks")
            }
