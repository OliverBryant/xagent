"""Tests for the Telegram sender ownership migration."""

import importlib.util
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20260726_add_task_telegram_user_id.py"
)
TABLE = "tasks"
COLUMN = "telegram_user_id"
INDEX = "ix_tasks_telegram_user_id"


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "telegram_task_owner_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def _transactional_offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context), context.begin_transaction():
        getattr(migration, operation)()
    return output.getvalue()


def test_migration_adds_and_removes_telegram_user_id() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

            inspector = sa.inspect(connection)
            assert COLUMN in {column["name"] for column in inspector.get_columns(TABLE)}
            assert INDEX in {index["name"] for index in inspector.get_indexes(TABLE)}

            migration.downgrade()
            inspector = sa.inspect(connection)
            assert COLUMN not in {
                column["name"] for column in inspector.get_columns(TABLE)
            }


def test_migration_noops_without_tasks_table() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_sqlite_offline_upgrade_and_downgrade_emit_plain_ddl() -> None:
    migration = _migration_module()

    upgrade_sql = _offline_sql(migration, "sqlite", "upgrade")
    downgrade_sql = _offline_sql(migration, "sqlite", "downgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(32)" in upgrade_sql
    assert f"CREATE INDEX {INDEX} ON {TABLE} ({COLUMN})" in upgrade_sql
    assert "CONCURRENTLY" not in upgrade_sql
    assert f"DROP INDEX {INDEX}" in downgrade_sql
    assert "CONCURRENTLY" not in downgrade_sql


def test_postgresql_offline_upgrade_and_downgrade_emit_concurrent_index_sql() -> None:
    migration = _migration_module()

    upgrade_sql = _offline_sql(migration, "postgresql", "upgrade")
    downgrade_sql = _offline_sql(migration, "postgresql", "downgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(32)" in upgrade_sql
    assert f"CREATE INDEX CONCURRENTLY {INDEX} ON {TABLE} ({COLUMN})" in upgrade_sql
    assert f"DROP INDEX CONCURRENTLY {INDEX}" in downgrade_sql
    assert f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}" in downgrade_sql


def test_postgresql_offline_concurrent_ddl_escapes_outer_transaction() -> None:
    migration = _migration_module()

    for operation in ("upgrade", "downgrade"):
        sql = _transactional_offline_sql(migration, "postgresql", operation)
        # CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction block,
        # so the generated script must COMMIT before emitting it.
        concurrent_at = sql.index("CONCURRENTLY")
        assert "COMMIT;" in sql[:concurrent_at]


def test_postgresql_online_upgrade_rebuilds_an_invalid_index() -> None:
    """A failed CREATE INDEX CONCURRENTLY leaves the index invalid. IF NOT
    EXISTS would skip the rebuild, so the retry must drop it first."""

    migration = _migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")

    class _NoopAutocommit:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            return False

    for validity, expect_drop in ((False, True), (None, False), (True, False)):
        created: list[dict] = []
        dropped: list[dict] = []
        with Operations.context(context):
            with (
                patch.object(context, "autocommit_block", _NoopAutocommit),
                patch.object(migration.op, "get_context", return_value=context),
                patch.object(migration, "_online_table_exists", return_value=True),
                patch.object(migration, "_online_columns", return_value={COLUMN}),
                patch.object(
                    migration, "_postgres_index_validity", return_value=validity
                ),
                patch.object(
                    migration.op,
                    "create_index",
                    side_effect=lambda *a, **kw: created.append(kw),
                ),
                patch.object(
                    migration.op,
                    "drop_index",
                    side_effect=lambda *a, **kw: dropped.append(kw),
                ),
            ):
                migration.upgrade()

        assert bool(dropped) is expect_drop, validity
        if validity is True:
            # Already usable: no rebuild at all.
            assert created == []
        else:
            assert created == [{"if_not_exists": True, "postgresql_concurrently": True}]
