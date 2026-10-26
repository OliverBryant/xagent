"""Migration coverage for ordered Auto binding storage."""

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261026_auto_sequential_failover.py"
)
REVISION = "20261026_auto_sequential_failover"
DOWN_REVISION = "20261023_add_mcp_signed_uploads"
CONFIG_TABLE = "auto_model_configs"
CANDIDATE_TABLE = "auto_model_candidates"
PROFILE_CONSTRAINT = "uq_auto_candidate_routing_model"
TARGET_CONSTRAINT = "uq_auto_candidate_target_model"
POSITION_CONSTRAINT = "uq_auto_candidate_position"


def _migration():
    return load_migration_module(MIGRATION_PATH, "auto_order_migration")


def _old_schema(metadata: sa.MetaData) -> tuple[sa.Table, sa.Table]:
    configs = sa.Table(
        CONFIG_TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
    )
    candidates = sa.Table(
        CANDIDATE_TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("routing_model_id", sa.String(200), nullable=False),
        sa.Column("target_model_id", sa.Integer, nullable=False),
        sa.UniqueConstraint("config_id", "routing_model_id", name=PROFILE_CONSTRAINT),
        sa.UniqueConstraint("config_id", "target_model_id", name=TARGET_CONSTRAINT),
    )
    return configs, candidates


def _constraint_names(connection) -> set[str]:
    return {
        str(constraint["name"])
        for constraint in sa.inspect(connection).get_unique_constraints(CANDIDATE_TABLE)
        if constraint.get("name")
    }


def _column_names(connection, table_name: str) -> set[str]:
    return {
        str(column["name"]) for column in sa.inspect(connection).get_columns(table_name)
    }


def _upgrade(migration, connection) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        migration.upgrade()


def _downgrade(migration, connection) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        migration.downgrade()


def _exercise_upgrade_and_reentry(engine) -> None:
    migration = _migration()
    metadata = sa.MetaData()
    configs, candidates = _old_schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(configs.insert(), [{"id": 1}, {"id": 2}])
        connection.execute(
            candidates.insert(),
            [
                {
                    "id": 12,
                    "config_id": 1,
                    "routing_model_id": "z-ai/glm-5.3",
                    "target_model_id": 120,
                },
                {
                    "id": 10,
                    "config_id": 1,
                    "routing_model_id": "deepseek/deepseek-chat",
                    "target_model_id": 100,
                },
                {
                    "id": 20,
                    "config_id": 2,
                    "routing_model_id": "openai/gpt-5.5",
                    "target_model_id": 200,
                },
            ],
        )
        _upgrade(migration, connection)

        upgraded = sa.Table(CANDIDATE_TABLE, sa.MetaData(), autoload_with=connection)
        connection.execute(
            upgraded.insert().values(
                id=11,
                config_id=1,
                routing_model_id="deepseek/deepseek-chat",
                target_model_id=110,
                position=2,
            )
        )
        rows_before_reentry = connection.execute(
            sa.select(
                upgraded.c.config_id,
                upgraded.c.id,
                upgraded.c.routing_model_id,
                upgraded.c.position,
            ).order_by(upgraded.c.config_id, upgraded.c.position)
        ).all()
        assert rows_before_reentry == [
            (1, 10, "deepseek/deepseek-chat", 0),
            (1, 12, "z-ai/glm-5.3", 1),
            (1, 11, "deepseek/deepseek-chat", 2),
            (2, 20, "openai/gpt-5.5", 0),
        ]

        assert _constraint_names(connection) == {
            TARGET_CONSTRAINT,
            POSITION_CONSTRAINT,
        }
        assert "mode" in _column_names(connection, CONFIG_TABLE)
        modes = connection.execute(
            sa.text(f"SELECT id, mode FROM {CONFIG_TABLE} ORDER BY id")
        ).all()
        assert modes == [(1, "routing"), (2, "routing")]

        # Simulate an interrupted migration after it removed/recreated one
        # constraint. Re-entry must restore the missing constraint without
        # renumbering already-valid order or restoring profile uniqueness.
        with Operations.context(MigrationContext.configure(connection)) as operations:
            with operations.batch_alter_table(CANDIDATE_TABLE) as batch_op:
                batch_op.drop_constraint(POSITION_CONSTRAINT, type_="unique")
        _upgrade(migration, connection)

        reflected = sa.Table(CANDIDATE_TABLE, sa.MetaData(), autoload_with=connection)
        rows_after_reentry = connection.execute(
            sa.select(
                reflected.c.config_id,
                reflected.c.id,
                reflected.c.routing_model_id,
                reflected.c.position,
            ).order_by(reflected.c.config_id, reflected.c.position)
        ).all()
        assert rows_after_reentry == rows_before_reentry
        assert _constraint_names(connection) == {
            TARGET_CONSTRAINT,
            POSITION_CONSTRAINT,
        }


def _exercise_partial_position_repair(engine) -> None:
    migration = _migration()
    metadata = sa.MetaData()
    sa.Table(
        CONFIG_TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "mode",
            sa.String(32),
            nullable=False,
            server_default="routing",
        ),
    )
    candidates = sa.Table(
        CANDIDATE_TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("routing_model_id", sa.String(200), nullable=False),
        sa.Column("target_model_id", sa.Integer, nullable=False),
        sa.Column("position", sa.Integer, nullable=True),
        sa.UniqueConstraint("config_id", "routing_model_id", name=PROFILE_CONSTRAINT),
        sa.UniqueConstraint("config_id", "target_model_id", name=TARGET_CONSTRAINT),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            candidates.insert(),
            [
                {
                    "id": 7,
                    "config_id": 1,
                    "routing_model_id": "profile-b",
                    "target_model_id": 70,
                    "position": 0,
                },
                {
                    "id": 3,
                    "config_id": 1,
                    "routing_model_id": "profile-a",
                    "target_model_id": 30,
                    "position": 0,
                },
            ],
        )
        _upgrade(migration, connection)

        repaired = sa.Table(CANDIDATE_TABLE, sa.MetaData(), autoload_with=connection)
        assert connection.execute(
            sa.select(repaired.c.id, repaired.c.position).order_by(repaired.c.position)
        ).all() == [(3, 0), (7, 1)]
        position_column = next(
            column
            for column in sa.inspect(connection).get_columns(CANDIDATE_TABLE)
            if column["name"] == "position"
        )
        assert position_column["nullable"] is False
        assert _constraint_names(connection) == {
            TARGET_CONSTRAINT,
            POSITION_CONSTRAINT,
        }


def _exercise_safe_and_refused_downgrades(engine) -> None:
    migration = _migration()
    metadata = sa.MetaData()
    configs, candidates = _old_schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(configs.insert().values(id=1))
        connection.execute(
            candidates.insert().values(
                id=1,
                config_id=1,
                routing_model_id="profile-a",
                target_model_id=10,
            )
        )
        _upgrade(migration, connection)
        _downgrade(migration, connection)
        _downgrade(migration, connection)

        assert "position" not in _column_names(connection, CANDIDATE_TABLE)
        assert "mode" not in _column_names(connection, CONFIG_TABLE)
        assert _constraint_names(connection) == {
            PROFILE_CONSTRAINT,
            TARGET_CONSTRAINT,
        }

    # Use a fresh database because the successful downgrade restored the old
    # profile constraint, which correctly cannot represent the conflict case.
    conflict_metadata = sa.MetaData()
    conflict_configs, conflict_candidates = _old_schema(conflict_metadata)
    conflict_metadata.drop_all(engine)
    conflict_metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(conflict_configs.insert().values(id=1))
        connection.execute(
            conflict_candidates.insert().values(
                id=1,
                config_id=1,
                routing_model_id="profile-a",
                target_model_id=10,
            )
        )
        _upgrade(migration, connection)
        upgraded = sa.Table(CANDIDATE_TABLE, sa.MetaData(), autoload_with=connection)
        connection.execute(
            upgraded.insert().values(
                id=2,
                config_id=1,
                routing_model_id="profile-a",
                target_model_id=20,
                position=1,
            )
        )

        with pytest.raises(RuntimeError, match="without losing data"):
            _downgrade(migration, connection)

        protected = sa.Table(CANDIDATE_TABLE, sa.MetaData(), autoload_with=connection)
        assert {"position"} <= set(protected.c.keys())
        assert "mode" in _column_names(connection, CONFIG_TABLE)
        assert (
            connection.execute(
                sa.select(sa.func.count()).select_from(protected)
            ).scalar_one()
            == 2
        )
        assert _constraint_names(connection) == {
            TARGET_CONSTRAINT,
            POSITION_CONSTRAINT,
        }


def test_revision_metadata() -> None:
    migration = _migration()
    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION


def test_sqlite_upgrade_and_constraint_reentry() -> None:
    _exercise_upgrade_and_reentry(sa.create_engine("sqlite://"))


def test_sqlite_repairs_partially_applied_positions() -> None:
    _exercise_partial_position_repair(sa.create_engine("sqlite://"))


def test_sqlite_downgrade_is_safe_or_refuses_loss() -> None:
    _exercise_safe_and_refused_downgrades(sa.create_engine("sqlite://"))


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_auto_order") as make:
        yield make


@pytest.mark.postgresql
def test_postgresql_upgrade_and_constraint_reentry(
    postgresql_engine_factory,
) -> None:
    _exercise_upgrade_and_reentry(postgresql_engine_factory("upgrade"))


@pytest.mark.postgresql
def test_postgresql_repairs_partially_applied_positions(
    postgresql_engine_factory,
) -> None:
    _exercise_partial_position_repair(postgresql_engine_factory("partial"))


@pytest.mark.postgresql
def test_postgresql_downgrade_is_safe_or_refuses_loss(
    postgresql_engine_factory,
) -> None:
    _exercise_safe_and_refused_downgrades(postgresql_engine_factory("downgrade"))
