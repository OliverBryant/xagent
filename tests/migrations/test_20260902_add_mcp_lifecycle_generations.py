from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.db.config import create_alembic_config
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

REVISION = "20260902_mcp_generations"
DOWN_REVISION = "20260901_taskstatus_waiting_for_user"
MIGRATIONS_DIR = Path(__file__).parents[2] / "src" / "xagent" / "migrations"


def _create_legacy_schema(connection: sa.Connection) -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE public_mcp_apps ("
            "id INTEGER PRIMARY KEY, app_id VARCHAR(100) NOT NULL UNIQUE)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE TABLE user_mcpservers ("
            "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
            "mcpserver_id INTEGER NOT NULL, "
            "CONSTRAINT uq_user_mcpservers UNIQUE (user_id, mcpserver_id))"
        )
    )
    connection.execute(
        sa.text(
            "INSERT INTO public_mcp_apps (id, app_id) VALUES (1, 'one'), (2, 'two')"
        )
    )
    connection.execute(
        sa.text(
            "INSERT INTO user_mcpservers (id, user_id, mcpserver_id) "
            "VALUES (1, 10, 20), (2, 11, 20)"
        )
    )
    connection.execute(
        sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        sa.text("INSERT INTO alembic_version VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _generations(connection: sa.Connection, table: str, column: str) -> list[uuid.UUID]:
    values = connection.scalars(
        sa.text(f"SELECT {column} FROM {table} ORDER BY id")  # noqa: S608
    )
    return [uuid.UUID(str(value)) for value in values]


def _assert_generation_schema(connection: sa.Connection) -> None:
    inspector = sa.inspect(connection)
    for table, column, unique_name in (
        ("public_mcp_apps", "generation", "uq_public_mcp_apps_generation"),
        (
            "user_mcpservers",
            "lifecycle_generation",
            "uq_user_mcpservers_lifecycle_generation",
        ),
    ):
        columns = {item["name"]: item for item in inspector.get_columns(table)}
        assert columns[column]["nullable"] is False
        assert columns[column]["default"] is not None
        uniques = {
            item["name"]: item for item in inspector.get_unique_constraints(table)
        }
        assert uniques[unique_name]["column_names"] == [column]
        values = _generations(connection, table, column)
        assert len(values) == len(set(values)) == 2


def _assert_database_defaults(connection: sa.Connection) -> None:
    connection.execute(
        sa.text(
            "INSERT INTO public_mcp_apps (id, app_id) VALUES (3, 'database-default')"
        )
    )
    connection.execute(
        sa.text(
            "INSERT INTO user_mcpservers (id, user_id, mcpserver_id) VALUES (3, 12, 20)"
        )
    )
    for table, column in (
        ("public_mcp_apps", "generation"),
        ("user_mcpservers", "lifecycle_generation"),
    ):
        generated = uuid.UUID(
            str(
                connection.scalar(
                    sa.text(
                        f"SELECT {column} FROM {table} WHERE id = 3"  # noqa: S608
                    )
                )
            )
        )
        assert generated.version == 4


def _assert_constraints(connection: sa.Connection) -> None:
    for table, column in (
        ("public_mcp_apps", "generation"),
        ("user_mcpservers", "lifecycle_generation"),
    ):
        generation = connection.scalar(
            sa.text(f"SELECT {column} FROM {table} WHERE id = 1")  # noqa: S608
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                sa.text(
                    f"UPDATE {table} SET {column} = :generation WHERE id = 2"  # noqa: S608
                ),
                {"generation": generation},
            )
        connection.rollback()

        with pytest.raises((IntegrityError, sa.exc.StatementError, sa.exc.DataError)):
            connection.execute(
                sa.text(
                    f"UPDATE {table} SET {column} = '' WHERE id = 1"  # noqa: S608
                )
            )
        connection.rollback()


def _upgrade_cycle(engine: sa.Engine) -> None:
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.connect() as connection:
        _create_legacy_schema(connection)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        _assert_generation_schema(connection)
        first_catalog = set(_generations(connection, "public_mcp_apps", "generation"))
        first_associations = set(
            _generations(connection, "user_mcpservers", "lifecycle_generation")
        )

        command.downgrade(config, DOWN_REVISION)
        assert "generation" not in {
            item["name"]
            for item in sa.inspect(connection).get_columns("public_mcp_apps")
        }
        assert "lifecycle_generation" not in {
            item["name"]
            for item in sa.inspect(connection).get_columns("user_mcpservers")
        }

        command.upgrade(config, REVISION)
        _assert_generation_schema(connection)
        assert first_catalog.isdisjoint(
            _generations(connection, "public_mcp_apps", "generation")
        )
        assert first_associations.isdisjoint(
            _generations(connection, "user_mcpservers", "lifecycle_generation")
        )
        _assert_database_defaults(connection)
        connection.commit()

    with engine.connect() as connection:
        _assert_constraints(connection)


def test_sqlite_upgrade_backfills_constraints_and_round_trips() -> None:
    _upgrade_cycle(sa.create_engine("sqlite:///:memory:"))


@pytest.fixture()
def postgresql_engine_factory():
    with disposable_database_factory("xagent_mcp_generations") as make:
        yield make


@pytest.mark.postgresql
def test_postgresql_upgrade_backfills_constraints_and_round_trips(
    postgresql_engine_factory,
) -> None:
    _upgrade_cycle(postgresql_engine_factory("upgrade"))

    fresh_engine = postgresql_engine_factory("fresh_models")
    User.__table__.create(fresh_engine)
    MCPServer.__table__.create(fresh_engine)
    PublicMCPApp.__table__.create(fresh_engine)
    UserMCPServer.__table__.create(fresh_engine)
    with Session(fresh_engine) as db:
        user = User(username="postgres-generation", password_hash="x")
        server = MCPServer(
            name="postgres-generation",
            managed="external",
            transport="streamable_http",
            restart_policy="no",
        )
        app = PublicMCPApp(app_id="postgres-generation", name="Postgres generation")
        db.add_all([user, server, app])
        db.flush()
        association = UserMCPServer(user_id=user.id, mcpserver_id=server.id)
        db.add(association)
        db.commit()

        assert isinstance(app.generation, uuid.UUID)
        assert isinstance(association.lifecycle_generation, uuid.UUID)
