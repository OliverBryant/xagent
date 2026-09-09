from __future__ import annotations

from pathlib import Path
import uuid

import sqlalchemy as sa
from alembic import command

from xagent.db.config import create_alembic_config

REVISION = "20260909_actor_mcp_connections"
DOWN_REVISION = "20260901_seed_zendesk_mcp_app"
TABLE = "actor_mcp_server_connections"
MIGRATIONS_DIR = Path(__file__).parents[2] / "src" / "xagent" / "migrations"


def _legacy_schema(connection: sa.Connection) -> None:
    connection.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY NOT NULL)"))
    connection.execute(
        sa.text(
            "CREATE TABLE public_mcp_apps ("
            "id INTEGER PRIMARY KEY NOT NULL, app_id VARCHAR(100) NOT NULL UNIQUE, "
            "generation CHAR(32) NOT NULL UNIQUE)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE TABLE alembic_version ("
            "version_num VARCHAR(255) NOT NULL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text("INSERT INTO alembic_version VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def test_sqlite_upgrade_constraints_and_downgrade() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.connect() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _legacy_schema(connection)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)

        inspector = sa.inspect(connection)
        columns = {item["name"]: item for item in inspector.get_columns(TABLE)}
        assert columns["resource_owner_key"]["type"].length == 512
        assert columns["app_id"]["type"].length == 100
        assert columns["encrypted_env"]["nullable"] is True
        assert columns["lifecycle_generation"]["nullable"] is False
        assert columns["lifecycle_generation"]["default"] is not None
        assert {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints(TABLE)
        } >= {
            ("lifecycle_generation",),
            ("user_id", "resource_owner_key", "app_id"),
            ("user_id", "resource_owner_key", "catalog_app_generation"),
        }
        foreign_keys = {
            tuple(item["constrained_columns"]): (
                item["referred_table"],
                item["options"].get("ondelete"),
            )
            for item in inspector.get_foreign_keys(TABLE)
        }
        assert foreign_keys == {
            ("catalog_app_generation",): ("public_mcp_apps", "CASCADE"),
            ("user_id",): ("users", "CASCADE"),
        }

        connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
        app_generation = uuid.uuid4()
        connection.execute(
            sa.text(
                "INSERT INTO public_mcp_apps (id, app_id, generation) "
                "VALUES (1, 'test-app', :generation)"
            ),
            {"generation": app_generation.hex},
        )
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(user_id, resource_owner_key, app_id, catalog_app_generation) "
                "VALUES (1, 'toby:test', 'test-app', :generation)"
            ),
            {"generation": app_generation.hex},
        )
        generation = uuid.UUID(
            str(connection.scalar(sa.text(f"SELECT lifecycle_generation FROM {TABLE}")))
        )
        assert generation.version == 4
        connection.execute(sa.text("DELETE FROM public_mcp_apps WHERE id = 1"))
        assert connection.scalar(sa.text(f"SELECT count(*) FROM {TABLE}")) == 0

        command.downgrade(config, DOWN_REVISION)
        assert TABLE not in sa.inspect(connection).get_table_names()

    engine.dispose()
