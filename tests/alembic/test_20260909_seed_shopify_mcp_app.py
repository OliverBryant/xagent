"""Tests for the provenance-safe Shopify connector seed."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260909_seed_shopify_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_shopify_migration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_tables(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL,
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE
            )
            """
        )
    )


def test_upgrade_inserts_provenance_and_personal_credential_metadata(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT transport, launch_config FROM public_mcp_apps "
                "WHERE app_id='shopify'"
            )
        ).first()

    assert row is not None
    assert row[0] == "stdio"
    assert '"credential_scope": "personal"' in row[1]
    assert '"builtin_provenance"' in row[1]


def test_upgrade_is_idempotent_for_provenance_owned_row(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert count == 1


def test_upgrade_refuses_custom_catalog_collision(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('shopify', 'Custom Shopify', 'stdio', '{\"command\": \"custom\"}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="custom public_mcp_apps"):
                migration.upgrade()
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert row == "Custom Shopify"


def test_upgrade_refuses_custom_server_name_collision(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(text("INSERT INTO mcp_servers (name) VALUES ('shopify')"))
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="custom mcp_servers"):
                migration.upgrade()
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert count == 0


def test_downgrade_only_deletes_provenance_owned_row(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert count == 0


def test_downgrade_preserves_custom_same_id(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as connection:
        _create_tables(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES ('shopify', 'Custom Shopify', 'stdio', '{\"command\": \"custom\"}')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        name = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='shopify'")
        ).scalar_one()
    assert name == "Custom Shopify"


def test_seed_row_matches_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration()
    registry = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "shopify"
    )
    assert migration.ROW == registry
