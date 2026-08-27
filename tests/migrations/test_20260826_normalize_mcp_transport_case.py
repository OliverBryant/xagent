"""Tests for migration 20260826_normalize_mcp_transport_case.

Covers the backfill that lowercases/trims stored MCP `transport` values:
- mixed-case and padded rows are canonicalized in both affected tables
- already-canonical rows are left byte-identical (idempotent)
- NULL transports are preserved, not rewritten to ''
- the migration tolerates a database where one of the tables is absent

Following this repo's existing migration-test convention (see
tests/migrations/test_custom_api_migration.py): rather than replaying the
full alembic history, the minimal pre-migration schema is created directly
with raw DDL, stamped to this migration's parent revision, and only the
migration under test is run against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

PARENT_REVISION = "20260825_add_slack_channels_join_scope"
TARGET_REVISION = "20260826_normalize_mcp_transport_case"

_CREATE_MCP_SERVERS_TABLE = """
CREATE TABLE mcp_servers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    transport VARCHAR(50),
    url VARCHAR(500)
)
"""

_CREATE_PUBLIC_MCP_APPS_TABLE = """
CREATE TABLE public_mcp_apps (
    id INTEGER PRIMARY KEY,
    app_id VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    transport VARCHAR(50)
)
"""


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test.db'}"


def _stamp_parent(db_url: str, *, create_tables: tuple[str, ...]) -> Config:
    config = Config()
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("script_location", "src/xagent/migrations")

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO alembic_version (version_num) "
                f"VALUES ('{PARENT_REVISION}')"
            )
        )
        for ddl in create_tables:
            conn.execute(text(ddl))
    engine.dispose()
    return config


@pytest.fixture
def config_at_parent_revision(db_url: str) -> Config:
    return _stamp_parent(
        db_url,
        create_tables=(_CREATE_MCP_SERVERS_TABLE, _CREATE_PUBLIC_MCP_APPS_TABLE),
    )


def test_backfill_normalizes_mcp_servers_transport(
    db_url: str, config_at_parent_revision: Config
) -> None:
    engine = create_engine(db_url)
    with engine.begin() as conn:
        for row_id, transport in enumerate(
            ["Streamable_HTTP", " streamable_http ", "STDIO", "sse"], start=1
        ):
            conn.execute(
                text(
                    "INSERT INTO mcp_servers (id, name, transport) "
                    "VALUES (:id, :name, :transport)"
                ),
                {"id": row_id, "name": f"s{row_id}", "transport": transport},
            )
    engine.dispose()

    command.upgrade(config_at_parent_revision, TARGET_REVISION)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        stored = dict(conn.execute(text("SELECT id, transport FROM mcp_servers")).all())
    engine.dispose()

    assert stored == {
        1: "streamable_http",
        2: "streamable_http",
        3: "stdio",
        4: "sse",
    }


def test_backfill_normalizes_public_mcp_apps_transport(
    db_url: str, config_at_parent_revision: Config
) -> None:
    """The catalog table matters most: a shared catalog row's transport is
    never rewritten by the connect path, so without this backfill it would
    stay mixed-case indefinitely."""
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO public_mcp_apps (id, app_id, name, transport) "
                "VALUES (1, 'mixed', 'Mixed', 'Streamable_HTTP')"
            )
        )
    engine.dispose()

    command.upgrade(config_at_parent_revision, TARGET_REVISION)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        transport = conn.execute(
            text("SELECT transport FROM public_mcp_apps WHERE id = 1")
        ).scalar_one()
    engine.dispose()

    assert transport == "streamable_http"


def test_backfill_preserves_null_transport(
    db_url: str, config_at_parent_revision: Config
) -> None:
    """A NULL transport must stay NULL rather than become '' -- the WHERE
    clause is NULL-safe, so those rows are skipped entirely."""
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO mcp_servers (id, name, transport) VALUES (1, 's1', NULL)")
        )
    engine.dispose()

    command.upgrade(config_at_parent_revision, TARGET_REVISION)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        transport = conn.execute(
            text("SELECT transport FROM mcp_servers WHERE id = 1")
        ).scalar_one()
    engine.dispose()

    assert transport is None


def test_backfill_is_idempotent_on_already_canonical_rows(
    db_url: str, config_at_parent_revision: Config
) -> None:
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO mcp_servers (id, name, transport) "
                "VALUES (1, 's1', 'streamable_http')"
            )
        )
    engine.dispose()

    command.upgrade(config_at_parent_revision, TARGET_REVISION)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        transport = conn.execute(
            text("SELECT transport FROM mcp_servers WHERE id = 1")
        ).scalar_one()
    engine.dispose()

    assert transport == "streamable_http"


def test_backfill_tolerates_missing_table(db_url: str) -> None:
    """A database without one of the two tables (e.g. a partially initialized
    deployment) must not crash the upgrade."""
    config = _stamp_parent(db_url, create_tables=(_CREATE_MCP_SERVERS_TABLE,))

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO mcp_servers (id, name, transport) "
                "VALUES (1, 's1', 'Streamable_HTTP')"
            )
        )
    engine.dispose()

    command.upgrade(config, TARGET_REVISION)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        transport = conn.execute(
            text("SELECT transport FROM mcp_servers WHERE id = 1")
        ).scalar_one()
    engine.dispose()

    assert transport == "streamable_http"
