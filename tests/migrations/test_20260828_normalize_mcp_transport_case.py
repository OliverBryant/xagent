r"""Tests for migration 20260828_normalize_mcp_transport_case (#1831).

Covers the backfill that canonicalizes stored MCP `transport` values:

- mixed-case and whitespace-padded rows are canonicalized in both tables
- the padding grammar matches the write-side helper's ``str.strip().lower()``,
  including TAB/LF/CR padding that single-argument SQL ``TRIM()`` would miss
- a value outside the covered set is left byte-identical, so the migration can
  never reshape an unrecognized string into different garbage
- already-canonical rows are untouched, and a real re-run is a no-op
- NULL transports stay NULL rather than becoming ''
- a missing table, and a table without the column, are tolerated
- SQL is generated in offline (``--sql``) mode, where the bind is a
  MockConnection that cannot be reflected

Following this repo's migration-test convention (see
tests/migrations/test_20260813_trace_json_columns_to_jsonb.py): the minimal
pre-migration schema is built directly, stamped to the parent revision, and
only the migration under test is run against it. Everything substantive runs
under both SQLite and ``@pytest.mark.postgresql`` so the dialect branch in
``_normalized_expression`` (SQLite ``char()`` vs PostgreSQL ``chr()``) is
exercised on a real server rather than assumed.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

PARENT_REVISION = "20260826_seed_deputy_mcp_app"
TARGET_REVISION = "20260828_normalize_mcp_transport_case"

TABLES = ("mcp_servers", "public_mcp_apps")
ALL_TABLES = ("alembic_version",) + TABLES

# Whitespace that Python's str.strip() removes but single-argument SQL TRIM()
# does not. Written via chr() so no editor can silently rewrite them.
TAB = chr(9)
LF = chr(10)
CR = chr(13)


def _migration_module() -> ModuleType:
    import xagent.migrations as migrations_pkg

    migrations_dir = Path(next(iter(migrations_pkg.__path__)))
    path = migrations_dir / "versions" / f"{TARGET_REVISION}.py"
    spec = importlib.util.spec_from_file_location(TARGET_REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_like_helper(value: str) -> str:
    """The write-side helper's grammar, restated.

    ``normalize_transport()`` ships in the write-side layer (#1829), which this
    migration does not depend on at import time; restating its one-line body
    here keeps the test runnable against this branch alone while still pinning
    the migration's SQL to the helper's semantics rather than to itself.
    """
    return str(value or "").strip().lower()


def _pre_migration_metadata() -> sa.MetaData:
    """The two MCP tables reduced to the column the migration touches plus the
    identity columns -- not the full production schema."""
    metadata = sa.MetaData()
    sa.Table(
        "mcp_servers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("transport", sa.String(50)),
        # Carried although the migration never writes it: that a backfill of
        # `transport` leaves the rest of the row verbatim is a contract, and a
        # fixture without a second column cannot pin it.
        sa.Column("url", sa.String(500)),
    )
    sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("transport", sa.String(50)),
    )
    return metadata


def _stamp_parent_revision(engine: sa.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO alembic_version (version_num) VALUES "
                f"('{PARENT_REVISION}')"
            )
        )


def _alembic_config(engine: sa.engine.Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", "src/xagent/migrations")
    config.set_main_option(
        "sqlalchemy.url", engine.url.render_as_string(hide_password=False)
    )
    return config


def _upgrade(engine: sa.engine.Engine, revision: str = TARGET_REVISION) -> None:
    command.upgrade(_alembic_config(engine), revision)


def _insert(
    engine: sa.engine.Engine, table: str, rows: dict[int, str | None], **extra: str
) -> None:
    with engine.begin() as conn:
        for row_id, transport in rows.items():
            if table == "mcp_servers":
                conn.execute(
                    text(
                        "INSERT INTO mcp_servers (id, name, transport, url) "
                        "VALUES (:id, :name, :transport, :url)"
                    ),
                    {
                        "id": row_id,
                        "name": f"s{row_id}",
                        "transport": transport,
                        "url": extra.get("url", "https://example.test/mcp"),
                    },
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO public_mcp_apps (id, app_id, name, transport) "
                        "VALUES (:id, :app_id, :name, :transport)"
                    ),
                    {
                        "id": row_id,
                        "app_id": f"app-{row_id}",
                        "name": f"a{row_id}",
                        "transport": transport,
                    },
                )


def _stored(engine: sa.engine.Engine, table: str) -> dict[int, str | None]:
    with engine.begin() as conn:
        return dict(
            conn.execute(text(f"SELECT id, transport FROM {table} ORDER BY id")).all()
        )


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


@pytest.fixture
def sqlite_engine(tmp_path: Path) -> sa.engine.Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'transport.db'}")
    _pre_migration_metadata().create_all(bind=engine)
    _stamp_parent_revision(engine)
    return engine


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture
def postgres_engine():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    with engine.begin() as conn:
        for table in ALL_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    _pre_migration_metadata().create_all(bind=engine)
    _stamp_parent_revision(engine)
    yield engine
    with engine.begin() as conn:
        for table in ALL_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    engine.dispose()


# --------------------------------------------------------------------------
# Shared behavior, run against both dialects
# --------------------------------------------------------------------------

# id -> (stored value, expected value after the backfill). Expectations are
# written out literally rather than computed from the migration, so a change to
# the migration's own expression cannot silently move the target.
BACKFILL_CASES: dict[int, tuple[str | None, str | None]] = {
    1: ("Streamable_HTTP", "streamable_http"),
    2: (" streamable_http ", "streamable_http"),
    3: (TAB + "streamable_http", "streamable_http"),
    4: (LF + " STDIO " + CR + LF, "stdio"),
    5: ("sse", "sse"),
    6: ("WebSocket", "websocket"),
    7: (None, None),
    # Outside the covered set: left byte-identical, including its padding, so
    # the migration cannot reshape an unrecognized value into different garbage.
    8: ("Not-A-Transport", "Not-A-Transport"),
    9: (TAB + "str" + TAB + "eamable" + LF, TAB + "str" + TAB + "eamable" + LF),
    10: ("   ", "   "),
    # Interior padding characters that would *spell* a canonical transport if
    # they were deleted rather than translated to a space. The helper leaves
    # these alone, so the migration must too: deleting would let the backfill
    # invent 'stdio' out of a value no reader recognizes, which is precisely
    # what bounding the rewrite to the covered set is meant to prevent.
    11: ("st" + TAB + "dio", "st" + TAB + "dio"),
    12: ("streamable_" + LF + "http", "streamable_" + LF + "http"),
}


class _BackfillContract:
    """The behavior every supported dialect must share.

    Subclassed per dialect below so SQLite and PostgreSQL run the identical
    assertions -- the point of the dialect branch in ``_normalized_expression``
    is that they are indistinguishable from the caller's side.
    """

    def test_backfill_canonicalizes_mcp_servers(self, engine) -> None:
        _insert(engine, "mcp_servers", {k: v for k, (v, _) in BACKFILL_CASES.items()})

        _upgrade(engine)

        expected = {k: want for k, (_, want) in BACKFILL_CASES.items()}
        assert _stored(engine, "mcp_servers") == expected

    def test_backfill_canonicalizes_public_mcp_apps(self, engine) -> None:
        """The catalog table matters most: a shared catalog row's transport is
        never rewritten by the connect path, so without this backfill it would
        stay non-canonical indefinitely."""
        _insert(
            engine, "public_mcp_apps", {k: v for k, (v, _) in BACKFILL_CASES.items()}
        )

        _upgrade(engine)

        expected = {k: want for k, (_, want) in BACKFILL_CASES.items()}
        assert _stored(engine, "public_mcp_apps") == expected

    def test_padding_grammar_matches_the_write_side_helper(self, engine) -> None:
        """The whitespace axis, pinned against the helper rather than the SQL.

        A tab-padded row is the case single-argument SQL TRIM() misses: the
        runtime classifier normalizes it and admits the row to the OAuth token
        exchange, while the core serializer exact-matches, fails, and drops the
        row's url -- the same split classification this migration exists to
        remove, on the whitespace axis instead of the case axis.
        """
        padded = {
            1: TAB + "streamable_http",
            2: "streamable_http" + LF,
            3: CR + "SSE" + CR,
            4: TAB + " " + LF + "WebSocket" + CR + " ",
        }
        _insert(engine, "mcp_servers", padded)

        _upgrade(engine)

        stored = _stored(engine, "mcp_servers")
        assert stored == {
            row_id: _normalize_like_helper(value) for row_id, value in padded.items()
        }

    def test_backfill_leaves_other_columns_verbatim(self, engine) -> None:
        _insert(engine, "mcp_servers", {1: "Streamable_HTTP"}, url="https://kept/mcp")

        _upgrade(engine)

        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT transport, url, name FROM mcp_servers WHERE id = 1")
            ).one()
        assert tuple(row) == ("streamable_http", "https://kept/mcp", "s1")

    def test_backfill_is_idempotent_across_a_real_rerun(self, engine) -> None:
        """A genuine second run, not merely one pass over canonical rows.

        The migration is stamped after the first upgrade, so re-running it means
        stepping the version back and upgrading again -- otherwise this asserts
        nothing beyond the already-canonical case.
        """
        _insert(engine, "mcp_servers", {k: v for k, (v, _) in BACKFILL_CASES.items()})

        _upgrade(engine)
        after_first = _stored(engine, "mcp_servers")

        with engine.begin() as conn:
            conn.execute(
                text("UPDATE alembic_version SET version_num = :parent"),
                {"parent": PARENT_REVISION},
            )
        _upgrade(engine)

        assert _stored(engine, "mcp_servers") == after_first

    def test_null_transport_is_preserved(self, engine) -> None:
        """A NULL transport must stay NULL rather than become '': both WHERE
        comparisons are NULL-safe, so the row is skipped entirely."""
        _insert(engine, "mcp_servers", {1: None})

        _upgrade(engine)

        assert _stored(engine, "mcp_servers") == {1: None}


class TestSqliteBackfill(_BackfillContract):
    @pytest.fixture
    def engine(self, sqlite_engine: sa.engine.Engine) -> sa.engine.Engine:
        return sqlite_engine


@pytest.mark.postgresql
class TestPostgresBackfill(_BackfillContract):
    @pytest.fixture
    def engine(self, postgres_engine):
        return postgres_engine


# --------------------------------------------------------------------------
# Schema tolerance (SQLite is sufficient: the guard is dialect-independent
# reflection, and the online path is shared)
# --------------------------------------------------------------------------


def _engine_with_tables(tmp_path: Path, metadata: sa.MetaData) -> sa.engine.Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'partial.db'}")
    metadata.create_all(bind=engine)
    _stamp_parent_revision(engine)
    return engine


def test_backfill_tolerates_a_missing_table(tmp_path: Path) -> None:
    """A database without one of the two tables (a partially initialized
    deployment) must not crash the upgrade."""
    metadata = sa.MetaData()
    full = _pre_migration_metadata()
    full.tables["mcp_servers"].to_metadata(metadata)
    engine = _engine_with_tables(tmp_path, metadata)
    _insert(engine, "mcp_servers", {1: "Streamable_HTTP"})

    _upgrade(engine)

    assert _stored(engine, "mcp_servers") == {1: "streamable_http"}


def test_backfill_tolerates_a_table_without_the_transport_column(
    tmp_path: Path,
) -> None:
    """The column guard must actually fire: without a test, a typo in the table
    or column name would make the guard skip every table and the backfill would
    silently no-op in production."""
    metadata = sa.MetaData()
    sa.Table(
        "mcp_servers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
    )
    _pre_migration_metadata().tables["public_mcp_apps"].to_metadata(metadata)
    engine = _engine_with_tables(tmp_path, metadata)
    _insert(engine, "public_mcp_apps", {1: "Streamable_HTTP"})

    _upgrade(engine)

    # The column-less table was skipped rather than raising, and the other
    # table was still backfilled.
    assert _stored(engine, "public_mcp_apps") == {1: "streamable_http"}


# --------------------------------------------------------------------------
# Offline (--sql) generation
# --------------------------------------------------------------------------


def _offline_sql(dialect: str, operation: str = "upgrade") -> str:
    """Generate the migration's SQL the way ``alembic upgrade --sql`` does.

    Driven through ``command.upgrade(sql=True)`` and the repository's own
    ``env.py`` offline path rather than a hand-built MigrationContext, so the
    MockConnection that broke the previous draft is the one actually in play.
    """
    url = {
        "sqlite": "sqlite://",
        "postgresql": "postgresql://user:pw@localhost/db",
    }[dialect]
    output = StringIO()
    config = Config(stdout=output)
    config.set_main_option("script_location", "src/xagent/migrations")
    config.set_main_option("sqlalchemy.url", url)

    revisions = (
        f"{PARENT_REVISION}:{TARGET_REVISION}"
        if operation == "upgrade"
        else f"{TARGET_REVISION}:{PARENT_REVISION}"
    )
    # Alembic writes the offline script to the process stdout, not to
    # Config.stdout, so capture that as well as passing the buffer in.
    with redirect_stdout(output):
        getattr(command, operation)(config, revisions, sql=True)
    return output.getvalue()


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
class TestOfflineSql:
    def test_upgrade_emits_an_update_for_both_tables(self, dialect: str) -> None:
        """The regression this migration's offline branch exists for.

        Reflecting the bind raised NoInspectionAvailable on the MockConnection
        that ``alembic upgrade --sql`` supplies, so neither UPDATE was emitted
        and the artifact could not be generated at all.
        """
        sql = _offline_sql(dialect)

        for table in TABLES:
            assert f"UPDATE {table}" in sql

    def test_upgrade_emits_the_padding_translation(self, dialect: str) -> None:
        """Offline SQL must carry the same whitespace grammar as online, and
        must spell it with the dialect's own character function."""
        sql = _offline_sql(dialect)
        char_fn = {"sqlite": "char", "postgresql": "chr"}[dialect]

        for code_point in (9, 10, 13):
            assert f"{char_fn}({code_point})" in sql

    def test_upgrade_bounds_the_rewrite_to_the_covered_set(self, dialect: str) -> None:
        sql = _offline_sql(dialect)

        for value in _migration_module()._CANONICAL_TRANSPORTS:
            assert f"'{value}'" in sql

    def test_offline_sql_carries_no_literal_control_characters(
        self, dialect: str
    ) -> None:
        """A raw tab or newline inside a SQL string literal is a hazard for
        whoever applies the generated script by hand; the code points are
        spelled via the dialect's character function instead."""
        sql = _offline_sql(dialect)
        statements = [line for line in sql.splitlines() if TAB in line]

        assert statements == []
        assert CR not in sql

    def test_offline_sql_carries_no_bind_parameters(self, dialect: str) -> None:
        sql = _offline_sql(dialect)

        assert "%(" not in sql
        assert ":param" not in sql
        assert "?" not in sql

    def test_downgrade_touches_neither_data_table(self, dialect: str) -> None:
        """The downgrade is deliberately a no-op (the original spellings are
        not recorded anywhere). Alembic's own alembic_version bookkeeping is
        expected; neither MCP table may be written."""
        sql = _offline_sql(dialect, "downgrade")

        for table in TABLES:
            assert table not in sql
