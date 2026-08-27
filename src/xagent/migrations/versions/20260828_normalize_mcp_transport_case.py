r"""Normalize stored MCP transport values to their canonical form

Revision ID: 20260828_normalize_mcp_transport_case
Revises: 20260826_seed_deputy_mcp_app
Create Date: 2026-08-28

`transport` is a free-form string on the MCP API models, so rows written
before the write-time normalizing validators shipped may hold a mixed-case or
whitespace-padded value (e.g. "Streamable_HTTP", "\tstreamable_http"). Such a
row is classified as connectable by the normalizing half of the MCP OAuth
feature and rejected by the exact-matching half: the web runtime treats it as
an HTTP server and runs the per-user token exchange, while the core
serializer's exact `transport in ["sse", "websocket", "streamable_http"]` test
fails and drops the row's `url` from the connection dict entirely. Backfill
the two web-layer tables once so the stored values agree with what every
reader expects.

Covered value set. A row is rewritten only when normalizing it yields one of
the four transports the application actually dispatches on (see
_CANONICAL_TRANSPORTS below). That bound makes the rewrite safe to audit: the
migration can only ever move a row onto a value the readers already agree on,
never invent one, and an unrecognized string is left exactly as stored rather
than being reshaped into different garbage.

Whitespace grammar. The write-side helper `normalize_transport()` canonicalizes
with Python's `str.strip().lower()`, which strips tabs and newlines as well as
ASCII spaces; single-argument SQL `TRIM()` strips only ASCII spaces. The
TAB/LF/CR characters are therefore translated to spaces before `TRIM` runs, so
padding written with them is stripped too. Translating rather than deleting
keeps `TRIM`'s edges-only semantics: an interior character is never removed,
and the covered-value-set bound above means an interior space cannot change
which row is eligible. Without this a tab-padded row survives the one-shot
backfill and stays in the split-classification state described above
indefinitely.

Rollout ordering. This is a one-shot UPDATE with no CHECK constraint or trigger
behind it, so it converges only if no application instance running the
pre-validator models is still serving admin writes when it lands; the startup
migration lock serializes migrators, not request sessions. The expand phase
that provides that guarantee is the two changes this migration is sequenced
behind:

  1. Write-side canonicalization (#1829) -- every write path stores a canonical
     transport, so no new non-canonical row is created.
  2. Read-side tolerance (#1830) -- web-layer reads normalize on read, so a
     not-yet-backfilled row behaves like its canonical equivalent.

Deploy 1 and 2, drain the old instances, then run this backfill. Run out of
order it is still safe -- it only ever rewrites a value to the form every
reader already agrees on -- but it is not guaranteed to leave the table
canonical, because a surviving old writer can insert a fresh mixed-case row
after the UPDATE commits.

Idempotent: the normalizing expression is a no-op on already-canonical values,
and the WHERE clause skips rows that are already normalized.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260828_normalize_mcp_transport_case"
down_revision = "20260826_seed_deputy_mcp_app"
branch_labels = None
depends_on = None

_TABLES = ("mcp_servers", "public_mcp_apps")

# The transports the application dispatches on. Mirrors the stdio branch and
# HTTP_MCP_TRANSPORTS in the MCP model/runtime; a row is only rewritten when it
# normalizes onto one of these.
_CANONICAL_TRANSPORTS = ("stdio", "sse", "websocket", "streamable_http")

# The whitespace characters Python's str.strip() removes that single-argument
# SQL TRIM() does not. Spelled as code points so no literal control character
# is embedded in the emitted SQL -- an offline (--sql) artifact carrying a raw
# newline inside a string literal is a hazard for whoever runs it by hand.
_PADDING_WHITESPACE = (9, 10, 13)  # TAB, LF, CR

# PostgreSQL spells the code-point-to-character function CHR(); SQLite spells it
# CHAR(). Both take one integer argument and need no extension.
_CHAR_FUNCTION = {"postgresql": "chr", "sqlite": "char"}


def _normalized_expression(dialect_name: str) -> str:
    """SQL canonicalizing `transport` the way normalize_transport() does.

    On a dialect whose character function is not known here this degrades to a
    bare LOWER(TRIM(...)): case and ASCII-space padding are still fixed, which
    is the behavior every supported dialect had before, so an unknown dialect
    gets a narrower backfill rather than invalid SQL.
    """
    char_fn = _CHAR_FUNCTION.get(dialect_name)
    expression = "transport"
    if char_fn is not None:
        for code_point in _PADDING_WHITESPACE:
            expression = f"REPLACE({expression}, {char_fn}({code_point}), ' ')"
    return f"LOWER(TRIM({expression}))"


def _tables_with_transport() -> list[str]:
    """The subset of _TABLES this database actually has, with the column.

    Online only: reflection needs a real bind. Offline callers use _TABLES
    directly, which is sound because the names are compile-time constants.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    present = []
    for table in _TABLES:
        if table not in existing:
            continue
        columns = {col["name"] for col in inspector.get_columns(table)}
        if "transport" in columns:
            present.append(table)
    return present


def _backfill(table: str, normalized: str) -> None:
    canonical_list = ", ".join(f"'{value}'" for value in _CANONICAL_TRANSPORTS)
    # NULL transports are left alone: both comparisons below are NULL-safe
    # (they evaluate to NULL, not true), so those rows are skipped rather than
    # rewritten to ''.
    op.execute(
        f"""
        UPDATE {table}
        SET transport = {normalized}
        WHERE transport IS NOT NULL
          AND transport <> {normalized}
          AND {normalized} IN ({canonical_list})
        """
    )


def upgrade() -> None:
    from alembic import context

    normalized = _normalized_expression(op.get_context().dialect.name)

    if context.is_offline_mode():
        # Offline (--sql) supplies a MockConnection that cannot be inspected,
        # so reflection is skipped and both UPDATEs are emitted unconditionally.
        # The table names are compile-time constants, and an UPDATE against a
        # table the target database lacks surfaces when the operator applies the
        # script -- the right place for it, since offline mode cannot know the
        # target's shape.
        for table in _TABLES:
            _backfill(table, normalized)
        return

    for table in _tables_with_transport():
        _backfill(table, normalized)


def downgrade() -> None:
    # Deliberately not reversible: the original mixed-case/padded spellings are
    # not recorded anywhere, and restoring them would reintroduce rows the
    # application cannot connect. Normalized values remain valid for every
    # earlier revision, so leaving them in place is safe.
    pass
