"""Explicit, resumable maintenance for existing LanceDB memory tables."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import pyarrow as pa  # type: ignore
from filelock import FileLock, Timeout

from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .scope_columns import (
    SCOPE_DIMS_COLUMN,
    USER_ID_COLUMN,
    derive_scope_columns,
)

MAINTENANCE_METADATA_KEY = b"xagent.memory.scope_maintenance"
MAINTENANCE_VERSION = b"1"
DEFAULT_BATCH_SIZE = 512
DEFAULT_LOCK_TIMEOUT = 10.0


class MaintenanceStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID_LEGACY_DATA = "invalid_legacy_data"


@dataclass(frozen=True)
class MaintenanceOutcome:
    status: MaintenanceStatus
    scanned_rows: int = 0
    updated_rows: int = 0
    cas_skipped_rows: int = 0
    batches_committed: int = 0
    detail: str | None = None


class MaintenanceLockTimeout(RuntimeError):
    """Raised when another process holds a table's maintenance lock too long."""


def _checkpoint(_stage: str, _batch: int | None = None) -> None:
    """Test seam for deterministic interruption; intentionally does nothing."""


def _is_complete(schema: Any) -> bool:
    names = set(schema.names)
    if not {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= names:
        return False
    metadata = schema.field(USER_ID_COLUMN).metadata or {}
    return metadata.get(MAINTENANCE_METADATA_KEY) == MAINTENANCE_VERSION


def _lock_path(connection: Any, table_name: str) -> str:
    uri = str(getattr(connection, "uri", "") or "")
    if not uri or "://" in uri or not os.path.isdir(uri):
        raise ValueError("LanceDB maintenance requires a writable local database URI")
    digest = hashlib.sha256(table_name.encode()).hexdigest()[:16]
    return os.path.join(uri, f".memory-maintenance-{digest}.lock")


def _read_rows(table: Any) -> list[dict[str, Any]]:
    names = set(table.schema.names)
    required = {"id", "metadata"}
    if not required <= names:
        missing = ", ".join(sorted(required - names))
        raise ValueError(f"memory table is missing required columns: {missing}")
    projected = ["id", "metadata"]
    projected += [c for c in (USER_ID_COLUMN, SCOPE_DIMS_COLUMN) if c in names]
    return cast(
        list[dict[str, Any]],
        table.search().select(projected).limit(None).to_arrow().to_pylist(),
    )


def _invalid_ids(rows: list[dict[str, Any]]) -> str | None:
    ids = [row["id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        return "legacy IDs must be non-empty strings"
    if len(ids) != len(set(ids)):
        return "legacy IDs must be unique"
    return None


def _needs_update(row: dict[str, Any]) -> bool:
    expected = derive_scope_columns(row["metadata"])
    return (row.get(USER_ID_COLUMN), row.get(SCOPE_DIMS_COLUMN)) != expected


def _source(rows: list[dict[str, Any]]) -> object:
    derived = [derive_scope_columns(row["metadata"]) for row in rows]
    return cast(
        object,
        pa.table(
            {
                "id": pa.array([row["id"] for row in rows], pa.string()),
                "metadata": pa.array([row["metadata"] for row in rows], pa.string()),
                USER_ID_COLUMN: pa.array([item[0] for item in derived], pa.int64()),
                SCOPE_DIMS_COLUMN: pa.array(
                    [item[1] for item in derived], pa.list_(pa.string())
                ),
            }
        ),
    )


def _backfill_batch(table: Any, rows: list[dict[str, Any]]) -> int:
    condition = (
        "(target.metadata = source.metadata) OR "
        "((target.metadata IS NULL) AND (source.metadata IS NULL))"
    )
    result = (
        table.merge_insert("id")
        .when_matched_update_all(where=condition)
        .execute(_source(rows))
    )
    return int(result.num_updated_rows)


def maintain_lancedb_memory_table(
    connection: Any,
    table_name: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> MaintenanceOutcome:
    """Backfill scope projections under an explicit, serialized admin boundary."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise ValueError("lock_timeout must be a finite positive duration")

    lock = FileLock(_lock_path(connection, table_name), timeout=lock_timeout)
    try:
        lock.acquire()
    except Timeout as exc:
        raise MaintenanceLockTimeout(
            f"Timed out after {lock_timeout}s acquiring maintenance lock for "
            f"table {table_name!r}; stop the other maintenance process and retry"
        ) from exc

    table = None
    try:
        table = connection.open_table(table_name)
        if _is_complete(table.schema):
            return MaintenanceOutcome(MaintenanceStatus.COMPLETE)

        rows = _read_rows(table)
        invalid = _invalid_ids(rows)
        if invalid:
            return MaintenanceOutcome(
                MaintenanceStatus.INVALID_LEGACY_DATA,
                scanned_rows=len(rows),
                detail=invalid,
            )

        names = set(table.schema.names)
        missing = [
            field
            for field in (
                pa.field(USER_ID_COLUMN, pa.int64()),
                pa.field(SCOPE_DIMS_COLUMN, pa.list_(pa.string())),
            )
            if field.name not in names
        ]
        if missing:
            table.add_columns(missing)
            _checkpoint("columns_added")

        candidates = [row for row in rows if missing or _needs_update(row)]
        updated = commits = skipped = 0
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            changed = _backfill_batch(table, batch)
            updated += changed
            skipped += len(batch) - changed
            commits += 1
            _checkpoint("batch_committed", commits)

        final_rows = _read_rows(table)
        final_invalid = _invalid_ids(final_rows)
        remaining = sum(_needs_update(row) for row in final_rows)
        if final_invalid or skipped or remaining:
            return MaintenanceOutcome(
                MaintenanceStatus.INCOMPLETE,
                scanned_rows=len(rows),
                updated_rows=updated,
                cas_skipped_rows=skipped,
                batches_committed=commits,
                detail=final_invalid or "concurrent changes require another pass",
            )

        _checkpoint("before_completion")
        table.update_field_metadata(
            {
                "path": USER_ID_COLUMN,
                "metadata": {
                    MAINTENANCE_METADATA_KEY.decode(): MAINTENANCE_VERSION.decode()
                },
            }
        )
        return MaintenanceOutcome(
            MaintenanceStatus.COMPLETE,
            scanned_rows=len(rows),
            updated_rows=updated,
            batches_committed=commits,
        )
    finally:
        _safe_close_table(table)
        lock.release()
