from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Union
from uuid import uuid4

import pyarrow as pa  # type: ignore

from ...providers.vector_store.lancedb import (
    LanceDBConnectionManager,
    LanceDBVectorStore,
)
from ..model.embedding import BaseEmbedding, DashScopeEmbedding
from ..model.embedding.adapter import create_embedding_adapter
from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .base import MemoryStore
from .core import MemoryNote, MemoryResponse
from .schema_migration import (
    MemoryMismatchKind,
    classify_memory_schema_mismatch,
    migrate_table_swap,
)

logger = logging.getLogger(__name__)


class LanceDBMemoryStore(MemoryStore):
    """LanceDB-based memory store implementation with vector search capabilities."""

    _embedding_model: Optional[BaseEmbedding]

    def __init__(
        self,
        db_dir: str,
        collection_name: str = "memories",
        embedding_model: Optional[Union[BaseEmbedding, EmbeddingModelConfig]] = None,
        similarity_threshold: float = 1.0,
        **embedding_kwargs: Any,
    ):
        """
        Initialize LanceDB memory store.

        Args:
            db_dir: Database directory path
            collection_name: Collection name for storing memories
            embedding_model: Optional BaseEmbedding instance or EmbeddingModel config
            similarity_threshold: Cosine distance threshold for vector search (lower = more strict)
            **embedding_kwargs: Additional arguments for embedding model
        """
        self._collection_name = collection_name

        # Handle different types of embedding_model input
        if embedding_model is None:
            # Try to create a default embedding model only if embedding_kwargs are provided
            if embedding_kwargs:
                try:
                    self._embedding_model = DashScopeEmbedding(**embedding_kwargs)
                except Exception:
                    # If embedding model creation fails, set to None (will use fallback)
                    self._embedding_model = None
                    logger.warning(
                        "Failed to create embedding model, will use fallback text search"
                    )
            else:
                self._embedding_model = None
                logger.info(
                    "No embedding model provided, will use fallback text search"
                )
        elif isinstance(embedding_model, BaseEmbedding):
            self._embedding_model = embedding_model
        elif isinstance(embedding_model, EmbeddingModelConfig):
            self._embedding_model = create_embedding_adapter(embedding_model)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {type(embedding_model)}"
            )
        self._similarity_threshold = similarity_threshold
        self._vector_store = LanceDBVectorStore(db_dir, collection_name)
        self._conn_manager = LanceDBConnectionManager()
        self._ensure_table_schema()

    def _ensure_table_schema(self) -> None:
        """Ensure the table has the correct schema for memory storage.

        If the table is missing a required column, migrate it in place
        (preserving all rows) instead of dropping and recreating it. This path
        runs on every store construction, so a wipe here would destroy data with
        no write in flight. On any migration failure the original table is left
        intact and the error propagates (out of ``__init__``); we never fall back
        to a wipe. Note the migration branch may perform a batched re-embed when
        a table is both missing a base column and vector-mismatched.
        """
        conn = self._vector_store.get_raw_connection()

        # Determine whether the table already exists and read its columns.
        table = None
        try:
            table = conn.open_table(self._collection_name)
            column_names = set(table.schema.names)
        except Exception:
            # Table doesn't exist yet, create it with the basic schema.
            logger.info(f"Creating table {self._collection_name} with basic schema")
            self._create_empty_table()
            return
        finally:
            _safe_close_table(table)

        # Table exists. Init's trigger is a missing required non-vector column;
        # a vector-dimension mismatch is detected and migrated lazily on the
        # add() path instead. Route the resolution through the shared classifier
        # and transform-then-swap primitive so we migrate rather than wipe.
        if not {"id", "text", "metadata"} <= column_names:
            logger.warning(
                f"Table {self._collection_name} has incompatible schema, "
                "migrating in place"
            )
            self._resolve_schema_mismatch(
                conn, self._current_embedding_dim(), raise_when_compatible=False
            )

    def _create_empty_table(self) -> None:
        """Create an empty table with the correct schema."""
        conn = self._vector_store.get_raw_connection()

        # Check if we have an embedding model
        if self._embedding_model:
            # Create table with vector support
            try:
                # Generate a sample embedding to get dimension
                sample_embedding = self._get_embedding("sample")
                if sample_embedding:
                    # Create sample data with vector
                    sample_data = [
                        {
                            "id": "sample",
                            "text": "sample",
                            "metadata": "{}",
                            "vector": sample_embedding,
                        }
                    ]
                else:
                    # Fallback to non-vector schema
                    sample_data = [{"id": "sample", "text": "sample", "metadata": "{}"}]
            except Exception:
                # If embedding fails, use non-vector schema
                sample_data = [{"id": "sample", "text": "sample", "metadata": "{}"}]
        else:
            # No embedding model, create without vector column
            sample_data = [{"id": "sample", "text": "sample", "metadata": "{}"}]

        # Create table with appropriate schema
        table = conn.create_table(self._collection_name, data=sample_data)
        # Remove sample data
        table.delete("id = 'sample'")

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text using the configured embedding model."""
        if not self._embedding_model or not text.strip():
            return None

        try:
            result = self._embedding_model.encode(text)
            # encode should return list[float] for single text input
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    # Got list[list[float]], return the first embedding
                    return result[0]
                elif len(result) > 0 and isinstance(result[0], (int, float)):
                    # Got list[float], return as is
                    return result  # type: ignore[return-value]
            logger.warning(f"Unexpected embedding result format: {type(result)}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {e}")
            return None

    def _current_embedding_dim(self) -> Optional[int]:
        """Return the vector dimension the store currently produces, or None.

        None means no embedding model is available and the store operates in
        vector-less (text-search) mode.
        """
        if not self._embedding_model:
            return None
        try:
            dim = self._embedding_model.get_dimension()
            if dim:
                return int(dim)
        except Exception:
            pass
        sample = self._get_embedding("sample")
        return len(sample) if sample else None

    def _embed_texts_batch(
        self, texts: list[str], target_dim: int
    ) -> list[list[float]]:
        """Re-embed all texts in a single batched encode call (all-or-nothing).

        Raises if no model is available, the batch shape is unexpected, or any
        row's embedding is missing or has the wrong dimension. The caller relies
        on this raising so the migration aborts with the original table intact
        (never a partially-vectorized table).
        """
        if not self._embedding_model:
            raise RuntimeError(
                "Cannot rebuild vector column without an embedding model"
            )
        if not texts:
            return []
        result = self._embedding_model.encode(texts)
        if not isinstance(result, list) or len(result) != len(texts):
            raise RuntimeError(
                f"Embedding batch returned unexpected shape for {len(texts)} rows"
            )
        vectors: list[list[float]] = []
        for index, vector in enumerate(result):
            if not isinstance(vector, list) or len(vector) != target_dim:
                raise RuntimeError(
                    f"Re-embedding row {index} failed or produced the wrong "
                    f"dimension (expected {target_dim})"
                )
            vectors.append([float(value) for value in vector])
        return vectors

    def _build_migrated_table(self, existing: Any, target_dim: Optional[int]) -> Any:
        """Transform for the migration primitive: rebuild rows at the target schema.

        Preserves every existing row's id/text/metadata. When ``target_dim`` is
        set, re-embeds all rows into a fresh fixed-width vector column; when it
        is None, produces a vector-less table (text-only search).
        """
        row_count = existing.num_rows
        names = set(existing.schema.names)

        def _string_column(name: str) -> Any:
            # Stay in the Arrow format and let PyArrow cast natively instead of
            # materializing Python lists per element.
            if name in names:
                return existing.column(name).cast(pa.string())
            return pa.array([None] * row_count, pa.string())

        columns: dict[str, Any] = {
            "id": _string_column("id"),
            "text": _string_column("text"),
            "metadata": _string_column("metadata"),
        }

        if target_dim is not None:
            # The batched embedding interface needs Python strings; reuse the
            # already-cast text column so it is only materialized once.
            texts = [text or "" for text in columns["text"].to_pylist()]
            vectors = self._embed_texts_batch(texts, target_dim)
            columns["vector"] = pa.array(vectors, pa.list_(pa.float32(), target_dim))

        return pa.table(columns)

    def _backfill_missing_columns(self, conn: Any, columns: tuple[str, ...]) -> None:
        """Add missing non-vector columns in place (no data loss, no rebuild)."""
        table = conn.open_table(self._collection_name)
        try:
            for column in columns:
                # All non-vector memory columns are strings.
                table.add_columns({column: "cast(null as string)"})
        finally:
            _safe_close_table(table)

    def _resolve_schema_mismatch(
        self, conn: Any, expected_dim: Optional[int], *, raise_when_compatible: bool
    ) -> None:
        """Classify and safely resolve a schema mismatch (shared by add/init).

        Missing non-vector columns are backfilled in place; a vector
        dimension/presence change rebuilds the table via transform-then-swap.
        On any failure the original table is left intact and the error
        propagates; no path drops or empties the table.

        When the schema is classified compatible, ``raise_when_compatible``
        controls behavior: the ``add()`` path passes ``True`` (its insert failed,
        so a compatible schema means an unexpected error to surface rather than
        silently drop); the init path passes ``False`` (nothing to migrate).
        """
        table = conn.open_table(self._collection_name)
        try:
            schema = table.schema
        finally:
            _safe_close_table(table)

        mismatch = classify_memory_schema_mismatch(schema, expected_dim)

        if mismatch.kind is MemoryMismatchKind.MISSING_NON_VECTOR_COLUMN:
            self._backfill_missing_columns(conn, mismatch.missing_columns)
        elif mismatch.kind is MemoryMismatchKind.VECTOR_REBUILD:
            target_dim = expected_dim
            migrate_table_swap(
                conn,
                self._collection_name,
                lambda existing: self._build_migrated_table(existing, target_dim),
            )
        elif raise_when_compatible:
            # add() failed but the schema looks compatible: do NOT drop the
            # table. Surface the original failure to the caller.
            raise RuntimeError(
                "add() failed but no resolvable schema mismatch was detected"
            )

    def _migrate_schema_mismatch(self, conn: Any, record: dict[str, Any]) -> None:
        """Resolve the schema mismatch that made an ``add()`` insert fail."""
        # The dimension we are trying to store now determines the target schema.
        if record.get("vector"):
            expected_dim: Optional[int] = len(record["vector"])
        elif self._embedding_model:
            expected_dim = self._current_embedding_dim()
        else:
            expected_dim = None

        self._resolve_schema_mismatch(conn, expected_dim, raise_when_compatible=True)

    def _insert_record(self, table: Any, record: dict[str, Any]) -> None:
        """Insert a record, adapting it to the (possibly migrated) table schema."""
        schema_names = set(table.schema.names)
        if "vector" in record and "vector" not in schema_names:
            record = {k: v for k, v in record.items() if k != "vector"}
        table.add([record])

    def _memory_note_to_dict(self, note: MemoryNote) -> dict[str, Any]:
        """Convert MemoryNote to dictionary for storage."""
        # Get embedding for the content
        content_text = (
            note.content.decode() if isinstance(note.content, bytes) else note.content
        )
        embedding = self._get_embedding(content_text)

        # Prepare metadata
        metadata = {
            "content": note.content,
            "keywords": note.keywords,
            "tags": note.tags,
            "category": note.category,
            "timestamp": note.timestamp.isoformat(),
            "mime_type": note.mime_type,
            **note.metadata,
        }

        return {
            "id": note.id,
            "vector": embedding,
            "text": note.content,
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }

    def _dict_to_memory_note(self, data: dict[str, Any]) -> MemoryNote:
        """Convert dictionary from storage to MemoryNote."""
        try:
            metadata = json.loads(data.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return MemoryNote(
            id=data.get("id"),
            content=metadata.pop("content", data.get("text", "")),
            keywords=metadata.pop("keywords", []),
            tags=metadata.pop("tags", []),
            category=metadata.pop("category", "general"),
            timestamp=metadata.pop("timestamp", None),
            mime_type=metadata.pop("mime_type", "text/plain"),
            metadata=metadata,
        )

    def _apply_filters(self, note: MemoryNote, filters: dict[str, Any]) -> bool:
        """Apply filters to a MemoryNote for vector search results."""
        for key, value in filters.items():
            # Special handling for category - check note.category first
            if key == "category":
                if str(note.category) != str(value):
                    return False
            elif key == "metadata":
                # Handle nested metadata filters
                if not self._apply_metadata_filters(note.metadata, value):
                    return False
            else:
                # For other fields, check metadata
                if str(note.metadata.get(key, "")) != str(value):
                    return False
        return True

    def _apply_text_search_filters(
        self, metadata_dict: dict[str, Any], filters: dict[str, Any]
    ) -> bool:
        """Apply filters to metadata dict for text search results."""
        for key, value in filters.items():
            if key == "metadata":
                # Handle nested metadata filters
                if not self._apply_metadata_filters(metadata_dict, value):
                    return False
            else:
                # Direct field comparison
                if str(metadata_dict.get(key, "")) != str(value):
                    return False
        return True

    def _apply_metadata_filters(
        self, metadata: dict[str, Any], metadata_filters: dict[str, Any]
    ) -> bool:
        """Apply nested metadata filters."""
        for key, value in metadata_filters.items():
            if str(metadata.get(key, "")) != str(value):
                return False
        return True

    def add(self, note: MemoryNote) -> MemoryResponse:
        """Add a memory note to the store."""
        try:
            # Generate ID if not provided
            if not note.id:
                note.id = str(uuid4())

            # Convert to storage format
            data = self._memory_note_to_dict(note)

            # Add to vector store - use a consistent approach
            conn = self._vector_store.get_raw_connection()
            table = None
            try:
                table = conn.open_table(self._collection_name)

                # Prepare record for insertion
                record = {
                    "id": data["id"],
                    "text": data["text"],
                    "metadata": data["metadata"],
                }

                # Add vector if available
                if data["vector"]:
                    record["vector"] = data["vector"]

                # Try to add the record. On a schema mismatch, migrate the
                # existing table in place (preserving all rows) instead of
                # dropping and recreating it.
                try:
                    table.add([record])
                except Exception as add_error:
                    logger.warning(
                        f"add() failed on possible schema mismatch: {add_error}; "
                        "attempting safe in-place migration"
                    )
                    _safe_close_table(table)
                    table = None
                    # Migrate safely; on any failure the original table is left
                    # intact and we surface an error WITHOUT dropping data.
                    try:
                        self._migrate_schema_mismatch(conn, record)
                    except Exception as migrate_error:
                        logger.error(
                            "Safe schema migration failed; table left intact: %s",
                            migrate_error,
                        )
                        return MemoryResponse(
                            success=False,
                            error=f"Failed to add memory: {migrate_error}",
                            memory_id=data["id"],
                        )
                    # Retry the insert against the migrated schema.
                    table = conn.open_table(self._collection_name)
                    self._insert_record(table, record)
            finally:
                _safe_close_table(table)

            return MemoryResponse(success=True, memory_id=data["id"])

        except Exception as e:
            logger.error(f"Failed to add memory note {note.id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to add memory: {str(e)}",
                memory_id=note.id,
            )

    def get(self, note_id: str) -> MemoryResponse:
        """Retrieve a memory note by its ID."""
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )

            # Search by ID
            results = table.search().where(f"id = '{note_id}'").to_pandas()

            if len(results) == 0:
                return MemoryResponse(
                    success=False,
                    error="Memory not found",
                    memory_id=note_id,
                )

            # Convert to MemoryNote
            data = results.iloc[0].to_dict()
            note = self._dict_to_memory_note(data)

            return MemoryResponse(
                success=True,
                memory_id=note_id,
                content=note,
            )

        except Exception as e:
            logger.error(f"Failed to get memory note {note_id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to get memory: {str(e)}",
                memory_id=note_id,
            )
        finally:
            _safe_close_table(table)

    def update(self, note: MemoryNote) -> MemoryResponse:
        """Update an existing memory note."""
        try:
            # Check if memory exists
            get_response = self.get(note.id)
            if not get_response.success:
                return MemoryResponse(
                    success=False,
                    error="Memory not found",
                    memory_id=note.id,
                )

            # Delete old record
            self.delete(note.id)

            # Add updated record
            return self.add(note)

        except Exception as e:
            logger.error(f"Failed to update memory note {note.id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to update memory: {str(e)}",
                memory_id=note.id,
            )

    def delete(self, note_id: str) -> MemoryResponse:
        """Delete a memory note by its ID."""
        try:
            success = self._vector_store.delete_vectors([note_id])

            if success:
                return MemoryResponse(success=True, memory_id=note_id)
            else:
                return MemoryResponse(
                    success=False,
                    error="Failed to delete memory",
                    memory_id=note_id,
                )

        except Exception as e:
            logger.error(f"Failed to delete memory note {note_id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to delete memory: {str(e)}",
                memory_id=note_id,
            )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[MemoryNote]:
        """Search memory notes by query text with optional filters."""
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            results = []

            # Try vector search first
            try:
                query_embedding = self._get_embedding(query)
                if query_embedding:
                    # Check if vector column exists and has the right dimension
                    sample_df = table.search().limit(1).to_pandas()
                    if not sample_df.empty and "vector" in sample_df.columns:
                        # Try vector search
                        try:
                            vector_df = (
                                table.search(
                                    query_embedding, vector_column_name="vector"
                                )
                                .limit(k)
                                .to_pandas()
                            )

                            for _, row in vector_df.iterrows():
                                # Check similarity threshold
                                threshold = (
                                    similarity_threshold
                                    if similarity_threshold is not None
                                    else self._similarity_threshold
                                )
                                distance = row.get("_distance", float("inf"))
                                if distance > threshold:
                                    logger.info(
                                        f"Skipping result with distance {distance} > threshold {threshold}"
                                    )
                                    continue

                                logger.info(
                                    f"Accepting result with distance {distance} <= threshold {threshold}"
                                )

                                note_data = {
                                    "id": row.get("id", ""),
                                    "text": row.get("text", ""),
                                    "metadata": row.get("metadata", "{}"),
                                }
                                note = self._dict_to_memory_note(note_data)

                                # Apply metadata filters to vector search results too
                                if filters:
                                    filter_match = self._apply_filters(note, filters)
                                    if not filter_match:
                                        continue

                                results.append(note)
                        except Exception as vector_error:
                            logger.warning(
                                f"Vector search failed, falling back to text search: {vector_error}"
                            )
            except Exception as embedding_error:
                logger.warning(
                    f"Embedding generation failed, using text search: {embedding_error}"
                )

            # Fallback to text search if no vector results or vector search failed
            if not results:
                # Text search
                df = table.search().to_pandas()

                # Filter by query text and apply filters
                for _, row in df.iterrows():
                    text = row.get("text", "")
                    metadata = row.get("metadata", "{}")

                    try:
                        metadata_dict = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata_dict = {}

                    # Apply metadata filters if specified
                    if filters:
                        filter_match = self._apply_text_search_filters(
                            metadata_dict, filters
                        )
                        if not filter_match:
                            continue

                    # Simple text matching
                    if not query or query.lower() in text.lower():
                        note_data = {
                            "id": row.get("id", ""),
                            "text": text,
                            "metadata": metadata,
                        }
                        note = self._dict_to_memory_note(note_data)
                        results.append(note)

                        if len(results) >= k:
                            break

            return results[:k]

        except Exception as e:
            logger.error(f"Failed to search memories with query '{query[:50]}...': {e}")
            return []
        finally:
            _safe_close_table(table)

    def clear(self) -> None:
        """Clear all memory notes from the store."""
        try:
            self._vector_store.clear()
        except Exception as e:
            logger.error(f"Failed to clear memory store: {e}")

    # Filters that search() does not understand and must be applied here.
    # Everything else (category, nested `metadata`, direct-field equality) is
    # delegated to search()'s proven filter path so list_all cannot diverge.
    _LIST_ONLY_FILTERS = frozenset({"date_from", "date_to", "tags", "keywords"})

    def _matches_list_only_filters(
        self, note: MemoryNote, filters: dict[str, Any]
    ) -> bool:
        """Apply the date-range/tags/keywords filters search() does not handle."""
        if "date_from" in filters and note.timestamp < filters["date_from"]:
            return False
        if "date_to" in filters and note.timestamp > filters["date_to"]:
            return False
        if "tags" in filters and not all(tag in note.tags for tag in filters["tags"]):
            return False
        if "keywords" in filters and not all(
            keyword in note.keywords for keyword in filters["keywords"]
        ):
            return False
        return True

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        """List all memory notes with optional filtering.

        Delegates category / nested-``metadata`` / direct-field filters to
        search() (its filter logic is the single source of truth, including the
        nested ``{"metadata": {...}}`` shape the user-isolation layer relies on),
        and applies the date-range/tags/keywords filters here. Results are sorted
        newest-first to match InMemoryStore.list_all.
        """
        try:
            filters = filters or {}
            # Let search() handle everything it understands...
            search_filters = {
                k: v for k, v in filters.items() if k not in self._LIST_ONLY_FILTERS
            }
            notes = self.search(query="", k=10000, filters=search_filters or None)
            # ...then apply the list-only filters it does not.
            list_only = {
                k: v for k, v in filters.items() if k in self._LIST_ONLY_FILTERS
            }
            if list_only:
                notes = [
                    n for n in notes if self._matches_list_only_filters(n, list_only)
                ]
            # Mirror InMemoryStore: newest first.
            notes.sort(key=lambda n: n.timestamp, reverse=True)
            return notes
        except Exception as e:
            logger.error(f"Failed to list all memories: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the memory store."""
        try:
            # Get all memories to calculate stats
            all_memories = self.list_all()

            total_count = len(all_memories)
            category_counts: dict[str, int] = {}
            tag_counts: dict[str, int] = {}

            for note in all_memories:
                # Count by category
                category_counts[note.category] = (
                    category_counts.get(note.category, 0) + 1
                )

                # Count tags
                for tag in note.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_count": total_count,
                "category_counts": category_counts,
                "tag_counts": tag_counts,
                "memory_store_type": "lancedb",
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "total_count": 0,
                "category_counts": {},
                "tag_counts": {},
                "memory_store_type": "lancedb",
                "error": str(e),
            }
