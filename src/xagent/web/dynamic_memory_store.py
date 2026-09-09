"""Dynamic memory store manager for web application."""

import logging
import os
import threading
from typing import Optional, Union, cast

from ..core.memory.in_memory import InMemoryMemoryStore
from ..core.memory.lancedb import LanceDBMemoryStore
from ..core.memory.lancedb_maintenance import (
    MaintenanceStatus,
    maintain_lancedb_memory_table,
)
from ..core.memory.vector_compatibility import (
    VectorCompatibility,
    canonical_embedding_identity,
    create_or_recreate_vector_capable_table,
)
from ..core.model import EmbeddingModelConfig
from ..core.storage.manager import get_storage_root
from ..core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .models.database import get_db
from .models.model import Model as DBModel
from .models.user import UserDefaultModel, UserModel
from .services.db_runtime import is_database_pool_timeout
from .user_isolated_memory import UserIsolatedMemoryStore

logger = logging.getLogger(__name__)

# Type alias for our memory store types that includes user isolation
MemoryStoreType = Union[
    InMemoryMemoryStore, LanceDBMemoryStore, UserIsolatedMemoryStore
]


def _embedding_model_fingerprint(model: Optional[DBModel]) -> Optional[tuple]:
    """Identity of an embedding model config, including reconfigurations.

    ``updated_at`` changes when the model row is edited (API key rotation,
    endpoint change), so comparing the fingerprint instead of only the id
    lets the store pick up new credentials without a backend restart.
    """
    if model is None:
        return None
    return (model.id, str(model.updated_at))


def _embedding_model_config(model: DBModel) -> EmbeddingModelConfig:
    """Preserve the complete shared embedding configuration and credential."""
    api_key = model.api_key
    return EmbeddingModelConfig(
        id=str(model.model_id),
        model_provider=str(model.model_provider),
        model_name=str(model.model_name),
        api_key=str(api_key) if api_key is not None else None,
        base_url=str(model.base_url) if model.base_url else None,
        dimension=int(model.dimension) if model.dimension is not None else None,
        instruct=getattr(model, "instruct", None),
        max_retries=int(model.max_retries) if model.max_retries is not None else 10,
    )


class DynamicMemoryStoreManager:
    """Dynamic memory store manager that supports lazy initialization and reconfiguration."""

    def __init__(self, similarity_threshold: Optional[float] = None):
        """
        Initialize the dynamic memory store manager.

        Args:
            similarity_threshold: Optional similarity threshold for vector search.
        """
        self._similarity_threshold = similarity_threshold
        self._memory_store: Optional[MemoryStoreType] = None
        self._lock = threading.RLock()
        self._last_embedding_model_id: Optional[int] = None
        # (id, updated_at) of the embedding model the store was built with.
        # Comparing the full fingerprint (not just the id) makes API key or
        # endpoint rotation on the same model take effect without a restart.
        self._last_embedding_model_fingerprint: Optional[tuple] = None
        self._is_lancedb: bool = False

        # Initialize with in-memory store (will be replaced with LanceDB when embedding model is configured)
        self._initialize_in_memory_store()

    def _initialize_in_memory_store(self) -> None:
        """Initialize with basic in-memory store."""
        with self._lock:
            in_memory_store = InMemoryMemoryStore()
            self._memory_store = UserIsolatedMemoryStore(in_memory_store)
            self._is_lancedb = False
            self._last_embedding_model_id = None
            self._last_embedding_model_fingerprint = None
            logger.info("Initialized with in-memory store")

    def _get_embedding_model_from_db(
        self, *, fail_fast: bool = False
    ) -> Optional[DBModel]:
        """Resolve one deterministic shared/admin default for the shared table."""
        try:
            db = next(get_db())
            try:
                from .services.model_service import _get_visible_user_ids

                default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .join(
                        UserModel,
                        (UserModel.model_id == DBModel.id)
                        & (UserModel.user_id == UserDefaultModel.user_id),
                    )
                    .filter(
                        UserDefaultModel.config_type == "embedding",
                        UserDefaultModel.user_id.in_(_get_visible_user_ids(db, None)),
                        UserModel.is_shared.is_(True),
                        DBModel.category == "embedding",
                        DBModel.is_active,
                    )
                    .order_by(UserDefaultModel.user_id, UserDefaultModel.id)
                    .first()
                )
                return default.model if default is not None else None
            finally:
                db.close()
        except Exception as e:
            if fail_fast or is_database_pool_timeout(e):
                raise
            logger.error(f"Error checking for embedding model: {e}")
            return None

    def _create_lancedb_store(
        self, embedding_model: DBModel
    ) -> UserIsolatedMemoryStore:
        """Create LanceDB store with the given embedding model."""
        legacy_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "memory_store",
        )
        if os.path.exists(legacy_dir) and os.listdir(legacy_dir):
            logger.info(f"Using legacy memory store location: {legacy_dir}")
            db_dir = legacy_dir
        else:
            new_dir = get_storage_root() / "memory_store"
            os.makedirs(new_dir, exist_ok=True)
            db_dir = str(new_dir)

        lancedb_store = LanceDBMemoryStore(
            db_dir=db_dir,
            embedding_model=_embedding_model_config(embedding_model),
            similarity_threshold=self._similarity_threshold or 1.5,
            initialize_schema=False,
            include_null_vector_fallback=True,
        )
        logger.info("Created LanceDB store with shared embedding model")
        return UserIsolatedMemoryStore(lancedb_store)

    def run_startup_compatibility_lifecycle(self) -> None:
        """Admit and maintain shared memory before runtime writers start."""
        with self._lock:
            model = self._get_embedding_model_from_db(fail_fast=True)
            if model is None:
                return
            embedding_config = _embedding_model_config(model)
            canonical_embedding_identity(embedding_config)
            new_store = self._create_lancedb_store(model)
            base_store = new_store._base_store
            if not isinstance(base_store, LanceDBMemoryStore):
                raise TypeError(
                    "startup memory lifecycle requires a LanceDB base store"
                )
            connection = base_store._vector_store.get_raw_connection()
            table_name = base_store._collection_name

            table = None
            try:
                try:
                    table = connection.open_table(table_name)
                except ValueError as error:
                    if "was not found" not in str(error):
                        raise
                if table is not None:
                    outcome = maintain_lancedb_memory_table(connection, table_name)
                    if outcome.status is MaintenanceStatus.INVALID_LEGACY_DATA:
                        base_store._embedding_model = None
                    elif outcome.status is not MaintenanceStatus.COMPLETE:
                        raise RuntimeError(
                            f"memory maintenance failed: {outcome.status.value}: {outcome.detail}"
                        )
            finally:
                _safe_close_table(table)

            if base_store._embedding_model is not None:
                compatibility = create_or_recreate_vector_capable_table(
                    connection, table_name, embedding_config
                )
                if compatibility is VectorCompatibility.MISMATCHING:
                    base_store._embedding_model = None
                elif table is None:
                    outcome = maintain_lancedb_memory_table(connection, table_name)
                    if outcome.status is not MaintenanceStatus.COMPLETE:
                        raise RuntimeError(
                            f"memory maintenance failed: {outcome.status.value}: {outcome.detail}"
                        )

            self._memory_store = new_store
            self._is_lancedb = True
            self._last_embedding_model_id = cast(int, model.id)
            self._last_embedding_model_fingerprint = _embedding_model_fingerprint(model)

    def _check_and_update_store(self) -> None:
        """Check if embedding model configuration has changed and update store accordingly."""
        with self._lock:
            embedding_model = self._get_embedding_model_from_db()
            current_model_id = embedding_model.id if embedding_model else None
            current_fingerprint = _embedding_model_fingerprint(embedding_model)

            # Check if we need to update the store
            should_update = False

            if embedding_model and not self._is_lancedb:
                # We have an embedding model but using in-memory store
                should_update = True
                logger.info("Embedding model detected, upgrading to LanceDB store")
            elif (
                embedding_model
                and self._is_lancedb
                and current_fingerprint != self._last_embedding_model_fingerprint
            ):
                # Embedding model changed, or the same model was reconfigured
                # (e.g. API key rotation) — rebuild so the new config is used.
                should_update = True
                logger.info(
                    "Embedding model configuration changed, updating LanceDB store"
                )
            elif not embedding_model and self._is_lancedb:
                # No embedding model available but using LanceDB (shouldn't happen normally)
                should_update = True
                logger.info(
                    "No embedding model available, falling back to in-memory store"
                )

            if should_update:
                if embedding_model:
                    try:
                        new_store = self._create_lancedb_store(embedding_model)
                    except Exception as error:
                        logger.error("Error creating LanceDB store: %s", error)
                        return
                    self._memory_store = new_store
                    self._is_lancedb = True
                    self._last_embedding_model_id = current_model_id  # type: ignore[assignment]
                    self._last_embedding_model_fingerprint = current_fingerprint
                    logger.info("Switched to LanceDB memory store")
                else:
                    self._initialize_in_memory_store()
                    logger.info("Switched to in-memory memory store")

    def get_memory_store(self) -> MemoryStoreType:
        """
        Get the current memory store, initializing or updating as necessary.

        Returns:
            Current memory store instance
        """
        return self._memory_store  # type: ignore[return-value]

    def force_reinitialize(self) -> None:
        """Force reinitialization of the memory store."""
        with self._lock:
            self._initialize_in_memory_store()
            self._check_and_update_store()
            logger.info("Force reinitialized memory store")

    def check_embedding_model_change(self) -> bool:
        """Check if embedding model configuration has changed and update if necessary.

        Returns:
            True if the store was updated, False otherwise.
        """
        with self._lock:
            old_is_lancedb = self._is_lancedb
            old_fingerprint = self._last_embedding_model_fingerprint

            self._check_and_update_store()

            # Return true if anything changed
            return (
                old_is_lancedb != self._is_lancedb
                or old_fingerprint != self._last_embedding_model_fingerprint
            )

    def get_store_info(self) -> dict:
        """
        Get information about the current memory store.

        Returns:
            Dictionary with store information
        """
        with self._lock:
            base_store = (
                self._memory_store._base_store
                if isinstance(self._memory_store, UserIsolatedMemoryStore)
                else self._memory_store
            )

            return {
                "store_type": type(base_store).__name__,
                "is_lancedb": self._is_lancedb,
                "embedding_model_id": self._last_embedding_model_id,
                "similarity_threshold": self._similarity_threshold,
                "supports_vector_search": self._is_lancedb,
            }


# Global instance
_dynamic_manager: Optional[DynamicMemoryStoreManager] = None
_manager_lock = threading.Lock()


def get_memory_store_manager(
    similarity_threshold: Optional[float] = None,
) -> DynamicMemoryStoreManager:
    """Get or create the global memory store manager."""
    global _dynamic_manager

    if _dynamic_manager is None:
        with _manager_lock:
            if _dynamic_manager is None:
                _dynamic_manager = DynamicMemoryStoreManager(similarity_threshold)

    return _dynamic_manager


def get_memory_store() -> MemoryStoreType:
    """Get the current memory store (for backward compatibility)."""
    manager = get_memory_store_manager()
    return manager.get_memory_store()


def force_reinitialize_memory_store() -> None:
    """Force reinitialization of the memory store."""
    manager = get_memory_store_manager()
    manager.force_reinitialize()
