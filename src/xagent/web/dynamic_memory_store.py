"""Dynamic memory store manager for web application."""

import logging
import os
import threading
from typing import Optional, Union

from ..core.memory.in_memory import InMemoryMemoryStore
from ..core.memory.lancedb import LanceDBMemoryStore
from ..core.model import EmbeddingModelConfig
from ..core.model.embedding import create_embedding_adapter
from ..core.storage.manager import get_storage_root
from .models.database import get_db
from .models.model import Model as DBModel
from .models.user import UserDefaultModel
from .services.db_runtime import is_database_pool_timeout
from .user_isolated_memory import UserIsolatedMemoryStore, current_user_id

logger = logging.getLogger(__name__)

MEMORY_BACKEND_UNAVAILABLE_REASON = "required_memory_backend_unavailable"


class MemoryBackendUnavailableError(RuntimeError):
    """Raised when an opt-in memory backend capability cannot be provided."""


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
        self._model_lookup_failed = False

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

    def _get_embedding_model_from_db(self) -> Optional[DBModel]:
        """Get the current embedding model from database."""
        try:
            db = next(get_db())
            try:
                # Get current user ID from context
                user_id = current_user_id.get()

                from .services.model_service import _is_model_visible_to_user

                if user_id:
                    # First, try to get user's default embedding model
                    user_default = (
                        db.query(UserDefaultModel)
                        .filter(
                            UserDefaultModel.user_id == user_id,
                            UserDefaultModel.config_type == "embedding",
                        )
                        .first()
                    )

                    if user_default:
                        # Get the actual model
                        embedding_model = (
                            db.query(DBModel)
                            .filter(
                                DBModel.id == user_default.model_id,
                                DBModel.category == "embedding",
                                DBModel.is_active,
                            )
                            .first()
                        )
                        if embedding_model:
                            if not _is_model_visible_to_user(
                                db, embedding_model.id, user_id
                            ):
                                logger.warning(
                                    f"User default embedding model {user_default.model_id} is no longer visible"
                                )
                                # fall through to system fallback
                            else:
                                logger.info(
                                    f"Found user's default embedding model: {embedding_model.model_id}"
                                )
                                return embedding_model
                        else:
                            logger.warning(
                                f"User default embedding model {user_default.model_id} not found or inactive"
                            )

                # Fallback: look for first active embedding model visible to user
                all_active_embeddings = (
                    db.query(DBModel)
                    .filter(
                        DBModel.category == "embedding",
                        DBModel.is_active,
                    )
                    .all()
                )

                for embedding_model in all_active_embeddings:
                    if _is_model_visible_to_user(db, embedding_model.id, user_id):
                        logger.info(
                            f"Using visible active embedding model: {embedding_model.model_id}"
                        )
                        return embedding_model

                logger.info("No visible active embedding model found")
                return None
            finally:
                db.close()
        except Exception as e:
            if is_database_pool_timeout(e):
                raise
            self._model_lookup_failed = True
            logger.error(f"Error checking for embedding model: {e}")
            return None

    def _create_lancedb_store(
        self, embedding_model: Optional[DBModel]
    ) -> UserIsolatedMemoryStore:
        """Create LanceDB store with the given embedding model."""
        # Check legacy location (project root) first for backward compatibility
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

        embedding_adapter = None
        if embedding_model is not None:
            embedding_adapter = create_embedding_adapter(
                EmbeddingModelConfig(
                    id=str(embedding_model.model_id),
                    model_name=embedding_model.model_name,
                    model_provider=embedding_model.model_provider,
                    api_key=str(embedding_model.api_key),
                    base_url=embedding_model.base_url,
                    dimension=embedding_model.dimension,
                )
            )
        lancedb_store = LanceDBMemoryStore(
            db_dir=db_dir,
            embedding_model=embedding_adapter,
            similarity_threshold=self._similarity_threshold or 1.5,
        )
        logger.info("Created LanceDB memory store")
        return UserIsolatedMemoryStore(lancedb_store)

    def _check_and_update_store(
        self,
        *,
        require_persistence: bool = False,
        require_vector_search: bool = False,
    ) -> None:
        """Check if embedding model configuration has changed and update store accordingly."""
        with self._lock:
            strict = require_persistence or require_vector_search
            self._model_lookup_failed = False
            try:
                embedding_model = self._get_embedding_model_from_db()
            except Exception as exc:
                if strict:
                    raise MemoryBackendUnavailableError(
                        MEMORY_BACKEND_UNAVAILABLE_REASON
                    ) from exc
                raise
            if strict and self._model_lookup_failed:
                raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)
            current_model_id = embedding_model.id if embedding_model else None
            current_fingerprint = _embedding_model_fingerprint(embedding_model)

            # Check if we need to update the store
            should_update = False

            if embedding_model and (
                not self._is_lancedb or self._last_embedding_model_fingerprint is None
            ):
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
            elif not embedding_model and self._is_lancedb and not require_persistence:
                # No embedding model available but using LanceDB (shouldn't happen normally)
                should_update = True
                logger.info(
                    "No embedding model available, falling back to in-memory store"
                )

            elif not embedding_model and require_persistence and not self._is_lancedb:
                should_update = True

            if require_vector_search and embedding_model is None:
                raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

            if should_update:
                if embedding_model or require_persistence:
                    try:
                        memory_store = self._create_lancedb_store(embedding_model)
                    except Exception as exc:
                        logger.exception("Error creating LanceDB memory store")
                        self._initialize_in_memory_store()
                        if strict:
                            raise MemoryBackendUnavailableError(
                                MEMORY_BACKEND_UNAVAILABLE_REASON
                            ) from exc
                        return
                    self._memory_store = memory_store
                    self._is_lancedb = True
                    self._last_embedding_model_id = current_model_id
                    self._last_embedding_model_fingerprint = current_fingerprint
                    logger.info("Switched to LanceDB memory store")
                else:
                    self._initialize_in_memory_store()
                    logger.info("Switched to in-memory memory store")

    def get_memory_store(
        self,
        *,
        require_persistence: bool = False,
        require_vector_search: bool = False,
    ) -> MemoryStoreType:
        """
        Get the current memory store, initializing or updating as necessary.

        Returns:
            Current memory store instance
        """
        self._check_and_update_store(
            require_persistence=require_persistence,
            require_vector_search=require_vector_search,
        )
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
                "supports_vector_search": (
                    self._is_lancedb
                    and self._last_embedding_model_fingerprint is not None
                ),
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


def get_memory_store(
    *,
    require_persistence: bool = False,
    require_vector_search: bool = False,
) -> MemoryStoreType:
    """Get the current memory store (for backward compatibility)."""
    manager = get_memory_store_manager()
    return manager.get_memory_store(
        require_persistence=require_persistence,
        require_vector_search=require_vector_search,
    )


def force_reinitialize_memory_store() -> None:
    """Force reinitialization of the memory store."""
    manager = get_memory_store_manager()
    manager.force_reinitialize()
