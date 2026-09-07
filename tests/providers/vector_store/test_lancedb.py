"""
Tests for LanceDB vector store implementation.

This module tests both the connection management and vector store functionality
provided by the LanceDB provider.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa  # type: ignore
import pytest

from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
from xagent.providers.vector_store.lancedb import (
    LanceDBConnectionManager,
    LanceDBVectorStore,
    get_connection,
    get_connection_from_env,
)


@pytest.fixture
def temp_db_dir(tmp_path):
    """Create temporary database directory."""
    db_dir = tmp_path / "test_lancedb"
    db_dir.mkdir()
    return str(db_dir)


@pytest.fixture
def connection_manager():
    """Create connection manager instance."""
    return LanceDBConnectionManager()


class TestLanceDBConnectionManager:
    """Test LanceDB connection manager."""

    def test_get_connection_creates_directory(self, tmp_path, connection_manager):
        """Test that get_connection creates directory if it doesn't exist."""
        db_dir = str(tmp_path / "new_db")
        assert not Path(db_dir).exists()

        conn = connection_manager.get_connection(db_dir)
        assert conn is not None
        assert Path(db_dir).exists()

    def test_get_connection_caching(self, temp_db_dir, connection_manager):
        """Test connection caching."""
        # First call
        conn1 = connection_manager.get_connection(temp_db_dir)

        # Second call should return cached connection
        conn2 = connection_manager.get_connection(temp_db_dir)

        assert conn1 is conn2

    def test_get_connection_empty_dir_raises_error(self, connection_manager):
        """Test that empty directory path raises error."""
        with pytest.raises(
            ValueError, match="LanceDB directory path must be non-empty"
        ):
            connection_manager.get_connection("")

    def test_get_connection_from_env_success(self, temp_db_dir, connection_manager):
        """Test getting connection from environment variable."""
        with patch.dict(os.environ, {"TEST_LANCEDB_DIR": temp_db_dir}):
            conn = connection_manager.get_connection_from_env("TEST_LANCEDB_DIR")
            assert conn is not None

    def test_get_connection_from_env_missing_var(self, connection_manager):
        """Test error when environment variable is missing."""
        with pytest.raises(
            KeyError, match="Environment variable MISSING_VAR is not set"
        ):
            connection_manager.get_connection_from_env("MISSING_VAR")

    def test_get_connection_from_env_empty_var(self, connection_manager):
        """Test error when environment variable is empty."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            with pytest.raises(
                ValueError, match="Environment variable EMPTY_VAR is empty"
            ):
                connection_manager.get_connection_from_env("EMPTY_VAR")

    def test_connection_expiry(self, temp_db_dir, connection_manager):
        """Test connection expiry mechanism."""
        # Mock short TTL for testing
        with patch("xagent.providers.vector_store.lancedb.CONNECTION_TTL", 1):
            # Get initial connection
            connection_manager.get_connection(temp_db_dir)

            # Wait for expiry
            time.sleep(1.1)

            # Get connection again - should create new one
            conn2 = connection_manager.get_connection(temp_db_dir)

            # Note: We can't easily test if they're different instances
            # because LanceDB might return the same connection object
            # But we can verify the mechanism doesn't crash
            assert conn2 is not None


class TestLanceDBVectorStore:
    """Test LanceDB vector store implementation."""

    @pytest.fixture
    def vector_store(self, temp_db_dir):
        """Create vector store instance."""
        return LanceDBVectorStore(temp_db_dir, "test_vectors")

    def test_init_creates_table(self, temp_db_dir):
        """Test that initialization creates the vector table."""
        store = LanceDBVectorStore(temp_db_dir, "test_collection")
        assert store is not None

        # Verify table exists by trying to open it
        conn = store.get_raw_connection()
        table = conn.open_table("test_collection")
        assert table is not None

    @pytest.mark.parametrize("failure", [RuntimeError("open I/O"), OSError("disk I/O")])
    def test_ensure_table_does_not_hide_open_failures(self, failure):
        store = LanceDBVectorStore.__new__(LanceDBVectorStore)
        store._collection_name = "memory"
        store._conn = SimpleNamespace(
            open_table=lambda _name: (_ for _ in ()).throw(failure)
        )

        with pytest.raises(type(failure), match="I/O"):
            store._ensure_table()

    def test_concurrent_create_rejects_an_incompatible_winner(self):
        class Connection:
            def __init__(self):
                self.opens = 0

            def open_table(self, _name):
                self.opens += 1
                if self.opens == 1:
                    raise ValueError("Table 'memory' was not found")
                return SimpleNamespace(
                    schema=pa.schema([pa.field("text", pa.string())]),
                    close=lambda: None,
                )

            def create_table(self, _name, data):
                raise ValueError("Table 'memory' already exists")

        store = LanceDBVectorStore.__new__(LanceDBVectorStore)
        store._collection_name = "memory"
        store._conn = Connection()
        seed = pa.table({"id": ["sample"], "text": ["sample"]})

        with pytest.raises(ValueError, match="incompatible schema"):
            store._ensure_table(seed)

    def test_add_vectors_basic(self, vector_store):
        """Test basic vector addition."""
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        metadatas = [{"text": "first"}, {"text": "second"}]

        ids = vector_store.add_vectors(vectors, metadatas=metadatas)

        assert len(ids) == 2
        assert all(isinstance(id_, str) for id_ in ids)

    def test_add_vectors_with_ids(self, vector_store):
        """Test vector addition with custom IDs."""
        vectors = [[1.0, 2.0, 3.0]]
        ids = ["custom_id_1"]
        metadatas = [{"text": "test"}]

        returned_ids = vector_store.add_vectors(vectors, ids=ids, metadatas=metadatas)

        assert returned_ids == ids

    def test_search_vectors(self, vector_store):
        """Test vector search."""
        # Add some vectors
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        metadatas = [{"text": "x"}, {"text": "y"}, {"text": "z"}]

        vector_store.add_vectors(vectors, metadatas=metadatas)

        # Search for similar vector
        query_vector = [1.0, 0.1, 0.0]  # Should be closest to first vector
        results = vector_store.search_vectors(query_vector, top_k=2)

        assert len(results) <= 2
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert all("metadata" in result for result in results)

    def test_delete_vectors(self, vector_store):
        """Test vector deletion."""
        # Add vectors
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        metadatas = [{"text": "first"}, {"text": "second"}]

        ids = vector_store.add_vectors(vectors, metadatas=metadatas)

        # Delete first vector
        success = vector_store.delete_vectors([ids[0]])
        assert success is True

        # Search should return fewer results
        results = vector_store.search_vectors([1.0, 2.0, 3.0], top_k=10)
        remaining_ids = [r["id"] for r in results]
        assert ids[0] not in remaining_ids

    def test_clear_store(self, vector_store):
        """Test clearing the vector store."""
        # Add vectors
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        vector_store.add_vectors(vectors)

        # Clear store
        vector_store.clear()

        # Search should return no results
        results = vector_store.search_vectors([1.0, 2.0, 3.0], top_k=10)
        assert len(results) == 0

    def test_get_raw_connection(self, vector_store):
        """Test getting raw connection."""
        conn = vector_store.get_raw_connection()
        assert conn is not None

        # Should be able to use connection for advanced operations
        tables = list_table_names(conn)
        assert "test_vectors" in tables


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_connection(self, temp_db_dir):
        """Test get_connection convenience function."""
        conn = get_connection(temp_db_dir)
        assert conn is not None

    def test_get_connection_from_env(self, temp_db_dir):
        """Test get_connection_from_env convenience function."""
        with patch.dict(os.environ, {"TEST_DB_DIR": temp_db_dir}):
            conn = get_connection_from_env("TEST_DB_DIR")
            assert conn is not None


@pytest.mark.integration
class TestLanceDBIntegration:
    """Integration tests for LanceDB functionality."""

    def test_end_to_end_workflow(self, temp_db_dir):
        """Test complete workflow from creation to search."""
        # Create store
        store = LanceDBVectorStore(temp_db_dir, "integration_test")

        # Add vectors with metadata
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
        metadatas = [
            {"text": "red", "category": "color"},
            {"text": "green", "category": "color"},
            {"text": "blue", "category": "color"},
            {"text": "yellow", "category": "color"},
        ]

        ids = store.add_vectors(vectors, metadatas=metadatas)
        assert len(ids) == 4

        # Search for similar vectors
        query = [1.0, 0.1, 0.0]  # Should be closest to "red"
        results = store.search_vectors(query, top_k=2)

        assert len(results) <= 2
        assert results[0]["metadata"]["text"] in [
            "red",
            "yellow",
        ]  # Should find red or yellow

        # Delete one vector
        success = store.delete_vectors([ids[0]])
        assert success

        # Search again - should have fewer results
        results_after_delete = store.search_vectors(query, top_k=10)
        assert len(results_after_delete) == 3

        # Clear everything
        store.clear()
        results_after_clear = store.search_vectors(query, top_k=10)
        assert len(results_after_clear) == 0
