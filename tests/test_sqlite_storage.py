"""Tests for SQLite-based embeddings storage.

Tests the hybrid SQLite + FAISS storage system:
- SQLite stores chunk metadata
- FAISS stores vectors for similarity search
"""

import tempfile
from pathlib import Path

import pytest

from nodestradamus.utils.db import (
    bulk_insert_chunks,
    close_all_connections,
    delete_all_chunks,
    delete_chunks_by_file,
    get_all_chunks,
    get_chunk_by_faiss_id,
    get_chunk_count,
    get_chunk_hashes,
    get_chunks_by_faiss_ids,
    get_chunks_by_file,
    get_chunks_by_scope,
    get_connection,
    get_db_path,
    get_files_with_chunks,
    get_max_faiss_id,
    insert_chunk,
    transaction,
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def cleanup_connections():
    """Clean up database connections after each test."""
    yield
    close_all_connections()


class TestDatabaseSetup:
    """Tests for database initialization and connection."""

    def test_get_db_path(self, temp_repo: Path) -> None:
        """Test database path computation."""
        db_path = get_db_path(None, str(temp_repo))
        assert db_path == temp_repo / ".nodestradamus" / "nodestradamus.db"

    def test_get_db_path_with_workspace(self, temp_repo: Path) -> None:
        """Test database path with workspace isolation."""
        workspace = temp_repo / "workspace"
        repo = temp_repo / "repo"
        workspace.mkdir()
        repo.mkdir()

        db_path = get_db_path(str(workspace), str(repo))
        assert ".nodestradamus" in str(db_path)
        assert "cache" in str(db_path)
        assert db_path.name == "nodestradamus.db"

    def test_get_connection_creates_db(self, temp_repo: Path) -> None:
        """Test that get_connection creates database and schema."""
        conn = get_connection(None, str(temp_repo))
        assert conn is not None

        # Verify schema exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        )
        assert cursor.fetchone() is not None

    def test_transaction_commits_on_success(self, temp_repo: Path) -> None:
        """Test that successful transactions are committed."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(
                conn,
                file_path="test.py",
                symbol_name="test_func",
                line_start=1,
                line_end=10,
                content_hash="abc123",
                snippet="def test_func():",
                language="python",
                faiss_id=0,
                model_version="test-model",
            )

        # Verify in new connection
        conn = get_connection(None, str(temp_repo))
        chunks = get_all_chunks(conn)
        assert len(chunks) == 1

    def test_transaction_rolls_back_on_error(self, temp_repo: Path) -> None:
        """Test that failed transactions are rolled back."""
        try:
            with transaction(None, str(temp_repo)) as conn:
                insert_chunk(
                    conn,
                    file_path="test.py",
                    symbol_name="test_func",
                    line_start=1,
                    line_end=10,
                    content_hash="abc123",
                    snippet="def test_func():",
                    language="python",
                    faiss_id=0,
                    model_version="test-model",
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify nothing was committed
        conn = get_connection(None, str(temp_repo))
        chunks = get_all_chunks(conn)
        assert len(chunks) == 0


class TestChunkCRUD:
    """Tests for chunk CRUD operations."""

    def test_insert_chunk(self, temp_repo: Path) -> None:
        """Test inserting a single chunk."""
        with transaction(None, str(temp_repo)) as conn:
            chunk_id = insert_chunk(
                conn,
                file_path="src/main.py",
                symbol_name="main",
                line_start=1,
                line_end=20,
                content_hash="hash123",
                snippet="def main():",
                language="python",
                faiss_id=0,
                model_version="model-v1",
            )
            assert chunk_id > 0

    def test_bulk_insert_chunks(self, temp_repo: Path) -> None:
        """Test bulk inserting multiple chunks."""
        chunks = [
            {
                "file_path": "src/a.py",
                "symbol_name": "func_a",
                "line_start": 1,
                "line_end": 10,
                "content_hash": "hash_a",
                "snippet": "def func_a():",
                "language": "python",
                "faiss_id": 0,
                "model_version": "model-v1",
            },
            {
                "file_path": "src/b.py",
                "symbol_name": "func_b",
                "line_start": 1,
                "line_end": 15,
                "content_hash": "hash_b",
                "snippet": "def func_b():",
                "language": "python",
                "faiss_id": 1,
                "model_version": "model-v1",
            },
            {
                "file_path": "src/c.py",
                "symbol_name": "func_c",
                "line_start": 1,
                "line_end": 5,
                "content_hash": "hash_c",
                "snippet": "def func_c():",
                "language": "python",
                "faiss_id": 2,
                "model_version": "model-v1",
            },
        ]

        with transaction(None, str(temp_repo)) as conn:
            ids = bulk_insert_chunks(conn, chunks)
            assert len(ids) == 3
            assert all(id > 0 for id in ids)

    def test_get_chunks_by_file(self, temp_repo: Path) -> None:
        """Test retrieving chunks by file path."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(
                conn, "src/main.py", "func1", 1, 10, "h1", "...", "python", 0, "v1"
            )
            insert_chunk(
                conn, "src/main.py", "func2", 11, 20, "h2", "...", "python", 1, "v1"
            )
            insert_chunk(
                conn, "src/utils.py", "helper", 1, 5, "h3", "...", "python", 2, "v1"
            )

        conn = get_connection(None, str(temp_repo))
        main_chunks = get_chunks_by_file(conn, "src/main.py")
        assert len(main_chunks) == 2

        utils_chunks = get_chunks_by_file(conn, "src/utils.py")
        assert len(utils_chunks) == 1

    def test_get_chunks_by_scope(self, temp_repo: Path) -> None:
        """Test retrieving chunks by path prefix scope."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(
                conn, "src/auth/login.py", "login", 1, 10, "h1", "...", "python", 0, "v1"
            )
            insert_chunk(
                conn, "src/auth/logout.py", "logout", 1, 10, "h2", "...", "python", 1, "v1"
            )
            insert_chunk(
                conn, "src/api/routes.py", "routes", 1, 10, "h3", "...", "python", 2, "v1"
            )
            insert_chunk(
                conn, "tests/test_auth.py", "test", 1, 10, "h4", "...", "python", 3, "v1"
            )

        conn = get_connection(None, str(temp_repo))

        auth_chunks = get_chunks_by_scope(conn, "src/auth/")
        assert len(auth_chunks) == 2

        src_chunks = get_chunks_by_scope(conn, "src/")
        assert len(src_chunks) == 3

        test_chunks = get_chunks_by_scope(conn, "tests/")
        assert len(test_chunks) == 1

    def test_get_chunk_by_faiss_id(self, temp_repo: Path) -> None:
        """Test retrieving a chunk by FAISS ID."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(
                conn, "test.py", "func", 1, 10, "hash", "snippet", "python", 42, "v1"
            )

        conn = get_connection(None, str(temp_repo))
        chunk = get_chunk_by_faiss_id(conn, 42)
        assert chunk is not None
        assert chunk["file_path"] == "test.py"
        assert chunk["faiss_id"] == 42

        missing = get_chunk_by_faiss_id(conn, 999)
        assert missing is None

    def test_get_chunks_by_faiss_ids(self, temp_repo: Path) -> None:
        """Test retrieving multiple chunks by FAISS IDs."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "a", 1, 10, "h1", "...", "python", 0, "v1")
            insert_chunk(conn, "b.py", "b", 1, 10, "h2", "...", "python", 1, "v1")
            insert_chunk(conn, "c.py", "c", 1, 10, "h3", "...", "python", 2, "v1")

        conn = get_connection(None, str(temp_repo))
        chunks = get_chunks_by_faiss_ids(conn, [0, 2])
        assert len(chunks) == 2

        file_paths = {c["file_path"] for c in chunks}
        assert "a.py" in file_paths
        assert "c.py" in file_paths

    def test_get_all_chunks(self, temp_repo: Path) -> None:
        """Test retrieving all chunks."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "a", 1, 10, "h1", "...", "python", 0, "v1")
            insert_chunk(conn, "b.py", "b", 1, 10, "h2", "...", "python", 1, "v1")

        conn = get_connection(None, str(temp_repo))
        chunks = get_all_chunks(conn)
        assert len(chunks) == 2

    def test_get_all_chunks_filtered_by_model(self, temp_repo: Path) -> None:
        """Test filtering chunks by model version."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "a", 1, 10, "h1", "...", "python", 0, "v1")
            insert_chunk(conn, "b.py", "b", 1, 10, "h2", "...", "python", 1, "v2")

        conn = get_connection(None, str(temp_repo))
        v1_chunks = get_all_chunks(conn, model_version="v1")
        assert len(v1_chunks) == 1
        assert v1_chunks[0]["file_path"] == "a.py"


class TestChunkHashing:
    """Tests for content hash tracking."""

    def test_get_chunk_hashes(self, temp_repo: Path) -> None:
        """Test retrieving content hashes for change detection."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(
                conn, "a.py", "func_a", 1, 10, "hash_a", "...", "python", 0, "v1"
            )
            insert_chunk(
                conn, "b.py", "func_b", 5, 15, "hash_b", "...", "python", 1, "v1"
            )

        conn = get_connection(None, str(temp_repo))
        hashes = get_chunk_hashes(conn)

        assert "a.py:1:10" in hashes
        assert hashes["a.py:1:10"] == "hash_a"
        assert "b.py:5:15" in hashes
        assert hashes["b.py:5:15"] == "hash_b"


class TestChunkDeletion:
    """Tests for chunk deletion operations."""

    def test_delete_chunks_by_file(self, temp_repo: Path) -> None:
        """Test deleting chunks by file path."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "keep.py", "f1", 1, 10, "h1", "...", "python", 0, "v1")
            insert_chunk(conn, "delete.py", "f2", 1, 10, "h2", "...", "python", 1, "v1")
            insert_chunk(conn, "delete.py", "f3", 11, 20, "h3", "...", "python", 2, "v1")

        with transaction(None, str(temp_repo)) as conn:
            deleted = delete_chunks_by_file(conn, "delete.py")
            assert deleted == 2

        conn = get_connection(None, str(temp_repo))
        remaining = get_all_chunks(conn)
        assert len(remaining) == 1
        assert remaining[0]["file_path"] == "keep.py"

    def test_delete_all_chunks(self, temp_repo: Path) -> None:
        """Test deleting all chunks."""
        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "f1", 1, 10, "h1", "...", "python", 0, "v1")
            insert_chunk(conn, "b.py", "f2", 1, 10, "h2", "...", "python", 1, "v1")

        with transaction(None, str(temp_repo)) as conn:
            deleted = delete_all_chunks(conn)
            assert deleted == 2

        conn = get_connection(None, str(temp_repo))
        assert get_chunk_count(conn) == 0


class TestChunkMetadata:
    """Tests for chunk metadata queries."""

    def test_get_chunk_count(self, temp_repo: Path) -> None:
        """Test getting total chunk count."""
        conn = get_connection(None, str(temp_repo))
        assert get_chunk_count(conn) == 0

        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "f", 1, 10, "h", "...", "python", 0, "v1")
            insert_chunk(conn, "b.py", "g", 1, 10, "h", "...", "python", 1, "v1")

        conn = get_connection(None, str(temp_repo))
        assert get_chunk_count(conn) == 2

    def test_get_max_faiss_id(self, temp_repo: Path) -> None:
        """Test getting maximum FAISS ID."""
        conn = get_connection(None, str(temp_repo))
        assert get_max_faiss_id(conn) == -1  # Empty table

        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "f", 1, 10, "h", "...", "python", 5, "v1")
            insert_chunk(conn, "b.py", "g", 1, 10, "h", "...", "python", 10, "v1")

        conn = get_connection(None, str(temp_repo))
        assert get_max_faiss_id(conn) == 10

    def test_get_files_with_chunks(self, temp_repo: Path) -> None:
        """Test getting set of files with chunks."""
        conn = get_connection(None, str(temp_repo))
        assert get_files_with_chunks(conn) == set()

        with transaction(None, str(temp_repo)) as conn:
            insert_chunk(conn, "a.py", "f1", 1, 10, "h1", "...", "python", 0, "v1")
            insert_chunk(conn, "a.py", "f2", 11, 20, "h2", "...", "python", 1, "v1")
            insert_chunk(conn, "b.py", "g", 1, 10, "h3", "...", "python", 2, "v1")

        conn = get_connection(None, str(temp_repo))
        files = get_files_with_chunks(conn)
        assert files == {"a.py", "b.py"}
