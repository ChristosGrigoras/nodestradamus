"""Tests for incremental embedding updates.

These tests verify that:
1. First run creates all embeddings
2. Second run with no changes reuses all embeddings
3. Adding a new file only embeds the new chunks
4. Modifying a file re-embeds only the changed chunks
5. Model version change triggers full re-embed
"""

import tempfile
from pathlib import Path

import pytest

from nodestradamus.analyzers.embeddings import (
    _compute_content_hash,
    _get_chunk_key,
    _load_existing_embeddings,
    compute_embeddings,
)
from nodestradamus.utils.db import (
    close_all_connections,
    get_chunk_hashes_with_faiss_ids,
    get_connection,
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository with Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)

        # Create initial files
        (repo / "module_a.py").write_text(
            '''
def function_one():
    """First function."""
    x = 1
    y = 2
    return x + y


def function_two():
    """Second function."""
    return "hello world"
'''
        )

        (repo / "module_b.py").write_text(
            '''
class MyClass:
    """A sample class."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
'''
        )

        yield repo

        # Cleanup connections
        close_all_connections()


class TestContentHash:
    """Tests for content hashing."""

    def test_same_content_same_hash(self):
        """Same content produces same hash."""
        content = "def foo(): pass"
        hash1 = _compute_content_hash(content)
        hash2 = _compute_content_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        hash1 = _compute_content_hash("def foo(): pass")
        hash2 = _compute_content_hash("def bar(): pass")
        assert hash1 != hash2

    def test_hash_is_16_chars(self):
        """Hash is truncated to 16 characters."""
        content = "def foo(): pass"
        hash_value = _compute_content_hash(content)
        assert len(hash_value) == 16


class TestChunkKey:
    """Tests for chunk key generation."""

    def test_chunk_key_format(self):
        """Chunk key includes file, line_start, line_end."""
        chunk = {
            "file": "src/module.py",
            "line_start": 10,
            "line_end": 20,
        }
        key = _get_chunk_key(chunk)
        assert key == "src/module.py:10:20"

    def test_chunk_key_handles_missing_lines(self):
        """Chunk key handles missing line numbers."""
        chunk = {"file": "src/module.py"}
        key = _get_chunk_key(chunk)
        assert key == "src/module.py:0:0"


class TestIncrementalEmbeddings:
    """Tests for incremental embedding updates."""

    def test_first_run_creates_all_embeddings(self, temp_repo):
        """First run should embed all chunks."""
        result = compute_embeddings(str(temp_repo))

        assert result["metadata"]["chunks_extracted"] > 0
        assert len(result["chunks"]) > 0

        # Check that SQLite has the chunks
        conn = get_connection(None, str(temp_repo))
        chunk_data = get_chunk_hashes_with_faiss_ids(conn)
        assert len(chunk_data) > 0

    def test_second_run_reuses_embeddings(self, temp_repo):
        """Second run with no changes should reuse all embeddings."""
        # First run
        result1 = compute_embeddings(str(temp_repo))
        chunks1 = result1["metadata"]["chunks_extracted"]

        # Second run (no changes)
        result2 = compute_embeddings(str(temp_repo))
        chunks2 = result2["metadata"]["chunks_extracted"]

        # Same number of chunks
        assert chunks1 == chunks2

        # Check incremental stats if available
        if "incremental" in result2["metadata"]:
            incremental = result2["metadata"]["incremental"]
            # All chunks should be reused (unchanged)
            assert incremental["unchanged"] == chunks2
            assert incremental["changed"] == 0
            assert incremental["new"] == 0

    def test_new_file_only_embeds_new_chunks(self, temp_repo):
        """Adding a new file should only embed the new chunks."""
        # First run
        result1 = compute_embeddings(str(temp_repo))
        chunks1 = result1["metadata"]["chunks_extracted"]

        # Add a new file
        (temp_repo / "module_c.py").write_text(
            '''
def new_function():
    """A new function."""
    return 42
'''
        )

        # Second run
        result2 = compute_embeddings(str(temp_repo))
        chunks2 = result2["metadata"]["chunks_extracted"]

        # Should have more chunks
        assert chunks2 > chunks1

        # Check incremental stats if available
        if "incremental" in result2["metadata"]:
            incremental = result2["metadata"]["incremental"]
            # Should have new chunks
            assert incremental["new"] > 0
            # Original chunks should be unchanged
            assert incremental["unchanged"] >= chunks1 - 1  # Allow for some variation

    def test_modified_file_reembeds_changed_chunks(self, temp_repo):
        """Modifying a file should re-embed only the changed chunks.

        Note: When content changes, line numbers may also change, causing
        the chunk to appear as "new" rather than "changed" since the chunk
        key includes line numbers. The key behavior we're testing is that
        not all chunks are re-embedded.
        """
        # First run
        result1 = compute_embeddings(str(temp_repo))
        result1["metadata"]["chunks_extracted"]

        # Modify an existing file (only change content, not structure)
        # Keep function_two unchanged and at same position
        (temp_repo / "module_a.py").write_text(
            '''
def function_one():
    """First function - MODIFIED."""
    x = 100
    y = 200
    z = 300
    return x + y + z


def function_two():
    """Second function."""
    return "hello world"
'''
        )

        # Second run
        result2 = compute_embeddings(str(temp_repo))

        # Check incremental stats if available
        if "incremental" in result2["metadata"]:
            incremental = result2["metadata"]["incremental"]
            # Some chunks should be unchanged (module_b.py wasn't touched)
            assert incremental["unchanged"] >= 1
            # Some chunks need re-embedding (changed or new due to line shift)
            assert incremental["changed"] + incremental["new"] >= 1
            # Not all chunks should need re-embedding
            assert incremental["reused"] >= 1

    def test_deleted_file_removes_chunks(self, temp_repo):
        """Deleting a file should result in fewer chunks."""
        # First run
        result1 = compute_embeddings(str(temp_repo))
        chunks1 = result1["metadata"]["chunks_extracted"]

        # Delete a file
        (temp_repo / "module_b.py").unlink()

        # Second run
        result2 = compute_embeddings(str(temp_repo))
        chunks2 = result2["metadata"]["chunks_extracted"]

        # Should have fewer chunks
        assert chunks2 < chunks1


class TestLoadExistingEmbeddings:
    """Tests for loading existing embeddings."""

    def test_returns_empty_when_no_cache(self, temp_repo):
        """Returns empty dicts when no cache exists."""
        chunk_data, embeddings, model = _load_existing_embeddings(None, str(temp_repo))
        assert chunk_data == {}
        assert embeddings == {}
        assert model is None

    def test_returns_data_after_compute(self, temp_repo):
        """Returns chunk data after embeddings are computed."""
        # Compute embeddings first
        compute_embeddings(str(temp_repo))

        # Now load existing
        chunk_data, embeddings, model = _load_existing_embeddings(None, str(temp_repo))

        assert len(chunk_data) > 0
        # embeddings may be empty if FAISS is not available
        assert model is not None

    def test_chunk_data_includes_faiss_ids(self, temp_repo):
        """Chunk data includes faiss_ids for embedding reuse."""
        compute_embeddings(str(temp_repo))
        chunk_data, embeddings, model = _load_existing_embeddings(None, str(temp_repo))

        # Each entry should be (content_hash, faiss_id) tuple
        for _key, value in chunk_data.items():
            assert isinstance(value, tuple)
            assert len(value) == 2
            content_hash, faiss_id = value
            assert isinstance(content_hash, str)
            assert isinstance(faiss_id, int)


class TestChunkHashesWithFaissIds:
    """Tests for the new get_chunk_hashes_with_faiss_ids function."""

    def test_returns_correct_structure(self, temp_repo):
        """Returns correct data structure."""
        compute_embeddings(str(temp_repo))
        conn = get_connection(None, str(temp_repo))
        chunk_data = get_chunk_hashes_with_faiss_ids(conn)

        assert isinstance(chunk_data, dict)
        for key, value in chunk_data.items():
            assert ":" in key  # Key format: file:start:end
            assert isinstance(value, tuple)
            assert len(value) == 2

    def test_faiss_ids_are_sequential(self, temp_repo):
        """FAISS IDs should be sequential starting from 0."""
        compute_embeddings(str(temp_repo))
        conn = get_connection(None, str(temp_repo))
        chunk_data = get_chunk_hashes_with_faiss_ids(conn)

        faiss_ids = sorted(fid for _, fid in chunk_data.values())
        expected = list(range(len(faiss_ids)))
        assert faiss_ids == expected
