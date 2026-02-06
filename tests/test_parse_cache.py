"""Tests for parse cache functionality."""

import time
from pathlib import Path

import msgpack

from nodestradamus.analyzers.code_parser import (
    PARSE_CACHE_VERSION,
    FileCacheEntry,
    ParseCache,
    is_file_stale,
    load_parse_cache,
    parse_directory,
    save_parse_cache,
)


class TestFileCacheEntry:
    """Tests for FileCacheEntry dataclass."""

    def test_creates_entry(self) -> None:
        """Should create a FileCacheEntry with all fields."""
        entry = FileCacheEntry(
            mtime=1234567890.0,
            size=100,
            nodes=[{"id": "test", "name": "test"}],
            edges=[{"from": "a", "to": "b"}],
        )
        assert entry.mtime == 1234567890.0
        assert entry.size == 100
        assert len(entry.nodes) == 1
        assert len(entry.edges) == 1


class TestParseCache:
    """Tests for ParseCache dataclass."""

    def test_creates_cache(self) -> None:
        """Should create a ParseCache with version and files."""
        cache = ParseCache(
            version="1.0",
            created_at="2024-01-01T00:00:00Z",
            files={},
        )
        assert cache.version == "1.0"
        assert cache.files == {}


class TestLoadSaveCache:
    """Tests for cache loading and saving."""

    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        """Should save and load a cache successfully."""
        # Create a cache
        entry = FileCacheEntry(
            mtime=1234567890.0,
            size=42,
            nodes=[{"id": "py:test.py", "name": "test.py", "type": "module"}],
            edges=[],
        )
        cache = ParseCache(
            version=PARSE_CACHE_VERSION,
            created_at="2024-01-01T00:00:00Z",
            files={"test.py": entry},
        )

        # Save cache
        save_parse_cache(tmp_path, cache)

        # Verify cache file exists
        cache_path = tmp_path / ".nodestradamus" / "parse_cache.msgpack"
        assert cache_path.exists()

        # Load cache
        loaded = load_parse_cache(tmp_path)
        assert loaded is not None
        assert loaded.version == PARSE_CACHE_VERSION
        assert "test.py" in loaded.files
        assert loaded.files["test.py"].mtime == 1234567890.0
        assert loaded.files["test.py"].size == 42

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        """Should return None for nonexistent cache."""
        loaded = load_parse_cache(tmp_path)
        assert loaded is None

    def test_load_invalid_msgpack_cache(self, tmp_path: Path) -> None:
        """Should return None for invalid msgpack cache."""
        cache_dir = tmp_path / ".nodestradamus"
        cache_dir.mkdir(parents=True)
        (cache_dir / "parse_cache.msgpack").write_bytes(b"not valid msgpack")

        loaded = load_parse_cache(tmp_path)
        assert loaded is None

    def test_load_wrong_version_cache(self, tmp_path: Path) -> None:
        """Should return None for wrong version cache."""
        cache_dir = tmp_path / ".nodestradamus"
        cache_dir.mkdir(parents=True)
        # Write msgpack data with wrong version
        with (cache_dir / "parse_cache.msgpack").open("wb") as f:
            msgpack.pack({"version": "0.0", "created_at": "", "files": {}}, f)

        loaded = load_parse_cache(tmp_path)
        assert loaded is None


class TestIsFileStale:
    """Tests for file staleness checking."""

    def test_file_not_in_cache_is_stale(self, tmp_path: Path) -> None:
        """File not in cache should be considered stale."""
        cache = ParseCache(version="1.0", created_at="", files={})
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")

        assert is_file_stale(test_file, "test.py", cache) is True

    def test_unchanged_file_not_stale(self, tmp_path: Path) -> None:
        """Unchanged file should not be stale."""
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        stat = test_file.stat()

        entry = FileCacheEntry(
            mtime=stat.st_mtime,
            size=stat.st_size,
            nodes=[],
            edges=[],
        )
        cache = ParseCache(
            version="1.0",
            created_at="",
            files={"test.py": entry},
        )

        assert is_file_stale(test_file, "test.py", cache) is False

    def test_modified_file_is_stale(self, tmp_path: Path) -> None:
        """Modified file should be stale."""
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        old_stat = test_file.stat()

        entry = FileCacheEntry(
            mtime=old_stat.st_mtime,
            size=old_stat.st_size,
            nodes=[],
            edges=[],
        )
        cache = ParseCache(
            version="1.0",
            created_at="",
            files={"test.py": entry},
        )

        # Modify the file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("pass\npass")

        assert is_file_stale(test_file, "test.py", cache) is True

    def test_deleted_file_is_stale(self, tmp_path: Path) -> None:
        """Deleted file should be stale."""
        deleted_path = tmp_path / "deleted.py"
        entry = FileCacheEntry(
            mtime=12345.0,
            size=10,
            nodes=[],
            edges=[],
        )
        cache = ParseCache(
            version="1.0",
            created_at="",
            files={"deleted.py": entry},
        )

        assert is_file_stale(deleted_path, "deleted.py", cache) is True


class TestParseDirectoryWithCache:
    """Tests for parse_directory with caching."""

    def test_first_parse_creates_cache(self, tmp_path: Path) -> None:
        """First parse should create a cache file."""
        # Create a Python file
        (tmp_path / "module.py").write_text("def foo():\n    pass\n")

        # Parse directory
        result = parse_directory(tmp_path, use_cache=True)

        # Verify parsing worked
        assert result["metadata"]["file_count"] == 1
        assert result["metadata"]["parsed_count"] == 1
        assert result["metadata"]["cached_count"] == 0

        # Verify cache was created
        cache_path = tmp_path / ".nodestradamus" / "parse_cache.msgpack"
        assert cache_path.exists()

    def test_second_parse_uses_cache(self, tmp_path: Path) -> None:
        """Second parse should use cached results."""
        # Create a Python file
        (tmp_path / "module.py").write_text("def foo():\n    pass\n")

        # First parse
        result1 = parse_directory(tmp_path, use_cache=True)
        assert result1["metadata"]["parsed_count"] == 1
        assert result1["metadata"]["cached_count"] == 0

        # Second parse (should use cache)
        result2 = parse_directory(tmp_path, use_cache=True)
        assert result2["metadata"]["parsed_count"] == 0
        assert result2["metadata"]["cached_count"] == 1

        # Results should be the same
        assert len(result1["nodes"]) == len(result2["nodes"])
        assert len(result1["edges"]) == len(result2["edges"])

    def test_modified_file_reparsed(self, tmp_path: Path) -> None:
        """Modified file should be reparsed."""
        py_file = tmp_path / "module.py"
        py_file.write_text("def foo():\n    pass\n")

        # First parse
        result1 = parse_directory(tmp_path, use_cache=True)
        assert result1["metadata"]["parsed_count"] == 1

        # Modify the file
        time.sleep(0.01)  # Ensure mtime changes
        py_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        # Second parse (should reparse the modified file)
        result2 = parse_directory(tmp_path, use_cache=True)
        assert result2["metadata"]["parsed_count"] == 1
        assert result2["metadata"]["cached_count"] == 0

        # Should have more functions now
        functions1 = [n for n in result1["nodes"] if n["type"] == "function"]
        functions2 = [n for n in result2["nodes"] if n["type"] == "function"]
        assert len(functions2) > len(functions1)

    def test_added_file_parsed(self, tmp_path: Path) -> None:
        """Added file should be parsed."""
        # Create first file
        (tmp_path / "module1.py").write_text("def foo():\n    pass\n")

        # First parse
        result1 = parse_directory(tmp_path, use_cache=True)
        assert result1["metadata"]["file_count"] == 1

        # Add second file
        (tmp_path / "module2.py").write_text("def bar():\n    pass\n")

        # Second parse
        result2 = parse_directory(tmp_path, use_cache=True)
        assert result2["metadata"]["file_count"] == 2
        assert result2["metadata"]["parsed_count"] == 1  # Only the new file
        assert result2["metadata"]["cached_count"] == 1  # Old file from cache

    def test_deleted_file_removed_from_cache(self, tmp_path: Path) -> None:
        """Deleted file should be removed from cache."""
        # Create two files
        file1 = tmp_path / "module1.py"
        file2 = tmp_path / "module2.py"
        file1.write_text("def foo():\n    pass\n")
        file2.write_text("def bar():\n    pass\n")

        # First parse
        result1 = parse_directory(tmp_path, use_cache=True)
        assert result1["metadata"]["file_count"] == 2

        # Delete one file
        file2.unlink()

        # Second parse
        result2 = parse_directory(tmp_path, use_cache=True)
        assert result2["metadata"]["file_count"] == 1

        # Cache should only have one file now
        cache = load_parse_cache(tmp_path)
        assert cache is not None
        assert len(cache.files) == 1

    def test_cache_disabled(self, tmp_path: Path) -> None:
        """Should work with cache disabled."""
        (tmp_path / "module.py").write_text("def foo():\n    pass\n")

        # Parse with cache disabled
        result = parse_directory(tmp_path, use_cache=False)

        assert result["metadata"]["file_count"] == 1
        # No cache file should be created
        cache_path = tmp_path / ".nodestradamus" / "parse_cache.msgpack"
        assert not cache_path.exists()

    def test_mixed_languages(self, tmp_path: Path) -> None:
        """Should cache multiple languages correctly."""
        # Create files in different languages
        (tmp_path / "module.py").write_text("def foo():\n    pass\n")
        (tmp_path / "script.sh").write_text("#!/bin/bash\necho hello\n")

        # First parse
        result1 = parse_directory(tmp_path, use_cache=True)
        assert result1["metadata"]["file_count"] == 2
        assert result1["metadata"]["parsed_count"] == 2

        # Second parse (should use cache for both)
        result2 = parse_directory(tmp_path, use_cache=True)
        assert result2["metadata"]["file_count"] == 2
        assert result2["metadata"]["cached_count"] == 2
        assert result2["metadata"]["parsed_count"] == 0
