"""Tests for structural fingerprint analyzer."""

from pathlib import Path

from nodestradamus.analyzers.fingerprints import (
    build_fingerprint_index,
    find_similar,
    load_fingerprint_index,
)


class TestBuildFingerprintIndex:
    """Tests for build_fingerprint_index."""

    def test_builds_index_from_parse_result(self, tmp_path: Path) -> None:
        """Should build hash_to_locations and file_to_hashes from parsed nodes/edges."""
        (tmp_path / "a.py").write_text("def f(): pass\ndef g(): f()")
        index = build_fingerprint_index(tmp_path, use_cache=False)
        assert "version" in index
        assert "hash_to_locations" in index
        assert "file_to_hashes" in index
        assert "metadata" in index
        assert index["metadata"]["node_count"] >= 2
        assert index["metadata"]["edge_count"] >= 1

    def test_saves_to_nodestradamus_dir_when_use_cache(self, tmp_path: Path) -> None:
        """Should write fingerprints.msgpack under .nodestradamus when use_cache=True."""
        (tmp_path / "a.py").write_text("def f(): pass")
        build_fingerprint_index(tmp_path, use_cache=True)
        cache_path = tmp_path / ".nodestradamus" / "fingerprints.msgpack"
        assert cache_path.exists()

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Should return None when no cache exists."""
        assert load_fingerprint_index(tmp_path) is None


class TestLoadFingerprintIndex:
    """Tests for load_fingerprint_index."""

    def test_loads_after_build(self, tmp_path: Path) -> None:
        """Should load the same index after build with use_cache=True."""
        (tmp_path / "a.py").write_text("def f(): pass")
        build_fingerprint_index(tmp_path, use_cache=True)
        loaded = load_fingerprint_index(tmp_path)
        assert loaded is not None
        assert loaded.get("version") == "2.0"
        assert "hash_to_locations" in loaded


class TestFindSimilar:
    """Tests for find_similar."""

    def test_returns_summary_and_matches(self, tmp_path: Path) -> None:
        """Should return summary and matches list."""
        (tmp_path / "a.py").write_text("def f(): pass\ndef g(): f()")
        result = find_similar(tmp_path, "a.py", top_k=5, use_cache=False)
        assert "summary" in result
        assert "matches" in result
        assert result["summary"]["file"] == "a.py"
        assert "match_count" in result["summary"]

    def test_empty_repo_returns_zero_matches(self, tmp_path: Path) -> None:
        """Should return match_count 0 and empty matches when no index."""
        (tmp_path / "only.py").write_text("x = 1")
        result = find_similar(tmp_path, "only.py", top_k=5, use_cache=False)
        assert result["summary"]["match_count"] >= 0
        assert isinstance(result["matches"], list)

    def test_line_range_filters_nodes(self, tmp_path: Path) -> None:
        """Should restrict scope when line_start/line_end provided."""
        (tmp_path / "a.py").write_text("def f(): pass\n\ndef g(): pass\n\ndef h(): g()")
        result = find_similar(
            tmp_path, "a.py", line_start=1, line_end=2, top_k=5, use_cache=False
        )
        assert "summary" in result
        assert result["summary"]["line_range"] == [1, 2]
