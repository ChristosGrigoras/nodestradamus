"""Tests for snapshot save/load and get_changes_since_last."""

from pathlib import Path

from nodestradamus.utils.snapshots import (
    diff_summaries,
    get_snapshot_dir,
    load_snapshot,
    save_snapshot,
)


class TestGetSnapshotDir:
    """Tests for get_snapshot_dir."""

    def test_uses_repo_hash_in_path(self, tmp_path: Path) -> None:
        """Snapshot dir should be workspace/.nodestradamus/snapshots/<repo_hash>."""
        workspace = tmp_path / "ws"
        repo = tmp_path / "repo"
        repo.mkdir()
        workspace.mkdir()
        d = get_snapshot_dir(workspace, repo)
        assert d == workspace / ".nodestradamus" / "snapshots" / d.name
        assert len(d.name) == 16

    def test_same_repo_same_hash(self, tmp_path: Path) -> None:
        """Same repo path should yield same snapshot dir name."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        repo = tmp_path / "repo"
        repo.mkdir()
        d1 = get_snapshot_dir(workspace, repo)
        d2 = get_snapshot_dir(workspace, repo)
        assert d1.name == d2.name


class TestSaveLoadSnapshot:
    """Tests for save_snapshot and load_snapshot."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Should persist and load summary with timestamp."""
        workspace = tmp_path / "ws"
        repo = tmp_path / "repo"
        workspace.mkdir()
        repo.mkdir()
        summary = {"total_nodes": 10, "files": ["a.py", "b.py"]}
        save_snapshot(workspace, repo, "analyze_deps", summary)
        loaded, saved_at = load_snapshot(workspace, repo, "analyze_deps")
        assert loaded is not None
        assert loaded["total_nodes"] == 10
        assert loaded["files"] == ["a.py", "b.py"]
        assert saved_at is not None

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        """Should return (None, None) when no snapshot exists."""
        workspace = tmp_path / "ws"
        repo = tmp_path / "repo"
        workspace.mkdir()
        repo.mkdir()
        loaded, saved_at = load_snapshot(workspace, repo, "project_scout")
        assert loaded is None
        assert saved_at is None


class TestDiffSummaries:
    """Tests for diff_summaries."""

    def test_identical_returns_empty_delta(self) -> None:
        """Same dict should yield no added/removed/changed."""
        d = {"a": 1, "b": [1, 2]}
        delta = diff_summaries(d, d)
        assert delta["added_keys"] == []
        assert delta["removed_keys"] == []
        assert delta["changed"] == {}

    def test_added_key(self) -> None:
        """New key in current should appear in added_keys."""
        prev = {"a": 1}
        cur = {"a": 1, "b": 2}
        delta = diff_summaries(cur, prev)
        assert "b" in delta["added_keys"]
        assert "a" not in delta["added_keys"]

    def test_removed_key(self) -> None:
        """Missing key in current should appear in removed_keys."""
        prev = {"a": 1, "b": 2}
        cur = {"a": 1}
        delta = diff_summaries(cur, prev)
        assert "b" in delta["removed_keys"]

    def test_changed_value(self) -> None:
        """Different value should appear in changed."""
        prev = {"a": 1}
        cur = {"a": 2}
        delta = diff_summaries(cur, prev)
        assert delta["changed"]["a"] == (1, 2)

    def test_list_added_removed_items(self) -> None:
        """List values should produce added_items and removed_items where applicable."""
        prev = {"files": ["a.py", "b.py"]}
        cur = {"files": ["a.py", "c.py"]}
        delta = diff_summaries(cur, prev)
        assert "files" in delta["added_items"] or "files" in delta["removed_items"]
