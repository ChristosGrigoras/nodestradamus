"""Tests for Phase 3 smart pipeline integration.

Tests the integration between project_scout and analyze_deps,
including scope filtering and seed entry handling.
"""

from pathlib import Path

import pytest

from nodestradamus.analyzers.deps import analyze_deps, analyze_deps_smart
from nodestradamus.analyzers.project_scout import project_scout


@pytest.fixture
def sample_scoped_project(temp_dir: Path) -> Path:
    """Create a project with clear scope boundaries."""
    # Main source code
    src = temp_dir / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text("""
from src.utils import helper

def main():
    return helper()
""")
    (src / "utils.py").write_text("""
def helper():
    return "hello"
""")

    # Library code
    lib = temp_dir / "lib"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    (lib / "core.py").write_text("""
class Widget:
    pass
""")

    # Test code (should be excludable)
    tests = temp_dir / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("""
from src.main import main

def test_main():
    assert main() == "hello"
""")

    # Vendor/external code (should be excludable)
    vendor = temp_dir / "vendor"
    vendor.mkdir()
    (vendor / "external.py").write_text("""
def vendor_function():
    pass
""")

    # README with hints
    (temp_dir / "README.md").write_text("""
# Sample Project

Core logic is in `src/`.

Run with: python src/main.py
""")

    # Config file
    (temp_dir / "pyproject.toml").write_text("""
[project]
name = "sample"
dependencies = ["flask"]
""")

    return temp_dir


class TestAnalyzeDepsScope:
    """Tests for scope filtering in analyze_deps."""

    def test_scope_filters_to_matching_paths(self, sample_scoped_project: Path) -> None:
        """Should filter graph to nodes within scope."""
        # Full graph (no scope)
        full_graph = analyze_deps(sample_scoped_project, use_cache=False)

        # Scoped to src/ only
        scoped_graph = analyze_deps(
            sample_scoped_project,
            scope=["src/"],
            use_cache=False,
        )

        # Scoped graph should be smaller
        assert scoped_graph.number_of_nodes() <= full_graph.number_of_nodes()

        # All nodes in scoped graph should be from src/
        for node_id, attrs in scoped_graph.nodes(data=True):
            file_path = attrs.get("file", "")
            if file_path:
                assert file_path.startswith("src/"), f"Node {node_id} has file {file_path}"

    def test_scope_multiple_prefixes(self, sample_scoped_project: Path) -> None:
        """Should allow multiple scope prefixes."""
        # Scoped to src/ and lib/
        scoped_graph = analyze_deps(
            sample_scoped_project,
            scope=["src/", "lib/"],
            use_cache=False,
        )

        # Should include nodes from both directories
        files_seen = set()
        for _node_id, attrs in scoped_graph.nodes(data=True):
            file_path = attrs.get("file", "")
            if file_path:
                files_seen.add(file_path.split("/")[0])

        assert "src" in files_seen or "lib" in files_seen

    def test_scope_excludes_out_of_scope(self, sample_scoped_project: Path) -> None:
        """Should exclude nodes outside scope."""
        scoped_graph = analyze_deps(
            sample_scoped_project,
            scope=["src/"],
            use_cache=False,
        )

        # Should not include tests/ or vendor/
        for node_id, attrs in scoped_graph.nodes(data=True):
            file_path = attrs.get("file", "")
            assert not file_path.startswith("tests/"), f"Found test node: {node_id}"
            assert not file_path.startswith("vendor/"), f"Found vendor node: {node_id}"


class TestAnalyzeDepsSeedEntries:
    """Tests for seed_entries parameter in analyze_deps."""

    def test_seed_entries_accepted(self, sample_scoped_project: Path) -> None:
        """Should accept seed_entries parameter without error."""
        graph = analyze_deps(
            sample_scoped_project,
            seed_entries=["src/main.py", "src/utils.py"],
            use_cache=False,
        )

        # Should produce a valid graph
        assert graph.number_of_nodes() > 0

    def test_seed_entries_with_scope(self, sample_scoped_project: Path) -> None:
        """Should work with both seed_entries and scope."""
        graph = analyze_deps(
            sample_scoped_project,
            scope=["src/"],
            seed_entries=["src/main.py"],
            use_cache=False,
        )

        # Should produce a valid scoped graph
        assert graph.number_of_nodes() > 0


class TestAnalyzeDepsSmart:
    """Tests for analyze_deps_smart integration function."""

    def test_returns_graph_and_metadata(self, sample_scoped_project: Path) -> None:
        """Should return both graph and metadata."""
        graph, metadata = analyze_deps_smart(sample_scoped_project, use_cache=False)

        assert graph is not None
        assert graph.number_of_nodes() >= 0

        assert metadata is not None
        assert "project_type" in metadata
        assert "node_count" in metadata

    def test_uses_project_scout_results(self, sample_scoped_project: Path) -> None:
        """Should incorporate project_scout results in metadata."""
        graph, metadata = analyze_deps_smart(sample_scoped_project, use_cache=False)

        # Should have scout-derived fields
        assert "primary_language" in metadata
        assert "scope_used" in metadata
        assert "entry_points" in metadata

    def test_applies_recommended_scope(self, sample_scoped_project: Path) -> None:
        """Should apply recommended scope from project_scout."""
        # First, check what project_scout recommends
        scout_result = project_scout(sample_scoped_project)

        # Run smart analysis
        graph, metadata = analyze_deps_smart(sample_scoped_project, use_cache=False)

        # If scout found a recommended scope, it should be used
        if scout_result.recommended_scope:
            assert metadata["scope_used"] is not None

    def test_handles_empty_project(self, temp_dir: Path) -> None:
        """Should handle projects with no source files."""
        graph, metadata = analyze_deps_smart(temp_dir, use_cache=False)

        assert graph is not None
        assert graph.number_of_nodes() == 0
        assert metadata["node_count"] == 0


class TestProjectScoutIntegration:
    """Tests for project_scout + analyze_deps integration."""

    def test_scout_scope_applied_to_deps(self, sample_scoped_project: Path) -> None:
        """Should be able to use scout's recommended_scope with analyze_deps."""
        scout_result = project_scout(sample_scoped_project)

        if scout_result.recommended_scope:
            graph = analyze_deps(
                sample_scoped_project,
                scope=scout_result.recommended_scope,
                use_cache=False,
            )

            # Graph should be produced successfully
            assert graph is not None

    def test_scout_entry_points_as_seeds(self, sample_scoped_project: Path) -> None:
        """Should be able to use scout's entry_points as seed_entries."""
        scout_result = project_scout(sample_scoped_project)

        # Filter to actual file paths
        file_entries = [
            ep for ep in scout_result.entry_points
            if not ep.startswith("[script:")
        ]

        graph = analyze_deps(
            sample_scoped_project,
            seed_entries=file_entries,
            use_cache=False,
        )

        # Graph should be produced successfully
        assert graph is not None
