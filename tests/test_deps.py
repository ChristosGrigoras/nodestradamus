"""Tests for unified dependency analyzer."""

from pathlib import Path

import networkx as nx
import pytest

from nodestradamus.analyzers.deps import (
    _build_graph,
    _detect_languages,
    _merge_graphs,
    analyze_deps,
    graph_metadata,
)


class TestDetectLanguages:
    """Tests for language auto-detection."""

    def test_detects_python(self, tmp_path: Path) -> None:
        """Should detect Python files."""
        (tmp_path / "app.py").write_text("print('hello')")

        languages = _detect_languages(tmp_path)

        assert "python" in languages
        assert "typescript" not in languages

    def test_detects_typescript(self, tmp_path: Path) -> None:
        """Should detect TypeScript files."""
        (tmp_path / "app.ts").write_text("console.log('hello')")

        languages = _detect_languages(tmp_path)

        assert "typescript" in languages
        assert "python" not in languages

    def test_detects_both(self, tmp_path: Path) -> None:
        """Should detect both Python and TypeScript."""
        (tmp_path / "app.py").write_text("print('hello')")
        (tmp_path / "app.ts").write_text("console.log('hello')")

        languages = _detect_languages(tmp_path)

        assert "python" in languages
        assert "typescript" in languages

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Should not detect files in node_modules."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "lib.js").write_text("export const x = 1")

        languages = _detect_languages(tmp_path)

        assert "typescript" not in languages

    def test_returns_empty_for_no_source_files(self, tmp_path: Path) -> None:
        """Should return empty list for non-source files."""
        (tmp_path / "readme.md").write_text("# Hello")

        languages = _detect_languages(tmp_path)

        assert languages == []


class TestBuildGraph:
    """Tests for building NetworkX graph from raw data."""

    def test_creates_nodes_with_attributes(self) -> None:
        """Should create nodes with all attributes."""
        nodes = [
            {
                "id": "py:app.py::main",
                "type": "function",
                "file": "app.py",
                "name": "main",
                "line": 10,
            }
        ]

        G = _build_graph(nodes, [])

        assert "py:app.py::main" in G.nodes()
        attrs = G.nodes["py:app.py::main"]
        assert attrs["type"] == "function"
        assert attrs["file"] == "app.py"
        assert attrs["name"] == "main"
        assert attrs["line"] == 10

    def test_creates_directed_edges(self) -> None:
        """Should create directed edges with attributes."""
        nodes = [
            {"id": "a", "type": "function", "file": "a.py", "name": "a"},
            {"id": "b", "type": "function", "file": "b.py", "name": "b"},
        ]
        edges = [
            {"from": "a", "to": "b", "type": "calls", "resolved": True},
        ]

        G = _build_graph(nodes, edges)

        assert G.has_edge("a", "b")
        assert not G.has_edge("b", "a")  # Directed
        assert G["a"]["b"]["type"] == "calls"
        assert G["a"]["b"]["resolved"] is True

    def test_handles_empty_input(self) -> None:
        """Should return empty graph for empty input."""
        G = _build_graph([], [])

        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestMergeGraphs:
    """Tests for merging multiple graphs."""

    def test_merges_nodes_and_edges(self) -> None:
        """Should combine nodes and edges from all graphs."""
        G1 = nx.DiGraph()
        G1.add_node("a", type="function")
        G1.add_edge("a", "b", type="calls")

        G2 = nx.DiGraph()
        G2.add_node("c", type="class")
        G2.add_edge("c", "d", type="imports")

        merged = _merge_graphs(G1, G2)

        assert "a" in merged.nodes()
        assert "c" in merged.nodes()
        assert merged.has_edge("a", "b")
        assert merged.has_edge("c", "d")

    def test_preserves_attributes(self) -> None:
        """Should preserve node and edge attributes."""
        G1 = nx.DiGraph()
        G1.add_node("a", type="function", file="a.py")
        G1.add_edge("a", "b", type="calls", resolved=True)

        merged = _merge_graphs(G1)

        assert merged.nodes["a"]["type"] == "function"
        assert merged["a"]["b"]["resolved"] is True

    def test_deduplicates_nodes(self) -> None:
        """Should not duplicate nodes that exist in multiple graphs."""
        G1 = nx.DiGraph()
        G1.add_node("shared", type="module")

        G2 = nx.DiGraph()
        G2.add_node("shared", type="module")

        merged = _merge_graphs(G1, G2)

        assert merged.number_of_nodes() == 1


class TestAnalyzeDeps:
    """Integration tests for analyze_deps."""

    def test_analyzes_python_files(self, tmp_path: Path) -> None:
        """Should analyze Python files and return graph."""
        (tmp_path / "app.py").write_text("""
def main():
    helper()

def helper():
    pass
""")

        G = analyze_deps(tmp_path, languages=["python"])

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() >= 2  # main, helper

    def test_analyzes_typescript_files(self, tmp_path: Path) -> None:
        """Should analyze TypeScript files and return graph."""
        (tmp_path / "app.ts").write_text("""
export function main() {
    helper();
}

function helper() {}
""")

        G = analyze_deps(tmp_path, languages=["typescript"])

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() >= 1

    def test_auto_detects_languages(self, tmp_path: Path) -> None:
        """Should auto-detect languages when not specified."""
        (tmp_path / "app.py").write_text("def main(): pass")

        G = analyze_deps(tmp_path)  # No languages specified

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() >= 1

    def test_returns_empty_graph_for_no_source_files(self, tmp_path: Path) -> None:
        """Should return empty graph when no source files found."""
        (tmp_path / "readme.md").write_text("# Hello")

        G = analyze_deps(tmp_path)

        assert G.number_of_nodes() == 0

    def test_raises_for_invalid_path(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-existent path."""
        with pytest.raises(ValueError, match="Not a directory"):
            analyze_deps(tmp_path / "nonexistent")


class TestGraphMetadata:
    """Tests for graph_metadata extraction."""

    def test_counts_nodes_and_edges(self) -> None:
        """Should count nodes and edges."""
        G = nx.DiGraph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([("a", "b"), ("b", "c")])

        meta = graph_metadata(G)

        assert meta["node_count"] == 3
        assert meta["edge_count"] == 2

    def test_groups_by_type(self) -> None:
        """Should count nodes by type."""
        G = nx.DiGraph()
        G.add_node("a", type="function")
        G.add_node("b", type="function")
        G.add_node("c", type="class")
        G.add_edge("a", "b", type="calls")
        G.add_edge("a", "c", type="imports")

        meta = graph_metadata(G)

        assert meta["node_types"]["function"] == 2
        assert meta["node_types"]["class"] == 1
        assert meta["edge_types"]["calls"] == 1
        assert meta["edge_types"]["imports"] == 1
