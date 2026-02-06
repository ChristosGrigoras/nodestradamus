"""Tests for merge_graphs.py."""

import json
from pathlib import Path

from scripts.merge_graphs import (
    detect_graph_type,
    find_cross_language_connections,
    load_graph,
    merge_graphs,
    normalize_node,
)


class TestLoadGraph:
    """Tests for load_graph function."""

    def test_load_valid_json(self, temp_dir: Path) -> None:
        """Test loading a valid JSON graph file."""
        graph_file = temp_dir / "graph.json"
        graph_file.write_text(json.dumps({
            "nodes": ["a", "b"],
            "edges": [{"from": "a", "to": "b"}],
        }))

        result = load_graph(graph_file)

        assert result is not None
        assert result["nodes"] == ["a", "b"]

    def test_load_nonexistent_file(self, temp_dir: Path) -> None:
        """Test loading nonexistent file returns None."""
        result = load_graph(temp_dir / "nonexistent.json")
        assert result is None

    def test_load_invalid_json(self, temp_dir: Path) -> None:
        """Test loading invalid JSON returns None."""
        bad_file = temp_dir / "bad.json"
        bad_file.write_text("not valid json {")

        result = load_graph(bad_file)
        assert result is None


class TestNormalizeNode:
    """Tests for normalize_node function."""

    def test_python_node_prefix(self) -> None:
        """Test that Python nodes get py: prefix."""
        result = normalize_node("main.py::func", "python")
        assert result == "py:main.py::func"

    def test_typescript_node_prefix(self) -> None:
        """Test that TypeScript nodes get ts: prefix."""
        result = normalize_node("main.ts::func", "typescript")
        assert result == "ts:main.ts::func"

    def test_no_prefix_without_namespace(self) -> None:
        """Test that nodes without :: don't get prefix."""
        result = normalize_node("main.py", "python")
        assert result == "main.py"

    def test_no_prefix_for_other_types(self) -> None:
        """Test that other source types don't get prefix."""
        result = normalize_node("file::func", "cooccurrence")
        assert result == "file::func"


class TestMergeGraphs:
    """Tests for merge_graphs function."""

    def test_merge_empty_list(self) -> None:
        """Test merging empty list of graphs."""
        result = merge_graphs([])

        assert result["nodes"] == []
        assert result["edges"] == []
        assert "metadata" in result

    def test_merge_single_graph(self, sample_graph: dict) -> None:
        """Test merging a single graph."""
        result = merge_graphs([(sample_graph, "python")])

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["metadata"]["sources"] == ["python"]

    def test_merge_multiple_graphs(self) -> None:
        """Test merging multiple graphs."""
        graph1 = {
            "nodes": ["a::func1"],
            "edges": [{"from": "a::func1", "to": "b", "type": "calls"}],
        }
        graph2 = {
            "nodes": ["c::func2"],
            "edges": [{"from": "c::func2", "to": "d", "type": "imports"}],
        }

        result = merge_graphs([(graph1, "python"), (graph2, "typescript")])

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 2
        assert result["metadata"]["sources"] == ["python", "typescript"]

    def test_merge_deduplicates_edges(self) -> None:
        """Test that duplicate edges are removed."""
        graph1 = {
            "nodes": ["a"],
            "edges": [{"from": "a", "to": "b", "type": "calls"}],
        }
        graph2 = {
            "nodes": ["a"],
            "edges": [{"from": "a", "to": "b", "type": "calls"}],  # Same edge
        }

        result = merge_graphs([(graph1, "python"), (graph2, "python")])

        # Should only have one edge after deduplication
        assert len(result["edges"]) == 1

    def test_merge_collects_files(self) -> None:
        """Test that files are collected from all graphs."""
        graph1 = {"nodes": [], "edges": [], "files": ["a.py", "b.py"]}
        graph2 = {"nodes": [], "edges": [], "files": ["c.ts", "d.ts"]}

        result = merge_graphs([(graph1, "python"), (graph2, "typescript")])

        assert len(result["files"]) == 4
        assert "a.py" in result["files"]
        assert "c.ts" in result["files"]

    def test_merge_collects_errors(self) -> None:
        """Test that errors are collected from all graphs."""
        graph1 = {"nodes": [], "edges": [], "errors": [{"file": "a.py", "error": "syntax"}]}
        graph2 = {"nodes": [], "edges": [], "errors": [{"file": "b.ts", "error": "parse"}]}

        result = merge_graphs([(graph1, "python"), (graph2, "typescript")])

        assert len(result["errors"]) == 2

    def test_merge_handles_string_errors(self) -> None:
        """Test that string errors are converted to dicts."""
        graph = {"nodes": [], "edges": [], "errors": ["some error"]}

        result = merge_graphs([(graph, "python")])

        assert len(result["errors"]) == 1
        assert result["errors"][0]["message"] == "some error"


class TestFindCrossLanguageConnections:
    """Tests for find_cross_language_connections function."""

    def test_no_cooccurrence_graph(self) -> None:
        """Test when there's no cooccurrence graph."""
        graphs = [
            ({"nodes": [], "edges": []}, "python"),
            ({"nodes": [], "edges": []}, "typescript"),
        ]

        result = find_cross_language_connections(graphs)
        assert result == []

    def test_finds_cross_language_edges(self) -> None:
        """Test finding edges between Python and TypeScript files."""
        graphs = [
            ({"nodes": [], "edges": [], "files": ["main.py"]}, "python"),
            ({"nodes": [], "edges": [], "files": ["main.ts"]}, "typescript"),
            ({
                "nodes": [],
                "edges": [
                    {"from": "main.py", "to": "main.ts", "strength": 0.8, "count": 5}
                ]
            }, "cooccurrence"),
        ]

        result = find_cross_language_connections(graphs)

        assert len(result) == 1
        assert result[0]["type"] == "co-occurs-cross-language"
        assert result[0]["from"] == "main.py"
        assert result[0]["to"] == "main.ts"

    def test_ignores_same_language_edges(self) -> None:
        """Test that same-language edges are not included."""
        graphs = [
            ({"nodes": [], "edges": [], "files": ["a.py", "b.py"]}, "python"),
            ({
                "nodes": [],
                "edges": [
                    {"from": "a.py", "to": "b.py", "strength": 0.8}
                ]
            }, "cooccurrence"),
        ]

        result = find_cross_language_connections(graphs)
        assert len(result) == 0


class TestDetectGraphType:
    """Tests for detect_graph_type function."""

    def test_detect_from_filename(self) -> None:
        """Test detecting type from filename."""
        assert detect_graph_type({}, "python-deps.json") == "python"
        assert detect_graph_type({}, "ts-deps.json") == "typescript"
        assert detect_graph_type({}, "app-js-deps.json") == "typescript"
        assert detect_graph_type({}, "co-occurrence.json") == "cooccurrence"
        assert detect_graph_type({}, "cooccurrence.json") == "cooccurrence"

    def test_detect_from_generator_metadata(self) -> None:
        """Test detecting type from generator metadata."""
        py_graph = {"metadata": {"generator": "analyze_python_deps.py"}}
        ts_graph = {"metadata": {"generator": "analyze_ts_deps.py"}}
        cooccur_graph = {"metadata": {"generator": "analyze_git_cooccurrence.py"}}

        assert detect_graph_type(py_graph, "graph.json") == "python"
        assert detect_graph_type(ts_graph, "graph.json") == "typescript"
        assert detect_graph_type(cooccur_graph, "graph.json") == "cooccurrence"

    def test_detect_from_edge_types(self) -> None:
        """Test detecting type from edge types."""
        cooccur_graph = {"edges": [{"type": "co-occurs"}]}
        ts_graph = {"edges": [{"type": "imports"}]}

        assert detect_graph_type(cooccur_graph, "graph.json") == "cooccurrence"
        assert detect_graph_type(ts_graph, "graph.json") == "typescript"

    def test_unknown_type(self) -> None:
        """Test that unknown types return 'unknown'."""
        assert detect_graph_type({}, "random.json") == "unknown"


class TestIntegration:
    """Integration tests for the merge system."""

    def test_merge_sample_graphs(self, sample_graph: dict) -> None:
        """Test merging sample graph fixtures."""
        # Create a second sample graph
        graph2 = {
            "nodes": ["utils.ts::helper"],
            "node_details": [
                {"name": "utils.ts::helper", "type": "function", "line": 5}
            ],
            "edges": [],
            "errors": [],
            "metadata": {"generator": "analyze_ts_deps.py"},
        }

        result = merge_graphs([
            (sample_graph, "python"),
            (graph2, "typescript"),
        ])

        assert len(result["nodes"]) >= 2
        assert "metadata" in result
        assert result["metadata"]["stats"]["total_nodes"] >= 2

    def test_full_merge_workflow(self, temp_dir: Path) -> None:
        """Test full workflow of loading and merging graphs."""
        # Create sample graph files
        py_graph = temp_dir / "python-deps.json"
        py_graph.write_text(json.dumps({
            "nodes": ["main.py::main"],
            "edges": [{"from": "main.py::main", "to": "utils", "type": "calls"}],
            "metadata": {"generator": "analyze_python_deps.py"},
        }))

        ts_graph = temp_dir / "ts-deps.json"
        ts_graph.write_text(json.dumps({
            "nodes": ["main.ts::main"],
            "edges": [{"from": "main.ts::main", "to": "./utils", "type": "imports"}],
            "metadata": {"generator": "analyze_ts_deps.py"},
        }))

        # Load and merge
        graphs = []
        for filepath in temp_dir.glob("*.json"):
            graph = load_graph(filepath)
            if graph:
                graph_type = detect_graph_type(graph, filepath.name)
                graphs.append((graph, graph_type))

        result = merge_graphs(graphs)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 2
        assert "python" in result["metadata"]["sources"]
        assert "typescript" in result["metadata"]["sources"]
