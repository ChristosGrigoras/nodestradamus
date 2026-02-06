"""Tests for string topology analysis with NetworkX."""

from pathlib import Path

from nodestradamus.analyzers.string_topology import (
    analyze_string_topology,
    build_bipartite_graph,
    find_string_usages,
    identify_significant_strings,
)
from nodestradamus.models.graph import (
    StringContext,
    StringRefGraph,
    StringRefMetadata,
    StringRefNode,
)


class TestBuildBipartiteGraph:
    """Tests for building the bipartite graph."""

    def test_creates_nodes_for_files_and_strings(self) -> None:
        """Test that nodes are created for both files and strings."""
        refs = StringRefGraph(
            strings=[
                StringRefNode(
                    value="config.yaml",
                    file="app.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="config.yaml",
                    file="utils.py",
                    contexts=[StringContext(line=5)],
                ),
            ],
            file_count=2,
            metadata=StringRefMetadata(),
        )

        G = build_bipartite_graph(refs)

        # Should have 2 file nodes and 1 string node (same string in multiple files)
        file_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "file"]
        string_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "string"]

        assert len(file_nodes) == 2
        assert len(string_nodes) == 1

    def test_creates_edges_between_files_and_strings(self) -> None:
        """Test that edges connect files to their strings."""
        refs = StringRefGraph(
            strings=[
                StringRefNode(
                    value="shared_channel",
                    file="producer.py",
                    contexts=[StringContext(line=10, call_site="publish")],
                ),
                StringRefNode(
                    value="shared_channel",
                    file="consumer.py",
                    contexts=[StringContext(line=20, call_site="subscribe")],
                ),
            ],
            file_count=2,
            metadata=StringRefMetadata(),
        )

        G = build_bipartite_graph(refs)

        # String should be connected to both files
        string_node = "string:shared_channel"
        neighbors = list(G.neighbors(string_node))

        assert len(neighbors) == 2
        assert "file:producer.py" in neighbors
        assert "file:consumer.py" in neighbors

    def test_aggregates_contexts_on_edges(self) -> None:
        """Test that contexts are stored on edges."""
        refs = StringRefGraph(
            strings=[
                StringRefNode(
                    value="api_endpoint",
                    file="client.py",
                    contexts=[
                        StringContext(line=5, call_site="fetch"),
                        StringContext(line=10, call_site="post"),
                    ],
                ),
            ],
            file_count=1,
            metadata=StringRefMetadata(),
        )

        G = build_bipartite_graph(refs)

        # Check edge has contexts
        edge_data = G["file:client.py"]["string:api_endpoint"]
        assert "contexts" in edge_data
        assert len(edge_data["contexts"]) == 2


class TestIdentifySignificantStrings:
    """Tests for identifying significant strings based on topology."""

    def test_filters_by_min_files(self) -> None:
        """Test that strings below min_files threshold are excluded."""
        refs = StringRefGraph(
            strings=[
                # Appears in 2 files - should be significant
                StringRefNode(
                    value="shared_config",
                    file="app.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="shared_config",
                    file="lib.py",
                    contexts=[StringContext(line=1)],
                ),
                # Appears in 1 file - should be filtered
                StringRefNode(
                    value="local_only",
                    file="app.py",
                    contexts=[StringContext(line=5)],
                ),
            ],
            file_count=2,
            metadata=StringRefMetadata(),
        )

        G = build_bipartite_graph(refs)
        significant = identify_significant_strings(G, min_files=2)

        values = [s["value"] for s in significant]
        assert "shared_config" in values
        assert "local_only" not in values

    def test_includes_file_list(self) -> None:
        """Test that significant strings include list of referencing files."""
        refs = StringRefGraph(
            strings=[
                StringRefNode(
                    value="channel_name",
                    file="producer.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="channel_name",
                    file="consumer.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="channel_name",
                    file="monitor.py",
                    contexts=[StringContext(line=1)],
                ),
            ],
            file_count=3,
            metadata=StringRefMetadata(),
        )

        G = build_bipartite_graph(refs)
        significant = identify_significant_strings(G, min_files=2)

        assert len(significant) == 1
        assert "channel_name" in significant[0]["value"]
        assert len(significant[0]["referenced_by"]) == 3
        assert "producer.py" in significant[0]["referenced_by"]
        assert "consumer.py" in significant[0]["referenced_by"]
        assert "monitor.py" in significant[0]["referenced_by"]

    def test_sorts_by_reference_count(self) -> None:
        """Test that results are sorted by reference count (descending)."""
        refs = StringRefGraph(
            strings=[
                # 2 files
                StringRefNode(
                    value="less_used",
                    file="a.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="less_used",
                    file="b.py",
                    contexts=[StringContext(line=1)],
                ),
                # 3 files
                StringRefNode(
                    value="more_used",
                    file="a.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="more_used",
                    file="b.py",
                    contexts=[StringContext(line=1)],
                ),
                StringRefNode(
                    value="more_used",
                    file="c.py",
                    contexts=[StringContext(line=1)],
                ),
            ],
            file_count=3,
            metadata=StringRefMetadata(),
        )

        G = build_bipartite_graph(refs)
        significant = identify_significant_strings(G, min_files=2)

        # more_used should come first (3 refs > 2 refs)
        assert significant[0]["value"] == "more_used"
        assert significant[1]["value"] == "less_used"


class TestAnalyzeStringTopology:
    """Integration tests for full topology analysis."""

    def test_analyzes_python_files(self, tmp_path: Path) -> None:
        """Test end-to-end analysis of Python files."""
        # Create files that share a string
        (tmp_path / "producer.py").write_text('''
redis.publish("events/notifications", data)
''')
        (tmp_path / "consumer.py").write_text('''
redis.subscribe("events/notifications")
''')
        (tmp_path / "unrelated.py").write_text('''
print("hello world")
''')

        result = analyze_string_topology(
            tmp_path,
            min_files=2,
            include_python=True,
            include_typescript=False,
        )

        # The shared channel should be identified as significant
        values = [s.value for s in result.significant_strings]
        assert "events/notifications" in values

        # Check metadata
        assert result.metadata["significant_strings"] >= 1

    def test_analyzes_typescript_files(self, tmp_path: Path) -> None:
        """Test end-to-end analysis of TypeScript files."""
        # Create files that share a string
        (tmp_path / "api.ts").write_text('''
fetch("https://api.example.com/users");
''')
        (tmp_path / "client.ts").write_text('''
axios.get("https://api.example.com/users");
''')

        result = analyze_string_topology(
            tmp_path,
            min_files=2,
            include_python=False,
            include_typescript=True,
        )

        values = [s.value for s in result.significant_strings]
        assert "https://api.example.com/users" in values

    def test_includes_single_use_when_requested(self, tmp_path: Path) -> None:
        """Test that single-use strings can be included."""
        (tmp_path / "app.py").write_text('''
config = "unique_config_path"
shared = "shared_value"
''')
        (tmp_path / "lib.py").write_text('''
also_shared = "shared_value"
''')

        # Without single use
        result_filtered = analyze_string_topology(
            tmp_path,
            min_files=2,
            include_single_use=False,
            include_python=True,
            include_typescript=False,
        )

        # With single use
        result_all = analyze_string_topology(
            tmp_path,
            min_files=2,
            include_single_use=True,
            include_python=True,
            include_typescript=False,
        )

        # Filtered should have fewer strings
        assert len(result_all.significant_strings) >= len(result_filtered.significant_strings)

    def test_calculates_importance_scores(self, tmp_path: Path) -> None:
        """Test that importance scores are calculated."""
        (tmp_path / "a.py").write_text('x = "shared"')
        (tmp_path / "b.py").write_text('y = "shared"')

        result = analyze_string_topology(
            tmp_path,
            min_files=2,
            include_python=True,
            include_typescript=False,
        )

        for s in result.significant_strings:
            assert 0.0 <= s.importance_score <= 1.0


class TestFindStringUsages:
    """Tests for finding usages of a specific string."""

    def test_finds_all_usages(self, tmp_path: Path) -> None:
        """Test that all usages of a string are found."""
        (tmp_path / "producer.py").write_text('''
redis.publish("channel/events", data)
''')
        (tmp_path / "consumer.py").write_text('''
redis.subscribe("channel/events")
''')
        (tmp_path / "other.py").write_text('''
print("unrelated string")
''')

        result = find_string_usages(
            tmp_path,
            "channel/events",
            include_python=True,
            include_typescript=False,
        )

        assert result["target_string"] == "channel/events"
        assert len(result["files"]) == 2
        assert "producer.py" in result["files"]
        assert "consumer.py" in result["files"]
        assert result["total_usages"] == 2

    def test_returns_empty_for_missing_string(self, tmp_path: Path) -> None:
        """Test handling of string not found in codebase."""
        (tmp_path / "app.py").write_text('x = "existing"')

        result = find_string_usages(
            tmp_path,
            "nonexistent_string",
            include_python=True,
            include_typescript=False,
        )

        assert result["target_string"] == "nonexistent_string"
        assert len(result["files"]) == 0
        assert result["total_usages"] == 0

    def test_includes_context_details(self, tmp_path: Path) -> None:
        """Test that usage contexts are included."""
        (tmp_path / "service.py").write_text('''
class MyService:
    def connect(self):
        db.open("connection_string")
''')

        result = find_string_usages(
            tmp_path,
            "connection_string",
            include_python=True,
            include_typescript=False,
        )

        assert len(result["usages_by_file"]) == 1
        usages = result["usages_by_file"]["service.py"]
        assert len(usages) == 1
        assert usages[0]["call_site"] == "db.open"
        assert usages[0]["enclosing_function"] == "connect"
        assert usages[0]["enclosing_class"] == "MyService"
