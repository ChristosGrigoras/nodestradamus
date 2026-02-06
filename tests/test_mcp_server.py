"""Tests for MCP server initialization and tools."""

import networkx as nx
import pytest

from nodestradamus.mcp.server import SERVER_NAME, SERVER_VERSION, create_server
from nodestradamus.mcp.tools.definitions import (
    ANALYZE_COOCCURRENCE_TOOL,
    ANALYZE_DEPS_TOOL,
    ANALYZE_GRAPH_TOOL,
    ANALYZE_STRINGS_TOOL,
    CODEBASE_HEALTH_TOOL,
    GET_IMPACT_TOOL,
    SEMANTIC_ANALYSIS_TOOL,
)


class TestMCPServer:
    """Tests for MCP server creation and configuration."""

    def test_create_server_returns_server_instance(self) -> None:
        """Server creation returns a valid Server instance."""
        server = create_server()
        assert server is not None
        assert server.name == SERVER_NAME

    def test_server_name_is_nodestradamus(self) -> None:
        """Server name is 'nodestradamus'."""
        assert SERVER_NAME == "nodestradamus"

    def test_server_version_matches_package(self) -> None:
        """Server version matches package version."""
        from nodestradamus import __version__

        assert SERVER_VERSION == __version__


class TestToolDefinitions:
    """Tests for MCP tool definitions."""

    def test_analyze_deps_tool_has_required_fields(self) -> None:
        """analyze_deps tool has name, description, and schema."""
        assert ANALYZE_DEPS_TOOL.name == "analyze_deps"
        assert ANALYZE_DEPS_TOOL.description is not None
        assert "repo_path" in ANALYZE_DEPS_TOOL.inputSchema["properties"]
        assert "repo_path" in ANALYZE_DEPS_TOOL.inputSchema["required"]
        assert "languages" in ANALYZE_DEPS_TOOL.inputSchema["properties"]

    def test_analyze_cooccurrence_tool_has_required_fields(self) -> None:
        """analyze_cooccurrence tool has name, description, and schema."""
        assert ANALYZE_COOCCURRENCE_TOOL.name == "analyze_cooccurrence"
        assert ANALYZE_COOCCURRENCE_TOOL.description is not None
        assert "repo_path" in ANALYZE_COOCCURRENCE_TOOL.inputSchema["properties"]
        assert "commits" in ANALYZE_COOCCURRENCE_TOOL.inputSchema["properties"]

    def test_get_impact_tool_has_required_fields(self) -> None:
        """get_impact tool has name, description, and schema."""
        assert GET_IMPACT_TOOL.name == "get_impact"
        assert GET_IMPACT_TOOL.description is not None
        assert "repo_path" in GET_IMPACT_TOOL.inputSchema["properties"]
        assert "file_path" in GET_IMPACT_TOOL.inputSchema["properties"]
        assert "symbol" in GET_IMPACT_TOOL.inputSchema["properties"]
        assert "depth" in GET_IMPACT_TOOL.inputSchema["properties"]

    def test_analyze_graph_tool_has_algorithm_param(self) -> None:
        """analyze_graph tool has algorithm parameter with all options."""
        assert ANALYZE_GRAPH_TOOL.name == "analyze_graph"
        assert "repo_path" in ANALYZE_GRAPH_TOOL.inputSchema["properties"]
        assert "algorithm" in ANALYZE_GRAPH_TOOL.inputSchema["properties"]
        algo_schema = ANALYZE_GRAPH_TOOL.inputSchema["properties"]["algorithm"]
        assert set(algo_schema["enum"]) == {
            "pagerank",
            "betweenness",
            "communities",
            "cycles",
            "path",
            "hierarchy",
            "layers",
        }

    def test_analyze_strings_tool_has_mode_param(self) -> None:
        """analyze_strings tool has mode parameter."""
        assert ANALYZE_STRINGS_TOOL.name == "analyze_strings"
        assert "mode" in ANALYZE_STRINGS_TOOL.inputSchema["properties"]
        mode_schema = ANALYZE_STRINGS_TOOL.inputSchema["properties"]["mode"]
        assert set(mode_schema["enum"]) == {"refs", "usages", "filter"}

    def test_semantic_analysis_tool_has_mode_param(self) -> None:
        """semantic_analysis tool has mode parameter."""
        assert SEMANTIC_ANALYSIS_TOOL.name == "semantic_analysis"
        assert "mode" in SEMANTIC_ANALYSIS_TOOL.inputSchema["properties"]
        mode_schema = SEMANTIC_ANALYSIS_TOOL.inputSchema["properties"]["mode"]
        assert set(mode_schema["enum"]) == {"search", "similar", "duplicates", "embeddings"}

    def test_codebase_health_tool_has_checks_param(self) -> None:
        """codebase_health tool has checks parameter."""
        assert CODEBASE_HEALTH_TOOL.name == "codebase_health"
        assert "repo_path" in CODEBASE_HEALTH_TOOL.inputSchema["properties"]
        assert "checks" in CODEBASE_HEALTH_TOOL.inputSchema["properties"]


class TestAnalyzers:
    """Tests for analyzer functions."""

    def test_analyze_deps_returns_networkx_graph(self, sample_python_dir: str) -> None:
        """analyze_deps returns a NetworkX DiGraph."""
        from nodestradamus.analyzers import analyze_deps

        G = analyze_deps(sample_python_dir)

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0

    def test_analyze_deps_finds_functions(self, sample_python_dir: str) -> None:
        """analyze_deps finds function definitions."""
        from nodestradamus.analyzers import analyze_deps

        G = analyze_deps(sample_python_dir)

        # Check that we found some functions
        function_nodes = [
            node for node, attrs in G.nodes(data=True)
            if attrs.get("type") == "function"
        ]
        assert len(function_nodes) > 0

    def test_get_impact_returns_report(self, sample_python_dir: str) -> None:
        """get_impact returns an ImpactReport."""
        from nodestradamus.analyzers.impact import get_impact

        report = get_impact(sample_python_dir, "main.py")

        assert report is not None
        assert report.target is not None
        assert report.risk_assessment is not None

    def test_get_impact_matches_correct_file(self, sample_python_dir: str) -> None:
        """get_impact matches the requested file, not unrelated files."""
        from nodestradamus.analyzers.impact import get_impact

        report = get_impact(sample_python_dir, "services/user.py")

        # Target should contain the requested file path, not a random match
        assert "user" in report.target.lower()
        # Should NOT match main.py or utils.py due to substring matching
        assert "main.py" not in report.target

    def test_get_impact_with_symbol_matches_exact_symbol(
        self, sample_python_dir: str
    ) -> None:
        """get_impact with symbol returns exact match, not first substring."""
        from nodestradamus.analyzers.impact import get_impact

        report = get_impact(sample_python_dir, "main.py", symbol="main")

        # Target should include the symbol name
        assert "main" in report.target


class TestGraphFiltering:
    """Tests for graph algorithm filtering (exclude tests, external imports)."""

    @pytest.mark.skip(reason="_is_test_file not implemented yet")
    def test_is_test_file_patterns(self) -> None:
        """Test file pattern detection."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _is_test_file

        # Should match test files
        assert _is_test_file("test_main.py") is True
        assert _is_test_file("main_test.py") is True
        assert _is_test_file("tests/test_utils.py") is True
        assert _is_test_file("test/test_utils.py") is True
        assert _is_test_file("src/__tests__/Button.test.tsx") is True
        assert _is_test_file("components/Button.spec.ts") is True

        # Should NOT match source files
        assert _is_test_file("main.py") is False
        assert _is_test_file("utils.py") is False
        assert _is_test_file("src/models.py") is False
        assert _is_test_file("app.tsx") is False
        assert _is_test_file("") is False

    def test_is_external_import(self) -> None:
        """External import detection."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _is_external_import

        # Should be external (stdlib with no file path)
        assert _is_external_import("typing", "") is True
        assert _is_external_import("collections.abc", "") is True
        assert _is_external_import("os.path", "") is True
        assert _is_external_import("json", "") is True

        # Should NOT be external (has a file path)
        assert _is_external_import("mypackage.utils", "src/mypackage/utils.py") is False
        assert _is_external_import("main", "main.py") is False

    def test_filter_graph_excludes_tests(self) -> None:
        """Filter graph excludes test files in source_only scope."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _filter_graph_for_ranking

        G = nx.DiGraph()
        G.add_node("main.py::main", file="main.py", type="function")
        G.add_node("utils.py::helper", file="utils.py", type="function")
        G.add_node("test_main.py::test_main", file="test_main.py", type="function")
        G.add_node("tests/test_utils.py::test_helper", file="tests/test_utils.py", type="function")
        G.add_edge("test_main.py::test_main", "main.py::main")

        filtered = _filter_graph_for_ranking(G, scope="source_only", exclude_external=False)

        assert filtered.number_of_nodes() == 2
        assert "main.py::main" in filtered.nodes()
        assert "utils.py::helper" in filtered.nodes()
        assert "test_main.py::test_main" not in filtered.nodes()
        assert "tests/test_utils.py::test_helper" not in filtered.nodes()

    def test_filter_graph_excludes_external(self) -> None:
        """Filter graph excludes external imports."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _filter_graph_for_ranking

        G = nx.DiGraph()
        G.add_node("main.py::main", file="main.py", type="function")
        G.add_node("typing", file="", type="module")  # external
        G.add_node("json", file="", type="module")  # external
        G.add_node("mypackage.utils", file="src/mypackage/utils.py", type="module")
        G.add_edge("main.py::main", "typing")
        G.add_edge("main.py::main", "mypackage.utils")

        filtered = _filter_graph_for_ranking(G, scope="all", exclude_external=True)

        assert "main.py::main" in filtered.nodes()
        assert "mypackage.utils" in filtered.nodes()
        assert "typing" not in filtered.nodes()
        assert "json" not in filtered.nodes()

    def test_filter_graph_tests_only_scope(self) -> None:
        """Filter graph with tests_only scope includes only tests."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _filter_graph_for_ranking

        G = nx.DiGraph()
        G.add_node("main.py::main", file="main.py", type="function")
        G.add_node("test_main.py::test_main", file="test_main.py", type="function")
        G.add_edge("test_main.py::test_main", "main.py::main")

        filtered = _filter_graph_for_ranking(G, scope="tests_only", exclude_external=False)

        assert filtered.number_of_nodes() == 1
        assert "test_main.py::test_main" in filtered.nodes()
        assert "main.py::main" not in filtered.nodes()

    def test_filter_graph_all_scope_includes_everything(self) -> None:
        """Filter graph with all scope includes everything except externals if specified."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _filter_graph_for_ranking

        G = nx.DiGraph()
        G.add_node("main.py::main", file="main.py", type="function")
        G.add_node("test_main.py::test_main", file="test_main.py", type="function")
        G.add_node("typing", file="", type="module")

        filtered = _filter_graph_for_ranking(G, scope="all", exclude_external=False)
        assert filtered.number_of_nodes() == 3

        filtered = _filter_graph_for_ranking(G, scope="all", exclude_external=True)
        assert filtered.number_of_nodes() == 2


class TestHierarchyClassification:
    """Tests for hierarchy algorithm's classification of unknown nodes as [stdlib]/[external]."""

    def test_unknown_nodes_classified_as_stdlib_or_external(self) -> None:
        """Nodes without file paths should be classified as [stdlib] or [external]."""
        import json

        from nodestradamus.mcp.tools.handlers.graph_algorithms import _run_hierarchy

        G = nx.DiGraph()
        G.add_node("py:src/main.py::main", file="src/main.py", type="function")
        G.add_node("typing", file="", type="module")  # stdlib
        G.add_node("requests", file="", type="module")  # external
        G.add_edge("py:src/main.py::main", "typing", type="imports")
        G.add_edge("py:src/main.py::main", "requests", type="imports")

        result = json.loads(_run_hierarchy(G, {"level": "module"}))

        node_ids = {n["id"] for n in result["hierarchy_nodes"]}
        # Should have classified nodes, not "unknown"
        assert "unknown" not in node_ids
        # Should have [stdlib] (for typing) and/or [external] (for requests)
        assert "[stdlib]" in node_ids or "[external]" in node_ids
        # Source file should still be present
        assert "src/main.py" in node_ids

    def test_classify_unknown_node_separates_stdlib_and_external(self) -> None:
        """_classify_unknown_node should split children into [stdlib] and [external]."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _classify_unknown_node

        node = {
            "id": "unknown",
            "level": "module",
            "child_count": 4,
            "children": ["typing", "json", "requests", "fastapi"],
        }

        result = _classify_unknown_node(node, "module")

        # Should have two nodes: [stdlib] and [external]
        assert len(result) == 2
        ids = {n["id"] for n in result}
        assert "[stdlib]" in ids
        assert "[external]" in ids

        # Check correct classification
        stdlib_node = next(n for n in result if n["id"] == "[stdlib]")
        external_node = next(n for n in result if n["id"] == "[external]")
        assert "typing" in stdlib_node["children"]
        assert "json" in stdlib_node["children"]
        assert "requests" in external_node["children"]
        assert "fastapi" in external_node["children"]

    def test_exclude_external_removes_stdlib_and_external_nodes(self) -> None:
        """exclude_external=true should filter out [stdlib] and [external] nodes."""
        import json

        from nodestradamus.mcp.tools.handlers.graph_algorithms import _run_hierarchy

        G = nx.DiGraph()
        G.add_node("py:src/main.py::main", file="src/main.py", type="function")
        G.add_node("typing", file="", type="module")  # stdlib
        G.add_node("requests", file="", type="module")  # external
        G.add_edge("py:src/main.py::main", "typing", type="imports")
        G.add_edge("py:src/main.py::main", "requests", type="imports")

        result = json.loads(_run_hierarchy(G, {"level": "module", "exclude_external": True}))

        node_ids = {n["id"] for n in result["hierarchy_nodes"]}
        # External nodes should be filtered out
        assert "[stdlib]" not in node_ids
        assert "[external]" not in node_ids
        assert "unknown" not in node_ids
        # Source file should remain
        assert "src/main.py" in node_ids
        # Metadata should reflect filtering
        assert result["filter_applied"]["exclude_external"] is True

    def test_hierarchy_response_includes_external_counts(self) -> None:
        """Hierarchy response should include stdlib_count and external_count metadata."""
        import json

        from nodestradamus.mcp.tools.handlers.graph_algorithms import _run_hierarchy

        G = nx.DiGraph()
        G.add_node("py:src/main.py::main", file="src/main.py", type="function")
        G.add_node("typing", file="", type="module")  # stdlib
        G.add_node("json", file="", type="module")  # stdlib
        G.add_node("requests", file="", type="module")  # external
        G.add_edge("py:src/main.py::main", "typing", type="imports")
        G.add_edge("py:src/main.py::main", "json", type="imports")
        G.add_edge("py:src/main.py::main", "requests", type="imports")

        result = json.loads(_run_hierarchy(G, {"level": "module"}))

        # Should have counts in response
        assert "stdlib_count" in result
        assert "external_count" in result
        assert result["stdlib_count"] == 2  # typing, json
        assert result["external_count"] == 1  # requests


class TestCommunityClassification:
    """Tests for community stdlib/external classification (M2)."""

    def test_classify_community_stdlib_majority(self) -> None:
        """Stdlib-dominated communities are labeled 'stdlib'."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _classify_community

        # Arrange: Graph with nodes, most without file paths and in STDLIB_MODULES
        G = nx.DiGraph()
        G.add_node("os")  # stdlib, no file
        G.add_node("json")  # stdlib, no file
        G.add_node("asyncio")  # stdlib, no file
        G.add_node("py:src/main.py", file="src/main.py")  # source
        members = ["os", "json", "asyncio", "py:src/main.py"]

        # Act
        result = _classify_community(members, G)

        # Assert
        assert result == "stdlib"

    def test_classify_community_external_majority(self) -> None:
        """Third-party-dominated communities are labeled 'external'."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _classify_community

        # Arrange: Graph with nodes, most are third-party (not in STDLIB_MODULES)
        G = nx.DiGraph()
        G.add_node("numpy")  # third-party, no file
        G.add_node("pandas")  # third-party, no file
        G.add_node("httpx")  # third-party, no file
        G.add_node("py:src/main.py", file="src/main.py")  # source
        members = ["numpy", "pandas", "httpx", "py:src/main.py"]

        # Act
        result = _classify_community(members, G)

        # Assert
        assert result == "external"

    def test_classify_community_source_majority(self) -> None:
        """Source-dominated communities are labeled 'source'."""
        from nodestradamus.mcp.tools.handlers.graph_algorithms import _classify_community

        # Arrange: Graph with mostly source nodes
        G = nx.DiGraph()
        G.add_node("py:src/main.py", file="src/main.py")
        G.add_node("py:src/utils.py", file="src/utils.py")
        G.add_node("py:src/models.py", file="src/models.py")
        G.add_node("os")  # stdlib
        members = ["py:src/main.py", "py:src/utils.py", "py:src/models.py", "os"]

        # Act
        result = _classify_community(members, G)

        # Assert
        assert result == "source"

    @pytest.mark.asyncio
    async def test_communities_summary_includes_stdlib_count(self) -> None:
        """Communities summary includes stdlib_modules count."""
        import json

        from nodestradamus.mcp.tools.handlers.graph_algorithms import _run_communities

        # Arrange: Graph with source, tests, stdlib, and external nodes
        G = nx.DiGraph()
        G.add_node("py:src/main.py", file="src/main.py", type="function")
        G.add_node("py:tests/test_main.py", file="tests/test_main.py", type="function")
        G.add_node("os", file="", type="module")  # stdlib
        G.add_node("numpy", file="", type="module")  # external
        G.add_edge("py:src/main.py", "os", type="imports")
        G.add_edge("py:tests/test_main.py", "numpy", type="imports")

        # Act
        result = json.loads(await _run_communities(G, {"summary_only": True}))

        # Assert: summary contains stdlib_modules key
        assert "summary" in result
        assert "stdlib_modules" in result["summary"]
        assert isinstance(result["summary"]["stdlib_modules"], int)


class TestImpactOutputImprovements:
    """Tests for get_impact output improvements: compact mode, same-file filtering, depth summary."""

    def test_extract_file_from_node_id(self) -> None:
        """Node ID file extraction works correctly."""
        from nodestradamus.analyzers.impact import _extract_file_from_node_id

        assert _extract_file_from_node_id("py:src/main.py::main") == "src/main.py"
        assert _extract_file_from_node_id("py:utils.py") == "utils.py"
        assert _extract_file_from_node_id("ts:src/app.tsx::App") == "src/app.tsx"
        assert _extract_file_from_node_id("main.py") == "main.py"

    def test_extract_symbol_from_node_id(self) -> None:
        """Node ID symbol extraction works correctly."""
        from nodestradamus.analyzers.impact import _extract_symbol_from_node_id

        assert _extract_symbol_from_node_id("py:src/main.py::main") == "main"
        assert _extract_symbol_from_node_id("py:utils.py") == ""
        assert _extract_symbol_from_node_id("ts:src/app.tsx::App") == "App"

    @pytest.mark.skip(reason="_is_test_file not implemented yet")
    def test_is_test_file_in_impact(self) -> None:
        """Test file detection in impact module matches graph_algorithms."""
        from nodestradamus.analyzers.impact import _is_test_file

        # Should match test files
        assert _is_test_file("test_main.py") is True
        assert _is_test_file("main_test.py") is True
        assert _is_test_file("tests/test_utils.py") is True
        assert _is_test_file("src/__tests__/Button.test.tsx") is True

        # Should NOT match source files
        assert _is_test_file("main.py") is False
        assert _is_test_file("src/utils.py") is False

    def test_group_by_file(self) -> None:
        """Group by file organizes targets correctly."""
        from nodestradamus.analyzers.impact import _group_by_file
        from nodestradamus.models.graph import ImpactTarget

        targets = [
            ImpactTarget(id="py:main.py::func1", depth=1),
            ImpactTarget(id="py:main.py::func2", depth=1),
            ImpactTarget(id="py:utils.py::helper", depth=2),
            ImpactTarget(id="py:utils.py::other", depth=2),
            ImpactTarget(id="py:config.py", depth=1),  # File-level node
        ]

        grouped = _group_by_file(targets)

        assert "main.py" in grouped
        assert set(grouped["main.py"]) == {"func1", "func2"}
        assert "utils.py" in grouped
        assert set(grouped["utils.py"]) == {"helper", "other"}
        assert "config.py" in grouped
        assert grouped["config.py"] == []

    def test_build_depth_summary(self) -> None:
        """Depth summary correctly counts at each level."""
        from nodestradamus.analyzers.impact import _build_depth_summary
        from nodestradamus.models.graph import ImpactTarget

        upstream = [
            ImpactTarget(id="a", depth=1),
            ImpactTarget(id="b", depth=1),
            ImpactTarget(id="c", depth=2),
        ]
        downstream = [
            ImpactTarget(id="x", depth=1),
            ImpactTarget(id="y", depth=2),
            ImpactTarget(id="z", depth=2),
        ]

        summary = _build_depth_summary(upstream, downstream)

        assert summary["upstream"]["depth_1"] == 2
        assert summary["upstream"]["depth_2"] == 1
        assert summary["downstream"]["depth_1"] == 1
        assert summary["downstream"]["depth_2"] == 2

    def test_get_impact_compact_mode(self, sample_python_dir: str) -> None:
        """get_impact with compact=True returns grouped results."""
        from nodestradamus.analyzers.impact import get_impact

        report = get_impact(sample_python_dir, "main.py", compact=True)

        # Should have the grouped fields populated
        assert report.depth_summary is not None
        assert "upstream" in report.depth_summary
        assert "downstream" in report.depth_summary
        # upstream_by_file and downstream_by_file should be dicts
        assert isinstance(report.upstream_by_file, dict)
        assert isinstance(report.downstream_by_file, dict)

    def test_get_impact_exclude_same_file(self, sample_python_dir: str) -> None:
        """get_impact with exclude_same_file=True filters same-file symbols."""
        from nodestradamus.analyzers.impact import _extract_file_from_node_id, get_impact

        report = get_impact(sample_python_dir, "main.py", exclude_same_file=True)
        target_file = _extract_file_from_node_id(report.target)

        # No upstream/downstream should have the same file as target
        for t in report.upstream:
            assert _extract_file_from_node_id(t.id) != target_file

        for t in report.downstream:
            assert _extract_file_from_node_id(t.id) != target_file

    def test_get_impact_include_same_file(self, sample_python_dir: str) -> None:
        """get_impact with exclude_same_file=False includes same-file symbols."""
        from nodestradamus.analyzers.impact import get_impact

        # Get both versions
        report_excluded = get_impact(
            sample_python_dir, "main.py", exclude_same_file=True
        )
        report_included = get_impact(
            sample_python_dir, "main.py", exclude_same_file=False
        )

        # The included version should have >= symbols (may be equal if no same-file deps)
        total_excluded = len(report_excluded.upstream) + len(report_excluded.downstream)
        total_included = len(report_included.upstream) + len(report_included.downstream)
        assert total_included >= total_excluded
