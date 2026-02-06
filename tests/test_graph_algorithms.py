"""Tests for graph algorithms.

Tests run against both NetworkX and Rust backends when available.
"""

import networkx as nx
import pytest

from nodestradamus.analyzers import graph_algorithms
from nodestradamus.analyzers.graph_algorithms import (
    ancestors_at_depth,
    betweenness,
    community_metagraph,
    descendants_at_depth,
    detect_communities,
    find_cycles,
    hierarchical_view,
    pagerank,
    shortest_path,
    strongly_connected,
    top_n,
    validate_layers,
)


@pytest.fixture(params=["networkx", "rust"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Fixture to test both NetworkX and Rust backends.

    Skips Rust tests if the extension is not available.
    """
    if request.param == "networkx":
        monkeypatch.setattr(graph_algorithms, "_HAS_RUST", False)
    elif request.param == "rust":
        if not graph_algorithms._HAS_RUST:
            pytest.skip("Rust extension not available")
    return request.param


class TestPageRank:
    """Tests for PageRank importance ranking."""

    def test_high_score_for_heavily_depended_node(self, backend: str) -> None:
        """Nodes with many incoming edges should rank higher."""
        G = nx.DiGraph()
        # Many nodes depend on "core"
        G.add_edges_from([
            ("a", "core"),
            ("b", "core"),
            ("c", "core"),
            ("d", "core"),
            ("e", "a"),  # Only one depends on "a"
        ])

        scores = pagerank(G)

        assert scores["core"] > scores["a"]
        assert scores["core"] > scores["e"]

    def test_returns_empty_for_empty_graph(self, backend: str) -> None:
        """Empty graph returns empty dict."""
        G = nx.DiGraph()
        assert pagerank(G) == {}

    def test_scores_sum_to_one(self, backend: str) -> None:
        """PageRank scores should sum to approximately 1."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

        scores = pagerank(G)
        total = sum(scores.values())

        assert abs(total - 1.0) < 0.01


class TestBetweenness:
    """Tests for betweenness centrality."""

    def test_bridge_node_has_high_betweenness(self) -> None:
        """Node connecting two clusters should have high betweenness."""
        G = nx.DiGraph()
        # Two clusters connected by "bridge"
        G.add_edges_from([
            ("a1", "bridge"),
            ("a2", "bridge"),
            ("bridge", "b1"),
            ("bridge", "b2"),
        ])

        scores = betweenness(G)

        assert scores["bridge"] > scores["a1"]
        assert scores["bridge"] > scores["b1"]

    def test_returns_empty_for_empty_graph(self) -> None:
        """Empty graph returns empty dict."""
        G = nx.DiGraph()
        assert betweenness(G) == {}

    def test_normalized_scores_between_zero_and_one(self) -> None:
        """Normalized betweenness should be in [0, 1]."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        scores = betweenness(G, normalized=True)

        for score in scores.values():
            assert 0.0 <= score <= 1.0


class TestDetectCommunities:
    """Tests for community detection."""

    def test_finds_separate_clusters(self) -> None:
        """Should identify distinct clusters."""
        G = nx.Graph()
        # Two separate clusters
        G.add_edges_from([
            ("a1", "a2"), ("a2", "a3"), ("a1", "a3"),  # Cluster A
            ("b1", "b2"), ("b2", "b3"), ("b1", "b3"),  # Cluster B
        ])

        communities = detect_communities(G)

        assert len(communities) == 2
        cluster_a = {"a1", "a2", "a3"}
        cluster_b = {"b1", "b2", "b3"}
        assert cluster_a in communities or cluster_b in communities

    def test_works_with_directed_graph(self) -> None:
        """Should convert directed to undirected and find communities."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])

        communities = detect_communities(G)

        # Should find at least one community
        assert len(communities) >= 1

    def test_returns_empty_for_empty_graph(self) -> None:
        """Empty graph returns empty list."""
        G = nx.Graph()
        assert detect_communities(G) == []


class TestFindCycles:
    """Tests for circular dependency detection."""

    def test_finds_simple_cycle_with_different_files(self) -> None:
        """Should detect a circular dependency across different files."""
        G = nx.DiGraph()
        # Cycle across 3 different files
        G.add_node("a", file="file_a.py")
        G.add_node("b", file="file_b.py")
        G.add_node("c", file="file_c.py")
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

        cycles = find_cycles(G)

        assert len(cycles) == 1
        cycle_nodes = set(cycles[0])
        assert cycle_nodes == {"a", "b", "c"}

    def test_filters_intra_file_cycles_by_default(self) -> None:
        """Should filter out cycles within the same file (file ↔ symbol)."""
        G = nx.DiGraph()
        # Simulate file ↔ symbol containment cycle (false positive)
        G.add_node("py:src/utils.py", file="src/utils.py", type="module")
        G.add_node("py:src/utils.py::helper", file="src/utils.py", type="function")
        G.add_edges_from([
            ("py:src/utils.py", "py:src/utils.py::helper"),  # contains
            ("py:src/utils.py::helper", "py:src/utils.py"),  # defined_in
        ])

        cycles = find_cycles(G, cross_file_only=True)

        # Should filter out this intra-file cycle
        assert cycles == []

    def test_includes_intra_file_cycles_when_disabled(self) -> None:
        """Should include intra-file cycles when cross_file_only=False."""
        G = nx.DiGraph()
        G.add_node("py:src/utils.py", file="src/utils.py")
        G.add_node("py:src/utils.py::helper", file="src/utils.py")
        G.add_edges_from([
            ("py:src/utils.py", "py:src/utils.py::helper"),
            ("py:src/utils.py::helper", "py:src/utils.py"),
        ])

        cycles = find_cycles(G, cross_file_only=False)

        # Should find the intra-file cycle
        assert len(cycles) == 1

    def test_finds_cross_file_cycle_among_mixed_cycles(self) -> None:
        """Should find real cycles even when false positives exist."""
        G = nx.DiGraph()
        # Intra-file cycle (false positive)
        G.add_node("py:src/a.py", file="src/a.py", type="module")
        G.add_node("py:src/a.py::func", file="src/a.py", type="function")
        G.add_edges_from([
            ("py:src/a.py", "py:src/a.py::func"),
            ("py:src/a.py::func", "py:src/a.py"),
        ])

        # Real cross-file cycle
        G.add_node("py:src/b.py", file="src/b.py", type="module")
        G.add_node("py:src/c.py", file="src/c.py", type="module")
        G.add_edges_from([
            ("py:src/a.py", "py:src/b.py"),
            ("py:src/b.py", "py:src/c.py"),
            ("py:src/c.py", "py:src/a.py"),
        ])

        cycles = find_cycles(G, cross_file_only=True)

        # Should find only the real cross-file cycle
        assert len(cycles) == 1
        cycle_nodes = set(cycles[0])
        assert cycle_nodes == {"py:src/a.py", "py:src/b.py", "py:src/c.py"}

    def test_extracts_file_from_node_id_fallback(self) -> None:
        """Should extract file from node ID when file attribute is missing."""
        G = nx.DiGraph()
        # Nodes without file attribute (fallback to parsing node ID)
        G.add_node("py:src/models.py")
        G.add_node("py:src/validators.py")
        G.add_edges_from([
            ("py:src/models.py", "py:src/validators.py"),
            ("py:src/validators.py", "py:src/models.py"),
        ])

        cycles = find_cycles(G, cross_file_only=True)

        # Should detect the cross-file cycle using node ID parsing
        assert len(cycles) == 1

    def test_returns_empty_for_dag(self) -> None:
        """DAG (no cycles) should return empty list."""
        G = nx.DiGraph()
        G.add_node("a", file="a.py")
        G.add_node("b", file="b.py")
        G.add_node("c", file="c.py")
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])

        cycles = find_cycles(G)

        assert cycles == []

    def test_finds_multiple_cross_file_cycles(self) -> None:
        """Should find all cross-file cycles in the graph."""
        G = nx.DiGraph()
        G.add_node("a", file="a.py")
        G.add_node("b", file="b.py")
        G.add_node("c", file="c.py")
        G.add_node("d", file="d.py")
        G.add_edges_from([
            ("a", "b"), ("b", "a"),  # Cycle 1: a ↔ b
            ("c", "d"), ("d", "c"),  # Cycle 2: c ↔ d
        ])

        cycles = find_cycles(G)

        assert len(cycles) == 2

    def test_returns_empty_for_empty_graph(self) -> None:
        """Empty graph returns empty list."""
        G = nx.DiGraph()
        assert find_cycles(G) == []


class TestShortestPath:
    """Tests for shortest path finding."""

    def test_finds_direct_path(self) -> None:
        """Should find path between connected nodes."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        path = shortest_path(G, "a", "d")

        assert path == ["a", "b", "c", "d"]

    def test_returns_none_for_no_path(self) -> None:
        """Should return None when no path exists."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("c", "d")])  # Disconnected

        path = shortest_path(G, "a", "d")

        assert path is None

    def test_returns_none_for_missing_nodes(self) -> None:
        """Should return None for non-existent nodes."""
        G = nx.DiGraph()
        G.add_edge("a", "b")

        assert shortest_path(G, "x", "y") is None
        assert shortest_path(G, "a", "z") is None


class TestStronglyConnected:
    """Tests for strongly connected components."""

    def test_finds_coupled_groups(self, backend: str) -> None:
        """Should identify mutually dependent modules."""
        G = nx.DiGraph()
        # A group where all nodes can reach each other
        G.add_edges_from([
            ("a", "b"), ("b", "c"), ("c", "a"),
        ])
        # An isolated node
        G.add_node("d")

        sccs = strongly_connected(G)

        assert len(sccs) == 1
        assert sccs[0] == {"a", "b", "c"}

    def test_excludes_single_node_components(self, backend: str) -> None:
        """Should only return components with 2+ nodes."""
        G = nx.DiGraph()
        G.add_nodes_from(["a", "b", "c"])

        sccs = strongly_connected(G)

        assert sccs == []

    def test_sorts_by_size_descending(self, backend: str) -> None:
        """Larger components should come first."""
        G = nx.DiGraph()
        # Large cycle
        G.add_edges_from([
            ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ])
        # Small cycle
        G.add_edges_from([("x", "y"), ("y", "x")])

        sccs = strongly_connected(G)

        assert len(sccs) == 2
        assert len(sccs[0]) > len(sccs[1])

    def test_returns_empty_for_empty_graph(self, backend: str) -> None:
        """Empty graph returns empty list."""
        G = nx.DiGraph()
        assert strongly_connected(G) == []


class TestTopN:
    """Tests for top_n helper."""

    def test_returns_top_items(self) -> None:
        """Should return the N highest scoring items."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.9}

        result = top_n(scores, n=2)

        assert result == [("d", 0.9), ("b", 0.5)]

    def test_ascending_order(self) -> None:
        """Should support ascending order."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.3}

        result = top_n(scores, n=2, descending=False)

        assert result == [("a", 0.1), ("c", 0.3)]

    def test_returns_all_if_n_exceeds_length(self) -> None:
        """Should return all items if n > len(scores)."""
        scores = {"a": 0.1, "b": 0.2}

        result = top_n(scores, n=10)

        assert len(result) == 2


class TestAncestorsAtDepth:
    """Tests for ancestors_at_depth traversal."""

    def test_finds_immediate_ancestors(self, backend: str) -> None:
        """Should find direct predecessors at depth 1."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])

        result = ancestors_at_depth(G, "c", max_depth=1)

        assert result == {"b": 1}

    def test_respects_max_depth(self, backend: str) -> None:
        """Should not exceed max_depth."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        result = ancestors_at_depth(G, "d", max_depth=2)

        assert "c" in result
        assert "b" in result
        assert "a" not in result

    def test_returns_empty_for_missing_node(self, backend: str) -> None:
        """Should return empty dict for non-existent node."""
        G = nx.DiGraph()
        G.add_edge("a", "b")

        assert ancestors_at_depth(G, "x", max_depth=3) == {}


class TestDescendantsAtDepth:
    """Tests for descendants_at_depth traversal."""

    def test_finds_immediate_descendants(self, backend: str) -> None:
        """Should find direct successors at depth 1."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])

        result = descendants_at_depth(G, "a", max_depth=1)

        assert result == {"b": 1}

    def test_respects_max_depth(self, backend: str) -> None:
        """Should not exceed max_depth."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        result = descendants_at_depth(G, "a", max_depth=2)

        assert "b" in result
        assert "c" in result
        assert "d" not in result

    def test_returns_empty_for_missing_node(self, backend: str) -> None:
        """Should return empty dict for non-existent node."""
        G = nx.DiGraph()
        G.add_edge("a", "b")

        assert descendants_at_depth(G, "x", max_depth=3) == {}


class TestCommunityMetagraph:
    """Tests for community_metagraph meta-graph analysis."""

    def test_computes_cohesion_for_isolated_community(self) -> None:
        """Isolated community with no external edges should have cohesion=1.0."""
        G = nx.DiGraph()
        # Single tight cluster with internal edges only
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

        communities = [{"a", "b", "c"}]
        metrics, inter_edges = community_metagraph(G, communities)

        assert len(metrics) == 1
        assert metrics[0]["cohesion"] == 1.0
        assert metrics[0]["internal_edges"] == 3
        assert metrics[0]["external_edges"] == 0
        assert inter_edges == []

    def test_finds_inter_community_edges(self) -> None:
        """Should detect edges between communities."""
        G = nx.DiGraph()
        # Two clusters connected by edges
        G.add_edges_from([
            ("a1", "a2"), ("a2", "a1"),  # Cluster A internal
            ("b1", "b2"), ("b2", "b1"),  # Cluster B internal
            ("a1", "b1"), ("a2", "b2"),  # Cross-cluster edges
        ])

        communities = [{"a1", "a2"}, {"b1", "b2"}]
        metrics, inter_edges = community_metagraph(G, communities)

        assert len(metrics) == 2
        assert len(inter_edges) >= 1

        # Find the inter-edge from cluster 0 to cluster 1
        edge = next((e for e in inter_edges if e["source"] == 0 and e["target"] == 1), None)
        assert edge is not None
        assert edge["edge_count"] == 2

    def test_cohesion_decreases_with_external_edges(self) -> None:
        """More external edges should lower cohesion."""
        G = nx.DiGraph()
        # Cluster with some external edges
        G.add_edges_from([
            ("a", "b"), ("b", "c"),  # Internal: 2
            ("a", "x"), ("b", "x"), ("c", "x"),  # External: 3
        ])

        communities = [{"a", "b", "c"}, {"x"}]
        metrics, _ = community_metagraph(G, communities)

        cluster_a_metrics = next(m for m in metrics if m["module_id"] == 0)
        assert cluster_a_metrics["internal_edges"] == 2
        assert cluster_a_metrics["external_edges"] == 3  # efferent only for cluster A
        assert cluster_a_metrics["cohesion"] < 1.0
        # cohesion = 2 / (2 + 3) = 0.4
        assert abs(cluster_a_metrics["cohesion"] - 0.4) < 0.01

    def test_computes_afferent_and_efferent_coupling(self) -> None:
        """Should track incoming and outgoing cross-community edges."""
        G = nx.DiGraph()
        G.add_edges_from([
            ("a", "b"),  # Internal to cluster A
            ("a", "x"),  # A -> X (efferent for A, afferent for X)
            ("x", "a"),  # X -> A (afferent for A, efferent for X)
        ])

        communities = [{"a", "b"}, {"x"}]
        metrics, _ = community_metagraph(G, communities)

        cluster_a = next(m for m in metrics if m["module_id"] == 0)
        cluster_x = next(m for m in metrics if m["module_id"] == 1)

        assert cluster_a["afferent_coupling"] == 1  # x -> a
        assert cluster_a["efferent_coupling"] == 1  # a -> x

        assert cluster_x["afferent_coupling"] == 1  # a -> x
        assert cluster_x["efferent_coupling"] == 1  # x -> a

    def test_instability_calculation(self) -> None:
        """Instability = efferent / (afferent + efferent)."""
        G = nx.DiGraph()
        G.add_edges_from([
            ("a", "x"),  # efferent for A
            ("a", "y"),  # efferent for A
            ("z", "a"),  # afferent for A
        ])

        communities = [{"a"}, {"x", "y", "z"}]
        metrics, _ = community_metagraph(G, communities)

        cluster_a = next(m for m in metrics if m["module_id"] == 0)
        # instability = 2 / (1 + 2) = 0.6667
        assert abs(cluster_a["instability"] - 0.6667) < 0.01

    def test_returns_empty_for_empty_graph(self) -> None:
        """Empty graph returns empty results."""
        G = nx.DiGraph()
        metrics, inter_edges = community_metagraph(G)

        assert metrics == []
        assert inter_edges == []

    def test_detects_communities_if_not_provided(self) -> None:
        """Should auto-detect communities when not provided."""
        G = nx.DiGraph()
        # Two disconnected clusters
        G.add_edges_from([("a", "b"), ("b", "a")])
        G.add_edges_from([("x", "y"), ("y", "x")])

        metrics, inter_edges = community_metagraph(G, communities=None)

        # Should detect 2 communities
        assert len(metrics) == 2
        # No inter-community edges since clusters are disconnected
        assert inter_edges == []

    def test_inter_edges_sorted_by_count_descending(self) -> None:
        """Inter-community edges should be sorted by edge_count descending."""
        G = nx.DiGraph()
        G.add_edges_from([
            ("a", "x"),  # 1 edge A->X
            ("b", "y"), ("b", "z"), ("b", "w"),  # 3 edges B->Y
        ])

        communities = [{"a"}, {"b"}, {"x", "y", "z", "w"}]
        _, inter_edges = community_metagraph(G, communities)

        # Should be sorted by edge_count descending
        if len(inter_edges) >= 2:
            assert inter_edges[0]["edge_count"] >= inter_edges[1]["edge_count"]


class TestHierarchicalView:
    """Tests for hierarchical_view graph collapsing."""

    def test_module_level_aggregates_by_file(self) -> None:
        """Module level should collapse nodes to file level."""
        G = nx.DiGraph()
        G.add_node("py:src/api/users.py::get_user", file="src/api/users.py", type="function")
        G.add_node("py:src/api/users.py::list_users", file="src/api/users.py", type="function")
        G.add_node("py:src/models/user.py::User", file="src/models/user.py", type="class")
        G.add_edges_from([
            ("py:src/api/users.py::get_user", "py:src/models/user.py::User"),
            ("py:src/api/users.py::list_users", "py:src/models/user.py::User"),
        ])

        nodes, edges = hierarchical_view(G, level="module")

        # Should have 2 hierarchy nodes (2 files)
        assert len(nodes) == 2
        node_ids = {n["id"] for n in nodes}
        assert "src/api/users.py" in node_ids
        assert "src/models/user.py" in node_ids

        # Should have 1 aggregated edge (api -> models)
        assert len(edges) == 1
        assert edges[0]["edge_count"] == 2

    def test_package_level_aggregates_by_directory(self) -> None:
        """Package level should collapse nodes to directory level."""
        G = nx.DiGraph()
        G.add_node("a", file="src/api/users.py", type="function")
        G.add_node("b", file="src/api/auth.py", type="function")
        G.add_node("c", file="src/models/user.py", type="class")
        G.add_edges_from([("a", "c"), ("b", "c")])

        nodes, edges = hierarchical_view(G, level="package")

        # Should have 2 packages
        node_ids = {n["id"] for n in nodes}
        assert "src/api/" in node_ids
        assert "src/models/" in node_ids

    def test_function_level_returns_original_structure(self) -> None:
        """Function level should return original graph without aggregation."""
        G = nx.DiGraph()
        G.add_node("a", file="f1.py", type="function")
        G.add_node("b", file="f2.py", type="function")
        G.add_edge("a", "b", type="calls")

        nodes, edges = hierarchical_view(G, level="function")

        assert len(nodes) == 2
        assert len(edges) == 1

    def test_self_loops_excluded_in_aggregated_view(self) -> None:
        """Edges within the same hierarchy node should be excluded."""
        G = nx.DiGraph()
        G.add_node("a", file="src/utils.py", type="function")
        G.add_node("b", file="src/utils.py", type="function")
        G.add_edge("a", "b", type="calls")

        nodes, edges = hierarchical_view(G, level="module")

        # One hierarchy node, no edges (internal edge becomes self-loop)
        assert len(nodes) == 1
        assert len(edges) == 0

    def test_edge_types_aggregated(self) -> None:
        """Edge types should be counted in aggregated edges."""
        G = nx.DiGraph()
        G.add_node("a1", file="a.py", type="function")
        G.add_node("a2", file="a.py", type="function")
        G.add_node("b", file="b.py", type="class")
        G.add_edge("a1", "b", type="calls")
        G.add_edge("a2", "b", type="imports")

        nodes, edges = hierarchical_view(G, level="module")

        assert len(edges) == 1
        assert edges[0]["edge_count"] == 2
        assert edges[0]["edge_types"] == {"calls": 1, "imports": 1}

    def test_invalid_level_raises_error(self) -> None:
        """Invalid level should raise ValueError."""
        G = nx.DiGraph()
        G.add_node("a", file="a.py")

        with pytest.raises(ValueError, match="Invalid level"):
            hierarchical_view(G, level="invalid")

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph should return empty lists."""
        G = nx.DiGraph()
        nodes, edges = hierarchical_view(G, level="module")

        assert nodes == []
        assert edges == []

    def test_children_limited_for_display(self) -> None:
        """Children list should be limited to prevent huge output."""
        G = nx.DiGraph()
        # Add 20 nodes in the same file
        for i in range(20):
            G.add_node(f"func_{i}", file="big_file.py", type="function")

        nodes, _ = hierarchical_view(G, level="module")

        assert len(nodes) == 1
        assert nodes[0]["child_count"] == 20
        assert len(nodes[0]["children"]) == 10  # Limited to 10


class TestValidateLayers:
    """Tests for validate_layers architecture validation."""

    def test_valid_top_to_bottom_dependencies(self) -> None:
        """Dependencies from top to bottom layers should be valid."""
        G = nx.DiGraph()
        G.add_node("api_handler", file="api/users.py", type="function")
        G.add_node("user_service", file="services/user.py", type="function")
        G.add_node("user_model", file="models/user.py", type="class")
        G.add_edges_from([
            ("api_handler", "user_service"),
            ("user_service", "user_model"),
        ])

        layers = [{"api/"}, {"services/"}, {"models/"}]
        violations, valid, total = validate_layers(G, layers)

        assert violations == []
        assert valid == 2
        assert total == 2

    def test_bottom_to_top_dependency_is_violation(self) -> None:
        """Dependencies from bottom to top layers should be violations."""
        G = nx.DiGraph()
        G.add_node("api_handler", file="api/users.py", type="function")
        G.add_node("user_model", file="models/user.py", type="class")
        # Violation: model depends on API
        G.add_edge("user_model", "api_handler", type="imports")

        layers = [{"api/"}, {"models/"}]
        layer_names = ["API", "Models"]
        violations, valid, total = validate_layers(G, layers, layer_names)

        assert len(violations) == 1
        assert violations[0]["source_layer_name"] == "Models"
        assert violations[0]["target_layer_name"] == "API"
        assert violations[0]["severity"] == "error"

    def test_same_layer_dependencies_are_valid(self) -> None:
        """Dependencies within the same layer should be valid."""
        G = nx.DiGraph()
        G.add_node("auth_service", file="services/auth.py", type="function")
        G.add_node("user_service", file="services/user.py", type="function")
        G.add_edge("auth_service", "user_service", type="calls")

        layers = [{"api/"}, {"services/"}, {"models/"}]
        violations, valid, total = validate_layers(G, layers)

        assert violations == []
        assert valid == 1
        assert total == 1

    def test_unclassified_nodes_ignored(self) -> None:
        """Nodes not in any layer should be ignored."""
        G = nx.DiGraph()
        G.add_node("api_handler", file="api/users.py", type="function")
        G.add_node("unknown", file="utils/helper.py", type="function")
        G.add_edge("api_handler", "unknown", type="calls")

        layers = [{"api/"}, {"services/"}]
        violations, valid, total = validate_layers(G, layers)

        # Edge involves unclassified node, so not counted
        assert violations == []
        assert total == 0

    def test_compliance_rate_calculation(self) -> None:
        """Compliance rate should be calculated correctly."""
        G = nx.DiGraph()
        G.add_node("a", file="api/a.py", type="function")
        G.add_node("s", file="services/s.py", type="function")
        G.add_node("m", file="models/m.py", type="class")
        G.add_edges_from([
            ("a", "s"),  # Valid
            ("s", "m"),  # Valid
            ("m", "a"),  # Violation
        ])

        layers = [{"api/"}, {"services/"}, {"models/"}]
        violations, valid, total = validate_layers(G, layers)

        assert len(violations) == 1
        assert valid == 2
        assert total == 3

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph should return no violations."""
        G = nx.DiGraph()
        violations, valid, total = validate_layers(G, [{"api/"}, {"models/"}])

        assert violations == []
        assert valid == 0
        assert total == 0

    def test_empty_layers_returns_empty(self) -> None:
        """Empty layers list should return no violations."""
        G = nx.DiGraph()
        G.add_node("a", file="api/a.py")
        violations, valid, total = validate_layers(G, [])

        assert violations == []
        assert valid == 0
        assert total == 0

    def test_list_layers_normalized_to_sets(self) -> None:
        """Lists in layers should be normalized to sets."""
        G = nx.DiGraph()
        G.add_node("a", file="api/a.py", type="function")
        G.add_node("m", file="models/m.py", type="class")
        G.add_edge("a", "m", type="imports")

        # Pass lists instead of sets
        layers = [["api/"], ["models/"]]
        violations, valid, total = validate_layers(G, layers)

        assert violations == []
        assert valid == 1

    def test_multiple_patterns_per_layer(self) -> None:
        """Multiple path patterns should work per layer."""
        G = nx.DiGraph()
        G.add_node("handler", file="api/handler.py", type="function")
        G.add_node("route", file="routes/user.py", type="function")
        G.add_node("model", file="models/user.py", type="class")
        G.add_edges_from([
            ("handler", "model"),
            ("route", "model"),
        ])

        # api/ and routes/ are both in the top layer
        layers = [{"api/", "routes/"}, {"models/"}]
        violations, valid, total = validate_layers(G, layers)

        assert violations == []
        assert valid == 2

    def test_violation_includes_edge_type(self) -> None:
        """Violations should include the edge type."""
        G = nx.DiGraph()
        G.add_node("m", file="models/m.py", type="class")
        G.add_node("a", file="api/a.py", type="function")
        G.add_edge("m", "a", type="imports")

        layers = [{"api/"}, {"models/"}]
        violations, _, _ = validate_layers(G, layers)

        assert len(violations) == 1
        assert violations[0]["edge_type"] == "imports"
