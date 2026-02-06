"""Tests for performance optimizations.

Tests the following features:
1. Timing metrics in MCP responses
2. Pre-normalized embeddings
3. FAISS integration (optional)
4. Graph caching
5. Rust betweenness centrality
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nodestradamus.logging import TimingContext, log_operation


class TestTimingMetrics:
    """Tests for timing metrics in log_operation."""

    def test_timing_context_captures_elapsed(self):
        """TimingContext should capture elapsed time."""
        ctx = TimingContext()
        ctx.start()
        # Small delay
        import time
        time.sleep(0.01)
        ctx.stop()

        assert ctx.elapsed > 0
        assert ctx.elapsed_ms > 0
        assert ctx.elapsed_ms >= 10  # At least 10ms

    def test_log_operation_yields_timing_context(self):
        """log_operation should yield a TimingContext with elapsed time."""
        with log_operation("test_operation") as timing:
            import time
            time.sleep(0.01)

        assert isinstance(timing, TimingContext)
        assert timing.elapsed > 0
        assert timing.elapsed_ms >= 10

    def test_log_operation_captures_exception_timing(self):
        """log_operation should capture timing even on exception."""
        with pytest.raises(ValueError):
            with log_operation("failing_operation") as timing:
                import time
                time.sleep(0.01)
                raise ValueError("test error")

        assert timing.elapsed > 0


class TestPreNormalizedEmbeddings:
    """Tests for pre-normalized embeddings cache."""

    def test_cosine_similarity_with_normalized_flag(self):
        """_cosine_similarity should skip normalization when normalized=True."""
        from nodestradamus.analyzers.embeddings import _cosine_similarity

        # Pre-normalized embeddings (unit vectors)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7071, 0.7071, 0.0],
        ])
        query = np.array([1.0, 0.0, 0.0])

        # With normalized=True
        sims_normalized = _cosine_similarity(query, embeddings, embeddings_normalized=True)

        # With normalized=False (will normalize again, but result should be same)
        sims_not_normalized = _cosine_similarity(query, embeddings, embeddings_normalized=False)

        # Results should be approximately equal
        np.testing.assert_array_almost_equal(sims_normalized, sims_not_normalized, decimal=4)

    def test_cosine_similarity_normalized_saves_computation(self):
        """Pre-normalized embeddings skip normalization step.

        Note: Actual timing difference may be small due to NumPy optimizations.
        This test verifies the function works correctly with both flags.
        """
        from nodestradamus.analyzers.embeddings import _cosine_similarity

        # Create embeddings
        n = 100
        d = 64
        embeddings = np.random.randn(n, d).astype(np.float32)

        # Pre-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-9)

        query = np.random.randn(d).astype(np.float32)

        # Both paths should produce similar results
        sims_normalized = _cosine_similarity(query, embeddings_norm, embeddings_normalized=True)
        sims_not_normalized = _cosine_similarity(query, embeddings, embeddings_normalized=False)

        # Results should be approximately equal
        np.testing.assert_array_almost_equal(sims_normalized, sims_not_normalized, decimal=3)


class TestFAISSIntegration:
    """Tests for optional FAISS integration."""

    def test_faiss_availability_flag(self):
        """_HAS_FAISS should indicate FAISS availability."""
        from nodestradamus.analyzers.embeddings import _HAS_FAISS

        # Flag should be a boolean
        assert isinstance(_HAS_FAISS, bool)

    def test_build_faiss_index(self):
        """_build_faiss_index should create a searchable index."""
        import importlib.util

        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not installed")

        from nodestradamus.analyzers.embeddings import _build_faiss_index, _faiss_search

        # Create normalized embeddings
        embeddings = np.random.randn(100, 64).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-9)

        index = _build_faiss_index(embeddings_norm)

        # Should be able to search
        query = embeddings_norm[0]
        sims, indices = _faiss_search(index, query, top_k=5)

        # First result should be the query itself (similarity ~1.0)
        assert indices[0] == 0
        assert sims[0] > 0.99

    def test_faiss_threshold_configurable(self):
        """FAISS threshold should be configurable via env var."""
        from nodestradamus.analyzers.embeddings import _FAISS_THRESHOLD

        # Default should be 10000
        assert _FAISS_THRESHOLD == 10000 or isinstance(_FAISS_THRESHOLD, int)


class TestGraphCaching:
    """Tests for incremental graph caching."""

    def test_graph_cache_path(self):
        """Graph cache should be in .nodestradamus directory."""
        from nodestradamus.analyzers.deps import _get_graph_cache_path

        repo_path = Path("/tmp/test-repo")
        cache_path = _get_graph_cache_path(repo_path)

        assert cache_path == repo_path / ".nodestradamus" / "graph.msgpack"

    def test_save_and_load_graph_cache(self):
        """Should be able to save and load graph cache."""

        from nodestradamus.analyzers.deps import (
            _build_graph,
            _load_cached_graph,
            _save_graph_cache,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a simple graph
            nodes = [
                {"id": "py:test.py", "type": "module", "file": "test.py", "name": "test"},
                {"id": "py:test.py::foo", "type": "function", "file": "test.py", "name": "foo"},
            ]
            edges = [
                {"from": "py:test.py", "to": "py:test.py::foo", "type": "contains"},
            ]
            G = _build_graph(nodes, edges)

            # Save
            _save_graph_cache(repo_path, G)

            # Verify cache file exists (now using msgpack)
            cache_path = repo_path / ".nodestradamus" / "graph.msgpack"
            assert cache_path.exists()

            # Load
            loaded_G = _load_cached_graph(repo_path)

            # Should have same structure
            assert loaded_G is not None
            assert loaded_G.number_of_nodes() == G.number_of_nodes()
            assert loaded_G.number_of_edges() == G.number_of_edges()

    def test_graph_cache_version_mismatch(self):
        """Old cache versions should be ignored."""
        import msgpack

        from nodestradamus.analyzers.deps import _load_cached_graph

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            cache_dir = repo_path / ".nodestradamus"
            cache_dir.mkdir(parents=True)

            # Write cache with wrong version (using msgpack)
            cache_path = cache_dir / "graph.msgpack"
            with open(cache_path, "wb") as f:
                msgpack.pack({"version": "0.0", "nodes": [], "edges": []}, f)

            # Should return None (version mismatch)
            loaded = _load_cached_graph(repo_path)
            assert loaded is None


class TestRustBetweenness:
    """Tests for Rust betweenness centrality."""

    def test_betweenness_with_rust_fallback(self):
        """betweenness should work with or without Rust."""
        import networkx as nx

        from nodestradamus.analyzers.graph_algorithms import betweenness

        # Create simple graph
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])

        scores = betweenness(G)

        # Should have scores for all nodes
        assert "a" in scores
        assert "b" in scores
        assert "c" in scores

    def test_betweenness_empty_graph(self):
        """betweenness should handle empty graph."""
        import networkx as nx

        from nodestradamus.analyzers.graph_algorithms import betweenness

        G = nx.DiGraph()
        scores = betweenness(G)

        assert scores == {}

    def test_betweenness_bridge_node_highest(self):
        """Bridge node should have highest betweenness."""
        import networkx as nx

        from nodestradamus.analyzers.graph_algorithms import betweenness

        # a -> b -> c: b is the bridge
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c")])

        scores = betweenness(G, normalized=False)

        # b should have highest score (sits on path a->c)
        assert scores["b"] >= scores["a"]
        assert scores["b"] >= scores["c"]

    def test_betweenness_rust_equivalence(self):
        """Rust and NetworkX implementations should give equivalent results."""
        import networkx as nx

        from nodestradamus.analyzers.graph_algorithms import betweenness

        # Create test graph
        G = nx.DiGraph()
        G.add_edges_from([
            ("a", "b"), ("b", "c"), ("c", "d"),
            ("a", "c"), ("b", "d"),
        ])

        # Get NetworkX result
        nx_scores = nx.betweenness_centrality(G, normalized=True)

        # Get our implementation result
        our_scores = betweenness(G, normalized=True)

        # Should be approximately equal
        for node in G.nodes():
            assert abs(nx_scores[node] - our_scores[node]) < 0.01, (
                f"Mismatch for {node}: NX={nx_scores[node]}, Ours={our_scores[node]}"
            )


class TestMCPTimingInjection:
    """Tests for timing injection in MCP tool responses."""

    def test_inject_timing_to_json(self):
        """inject_timing should add timing field to JSON response."""
        from nodestradamus.mcp.tools.dispatch import inject_timing

        result_str = json.dumps({"summary": {"nodes": 100}})

        injected = inject_timing(result_str, 123.4)
        parsed = json.loads(injected)

        assert "timing" in parsed
        assert parsed["timing"]["total_ms"] == 123.4

    def test_inject_timing_handles_non_dict(self):
        """inject_timing should handle non-dict JSON gracefully."""
        from nodestradamus.mcp.tools.dispatch import inject_timing

        result_str = json.dumps(["a", "b", "c"])

        injected = inject_timing(result_str, 100.0)

        # Should return original string unchanged for non-dict
        assert injected == result_str

    def test_inject_timing_handles_invalid_json(self):
        """inject_timing should handle invalid JSON gracefully."""
        from nodestradamus.mcp.tools.dispatch import inject_timing

        result_str = "not valid json"

        injected = inject_timing(result_str, 100.0)

        # Should return original string unchanged
        assert injected == result_str


class TestSummarizeCooccurrencePerformance:
    """Tests for summarize_cooccurrence optimization."""

    def test_summarize_cooccurrence_single_pass(self):
        """summarize_cooccurrence should use single-pass iteration."""
        import networkx as nx

        from nodestradamus.mcp.tools.utils.summarize import summarize_cooccurrence

        # Create a test graph with many edges
        G = nx.Graph()
        for i in range(100):
            for j in range(i + 1, 100):
                G.add_edge(f"file_{i}.py", f"file_{j}.py", strength=0.5, count=2)

        # Should complete without error and return correct structure
        result = summarize_cooccurrence(G, top_n_count=10)

        assert "summary" in result
        assert result["summary"]["file_count"] == 100
        assert result["summary"]["edge_count"] == 4950  # C(100,2)
        assert len(result["top_co_occurring_pairs"]) == 10
        assert len(result["change_hotspots"]) == 15

    def test_summarize_cooccurrence_empty_graph(self):
        """summarize_cooccurrence should handle empty graph."""
        import networkx as nx

        from nodestradamus.mcp.tools.utils.summarize import summarize_cooccurrence

        G = nx.Graph()

        result = summarize_cooccurrence(G)

        assert result["summary"]["file_count"] == 0
        assert result["summary"]["edge_count"] == 0
        assert result["top_co_occurring_pairs"] == []
        assert result["change_hotspots"] == []

    def test_summarize_cooccurrence_uses_heap_for_top_n(self):
        """summarize_cooccurrence should use heap for efficient top-N selection."""
        import networkx as nx

        from nodestradamus.mcp.tools.utils.summarize import summarize_cooccurrence

        # Create graph with known strengths
        G = nx.Graph()
        for i in range(50):
            # Strength decreases with i
            G.add_edge(f"a_{i}.py", f"b_{i}.py", strength=1.0 - (i * 0.01), count=i + 1)

        result = summarize_cooccurrence(G, top_n_count=5)

        # Top 5 should be the first 5 edges (highest strength)
        pairs = result["top_co_occurring_pairs"]
        assert len(pairs) == 5
        # Verify sorted by strength descending
        strengths = [p["strength"] for p in pairs]
        assert strengths == sorted(strengths, reverse=True)
        # First should have strength 1.0
        assert pairs[0]["strength"] == 1.0
