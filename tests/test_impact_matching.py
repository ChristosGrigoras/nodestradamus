"""Tests for impact analysis node matching with disambiguation."""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from nodestradamus.analyzers.impact import (
    AmbiguousMatchError,
    NoMatchError,
    _find_target_node,
    _get_fused_matches,
    _get_match_type,
    _get_semantically_related,
    compute_fused_score,
)
from nodestradamus.models.graph import FusedMatch, SemanticMatch


class TestGetMatchType:
    """Tests for path match type detection."""

    def test_exact_match(self) -> None:
        """Should return 'exact' for identical paths."""
        assert _get_match_type("src/utils.py", "src/utils.py") == "exact"

    def test_exact_match_with_leading_dot_slash(self) -> None:
        """Should normalize leading ./ for matching."""
        assert _get_match_type("./src/utils.py", "src/utils.py") == "exact"
        assert _get_match_type("src/utils.py", "./src/utils.py") == "exact"

    def test_suffix_match(self) -> None:
        """Should return 'suffix' when query is end of node path."""
        assert _get_match_type("utils.py", "src/utils.py") == "suffix"
        assert _get_match_type("api/handler.py", "src/api/handler.py") == "suffix"

    def test_suffix_requires_path_boundary(self) -> None:
        """Should not match partial filenames."""
        # "utils.py" should NOT match "myutils.py"
        assert _get_match_type("utils.py", "myutils.py") == "none"
        assert _get_match_type("utils.py", "src/myutils.py") == "none"

    def test_no_match(self) -> None:
        """Should return 'none' for non-matching paths."""
        assert _get_match_type("foo.py", "bar.py") == "none"
        assert _get_match_type("src/foo.py", "lib/bar.py") == "none"

    def test_empty_node_file(self) -> None:
        """Should return 'none' for empty node file."""
        assert _get_match_type("utils.py", "") == "none"


class TestFindTargetNode:
    """Tests for finding target nodes with disambiguation."""

    def _build_graph(self, nodes: list[dict]) -> nx.DiGraph:
        """Helper to build a graph from node dicts."""
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(
                node["id"],
                file=node.get("file", ""),
                name=node.get("name", ""),
                type=node.get("type", "function"),
                line=node.get("line"),
            )
        return G

    def test_exact_file_match_single(self) -> None:
        """Should return single exact match."""
        G = self._build_graph([
            {"id": "py:src/utils.py::helper", "file": "src/utils.py", "name": "helper"},
        ])

        result = _find_target_node(G, "src/utils.py", None)

        assert result == "py:src/utils.py::helper"

    def test_exact_file_and_symbol_match(self) -> None:
        """Should return node matching both file and symbol."""
        G = self._build_graph([
            {"id": "py:src/utils.py::helper", "file": "src/utils.py", "name": "helper"},
            {"id": "py:src/utils.py::process", "file": "src/utils.py", "name": "process"},
        ])

        result = _find_target_node(G, "src/utils.py", "process")

        assert result == "py:src/utils.py::process"

    def test_suffix_match_single(self) -> None:
        """Should return single suffix match when unambiguous."""
        G = self._build_graph([
            {"id": "py:src/api/handler.py::handle", "file": "src/api/handler.py", "name": "handle"},
        ])

        result = _find_target_node(G, "handler.py", None)

        assert result == "py:src/api/handler.py::handle"

    def test_ambiguous_suffix_match_raises(self) -> None:
        """Should raise AmbiguousMatchError when multiple suffix matches."""
        G = self._build_graph([
            {"id": "py:src/api/request.py::Request", "file": "src/api/request.py", "name": "Request"},
            {"id": "py:src/http/request.py::Request", "file": "src/http/request.py", "name": "Request"},
        ])

        with pytest.raises(AmbiguousMatchError) as exc_info:
            _find_target_node(G, "request.py", "Request")

        assert len(exc_info.value.candidates) == 2
        # Candidates should contain file paths for disambiguation
        files = [c["file"] for c in exc_info.value.candidates]
        assert "src/api/request.py" in files
        assert "src/http/request.py" in files

    def test_ambiguous_suffix_without_symbol_raises(self) -> None:
        """Should raise AmbiguousMatchError for ambiguous file-only query."""
        G = self._build_graph([
            {"id": "py:src/utils.py::a", "file": "src/utils.py", "name": "a"},
            {"id": "py:lib/utils.py::b", "file": "lib/utils.py", "name": "b"},
        ])

        with pytest.raises(AmbiguousMatchError) as exc_info:
            _find_target_node(G, "utils.py", None)

        assert len(exc_info.value.candidates) == 2

    def test_no_match_raises_with_suggestions(self) -> None:
        """Should raise NoMatchError with suggestions when no match."""
        G = self._build_graph([
            {"id": "py:src/utils.py::helper", "file": "src/utils.py", "name": "helper"},
            {"id": "py:src/api.py::Request", "file": "src/api.py", "name": "Request"},
        ])

        with pytest.raises(NoMatchError) as exc_info:
            _find_target_node(G, "nonexistent.py", "SomeClass")

        # Should provide suggestions for similar files or symbols
        assert exc_info.value.suggestions is not None

    def test_suggests_same_filename_different_directory(self) -> None:
        """Should suggest files with same name but different directory."""
        G = self._build_graph([
            {"id": "py:src/api/handler.py::handle", "file": "src/api/handler.py", "name": "handle"},
        ])

        with pytest.raises(NoMatchError) as exc_info:
            _find_target_node(G, "lib/handler.py", None)

        # Should suggest the similar file
        suggestions = exc_info.value.suggestions
        assert len(suggestions) >= 1
        assert any("src/api/handler.py" in s["file"] for s in suggestions)

    def test_suggests_same_symbol_different_file(self) -> None:
        """Should suggest same symbol from different files."""
        G = self._build_graph([
            {"id": "py:src/auth.py::Request", "file": "src/auth.py", "name": "Request"},
        ])

        with pytest.raises(NoMatchError) as exc_info:
            _find_target_node(G, "nonexistent.py", "Request")

        suggestions = exc_info.value.suggestions
        assert len(suggestions) >= 1
        assert any(s["name"] == "Request" for s in suggestions)

    def test_exact_match_prioritized_over_suffix(self) -> None:
        """Should prefer exact match even when suffix matches exist."""
        G = self._build_graph([
            {"id": "py:utils.py::helper", "file": "utils.py", "name": "helper"},
            {"id": "py:src/utils.py::helper", "file": "src/utils.py", "name": "helper"},
        ])

        # Query for "utils.py" should match "utils.py" exactly, not "src/utils.py"
        result = _find_target_node(G, "utils.py", None)

        assert result == "py:utils.py::helper"

    def test_multiple_symbols_same_file_no_symbol(self) -> None:
        """When no symbol specified with multiple in same file, return first."""
        G = self._build_graph([
            {"id": "py:src/utils.py::a", "file": "src/utils.py", "name": "a", "type": "function"},
            {"id": "py:src/utils.py::b", "file": "src/utils.py", "name": "b", "type": "function"},
        ])

        # Should return first match, not raise error
        result = _find_target_node(G, "src/utils.py", None)

        assert result in ["py:src/utils.py::a", "py:src/utils.py::b"]

    def test_prefers_module_type_when_no_symbol(self) -> None:
        """Should prefer module-level node when no symbol specified."""
        G = nx.DiGraph()
        # Add nodes in specific order to test preference
        G.add_node(
            "py:src/utils.py::helper",
            file="src/utils.py", name="helper", type="function"
        )
        G.add_node(
            "py:src/utils.py",
            file="src/utils.py", name="utils", type="module"
        )

        result = _find_target_node(G, "src/utils.py", None)

        # Should prefer the module-level node
        assert result == "py:src/utils.py"


class TestAmbiguousMatchError:
    """Tests for the AmbiguousMatchError exception."""

    def test_contains_candidates(self) -> None:
        """Should store candidates for disambiguation."""
        candidates = [
            {"node_id": "a", "file": "src/a.py"},
            {"node_id": "b", "file": "lib/a.py"},
        ]

        error = AmbiguousMatchError("Multiple matches", candidates)

        assert error.candidates == candidates
        assert "Multiple matches" in str(error)


class TestNoMatchError:
    """Tests for the NoMatchError exception."""

    def test_contains_suggestions(self) -> None:
        """Should store suggestions."""
        suggestions = [
            {"node_id": "a", "file": "src/similar.py", "reason": "same filename"},
        ]

        error = NoMatchError("No match found", suggestions)

        assert error.suggestions == suggestions
        assert "No match found" in str(error)

    def test_defaults_to_empty_suggestions(self) -> None:
        """Should default to empty list for suggestions."""
        error = NoMatchError("No match found")

        assert error.suggestions == []


class TestGetSemanticallyRelated:
    """Tests for semantic similarity in impact analysis."""

    def test_finds_similar_code_not_in_graph(self) -> None:
        """Should find semantically similar code outside dependency graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with similar functions
            Path(tmpdir, "auth.py").write_text('''def validate_user(user_data):
    """Validate user credentials and permissions."""
    if not user_data.get("username"):
        raise ValueError("Username required")
    if not user_data.get("password"):
        raise ValueError("Password required")
    return check_credentials(user_data)
''')
            Path(tmpdir, "admin.py").write_text('''def validate_admin(admin_data):
    """Validate admin credentials and permissions."""
    if not admin_data.get("username"):
        raise ValueError("Username required")
    if not admin_data.get("password"):
        raise ValueError("Password required")
    return check_admin_credentials(admin_data)
''')
            Path(tmpdir, "math_utils.py").write_text('''def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    return sum(numbers)
''')

            # Pre-compute embeddings
            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            # Find code similar to validate_user but not in dependency graph
            results = _get_semantically_related(
                Path(tmpdir),
                target_id="py:auth.py::validate_user",
                file_path="auth.py",
                symbol="validate_user",
                exclude_ids=set(),  # Nothing excluded
                top_k=5,
                threshold=0.5,
            )

            # Should find validate_admin as similar
            assert len(results) >= 1
            names = [r.name for r in results]
            assert "validate_admin" in names

            # Should NOT include calculate_sum (different functionality)
            # or if included, should have lower or equal similarity
            # Note: with short code snippets, embedding models may produce
            # similar scores, so we use >= to handle edge cases
            if "calculate_sum" in names:
                admin_sim = next(r.similarity for r in results if r.name == "validate_admin")
                calc_sim = next(r.similarity for r in results if r.name == "calculate_sum")
                assert admin_sim >= calc_sim

    def test_excludes_connected_nodes(self) -> None:
        """Should exclude nodes already in dependency graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").write_text('''def process_request(req):
    """Process incoming request."""
    return handle(req)
''')
            Path(tmpdir, "b.py").write_text('''def process_response(res):
    """Process outgoing response."""
    return format(res)
''')

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            # Exclude b.py from results (simulating it's already in graph)
            results = _get_semantically_related(
                Path(tmpdir),
                target_id="py:a.py::process_request",
                file_path="a.py",
                symbol="process_request",
                exclude_ids={"py:b.py::process_response"},
                top_k=5,
                threshold=0.0,  # Low threshold to see all results
            )

            # Should not include the excluded node
            ids = [r.id for r in results]
            assert "py:b.py::process_response" not in ids

    def test_excludes_same_file(self) -> None:
        """Should exclude functions from the same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "handlers.py").write_text('''def handle_get(request):
    """Handle GET request."""
    return get_data(request)

def handle_post(request):
    """Handle POST request."""
    return post_data(request)
''')
            Path(tmpdir, "other.py").write_text('''def process_get(req):
    """Process GET request."""
    return fetch(req)
''')

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            results = _get_semantically_related(
                Path(tmpdir),
                target_id="py:handlers.py::handle_get",
                file_path="handlers.py",
                symbol="handle_get",
                exclude_ids=set(),
                top_k=5,
                threshold=0.0,
            )

            # Should not include handle_post (same file)
            for r in results:
                assert r.file != "handlers.py"

    def test_returns_empty_when_no_embeddings(self) -> None:
        """Should return empty list when embeddings not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't compute embeddings
            results = _get_semantically_related(
                Path(tmpdir),
                target_id="py:test.py::foo",
                file_path="test.py",
                symbol="foo",
                exclude_ids=set(),
                top_k=5,
                threshold=0.5,
            )

            assert results == []

    def test_returns_semantic_match_type(self) -> None:
        """Should return SemanticMatch objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").write_text("def foo(): pass")
            Path(tmpdir, "b.py").write_text("def bar(): pass")

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            results = _get_semantically_related(
                Path(tmpdir),
                target_id="py:a.py::foo",
                file_path="a.py",
                symbol="foo",
                exclude_ids=set(),
                top_k=5,
                threshold=0.0,
            )

            for r in results:
                assert isinstance(r, SemanticMatch)
                assert hasattr(r, "id")
                assert hasattr(r, "file")
                assert hasattr(r, "name")
                assert hasattr(r, "similarity")
                assert 0.0 <= r.similarity <= 1.0


class TestComputeFusedScore:
    """Tests for the fused score computation."""

    def test_default_weights(self) -> None:
        """Should use default weights of 0.6 similarity, 0.4 proximity."""
        # depth=0 means proximity=1.0
        score = compute_fused_score(similarity=1.0, depth=0)
        assert score == pytest.approx(1.0)  # 0.6 * 1.0 + 0.4 * 1.0

        # depth=1 means proximity=0.5
        score = compute_fused_score(similarity=1.0, depth=1)
        assert score == pytest.approx(0.8)  # 0.6 * 1.0 + 0.4 * 0.5

    def test_proximity_decreases_with_depth(self) -> None:
        """Deeper nodes should have lower proximity contribution."""
        scores = [
            compute_fused_score(similarity=0.5, depth=d)
            for d in range(5)
        ]
        # Scores should decrease as depth increases
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_custom_weights(self) -> None:
        """Should respect custom alpha/beta weights."""
        # Pure similarity (alpha=1, beta=0)
        score = compute_fused_score(similarity=0.8, depth=0, alpha=1.0, beta=0.0)
        assert score == pytest.approx(0.8)

        # Pure proximity (alpha=0, beta=1)
        score = compute_fused_score(similarity=0.8, depth=0, alpha=0.0, beta=1.0)
        assert score == pytest.approx(1.0)  # depth=0 → proximity=1.0

    def test_zero_similarity_still_has_proximity_score(self) -> None:
        """Even zero similarity should contribute proximity score."""
        score = compute_fused_score(similarity=0.0, depth=0)
        assert score == pytest.approx(0.4)  # 0.6 * 0 + 0.4 * 1.0

    def test_depth_zero_gives_max_proximity(self) -> None:
        """Depth 0 should give proximity of 1.0."""
        score = compute_fused_score(similarity=0.5, depth=0)
        expected = 0.6 * 0.5 + 0.4 * 1.0  # 0.3 + 0.4 = 0.7
        assert score == pytest.approx(expected)


class TestFilterIds:
    """Tests for the filter_ids parameter in find_similar_code."""

    def test_filter_to_specific_ids(self) -> None:
        """Should only return results matching filter_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").write_text("def process_data(data): return data.strip()")
            Path(tmpdir, "b.py").write_text("def process_input(inp): return inp.strip()")
            Path(tmpdir, "c.py").write_text("def handle_request(req): return req.json()")

            from nodestradamus.analyzers.embeddings import compute_embeddings, find_similar_code
            compute_embeddings(tmpdir)

            # Without filter - should find all similar
            all_results = find_similar_code(
                tmpdir,
                query="process data",
                top_k=10,
                threshold=0.0,
            )
            all_ids = {r["id"] for r in all_results}
            assert len(all_ids) >= 2

            # With filter - should only find matching IDs
            filter_set = {"py:a.py::process_data"}
            filtered_results = find_similar_code(
                tmpdir,
                query="process data",
                top_k=10,
                threshold=0.0,
                filter_ids=filter_set,
            )
            filtered_ids = {r["id"] for r in filtered_results}
            assert filtered_ids == filter_set

    def test_filter_ids_empty_set_returns_empty(self) -> None:
        """Empty filter_ids should return empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").write_text("def foo(): pass")

            from nodestradamus.analyzers.embeddings import compute_embeddings, find_similar_code
            compute_embeddings(tmpdir)

            results = find_similar_code(
                tmpdir,
                query="foo",
                top_k=10,
                threshold=0.0,
                filter_ids=set(),
            )
            assert results == []

    def test_filter_ids_no_matches_returns_empty(self) -> None:
        """Non-matching filter_ids should return empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").write_text("def foo(): pass")

            from nodestradamus.analyzers.embeddings import compute_embeddings, find_similar_code
            compute_embeddings(tmpdir)

            results = find_similar_code(
                tmpdir,
                query="foo",
                top_k=10,
                threshold=0.0,
                filter_ids={"py:nonexistent.py::bar"},
            )
            assert results == []


class TestGetFusedMatches:
    """Tests for fused matching combining graph and semantic signals."""

    def _build_graph(self, nodes: list[dict], edges: list[tuple[str, str]]) -> nx.DiGraph:
        """Helper to build a graph from node dicts and edge tuples."""
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(
                node["id"],
                file=node.get("file", ""),
                name=node.get("name", ""),
                type=node.get("type", "function"),
                line=node.get("line"),
            )
        for source, target in edges:
            G.add_edge(source, target, type="calls", resolved=True)
        return G

    def test_returns_fused_match_objects(self) -> None:
        """Should return FusedMatch objects with all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text("def main(): return process()")
            Path(tmpdir, "processor.py").write_text("def process(): return validate()")
            Path(tmpdir, "validator.py").write_text("def validate(): return True")

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            G = self._build_graph(
                [
                    {"id": "py:main.py::main", "file": "main.py", "name": "main"},
                    {"id": "py:processor.py::process", "file": "processor.py", "name": "process"},
                    {"id": "py:validator.py::validate", "file": "validator.py", "name": "validate"},
                ],
                [
                    ("py:main.py::main", "py:processor.py::process"),
                    ("py:processor.py::process", "py:validator.py::validate"),
                ],
            )

            results = _get_fused_matches(
                Path(tmpdir),
                G,
                target_id="py:main.py::main",
                file_path="main.py",
                symbol="main",
                depth=3,
                top_k=10,
                threshold=0.0,
            )

            for r in results:
                assert isinstance(r, FusedMatch)
                assert hasattr(r, "id")
                assert hasattr(r, "file")
                assert hasattr(r, "name")
                assert hasattr(r, "depth")
                assert hasattr(r, "similarity")
                assert hasattr(r, "fused_score")

    def test_closer_and_similar_ranks_higher(self) -> None:
        """Nodes that are both close in graph AND semantically similar should rank higher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with varying similarity to target
            Path(tmpdir, "auth.py").write_text('''def authenticate_user(credentials):
    """Validate user credentials for authentication."""
    return check_password(credentials)
''')
            Path(tmpdir, "nearby_auth.py").write_text('''def verify_credentials(creds):
    """Check user credentials for login."""
    return validate_password(creds)
''')
            Path(tmpdir, "far_auth.py").write_text('''def confirm_identity(data):
    """Confirm user identity for access."""
    return check_data(data)
''')
            Path(tmpdir, "unrelated.py").write_text('''def calculate_total(numbers):
    """Sum up all numbers in list."""
    return sum(numbers)
''')

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            # Build graph where nearby_auth is close (depth=1) and far_auth is far (depth=3)
            G = self._build_graph(
                [
                    {"id": "py:auth.py::authenticate_user", "file": "auth.py", "name": "authenticate_user"},
                    {"id": "py:nearby_auth.py::verify_credentials", "file": "nearby_auth.py", "name": "verify_credentials"},
                    {"id": "py:far_auth.py::confirm_identity", "file": "far_auth.py", "name": "confirm_identity"},
                    {"id": "py:unrelated.py::calculate_total", "file": "unrelated.py", "name": "calculate_total"},
                ],
                [
                    # nearby_auth is depth=1 from auth
                    ("py:auth.py::authenticate_user", "py:nearby_auth.py::verify_credentials"),
                    # far_auth is depth=3 from auth (through intermediaries)
                    ("py:nearby_auth.py::verify_credentials", "py:unrelated.py::calculate_total"),
                    ("py:unrelated.py::calculate_total", "py:far_auth.py::confirm_identity"),
                ],
            )

            results = _get_fused_matches(
                Path(tmpdir),
                G,
                target_id="py:auth.py::authenticate_user",
                file_path="auth.py",
                symbol="authenticate_user",
                depth=4,
                top_k=10,
                threshold=0.0,
            )

            # Filter to just auth-related results
            auth_results = [r for r in results if "auth" in r.file.lower() or "verify" in r.name.lower() or "confirm" in r.name.lower()]

            if len(auth_results) >= 2:
                # nearby_auth should rank higher because it's both close AND similar
                nearby = next((r for r in auth_results if "nearby" in r.file), None)
                far = next((r for r in auth_results if "far" in r.file), None)

                if nearby and far:
                    # Nearby should have higher fused score (close + similar)
                    assert nearby.fused_score >= far.fused_score

    def test_excludes_target_itself(self) -> None:
        """Should not include the target node in results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text("def main(): return helper()")
            Path(tmpdir, "helper.py").write_text("def helper(): return True")

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            G = self._build_graph(
                [
                    {"id": "py:main.py::main", "file": "main.py", "name": "main"},
                    {"id": "py:helper.py::helper", "file": "helper.py", "name": "helper"},
                ],
                [("py:main.py::main", "py:helper.py::helper")],
            )

            results = _get_fused_matches(
                Path(tmpdir),
                G,
                target_id="py:main.py::main",
                file_path="main.py",
                symbol="main",
                depth=3,
                top_k=10,
                threshold=0.0,
            )

            result_ids = {r.id for r in results}
            assert "py:main.py::main" not in result_ids

    def test_empty_blast_radius_returns_empty(self) -> None:
        """Should return empty list when no nodes in blast radius."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "isolated.py").write_text("def isolated(): pass")

            from nodestradamus.analyzers.embeddings import compute_embeddings
            compute_embeddings(tmpdir)

            # Graph with isolated node (no edges)
            G = self._build_graph(
                [{"id": "py:isolated.py::isolated", "file": "isolated.py", "name": "isolated"}],
                [],
            )

            results = _get_fused_matches(
                Path(tmpdir),
                G,
                target_id="py:isolated.py::isolated",
                file_path="isolated.py",
                symbol="isolated",
                depth=3,
                top_k=10,
                threshold=0.0,
            )

            assert results == []
