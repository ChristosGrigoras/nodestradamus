"""Tests for consolidated MCP tools.

Verifies that the consolidated tools work correctly with different
algorithm/mode parameters.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from nodestradamus.mcp.tools.handlers import (
    handle_analyze_graph,
    handle_analyze_strings,
    handle_codebase_health,
    handle_semantic_analysis,
)

# Use sample_python fixture for testing
FIXTURE_PATH = str(Path(__file__).parent / "fixtures" / "sample_python")


def _normalize_pagerank_results(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize PageRank results for order-independent comparison.

    Items with equal importance scores may appear in any order,
    so we sort by (importance DESC, node ASC) for deterministic comparison.
    """
    if "most_important_code" in data:
        data["most_important_code"] = sorted(
            data["most_important_code"],
            key=lambda x: (-x.get("importance", 0), x.get("node", "")),
        )
    return data


class TestAnalyzeGraphAlgorithms:
    """Test that analyze_graph works with all algorithm options."""

    @pytest.mark.asyncio
    async def test_pagerank_returns_results(self) -> None:
        """analyze_graph with algorithm='pagerank' returns importance ranking."""
        result = await handle_analyze_graph(
            {
                "repo_path": FIXTURE_PATH,
                "algorithm": "pagerank",
                "top_n": 10,
            }
        )
        parsed = json.loads(result)
        assert "most_important_code" in parsed
        assert "total_nodes_analyzed" in parsed
        assert len(parsed["most_important_code"]) <= 10

    @pytest.mark.asyncio
    async def test_betweenness_returns_results(self) -> None:
        """analyze_graph with algorithm='betweenness' returns bottleneck ranking."""
        result = await handle_analyze_graph(
            {
                "repo_path": FIXTURE_PATH,
                "algorithm": "betweenness",
                "top_n": 10,
            }
        )
        parsed = json.loads(result)
        assert "bottlenecks" in parsed
        assert "total_nodes_analyzed" in parsed
        assert "explanation" in parsed

    @pytest.mark.asyncio
    async def test_communities_returns_results(self) -> None:
        """analyze_graph with algorithm='communities' returns module clusters."""
        result = await handle_analyze_graph(
            {
                "repo_path": FIXTURE_PATH,
                "algorithm": "communities",
            }
        )
        parsed = json.loads(result)
        assert "modules" in parsed
        assert "total_modules" in parsed
        assert "total_nodes" in parsed

    @pytest.mark.asyncio
    async def test_cycles_returns_results(self) -> None:
        """analyze_graph with algorithm='cycles' returns circular dependencies."""
        result = await handle_analyze_graph(
            {
                "repo_path": FIXTURE_PATH,
                "algorithm": "cycles",
                "max_cycles": 10,
            }
        )
        parsed = json.loads(result)
        assert "circular_dependencies" in parsed
        assert "total_cycles_found" in parsed

    @pytest.mark.asyncio
    async def test_path_returns_results(self) -> None:
        """analyze_graph with algorithm='path' returns shortest path."""
        # Use node IDs that exist in the sample_python fixture
        source = "py:main.py::main"
        target = "py:utils.py::helper_function"

        result = await handle_analyze_graph(
            {
                "repo_path": FIXTURE_PATH,
                "algorithm": "path",
                "source": source,
                "target": target,
            }
        )
        parsed = json.loads(result)
        # May have path or error if no path exists
        assert "path" in parsed or "error" in parsed


class TestAnalyzeStringsModes:
    """Test that analyze_strings works with all mode options."""

    @pytest.mark.asyncio
    async def test_refs_returns_results(self) -> None:
        """analyze_strings with mode='refs' returns significant strings."""
        result = await handle_analyze_strings(
            {
                "repo_path": FIXTURE_PATH,
                "mode": "refs",
                "min_files": 1,
                "top_n": 10,
            }
        )
        parsed = json.loads(result)
        assert "significant_strings" in parsed or "metadata" in parsed

    @pytest.mark.asyncio
    async def test_usages_returns_results(self) -> None:
        """analyze_strings with mode='usages' returns string usage locations."""
        result = await handle_analyze_strings(
            {
                "repo_path": FIXTURE_PATH,
                "mode": "usages",
                "target_string": "helper_function",
            }
        )
        parsed = json.loads(result)
        # Result structure depends on implementation
        assert parsed is not None

    @pytest.mark.asyncio
    async def test_filter_returns_results(self) -> None:
        """analyze_strings with mode='filter' filters noise from strings."""
        test_strings = [
            {"value": "string", "reference_count": 10},
            {"value": "my_config", "reference_count": 5},
            {"value": "ab", "reference_count": 3},  # Too short
        ]

        result = await handle_analyze_strings(
            {
                "repo_path": FIXTURE_PATH,
                "mode": "filter",
                "strings": test_strings,
                "exclude_types": True,
                "exclude_imports": True,
                "exclude_css": True,
                "min_length": 3,
            }
        )
        parsed = json.loads(result)
        assert "filtered_strings" in parsed
        assert "original_count" in parsed
        assert "filtered_count" in parsed
        assert "removed_breakdown" in parsed


class TestSemanticAnalysisModes:
    """Test that semantic_analysis works with all mode options."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        """semantic_analysis with mode='search' returns search results."""
        result = await handle_semantic_analysis(
            {
                "repo_path": FIXTURE_PATH,
                "mode": "search",
                "query": "helper function",
                "top_k": 5,
                "threshold": 0.3,
            }
        )
        parsed = json.loads(result)
        assert "query" in parsed
        assert "results" in parsed
        assert "count" in parsed

    @pytest.mark.asyncio
    async def test_similar_returns_results(self) -> None:
        """semantic_analysis with mode='similar' returns similar code."""
        result = await handle_semantic_analysis(
            {
                "repo_path": FIXTURE_PATH,
                "mode": "similar",
                "query": "def main",
                "top_k": 5,
                "threshold": 0.3,
            }
        )
        parsed = json.loads(result)
        assert "similar_code" in parsed
        assert "count" in parsed
        assert "query_type" in parsed

    @pytest.mark.asyncio
    async def test_duplicates_returns_results(self) -> None:
        """semantic_analysis with mode='duplicates' returns duplicate code."""
        result = await handle_semantic_analysis(
            {
                "repo_path": FIXTURE_PATH,
                "mode": "duplicates",
                "threshold": 0.9,
                "max_pairs": 10,
            }
        )
        parsed = json.loads(result)
        assert "duplicates" in parsed
        assert "count" in parsed
        assert "threshold" in parsed


class TestAnalyzeGraphValidation:
    """Test validation for analyze_graph consolidated tool."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self) -> None:
        """Raises error when repo_path is missing."""
        with pytest.raises(ValueError, match="repo_path is required"):
            await handle_analyze_graph(
                {
                    "algorithm": "pagerank",
                }
            )

    @pytest.mark.asyncio
    async def test_missing_algorithm(self) -> None:
        """Raises error when algorithm is missing."""
        with pytest.raises(ValueError, match="algorithm is required"):
            await handle_analyze_graph(
                {
                    "repo_path": FIXTURE_PATH,
                }
            )

    @pytest.mark.asyncio
    async def test_invalid_algorithm(self) -> None:
        """Raises error for unknown algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            await handle_analyze_graph(
                {
                    "repo_path": FIXTURE_PATH,
                    "algorithm": "invalid",
                }
            )

    @pytest.mark.asyncio
    async def test_path_missing_source(self) -> None:
        """Raises error when source is missing for path algorithm."""
        with pytest.raises(ValueError, match="source is required"):
            await handle_analyze_graph(
                {
                    "repo_path": FIXTURE_PATH,
                    "algorithm": "path",
                    "target": "some_target",
                }
            )

    @pytest.mark.asyncio
    async def test_path_missing_target(self) -> None:
        """Raises error when target is missing for path algorithm."""
        with pytest.raises(ValueError, match="target is required"):
            await handle_analyze_graph(
                {
                    "repo_path": FIXTURE_PATH,
                    "algorithm": "path",
                    "source": "some_source",
                }
            )


class TestAnalyzeStringsValidation:
    """Test validation for analyze_strings consolidated tool."""

    @pytest.mark.asyncio
    async def test_missing_mode(self) -> None:
        """Raises error when mode is missing."""
        with pytest.raises(ValueError, match="mode is required"):
            await handle_analyze_strings(
                {
                    "repo_path": FIXTURE_PATH,
                }
            )

    @pytest.mark.asyncio
    async def test_invalid_mode(self) -> None:
        """Raises error for unknown mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            await handle_analyze_strings(
                {
                    "repo_path": FIXTURE_PATH,
                    "mode": "invalid",
                }
            )

    @pytest.mark.asyncio
    async def test_usages_missing_target(self) -> None:
        """Raises error when target_string is missing for usages mode."""
        with pytest.raises(ValueError, match="target_string is required"):
            await handle_analyze_strings(
                {
                    "repo_path": FIXTURE_PATH,
                    "mode": "usages",
                }
            )

    @pytest.mark.asyncio
    async def test_filter_missing_strings(self) -> None:
        """Raises error when strings is missing for filter mode."""
        with pytest.raises(ValueError, match="strings is required"):
            await handle_analyze_strings(
                {
                    "repo_path": FIXTURE_PATH,
                    "mode": "filter",
                }
            )


class TestSemanticAnalysisValidation:
    """Test validation for semantic_analysis consolidated tool."""

    @pytest.mark.asyncio
    async def test_missing_mode(self) -> None:
        """Raises error when mode is missing."""
        with pytest.raises(ValueError, match="mode is required"):
            await handle_semantic_analysis(
                {
                    "repo_path": FIXTURE_PATH,
                }
            )

    @pytest.mark.asyncio
    async def test_invalid_mode(self) -> None:
        """Raises error for unknown mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            await handle_semantic_analysis(
                {
                    "repo_path": FIXTURE_PATH,
                    "mode": "invalid",
                }
            )

    @pytest.mark.asyncio
    async def test_search_missing_query(self) -> None:
        """Raises error when query is missing for search mode."""
        with pytest.raises(ValueError, match="query is required"):
            await handle_semantic_analysis(
                {
                    "repo_path": FIXTURE_PATH,
                    "mode": "search",
                }
            )

    @pytest.mark.asyncio
    async def test_similar_missing_input(self) -> None:
        """Raises error when no input is provided for similar mode."""
        with pytest.raises(ValueError, match="query, file_path, or symbol is required"):
            await handle_semantic_analysis(
                {
                    "repo_path": FIXTURE_PATH,
                    "mode": "similar",
                }
            )


class TestCodebaseHealthWorkflow:
    """Test the codebase_health workflow tool."""

    @pytest.mark.asyncio
    async def test_returns_unified_report(self) -> None:
        """codebase_health returns a unified report with summary and findings."""
        result = await handle_codebase_health(
            {
                "repo_path": FIXTURE_PATH,
                "checks": ["dead_code", "cycles", "bottlenecks"],
            }
        )
        parsed = json.loads(result)
        assert "summary" in parsed
        assert "findings" in parsed
        assert "health" in parsed["summary"]
        assert "checks_run" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_runs_all_checks_by_default(self) -> None:
        """codebase_health runs all checks when none specified."""
        result = await handle_codebase_health(
            {
                "repo_path": FIXTURE_PATH,
            }
        )
        parsed = json.loads(result)
        assert set(parsed["summary"]["checks_run"]) == {
            "dead_code",
            "duplicates",
            "cycles",
            "bottlenecks",
            "docs",
        }

    @pytest.mark.asyncio
    async def test_runs_single_check(self) -> None:
        """codebase_health can run a single check."""
        result = await handle_codebase_health(
            {
                "repo_path": FIXTURE_PATH,
                "checks": ["dead_code"],
            }
        )
        parsed = json.loads(result)
        assert "dead_code" in parsed["findings"]
        assert "dead_code_count" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_respects_max_items(self) -> None:
        """codebase_health respects max_items limit."""
        result = await handle_codebase_health(
            {
                "repo_path": FIXTURE_PATH,
                "checks": ["bottlenecks"],
                "max_items": 5,
            }
        )
        parsed = json.loads(result)
        assert len(parsed["findings"]["bottlenecks"]) <= 5

    @pytest.mark.asyncio
    async def test_missing_repo_path(self) -> None:
        """Raises error when repo_path is missing."""
        with pytest.raises(ValueError, match="repo_path is required"):
            await handle_codebase_health({})
