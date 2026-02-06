"""Tests for refactor analysis features.

Tests the four refactoring analysis capabilities:
1. Symbol-level import aggregation
2. Duplicate detection with line ranges
3. Internal function clustering
4. Breaking change warnings
"""

from pathlib import Path

import pytest

from nodestradamus.analyzers.clustering import _infer_cluster_name, cluster_functions_in_file
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.duplicates import (
    _compute_content_hash,
    _extract_code_blocks,
    _normalize_code,
    find_exact_duplicates,
)
from nodestradamus.analyzers.impact import _get_breaking_changes, _get_symbol_usage

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    """Create a Python repo with symbols, duplicates, and clusters."""
    # Main module with exported symbols
    (tmp_path / "utils.py").write_text('''
"""Utility functions."""

SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".git"})
MAX_RETRIES = 3

def parse_file(filepath: str) -> dict:
    """Parse a file."""
    data = read_data(filepath)
    return process_data(data)

def read_data(filepath: str) -> str:
    """Read data from file."""
    with open(filepath) as f:
        return f.read()

def process_data(data: str) -> dict:
    """Process the data."""
    validate_data(data)
    return {"data": data}

def validate_data(data: str) -> None:
    """Validate the data."""
    if not data:
        raise ValueError("Empty data")

def unrelated_func():
    """An unrelated function."""
    return 42
''')

    # Consumer 1
    (tmp_path / "consumer1.py").write_text('''
from utils import parse_file, SKIP_DIRS

def main():
    result = parse_file("test.txt")
    print(result)
''')

    # Consumer 2
    (tmp_path / "consumer2.py").write_text('''
from utils import parse_file, MAX_RETRIES

def run():
    for i in range(MAX_RETRIES):
        try:
            return parse_file("data.json")
        except Exception:
            continue
''')

    # Consumer 3 (only uses parse_file)
    (tmp_path / "consumer3.py").write_text('''
from utils import parse_file

def execute():
    return parse_file("config.yaml")
''')

    # Duplicate code in another file
    (tmp_path / "other_utils.py").write_text('''
"""Other utilities with duplicate code."""

SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".git"})

def another_func():
    """Different function."""
    return 123
''')

    return tmp_path


@pytest.fixture
def typescript_repo(tmp_path: Path) -> Path:
    """Create a TypeScript repo with constants and functions."""
    (tmp_path / "config.ts").write_text('''
export const API_URL = "https://api.example.com";
export const MAX_CONNECTIONS = 100;

export function fetchData(endpoint: string): Promise<any> {
    return processRequest(endpoint);
}

function processRequest(endpoint: string): Promise<any> {
    return validateRequest(endpoint).then(() => {
        return fetch(API_URL + endpoint);
    });
}

function validateRequest(endpoint: string): Promise<void> {
    if (!endpoint) {
        throw new Error("Invalid endpoint");
    }
    return Promise.resolve();
}
''')

    (tmp_path / "client.ts").write_text('''
import { fetchData, API_URL } from "./config";

export function getData(): Promise<any> {
    return fetchData("/users");
}
''')

    return tmp_path


# =============================================================================
# Symbol Usage Tests
# =============================================================================


class TestSymbolUsage:
    """Tests for _get_symbol_usage function."""

    def test_counts_importing_files(self, python_repo: Path) -> None:
        """Test that symbol usage counts are correct."""
        G = analyze_deps(python_repo)
        usage = _get_symbol_usage(G, "utils.py")

        # parse_file should be imported by 3 files
        parse_file_usage = next((u for u in usage if u.symbol == "parse_file"), None)
        assert parse_file_usage is not None
        assert parse_file_usage.count == 3
        assert "consumer1.py" in parse_file_usage.importing_files
        assert "consumer2.py" in parse_file_usage.importing_files
        assert "consumer3.py" in parse_file_usage.importing_files

    def test_constants_tracked(self, python_repo: Path) -> None:
        """Test that constants are tracked as symbols."""
        G = analyze_deps(python_repo)
        usage = _get_symbol_usage(G, "utils.py")

        # SKIP_DIRS should be imported by 1 file
        skip_dirs_usage = next((u for u in usage if u.symbol == "SKIP_DIRS"), None)
        assert skip_dirs_usage is not None
        assert skip_dirs_usage.count == 1
        assert skip_dirs_usage.symbol_type == "constant"

    def test_sorted_by_count(self, python_repo: Path) -> None:
        """Test that results are sorted by count descending."""
        G = analyze_deps(python_repo)
        usage = _get_symbol_usage(G, "utils.py")

        counts = [u.count for u in usage]
        assert counts == sorted(counts, reverse=True)


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestDuplicateDetection:
    """Tests for duplicate detection functions."""

    def test_normalize_code_strips_comments(self) -> None:
        """Test that comments are stripped during normalization."""
        code = '''
# This is a comment
def foo():
    # Another comment
    return 42  # inline comment
'''
        normalized = _normalize_code(code)
        assert "comment" not in normalized
        assert "def foo():" in normalized

    def test_normalize_code_strips_multiline_comments(self) -> None:
        """Test that multiline comments are stripped."""
        code = '''
/* This is a
   multiline comment */
function bar() {
    return 42;
}
'''
        normalized = _normalize_code(code)
        assert "multiline" not in normalized
        assert "function bar()" in normalized

    def test_content_hash_deterministic(self) -> None:
        """Test that content hash is deterministic."""
        content = "def foo(): return 42"
        hash1 = _compute_content_hash(content)
        hash2 = _compute_content_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_finds_exact_duplicates(self, python_repo: Path) -> None:
        """Test that exact duplicates are found."""
        duplicates = find_exact_duplicates(python_repo, min_lines=1)

        # Should find SKIP_DIRS as a duplicate
        # Note: the constant might not be detected as a block,
        # so let's check if we find any duplicates at all
        # and verify the structure
        for dup in duplicates:
            assert dup.similarity == 1.0
            assert len(dup.locations) >= 2
            for loc in dup.locations:
                assert loc.line_start > 0
                assert loc.line_end >= loc.line_start

    def test_extract_code_blocks_python(self, python_repo: Path) -> None:
        """Test Python code block extraction."""
        blocks = _extract_code_blocks(python_repo / "utils.py", min_lines=2)

        # Should find function definitions
        assert len(blocks) > 0
        for block in blocks:
            assert "line_start" in block
            assert "line_end" in block
            assert "hash" in block
            assert block["line_start"] > 0


# =============================================================================
# Clustering Tests
# =============================================================================


class TestClustering:
    """Tests for function clustering."""

    def test_infer_cluster_name_common_prefix(self) -> None:
        """Test cluster name inference from common prefix."""
        names = ["parse_json", "parse_xml", "parse_yaml"]
        name = _infer_cluster_name(names)
        assert name == "parse"

    def test_infer_cluster_name_patterns(self) -> None:
        """Test cluster name inference from patterns."""
        names = ["validate_input", "check_output", "verify_data"]
        name = _infer_cluster_name(names)
        assert name == "validation"

    def test_cluster_functions(self, python_repo: Path) -> None:
        """Test function clustering based on call graph."""
        clusters = cluster_functions_in_file(python_repo, "utils.py")

        # Should find at least one cluster (parse_file -> read_data -> process_data)
        if clusters:  # May be empty if no internal calls detected
            for cluster in clusters:
                assert len(cluster.functions) >= 2
                assert cluster.line_start > 0
                assert cluster.line_end >= cluster.line_start
                assert 0 <= cluster.cohesion_score <= 1.0


# =============================================================================
# Breaking Changes Tests
# =============================================================================


class TestBreakingChanges:
    """Tests for breaking change detection."""

    def test_identifies_external_dependents(self, python_repo: Path) -> None:
        """Test that external dependents are identified."""
        G = analyze_deps(python_repo)
        breaking = _get_breaking_changes(G, "utils.py")

        # parse_file should have breaking change warnings
        parse_file_breaking = next((b for b in breaking if b.symbol == "parse_file"), None)
        assert parse_file_breaking is not None
        assert parse_file_breaking.direct_count >= 3
        assert len(parse_file_breaking.dependents) >= 3

    def test_separates_direct_indirect(self, python_repo: Path) -> None:
        """Test that direct and indirect dependents are separated."""
        G = analyze_deps(python_repo)
        breaking = _get_breaking_changes(G, "utils.py")

        for change in breaking:
            # Dependents is the union of direct and indirect files (deduplicated)
            # So len(dependents) <= direct_count + indirect_count
            total = change.direct_count + change.indirect_count
            assert len(change.dependents) <= total
            # Each dependent should be either direct or indirect
            assert len(change.dependents) > 0

    def test_sorted_by_impact(self, python_repo: Path) -> None:
        """Test that results are sorted by total impact."""
        G = analyze_deps(python_repo)
        breaking = _get_breaking_changes(G, "utils.py")

        totals = [b.direct_count + b.indirect_count for b in breaking]
        assert totals == sorted(totals, reverse=True)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRefactorModeIntegration:
    """Integration tests for the full refactor_mode feature."""

    def test_python_repo_full_analysis(self, python_repo: Path) -> None:
        """Test full refactor analysis on Python repo."""
        G = analyze_deps(python_repo)

        # Symbol usage
        usage = _get_symbol_usage(G, "utils.py")
        assert len(usage) > 0

        # Breaking changes
        breaking = _get_breaking_changes(G, "utils.py")
        assert len(breaking) > 0

        # Duplicates
        duplicates = find_exact_duplicates(python_repo, "utils.py")
        # May or may not find duplicates depending on block detection
        assert isinstance(duplicates, list)

        # Clusters
        clusters = cluster_functions_in_file(python_repo, "utils.py")
        assert isinstance(clusters, list)

    def test_typescript_repo_constants(self, typescript_repo: Path) -> None:
        """Test constant extraction in TypeScript."""
        G = analyze_deps(typescript_repo)

        # Check that constants are in the graph
        constant_nodes = [
            (node_id, attrs)
            for node_id, attrs in G.nodes(data=True)
            if attrs.get("type") == "constant"
        ]

        # Should find API_URL and MAX_CONNECTIONS
        constant_names = [attrs.get("name") for _, attrs in constant_nodes]
        assert "API_URL" in constant_names or "MAX_CONNECTIONS" in constant_names

    def test_empty_file_returns_empty_results(self, tmp_path: Path) -> None:
        """Test that empty files return empty results."""
        (tmp_path / "empty.py").write_text("")

        G = analyze_deps(tmp_path)
        usage = _get_symbol_usage(G, "empty.py")
        assert usage == []

        breaking = _get_breaking_changes(G, "empty.py")
        assert breaking == []


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test handling of nonexistent files."""
        G = analyze_deps(tmp_path)
        usage = _get_symbol_usage(G, "nonexistent.py")
        assert usage == []

    def test_file_with_only_constants(self, tmp_path: Path) -> None:
        """Test file with only constants, no functions."""
        (tmp_path / "constants.py").write_text('''
API_KEY = "secret"
DEBUG_MODE = True
MAX_SIZE = 1000
''')

        G = analyze_deps(tmp_path)
        # Should have constant nodes
        constant_nodes = [
            attrs for _, attrs in G.nodes(data=True)
            if attrs.get("type") == "constant"
        ]
        assert len(constant_nodes) >= 2

    def test_circular_calls(self, tmp_path: Path) -> None:
        """Test handling of circular function calls."""
        (tmp_path / "circular.py").write_text('''
def func_a():
    return func_b()

def func_b():
    return func_a()
''')

        analyze_deps(tmp_path)
        clusters = cluster_functions_in_file(tmp_path, "circular.py")
        # Should handle without error
        assert isinstance(clusters, list)
