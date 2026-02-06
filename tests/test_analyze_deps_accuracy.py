"""Tests for analyze_deps accuracy using synthetic repo with known counts.

This test suite validates that analyze_deps correctly identifies ALL definitions
in a codebase, not just "at least some".

Synthetic Repo Structure:
========================

Python (tests/fixtures/synthetic_repo/python/):
- models.py: 3 classes (User, Product, Order)
- utils.py: 5 functions (format_date, parse_json, slugify, hash_password, validate_email)
- services/auth.py: 1 class (AuthService) + 3 functions
- services/users.py: 1 class (UserService) + 2 functions

TypeScript (tests/fixtures/synthetic_repo/typescript/):
- index.ts: 2 functions (main, bootstrap)
- lib/api.ts: 1 class (ApiClient) + 3 functions
- lib/utils.ts: 4 functions
- components/Button.tsx: 1 function
- components/Card.tsx: 2 functions
"""

from pathlib import Path

import pytest

from nodestradamus.analyzers.deps import analyze_deps

SYNTHETIC_REPO = Path(__file__).parent / "fixtures" / "synthetic_repo"
PYTHON_DIR = SYNTHETIC_REPO / "python"
TYPESCRIPT_DIR = SYNTHETIC_REPO / "typescript"


class TestPythonAccuracy:
    """Validate exact counts for Python analysis."""

    @pytest.fixture
    def python_graph(self):
        """Analyze Python portion of synthetic repo."""
        return analyze_deps(PYTHON_DIR, languages=["python"])

    def test_finds_all_classes(self, python_graph) -> None:
        """Should find exactly 5 classes."""
        classes = [
            (node, attrs)
            for node, attrs in python_graph.nodes(data=True)
            if attrs.get("type") == "class"
        ]

        class_names = {attrs["name"] for _, attrs in classes}
        expected_classes = {"User", "Product", "Order", "AuthService", "UserService"}

        # First, show what we found for debugging
        print(f"\nFound {len(classes)} classes: {class_names}")

        assert class_names == expected_classes, (
            f"Expected classes {expected_classes}, got {class_names}. "
            f"Missing: {expected_classes - class_names}, "
            f"Extra: {class_names - expected_classes}"
        )

    def test_finds_all_functions_including_methods(self, python_graph) -> None:
        """Should find all functions AND methods (analyzer doesn't distinguish).

        NOTE: Current analyzer behavior is that methods are stored as type="function"
        without a way to distinguish them from standalone functions. This test
        documents that behavior.
        """
        functions = [
            (node, attrs)
            for node, attrs in python_graph.nodes(data=True)
            if attrs.get("type") == "function"
        ]

        func_names = {attrs["name"] for _, attrs in functions}

        # Standalone functions
        expected_standalone = {
            "format_date", "parse_json", "slugify", "hash_password", "validate_email",
            "create_token", "verify_token", "refresh_token",
            "get_user_by_email", "count_users",
        }
        # Methods (also reported as "function" type)
        expected_methods = {
            "__init__", "save", "delete",  # From User/Product/Order (deduplicated)
            "login", "logout",  # From AuthService
            "get_user", "create_user", "list_users",  # From UserService
        }
        expected_all = expected_standalone | expected_methods

        print(f"\nFound {len(functions)} functions: {func_names}")

        assert func_names == expected_all, (
            f"Missing: {expected_all - func_names}, "
            f"Extra: {func_names - expected_all}"
        )

    def test_method_deduplication_issue(self, python_graph) -> None:
        """Document: methods with same name from different classes are deduplicated.

        User, Product, and Order all have __init__, save, delete methods.
        But the analyzer only reports each once (deduplicated by name within file).
        """
        functions = [
            (node, attrs)
            for node, attrs in python_graph.nodes(data=True)
            if attrs.get("type") == "function" and attrs.get("file") == "models.py"
        ]

        func_names = [attrs["name"] for _, attrs in functions]
        print(f"\nFunctions in models.py: {func_names}")

        # Each class has 3 methods, so we'd expect 9 methods if not deduplicated
        # But they're deduplicated by name, so we only get 3
        # This is a LIMITATION of the current analyzer
        assert len(func_names) == 3, (
            f"Expected 3 (deduplicated) but got {len(func_names)}. "
            "If this changes, the analyzer may have been fixed to track methods per-class."
        )

    def test_total_node_count(self, python_graph) -> None:
        """Verify total node count matches expectations."""
        total = python_graph.number_of_nodes()

        # Show breakdown by type
        by_type: dict[str, int] = {}
        for _, attrs in python_graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            by_type[node_type] = by_type.get(node_type, 0) + 1

        print(f"\nTotal nodes: {total}")
        print(f"By type: {by_type}")

        # We expect at minimum: 5 classes + 10 functions = 15
        # Methods may or may not be counted separately
        assert total >= 15, f"Expected at least 15 nodes, got {total}"

    def test_edges_are_resolved(self, python_graph) -> None:
        """Verify that internal function calls are resolved."""
        edges = list(python_graph.edges(data=True))

        resolved = [(u, v) for u, v, d in edges if d.get("resolved")]
        unresolved = [(u, v) for u, v, d in edges if not d.get("resolved")]

        print(f"\nTotal edges: {len(edges)}")
        print(f"Resolved: {len(resolved)}")
        print(f"Unresolved: {len(unresolved)}")

        # We expect some resolved edges (internal calls)
        assert len(resolved) > 0, "Expected some resolved edges for internal calls"


class TestTypeScriptAccuracy:
    """Validate exact counts for TypeScript analysis."""

    @pytest.fixture
    def ts_graph(self):
        """Analyze TypeScript portion of synthetic repo."""
        return analyze_deps(TYPESCRIPT_DIR, languages=["typescript"])

    def test_finds_all_classes(self, ts_graph) -> None:
        """Should find all type definitions (classes, interfaces, type aliases)."""
        classes = [
            (node, attrs)
            for node, attrs in ts_graph.nodes(data=True)
            if attrs.get("type") == "class"
        ]

        class_names = {attrs["name"] for _, attrs in classes}
        # Note: TypeScript interfaces and type aliases are intentionally
        # classified as "class" type since they're type definitions
        expected_classes = {"ApiClient", "ButtonProps", "CardProps"}

        print(f"\nFound {len(classes)} classes: {class_names}")

        assert class_names == expected_classes, (
            f"Expected classes {expected_classes}, got {class_names}. "
            f"Missing: {expected_classes - class_names}, "
            f"Extra: {class_names - expected_classes}"
        )

    def test_finds_all_functions(self, ts_graph) -> None:
        """Should find exactly 12 functions."""
        functions = [
            (node, attrs)
            for node, attrs in ts_graph.nodes(data=True)
            if attrs.get("type") == "function"
        ]

        func_names = {attrs["name"] for _, attrs in functions}
        expected_functions = {
            # index.ts
            "main",
            "bootstrap",
            # lib/api.ts
            "initApi",
            "fetchJson",
            "handleError",
            # lib/utils.ts
            "formatDate",
            "slugify",
            "debounce",
            "clamp",
            # components/Button.tsx
            "Button",
            # components/Card.tsx
            "Card",
            "CardHeader",
        }

        print(f"\nFound {len(functions)} functions: {func_names}")

        assert func_names == expected_functions, (
            f"Expected functions {expected_functions}, got {func_names}. "
            f"Missing: {expected_functions - func_names}, "
            f"Extra: {func_names - expected_functions}"
        )

    def test_total_node_count(self, ts_graph) -> None:
        """Verify total node count matches expectations."""
        total = ts_graph.number_of_nodes()

        by_type: dict[str, int] = {}
        for _, attrs in ts_graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            by_type[node_type] = by_type.get(node_type, 0) + 1

        print(f"\nTotal nodes: {total}")
        print(f"By type: {by_type}")

        # We expect at minimum: 1 class + 12 functions = 13
        assert total >= 13, f"Expected at least 13 nodes, got {total}"


class TestCombinedAccuracy:
    """Test analyzing both languages together."""

    def test_combined_analysis(self) -> None:
        """Analyze entire synthetic repo with auto-detection."""
        graph = analyze_deps(SYNTHETIC_REPO)

        total = graph.number_of_nodes()
        by_type: dict[str, int] = {}
        for _, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            by_type[node_type] = by_type.get(node_type, 0) + 1

        print("\nCombined analysis:")
        print(f"Total nodes: {total}")
        print(f"By type: {by_type}")

        # Minimum expected: 5 py classes + 1 ts class + 10 py functions + 12 ts functions = 28
        assert total >= 28, f"Expected at least 28 nodes, got {total}"


class TestDiagnostics:
    """Diagnostic tests to understand what the analyzer finds."""

    def test_list_all_python_nodes(self) -> None:
        """Print all nodes found in Python analysis for debugging."""
        graph = analyze_deps(PYTHON_DIR, languages=["python"])

        print("\n=== All Python Nodes ===")
        for node, attrs in sorted(graph.nodes(data=True)):
            print(f"  {node}")
            print(f"    type: {attrs.get('type')}")
            print(f"    name: {attrs.get('name')}")
            print(f"    file: {attrs.get('file')}")
            print()

    def test_list_all_typescript_nodes(self) -> None:
        """Print all nodes found in TypeScript analysis for debugging."""
        graph = analyze_deps(TYPESCRIPT_DIR, languages=["typescript"])

        print("\n=== All TypeScript Nodes ===")
        for node, attrs in sorted(graph.nodes(data=True)):
            print(f"  {node}")
            print(f"    type: {attrs.get('type')}")
            print(f"    name: {attrs.get('name')}")
            print(f"    file: {attrs.get('file')}")
            print()
