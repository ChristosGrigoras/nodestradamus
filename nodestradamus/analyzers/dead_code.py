"""Dead code detection analyzer.

Identifies unused code (functions, classes) that are not referenced
anywhere in the codebase.
"""

from typing import Any

import networkx as nx


def find_dead_code(
    G: nx.DiGraph,
    include_test_files: bool = False,
) -> list[dict[str, Any]]:
    """Find unused code (nodes with no incoming edges).

    Identifies functions and classes that are defined but never used
    anywhere in the codebase. These are candidates for removal.

    Args:
        G: Dependency graph from analyze_deps.
        include_test_files: If True, include test files in results.
                           Default False (test files often have "unused" code).

    Returns:
        List of unused nodes with file, name, type, and line number.
    """
    unused = []

    for node in G.nodes():
        attrs = G.nodes[node]

        # Skip nodes without file info (external imports, unresolved)
        file_path = attrs.get("file", "")
        if not file_path:
            continue

        # Skip test files by default
        if not include_test_files:
            if (
                "test_" in file_path
                or "_test.py" in file_path
                or "tests/" in file_path
                or "/test/" in file_path
                or ".test." in file_path
                or ".spec." in file_path
            ):
                continue

        # Skip file-level nodes (we want functions/classes)
        node_type = attrs.get("type", "")
        if node_type not in ("function", "class"):
            continue

        # Skip __init__, __main__, and other special methods
        name = attrs.get("name", "")
        if name.startswith("__") and name.endswith("__"):
            continue

        # Skip main entry points
        if name in ("main", "cli", "app", "run", "start"):
            continue

        # No incoming edges = nothing uses this
        if G.in_degree(node) == 0:
            unused.append(
                {
                    "id": node,
                    "file": file_path,
                    "name": name,
                    "type": node_type,
                    "line": attrs.get("line"),
                }
            )

    # Sort by file path for easier review
    unused.sort(key=lambda x: (x["file"], x.get("line") or 0))

    return unused


def find_orphaned_files(G: nx.DiGraph) -> list[dict[str, Any]]:
    """Find files that are not imported anywhere.

    These are files that exist in the codebase but are not referenced
    by any other file. They might be dead code at the file level.

    Args:
        G: Dependency graph from analyze_deps.

    Returns:
        List of orphaned files with path and node count.
    """
    # Group nodes by file
    files: dict[str, list[str]] = {}
    for node in G.nodes():
        file_path = G.nodes[node].get("file", "")
        if file_path:
            if file_path not in files:
                files[file_path] = []
            files[file_path].append(node)

    orphaned = []

    for file_path, nodes in files.items():
        # Skip test files
        if (
            "test_" in file_path
            or "_test.py" in file_path
            or "tests/" in file_path
            or "/test/" in file_path
        ):
            continue

        # Skip __init__.py files
        if file_path.endswith("__init__.py"):
            continue

        # Check if any node in this file has incoming edges from other files
        has_external_dependents = False
        for node in nodes:
            for pred in G.predecessors(node):
                pred_file = G.nodes[pred].get("file", "")
                if pred_file and pred_file != file_path:
                    has_external_dependents = True
                    break
            if has_external_dependents:
                break

        if not has_external_dependents:
            orphaned.append(
                {
                    "file": file_path,
                    "node_count": len(nodes),
                    "nodes": nodes[:5],  # First 5 nodes for reference
                }
            )

    # Sort by file path
    orphaned.sort(key=lambda x: str(x["file"]))

    return orphaned
