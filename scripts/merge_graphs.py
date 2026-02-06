#!/usr/bin/env python3
"""
Merge multiple dependency graphs into a unified graph.

Combines Python, TypeScript, and co-occurrence graphs for cross-language
dependency visualization.

Usage:
    python merge_graphs.py .cursor/graph/
    python merge_graphs.py .cursor/graph/ > .cursor/graph/unified.json
    python merge_graphs.py graph1.json graph2.json
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_graph(filepath: Path) -> dict[str, Any] | None:
    """Load a graph from a JSON file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)
        return None


def normalize_node(node: str, source_type: str) -> str:
    """Normalize node names for consistency across graphs."""
    # Add prefix for disambiguation if needed
    if source_type == "python" and "::" in node:
        return f"py:{node}"
    elif source_type == "typescript" and "::" in node:
        return f"ts:{node}"
    return node


def merge_graphs(graphs: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    """Merge multiple graphs into a unified structure.

    Args:
        graphs: List of (graph_dict, source_type) tuples

    Returns:
        Merged graph dictionary
    """
    all_nodes: set[str] = set()
    all_node_details: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    all_files: set[str] = set()
    sources: list[str] = []
    errors: list[dict[str, Any]] = []

    for graph, source_type in graphs:
        sources.append(source_type)

        # Collect nodes
        if "nodes" in graph:
            for node in graph["nodes"]:
                normalized = normalize_node(node, source_type)
                all_nodes.add(normalized)

        # Collect node details
        if "node_details" in graph:
            for detail in graph["node_details"]:
                detail_copy = detail.copy()
                detail_copy["source"] = source_type
                if "name" in detail_copy:
                    detail_copy["name"] = normalize_node(detail_copy["name"], source_type)
                all_node_details.append(detail_copy)

        # Collect files
        if "files" in graph:
            all_files.update(graph["files"])

        # Collect edges
        if "edges" in graph:
            for edge in graph["edges"]:
                edge_copy = edge.copy()
                edge_copy["source"] = source_type

                # Normalize node references in edges
                if "from" in edge_copy:
                    edge_copy["from"] = normalize_node(edge_copy["from"], source_type)
                if "to" in edge_copy:
                    edge_copy["to"] = normalize_node(edge_copy["to"], source_type)

                all_edges.append(edge_copy)

        # Collect errors
        if "errors" in graph:
            for error in graph["errors"]:
                error_copy = error.copy() if isinstance(error, dict) else {"message": str(error)}
                error_copy["source"] = source_type
                errors.append(error_copy)

    # Deduplicate edges (same from/to/type combination)
    seen_edges: set[tuple[str, str, str]] = set()
    unique_edges: list[dict[str, Any]] = []
    for edge in all_edges:
        key = (edge.get("from", ""), edge.get("to", ""), edge.get("type", ""))
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    # Build cross-language connections from co-occurrence data
    cross_language_edges = find_cross_language_connections(graphs)
    unique_edges.extend(cross_language_edges)

    return {
        "nodes": sorted(all_nodes),
        "node_details": all_node_details,
        "files": sorted(all_files),
        "edges": unique_edges,
        "errors": errors,
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "generator": "merge_graphs.py",
            "sources": sources,
            "stats": {
                "total_nodes": len(all_nodes),
                "total_edges": len(unique_edges),
                "total_files": len(all_files),
                "cross_language_edges": len(cross_language_edges),
            },
        },
    }


def find_cross_language_connections(graphs: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    """Find connections between files in different language graphs.

    Uses co-occurrence data to link Python and TypeScript files that
    change together frequently.
    """
    cross_edges: list[dict[str, Any]] = []

    # Separate graphs by type
    cooccurrence_graph = None
    python_files: set[str] = set()
    ts_files: set[str] = set()

    for graph, source_type in graphs:
        if source_type == "cooccurrence":
            cooccurrence_graph = graph
        elif source_type == "python":
            python_files.update(graph.get("files", []))
            # Also add node file paths
            for node in graph.get("nodes", []):
                if "::" in node:
                    python_files.add(node.split("::")[0])
        elif source_type == "typescript":
            ts_files.update(graph.get("files", []))

    if not cooccurrence_graph:
        return cross_edges

    # Look for co-occurrence edges between Python and TypeScript files
    for edge in cooccurrence_graph.get("edges", []):
        from_file = edge.get("from", "")
        to_file = edge.get("to", "")

        from_is_py = from_file.endswith(".py") or from_file in python_files
        from_is_ts = any(from_file.endswith(ext) for ext in [".ts", ".tsx", ".js", ".jsx"])
        to_is_py = to_file.endswith(".py") or to_file in python_files
        to_is_ts = any(to_file.endswith(ext) for ext in [".ts", ".tsx", ".js", ".jsx"])

        # Cross-language connection
        if (from_is_py and to_is_ts) or (from_is_ts and to_is_py):
            cross_edges.append({
                "from": from_file,
                "to": to_file,
                "type": "co-occurs-cross-language",
                "source": "merged",
                "strength": edge.get("strength", 0),
                "count": edge.get("count", 0),
            })

    return cross_edges


def detect_graph_type(graph: dict[str, Any], filename: str) -> str:
    """Detect the type of graph based on content and filename."""
    # Remove .json extension for pattern matching to avoid false positives
    name_lower = filename.lower().replace(".json", "")

    if "python" in name_lower or "py-deps" in name_lower:
        return "python"
    elif "ts-" in name_lower or "typescript" in name_lower or "-js" in name_lower:
        return "typescript"
    elif "cooccur" in name_lower or "co-occur" in name_lower:
        return "cooccurrence"

    # Try to detect from content
    metadata = graph.get("metadata", {})
    generator = metadata.get("generator", "")

    if "python" in generator:
        return "python"
    elif "ts" in generator:
        return "typescript"
    elif "cooccur" in generator:
        return "cooccurrence"

    # Check edge types
    for edge in graph.get("edges", [])[:10]:
        if edge.get("type") == "co-occurs":
            return "cooccurrence"
        if edge.get("type") == "imports":
            return "typescript"  # Python uses "calls"

    return "unknown"


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print("Usage: python merge_graphs.py <directory_or_files...>", file=sys.stderr)
        print("  python merge_graphs.py .cursor/graph/", file=sys.stderr)
        print("  python merge_graphs.py graph1.json graph2.json", file=sys.stderr)
        sys.exit(1)

    graphs: list[tuple[dict[str, Any], str]] = []

    for arg in args:
        path = Path(arg)

        if path.is_dir():
            # Load all JSON files from directory
            for filepath in sorted(path.glob("*.json")):
                # Skip unified output to prevent recursion
                if "unified" in filepath.name or "merged" in filepath.name:
                    continue

                graph = load_graph(filepath)
                if graph:
                    graph_type = detect_graph_type(graph, filepath.name)
                    graphs.append((graph, graph_type))

        elif path.is_file():
            graph = load_graph(path)
            if graph:
                graph_type = detect_graph_type(graph, path.name)
                graphs.append((graph, graph_type))

        else:
            print(f"Warning: {path} not found", file=sys.stderr)

    if not graphs:
        print("Error: No valid graph files found", file=sys.stderr)
        sys.exit(1)

    # Merge graphs
    merged = merge_graphs(graphs)

    # Output
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
