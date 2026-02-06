"""Graph summarization utilities.

Provides compact summaries of dependency and co-occurrence graphs
for MCP tool responses.
"""

from collections import Counter
from typing import Any

import networkx as nx

from nodestradamus.analyzers.deps import graph_metadata


def summarize_digraph(
    G: nx.DiGraph,
    top_n_count: int = 15,
    include_fields: bool = False,
) -> dict[str, Any]:
    """Create a compact summary of a dependency graph.

    Args:
        G: NetworkX directed graph.
        top_n_count: Number of top items to include.
        include_fields: Whether to include schema fields in output.

    Returns:
        Compact summary dict.
    """
    meta = graph_metadata(G)

    # Find external imports (edges to unresolved nodes)
    external_imports = set()
    internal_calls = 0
    fk_references = 0
    for _, target, attrs in G.edges(data=True):
        if attrs.get("resolved", False):
            internal_calls += 1
        else:
            if not target.startswith(("py:", "ts:", "sh:", "rs:", "sql:", "json:")):
                external_imports.add(target)
        if attrs.get("type") == "references_fk":
            fk_references += 1

    # Top nodes by connectivity
    outgoing = Counter(source for source, _ in G.edges())
    incoming = Counter(target for _, target in G.edges())

    top_callers = [
        {"name": short_name(node), "calls": count}
        for node, count in outgoing.most_common(top_n_count)
    ]
    top_called = [
        {"name": short_name(node), "called_by": count}
        for node, count in incoming.most_common(top_n_count)
    ]

    # Classes, functions, tables, and configs
    classes = []
    functions = []
    tables = []
    configs = []
    for _node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "")
        node_info: dict[str, Any] = {
            "name": attrs.get("name", ""),
            "file": attrs.get("file", ""),
            "line": attrs.get("line"),
        }

        # Include fields if requested and present
        if include_fields:
            fields = attrs.get("fields")
            if fields:
                node_info["fields"] = fields

        if node_type == "class":
            classes.append(node_info)
        elif node_type == "function":
            functions.append({"name": attrs.get("name", ""), "file": attrs.get("file", "")})
        elif node_type == "table":
            tables.append(node_info)
        elif node_type == "config":
            configs.append(node_info)

    # Group by directory
    directories: Counter[str] = Counter()
    files = set()
    for _, attrs in G.nodes(data=True):
        file_path = attrs.get("file", "")
        if file_path:
            files.add(file_path)
            dir_name = file_path.split("/")[0] if "/" in file_path else "."
            directories[dir_name] += 1

    result: dict[str, Any] = {
        "summary": {
            "total_files": len(files),
            "total_nodes": meta["node_count"],
            "total_edges": meta["edge_count"],
            "functions": meta["node_types"].get("function", 0),
            "classes": meta["node_types"].get("class", 0),
            "tables": meta["node_types"].get("table", 0),
            "configs": meta["node_types"].get("config", 0),
            "internal_calls": internal_calls,
            "external_imports": len(external_imports),
            "fk_references": fk_references,
        },
        "directories": dict(directories.most_common(10)),
        "external_imports": sorted(external_imports)[:30],
        "top_callers": top_callers[:10],
        "top_called": top_called[:10],
        "classes": classes[:top_n_count],
        "sample_functions": functions[:10],
    }

    # Add tables and configs if present
    if tables:
        result["tables"] = tables[:top_n_count]
    if configs:
        result["configs"] = configs[:top_n_count]

    return result


def summarize_cooccurrence(G: nx.Graph, top_n_count: int = 20) -> dict[str, Any]:
    """Create a compact summary of a co-occurrence graph.

    Optimized to iterate edges only once for better performance on large graphs.

    Args:
        G: NetworkX undirected graph.
        top_n_count: Number of top pairs to include.

    Returns:
        Compact summary dict.
    """
    import heapq

    edge_count = G.number_of_edges()
    if edge_count == 0:
        return {
            "summary": {
                "file_count": G.number_of_nodes(),
                "edge_count": 0,
                "avg_strength": 0.0,
                "max_strength": 0.0,
            },
            "top_co_occurring_pairs": [],
            "change_hotspots": [],
        }

    # Single pass over edges: compute stats, top pairs, and file counts
    file_counts: Counter[str] = Counter()
    strength_sum = 0.0
    max_strength = 0.0

    # Use heap for top N pairs (more efficient than full sort for large edge sets)
    top_pairs_heap: list[tuple[float, str, str, int]] = []

    for source, target, attrs in G.edges(data=True):
        strength = attrs.get("strength", 0.0)
        count = attrs.get("count", 1)

        # Track strength stats incrementally (no list allocation)
        strength_sum += strength
        if strength > max_strength:
            max_strength = strength

        # Track file counts
        file_counts[source] += count
        file_counts[target] += count

        # Maintain top N pairs using heap
        if len(top_pairs_heap) < top_n_count:
            heapq.heappush(top_pairs_heap, (strength, source, target, count))
        elif strength > top_pairs_heap[0][0]:
            heapq.heapreplace(top_pairs_heap, (strength, source, target, count))

    # Build metadata
    meta = {
        "file_count": G.number_of_nodes(),
        "edge_count": edge_count,
        "avg_strength": strength_sum / edge_count,
        "max_strength": max_strength,
    }

    # Extract top pairs from heap (sorted by strength descending)
    top_pairs_heap.sort(key=lambda x: x[0], reverse=True)
    top_pairs = [
        {
            "file_a": source,
            "file_b": target,
            "strength": strength,
            "times_together": count,
        }
        for strength, source, target, count in top_pairs_heap
    ]

    hotspots = [
        {"file": f, "co_changes": count} for f, count in file_counts.most_common(15)
    ]

    return {
        "summary": meta,
        "top_co_occurring_pairs": top_pairs,
        "change_hotspots": hotspots,
    }


def short_name(node_id: str) -> str:
    """Extract short name from node ID like 'py:path/file.py::func'.

    Args:
        node_id: Full node identifier.

    Returns:
        Shortened name for display.
    """
    if "::" in node_id:
        parts = node_id.split("::")
        file_part = parts[0].replace("py:", "").replace("ts:", "").split("/")[-1]
        return f"{file_part}::{parts[-1]}"
    return node_id
