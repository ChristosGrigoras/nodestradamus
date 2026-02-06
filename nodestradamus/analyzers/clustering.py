"""Internal function clustering for refactoring analysis.

Identifies logical groupings of related functions within a file
based on call relationships and naming patterns.
"""

import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.models.graph import FunctionCluster


def _infer_cluster_name(function_names: list[str]) -> str:
    """Infer a descriptive name for a cluster of functions.

    Uses common prefixes, suffixes, or keyword patterns.

    Args:
        function_names: List of function names in the cluster.

    Returns:
        Inferred cluster name.
    """
    if not function_names:
        return "misc"

    # Try to find common prefix
    if len(function_names) >= 2:
        # Find longest common prefix
        prefix = function_names[0]
        for name in function_names[1:]:
            while prefix and not name.startswith(prefix):
                prefix = prefix[:-1]
            if len(prefix) <= 2:
                break

        # If prefix is meaningful (3+ chars), use it
        if len(prefix) >= 3:
            # Clean up trailing underscores
            prefix = prefix.rstrip("_")
            if prefix:
                return prefix

    # Look for common patterns in names
    patterns = {
        r"(parse|extract|read)": "parsing",
        r"(write|save|store|persist)": "persistence",
        r"(validate|check|verify)": "validation",
        r"(format|render|display)": "formatting",
        r"(load|fetch|get)": "loading",
        r"(cache|memo)": "caching",
        r"(log|trace|debug)": "logging",
        r"(test|assert|expect)": "testing",
        r"(init|setup|configure)": "initialization",
        r"(clean|clear|reset)": "cleanup",
        r"(sql|query|database|db)": "database",
        r"(http|request|response|api)": "http",
        r"(auth|login|logout|session)": "authentication",
        r"(file|path|dir)": "filesystem",
    }

    # Count pattern matches
    pattern_counts: dict[str, int] = defaultdict(int)
    for name in function_names:
        name_lower = name.lower()
        for pattern, label in patterns.items():
            if re.search(pattern, name_lower):
                pattern_counts[label] += 1

    if pattern_counts:
        # Return the most common pattern
        return max(pattern_counts.items(), key=lambda x: x[1])[0]

    # Fallback: use first function's first word
    first_name = function_names[0]
    # Split on underscore or camelCase
    if "_" in first_name:
        return first_name.split("_")[0]
    else:
        # CamelCase: extract first word
        match = re.match(r"([a-z]+)", first_name)
        if match:
            return match.group(1)

    return "misc"


def _compute_cohesion(
    subgraph: nx.DiGraph,
    functions: set[str],
) -> float:
    """Compute cohesion score for a cluster.

    Cohesion = internal edges / (size * (size - 1))
    A fully connected cluster has cohesion 1.0.

    Args:
        subgraph: The subgraph containing just the cluster nodes.
        functions: Set of function node IDs in the cluster.

    Returns:
        Cohesion score between 0.0 and 1.0.
    """
    size = len(functions)
    if size <= 1:
        return 1.0

    # Count internal edges
    internal_edges = 0
    for source, target in subgraph.edges():
        if source in functions and target in functions:
            internal_edges += 1

    # Maximum possible edges in directed graph
    max_edges = size * (size - 1)

    if max_edges == 0:
        return 1.0

    return min(1.0, internal_edges / max_edges)


def cluster_functions_in_file(
    repo_path: Path,
    file_path: str,
    min_cluster_size: int = 2,
) -> list[FunctionCluster]:
    """Identify logical function groupings within a single file.

    Uses internal call relationships to find connected components,
    then infers cluster names from function naming patterns.

    Args:
        repo_path: Path to repository root.
        file_path: Relative path to the file to analyze.
        min_cluster_size: Minimum functions to form a cluster (default: 2).

    Returns:
        List of FunctionCluster sorted by line number.
    """
    repo_path = Path(repo_path).resolve()

    # Build the dependency graph
    G = analyze_deps(repo_path)

    # Find all function nodes in this file
    file_functions: dict[str, dict] = {}  # node_id -> {name, line}

    for node_id, attrs in G.nodes(data=True):
        if attrs.get("file") == file_path and attrs.get("type") == "function":
            file_functions[node_id] = {
                "name": attrs.get("name", ""),
                "line": attrs.get("line", 0),
            }

    if len(file_functions) < min_cluster_size:
        return []

    # Build subgraph of just internal calls within this file
    internal_calls = nx.DiGraph()

    for node_id in file_functions:
        internal_calls.add_node(node_id)

    for source, target, attrs in G.edges(data=True):
        if attrs.get("type") == "calls":
            if source in file_functions and target in file_functions:
                internal_calls.add_edge(source, target)

    # Find connected components (treat as undirected for clustering)
    undirected = internal_calls.to_undirected()
    components = list(nx.connected_components(undirected))

    # Build cluster objects
    clusters: list[FunctionCluster] = []

    for component in components:
        if len(component) < min_cluster_size:
            continue

        # Get function info for this component
        func_names = []
        func_lines = []

        for node_id in component:
            info = file_functions[node_id]
            func_names.append(info["name"])
            func_lines.append(info["line"])

        if not func_lines:
            continue

        # Compute cluster boundaries
        line_start = min(func_lines)
        line_end = max(func_lines)

        # Infer cluster name
        cluster_name = _infer_cluster_name(sorted(func_names))

        # Compute cohesion
        cohesion = _compute_cohesion(internal_calls, component)

        clusters.append(
            FunctionCluster(
                name=cluster_name,
                line_start=line_start,
                line_end=line_end,
                functions=sorted(func_names),
                cohesion_score=round(cohesion, 2),
            )
        )

    # Sort by line number
    clusters.sort(key=lambda c: c.line_start)

    return clusters
