"""NetworkX-based topology analysis for string references.

Uses graph algorithms to identify significant strings based on topology:
- Strings referenced by multiple files are likely important
- Strings appearing only once are likely noise (UI text, error messages)

This provides automatic signal/noise separation without hand-coded heuristics.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from nodestradamus.analyzers.string_refs import analyze_string_refs
from nodestradamus.models.graph import (
    SignificantString,
    StringAnalysisResult,
    StringContext,
    StringRefGraph,
)


def build_bipartite_graph(refs: StringRefGraph) -> nx.Graph:
    """Build a bipartite graph connecting files to strings.

    The graph has two types of nodes:
    - file nodes (bipartite=0): represent source files
    - string nodes (bipartite=1): represent string literals

    Edges connect files to the strings they contain.

    Args:
        refs: The string reference graph from extraction.

    Returns:
        NetworkX Graph with bipartite structure.
    """
    G = nx.Graph()

    for node in refs.strings:
        file_node = f"file:{node.file}"
        string_node = f"string:{node.value}"

        # Add file node if not exists
        if not G.has_node(file_node):
            G.add_node(file_node, bipartite=0, type="file", path=node.file)

        # Add string node if not exists
        if not G.has_node(string_node):
            G.add_node(string_node, bipartite=1, type="string", value=node.value)

        # Add edge with context as edge data
        # If edge already exists, we'll merge contexts later
        if G.has_edge(file_node, string_node):
            # Append contexts to existing edge
            G[file_node][string_node]["contexts"].extend([c.model_dump() for c in node.contexts])
        else:
            G.add_edge(
                file_node,
                string_node,
                contexts=[c.model_dump() for c in node.contexts],
            )

    return G


def get_string_degree(G: nx.Graph, string_node: str) -> int:
    """Get the number of files that reference a string.

    Args:
        G: The bipartite graph.
        string_node: The string node ID (prefixed with 'string:').

    Returns:
        Number of file nodes connected to this string.
    """
    return G.degree(string_node)


def identify_significant_strings(
    G: nx.Graph,
    min_files: int = 2,
) -> list[dict[str, Any]]:
    """Find strings that are referenced by multiple files.

    These are likely important runtime references like:
    - Config file paths
    - Channel names
    - API endpoints
    - Environment variable names

    Args:
        G: The bipartite graph.
        min_files: Minimum number of files to consider a string significant.

    Returns:
        List of significant string dicts with metadata.
    """
    significant = []

    # Get all string nodes
    string_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "string"]

    for node_id, node_data in string_nodes:
        degree = G.degree(node_id)

        if degree >= min_files:
            # Gather all files and contexts for this string
            files = []
            all_contexts = []

            for neighbor in G.neighbors(node_id):
                neighbor_data = G.nodes[neighbor]
                if neighbor_data.get("type") == "file":
                    files.append(neighbor_data.get("path", neighbor))
                    edge_data = G[node_id][neighbor]
                    all_contexts.extend(edge_data.get("contexts", []))

            significant.append(
                {
                    "value": node_data.get("value", node_id.replace("string:", "")),
                    "referenced_by": sorted(files),
                    "contexts": all_contexts,
                    "reference_count": degree,
                }
            )

    # Sort by reference count (most referenced first)
    significant.sort(key=lambda x: x["reference_count"], reverse=True)

    return significant


def calculate_importance_scores(
    significant_strings: list[dict[str, Any]],
    total_files: int,
) -> list[dict[str, Any]]:
    """Calculate normalized importance scores for significant strings.

    The importance score is based on:
    - Reference count relative to total files
    - Diversity of contexts (different call sites = more important)

    Args:
        significant_strings: List of significant string dicts.
        total_files: Total number of files in the codebase.

    Returns:
        Updated list with importance_score field.
    """
    if not significant_strings:
        return []

    max_refs = max(s["reference_count"] for s in significant_strings)

    for s in significant_strings:
        # Base score from reference count (0.5 - 1.0 range)
        ref_score = 0.5 + 0.5 * (s["reference_count"] / max_refs)

        # Bonus for diverse contexts (different call sites)
        call_sites = set()
        for ctx in s["contexts"]:
            if ctx.get("call_site"):
                call_sites.add(ctx["call_site"])

        # More diverse call sites = higher importance
        diversity_bonus = min(0.2, 0.05 * len(call_sites))

        # Coverage bonus (% of files that reference this string)
        coverage_bonus = min(0.1, 0.1 * (s["reference_count"] / max(1, total_files)))

        s["importance_score"] = min(1.0, ref_score + diversity_bonus + coverage_bonus)

    return significant_strings


def analyze_string_topology(
    repo_path: str | Path,
    min_files: int = 2,
    include_single_use: bool = False,
    include_python: bool = True,
    include_typescript: bool = True,
    include_sql: bool = True,
    include_rust: bool = True,
    include_bash: bool = True,
) -> StringAnalysisResult:
    """Analyze string references and rank by importance using graph topology.

    This is the main entry point for string topology analysis. It:
    1. Extracts all string literals from the codebase
    2. Builds a bipartite graph (files <-> strings)
    3. Uses graph degree to identify significant strings
    4. Calculates importance scores

    Args:
        repo_path: Absolute path to repository root.
        min_files: Minimum files to consider a string significant.
        include_single_use: Include strings that appear in only one file.
        include_python: Analyze Python files.
        include_typescript: Analyze TypeScript/JavaScript files.
        include_sql: Analyze SQL files.
        include_rust: Analyze Rust files.
        include_bash: Analyze Bash files.

    Returns:
        StringAnalysisResult with ranked significant strings.
    """
    repo_path = Path(repo_path).resolve()

    # Step 1: Extract all strings
    refs = analyze_string_refs(
        repo_path,
        include_python=include_python,
        include_typescript=include_typescript,
        include_sql=include_sql,
        include_rust=include_rust,
        include_bash=include_bash,
    )

    if not refs.strings:
        return StringAnalysisResult(
            significant_strings=[],
            metadata={
                "total_strings_found": 0,
                "significant_strings": 0,
                "noise_filtered": 0,
                "min_files_threshold": min_files,
            },
        )

    # Step 2: Build bipartite graph
    G = build_bipartite_graph(refs)

    # Count unique files
    file_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "file"]
    total_files = len(file_nodes)

    # Step 3: Identify significant strings
    effective_min = 1 if include_single_use else min_files
    significant = identify_significant_strings(G, min_files=effective_min)

    # Step 4: Calculate importance scores
    significant = calculate_importance_scores(significant, total_files)

    # Convert to Pydantic models
    result_strings = []
    for s in significant:
        contexts = [
            StringContext(
                call_site=c.get("call_site"),
                variable_name=c.get("variable_name"),
                enclosing_function=c.get("enclosing_function"),
                enclosing_class=c.get("enclosing_class"),
                line=c.get("line", 0),
            )
            for c in s["contexts"]
        ]

        result_strings.append(
            SignificantString(
                value=s["value"],
                referenced_by=s["referenced_by"],
                contexts=contexts,
                importance_score=s["importance_score"],
                reference_count=s["reference_count"],
            )
        )

    # Calculate noise filtered
    total_unique_strings = len([n for n, d in G.nodes(data=True) if d.get("type") == "string"])
    noise_filtered = total_unique_strings - len(result_strings)

    return StringAnalysisResult(
        significant_strings=result_strings,
        metadata={
            "total_strings_found": len(refs.strings),
            "unique_strings": total_unique_strings,
            "significant_strings": len(result_strings),
            "noise_filtered": noise_filtered,
            "min_files_threshold": min_files,
            "files_analyzed": total_files,
        },
    )


def find_string_usages(
    repo_path: str | Path,
    target_string: str,
    include_python: bool = True,
    include_typescript: bool = True,
    include_sql: bool = True,
    include_rust: bool = True,
    include_bash: bool = True,
) -> dict[str, Any]:
    """Find all usages of a specific string in the codebase.

    Useful for answering questions like:
    - "What services communicate via this channel?"
    - "What files use this config path?"

    Args:
        repo_path: Absolute path to repository root.
        target_string: The string to search for.
        include_python: Analyze Python files.
        include_typescript: Analyze TypeScript/JavaScript files.
        include_sql: Analyze SQL files.
        include_rust: Analyze Rust files.
        include_bash: Analyze Bash files.

    Returns:
        Dictionary with usages grouped by file.
    """
    repo_path = Path(repo_path).resolve()

    refs = analyze_string_refs(
        repo_path,
        include_python=include_python,
        include_typescript=include_typescript,
        include_sql=include_sql,
        include_rust=include_rust,
        include_bash=include_bash,
    )

    usages: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for node in refs.strings:
        if node.value == target_string:
            usages[node.file].extend([c.model_dump() for c in node.contexts])

    return {
        "target_string": target_string,
        "files": list(usages.keys()),
        "total_usages": sum(len(v) for v in usages.values()),
        "usages_by_file": dict(usages),
    }
