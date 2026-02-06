"""Graph algorithm handlers.

Consolidated handler for analyze_graph tool with all algorithms inlined:
- pagerank: Importance ranking
- betweenness: Bottleneck detection
- communities: Module clustering
- cycles: Circular dependency detection
- path: Shortest dependency path
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import networkx as nx

from nodestradamus.analyzers.constants import STDLIB_MODULES, is_test_file
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.graph_algorithms import (
    betweenness,
    community_metagraph,
    detect_communities,
    find_cycles,
    hierarchical_view,
    pagerank,
    sampled_betweenness,
    shortest_path,
    top_n,
    validate_layers,
)
from nodestradamus.analyzers.ignore import load_ignore_patterns
from nodestradamus.mcp.tools.utils.summarize import short_name


def _is_external_import(node_id: str, file_path: str) -> bool:
    """Check if a node represents an external/stdlib import.

    Args:
        node_id: The node identifier.
        file_path: The file path from node attributes (empty for externals).

    Returns:
        True if the node is external, False otherwise.
    """
    # External imports have no file path
    if not file_path:
        # Check if it's a known stdlib module
        root_module = node_id.split(".")[0]
        if root_module in STDLIB_MODULES:
            return True
        # If no file path and not clearly internal, treat as external
        return True
    return False


def _filter_graph_for_ranking(
    G: nx.DiGraph,
    scope: str = "source_only",
    exclude_external: bool = True,
) -> nx.DiGraph:
    """Create a filtered subgraph for ranking algorithms.

    Args:
        G: Original dependency graph.
        scope: 'source_only', 'tests_only', or 'all'.
        exclude_external: Whether to exclude external imports.

    Returns:
        Filtered subgraph with only the requested nodes.
    """
    nodes_to_keep = []

    for node_id in G.nodes():
        attrs = G.nodes[node_id]
        file_path = attrs.get("file", "")

        # Check external filter
        if exclude_external and _is_external_import(node_id, file_path):
            continue

        # Check scope filter
        is_test = is_test_file(file_path)
        if scope == "source_only" and is_test:
            continue
        elif scope == "tests_only" and not is_test:
            continue
        # scope == "all" includes everything

        nodes_to_keep.append(node_id)

    return G.subgraph(nodes_to_keep).copy()


async def handle_analyze_graph(arguments: dict[str, Any]) -> str:
    """Handle analyze_graph consolidated tool call.

    Dispatches to appropriate algorithm based on 'algorithm' parameter.
    The dependency graph is built once and reused for all algorithms.

    Args:
        arguments: Tool arguments with repo_path, algorithm, and algorithm-specific params.

    Returns:
        JSON string with algorithm results.

    Raises:
        ValueError: If required parameters are missing or algorithm is unknown.
    """
    repo_path = arguments.get("repo_path")
    algorithm = arguments.get("algorithm")

    if not repo_path:
        raise ValueError("repo_path is required")
    if not algorithm:
        raise ValueError("algorithm is required")

    # Load exclude patterns: use provided list, or auto-load from .nodestradamusignore + defaults
    exclude = arguments.get("exclude")
    if exclude is None:
        exclude = list(load_ignore_patterns(Path(repo_path)))

    # Build graph once for all algorithms
    G = await asyncio.to_thread(analyze_deps, repo_path, exclude=exclude)

    if G.number_of_nodes() == 0:
        return json.dumps({"error": "No nodes found in dependency graph"}, indent=2)

    # Dispatch to algorithm
    if algorithm == "pagerank":
        return await _run_pagerank(G, arguments)
    elif algorithm == "betweenness":
        return await _run_betweenness(G, arguments)
    elif algorithm == "communities":
        return await _run_communities(G, arguments)
    elif algorithm == "cycles":
        return await _run_cycles(G, arguments)
    elif algorithm == "path":
        return _run_path(G, arguments)
    elif algorithm == "hierarchy":
        return _run_hierarchy(G, arguments)
    elif algorithm == "layers":
        return _run_layers(G, arguments)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


async def _run_pagerank(G: nx.DiGraph, arguments: dict[str, Any]) -> str:
    """Run PageRank algorithm for importance ranking.

    Supports hierarchical (module/package-level) analysis for faster results on large graphs.

    Args:
        G: NetworkX directed graph.
        arguments: Tool arguments with top_n, scope, exclude_external, level.

    Returns:
        JSON string with ranked nodes by importance.
    """
    top_n_count = arguments.get("top_n", 20)
    scope = arguments.get("scope", "source_only")
    exclude_external = arguments.get("exclude_external", True)
    level = arguments.get("level")  # None, "module", or "package" for hierarchical

    # Filter graph based on scope and external filter
    filtered_G = _filter_graph_for_ranking(G, scope, exclude_external)

    if filtered_G.number_of_nodes() == 0:
        return json.dumps(
            {
                "most_important_code": [],
                "total_nodes_analyzed": 0,
                "original_nodes": G.number_of_nodes(),
                "filter_applied": {"scope": scope, "exclude_external": exclude_external},
                "message": "No nodes match the filter criteria. Try scope='all' or exclude_external=false.",
            },
            indent=2,
        )

    # Hierarchical mode: collapse to module/package level first
    analysis_graph = filtered_G
    method = "exact"

    if level in ("module", "package"):
        hier_nodes, hier_edges = await asyncio.to_thread(hierarchical_view, filtered_G, level=level)
        # Build collapsed graph
        analysis_graph = nx.DiGraph()
        for node in hier_nodes:
            if node["id"] not in ("[stdlib]", "[external]", "unknown"):
                analysis_graph.add_node(node["id"])
        for edge in hier_edges:
            if edge["source"] in analysis_graph and edge["target"] in analysis_graph:
                analysis_graph.add_edge(edge["source"], edge["target"])
        method = f"hierarchical-{level}"

    n = analysis_graph.number_of_nodes()
    scores = await asyncio.to_thread(pagerank, analysis_graph)
    ranked = top_n(scores, n=top_n_count)

    result = [
        {"node": short_name(node), "full_id": node, "importance": round(score, 6)}
        for node, score in ranked
    ]

    return json.dumps(
        {
            "most_important_code": result,
            "total_nodes_analyzed": n,
            "original_nodes": G.number_of_nodes(),
            "filter_applied": {
                "scope": scope,
                "exclude_external": exclude_external,
                "level": level,
            },
            "method": method,
        },
        indent=2,
    )


async def _run_betweenness(G: nx.DiGraph, arguments: dict[str, Any]) -> str:
    """Run betweenness centrality for bottleneck detection.

    Supports auto-sampling for large graphs and hierarchical (module-level) analysis.

    Args:
        G: NetworkX directed graph.
        arguments: Tool arguments with top_n, scope, exclude_external, sample_size, level.

    Returns:
        JSON string with nodes ranked by betweenness centrality.
    """
    top_n_count = arguments.get("top_n", 20)
    scope = arguments.get("scope", "source_only")
    exclude_external = arguments.get("exclude_external", True)
    sample_size = arguments.get("sample_size")  # None = auto-detect
    level = arguments.get("level")  # None, "module", or "package" for hierarchical

    # Auto-sampling threshold: graphs larger than this use sampling by default
    AUTO_SAMPLE_THRESHOLD = 10_000

    # Filter graph based on scope and external filter
    filtered_G = _filter_graph_for_ranking(G, scope, exclude_external)

    if filtered_G.number_of_nodes() == 0:
        return json.dumps(
            {
                "bottlenecks": [],
                "total_nodes_analyzed": 0,
                "original_nodes": G.number_of_nodes(),
                "filter_applied": {"scope": scope, "exclude_external": exclude_external},
                "message": "No nodes match the filter criteria. Try scope='all' or exclude_external=false.",
            },
            indent=2,
        )

    # Hierarchical mode: collapse to module/package level first
    analysis_graph = filtered_G

    if level in ("module", "package"):
        hier_nodes, hier_edges = await asyncio.to_thread(hierarchical_view, filtered_G, level=level)
        # Build collapsed graph
        analysis_graph = nx.DiGraph()
        for node in hier_nodes:
            if node["id"] not in ("[stdlib]", "[external]", "unknown"):
                analysis_graph.add_node(node["id"])
        for edge in hier_edges:
            if edge["source"] in analysis_graph and edge["target"] in analysis_graph:
                analysis_graph.add_edge(edge["source"], edge["target"])

    n = analysis_graph.number_of_nodes()

    # Determine computation method
    if sample_size is not None and sample_size > 0:
        # Explicit sampling requested
        k = min(sample_size, n)
        scores = await asyncio.to_thread(sampled_betweenness, analysis_graph, sample_size=k)
        method = f"sampled (k={k})"
    elif n > AUTO_SAMPLE_THRESHOLD:
        # Auto-sample for large graphs: sample 10% or max 500
        k = min(500, max(100, n // 10))
        scores = await asyncio.to_thread(sampled_betweenness, analysis_graph, sample_size=k)
        method = f"auto-sampled (k={k}, threshold={AUTO_SAMPLE_THRESHOLD:,})"
    else:
        # Exact computation for smaller graphs
        scores = await asyncio.to_thread(betweenness, analysis_graph)
        method = "exact"

    if level:
        method = f"hierarchical-{level} + {method}"

    ranked = top_n(scores, n=top_n_count)

    result = [
        {"node": short_name(node), "full_id": node, "betweenness": round(score, 6)}
        for node, score in ranked
    ]

    return json.dumps(
        {
            "bottlenecks": result,
            "total_nodes_analyzed": n,
            "original_nodes": G.number_of_nodes(),
            "filter_applied": {
                "scope": scope,
                "exclude_external": exclude_external,
                "level": level,
                "sample_size": sample_size,
            },
            "method": method,
            "explanation": "High betweenness = changes here ripple to many paths",
        },
        indent=2,
    )


def _compute_community_name(members: list[str], G: nx.DiGraph) -> str:
    """Compute a descriptive name for a community based on common prefix.

    Args:
        members: List of node IDs in the community.
        G: NetworkX graph for file path lookup.

    Returns:
        Descriptive name like "langchain_core.runnables.*" or "tests/*".
    """
    if not members:
        return "empty"

    # Extract file paths from members
    file_paths = []
    for node_id in members:
        file_path = G.nodes.get(node_id, {}).get("file", "")
        if file_path:
            file_paths.append(file_path)

    if not file_paths:
        # No file paths, try to find common prefix in node IDs
        # Remove language prefix (py:, ts:, etc.)
        clean_ids = []
        for node_id in members:
            if ":" in node_id:
                clean_ids.append(node_id.split(":", 1)[1])
            else:
                clean_ids.append(node_id)

        if not clean_ids:
            return "unknown"

        # Find common prefix
        prefix = clean_ids[0]
        for s in clean_ids[1:]:
            while not s.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        return prefix.rstrip("/:") + "/*" if prefix else "mixed"

    # Find common directory prefix
    parts_list = [p.split("/") for p in file_paths]
    if not parts_list:
        return "unknown"

    common_parts = []
    for i in range(min(len(p) for p in parts_list)):
        values = {p[i] for p in parts_list}
        if len(values) == 1:
            common_parts.append(values.pop())
        else:
            break

    if common_parts:
        return "/".join(common_parts) + "/*"

    # No common prefix, use most frequent directory
    first_dirs = [p[0] for p in parts_list if p]
    if first_dirs:
        most_common = max(set(first_dirs), key=first_dirs.count)
        return f"{most_common}/* (mixed)"

    return "mixed"


def _get_community_representative(
    members: list[str], G: nx.DiGraph, pr_scores: dict[str, float]
) -> str:
    """Get the most important node in a community (highest PageRank).

    Args:
        members: List of node IDs in the community.
        G: NetworkX graph.
        pr_scores: Pre-computed PageRank scores.

    Returns:
        Node ID of the representative member, or first member if no scores.
    """
    if not members:
        return ""

    # Find member with highest PageRank
    best_node = members[0]
    best_score = pr_scores.get(members[0], 0.0)

    for node_id in members[1:]:
        score = pr_scores.get(node_id, 0.0)
        if score > best_score:
            best_score = score
            best_node = node_id

    return short_name(best_node)


def _classify_community(members: list[str], G: nx.DiGraph) -> str:
    """Classify a community as source, tests, stdlib, or external.

    Args:
        members: List of node IDs in the community.
        G: NetworkX graph for file path lookup.

    Returns:
        "source", "tests", "stdlib", or "external".
    """
    test_count = 0
    stdlib_count = 0
    external_count = 0
    source_count = 0

    for node_id in members:
        file_path = G.nodes.get(node_id, {}).get("file", "")
        if not file_path:
            # No file path - check if stdlib or third-party
            # Pattern matches _is_external_import (line 108)
            root_module = node_id.split(".")[0]
            if root_module in STDLIB_MODULES:
                stdlib_count += 1
            else:
                external_count += 1
        elif is_test_file(file_path):
            test_count += 1
        else:
            source_count += 1

    # Classify based on majority (>50%)
    total = len(members)
    if stdlib_count > total * 0.5:
        return "stdlib"
    if external_count > total * 0.5:
        return "external"
    if test_count > total * 0.5:
        return "tests"
    return "source"


# Category sort order for communities (source first, then tests, then stdlib, then external)
_CATEGORY_ORDER = {"source": 0, "tests": 1, "stdlib": 2, "external": 3}


async def _run_communities(G, arguments: dict[str, Any] | None = None) -> str:
    """Run community detection for module clustering with meta-graph analysis.

    Args:
        G: NetworkX directed graph.
        arguments: Optional tool arguments with summary_only parameter.

    Returns:
        JSON string with detected communities/modules, inter-module edges,
        and cohesion/coupling metrics. Communities have descriptive names,
        representative members, and are sorted by category then size.
        When summary_only=true, returns concise summary with top 5 modules.
    """
    if arguments is None:
        arguments = {}

    summary_only = arguments.get("summary_only", False)
    communities = await asyncio.to_thread(detect_communities, G)

    # Pre-compute PageRank for representative selection
    try:
        pr_scores = await asyncio.to_thread(nx.pagerank, G)
    except Exception:
        pr_scores = {}

    # Compute meta-graph metrics
    metrics, inter_edges = await asyncio.to_thread(community_metagraph, G, communities)

    # Build metrics lookup by module_id (0-indexed from metagraph)
    metrics_by_id = {m["module_id"]: m for m in metrics}

    # Format communities with descriptive names and metrics
    formatted = []
    for i, community in enumerate(communities):
        members_list = list(community)
        module_metrics = metrics_by_id.get(i, {})
        category = _classify_community(members_list, G)

        formatted.append(
            {
                "module_id": i,
                "module_name": _compute_community_name(members_list, G),
                "representative": _get_community_representative(members_list, G, pr_scores),
                "category": category,
                "size": len(community),
                "cohesion": module_metrics.get("cohesion", 1.0),
                "afferent_coupling": module_metrics.get("afferent_coupling", 0),
                "efferent_coupling": module_metrics.get("efferent_coupling", 0),
                "instability": module_metrics.get("instability", 0.0),
                "members": [short_name(node) for node in sorted(community)][:20],
                "full_members": (
                    sorted(community)[:20] if len(community) <= 20 else f"({len(community)} total)"
                ),
            }
        )

    # Sort by category (source first) then by size (largest first)
    formatted.sort(key=lambda x: (_CATEGORY_ORDER.get(x["category"], 9), -x["size"]))

    # Summary mode (E7): return concise output with top 5 modules
    if summary_only:
        top_modules = []
        for m in formatted[:5]:
            top_modules.append({
                "module_name": m["module_name"],
                "representative": m["representative"],
                "category": m["category"],
                "size": m["size"],
                "cohesion": round(m["cohesion"], 2),
            })

        return json.dumps(
            {
                "summary": {
                    "total_modules": len(communities),
                    "total_nodes": G.number_of_nodes(),
                    "source_modules": sum(1 for m in formatted if m["category"] == "source"),
                    "test_modules": sum(1 for m in formatted if m["category"] == "tests"),
                    "stdlib_modules": sum(1 for m in formatted if m["category"] == "stdlib"),
                    "external_modules": sum(1 for m in formatted if m["category"] == "external"),
                },
                "top_modules": top_modules,
                "truncated": True,
                "message": f"Showing top 5 of {len(communities)} modules. Use summary_only=false for full results.",
            },
            indent=2,
        )

    # Format inter-module edges for output (limit to top 30)
    inter_module_edges = [
        {"source": e["source"], "target": e["target"], "edge_count": e["edge_count"]}
        for e in inter_edges[:30]
    ]

    return json.dumps(
        {
            "modules": formatted[:15],
            "inter_module_edges": inter_module_edges,
            "total_modules": len(communities),
            "total_nodes": G.number_of_nodes(),
            "total_inter_module_edges": len(inter_edges),
            "source_modules": sum(1 for m in formatted if m["category"] == "source"),
            "test_modules": sum(1 for m in formatted if m["category"] == "tests"),
            "stdlib_modules": sum(1 for m in formatted if m["category"] == "stdlib"),
            "external_modules": sum(1 for m in formatted if m["category"] == "external"),
        },
        indent=2,
    )


async def _run_cycles(G, arguments: dict[str, Any]) -> str:
    """Run cycle detection for circular dependencies.

    Args:
        G: NetworkX directed graph.
        arguments: Tool arguments with max_cycles.

    Returns:
        JSON string with detected circular dependencies.
    """
    max_cycles_count = arguments.get("max_cycles", 20)

    # Find real cross-file circular dependencies (filters out intra-file cycles)
    cycles = await asyncio.to_thread(find_cycles, G, cross_file_only=True)

    # Format cycles with short names
    formatted = []
    for cycle in cycles[:max_cycles_count]:
        # Extract unique files in the cycle
        files_in_cycle = set()
        for node in cycle:
            file_path = G.nodes[node].get("file", "")
            if file_path:
                files_in_cycle.add(file_path)

        formatted.append(
            {
                "length": len(cycle),
                "files_involved": len(files_in_cycle),
                "cycle": [short_name(node) for node in cycle],
                "files": sorted(files_in_cycle),
                "full_cycle": cycle,
            }
        )

    if not formatted:
        return json.dumps(
            {
                "circular_dependencies": [],
                "total_cycles_found": 0,
                "message": "No cross-file circular dependencies found. Your codebase has clean import structure!",
            },
            indent=2,
        )

    return json.dumps(
        {
            "circular_dependencies": formatted,
            "total_cycles_found": len(cycles),
            "showing": min(len(cycles), max_cycles_count),
            "note": "Only showing cross-file import cycles (intra-file containment cycles filtered)",
        },
        indent=2,
    )


def _run_path(G, arguments: dict[str, Any]) -> str:
    """Run shortest path algorithm.

    Args:
        G: NetworkX directed graph.
        arguments: Tool arguments with source and target.

    Returns:
        JSON string with shortest path or error if not found.

    Raises:
        ValueError: If source or target is not provided.
    """
    source = arguments.get("source")
    target = arguments.get("target")

    if not source:
        raise ValueError("source is required for path algorithm")
    if not target:
        raise ValueError("target is required for path algorithm")

    path = shortest_path(G, source, target)

    if path is None:
        # Try to find partial matches
        possible_sources = [n for n in G.nodes() if source in n]
        possible_targets = [n for n in G.nodes() if target in n]

        # Check if nodes exist at all
        source_exists = source in G.nodes() or len(possible_sources) > 0
        target_exists = target in G.nodes() or len(possible_targets) > 0

        # Build helpful explanation (E6)
        if not source_exists and not target_exists:
            explanation = (
                "Neither source nor target were found in the dependency graph. "
                "Check the node IDs using analyze_deps to see available nodes."
            )
        elif not source_exists:
            explanation = (
                f"Source node '{source}' not found in the graph. "
                "Check the node ID or use a partial match from possible_source_matches."
            )
        elif not target_exists:
            explanation = (
                f"Target node '{target}' not found in the graph. "
                "Check the node ID or use a partial match from possible_target_matches."
            )
        else:
            explanation = (
                "No direct import path found between source and target. "
                "The dependency graph tracks imports, not inheritance or runtime calls. "
                "The two nodes exist but are not connected through import relationships."
            )

        # Suggest alternative approaches
        suggestions = [
            "Use 'semantic_analysis mode=similar' to find conceptually related code between these files",
            "Try 'get_impact fusion_mode=true' to combine graph proximity with semantic similarity",
            "Check if there's an indirect connection through a common dependency",
        ]

        return json.dumps(
            {
                "path": None,
                "explanation": explanation,
                "suggestions": suggestions,
                "possible_source_matches": possible_sources[:5],
                "possible_target_matches": possible_targets[:5],
            },
            indent=2,
        )

    return json.dumps(
        {
            "path": [short_name(node) for node in path],
            "full_path": path,
            "length": len(path) - 1,
        },
        indent=2,
    )


def _classify_unknown_node(
    node: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:
    """Classify an 'unknown' hierarchy node into [stdlib] and [external] nodes.

    Args:
        node: A hierarchy node with id="unknown" containing child node IDs.
        level: The hierarchy level (package, module, class, function).

    Returns:
        List of classified nodes: one for [stdlib], one for [external], or both.
    """
    stdlib_children: list[str] = []
    external_children: list[str] = []

    for child_id in node.get("children", []):
        # Extract root module from node ID (handles "py:module.submod" format)
        clean_id = child_id
        if ":" in child_id:
            clean_id = child_id.split(":", 1)[1]
        root_module = clean_id.split(".")[0]

        if root_module in STDLIB_MODULES:
            stdlib_children.append(child_id)
        else:
            external_children.append(child_id)

    result: list[dict[str, Any]] = []

    if stdlib_children:
        result.append({
            "id": "[stdlib]",
            "level": level,
            "child_count": len(stdlib_children),
            "children": stdlib_children[:10],
        })

    if external_children:
        result.append({
            "id": "[external]",
            "level": level,
            "child_count": len(external_children),
            "children": external_children[:10],
        })

    return result


def _run_hierarchy(G, arguments: dict[str, Any]) -> str:
    """Run hierarchical view for collapsed graph visualization.

    Args:
        G: NetworkX directed graph.
        arguments: Tool arguments with level (package, module, class, function),
                   summary_only for concise output, exclude_external to filter.

    Returns:
        JSON string with aggregated nodes and edges at the specified level.
        When summary_only=true, returns concise summary with top 5 nodes.
        Nodes without file paths are classified as [stdlib] or [external].
    """
    level = arguments.get("level", "module")
    summary_only = arguments.get("summary_only", False)
    exclude_external = arguments.get("exclude_external", False)

    nodes, edges = hierarchical_view(G, level=level)

    # Classify "unknown" nodes into [stdlib] and [external]
    stdlib_count = 0
    external_count = 0
    classified_nodes: list[dict[str, Any]] = []

    for node in nodes:
        if node["id"] == "unknown":
            classified = _classify_unknown_node(node, level)
            for cn in classified:
                if cn["id"] == "[stdlib]":
                    stdlib_count = cn["child_count"]
                elif cn["id"] == "[external]":
                    external_count = cn["child_count"]
            classified_nodes.extend(classified)
        else:
            classified_nodes.append(node)

    # Filter external/stdlib if requested
    external_ids = {"[stdlib]", "[external]", "unknown"}
    if exclude_external:
        classified_nodes = [n for n in classified_nodes if n["id"] not in external_ids]
        edges = [
            e for e in edges
            if e["source"] not in external_ids and e["target"] not in external_ids
        ]

    nodes = classified_nodes

    # Sort nodes by child_count descending
    nodes.sort(key=lambda x: x["child_count"], reverse=True)

    # Summary mode (E7): return concise output
    if summary_only:
        top_nodes = []
        for n in nodes[:5]:
            top_nodes.append({
                "id": n["id"],
                "child_count": n["child_count"],
            })

        return json.dumps(
            {
                "summary": {
                    "level": level,
                    "total_hierarchy_nodes": len(nodes),
                    "total_hierarchy_edges": len(edges),
                    "compression_ratio": round(len(nodes) / max(G.number_of_nodes(), 1), 4),
                    "stdlib_count": stdlib_count,
                    "external_count": external_count,
                },
                "top_nodes": top_nodes,
                "truncated": True,
                "filter_applied": {"exclude_external": exclude_external},
                "message": f"Showing top 5 of {len(nodes)} {level}s. Use summary_only=false for full results.",
            },
            indent=2,
        )

    return json.dumps(
        {
            "level": level,
            "hierarchy_nodes": nodes[:50],  # Limit for display
            "hierarchy_edges": edges[:100],  # Limit for display
            "total_hierarchy_nodes": len(nodes),
            "total_hierarchy_edges": len(edges),
            "original_nodes": G.number_of_nodes(),
            "original_edges": G.number_of_edges(),
            "compression_ratio": round(len(nodes) / max(G.number_of_nodes(), 1), 4),
            "stdlib_count": stdlib_count,
            "external_count": external_count,
            "filter_applied": {"exclude_external": exclude_external},
        },
        indent=2,
    )


def _run_layers(G, arguments: dict[str, Any]) -> str:
    """Run layer validation for architectural compliance.

    Args:
        G: NetworkX directed graph.
        arguments: Tool arguments with layers and layer_names.

    Returns:
        JSON string with layer violations and compliance metrics.

    Raises:
        ValueError: If layers parameter is not provided.
    """
    layers_arg = arguments.get("layers")
    layer_names = arguments.get("layer_names")

    if not layers_arg:
        raise ValueError(
            "layers is required: list of lists/sets of node patterns, "
            "ordered from top (API) to bottom (infrastructure). "
            "Example: [[\"api/\", \"routes/\"], [\"services/\"], [\"models/\", \"db/\"]]"
        )

    # Convert to list of sets
    layers = [set(layer) if isinstance(layer, list) else layer for layer in layers_arg]

    violations, valid_count, total_count = validate_layers(G, layers, layer_names)

    # Format violations for display
    formatted_violations = []
    for v in violations[:30]:  # Limit for display
        formatted_violations.append({
            "source": short_name(v["source_node"]),
            "target": short_name(v["target_node"]),
            "source_full": v["source_node"],
            "target_full": v["target_node"],
            "direction": f"{v['source_layer_name']} → {v['target_layer_name']}",
            "edge_type": v["edge_type"],
            "severity": v["severity"],
        })

    compliance_rate = valid_count / total_count if total_count > 0 else 1.0

    # Determine status
    if len(violations) == 0:
        status = "clean"
        message = "No layer violations found. Architecture is compliant!"
    elif compliance_rate >= 0.9:
        status = "warning"
        message = f"Minor issues: {len(violations)} violations found, but {compliance_rate:.1%} compliant."
    else:
        status = "error"
        message = f"Architecture issues: {len(violations)} violations, only {compliance_rate:.1%} compliant."

    return json.dumps(
        {
            "status": status,
            "message": message,
            "layers": layer_names or [f"layer_{i}" for i in range(len(layers))],
            "violations": formatted_violations,
            "total_violations": len(violations),
            "valid_edges": valid_count,
            "total_classified_edges": total_count,
            "compliance_rate": round(compliance_rate, 4),
        },
        indent=2,
    )
