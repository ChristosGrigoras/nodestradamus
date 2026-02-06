"""Graph algorithms for code intelligence.

Provides algorithms to analyze dependency graphs and extract insights:
- PageRank for importance ranking
- Betweenness centrality for bottleneck detection
- Community detection for module clustering
- Cycle detection for circular dependencies
- Shortest path for dependency chains
- Strongly connected components for coupled modules

Uses Rust implementation when available for performance,
with automatic fallback to NetworkX.
"""

import networkx as nx

# Try to import Rust extension for accelerated graph algorithms
try:
    import nodestradamus_graph as _rust

    _HAS_RUST = True
except ImportError:
    _rust = None  # type: ignore[assignment]
    _HAS_RUST = False


def _edges_from_graph(G: nx.DiGraph) -> list[tuple[str, str]]:
    """Extract edge list for Rust functions.

    Args:
        G: NetworkX directed graph.

    Returns:
        List of (source, target) string tuples.
    """
    return [(str(u), str(v)) for u, v in G.edges()]


def pagerank(
    G: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
) -> dict[str, float]:
    """Rank nodes by importance (most depended upon).

    Higher scores indicate code that many other parts depend on.
    These are critical modules that require careful change management.

    Args:
        G: Directed dependency graph.
        alpha: Damping factor (default 0.85).
        max_iter: Maximum iterations for convergence.

    Returns:
        Dict mapping node ID to importance score (0-1).
    """
    if not G.nodes():
        return {}

    if _HAS_RUST:
        return _rust.pagerank(_edges_from_graph(G), alpha, max_iter)

    return nx.pagerank(G, alpha=alpha, max_iter=max_iter)


def betweenness(
    G: nx.DiGraph,
    normalized: bool = True,
) -> dict[str, float]:
    """Find bottleneck nodes that sit on many dependency paths.

    High betweenness nodes are change ripple risks - modifying them
    affects many dependency chains.

    Uses Rust implementation when available for O(nm) performance,
    with automatic fallback to NetworkX.

    Args:
        G: Directed dependency graph.
        normalized: Normalize scores to 0-1 range.

    Returns:
        Dict mapping node ID to betweenness score.
    """
    if not G.nodes():
        return {}

    if _HAS_RUST:
        return _rust.betweenness(_edges_from_graph(G), normalized)

    return nx.betweenness_centrality(G, normalized=normalized)


def detect_communities(G: nx.Graph | nx.DiGraph) -> list[set[str]]:
    """Cluster related code into modules using Louvain algorithm.

    Identifies groups of files/symbols that are tightly connected,
    revealing architectural boundaries.

    Args:
        G: Dependency graph (converted to undirected for analysis).

    Returns:
        List of sets, each set containing node IDs in a community.
    """
    if not G.nodes():
        return []

    # Convert to undirected for community detection
    undirected = G.to_undirected() if G.is_directed() else G

    # Louvain algorithm returns generator of sets
    communities = nx.community.louvain_communities(undirected)

    return list(communities)


def community_metagraph(
    G: nx.DiGraph,
    communities: list[set[str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build meta-graph showing inter-community relationships and metrics.

    Computes cohesion/coupling metrics for each community and identifies
    edges between communities to reveal architectural dependencies.

    Args:
        G: Directed dependency graph.
        communities: Pre-computed communities (optional). If None, will detect.

    Returns:
        Tuple of (community_metrics, inter_community_edges) where:
        - community_metrics: List of dicts with cohesion/coupling per community
        - inter_community_edges: List of dicts with edges between communities
    """
    if not G.nodes():
        return [], []

    # Detect communities if not provided
    if communities is None:
        communities = detect_communities(G)

    if not communities:
        return [], []

    # Build node -> community_id mapping
    node_to_community: dict[str, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Count edges: internal (within community) and external (between communities)
    # Also track afferent (incoming) and efferent (outgoing) per community
    community_ids = range(len(communities))
    internal_edges: dict[int, int] = dict.fromkeys(community_ids, 0)
    afferent_coupling: dict[int, int] = dict.fromkeys(community_ids, 0)
    efferent_coupling: dict[int, int] = dict.fromkeys(community_ids, 0)

    # Track inter-community edges: (source_community, target_community) -> count
    inter_edges: dict[tuple[int, int], int] = {}

    for source, target in G.edges():
        source_comm = node_to_community.get(source)
        target_comm = node_to_community.get(target)

        # Skip edges involving nodes not in any community
        if source_comm is None or target_comm is None:
            continue

        if source_comm == target_comm:
            # Internal edge
            internal_edges[source_comm] += 1
        else:
            # External edge - counts as efferent for source, afferent for target
            efferent_coupling[source_comm] += 1
            afferent_coupling[target_comm] += 1

            # Track inter-community edge
            edge_key = (source_comm, target_comm)
            inter_edges[edge_key] = inter_edges.get(edge_key, 0) + 1

    # Compute metrics for each community
    community_metrics = []
    for idx, community in enumerate(communities):
        internal = internal_edges[idx]
        afferent = afferent_coupling[idx]
        efferent = efferent_coupling[idx]
        external = afferent + efferent
        total = internal + external

        # Cohesion: ratio of internal to total edges (1.0 = fully cohesive)
        cohesion = internal / total if total > 0 else 1.0

        # Instability: ratio of efferent to total coupling (1.0 = fully unstable)
        total_coupling = afferent + efferent
        instability = efferent / total_coupling if total_coupling > 0 else 0.0

        community_metrics.append({
            "module_id": idx,
            "size": len(community),
            "internal_edges": internal,
            "external_edges": external,
            "cohesion": round(cohesion, 4),
            "afferent_coupling": afferent,
            "efferent_coupling": efferent,
            "instability": round(instability, 4),
        })

    # Format inter-community edges
    inter_community_edges = [
        {"source": src, "target": tgt, "edge_count": count}
        for (src, tgt), count in sorted(inter_edges.items(), key=lambda x: -x[1])
    ]

    return community_metrics, inter_community_edges


def find_cycles(
    G: nx.DiGraph,
    cross_file_only: bool = True,
) -> list[list[str]]:
    """Find circular dependencies in the graph.

    Circular dependencies can cause import issues and indicate
    poor module boundaries.

    By default, filters out intra-file cycles (e.g., file ↔ symbol within
    that file) which are not real import cycles. Set cross_file_only=False
    to include all cycles.

    Args:
        G: Directed dependency graph with optional 'file' attribute on nodes.
        cross_file_only: If True (default), only return cycles involving
                         nodes from different files.

    Returns:
        List of cycles, each cycle is a list of node IDs.
    """
    if not G.nodes():
        return []

    try:
        # simple_cycles finds all elementary cycles
        all_cycles = list(nx.simple_cycles(G))
    except nx.NetworkXNoCycle:
        return []

    if not cross_file_only:
        all_cycles.sort(key=len)
        return all_cycles

    # Filter to only cross-file cycles (real import cycles)
    cross_file_cycles = []
    for cycle in all_cycles:
        files_in_cycle = set()
        for node in cycle:
            # Get file from node attributes
            file_path = G.nodes[node].get("file", "")
            if file_path:
                files_in_cycle.add(file_path)
            else:
                # Fallback: extract file from node ID
                # Node IDs like "py:path/file.py::symbol" or "py:path/file.py"
                file_path = _extract_file_from_node_id(node)
                if file_path:
                    files_in_cycle.add(file_path)

        # Only include cycles with nodes from multiple files
        if len(files_in_cycle) >= 2:
            cross_file_cycles.append(cycle)

    # Sort by length for easier reading
    cross_file_cycles.sort(key=len)
    return cross_file_cycles


def _extract_file_from_node_id(node_id: str) -> str | None:
    """Extract file path from a node ID.

    Node ID formats:
    - "py:path/to/file.py" (file node)
    - "py:path/to/file.py::SymbolName" (symbol node)
    - "ts:path/to/file.ts::ClassName" (TypeScript symbol)

    Args:
        node_id: The node identifier.

    Returns:
        The file path, or None if not extractable.
    """
    # Remove language prefix
    if node_id.startswith(("py:", "ts:")):
        node_id = node_id[3:]

    # Split on "::" to separate file from symbol
    if "::" in node_id:
        return node_id.split("::")[0]

    # Assume it's a file node if no "::"
    return node_id if node_id else None


def shortest_path(
    G: nx.DiGraph,
    source: str,
    target: str,
) -> list[str] | None:
    """Find the shortest dependency chain between two nodes.

    Useful for understanding how one module depends on another.

    Args:
        G: Directed dependency graph.
        source: Starting node ID.
        target: Ending node ID.

    Returns:
        List of node IDs forming the path, or None if no path exists.
    """
    if source not in G or target not in G:
        return None

    try:
        return nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return None


def strongly_connected(G: nx.DiGraph) -> list[set[str]]:
    """Find tightly coupled module groups.

    Strongly connected components are groups where every node
    can reach every other node - indicating high coupling.

    Args:
        G: Directed dependency graph.

    Returns:
        List of sets containing node IDs in each component.
        Only returns components with 2+ nodes (actual coupling).
    """
    if not G.nodes():
        return []

    if _HAS_RUST:
        # Rust returns list of lists, convert to sets
        sccs = _rust.strongly_connected(_edges_from_graph(G))
        return [set(scc) for scc in sccs]

    # Get all SCCs and filter to those with multiple nodes
    sccs = nx.strongly_connected_components(G)
    coupled = [scc for scc in sccs if len(scc) > 1]

    # Sort by size (largest first)
    coupled.sort(key=len, reverse=True)

    return coupled


def top_n(
    scores: dict[str, float],
    n: int = 10,
    descending: bool = True,
) -> list[tuple[str, float]]:
    """Get top N items from a score dictionary.

    Helper function for ranking results.

    Args:
        scores: Dict mapping node ID to score.
        n: Number of items to return.
        descending: Sort descending (highest first).

    Returns:
        List of (node_id, score) tuples.
    """
    sorted_items = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=descending,
    )
    return sorted_items[:n]


def hierarchical_view(
    G: nx.DiGraph,
    level: str = "module",
) -> tuple[list[dict], list[dict]]:
    """Build a collapsed hierarchical view of the dependency graph.

    Aggregates nodes and edges at the specified hierarchy level:
    - "package": Directory-level view (e.g., "src/api/", "src/models/")
    - "module": File-level view (e.g., "src/api/users.py")
    - "class": Class-level view (includes standalone functions)
    - "function": Full detail (no aggregation)

    Args:
        G: Directed dependency graph with 'file', 'type', 'name' attributes.
        level: Hierarchy level to collapse to. Options: package, module, class, function.

    Returns:
        Tuple of (hierarchy_nodes, hierarchy_edges) where:
        - hierarchy_nodes: List of dicts with aggregated node info
        - hierarchy_edges: List of dicts with aggregated edge counts
    """
    if not G.nodes():
        return [], []

    valid_levels = {"package", "module", "class", "function"}
    if level not in valid_levels:
        raise ValueError(f"Invalid level '{level}'. Must be one of: {valid_levels}")

    if level == "function":
        # No aggregation needed - return original structure
        return _nodes_to_hierarchy_format(G), _edges_to_hierarchy_format(G)

    # Build node -> hierarchy_id mapping based on level
    node_to_hierarchy: dict[str, str] = {}
    hierarchy_children: dict[str, list[str]] = {}

    for node_id, attrs in G.nodes(data=True):
        file_path = attrs.get("file", "")
        node_type = attrs.get("type", "")

        if level == "package":
            # Aggregate to directory level
            hierarchy_id = _get_package_from_path(file_path) if file_path else "unknown"
        elif level == "module":
            # Aggregate to file level
            hierarchy_id = file_path if file_path else "unknown"
        else:  # level == "class"
            # Aggregate to class level (functions stay separate)
            if node_type in ("class", "module", "file"):
                hierarchy_id = node_id
            elif node_type in ("method", "function"):
                # Check if it's a method (has class parent) or standalone
                parent = _find_class_parent(G, node_id, file_path)
                hierarchy_id = parent if parent else node_id
            else:
                hierarchy_id = node_id

        node_to_hierarchy[node_id] = hierarchy_id
        if hierarchy_id not in hierarchy_children:
            hierarchy_children[hierarchy_id] = []
        hierarchy_children[hierarchy_id].append(node_id)

    # Build aggregated nodes
    hierarchy_nodes = []
    for hierarchy_id, children in hierarchy_children.items():
        hierarchy_nodes.append({
            "id": hierarchy_id,
            "level": level,
            "child_count": len(children),
            "children": children[:10],  # Limit for display
        })

    # Build aggregated edges
    edge_counts: dict[tuple[str, str], dict[str, int]] = {}
    for source, target, attrs in G.edges(data=True):
        source_hierarchy = node_to_hierarchy.get(source, source)
        target_hierarchy = node_to_hierarchy.get(target, target)

        # Skip self-loops in aggregated view
        if source_hierarchy == target_hierarchy:
            continue

        edge_key = (source_hierarchy, target_hierarchy)
        if edge_key not in edge_counts:
            edge_counts[edge_key] = {}

        edge_type = attrs.get("type", "unknown")
        edge_counts[edge_key][edge_type] = edge_counts[edge_key].get(edge_type, 0) + 1

    hierarchy_edges = []
    for (source, target), type_counts in sorted(
        edge_counts.items(), key=lambda x: -sum(x[1].values())
    ):
        hierarchy_edges.append({
            "source": source,
            "target": target,
            "edge_count": sum(type_counts.values()),
            "edge_types": type_counts,
        })

    return hierarchy_nodes, hierarchy_edges


def _get_package_from_path(file_path: str) -> str:
    """Extract package/directory from file path.

    Args:
        file_path: File path like "src/api/users.py"

    Returns:
        Directory path like "src/api/"
    """
    if not file_path:
        return "unknown"

    # Split and take parent directory
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) <= 1:
        return "/"
    return "/".join(parts[:-1]) + "/"


def _find_class_parent(G: nx.DiGraph, node_id: str, file_path: str) -> str | None:
    """Find the class that contains a method.

    Args:
        G: Dependency graph.
        node_id: The node ID to find parent for.
        file_path: File path of the node.

    Returns:
        Class node ID if found, None otherwise.
    """
    # Look for incoming "contains" edges or same-file class nodes
    for pred in G.predecessors(node_id):
        pred_attrs = G.nodes.get(pred, {})
        if pred_attrs.get("type") == "class" and pred_attrs.get("file") == file_path:
            return pred
    return None


def _nodes_to_hierarchy_format(G: nx.DiGraph) -> list[dict]:
    """Convert graph nodes to hierarchy format (no aggregation)."""
    return [
        {
            "id": node_id,
            "level": "function",
            "child_count": 1,
            "children": [node_id],
        }
        for node_id in G.nodes()
    ]


def _edges_to_hierarchy_format(G: nx.DiGraph) -> list[dict]:
    """Convert graph edges to hierarchy format (no aggregation)."""
    return [
        {
            "source": source,
            "target": target,
            "edge_count": 1,
            "edge_types": {attrs.get("type", "unknown"): 1},
        }
        for source, target, attrs in G.edges(data=True)
    ]


def validate_layers(
    G: nx.DiGraph,
    layers: list[set[str] | list[str]],
    layer_names: list[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Validate that dependencies respect layer ordering.

    In a layered architecture, upper layers can depend on lower layers,
    but lower layers should NOT depend on upper layers.

    Layer ordering: layers[0] is the top (e.g., API/presentation),
    layers[-1] is the bottom (e.g., domain/infrastructure).

    Args:
        G: Directed dependency graph.
        layers: Ordered list of sets, each containing node IDs or path patterns
                in that layer. layers[0] = top, can depend on any lower layer.
        layer_names: Optional names for each layer (for reporting).

    Returns:
        Tuple of (violations, valid_edge_count, total_classified_edges) where:
        - violations: List of dicts describing each violation
        - valid_edge_count: Edges respecting layer rules
        - total_classified_edges: Total edges between classified nodes
    """
    if not G.nodes():
        return [], 0, 0

    if not layers:
        return [], 0, 0

    # Normalize layers to sets
    normalized_layers: list[set[str]] = [
        set(layer) if isinstance(layer, list) else layer for layer in layers
    ]

    # Build node -> layer_index mapping
    # Support both exact match and path prefix matching
    node_to_layer: dict[str, int] = {}

    for node_id, attrs in G.nodes(data=True):
        file_path = attrs.get("file", "")

        for layer_idx, layer_patterns in enumerate(normalized_layers):
            # Check exact match first
            if node_id in layer_patterns:
                node_to_layer[node_id] = layer_idx
                break

            # Check file path prefix match
            for pattern in layer_patterns:
                if file_path and file_path.startswith(pattern.rstrip("/")):
                    node_to_layer[node_id] = layer_idx
                    break
            else:
                continue
            break

    # Check each edge for violations
    violations = []
    valid_count = 0
    total_count = 0

    for source, target, attrs in G.edges(data=True):
        source_layer = node_to_layer.get(source)
        target_layer = node_to_layer.get(target)

        # Skip edges involving unclassified nodes
        if source_layer is None or target_layer is None:
            continue

        total_count += 1

        # Valid: source layer <= target layer (upper depends on lower)
        # Invalid: source layer > target layer (lower depends on upper)
        if source_layer > target_layer:
            # Violation: lower layer depending on upper layer
            violations.append({
                "source_node": source,
                "target_node": target,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "source_layer_name": (
                    layer_names[source_layer]
                    if layer_names and source_layer < len(layer_names)
                    else f"layer_{source_layer}"
                ),
                "target_layer_name": (
                    layer_names[target_layer]
                    if layer_names and target_layer < len(layer_names)
                    else f"layer_{target_layer}"
                ),
                "edge_type": attrs.get("type", "unknown"),
                "severity": "error",
            })
        elif source_layer == target_layer:
            # Same-layer dependency - valid but worth noting
            valid_count += 1
        else:
            # Valid cross-layer dependency
            valid_count += 1

    return violations, valid_count, total_count


def ancestors_at_depth(
    G: nx.DiGraph,
    node: str,
    max_depth: int,
) -> dict[str, int]:
    """Find all ancestors up to a maximum depth.

    Args:
        G: Directed dependency graph.
        node: Starting node ID.
        max_depth: Maximum traversal depth.

    Returns:
        Dict mapping ancestor node ID to its depth from the target.
    """
    if node not in G:
        return {}

    if _HAS_RUST:
        return _rust.ancestors_at_depth(_edges_from_graph(G), node, max_depth)

    result: dict[str, int] = {}
    current_level = {node}
    visited = {node}

    for depth in range(1, max_depth + 1):
        next_level: set[str] = set()
        for n in current_level:
            for pred in G.predecessors(n):
                if pred not in visited:
                    result[pred] = depth
                    next_level.add(pred)
                    visited.add(pred)
        current_level = next_level
        if not current_level:
            break

    return result


def descendants_at_depth(
    G: nx.DiGraph,
    node: str,
    max_depth: int,
) -> dict[str, int]:
    """Find all descendants up to a maximum depth.

    Args:
        G: Directed dependency graph.
        node: Starting node ID.
        max_depth: Maximum traversal depth.

    Returns:
        Dict mapping descendant node ID to its depth from the target.
    """
    if node not in G:
        return {}

    if _HAS_RUST:
        return _rust.descendants_at_depth(_edges_from_graph(G), node, max_depth)

    result: dict[str, int] = {}
    current_level = {node}
    visited = {node}

    for depth in range(1, max_depth + 1):
        next_level: set[str] = set()
        for n in current_level:
            for succ in G.successors(n):
                if succ not in visited:
                    result[succ] = depth
                    next_level.add(succ)
                    visited.add(succ)
        current_level = next_level
        if not current_level:
            break

    return result


# =============================================================================
# SCALE ALGORITHMS - For large codebases (50K+ files, 2M+ nodes)
# =============================================================================


def sampled_betweenness(
    G: nx.DiGraph,
    sample_size: int = 100,
    seed: int = 42,
    normalized: bool = True,
) -> dict[str, float]:
    """Approximate betweenness centrality using random sampling.

    For large graphs, full betweenness is O(n*m) which is too slow.
    This uses random source sampling to approximate in O(k*m) time.

    Args:
        G: Directed dependency graph.
        sample_size: Number of random source nodes to sample.
        seed: Random seed for reproducibility.
        normalized: Normalize scores to 0-1 range.

    Returns:
        Dict mapping node ID to approximate betweenness score.
    """
    if not G.nodes():
        return {}

    if _HAS_RUST:
        return _rust.sampled_betweenness(
            _edges_from_graph(G), sample_size, seed, normalized
        )

    # NetworkX fallback: use k parameter for approximation
    n = G.number_of_nodes()
    k = min(sample_size, n)

    if k >= n:
        return nx.betweenness_centrality(G, normalized=normalized)

    return nx.betweenness_centrality(G, k=k, seed=seed, normalized=normalized)


def batch_ancestors(
    G: nx.DiGraph,
    targets: list[str],
    max_depth: int,
) -> dict[str, dict[str, int]]:
    """Find ancestors for multiple targets in one graph traversal.

    More efficient than calling ancestors_at_depth multiple times
    because the graph is only built once.

    Args:
        G: Directed dependency graph.
        targets: List of target node IDs.
        max_depth: Maximum traversal depth.

    Returns:
        Dict mapping target ID to dict of (ancestor ID -> depth).
    """
    if not G.nodes() or not targets:
        return {}

    if _HAS_RUST:
        return _rust.batch_ancestors(_edges_from_graph(G), targets, max_depth)

    # NetworkX fallback: call ancestors_at_depth for each target
    return {
        target: ancestors_at_depth(G, target, max_depth)
        for target in targets
    }


def batch_descendants(
    G: nx.DiGraph,
    sources: list[str],
    max_depth: int,
) -> dict[str, dict[str, int]]:
    """Find descendants for multiple sources in one graph traversal.

    More efficient than calling descendants_at_depth multiple times
    because the graph is only built once.

    Args:
        G: Directed dependency graph.
        sources: List of source node IDs.
        max_depth: Maximum traversal depth.

    Returns:
        Dict mapping source ID to dict of (descendant ID -> depth).
    """
    if not G.nodes() or not sources:
        return {}

    if _HAS_RUST:
        return _rust.batch_descendants(_edges_from_graph(G), sources, max_depth)

    # NetworkX fallback: call descendants_at_depth for each source
    return {
        source: descendants_at_depth(G, source, max_depth)
        for source in sources
    }


def extract_subgraph(
    G: nx.DiGraph,
    seed_nodes: list[str],
    max_depth: int = 2,
    include_incoming: bool = True,
    include_outgoing: bool = True,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract a subgraph containing nodes within max_depth of seed nodes.

    Useful for lazy loading: load only the relevant portion of a large graph
    instead of the entire graph.

    Args:
        G: Directed dependency graph.
        seed_nodes: Starting nodes for the subgraph extraction.
        max_depth: Maximum depth from seed nodes to include.
        include_incoming: Include incoming edges (ancestors).
        include_outgoing: Include outgoing edges (descendants).

    Returns:
        Tuple of (nodes, edges) where edges is list of (src, tgt) tuples.
    """
    if not G.nodes() or not seed_nodes:
        return [], []

    if _HAS_RUST:
        return _rust.subgraph(
            _edges_from_graph(G),
            seed_nodes,
            max_depth,
            include_incoming,
            include_outgoing,
        )

    # NetworkX fallback: BFS in both directions
    relevant_nodes: set[str] = set()
    queue: list[tuple[str, int]] = []

    # Initialize with seed nodes
    for seed in seed_nodes:
        if seed in G:
            relevant_nodes.add(seed)
            queue.append((seed, 0))

    # BFS
    visited: set[str] = set(relevant_nodes)
    while queue:
        node, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        # Outgoing edges (descendants)
        if include_outgoing:
            for succ in G.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    relevant_nodes.add(succ)
                    queue.append((succ, depth + 1))

        # Incoming edges (ancestors)
        if include_incoming:
            for pred in G.predecessors(node):
                if pred not in visited:
                    visited.add(pred)
                    relevant_nodes.add(pred)
                    queue.append((pred, depth + 1))

    # Collect edges where both endpoints are in relevant_nodes
    subgraph_edges = [
        (str(u), str(v))
        for u, v in G.edges()
        if u in relevant_nodes and v in relevant_nodes
    ]

    return list(relevant_nodes), subgraph_edges


class LazyEmbeddingGraph:
    """Lazy wrapper for graph + embeddings that loads on-demand.

    For large codebases, building the full graph and computing all
    embeddings upfront is expensive. This class enables:

    1. Load only a subgraph relevant to the current query
    2. Compute embeddings only for nodes in the subgraph
    3. Expand the loaded portion as needed

    Example usage:
        >>> lazy = LazyEmbeddingGraph(repo_path)
        >>> # Load subgraph around auth module
        >>> lazy.load_scope("src/auth/")
        >>> # Query only affects the loaded scope
        >>> results = lazy.find_similar("login function")
        >>> # Expand to include related modules
        >>> lazy.expand_from_nodes(["src/utils/crypto.py"])
    """

    def __init__(
        self,
        repo_path: str,
        workspace_path: str | None = None,
    ):
        """Initialize lazy loader.

        Args:
            repo_path: Path to the repository.
            workspace_path: Workspace path for isolated caching.
        """
        self._repo_path = repo_path
        self._workspace_path = workspace_path
        self._full_graph: nx.DiGraph | None = None
        self._loaded_nodes: set[str] = set()
        self._subgraph: nx.DiGraph | None = None
        self._embeddings_scope: str | None = None
        self._embeddings_loaded = False

    @property
    def repo_path(self) -> str:
        """Get the repository path."""
        return self._repo_path

    @property
    def workspace_path(self) -> str | None:
        """Get the workspace path."""
        return self._workspace_path

    @property
    def loaded_nodes(self) -> set[str]:
        """Get the set of currently loaded node IDs."""
        return self._loaded_nodes.copy()

    @property
    def is_loaded(self) -> bool:
        """Check if any data has been loaded."""
        return self._subgraph is not None and len(self._loaded_nodes) > 0

    def _ensure_full_graph(self) -> nx.DiGraph:
        """Load the full graph if not already loaded.

        Returns:
            The full dependency graph.
        """
        if self._full_graph is None:
            from nodestradamus.analyzers.deps import analyze_deps

            result = analyze_deps(
                self._repo_path,
                workspace_path=self._workspace_path,
            )
            self._full_graph = result.graph
        return self._full_graph

    def load_scope(self, scope: str) -> dict:
        """Load nodes matching a path prefix (scope).

        Args:
            scope: Path prefix to filter nodes (e.g., "src/auth/").

        Returns:
            Dict with load statistics.
        """
        G = self._ensure_full_graph()

        # Find nodes matching scope
        matching_nodes = [
            node for node in G.nodes()
            if str(node).startswith(scope) or scope in str(node)
        ]

        if not matching_nodes:
            return {"loaded": 0, "scope": scope}

        # Load subgraph around matching nodes
        nodes, edges = extract_subgraph(
            G,
            matching_nodes,
            max_depth=1,  # Include immediate neighbors
            include_incoming=True,
            include_outgoing=True,
        )

        # Update subgraph
        if self._subgraph is None:
            self._subgraph = nx.DiGraph()

        self._subgraph.add_nodes_from(nodes)
        self._subgraph.add_edges_from(edges)
        self._loaded_nodes.update(nodes)
        self._embeddings_scope = scope

        return {
            "loaded": len(nodes),
            "edges": len(edges),
            "scope": scope,
        }

    def load_from_nodes(
        self,
        seed_nodes: list[str],
        max_depth: int = 2,
    ) -> dict:
        """Load subgraph around specific seed nodes.

        Args:
            seed_nodes: Starting nodes for subgraph extraction.
            max_depth: How far to expand from seed nodes.

        Returns:
            Dict with load statistics.
        """
        G = self._ensure_full_graph()

        nodes, edges = extract_subgraph(
            G,
            seed_nodes,
            max_depth=max_depth,
            include_incoming=True,
            include_outgoing=True,
        )

        if self._subgraph is None:
            self._subgraph = nx.DiGraph()

        self._subgraph.add_nodes_from(nodes)
        self._subgraph.add_edges_from(edges)
        self._loaded_nodes.update(nodes)

        return {
            "loaded": len(nodes),
            "edges": len(edges),
            "seeds": len(seed_nodes),
        }

    def expand_from_nodes(
        self,
        seed_nodes: list[str],
        max_depth: int = 1,
    ) -> dict:
        """Expand the loaded subgraph from additional seed nodes.

        Similar to load_from_nodes but merges with existing subgraph.

        Args:
            seed_nodes: Additional nodes to expand from.
            max_depth: How far to expand.

        Returns:
            Dict with expansion statistics.
        """
        return self.load_from_nodes(seed_nodes, max_depth)

    def get_subgraph(self) -> nx.DiGraph | None:
        """Get the currently loaded subgraph.

        Returns:
            The loaded subgraph, or None if nothing loaded.
        """
        return self._subgraph

    def compute_scoped_embeddings(
        self,
        force: bool = False,
    ) -> dict:
        """Compute embeddings only for loaded scope.

        Args:
            force: Force recompute even if already loaded.

        Returns:
            Dict with embedding statistics.
        """
        if self._embeddings_loaded and not force:
            return {"status": "already_loaded", "scope": self._embeddings_scope}

        if not self._embeddings_scope and not self._loaded_nodes:
            return {"status": "no_scope", "error": "Call load_scope first"}

        from nodestradamus.analyzers.embeddings import compute_embeddings

        # Determine scope from loaded nodes
        if self._embeddings_scope:
            scope = self._embeddings_scope
        else:
            # Infer common prefix from loaded nodes
            if self._loaded_nodes:
                prefixes = [str(n).split("/")[0] for n in self._loaded_nodes]
                if prefixes:
                    scope = prefixes[0]
                else:
                    scope = None
            else:
                scope = None

        result = compute_embeddings(
            self._repo_path,
            workspace_path=self._workspace_path,
            scope=scope,
        )

        self._embeddings_loaded = True

        return {
            "status": "computed",
            "scope": scope,
            "chunks": result["metadata"].get("chunks_extracted", 0),
        }

    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Find similar code within the loaded scope.

        Args:
            query: Search query.
            top_k: Number of results.
            threshold: Minimum similarity score.

        Returns:
            List of similar code chunks.
        """
        from nodestradamus.analyzers.embeddings import find_similar_code

        # Use package/scope filtering if available
        package = self._embeddings_scope if self._embeddings_scope else None

        return find_similar_code(
            repo_path=self._repo_path,
            query=query,
            top_k=top_k,
            threshold=threshold,
            workspace_path=self._workspace_path,
            package=package,
        )

    def get_impact(
        self,
        node_id: str,
        direction: str = "both",
    ) -> dict:
        """Get impact analysis for a node within loaded subgraph.

        Args:
            node_id: The node to analyze.
            direction: "upstream", "downstream", or "both".

        Returns:
            Dict with impacted nodes.
        """
        if self._subgraph is None:
            return {"error": "No subgraph loaded"}

        if node_id not in self._subgraph:
            return {"error": f"Node {node_id} not in loaded subgraph"}

        result = {
            "node": node_id,
            "upstream": [],
            "downstream": [],
        }

        if direction in ("upstream", "both"):
            try:
                result["upstream"] = list(nx.ancestors(self._subgraph, node_id))
            except nx.NetworkXError:
                pass

        if direction in ("downstream", "both"):
            try:
                result["downstream"] = list(nx.descendants(self._subgraph, node_id))
            except nx.NetworkXError:
                pass

        return result

    def stats(self) -> dict:
        """Get statistics about the loaded graph.

        Returns:
            Dict with graph statistics.
        """
        full_nodes = 0
        full_edges = 0
        if self._full_graph is not None:
            full_nodes = self._full_graph.number_of_nodes()
            full_edges = self._full_graph.number_of_edges()

        loaded_nodes = len(self._loaded_nodes)
        loaded_edges = self._subgraph.number_of_edges() if self._subgraph else 0

        return {
            "full_graph": {
                "nodes": full_nodes,
                "edges": full_edges,
            },
            "loaded_subgraph": {
                "nodes": loaded_nodes,
                "edges": loaded_edges,
            },
            "load_ratio": loaded_nodes / full_nodes if full_nodes > 0 else 0,
            "embeddings_scope": self._embeddings_scope,
            "embeddings_loaded": self._embeddings_loaded,
        }
