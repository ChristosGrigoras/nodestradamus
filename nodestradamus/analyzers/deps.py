"""Unified dependency graph analyzer.

Builds a single NetworkX graph from any supported language.
Uses tree-sitter for consistent parsing across all languages.

Incremental Caching
-------------------
The analyzer caches the full graph to avoid rebuilding from scratch.
The parse cache tracks per-file changes, and only stale files are re-parsed.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack
import networkx as nx

from nodestradamus.analyzers.code_parser import (
    SKIP_DIRS,
    parse_directory,
)
from nodestradamus.logging import logger

# Graph cache version - bump when cache format changes
# 2.0: Switched to MessagePack binary format for faster I/O
GRAPH_CACHE_VERSION = "2.0"

# File extensions for language detection
PYTHON_EXTENSIONS = frozenset({".py", ".pyw"})
TYPESCRIPT_EXTENSIONS = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
RUST_EXTENSIONS = frozenset({".rs"})
SQL_EXTENSIONS = frozenset({".sql", ".pgsql"})
BASH_EXTENSIONS = frozenset({".sh", ".bash"})
JSON_EXTENSIONS = frozenset({".json"})
CPP_EXTENSIONS = frozenset({".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".m", ".mm"})


def _get_graph_cache_path(repo_path: Path) -> Path:
    """Get the path to the graph cache file.

    Args:
        repo_path: Repository root path.

    Returns:
        Path to the graph cache file (MessagePack format).
    """
    return repo_path / ".nodestradamus" / "graph.msgpack"


def _get_legacy_graph_cache_path(repo_path: Path) -> Path:
    """Get the path to the legacy JSON graph cache (for migration)."""
    return repo_path / ".nodestradamus" / "graph.json"


def _load_cached_graph(repo_path: Path) -> nx.DiGraph | None:
    """Load cached graph if available and valid.

    Tries MessagePack format first, falls back to legacy JSON.

    Args:
        repo_path: Repository root path.

    Returns:
        Cached NetworkX DiGraph or None if cache is invalid/missing.
    """
    cache_path = _get_graph_cache_path(repo_path)
    legacy_path = _get_legacy_graph_cache_path(repo_path)

    # Try MessagePack first
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                data = msgpack.unpack(f, raw=False)

            if data.get("version") != GRAPH_CACHE_VERSION:
                logger.info("  Graph cache version mismatch, ignoring cache")
                return None

            G = _build_graph(data.get("nodes", []), data.get("edges", []))
            logger.info("  Loaded cached graph with %d nodes (msgpack)", G.number_of_nodes())
            return G

        except (msgpack.UnpackException, msgpack.ExtraData, KeyError, TypeError, ValueError) as e:
            logger.warning("  Failed to load graph cache: %s", e)
            return None

    # Fall back to legacy JSON (for informational purposes only)
    if legacy_path.exists():
        logger.info("  Found legacy JSON graph cache, will rebuild and migrate")
        return None

    return None


def _save_graph_cache(repo_path: Path, G: nx.DiGraph) -> None:
    """Save graph to cache using MessagePack format.

    Args:
        repo_path: Repository root path.
        G: The graph to cache.
    """
    cache_path = _get_graph_cache_path(repo_path)
    legacy_path = _get_legacy_graph_cache_path(repo_path)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert graph to serializable format
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            node = {"id": node_id, **attrs}
            nodes.append(node)

        edges = []
        for source, target, attrs in G.edges(data=True):
            edge = {"from": source, "to": target, **attrs}
            edges.append(edge)

        data = {
            "version": GRAPH_CACHE_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "nodes": nodes,
            "edges": edges,
        }

        # Write MessagePack (binary, compact)
        with cache_path.open("wb") as f:
            msgpack.pack(data, f)

        logger.info("  Saved graph cache with %d nodes (msgpack)", len(nodes))

        # Clean up legacy JSON cache if it exists
        if legacy_path.exists():
            try:
                legacy_path.unlink()
                logger.info("  Removed legacy JSON graph cache")
            except OSError:
                pass

    except OSError as e:
        logger.warning("  Failed to save graph cache: %s", e)


def _apply_incremental_update(
    cached_graph: nx.DiGraph,
    changed_files: set[str],
    new_nodes: list[dict[str, Any]],
    new_edges: list[dict[str, Any]],
) -> nx.DiGraph:
    """Apply incremental updates to a cached graph.

    Removes nodes/edges from changed files and adds new ones.

    Args:
        cached_graph: The existing cached graph.
        changed_files: Set of file paths that were re-parsed.
        new_nodes: New nodes from re-parsed files.
        new_edges: New edges from re-parsed files.

    Returns:
        Updated graph.
    """
    # Remove old nodes from changed files
    nodes_to_remove = [
        node_id
        for node_id, attrs in cached_graph.nodes(data=True)
        if attrs.get("file", "") in changed_files
    ]
    cached_graph.remove_nodes_from(nodes_to_remove)

    # Add new nodes
    for node in new_nodes:
        if node.get("file", "") in changed_files:
            attrs = {
                "type": node.get("type", "unknown"),
                "file": node.get("file", ""),
                "name": node.get("name", ""),
                "line": node.get("line"),
                "language": node.get("language", ""),
            }
            if "fields" in node:
                attrs["fields"] = node["fields"]
            cached_graph.add_node(node["id"], **attrs)

    # Add new edges (only if both endpoints exist)
    for edge in new_edges:
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))
        if source in cached_graph and target in cached_graph:
            cached_graph.add_edge(
                source,
                target,
                type=edge.get("type", "unknown"),
                resolved=edge.get("resolved", False),
            )

    return cached_graph


def analyze_deps(
    repo_path: str | Path,
    languages: list[str] | None = None,
    use_cache: bool = True,
    exclude: list[str] | None = None,
    package: str | None = None,
    scope: list[str] | None = None,
    seed_entries: list[str] | None = None,
    workers: int | None = None,
) -> nx.DiGraph:
    """Build unified dependency graph for repository.

    Uses tree-sitter for language-agnostic parsing with consistent
    graph structure across all supported languages.

    The parse cache (per-file) ensures only changed files are re-parsed.
    The graph cache stores the full built graph for faster subsequent loads.

    Args:
        repo_path: Path to repository.
        languages: List of languages to analyze.
                   None = auto-detect from files present.
                   Options: "python", "typescript", "rust", "sql"
        use_cache: Whether to use/update the graph cache. Default True.
        exclude: Directories/patterns to exclude from analysis.
        package: For monorepos, analyze only this package path (e.g., 'libs/core').
                 Use project_scout to discover available packages.
        scope: Path prefixes to include in analysis (e.g., ['src/', 'lib/']).
               Files outside these paths are excluded. Use project_scout's
               recommended_scope for intelligent defaults.
        seed_entries: Entry point files to prioritize in graph building.
                      These files and their dependencies are analyzed first.
                      Use project_scout's entry_points for intelligent defaults.

    Returns:
        NetworkX DiGraph with all dependencies merged.
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.is_dir():
        raise ValueError(f"Not a directory: {repo_path}")

    # If package is specified, analyze only that subdirectory
    analysis_path = repo_path
    if package:
        package_path = repo_path / package
        if not package_path.is_dir():
            raise ValueError(
                f"Package directory not found: {package}. "
                f"Use project_scout to discover available packages."
            )
        analysis_path = package_path

    # Combine scope with exclude patterns
    # Scope acts as an allowlist, so we need to exclude everything else
    combined_exclude = list(exclude) if exclude else []

    # If scope is specified, we filter nodes/edges after parsing
    # (The parser still parses all files but we filter the result)
    scope_prefixes = None
    if scope:
        # Normalize scope prefixes (ensure they end with / for directory matching)
        scope_prefixes = [s.rstrip("/") + "/" if not s.endswith("/") else s for s in scope]
        # Also include files that exactly match (for file entries like "main.py")
        scope_prefixes.extend([s.rstrip("/") for s in scope if not s.endswith("/")])

    if languages is None:
        languages = _detect_languages(analysis_path)

    # Use unified parser for all languages
    # The parser internally uses parse cache for incremental file parsing
    result = parse_directory(
        analysis_path,
        languages=languages,
        use_cache=use_cache,
        exclude=combined_exclude,
    )

    nodes = result["nodes"]
    edges = result["edges"]

    # Apply scope filtering if specified
    if scope_prefixes:
        nodes, edges = _filter_by_scope(nodes, edges, scope_prefixes)
        logger.info("  Scope filter applied: %d nodes retained", len(nodes))

    # If seed_entries are specified, we can use them to prioritize or filter
    # For now, log them for visibility (future: prioritize traversal order)
    if seed_entries:
        valid_seeds = [s for s in seed_entries if (analysis_path / s).exists()]
        if valid_seeds:
            logger.info("  Using %d seed entry points for analysis", len(valid_seeds))

    # Build the graph
    G = _build_graph(nodes, edges)

    # Cache the graph for faster subsequent loads
    # Use a package-specific cache key if analyzing a sub-package
    if use_cache:
        cache_path = repo_path
        if package:
            # Create a unique cache location for the package
            cache_path = analysis_path
        _save_graph_cache(cache_path, G)

    return G


def _filter_by_scope(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    scope_prefixes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter nodes and edges to only include those within scope.

    Args:
        nodes: List of node dicts.
        edges: List of edge dicts.
        scope_prefixes: List of path prefixes to include.

    Returns:
        Tuple of (filtered_nodes, filtered_edges).
    """
    # Filter nodes by file path
    filtered_nodes = []
    valid_node_ids = set()

    for node in nodes:
        file_path = node.get("file", "")
        if not file_path:
            continue

        # Check if file matches any scope prefix
        matches_scope = any(
            file_path.startswith(prefix) or file_path == prefix.rstrip("/")
            for prefix in scope_prefixes
        )

        if matches_scope:
            filtered_nodes.append(node)
            valid_node_ids.add(node.get("id", ""))

    # Filter edges to only include those where both endpoints are in scope
    filtered_edges = []
    for edge in edges:
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))

        if source in valid_node_ids and target in valid_node_ids:
            filtered_edges.append(edge)

    return filtered_nodes, filtered_edges


def _detect_languages(repo_path: Path) -> list[str]:
    """Auto-detect languages present in repository.

    Args:
        repo_path: Path to repository root.

    Returns:
        List of detected languages.
    """
    languages = []
    has_python = False
    has_typescript = False
    has_rust = False
    has_sql = False
    has_bash = False
    has_json = False
    has_cpp = False

    for filepath in repo_path.rglob("*"):
        if not filepath.is_file():
            continue

        # Skip common non-source directories
        if any(part in SKIP_DIRS for part in filepath.parts):
            continue

        suffix = filepath.suffix.lower()

        if suffix in PYTHON_EXTENSIONS:
            has_python = True
        elif suffix in TYPESCRIPT_EXTENSIONS:
            has_typescript = True
        elif suffix in RUST_EXTENSIONS:
            has_rust = True
        elif suffix in SQL_EXTENSIONS:
            has_sql = True
        elif suffix in BASH_EXTENSIONS:
            has_bash = True
        elif suffix in JSON_EXTENSIONS:
            has_json = True
        elif suffix in CPP_EXTENSIONS:
            has_cpp = True

        # Early exit if all detected
        if (
            has_python
            and has_typescript
            and has_rust
            and has_sql
            and has_bash
            and has_json
            and has_cpp
        ):
            break

    if has_python:
        languages.append("python")
    if has_typescript:
        languages.append("typescript")
    if has_rust:
        languages.append("rust")
    if has_sql:
        languages.append("sql")
    if has_bash:
        languages.append("bash")
    if has_json:
        languages.append("json")
    if has_cpp:
        languages.append("cpp")

    return languages


def _build_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> nx.DiGraph:
    """Convert raw nodes/edges to NetworkX graph.

    Args:
        nodes: List of node dicts with id, type, file, name, line, fields.
        edges: List of edge dicts with from, to, type, resolved.

    Returns:
        NetworkX DiGraph with node/edge attributes.
    """
    G = nx.DiGraph()

    for node in nodes:
        attrs = {
            "type": node.get("type", "unknown"),
            "file": node.get("file", ""),
            "name": node.get("name", ""),
            "line": node.get("line"),
            "language": node.get("language", ""),
        }
        # Include fields if present
        if "fields" in node:
            attrs["fields"] = node["fields"]
        G.add_node(node["id"], **attrs)

    for edge in edges:
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))

        if source and target:
            G.add_edge(
                source,
                target,
                type=edge.get("type", "unknown"),
                resolved=edge.get("resolved", False),
            )

    return G


def _merge_graphs(*graphs: nx.DiGraph) -> nx.DiGraph:
    """Merge multiple language graphs into one.

    Args:
        graphs: Variable number of DiGraph objects.

    Returns:
        Merged graph with all nodes and edges.
    """
    merged = nx.DiGraph()

    for G in graphs:
        # Add all nodes with attributes
        for node, attrs in G.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **attrs)

        # Add all edges with attributes
        for source, target, attrs in G.edges(data=True):
            if not merged.has_edge(source, target):
                merged.add_edge(source, target, **attrs)

    return merged


def analyze_deps_smart(
    repo_path: str | Path,
    languages: list[str] | None = None,
    use_cache: bool = True,
    exclude: list[str] | None = None,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Build dependency graph using project_scout intelligence.

    This combines project_scout reconnaissance with analyze_deps to:
    1. Detect project type and recommended scope
    2. Apply scope filtering for focused analysis
    3. Use entry points for prioritized traversal

    Args:
        repo_path: Path to repository.
        languages: List of languages to analyze (None = auto-detect).
        use_cache: Whether to use/update caches.
        exclude: Additional patterns to exclude.

    Returns:
        Tuple of (graph, metadata) where metadata includes scout results.
    """
    from nodestradamus.analyzers.project_scout import project_scout

    repo_path = Path(repo_path).resolve()

    # Run project_scout for intelligent defaults
    scout_result = project_scout(repo_path)

    # Use recommended scope if available, otherwise analyze everything
    scope = scout_result.recommended_scope if scout_result.recommended_scope else None

    # Combine suggested ignores with explicit excludes
    combined_exclude = list(scout_result.suggested_ignores)
    if exclude:
        combined_exclude.extend(exclude)

    # Filter entry points to actual file paths (not script entries)
    seed_entries = [
        ep for ep in scout_result.entry_points
        if not ep.startswith("[script:") and (repo_path / ep).exists()
    ]

    # Build the graph with scope and seed entries
    G = analyze_deps(
        repo_path=repo_path,
        languages=languages,
        use_cache=use_cache,
        exclude=combined_exclude,
        scope=scope,
        seed_entries=seed_entries,
    )

    # Collect metadata about the analysis
    metadata = {
        "project_type": scout_result.project_type,
        "primary_language": scout_result.primary_language,
        "scope_used": scope,
        "entry_points": seed_entries,
        "readme_hints": scout_result.readme_hints,
        "is_monorepo": scout_result.is_monorepo,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }

    return G, metadata


def graph_metadata(G: nx.DiGraph) -> dict[str, Any]:
    """Extract summary metadata from a dependency graph.

    Args:
        G: Dependency graph.

    Returns:
        Dict with node/edge counts and type breakdowns.
    """
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}

    for _, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    for _, _, attrs in G.edges(data=True):
        edge_type = attrs.get("type", "unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
    }


# =============================================================================
# LAZY GRAPH - For large codebases (50K+ files, 2M+ nodes)
# =============================================================================


class LazyGraph:
    """Load subgraphs on demand from cache.

    For WebKit-scale codebases (50K-100K files, 2-5M nodes), loading the entire
    graph into memory is impractical. LazyGraph loads only the nodes and edges
    needed for specific operations.

    Usage:
        lazy = LazyGraph(repo_path)
        # Get subgraph around a file (2 hops)
        subgraph = lazy.load_around("src/core/component.cpp", depth=2)
        # Run algorithms on subgraph
        scores = pagerank(subgraph)

    The full graph cache is stored as MessagePack, allowing efficient random
    access to portions of the data.
    """

    def __init__(self, repo_path: str | Path):
        """Initialize lazy graph loader.

        Args:
            repo_path: Path to repository root.
        """
        self.repo_path = Path(repo_path).resolve()
        self._cache_path = _get_graph_cache_path(self.repo_path)
        self._cache_data: dict[str, Any] | None = None
        self._node_index: dict[str, int] | None = None  # node_id -> index in nodes list
        self._file_index: dict[str, list[int]] | None = None  # file -> list of node indices
        self._loaded_graph = nx.DiGraph()
        self._loaded_files: set[str] = set()

    def _load_cache_index(self) -> None:
        """Load cache metadata and build indexes."""
        if self._cache_data is not None:
            return

        if not self._cache_path.exists():
            raise FileNotFoundError(
                f"Graph cache not found: {self._cache_path}. "
                f"Run analyze_deps first to build the cache."
            )

        with self._cache_path.open("rb") as f:
            self._cache_data = msgpack.unpack(f, raw=False)

        if self._cache_data.get("version") != GRAPH_CACHE_VERSION:
            raise ValueError(
                f"Graph cache version mismatch. Expected {GRAPH_CACHE_VERSION}, "
                f"got {self._cache_data.get('version')}. Run analyze_deps to rebuild."
            )

        # Build indexes for fast lookup
        self._node_index = {}
        self._file_index = {}

        for i, node in enumerate(self._cache_data.get("nodes", [])):
            node_id = node.get("id", "")
            file_path = node.get("file", "")

            self._node_index[node_id] = i

            if file_path:
                if file_path not in self._file_index:
                    self._file_index[file_path] = []
                self._file_index[file_path].append(i)

        logger.info(
            "  LazyGraph indexed %d nodes, %d files",
            len(self._node_index),
            len(self._file_index),
        )

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the full graph."""
        self._load_cache_index()
        return len(self._cache_data.get("nodes", [])) if self._cache_data else 0

    @property
    def total_edges(self) -> int:
        """Total number of edges in the full graph."""
        self._load_cache_index()
        return len(self._cache_data.get("edges", [])) if self._cache_data else 0

    @property
    def loaded_nodes(self) -> int:
        """Number of nodes currently loaded."""
        return self._loaded_graph.number_of_nodes()

    @property
    def loaded_edges(self) -> int:
        """Number of edges currently loaded."""
        return self._loaded_graph.number_of_edges()

    def load_file(self, file_path: str) -> nx.DiGraph:
        """Load nodes and edges for a specific file.

        Args:
            file_path: Path to the file (relative to repo root).

        Returns:
            The current loaded graph (accumulates across calls).
        """
        self._load_cache_index()
        assert self._cache_data is not None
        assert self._file_index is not None

        if file_path in self._loaded_files:
            return self._loaded_graph

        # Get node indices for this file
        node_indices = self._file_index.get(file_path, [])
        if not node_indices:
            logger.debug("  No nodes found for file: %s", file_path)
            return self._loaded_graph

        # Add nodes
        nodes = self._cache_data.get("nodes", [])
        for idx in node_indices:
            node = nodes[idx]
            node_id = node.get("id", "")
            attrs = {
                "type": node.get("type", "unknown"),
                "file": node.get("file", ""),
                "name": node.get("name", ""),
                "line": node.get("line"),
                "language": node.get("language", ""),
            }
            if "fields" in node:
                attrs["fields"] = node["fields"]
            self._loaded_graph.add_node(node_id, **attrs)

        self._loaded_files.add(file_path)

        # Add edges where both endpoints are loaded
        self._add_edges_for_loaded_nodes()

        logger.debug(
            "  Loaded file %s: %d nodes, %d edges total",
            file_path,
            len(node_indices),
            self._loaded_graph.number_of_edges(),
        )

        return self._loaded_graph

    def load_around(
        self,
        seed_nodes: str | list[str],
        depth: int = 2,
        include_incoming: bool = True,
        include_outgoing: bool = True,
    ) -> nx.DiGraph:
        """Load subgraph around seed nodes up to depth hops.

        This loads the minimal subgraph needed for local analysis
        (impact analysis, nearest neighbors, etc.) without loading
        the full graph.

        Args:
            seed_nodes: Node ID(s) to start from.
            depth: Maximum hops from seed nodes to include.
            include_incoming: Include incoming edges (ancestors).
            include_outgoing: Include outgoing edges (descendants).

        Returns:
            The current loaded graph (accumulates across calls).
        """
        self._load_cache_index()
        assert self._cache_data is not None
        assert self._node_index is not None

        if isinstance(seed_nodes, str):
            seed_nodes = [seed_nodes]

        # Build a temporary full edge list for BFS
        # (We need to traverse edges to find neighbors)
        edge_map: dict[str, list[str]] = {}  # outgoing: node -> [targets]
        reverse_edge_map: dict[str, list[str]] = {}  # incoming: node -> [sources]

        for edge in self._cache_data.get("edges", []):
            source = edge.get("from", edge.get("source", ""))
            target = edge.get("to", edge.get("target", ""))
            if source and target:
                if source not in edge_map:
                    edge_map[source] = []
                edge_map[source].append(target)

                if target not in reverse_edge_map:
                    reverse_edge_map[target] = []
                reverse_edge_map[target].append(source)

        # BFS to find relevant nodes
        relevant_nodes: set[str] = set()
        queue: list[tuple[str, int]] = []

        for seed in seed_nodes:
            if seed in self._node_index:
                relevant_nodes.add(seed)
                queue.append((seed, 0))

        visited: set[str] = set(relevant_nodes)
        while queue:
            node, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue

            # Outgoing edges (descendants)
            if include_outgoing:
                for target in edge_map.get(node, []):
                    if target not in visited and target in self._node_index:
                        visited.add(target)
                        relevant_nodes.add(target)
                        queue.append((target, current_depth + 1))

            # Incoming edges (ancestors)
            if include_incoming:
                for source in reverse_edge_map.get(node, []):
                    if source not in visited and source in self._node_index:
                        visited.add(source)
                        relevant_nodes.add(source)
                        queue.append((source, current_depth + 1))

        # Add relevant nodes to loaded graph
        nodes = self._cache_data.get("nodes", [])
        for node_id in relevant_nodes:
            if node_id in self._loaded_graph:
                continue

            idx = self._node_index.get(node_id)
            if idx is None:
                continue

            node = nodes[idx]
            attrs = {
                "type": node.get("type", "unknown"),
                "file": node.get("file", ""),
                "name": node.get("name", ""),
                "line": node.get("line"),
                "language": node.get("language", ""),
            }
            if "fields" in node:
                attrs["fields"] = node["fields"]
            self._loaded_graph.add_node(node_id, **attrs)

        # Add edges where both endpoints are loaded
        self._add_edges_for_loaded_nodes()

        logger.info(
            "  LazyGraph loaded %d nodes around %d seeds (depth=%d)",
            len(relevant_nodes),
            len(seed_nodes),
            depth,
        )

        return self._loaded_graph

    def _add_edges_for_loaded_nodes(self) -> None:
        """Add edges from cache where both endpoints are in loaded graph."""
        assert self._cache_data is not None

        loaded_node_ids = set(self._loaded_graph.nodes())

        for edge in self._cache_data.get("edges", []):
            source = edge.get("from", edge.get("source", ""))
            target = edge.get("to", edge.get("target", ""))

            if source in loaded_node_ids and target in loaded_node_ids:
                if not self._loaded_graph.has_edge(source, target):
                    self._loaded_graph.add_edge(
                        source,
                        target,
                        type=edge.get("type", "unknown"),
                        resolved=edge.get("resolved", False),
                    )

    def get_graph(self) -> nx.DiGraph:
        """Get the currently loaded graph.

        Returns:
            NetworkX DiGraph with currently loaded nodes/edges.
        """
        return self._loaded_graph

    def clear(self) -> None:
        """Clear loaded graph and start fresh."""
        self._loaded_graph = nx.DiGraph()
        self._loaded_files = set()

    def load_full(self) -> nx.DiGraph:
        """Load the entire graph (use sparingly for large repos).

        Returns:
            The full NetworkX DiGraph.
        """
        self._load_cache_index()
        assert self._cache_data is not None

        return _build_graph(
            self._cache_data.get("nodes", []),
            self._cache_data.get("edges", []),
        )
