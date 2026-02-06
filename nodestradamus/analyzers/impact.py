"""Impact analysis through graph traversal.

Determines what would be affected by changing a file or symbol.
Uses NetworkX for graph traversal.
"""

from collections import defaultdict
from pathlib import Path

import networkx as nx

from nodestradamus.analyzers.constants import STDLIB_MODULES, is_test_file
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.git_cooccurrence import AnalysisError, analyze_git_cooccurrence
from nodestradamus.analyzers.graph_algorithms import ancestors_at_depth, descendants_at_depth
from nodestradamus.models.graph import (
    BreakingChange,
    FusedMatch,
    ImpactReport,
    ImpactTarget,
    RiskAssessment,
    SemanticMatch,
    SymbolUsage,
)


def _is_external_node(node_id: str, file_path: str) -> bool:
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
        # Handle both "py:typing" and "typing" formats
        clean_id = node_id
        if ":" in node_id:
            clean_id = node_id.split(":", 1)[1]
        root_module = clean_id.split(".")[0]
        if root_module in STDLIB_MODULES:
            return True
        # If no file path and not clearly internal, treat as external
        return True
    return False


def _extract_file_from_node_id(node_id: str) -> str:
    """Extract the file path from a node ID.

    Node IDs have format: 'prefix:file_path::symbol' or 'prefix:file_path'

    Args:
        node_id: The node identifier.

    Returns:
        The file path component.
    """
    # Remove prefix (py:, ts:, etc.)
    if ":" in node_id:
        rest = node_id.split(":", 1)[1]
        # Split on :: to separate file from symbol
        if "::" in rest:
            return rest.split("::")[0]
        return rest
    return node_id


def _extract_symbol_from_node_id(node_id: str) -> str:
    """Extract the symbol name from a node ID.

    Args:
        node_id: The node identifier.

    Returns:
        The symbol name, or empty string if not present.
    """
    if "::" in node_id:
        return node_id.split("::")[-1]
    return ""

# File extensions for each language
PYTHON_EXTENSIONS = frozenset({".py", ".pyw"})
TYPESCRIPT_EXTENSIONS = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
SQL_EXTENSIONS = frozenset({".sql", ".pgsql"})


def _detect_language(file_path: str) -> str:
    """Detect the language of a file based on extension.

    Args:
        file_path: Path to the file.

    Returns:
        'python', 'typescript', 'sql', or 'unknown'.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in PYTHON_EXTENSIONS:
        return "python"
    elif suffix in TYPESCRIPT_EXTENSIONS:
        return "typescript"
    elif suffix in SQL_EXTENSIONS:
        return "sql"
    return "unknown"


class AmbiguousMatchError(Exception):
    """Raised when multiple nodes match a query and disambiguation is needed."""

    def __init__(self, message: str, candidates: list[dict]):
        super().__init__(message)
        self.candidates = candidates


class NoMatchError(Exception):
    """Raised when no nodes match a query."""

    def __init__(self, message: str, suggestions: list[dict] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []


def _find_target_node(
    G: nx.DiGraph,
    file_path: str,
    symbol: str | None,
) -> str:
    """Find the node ID matching the file path and optional symbol.

    Uses strict matching with proper disambiguation:
    1. Exact path match takes priority
    2. Suffix match only when unambiguous
    3. Raises AmbiguousMatchError with candidates when multiple matches
    4. Raises NoMatchError with suggestions when no matches

    Args:
        G: The dependency graph.
        file_path: Relative path to the file.
        symbol: Optional symbol name (function/class).

    Returns:
        Node ID if found unambiguously.

    Raises:
        AmbiguousMatchError: Multiple nodes match - provides candidates.
        NoMatchError: No nodes match - provides suggestions.
    """
    file_path = file_path.lstrip("./")

    candidates: list[dict] = []
    exact_matches: list[dict] = []

    for node, attrs in G.nodes(data=True):
        node_file = attrs.get("file", "")
        node_name = attrs.get("name", "")

        if not node_file:
            continue

        # Check file match quality
        match_type = _get_match_type(file_path, node_file)

        if match_type == "none":
            continue

        # If symbol specified, must match
        if symbol:
            if node_name != symbol:
                continue

        candidate = {
            "node_id": node,
            "file": node_file,
            "name": node_name,
            "type": attrs.get("type", "unknown"),
            "line": attrs.get("line"),
            "match_type": match_type,
        }

        if match_type == "exact":
            exact_matches.append(candidate)
        else:
            candidates.append(candidate)

    # Priority 1: Return single exact match
    if len(exact_matches) == 1:
        return exact_matches[0]["node_id"]

    # Priority 2: Multiple exact matches (same file, different symbols)
    if len(exact_matches) > 1:
        # If no symbol specified, prefer file-level or first function
        if not symbol:
            # Prefer module/file-level nodes
            for m in exact_matches:
                if m["type"] in ("module", "file"):
                    return m["node_id"]
            # Otherwise return first
            return exact_matches[0]["node_id"]
        # Multiple symbols with same name in exact file match - ambiguous
        raise AmbiguousMatchError(
            f"Multiple symbols named '{symbol}' found in '{file_path}'",
            exact_matches,
        )

    # Priority 3: Return single suffix match
    if len(candidates) == 1:
        return candidates[0]["node_id"]

    # Priority 4: Multiple suffix matches - ambiguous
    if len(candidates) > 1:
        target_desc = f"'{file_path}::{symbol}'" if symbol else f"'{file_path}'"
        raise AmbiguousMatchError(
            f"Multiple nodes match {target_desc}. Provide a more specific path.",
            candidates,
        )

    # No matches found - provide suggestions
    suggestions = _find_suggestions(G, file_path, symbol)
    target_desc = f"'{file_path}::{symbol}'" if symbol else f"'{file_path}'"
    raise NoMatchError(
        f"No nodes found matching {target_desc}",
        suggestions,
    )


def _get_match_type(query_path: str, node_file: str) -> str:
    """Determine how well a query path matches a node's file path.

    Args:
        query_path: The file path being searched for.
        node_file: The file path from a graph node.

    Returns:
        'exact' - Paths are identical
        'suffix' - Query is a suffix of node path (e.g., 'utils.py' matches 'src/utils.py')
        'none' - No match
    """
    if not node_file:
        return "none"

    # Normalize paths
    query_path = query_path.lstrip("./")
    node_file = node_file.lstrip("./")

    # Exact match
    if query_path == node_file:
        return "exact"

    # Suffix match - query is the end of node path
    # Must match at path boundary (/ or start of string)
    if node_file.endswith(query_path):
        # Check it's at a path boundary
        prefix_len = len(node_file) - len(query_path)
        if prefix_len == 0 or node_file[prefix_len - 1] == "/":
            return "suffix"

    return "none"


def _find_suggestions(
    G: nx.DiGraph,
    file_path: str,
    symbol: str | None,
) -> list[dict]:
    """Find potential matches to suggest when no exact match found.

    Args:
        G: The dependency graph.
        file_path: The file path that didn't match.
        symbol: Optional symbol that didn't match.

    Returns:
        List of suggestions with node info.
    """
    suggestions: list[dict] = []
    file_path = file_path.lstrip("./")
    filename = Path(file_path).name

    for node, attrs in G.nodes(data=True):
        node_file = attrs.get("file", "")
        node_name = attrs.get("name", "")

        if not node_file:
            continue

        # Suggest if filename matches (even if directory differs)
        if Path(node_file).name == filename:
            suggestions.append(
                {
                    "node_id": node,
                    "file": node_file,
                    "name": node_name,
                    "type": attrs.get("type", "unknown"),
                    "reason": "same filename, different directory",
                }
            )
            continue

        # Suggest if symbol matches anywhere
        if symbol and node_name == symbol:
            suggestions.append(
                {
                    "node_id": node,
                    "file": node_file,
                    "name": node_name,
                    "type": attrs.get("type", "unknown"),
                    "reason": f"same symbol '{symbol}', different file",
                }
            )

    # Deduplicate and limit
    seen = set()
    unique = []
    for s in suggestions:
        if s["node_id"] not in seen:
            seen.add(s["node_id"])
            unique.append(s)

    return unique[:10]


def compute_fused_score(
    similarity: float,
    depth: int,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> float:
    """Combine semantic similarity with graph proximity.

    Args:
        similarity: Semantic similarity score (0-1).
        depth: Graph distance from target (0 = target itself).
        alpha: Weight for semantic similarity (default: 0.6).
        beta: Weight for graph proximity (default: 0.4).

    Returns:
        Combined fused score (0-1).
    """
    # depth=0 → proximity=1.0, depth=1 → 0.5, depth=2 → 0.33
    proximity = 1.0 / (depth + 1)
    return alpha * similarity + beta * proximity


def _get_fused_matches(
    repo_path: Path,
    G: "nx.DiGraph",
    target_id: str,
    file_path: str,
    symbol: str | None,
    depth: int,
    top_k: int = 10,
    threshold: float = 0.3,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> list[FusedMatch]:
    """Get matches combining graph proximity and semantic similarity.

    This implements "lazy fusion" - using graph traversal to narrow down
    candidates before running semantic search, then computing joint scores.

    Args:
        repo_path: Path to repository.
        G: The dependency graph.
        target_id: The target node ID.
        file_path: Relative path to target file.
        symbol: Optional symbol name.
        depth: Graph depth for blast radius.
        top_k: Maximum results to return.
        threshold: Minimum semantic similarity threshold.
        alpha: Weight for semantic similarity (0-1).
        beta: Weight for graph proximity (0-1).

    Returns:
        List of FusedMatch objects ranked by fused_score.
    """
    try:
        from nodestradamus.analyzers.embeddings import find_similar_code
    except ImportError:
        return []

    # Step 1: Get blast radius from graph (upstream + downstream with depths)
    upstream_dict = ancestors_at_depth(G, target_id, depth)
    downstream_dict = descendants_at_depth(G, target_id, depth)

    # Combine with depth info
    blast_radius: dict[str, int] = {}
    for node_id, d in upstream_dict.items():
        blast_radius[node_id] = d
    for node_id, d in downstream_dict.items():
        # Use minimum depth if node appears in both
        if node_id in blast_radius:
            blast_radius[node_id] = min(blast_radius[node_id], d)
        else:
            blast_radius[node_id] = d

    if not blast_radius:
        return []

    # Step 2: Run semantic search constrained to blast radius
    blast_ids = set(blast_radius.keys())
    try:
        if symbol:
            similar = find_similar_code(
                str(repo_path),
                symbol=symbol,
                top_k=len(blast_ids),  # Get all matches in blast radius
                threshold=threshold,
                filter_ids=blast_ids,
            )
        else:
            similar = find_similar_code(
                str(repo_path),
                file_path=file_path,
                top_k=len(blast_ids),
                threshold=threshold,
                filter_ids=blast_ids,
            )
    except Exception:
        return []

    # Step 3: Compute joint scores
    fused_matches = []
    for result in similar:
        node_id = result.get("id", "")
        if node_id == target_id:
            continue  # Skip the target itself

        node_depth = blast_radius.get(node_id, depth)
        similarity = result.get("similarity", 0.0)
        fused_score = compute_fused_score(similarity, node_depth, alpha, beta)

        fused_matches.append(
            FusedMatch(
                id=node_id,
                file=result.get("file", ""),
                name=result.get("name", ""),
                depth=node_depth,
                similarity=similarity,
                fused_score=round(fused_score, 4),
            )
        )

    # Sort by fused score (descending)
    fused_matches.sort(key=lambda x: x.fused_score, reverse=True)

    return fused_matches[:top_k]


def _get_semantically_related(
    repo_path: Path,
    target_id: str,
    file_path: str,
    symbol: str | None,
    exclude_ids: set[str],
    top_k: int = 5,
    threshold: float = 0.6,
) -> list[SemanticMatch]:
    """Find code semantically similar to target but not in dependency graph.

    Args:
        repo_path: Path to repository.
        target_id: The target node ID.
        file_path: Relative path to target file.
        symbol: Optional symbol name.
        exclude_ids: Node IDs to exclude (already in upstream/downstream).
        top_k: Maximum results to return.
        threshold: Minimum similarity score.

    Returns:
        List of semantically similar code blocks.
    """
    try:
        from nodestradamus.analyzers.embeddings import find_similar_code
    except ImportError:
        # sentence-transformers not installed
        return []

    try:
        # Search by symbol name or file path
        if symbol:
            results = find_similar_code(
                str(repo_path),
                symbol=symbol,
                top_k=top_k + len(exclude_ids),  # Get extra to filter
                threshold=threshold,
            )
        else:
            results = find_similar_code(
                str(repo_path),
                file_path=file_path,
                top_k=top_k + len(exclude_ids),
                threshold=threshold,
            )
    except Exception:
        return []

    # Filter out nodes already in dependency graph
    semantic_matches = []
    for r in results:
        node_id = r.get("id", "")
        # Skip if it's the target itself or already in graph
        if node_id == target_id or node_id in exclude_ids:
            continue
        # Skip if same file (usually adjacent functions)
        if r.get("file", "") == file_path:
            continue

        semantic_matches.append(
            SemanticMatch(
                id=node_id,
                file=r.get("file", ""),
                name=r.get("name", ""),
                similarity=r.get("similarity", 0.0),
            )
        )

        if len(semantic_matches) >= top_k:
            break

    return semantic_matches


def _get_cooccurring_files(
    repo_path: Path,
    target_file: str,
) -> list[str]:
    """Get files that frequently change with the target file.

    Args:
        repo_path: Path to repository.
        target_file: Relative path to target file.

    Returns:
        List of co-occurring file paths.
    """
    try:
        G = analyze_git_cooccurrence(repo_path, commits=200, min_strength=0.2)
    except AnalysisError:
        return []

    target_file = target_file.lstrip("./")
    co_occurring: list[str] = []

    # Direct neighbors (if target is in graph)
    if target_file in G:
        for neighbor in G.neighbors(target_file):
            co_occurring.append(neighbor)

    # Also check if target_file is a substring match
    for source, target in G.edges():
        if target_file in source or source in target_file:
            co_occurring.append(target)
        elif target_file in target or target in target_file:
            co_occurring.append(source)

    return list(set(co_occurring))[:10]


def _group_by_file(targets: list[ImpactTarget]) -> dict[str, list[str]]:
    """Group impact targets by file path.

    Args:
        targets: List of ImpactTarget objects.

    Returns:
        Dictionary mapping file paths to lists of symbol names.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for t in targets:
        file_path = _extract_file_from_node_id(t.id)
        symbol_name = _extract_symbol_from_node_id(t.id)
        if symbol_name:
            grouped[file_path].append(symbol_name)
        else:
            # File-level node, just add the file
            if file_path not in grouped:
                grouped[file_path] = []
    return dict(grouped)


def _build_depth_summary(
    upstream: list[ImpactTarget],
    downstream: list[ImpactTarget],
) -> dict[str, dict[str, int]]:
    """Build a summary of counts at each depth level.

    Args:
        upstream: List of upstream dependencies.
        downstream: List of downstream dependencies.

    Returns:
        Dictionary with depth counts for upstream and downstream.
    """
    upstream_counts: dict[str, int] = defaultdict(int)
    downstream_counts: dict[str, int] = defaultdict(int)

    for t in upstream:
        upstream_counts[f"depth_{t.depth}"] += 1

    for t in downstream:
        downstream_counts[f"depth_{t.depth}"] += 1

    return {
        "upstream": dict(upstream_counts),
        "downstream": dict(downstream_counts),
    }


def _get_symbol_usage(
    G: nx.DiGraph,
    target_file: str,
) -> list[SymbolUsage]:
    """Aggregate which symbols from a file are imported by external files.

    Traverses incoming edges of type 'imports_symbol' to find all external
    files that import symbols defined in target_file.

    Args:
        G: The dependency graph.
        target_file: Relative path to the target file.

    Returns:
        List of SymbolUsage sorted by count (descending).
    """
    # Find all symbol nodes defined in target_file
    target_symbols: dict[str, dict] = {}  # node_id -> {name, type, line}
    for node_id, attrs in G.nodes(data=True):
        node_file = attrs.get("file", "")
        if node_file == target_file and "::" in node_id:
            symbol_name = _extract_symbol_from_node_id(node_id)
            if symbol_name:
                target_symbols[node_id] = {
                    "name": symbol_name,
                    "type": attrs.get("type", "unknown"),
                    "line": attrs.get("line"),
                }

    # For each symbol, find incoming imports_symbol edges
    symbol_importers: dict[str, set[str]] = defaultdict(set)

    for source, target, attrs in G.edges(data=True):
        if attrs.get("type") == "imports_symbol" and target in target_symbols:
            # Get the file that's importing
            source_file = _extract_file_from_node_id(source)
            # Only count external files
            if source_file and source_file != target_file:
                symbol_name = target_symbols[target]["name"]
                symbol_importers[symbol_name].add(source_file)

    # Build result list
    results: list[SymbolUsage] = []
    for symbol_name, importing_files in symbol_importers.items():
        # Find the symbol info
        symbol_info = None
        for _node_id, info in target_symbols.items():
            if info["name"] == symbol_name:
                symbol_info = info
                break

        results.append(
            SymbolUsage(
                symbol=symbol_name,
                symbol_type=symbol_info["type"] if symbol_info else "unknown",
                importing_files=sorted(importing_files),
                count=len(importing_files),
                line=symbol_info["line"] if symbol_info else None,
            )
        )

    # Sort by count descending
    results.sort(key=lambda x: x.count, reverse=True)
    return results


def _get_breaking_changes(
    G: nx.DiGraph,
    target_file: str,
) -> list[BreakingChange]:
    """Identify symbols that external code depends on.

    For each symbol defined in target_file, finds all external files
    that would break if the symbol were removed.

    Args:
        G: The dependency graph.
        target_file: Relative path to the target file.

    Returns:
        List of BreakingChange sorted by total dependents (descending).
    """
    # Find all symbol nodes defined in target_file
    target_symbols: dict[str, dict] = {}  # node_id -> {name, type}
    for node_id, attrs in G.nodes(data=True):
        node_file = attrs.get("file", "")
        if node_file == target_file and "::" in node_id:
            symbol_name = _extract_symbol_from_node_id(node_id)
            if symbol_name:
                target_symbols[node_id] = {
                    "name": symbol_name,
                    "type": attrs.get("type", "unknown"),
                }

    # For each symbol, compute direct and indirect dependents
    results: list[BreakingChange] = []

    for symbol_id, symbol_info in target_symbols.items():
        # Get all ancestors (things that depend on this symbol)
        try:
            upstream = ancestors_at_depth(G, symbol_id, 3)
        except Exception:
            continue

        # Separate direct (depth 1) from indirect (depth > 1)
        direct_files: set[str] = set()
        indirect_files: set[str] = set()

        for node_id, depth in upstream.items():
            node_file = _extract_file_from_node_id(node_id)
            # Only count external files
            if node_file and node_file != target_file:
                if depth == 1:
                    direct_files.add(node_file)
                else:
                    indirect_files.add(node_file)

        # Only include symbols with external dependents
        if direct_files or indirect_files:
            results.append(
                BreakingChange(
                    symbol=symbol_info["name"],
                    symbol_type=symbol_info["type"],
                    dependents=sorted(direct_files | indirect_files),
                    direct_count=len(direct_files),
                    indirect_count=len(indirect_files),
                )
            )

    # Sort by total dependents descending
    results.sort(key=lambda x: x.direct_count + x.indirect_count, reverse=True)
    return results


def get_impact(
    repo_path: str | Path,
    file_path: str,
    symbol: str | None = None,
    depth: int = 3,
    include_semantic: bool = False,
    fusion_mode: bool = False,
    compact: bool = True,
    exclude_same_file: bool = True,
    exclude_external: bool = True,
) -> ImpactReport:
    """Analyze the impact of changing a specific file or symbol.

    Uses unified dependency analyzer and NetworkX for traversal.

    Args:
        repo_path: Absolute path to repository root.
        file_path: Relative path to the file being changed.
        symbol: Optional specific function/class name.
        depth: How many levels of dependencies to traverse.
        include_semantic: Whether to include semantically similar code (slower,
            requires computing embeddings). Default: False. When fusion_mode is
            enabled, this only shows code similar but NOT in the dependency graph.
        fusion_mode: Whether to enable fused impact analysis that combines
            graph proximity with semantic similarity. Returns a single ranked
            list of matches with joint scores. Default: False.
        compact: Whether to group results by file (default: True). When True,
            also populates upstream_by_file and downstream_by_file.
        exclude_same_file: Whether to exclude symbols from the same file as
            target (default: True). Reduces noise when analyzing files with
            many internal functions.
        exclude_external: Whether to exclude external/stdlib imports from results
            (default: True). Filters out typing, collections, os, etc. from
            downstream dependencies to focus on project code.

    Returns:
        ImpactReport with upstream and downstream dependencies. When fusion_mode
        is enabled, also includes fused_matches with combined graph+semantic scores.
        When compact=True, includes upstream_by_file and downstream_by_file.
        Always includes depth_summary and external_filtered count.

    Raises:
        AmbiguousMatchError: Multiple nodes match - provides candidates for disambiguation.
        NoMatchError: No nodes match - provides suggestions.
    """
    repo_path = Path(repo_path).resolve()

    # Detect language for target ID prefix
    language = _detect_language(file_path)
    prefix_map = {"typescript": "ts", "python": "py", "rust": "rs", "sql": "sql"}
    prefix = prefix_map.get(language, "py")

    # Build unified dependency graph
    G = analyze_deps(repo_path)

    if G.number_of_nodes() == 0:
        target_id = f"{prefix}:{file_path}::{symbol}" if symbol else f"{prefix}:{file_path}"
        return ImpactReport(
            target=target_id,
            upstream=[],
            downstream=[],
            co_occurring_files=[],
            risk_assessment=RiskAssessment(),
            depth_summary={"upstream": {}, "downstream": {}},
        )

    # Find target node - may raise AmbiguousMatchError or NoMatchError
    target_id = _find_target_node(G, file_path, symbol)

    # Extract target file for same-file filtering
    target_file = _extract_file_from_node_id(target_id)

    # Find upstream (things that depend on target) using NetworkX
    upstream_dict = ancestors_at_depth(G, target_id, depth)
    upstream_all = [
        ImpactTarget(id=node, depth=d)
        for node, d in sorted(upstream_dict.items(), key=lambda x: x[1])
    ]

    # Find downstream (things target depends on) using NetworkX
    downstream_dict = descendants_at_depth(G, target_id, depth)
    downstream_all = [
        ImpactTarget(id=node, depth=d)
        for node, d in sorted(downstream_dict.items(), key=lambda x: x[1])
    ]

    # Apply same-file filtering if requested
    if exclude_same_file:
        upstream = [
            t for t in upstream_all
            if _extract_file_from_node_id(t.id) != target_file
        ]
        downstream = [
            t for t in downstream_all
            if _extract_file_from_node_id(t.id) != target_file
        ]
    else:
        upstream = upstream_all
        downstream = downstream_all

    # Track external counts before filtering
    external_upstream_count = 0
    external_downstream_count = 0

    # Apply external filtering if requested
    if exclude_external:
        filtered_upstream = []
        for t in upstream:
            node_file = G.nodes[t.id].get("file", "") if t.id in G else ""
            if _is_external_node(t.id, node_file):
                external_upstream_count += 1
            else:
                filtered_upstream.append(t)
        upstream = filtered_upstream

        filtered_downstream = []
        for t in downstream:
            node_file = G.nodes[t.id].get("file", "") if t.id in G else ""
            if _is_external_node(t.id, node_file):
                external_downstream_count += 1
            else:
                filtered_downstream.append(t)
        downstream = filtered_downstream

    # Build depth summary (always included, uses filtered lists)
    depth_summary = _build_depth_summary(upstream, downstream)

    # Group by file if compact mode
    upstream_by_file: dict[str, list[str]] = {}
    downstream_by_file: dict[str, list[str]] = {}
    if compact:
        upstream_by_file = _group_by_file(upstream)
        downstream_by_file = _group_by_file(downstream)

    # Get co-occurring files from git history
    co_occurring_files = _get_cooccurring_files(repo_path, file_path)

    # Get semantically related code (not in dependency graph)
    # Only compute if explicitly requested (slower, requires embeddings)
    semantically_related: list[SemanticMatch] = []
    if include_semantic:
        connected_ids = {t.id for t in upstream + downstream}
        semantically_related = _get_semantically_related(
            repo_path,
            target_id,
            file_path,
            symbol,
            exclude_ids=connected_ids,
            top_k=5,
            threshold=0.6,
        )

    # Get fused matches (graph proximity + semantic similarity)
    # Only compute if fusion_mode is enabled
    fused_matches: list[FusedMatch] = []
    if fusion_mode:
        fused_matches = _get_fused_matches(
            repo_path,
            G,
            target_id,
            file_path,
            symbol,
            depth=depth,
            top_k=10,
            threshold=0.3,
        )

    # Calculate risk assessment using proper test file detection
    test_files: set[str] = set()
    for t in upstream + downstream:
        node_file = _extract_file_from_node_id(t.id)
        if is_test_file(node_file):
            test_files.add(node_file)

    risk = RiskAssessment(
        direct_dependents=len([t for t in upstream if t.depth == 1]),
        indirect_dependents=len([t for t in upstream if t.depth > 1]),
        test_files_affected=len(test_files),
    )

    # Build external_filtered dict (only if filtering was applied)
    external_filtered: dict[str, int] = {}
    if exclude_external:
        external_filtered = {
            "upstream": external_upstream_count,
            "downstream": external_downstream_count,
        }

    return ImpactReport(
        target=target_id,
        upstream=upstream,
        downstream=downstream,
        co_occurring_files=co_occurring_files,
        semantically_related=semantically_related,
        fused_matches=fused_matches,
        risk_assessment=risk,
        upstream_by_file=upstream_by_file,
        downstream_by_file=downstream_by_file,
        depth_summary=depth_summary,
        external_filtered=external_filtered,
    )
