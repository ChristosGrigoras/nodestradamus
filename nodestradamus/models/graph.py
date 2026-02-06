"""Graph data models for dependency and co-occurrence analysis.

Includes Pydantic models for serialization and NetworkX conversion utilities.
"""

from datetime import datetime
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field


class FieldInfo(BaseModel):
    """A field/column in a schema-defining construct (class, table, interface, etc.)."""

    name: str = Field(description="Field/column name")
    type: str = Field(description="Type annotation or data type (normalized or raw)")
    nullable: bool = Field(default=True, description="Whether the field can be null/None")
    references: str | None = Field(
        default=None, description="Foreign key target for SQL (e.g., 'users.id')"
    )


class GraphNode(BaseModel):
    """A node in the dependency graph representing a code symbol."""

    id: str = Field(description="Unique identifier (e.g., 'py:src/auth.py::login')")
    type: Literal["function", "class", "method", "module", "file", "table", "config", "view", "cte", "procedure", "trigger", "constant"] = Field(
        description="Type of code symbol"
    )
    file: str = Field(description="Relative file path")
    name: str = Field(description="Symbol name")
    line: int | None = Field(default=None, description="Line number in file")
    fields: list[FieldInfo] | None = Field(
        default=None, description="Schema fields for classes/tables/interfaces"
    )


class GraphEdge(BaseModel):
    """An edge in the dependency graph representing a relationship."""

    source: str = Field(alias="from", description="Source node ID")
    target: str = Field(alias="to", description="Target node ID")
    type: Literal["calls", "inherits", "imports", "imports_symbol", "extends", "defined_in", "contains"] = Field(
        description="Type of relationship"
    )
    resolved: bool = Field(default=False, description="Whether target was resolved to a known node")

    model_config = {"populate_by_name": True}


class GraphMetadata(BaseModel):
    """Metadata about the generated graph."""

    analyzer: str = Field(description="Name of the analyzer that generated this graph")
    version: str = Field(description="Version of the analyzer")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None), description="When the graph was generated"
    )
    file_count: int = Field(default=0, description="Number of files analyzed")
    source_directory: str | None = Field(
        default=None, description="Root directory that was analyzed"
    )


class DependencyGraph(BaseModel):
    """Complete dependency graph with nodes, edges, and metadata."""

    nodes: list[GraphNode] = Field(default_factory=list, description="Graph nodes")
    edges: list[GraphEdge] = Field(default_factory=list, description="Graph edges")
    metadata: GraphMetadata = Field(description="Graph metadata")
    errors: list[dict] = Field(
        default_factory=list, description="Errors encountered during analysis"
    )


class CooccurrenceEdge(BaseModel):
    """An edge representing files that change together."""

    source: str = Field(alias="from", description="First file path")
    target: str = Field(alias="to", description="Second file path")
    type: Literal["co-occurs"] = Field(default="co-occurs")
    count: int = Field(description="Number of commits where both files changed")
    strength: float = Field(ge=0.0, le=1.0, description="Jaccard similarity (0-1)")

    model_config = {"populate_by_name": True}


class CooccurrenceMetadata(BaseModel):
    """Metadata for co-occurrence graph."""

    analyzer: str = Field(default="git_cooccurrence")
    version: str = Field(default="0.2.0")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
    commits_analyzed: int = Field(description="Number of commits analyzed")
    min_strength_threshold: float = Field(
        default=0.3, description="Minimum strength to include edge"
    )


class CooccurrenceGraph(BaseModel):
    """Graph of files that frequently change together."""

    nodes: list[str] = Field(default_factory=list, description="File paths")
    edges: list[CooccurrenceEdge] = Field(
        default_factory=list, description="Co-occurrence relationships"
    )
    metadata: CooccurrenceMetadata = Field(description="Graph metadata")


# Community Meta-Graph Models


class ModuleMetrics(BaseModel):
    """Cohesion and coupling metrics for a detected community/module."""

    module_id: int = Field(description="Community index (0-based)")
    size: int = Field(description="Number of nodes in the community")
    internal_edges: int = Field(description="Edges within the community")
    external_edges: int = Field(description="Edges crossing community boundary")
    cohesion: float = Field(
        ge=0.0, le=1.0,
        description="Ratio of internal to total edges (1.0 = fully cohesive)"
    )
    afferent_coupling: int = Field(
        description="Incoming edges from other communities"
    )
    efferent_coupling: int = Field(
        description="Outgoing edges to other communities"
    )
    instability: float = Field(
        ge=0.0, le=1.0,
        description="Ratio of efferent to total coupling (1.0 = fully unstable)"
    )


class InterModuleEdge(BaseModel):
    """An edge between two communities in the meta-graph."""

    source: int = Field(description="Source community ID")
    target: int = Field(description="Target community ID")
    edge_count: int = Field(description="Number of edges between communities")


# Hierarchical Graph Models


class HierarchyNode(BaseModel):
    """A node in a hierarchical (collapsed) graph view."""

    id: str = Field(description="Node identifier (e.g., directory path or module name)")
    level: Literal["package", "module", "class", "function"] = Field(
        description="Hierarchy level"
    )
    child_count: int = Field(description="Number of children aggregated into this node")
    children: list[str] = Field(
        default_factory=list,
        description="List of child node IDs (limited for display)",
    )


class HierarchyEdge(BaseModel):
    """An aggregated edge between hierarchy nodes."""

    source: str = Field(description="Source hierarchy node ID")
    target: str = Field(description="Target hierarchy node ID")
    edge_count: int = Field(description="Number of underlying edges aggregated")
    edge_types: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each edge type (calls, imports, inherits)",
    )


class HierarchicalGraph(BaseModel):
    """A collapsed view of the dependency graph at a specific hierarchy level."""

    level: Literal["package", "module", "class", "function"] = Field(
        description="The hierarchy level of this view"
    )
    nodes: list[HierarchyNode] = Field(
        default_factory=list, description="Aggregated nodes at this level"
    )
    edges: list[HierarchyEdge] = Field(
        default_factory=list, description="Aggregated edges between nodes"
    )
    total_original_nodes: int = Field(description="Total nodes in the original graph")
    total_original_edges: int = Field(description="Total edges in the original graph")


# Layer Validation Models


class LayerViolation(BaseModel):
    """A dependency that violates layer ordering rules."""

    source_node: str = Field(description="Node ID causing the violation")
    target_node: str = Field(description="Node ID being depended upon")
    source_layer: int = Field(description="Layer index of source (0 = top)")
    target_layer: int = Field(description="Layer index of target")
    edge_type: str = Field(description="Type of dependency edge (calls, imports, etc.)")
    severity: Literal["error", "warning"] = Field(
        default="error",
        description="error = violation, warning = same-layer dependency",
    )


class LayerValidationResult(BaseModel):
    """Result of validating a layered architecture."""

    layers: list[str] = Field(description="Layer names in order (top to bottom)")
    violations: list[LayerViolation] = Field(
        default_factory=list, description="Dependencies that violate layer rules"
    )
    valid_edges: int = Field(description="Number of edges respecting layer rules")
    violation_count: int = Field(description="Number of violating edges")
    compliance_rate: float = Field(
        ge=0.0, le=1.0, description="Ratio of valid edges to total"
    )


class ImpactTarget(BaseModel):
    """A dependency found during impact analysis."""

    id: str = Field(description="Node ID")
    depth: int = Field(description="Distance from the target in the graph")


class SemanticMatch(BaseModel):
    """A semantically similar code block found via embeddings."""

    id: str = Field(description="Node ID of similar code")
    file: str = Field(description="File path")
    name: str = Field(default="", description="Function/class name")
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")


class FusedMatch(BaseModel):
    """A match combining structural (graph) and semantic (embedding) signals.

    Used when fusion_mode is enabled in impact analysis to provide
    a single ranked list that considers both graph proximity and
    semantic similarity.
    """

    id: str = Field(description="Node ID")
    file: str = Field(description="File path")
    name: str = Field(default="", description="Function/class name")
    depth: int = Field(description="Graph distance from target (0 = target itself)")
    similarity: float = Field(ge=0.0, le=1.0, description="Semantic similarity score")
    fused_score: float = Field(
        ge=0.0, le=1.0, description="Combined score (alpha * similarity + beta * proximity)"
    )


class RiskAssessment(BaseModel):
    """Risk assessment for a change."""

    direct_dependents: int = Field(default=0, description="Number of direct dependents")
    indirect_dependents: int = Field(default=0, description="Number of indirect dependents")
    test_files_affected: int = Field(
        default=0, description="Number of test files that may be affected"
    )


class ImpactReport(BaseModel):
    """Report of what would be affected by changing a file or symbol."""

    target: str = Field(description="The file or symbol being analyzed")
    upstream: list[ImpactTarget] = Field(
        default_factory=list, description="Things that depend on target"
    )
    downstream: list[ImpactTarget] = Field(
        default_factory=list, description="Things that target depends on"
    )
    co_occurring_files: list[str] = Field(
        default_factory=list, description="Files that often change with target"
    )
    semantically_related: list[SemanticMatch] = Field(
        default_factory=list,
        description="Code that is semantically similar but not directly connected",
    )
    fused_matches: list["FusedMatch"] = Field(
        default_factory=list,
        description="Ranked matches combining graph proximity and semantic similarity (fusion_mode only)",
    )
    risk_assessment: RiskAssessment = Field(
        default_factory=RiskAssessment, description="Risk assessment"
    )
    # Grouped view (when compact=true)
    upstream_by_file: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Upstream dependencies grouped by file path",
    )
    downstream_by_file: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Downstream dependencies grouped by file path",
    )
    # Depth summary (always included)
    depth_summary: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Count of dependencies at each depth level",
    )
    # External filtering info (when exclude_external=true)
    external_filtered: dict[str, int] = Field(
        default_factory=dict,
        description="Count of external imports filtered from upstream/downstream",
    )
    # Refactor analysis (when refactor_mode=true)
    refactor_analysis: "RefactorAnalysis | None" = Field(
        default=None,
        description="Symbol-level refactoring analysis (only when refactor_mode=true)",
    )


class PackageInfo(BaseModel):
    """Information about a package in a monorepo."""

    name: str = Field(description="Package name (from pyproject.toml or package.json)")
    path: str = Field(description="Relative path to the package directory")
    language: str = Field(description="Primary language (python, typescript, etc.)")


class ProjectMetadata(BaseModel):
    """Reconnaissance metadata about a repository's structure and characteristics."""

    # Language distribution
    languages: dict[str, int] = Field(
        default_factory=dict,
        description="File count by language (e.g., {'typescript': 199, 'python': 25})",
    )
    primary_language: str | None = Field(
        default=None, description="Most common language in the repo"
    )

    # Structure
    key_directories: list[str] = Field(
        default_factory=list, description="Important directories (src/, tests/, etc.)"
    )
    entry_points: list[str] = Field(
        default_factory=list,
        description="Likely entry point files (index.ts, main.py, app.py)",
    )
    config_files: list[str] = Field(
        default_factory=list,
        description="Configuration files (package.json, pyproject.toml)",
    )

    # Git info
    has_git: bool = Field(default=False, description="Whether repo has git")
    recent_commit_count: int = Field(default=0, description="Commits in last 30 days")
    contributors: int = Field(default=0, description="Number of unique contributors")

    # Detected patterns
    frameworks: list[str] = Field(
        default_factory=list, description="Detected frameworks (express, react, pytest)"
    )
    package_managers: list[str] = Field(
        default_factory=list, description="Package managers (npm, pip, cargo)"
    )
    has_tests: bool = Field(default=False, description="Whether tests directory exists")
    has_ci: bool = Field(default=False, description="Whether CI config exists (.github/workflows/)")

    # Tool recommendations
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Nodestradamus tools suggested for this repo",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Example queries to run with suggested tools",
    )

    # Ignore patterns (LLM-driven)
    suggested_ignores: list[str] = Field(
        default_factory=list,
        description="Suggested ignore patterns based on detected frameworks/languages",
    )
    nodestradamusignore_exists: bool = Field(
        default=False,
        description="Whether .nodestradamusignore file exists in repo",
    )

    # Workflow guidance
    recommended_workflow: list[str] = Field(
        default_factory=list,
        description="Recommended sequence of Nodestradamus tools to run next",
    )
    next_steps: list[dict[str, str]] = Field(
        default_factory=list,
        description="Actionable next steps with tool name and description",
    )

    # Monorepo support
    is_monorepo: bool = Field(
        default=False,
        description="Whether this appears to be a monorepo with multiple packages",
    )
    packages: list[PackageInfo] = Field(
        default_factory=list,
        description="Detected packages in monorepo (empty for single-package repos)",
    )

    # Smart pipeline fields (Phase 3)
    project_type: Literal["app", "lib", "monorepo", "unknown"] = Field(
        default="unknown",
        description="Classified project type (app, lib, monorepo, or unknown)",
    )
    readme_hints: list[str] = Field(
        default_factory=list,
        description="Hints extracted from README (e.g., 'core logic in src/engine/')",
    )
    recommended_scope: list[str] = Field(
        default_factory=list,
        description="Recommended scope paths for focused analysis (e.g., ['src/', 'lib/'])",
    )

    # Lazy loading options (for large codebases / monorepos)
    lazy_options: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "When to use lazy/on-demand loading: LazyGraph, LazyEmbeddingGraph, "
            "lazy embedding. Each item has 'option', 'when', 'description'."
        ),
    )


# String Reference Models


class StringContext(BaseModel):
    """Context where a string literal appears in source code."""

    call_site: str | None = Field(
        default=None,
        description="Function/method the string is passed to (e.g., 'open', 'os.getenv')",
    )
    variable_name: str | None = Field(
        default=None,
        description="Variable the string is assigned to (e.g., 'CONFIG_PATH')",
    )
    enclosing_function: str | None = Field(
        default=None, description="Name of the function containing this string"
    )
    enclosing_class: str | None = Field(
        default=None, description="Name of the class containing this string"
    )
    line: int = Field(description="Line number where the string appears")


class StringRefNode(BaseModel):
    """A string literal extracted from source code with its contexts."""

    value: str = Field(description="The actual string content")
    file: str = Field(description="Relative path to the file containing this string")
    contexts: list[StringContext] = Field(
        default_factory=list,
        description="All contexts where this string appears in this file",
    )


class StringRefMetadata(BaseModel):
    """Metadata for string reference analysis."""

    analyzer: str = Field(default="string_refs")
    version: str = Field(default="0.1.0")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
    file_count: int = Field(default=0, description="Number of files analyzed")
    total_strings: int = Field(default=0, description="Total strings extracted")
    source_directory: str | None = Field(
        default=None, description="Root directory that was analyzed"
    )


class StringRefGraph(BaseModel):
    """Graph of string references across the codebase."""

    strings: list[StringRefNode] = Field(
        default_factory=list, description="All extracted string references"
    )
    file_count: int = Field(default=0, description="Number of files analyzed")
    metadata: StringRefMetadata = Field(description="Analysis metadata")
    errors: list[dict] = Field(
        default_factory=list, description="Errors encountered during analysis"
    )


class SignificantString(BaseModel):
    """A string identified as significant by topology analysis."""

    value: str = Field(description="The string content")
    referenced_by: list[str] = Field(description="List of files that reference this string")
    contexts: list[StringContext] = Field(description="All contexts across all files")
    importance_score: float = Field(
        ge=0.0, le=1.0, description="Normalized importance score based on topology"
    )
    reference_count: int = Field(description="Number of files referencing this string")


class StringAnalysisResult(BaseModel):
    """Result of string reference analysis with significance ranking."""

    significant_strings: list[SignificantString] = Field(
        default_factory=list, description="Strings ranked by importance"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Analysis metadata (total found, filtered, etc.)",
    )


# =============================================================================
# Refactor Analysis Models
# =============================================================================


class SymbolUsage(BaseModel):
    """Tracks which external files import a specific symbol."""

    symbol: str = Field(description="Symbol name (function, class, or constant)")
    symbol_type: str = Field(description="Type: 'function', 'class', or 'constant'")
    importing_files: list[str] = Field(
        default_factory=list,
        description="List of files that import this symbol",
    )
    count: int = Field(description="Number of files importing this symbol")
    line: int | None = Field(default=None, description="Line number where symbol is defined")


class DuplicateLocation(BaseModel):
    """Specific location of a duplicate code block."""

    file: str = Field(description="File path")
    line_start: int = Field(description="Starting line number")
    line_end: int = Field(description="Ending line number")
    preview: str = Field(description="First few lines of the code block")


class DuplicateBlock(BaseModel):
    """An exact or near-duplicate code block with locations."""

    content_hash: str = Field(description="Content hash (first 8 chars)")
    similarity: float = Field(
        ge=0.0, le=1.0,
        description="Similarity score (1.0 = exact match)",
    )
    locations: list[DuplicateLocation] = Field(
        default_factory=list,
        description="All locations where this duplicate appears",
    )


class FunctionCluster(BaseModel):
    """A logical grouping of related functions within a file."""

    name: str = Field(description="Inferred cluster name (e.g., 'SQL parsing')")
    line_start: int = Field(description="Starting line of the cluster")
    line_end: int = Field(description="Ending line of the cluster")
    functions: list[str] = Field(
        default_factory=list,
        description="Function names in this cluster",
    )
    cohesion_score: float = Field(
        ge=0.0, le=1.0,
        description="How tightly coupled the functions are (1.0 = fully cohesive)",
    )


class BreakingChange(BaseModel):
    """A symbol that external code depends on."""

    symbol: str = Field(description="Symbol name")
    symbol_type: str = Field(description="Type: 'function', 'class', or 'constant'")
    dependents: list[str] = Field(
        default_factory=list,
        description="Files that depend on this symbol",
    )
    direct_count: int = Field(description="Number of direct imports")
    indirect_count: int = Field(description="Number of transitive dependencies")


class RefactorAnalysis(BaseModel):
    """Extended analysis for refactoring tasks.

    Provides symbol-level precision for planning refactors:
    - Which symbols are used by external code
    - Duplicate code that could be consolidated
    - Logical function groupings for splitting files
    - Breaking changes if symbols are removed
    """

    symbol_usage: list[SymbolUsage] = Field(
        default_factory=list,
        description="Symbols imported by external files, sorted by usage count",
    )
    duplicates: list[DuplicateBlock] = Field(
        default_factory=list,
        description="Identical or near-identical code blocks",
    )
    clusters: list[FunctionCluster] = Field(
        default_factory=list,
        description="Logical function groupings within the file",
    )
    breaking_changes: list[BreakingChange] = Field(
        default_factory=list,
        description="Symbols with external dependencies (would break if removed)",
    )


# NetworkX Conversion Utilities


def digraph_to_json(G: nx.DiGraph) -> dict[str, Any]:
    """Serialize a NetworkX DiGraph to JSON-compatible dict.

    Args:
        G: NetworkX directed graph.

    Returns:
        Dict with nodes, edges, and metadata.
    """
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "type": attrs.get("type", "unknown"),
                "file": attrs.get("file", ""),
                "name": attrs.get("name", ""),
                "line": attrs.get("line"),
            }
        )

    edges = []
    for source, target, attrs in G.edges(data=True):
        edges.append(
            {
                "from": source,
                "to": target,
                "type": attrs.get("type", "unknown"),
                "resolved": attrs.get("resolved", False),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        },
    }


def json_to_digraph(data: dict[str, Any]) -> nx.DiGraph:
    """Deserialize a JSON dict to NetworkX DiGraph.

    Args:
        data: Dict with nodes and edges.

    Returns:
        NetworkX directed graph.
    """
    G = nx.DiGraph()

    for node in data.get("nodes", []):
        G.add_node(
            node["id"],
            type=node.get("type", "unknown"),
            file=node.get("file", ""),
            name=node.get("name", ""),
            line=node.get("line"),
        )

    for edge in data.get("edges", []):
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


def graph_to_json(G: nx.Graph) -> dict[str, Any]:
    """Serialize a NetworkX Graph to JSON-compatible dict.

    Args:
        G: NetworkX undirected graph.

    Returns:
        Dict with nodes and edges.
    """
    nodes = list(G.nodes())
    edges = []

    for source, target, attrs in G.edges(data=True):
        edges.append(
            {
                "from": source,
                "to": target,
                **attrs,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        },
    }


def json_to_graph(data: dict[str, Any]) -> nx.Graph:
    """Deserialize a JSON dict to NetworkX Graph.

    Args:
        data: Dict with nodes and edges.

    Returns:
        NetworkX undirected graph.
    """
    G = nx.Graph()

    for node in data.get("nodes", []):
        if isinstance(node, str):
            G.add_node(node)
        else:
            G.add_node(node.get("id", node), **node)

    for edge in data.get("edges", []):
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))
        attrs = {k: v for k, v in edge.items() if k not in ("from", "to", "source", "target")}
        if source and target:
            G.add_edge(source, target, **attrs)

    return G
