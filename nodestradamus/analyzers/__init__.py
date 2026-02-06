"""Analyzers for codebase intelligence."""

from nodestradamus.analyzers.dead_code import find_dead_code, find_orphaned_files
from nodestradamus.analyzers.deps import analyze_deps, graph_metadata
from nodestradamus.analyzers.docs import (
    DocAnalysisResult,
    DocReference,
    analyze_docs,
    extract_doc_references,
    validate_references,
)
from nodestradamus.analyzers.embedding_providers import (
    EmbeddingProvider,
    LocalProvider,
    MistralProvider,
    get_embedding_provider,
    reset_provider,
)
from nodestradamus.analyzers.embeddings import (
    compute_embeddings,
    detect_duplicates,
    find_similar_code,
    semantic_search,
)
from nodestradamus.analyzers.fingerprints import (
    build_fingerprint_index,
    find_similar,
    load_fingerprint_index,
)
from nodestradamus.analyzers.git_cooccurrence import (
    AnalysisError,
    analyze_git_cooccurrence,
    cooccurrence_metadata,
)
from nodestradamus.analyzers.graph_algorithms import (
    LazyEmbeddingGraph,
    ancestors_at_depth,
    betweenness,
    community_metagraph,
    descendants_at_depth,
    detect_communities,
    find_cycles,
    hierarchical_view,
    pagerank,
    sampled_betweenness,
    shortest_path,
    strongly_connected,
    top_n,
    validate_layers,
)
from nodestradamus.analyzers.ignore import (
    DEFAULT_IGNORES,
    FRAMEWORK_IGNORES,
    generate_nodestradamusignore_content,
    generate_suggested_ignores,
    load_ignore_patterns,
    nodestradamusignore_exists,
    should_ignore,
)
from nodestradamus.analyzers.impact import (
    AmbiguousMatchError,
    NoMatchError,
    get_impact,
)
from nodestradamus.analyzers.project_scout import project_scout
from nodestradamus.analyzers.string_extraction import (
    NOISE_PATTERNS,
    SKIP_DIRS,
    extract_python_strings,
    extract_sql_strings,
    extract_typescript_strings,
    is_noise,
)
from nodestradamus.analyzers.string_refs import (
    analyze_python_string_refs,
    analyze_sql_string_refs,
    analyze_string_refs,
    analyze_typescript_string_refs,
)
from nodestradamus.analyzers.string_topology import (
    analyze_string_topology,
    find_string_usages,
)

__all__ = [
    # Unified dependency analyzer
    "analyze_deps",
    "analyze_deps_smart",
    "graph_metadata",
    # Git co-occurrence
    "analyze_git_cooccurrence",
    "cooccurrence_metadata",
    "AnalysisError",
    # Graph algorithms
    "pagerank",
    "betweenness",
    "sampled_betweenness",
    "community_metagraph",
    "detect_communities",
    "find_cycles",
    "hierarchical_view",
    "LazyEmbeddingGraph",
    "shortest_path",
    "strongly_connected",
    "top_n",
    "validate_layers",
    "ancestors_at_depth",
    "descendants_at_depth",
    # Dead code detection
    "find_dead_code",
    "find_orphaned_files",
    # String analysis
    "analyze_string_refs",
    "analyze_python_string_refs",
    "analyze_typescript_string_refs",
    "analyze_sql_string_refs",
    "analyze_string_topology",
    "find_string_usages",
    # String extraction (low-level)
    "extract_python_strings",
    "extract_typescript_strings",
    "extract_sql_strings",
    "is_noise",
    "SKIP_DIRS",
    "NOISE_PATTERNS",
    # Embeddings and similarity
    "compute_embeddings",
    "find_similar_code",
    "semantic_search",
    "detect_duplicates",
    # Embedding providers
    "EmbeddingProvider",
    "LocalProvider",
    "MistralProvider",
    "get_embedding_provider",
    "reset_provider",
    # Fingerprints
    "build_fingerprint_index",
    "find_similar",
    "load_fingerprint_index",
    # Impact and scout
    "get_impact",
    "AmbiguousMatchError",
    "NoMatchError",
    "project_scout",
    # Documentation analysis
    "analyze_docs",
    "extract_doc_references",
    "validate_references",
    "DocAnalysisResult",
    "DocReference",
    # Ignore patterns
    "DEFAULT_IGNORES",
    "FRAMEWORK_IGNORES",
    "nodestradamusignore_exists",
    "generate_nodestradamusignore_content",
    "generate_suggested_ignores",
    "load_ignore_patterns",
    "should_ignore",
]
