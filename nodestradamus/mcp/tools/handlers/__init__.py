"""MCP tool handlers.

Each handler processes tool calls and returns JSON responses.
Handlers are organized by domain:
- core: analyze_deps, analyze_cooccurrence, get_impact, project_scout, analyze_docs
- graph_algorithms: analyze_graph (pagerank, betweenness, communities, cycles, path)
- semantic: semantic_analysis (search, similar, duplicates, embeddings)
- strings: analyze_strings (refs, usages, filter)
- cache: manage_cache (info, clear, list)
- workflows: codebase_health, quick_start, get_changes_since_last
- rules_audit: compare_rules_to_codebase
- rules_validation: validate_rules, detect_rule_conflicts
- fingerprints: find_similar
"""

from nodestradamus.mcp.tools.handlers.cache import handle_manage_cache
from nodestradamus.mcp.tools.handlers.core import (
    handle_analyze_cooccurrence,
    handle_analyze_deps,
    handle_analyze_docs,
    handle_get_impact,
    handle_project_scout,
)
from nodestradamus.mcp.tools.handlers.fingerprints import handle_find_similar
from nodestradamus.mcp.tools.handlers.graph_algorithms import handle_analyze_graph
from nodestradamus.mcp.tools.handlers.rules_audit import handle_compare_rules
from nodestradamus.mcp.tools.handlers.rules_validation import (
    handle_detect_conflicts,
    handle_validate_rules,
)
from nodestradamus.mcp.tools.handlers.semantic import handle_semantic_analysis
from nodestradamus.mcp.tools.handlers.strings import handle_analyze_strings
from nodestradamus.mcp.tools.handlers.workflows import (
    handle_codebase_health,
    handle_get_changes_since_last,
    handle_quick_start,
)

__all__ = [
    # Core handlers
    "handle_analyze_deps",
    "handle_analyze_cooccurrence",
    "handle_get_impact",
    "handle_project_scout",
    "handle_analyze_docs",
    "handle_find_similar",
    # Consolidated handlers
    "handle_analyze_graph",
    "handle_analyze_strings",
    "handle_semantic_analysis",
    # Workflow handlers
    "handle_quick_start",
    "handle_codebase_health",
    "handle_get_changes_since_last",
    "handle_compare_rules",
    # Rules validation handlers
    "handle_validate_rules",
    "handle_detect_conflicts",
    # Cache handlers
    "handle_manage_cache",
]
