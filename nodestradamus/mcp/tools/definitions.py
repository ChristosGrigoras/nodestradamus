"""MCP Tool schema definitions.

Contains all Tool objects that define the MCP interface for Nodestradamus.
Each Tool specifies its name, description, and JSON schema for inputs.
"""

from mcp.types import Tool

# =============================================================================
# CORE ANALYSIS TOOLS
# =============================================================================

ANALYZE_DEPS_TOOL = Tool(
    name="analyze_deps",
    description=(
        "Analyze Python, TypeScript, Rust, SQL, Bash, and JSON files to extract dependency graph. "
        "Auto-detects languages. Returns a compact summary by default. "
        "Use include_fields=true to extract schema fields from classes, tables, interfaces, and structs. "
        "After creating or editing files, call again to refresh the graph for the new changes."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "languages": {
                "type": "array",
                "items": {"type": "string", "enum": ["python", "typescript", "rust", "sql", "bash", "json"]},
                "description": "Languages to analyze. If not specified, auto-detects from files present.",
            },
            "full_graph": {
                "type": "boolean",
                "description": "Return full graph instead of summary (warning: can be very large)",
                "default": False,
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top items to include in summary (default: 15)",
                "default": 15,
            },
            "include_fields": {
                "type": "boolean",
                "description": "Include field-level schema info for classes/tables/interfaces in output",
                "default": False,
            },
            "json_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns for JSON config files to include (e.g., ['**/config/*.json'])",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Directories/patterns to exclude from analysis. "
                    "If not provided, uses default ignores. "
                    "Pass suggested_ignores from project_scout for comprehensive filtering."
                ),
            },
            "package": {
                "type": "string",
                "description": (
                    "For monorepos: analyze only this package path (e.g., 'libs/core'). "
                    "Use project_scout to discover available packages."
                ),
            },
            "workspace_path": {
                "type": "string",
                "description": "Optional workspace path; when set, saves a snapshot for get_changes_since_last",
            },
        },
        "required": ["repo_path"],
    },
)

ANALYZE_COOCCURRENCE_TOOL = Tool(
    name="analyze_cooccurrence",
    description="Analyze git history to identify files that frequently change together. Returns top co-occurring file pairs.",
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "commits": {
                "type": "integer",
                "description": "Number of recent commits to analyze (default: 500)",
                "default": 500,
            },
            "full_graph": {
                "type": "boolean",
                "description": "Return full graph instead of summary",
                "default": False,
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top pairs to include (default: 20)",
                "default": 20,
            },
        },
        "required": ["repo_path"],
    },
)

GET_IMPACT_TOOL = Tool(
    name="get_impact",
    description=(
        "Analyze the impact of changing a specific file or symbol. Shows what depends on it "
        "and what it depends on. Supports fusion_mode for combined graph+semantic ranking. "
        "Use refactor_mode for symbol-level analysis when planning refactors. "
        "After saving a new or edited file, call again to get impact for that file."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository root",
            },
            "file_path": {
                "type": "string",
                "description": "Relative path to the file being changed",
            },
            "symbol": {
                "type": "string",
                "description": "Optional specific function/class name",
            },
            "depth": {
                "type": "integer",
                "description": "How many levels of dependencies to traverse (default: 3)",
                "default": 3,
            },
            "include_semantic": {
                "type": "boolean",
                "description": "Include semantically similar code (slower, requires computing embeddings). Default: false",
                "default": False,
            },
            "fusion_mode": {
                "type": "boolean",
                "description": (
                    "Enable fused impact analysis. Combines graph proximity with semantic similarity "
                    "to produce a single ranked list of matches. Uses the graph to narrow search space, "
                    "then applies embeddings only to nodes in the blast radius. Default: false"
                ),
                "default": False,
            },
            "refactor_mode": {
                "type": "boolean",
                "description": (
                    "Enable refactoring analysis. Adds symbol_usage (which specific symbols are imported "
                    "by external files with counts), duplicates (identical code blocks with line ranges), "
                    "clusters (logical function groupings for splitting files), and breaking_changes "
                    "(symbols that external code depends on). Best for planning refactors or file splits. "
                    "Default: false"
                ),
                "default": False,
            },
            "compact": {
                "type": "boolean",
                "description": "Group results by file instead of flat list (default: true)",
                "default": True,
            },
            "exclude_same_file": {
                "type": "boolean",
                "description": "Exclude symbols from the same file as target (default: true)",
                "default": True,
            },
            "exclude_external": {
                "type": "boolean",
                "description": (
                    "Exclude external/stdlib imports from results (default: true). "
                    "Filters out typing, collections, os, threading, etc. from downstream dependencies "
                    "to focus on project code. Set to false to see all dependencies including stdlib."
                ),
                "default": True,
            },
        },
        "required": ["repo_path", "file_path"],
    },
)

PROJECT_SCOUT_TOOL = Tool(
    name="project_scout",
    description="Quick reconnaissance of a repository. Returns language distribution, key directories, frameworks, and recommended Nodestradamus tools. Run this first on unfamiliar codebases.",
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "workspace_path": {
                "type": "string",
                "description": "Optional workspace path; when set, saves a snapshot for get_changes_since_last",
            },
        },
        "required": ["repo_path"],
    },
)

ANALYZE_DOCS_TOOL = Tool(
    name="analyze_docs",
    description=(
        "Analyze documentation for stale references and coverage. "
        "Parses markdown files (README, docs/) to extract code references, "
        "validates them against the codebase, and reports broken links and "
        "undocumented exports."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "docs_path": {
                "type": "string",
                "description": "Subdirectory containing docs (default: auto-detect docs/)",
            },
            "include_readme": {
                "type": "boolean",
                "description": "Include README.md files in analysis (default: true)",
                "default": True,
            },
        },
        "required": ["repo_path"],
    },
)

# =============================================================================
# CONSOLIDATED TOOLS - These replace multiple individual tools with mode/algorithm params
# =============================================================================

ANALYZE_GRAPH_TOOL = Tool(
    name="analyze_graph",
    description=(
        "Run graph algorithms on the dependency graph. "
        "Algorithms: pagerank (importance ranking), betweenness (bottleneck detection), "
        "communities (module clustering with cohesion metrics, categories: source/tests/stdlib/external), "
        "cycles (circular dependency detection), "
        "path (shortest dependency path), hierarchy (collapsed view at package/module/class level), "
        "layers (validate layered architecture compliance)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "algorithm": {
                "type": "string",
                "enum": ["pagerank", "betweenness", "communities", "cycles", "path", "hierarchy", "layers"],
                "description": "Algorithm to run",
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top items to return (for pagerank/betweenness, default: 20)",
                "default": 20,
            },
            "max_cycles": {
                "type": "integer",
                "description": "Maximum cycles to return (for cycles algorithm, default: 20)",
                "default": 20,
            },
            "source": {
                "type": "string",
                "description": "Source node ID (required for path algorithm)",
            },
            "target": {
                "type": "string",
                "description": "Target node ID (required for path algorithm)",
            },
            "level": {
                "type": "string",
                "enum": ["package", "module", "class", "function"],
                "description": "Hierarchy level for collapsed view (for hierarchy algorithm, default: module)",
                "default": "module",
            },
            "layers": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "description": (
                    "Layer definitions for architecture validation (required for layers algorithm). "
                    "Ordered list of layers from top (API/presentation) to bottom (domain/infrastructure). "
                    "Each layer is a list of path prefixes. Example: [[\"api/\", \"routes/\"], [\"services/\"], [\"models/\"]]"
                ),
            },
            "layer_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Human-readable names for layers (for layers algorithm). Example: [\"API\", \"Services\", \"Domain\"]",
            },
            "exclude_tests": {
                "type": "boolean",
                "description": (
                    "Exclude test files from results (for pagerank/betweenness). "
                    "Filters out test_*.py, *_test.py, and files in tests/ directories. Default: true"
                ),
                "default": True,
            },
            "exclude_external": {
                "type": "boolean",
                "description": (
                    "Exclude external/third-party imports from results (for pagerank/betweenness/hierarchy). "
                    "Filters out standard library and packages without source files. "
                    "Default: true for pagerank/betweenness, false for hierarchy"
                ),
                "default": True,
            },
            "scope": {
                "type": "string",
                "enum": ["all", "source_only", "tests_only"],
                "description": (
                    "Scope of analysis (for pagerank/betweenness). "
                    "'source_only' excludes tests, 'tests_only' includes only tests, 'all' includes everything. "
                    "Overrides exclude_tests if set. Default: source_only"
                ),
                "default": "source_only",
            },
            "summary_only": {
                "type": "boolean",
                "description": (
                    "Return concise summary instead of full results (for communities/hierarchy). "
                    "Shows top 5 items with key metrics. Use for initial exploration of large codebases. "
                    "For hierarchy: nodes without file paths are classified as '[stdlib]' or '[external]'. "
                    "For communities: modules are classified as 'source', 'tests', 'stdlib', or 'external'. "
                    "Default: false"
                ),
                "default": False,
            },
        },
        "required": ["repo_path", "algorithm"],
    },
)

ANALYZE_STRINGS_TOOL = Tool(
    name="analyze_strings",
    description=(
        "Analyze string literals in codebase. "
        "Modes: refs (find shared strings across files), "
        "usages (find where a specific string is used), "
        "filter (clean noisy results from refs output)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "mode": {
                "type": "string",
                "enum": ["refs", "usages", "filter"],
                "description": "Analysis mode",
            },
            "target_string": {
                "type": "string",
                "description": "The string to search for (required for 'usages' mode)",
            },
            "strings": {
                "type": "array",
                "items": {"type": "object"},
                "description": "String objects from refs output (required for 'filter' mode)",
            },
            "min_files": {
                "type": "integer",
                "description": "Minimum files referencing a string (for refs mode, default: 2)",
                "default": 2,
            },
            "top_n": {
                "type": "integer",
                "description": "Limit results to top N strings (for refs mode, default: 50)",
                "default": 50,
            },
            "exclude_types": {
                "type": "boolean",
                "description": "Exclude type annotations (for filter mode, default: true)",
                "default": True,
            },
            "exclude_imports": {
                "type": "boolean",
                "description": "Exclude import paths (for filter mode, default: true)",
                "default": True,
            },
            "exclude_css": {
                "type": "boolean",
                "description": "Exclude CSS classes (for filter mode, default: true)",
                "default": True,
            },
            "summary_only": {
                "type": "boolean",
                "description": (
                    "Return concise summary instead of full results (for refs mode). "
                    "Shows top 5 strings with key metrics. Use for initial exploration. "
                    "Default: false"
                ),
                "default": False,
            },
        },
        "required": ["repo_path", "mode"],
    },
)

SEMANTIC_ANALYSIS_TOOL = Tool(
    name="semantic_analysis",
    description=(
        "Semantic code analysis using embeddings. "
        "Modes: search (natural language code search), "
        "similar (find related code to a query/file/symbol), "
        "duplicates (find copy-pasted code), "
        "embeddings (compute/refresh embedding cache). "
        "Results from search/similar include a 'snippet' field with a short code preview. "
        "Duplicates include 'preview_a', 'preview_b', and 'preview' fields for code previews."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository",
            },
            "mode": {
                "type": "string",
                "enum": ["search", "similar", "duplicates", "embeddings"],
                "description": "Analysis mode",
            },
            "query": {
                "type": "string",
                "description": "Natural language query or code snippet (for search/similar modes)",
            },
            "file_path": {
                "type": "string",
                "description": "Find code similar to this file (for similar mode)",
            },
            "symbol": {
                "type": "string",
                "description": "Find code similar to this function/class (for similar mode)",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 10)",
                "default": 10,
            },
            "threshold": {
                "type": "number",
                "description": "Minimum similarity score 0-1 (default: 0.5 for similar, 0.3 for search, 0.9 for duplicates)",
                "default": 0.5,
            },
            "max_pairs": {
                "type": "integer",
                "description": "Maximum duplicate pairs to return (for duplicates mode, default: 50)",
                "default": 50,
            },
            "chunk_by": {
                "type": "string",
                "enum": ["function", "file"],
                "description": "How to chunk code (for embeddings mode, default: function)",
                "default": "function",
            },
            "workspace_path": {
                "type": "string",
                "description": (
                    "Workspace path for cache isolation. When provided, caches are stored "
                    "in <workspace>/.nodestradamus/cache/ instead of the analyzed repo. "
                    "Use this for isolation between different Cursor windows."
                ),
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Directories/patterns to exclude from analysis. "
                    "If not provided, uses default ignores. "
                    "Pass suggested_ignores from project_scout for comprehensive filtering."
                ),
            },
            "package": {
                "type": "string",
                "description": (
                    "For monorepos: limit analysis to this package path (e.g., 'libs/core'). "
                    "Use project_scout to discover available packages. "
                    "Embeddings are cached separately per package."
                ),
            },
        },
        "required": ["repo_path", "mode"],
    },
)

# =============================================================================
# WORKFLOW AND CACHE TOOLS
# =============================================================================

QUICK_START_TOOL = Tool(
    name="quick_start",
    description=(
        "Run the optimal setup sequence for a new codebase. "
        "Executes: project_scout → analyze_deps → codebase_health → semantic_analysis (embeddings). "
        "Returns a combined report with repository overview, dependency summary, health check, "
        "and confirms embeddings are ready for fast semantic search. "
        "Use this as your first tool on any unfamiliar codebase. "
        "After adding new files or modules, run again to refresh the report."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "skip_embeddings": {
                "type": "boolean",
                "description": (
                    "Skip embedding computation (faster, but semantic search will be slow on first use). "
                    "Default: false"
                ),
                "default": False,
            },
            "health_checks": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["dead_code", "duplicates", "cycles", "bottlenecks"],
                },
                "description": "Which health checks to run (default: cycles, bottlenecks)",
                "default": ["cycles", "bottlenecks"],
            },
        },
        "required": ["repo_path"],
    },
)

CODEBASE_HEALTH_TOOL = Tool(
    name="codebase_health",
    description=(
        "Comprehensive codebase health check. Runs multiple analyses in one call. "
        "Checks: dead_code (unused functions/classes), duplicates (copy-pasted code), "
        "cycles (circular dependencies), bottlenecks (high-impact nodes), "
        "docs (stale documentation references). "
        "Returns a unified report with summary and findings. "
        "After creating or editing files, call again to include them in the health report."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "checks": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["dead_code", "duplicates", "cycles", "bottlenecks", "docs"],
                },
                "description": "Which checks to run (default: all)",
                "default": ["dead_code", "duplicates", "cycles", "bottlenecks", "docs"],
            },
            "max_items": {
                "type": "integer",
                "description": "Maximum items to return per check (default: 20)",
                "default": 20,
            },
            "scope": {
                "type": "string",
                "enum": ["all", "source_only", "tests_only"],
                "description": (
                    "Scope for bottleneck analysis. 'source_only' excludes test files (default), "
                    "'tests_only' includes only tests, 'all' includes everything."
                ),
                "default": "source_only",
            },
            "exclude_external": {
                "type": "boolean",
                "description": (
                    "Exclude external/stdlib imports from bottleneck analysis (default: true). "
                    "Filters out typing, os, collections, etc. to focus on project code."
                ),
                "default": True,
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Directories/patterns to exclude from analysis. "
                    "If not provided, auto-loads from .nodestradamusignore and defaults. "
                    "Pass suggested_ignores from project_scout for comprehensive filtering."
                ),
            },
            "workspace_path": {
                "type": "string",
                "description": "Optional workspace path; when set, saves a snapshot for get_changes_since_last",
            },
        },
        "required": ["repo_path"],
    },
)

FIND_SIMILAR_TOOL = Tool(
    name="find_similar",
    description=(
        "Find structurally similar code to a file or region. Uses Shazam-like fingerprinting "
        "from the parse cache (node type + edge type pairs). Returns locations with most "
        "overlapping structural patterns. Run analyze_deps first to ensure parse cache is warm."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository",
            },
            "file_path": {
                "type": "string",
                "description": "Relative path to the file (anchor for similarity)",
            },
            "line_start": {
                "type": "integer",
                "description": "Optional start line (1-based); omit for whole file",
            },
            "line_end": {
                "type": "integer",
                "description": "Optional end line (1-based); omit for whole file",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of similar locations to return (default: 15)",
                "default": 15,
            },
        },
        "required": ["repo_path", "file_path"],
    },
)

GET_CHANGES_SINCE_LAST_TOOL = Tool(
    name="get_changes_since_last",
    description=(
        "Compare current codebase state to the last Nodestradamus run. Loads snapshots saved when "
        "project_scout, analyze_deps, or codebase_health were called with workspace_path. "
        "Re-runs the selected tool(s) using caches and returns a diff (added/removed/changed). "
        "Use to focus on what changed since the assistant last analyzed the repo."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository",
            },
            "workspace_path": {
                "type": "string",
                "description": "Workspace path where snapshots are stored",
            },
            "tool": {
                "type": "string",
                "enum": ["project_scout", "analyze_deps", "codebase_health", "all"],
                "description": "Which tool snapshot(s) to diff (default: all)",
                "default": "all",
            },
        },
        "required": ["repo_path", "workspace_path"],
    },
)

MANAGE_CACHE_TOOL = Tool(
    name="manage_cache",
    description=(
        "Manage Nodestradamus embedding caches. "
        "Modes: info (show cache details for a repo), "
        "clear (delete cache for a repo), "
        "list (show all caches in workspace)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["info", "clear", "list"],
                "description": "Cache management mode",
            },
            "repo_path": {
                "type": "string",
                "description": "Repository path (required for info/clear modes)",
            },
            "workspace_path": {
                "type": "string",
                "description": (
                    "Workspace path. For 'list' mode, shows all caches in this workspace. "
                    "For 'info'/'clear' modes, specifies where to look for workspace-scoped cache."
                ),
            },
        },
        "required": ["mode"],
    },
)

COMPARE_RULES_TOOL = Tool(
    name="compare_rules_to_codebase",
    description=(
        "Compare existing rules (Cursor, OpenCode, or Claude Code) with codebase analysis. "
        "Discovers rules from .cursor/rules/, AGENTS.md, .claude/, or a custom path. "
        "Derives inferred facets (critical files, bottlenecks) from dependency analysis. "
        "Returns coverage (which hotspots are documented), gaps (undocumented hotspots), "
        "stale references (rule paths that don't exist), and recommendations. "
        "Works even when no rules exist — returns inferred facets and suggestions to create rules."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "rules_sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["cursor", "opencode", "claude"]},
                "description": (
                    "Which rule sources to check. Default: all three (auto-detect). "
                    "Or provide a single custom path string to use only that path."
                ),
            },
            "custom_rules_path": {
                "type": "string",
                "description": (
                    "Custom path to rules file or directory (overrides rules_sources). "
                    "Can be absolute or relative to repo_path."
                ),
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top pagerank/betweenness nodes for inferred facets (default: 15)",
                "default": 15,
            },
            "include_duplicates": {
                "type": "boolean",
                "description": "Run duplicate detection and include in inferred facets (slower, default: false)",
                "default": False,
            },
            "include_cycles": {
                "type": "boolean",
                "description": "Run cycle detection and include in report (default: false)",
                "default": False,
            },
        },
        "required": ["repo_path"],
    },
)

# =============================================================================
# RULES VALIDATION AND CONFLICT DETECTION
# =============================================================================

VALIDATE_RULES_TOOL = Tool(
    name="validate_rules",
    description=(
        "Validate rule files in .cursor/rules/, .claude/, or AGENTS.md. "
        "Checks YAML frontmatter syntax, required fields, unique rule numbering, "
        "referenced file existence, token budget compliance, and rule format consistency. "
        "Returns validation report with errors, warnings, and info per file."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "rules_source": {
                "type": "string",
                "enum": ["cursor", "opencode", "claude"],
                "description": (
                    "Which rule source to check. Default: auto-detect (cursor, then claude). "
                    "cursor = .cursor/rules/*.mdc, opencode = AGENTS.md, claude = .claude/"
                ),
            },
            "custom_rules_path": {
                "type": "string",
                "description": (
                    "Custom path to rules directory (overrides rules_source). "
                    "Can be absolute or relative to repo_path."
                ),
            },
        },
        "required": ["repo_path"],
    },
)

DETECT_RULE_CONFLICTS_TOOL = Tool(
    name="detect_rule_conflicts",
    description=(
        "Detect potential conflicts between AI rules. "
        "Analyzes rules for contradictory directives including: "
        "naming conventions (snake_case vs camelCase), testing frameworks (pytest vs unittest), "
        "import styles (absolute vs relative), documentation styles, and type hints. "
        "Returns conflicts grouped by severity (error, warning) with recommendations."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository to analyze",
            },
            "rules_source": {
                "type": "string",
                "enum": ["cursor", "opencode", "claude"],
                "description": (
                    "Which rule source to check. Default: auto-detect (cursor, then claude). "
                    "cursor = .cursor/rules/*.mdc, claude = .claude/"
                ),
            },
            "custom_rules_path": {
                "type": "string",
                "description": (
                    "Custom path to rules directory (overrides rules_source). "
                    "Can be absolute or relative to repo_path."
                ),
            },
        },
        "required": ["repo_path"],
    },
)

# =============================================================================
# ALL TOOLS LIST - Used by register_graph_tools
# =============================================================================

ALL_TOOLS = [
    # Core tools
    PROJECT_SCOUT_TOOL,
    ANALYZE_DEPS_TOOL,
    ANALYZE_COOCCURRENCE_TOOL,
    GET_IMPACT_TOOL,
    ANALYZE_DOCS_TOOL,
    # Consolidated tools
    ANALYZE_GRAPH_TOOL,
    ANALYZE_STRINGS_TOOL,
    SEMANTIC_ANALYSIS_TOOL,
    # Workflow tools
    QUICK_START_TOOL,
    CODEBASE_HEALTH_TOOL,
    COMPARE_RULES_TOOL,
    # Rules validation and conflicts
    VALIDATE_RULES_TOOL,
    DETECT_RULE_CONFLICTS_TOOL,
    # Fingerprint and snapshot diff
    FIND_SIMILAR_TOOL,
    GET_CHANGES_SINCE_LAST_TOOL,
    # Cache management
    MANAGE_CACHE_TOOL,
]

__all__ = [
    # Individual tools
    "ANALYZE_DEPS_TOOL",
    "ANALYZE_COOCCURRENCE_TOOL",
    "GET_IMPACT_TOOL",
    "PROJECT_SCOUT_TOOL",
    "ANALYZE_DOCS_TOOL",
    "ANALYZE_GRAPH_TOOL",
    "ANALYZE_STRINGS_TOOL",
    "SEMANTIC_ANALYSIS_TOOL",
    "QUICK_START_TOOL",
    "CODEBASE_HEALTH_TOOL",
    "COMPARE_RULES_TOOL",
    "VALIDATE_RULES_TOOL",
    "DETECT_RULE_CONFLICTS_TOOL",
    "FIND_SIMILAR_TOOL",
    "GET_CHANGES_SINCE_LAST_TOOL",
    "MANAGE_CACHE_TOOL",
    # List of all tools
    "ALL_TOOLS",
]
