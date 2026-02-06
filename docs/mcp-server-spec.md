# Nodestradamus MCP Server Specification

> **Implementation Prompt for Python MCP Server**
> 
> This document specifies the Nodestradamus MCP server implementation. Use this as the authoritative reference for building the server.

---

## Overview

**Nodestradamus** is an MCP (Model Context Protocol) server that provides codebase intelligence to AI assistants like Cursor, Claude Desktop, and other MCP-compatible clients.

**Core Value Proposition**: Graph-based code understanding + rule management that AI coding assistants don't have built-in.

> **New to Nodestradamus?** See the [Getting Started Workflow](getting-started-workflow.md) for the optimal tool sequence.

---

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.12+ | Existing codebase, mature MCP SDK |
| **MCP SDK** | `mcp` (official) | `pip install mcp` |
| **Async Runtime** | `asyncio` | Required by MCP SDK |
| **AST Parsing (Python)** | `ast` module | Native, no dependencies |
| **AST Parsing (TS/JS)** | `tree-sitter` | Already in use |
| **Git Operations** | `subprocess` | Call git binary |
| **Graph Algorithms** | `networkx` (optional) | For advanced features |

---

## Package Structure

```
nodestradamus/
├── __init__.py
├── __main__.py              # Entry point: python -m nodestradamus
├── cli.py                   # CLI commands (serve, analyze, etc.)
├── mcp/
│   ├── __init__.py
│   ├── server.py            # MCP server implementation
│   ├── context.py           # Workspace/cache context
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── definitions.py   # MCP Tool schemas
│   │   ├── dispatch.py      # Tool routing
│   │   ├── handlers/        # Tool implementations
│   │   │   ├── core.py       # project_scout, analyze_deps, get_impact, etc.
│   │   │   ├── graph_algorithms.py
│   │   │   ├── strings.py
│   │   │   ├── semantic.py
│   │   │   ├── workflows.py # quick_start, codebase_health
│   │   │   ├── cache.py      # manage_cache
│   │   │   ├── rules_audit.py # compare_rules_to_codebase, analyze_docs
│   │   │   └── ...
│   │   └── utils/
│   └── resources/
│       ├── __init__.py
│       └── graph_resources.py
├── analyzers/
│   ├── __init__.py
│   ├── deps.py              # Multi-language dependency analysis
│   ├── code_parser.py       # AST/parse cache
│   ├── git_cooccurrence.py  # Git co-occurrence
│   ├── project_scout.py    # Repo reconnaissance
│   ├── impact.py            # Impact analysis
│   ├── graph_algorithms.py # PageRank, betweenness, etc.
│   ├── embeddings.py       # Semantic embeddings
│   ├── docs.py              # Doc analysis (stale refs)
│   ├── string_extraction.py
│   ├── string_refs.py
│   └── ...
├── models/
│   ├── __init__.py
│   └── graph.py             # Graph data models
└── utils/
    ├── cache.py             # get_cache_dir, repo hash
    └── ...
```

---

## MCP Server Configuration

### Server Metadata

```python
SERVER_NAME = "nodestradamus"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSION = "2024-11-05"
```

### Capabilities

```python
capabilities = {
    "tools": True,
    "resources": True,
    "prompts": False  # Future: add prompt templates
}
```

---

## MCP Tools Specification

### Consolidated Tools (Recommended)

These consolidated tools replace multiple individual tools with mode/algorithm parameters for a better AI experience.

#### Tool: `analyze_graph`

**Purpose**: Run graph algorithms on the dependency graph.

```python
@tool
async def analyze_graph(
    repo_path: str,
    algorithm: str,  # pagerank, betweenness, communities, cycles, path
    top_n: int = 20,
    max_cycles: int = 20,
    source: str = None,  # for path algorithm
    target: str = None,  # for path algorithm
) -> dict:
    """
    Run graph algorithms on the dependency graph.
    
    Algorithms:
    - pagerank: Rank code by importance (most depended upon)
    - betweenness: Find bottleneck nodes where changes ripple widely
    - communities: Detect module clusters using Louvain algorithm
    - cycles: Find circular dependencies (import cycles)
    - path: Find shortest dependency path between two nodes
    """
```

#### Tool: `analyze_strings`

**Purpose**: Analyze string literals in the codebase.

```python
@tool
async def analyze_strings(
    repo_path: str,
    mode: str,  # refs, usages, filter
    target_string: str = None,  # for usages mode
    strings: list = None,  # for filter mode
    min_files: int = 2,
    top_n: int = 50,
) -> dict:
    """
    Analyze string literals in the codebase.
    
    Modes:
    - refs: Find shared strings across files
    - usages: Find where a specific string is used
    - filter: Clean noisy results from refs output
    """
```

#### Tool: `semantic_analysis`

**Purpose**: Semantic code analysis using embeddings.

```python
@tool
async def semantic_analysis(
    repo_path: str,
    mode: str,  # search, similar, duplicates, embeddings
    query: str = None,
    file_path: str = None,
    symbol: str = None,
    top_k: int = 10,
    threshold: float = 0.5,
) -> dict:
    """
    Semantic code analysis using embeddings.
    
    Modes:
    - search: Natural language code search
    - similar: Find related code to a query/file/symbol
    - duplicates: Find copy-pasted code
    - embeddings: Compute/refresh embedding cache
    """
```

---

### Workflow Tools

These tools orchestrate multiple analysis steps for common workflows.

#### Tool: `quick_start`

**Purpose**: Run the optimal setup sequence for a new codebase automatically.

```python
@tool
async def quick_start(
    repo_path: str,
    skip_embeddings: bool = False,
    health_checks: list = ["cycles", "bottlenecks"],
) -> dict:
    """
    Run optimal setup sequence for a new codebase:
    1. project_scout - Get overview and suggested_ignores
    2. analyze_deps - Build dependency graph (using suggested_ignores)
    3. codebase_health - Run health checks
    4. semantic_analysis embeddings - Pre-compute for fast search (optional)
    
    Args:
        repo_path: Absolute path to the repository
        skip_embeddings: Skip slow embedding step (default: false)
        health_checks: Which health checks to run (default: cycles, bottlenecks)
    
    Returns:
        Combined report with scout, deps, health, and embedding status
    """
```

**Output Schema**:
```json
{
    "status": "success",
    "steps_completed": ["project_scout", "analyze_deps", "codebase_health", "semantic_analysis_embeddings"],
    "scout": {
        "primary_language": "python",
        "languages": {"python": 150, "typescript": 20},
        "frameworks": ["pytest", "pydantic"],
        "suggested_ignores": ["node_modules", "venv", "__pycache__"],
        "lazy_options": [{"option": "LazyEmbeddingGraph", "when": "monorepo or 5K+ source files", "description": "Load scope with load_scope()..."}]
    },
    "deps": {
        "nodes": 450,
        "edges": 1200,
        "files": 85,
        "top_modules": [...]
    },
    "health": {
        "cycle_count": 2,
        "top_bottlenecks": [...]
    },
    "embeddings": {
        "chunks_indexed": 320,
        "ready_for_search": true
    },
    "timings": {
        "project_scout_ms": 150,
        "analyze_deps_ms": 800,
        "health_check_ms": 200,
        "embeddings_ms": 45000,
        "total_ms": 46150
    },
    "next_steps": [
        "semantic_analysis mode=search — Fast semantic code search",
        "analyze_graph algorithm=pagerank — Find most critical code"
    ]
}
```

---

#### Tool: `compare_rules_to_codebase`

**Purpose**: Compare existing rules with codebase analysis to find coverage, gaps, and stale references.

```python
@tool
async def compare_rules_to_codebase(
    repo_path: str,
    rules_sources: list = ["cursor", "opencode", "claude"],
    custom_rules_path: str = None,
    top_n: int = 15,
    include_duplicates: bool = False,
    include_cycles: bool = False,
) -> dict:
    """
    Compare existing rules with codebase analysis.
    
    Discovers rules from multiple sources:
    - Cursor: .cursor/rules/*.mdc
    - OpenCode: AGENTS.md, CLAUDE.md
    - Claude Code: .claude/CLAUDE.md, .claude/rules/**/*.md
    
    Derives inferred facets from codebase (critical files, bottlenecks).
    Compares rule content to hotspots and returns coverage analysis.
    
    Works even when no rules exist — returns inferred facets and 
    recommendations to help create rules.
    
    Args:
        repo_path: Absolute path to the repository
        rules_sources: Which rule sources to check (default: all three)
        custom_rules_path: Optional custom path to rules (overrides rules_sources)
        top_n: Number of top pagerank/betweenness nodes (default: 15)
        include_duplicates: Run duplicate detection (slower, default: false)
        include_cycles: Run cycle detection (default: false)
    
    Returns:
        Comparison report with coverage, gaps, stale refs, and recommendations
    """
```

**Output Schema**:
```json
{
    "existing_rules_summary": {
        "count": 5,
        "sources_checked": ["cursor", "opencode", "claude"],
        "sources_found": ["cursor"],
        "rules": [
            {"file": ".cursor/rules/100-python.mdc", "source": "cursor", "code_paths_count": 3}
        ]
    },
    "inferred_facets": {
        "structure": {
            "primary_language": "python",
            "languages": {"python": 50},
            "frameworks": ["pytest", "pydantic"],
            "is_monorepo": false
        },
        "critical_files": [
            {"node": "analyze_deps", "file": "nodestradamus/analyzers/deps.py", "importance": 0.0045}
        ],
        "bottlenecks": [
            {"node": "handle_request", "file": "nodestradamus/mcp/server.py", "betweenness": 0.003}
        ]
    },
    "coverage": {
        "nodestradamus/analyzers/deps.py": {"mentioned_in": ["100-project.mdc"], "is_covered": true}
    },
    "gaps": [
        {"path": "nodestradamus/mcp/server.py", "type": "bottleneck", "suggestion": "Consider adding guidance"}
    ],
    "stale": [
        {"rule": "100-project.mdc", "path": "old_module.py", "source": "cursor"}
    ],
    "recommendations": [
        "Low hotspot coverage (40%). Consider adding guidance for undocumented critical files.",
        "Top undocumented hotspots: nodestradamus/mcp/server.py, nodestradamus/cli.py"
    ],
    "summary": {
        "rules_count": 5,
        "hotspots_count": 15,
        "covered_count": 6,
        "gaps_count": 9,
        "stale_count": 1,
        "coverage_percent": 40.0
    }
}
```

**Use Cases**:
- Audit existing rules against actual codebase structure
- Find critical files that lack documentation in rules
- Detect stale references to renamed/deleted files
- Get recommendations for improving rule coverage
- Works on repos with no existing rules (returns inferred facets only)

---

### Legacy Tools (Deprecated)

The following tools are deprecated and will be removed in a future release.
Use the consolidated tools above instead.

| Old Tool | New Tool | Migration |
|----------|----------|-----------|
| `rank_importance` | `analyze_graph` | `algorithm="pagerank"` |
| `find_bottlenecks` | `analyze_graph` | `algorithm="betweenness"` |
| `detect_modules` | `analyze_graph` | `algorithm="communities"` |
| `find_circular_deps` | `analyze_graph` | `algorithm="cycles"` |
| `dependency_path` | `analyze_graph` | `algorithm="path"` |
| `analyze_string_refs` | `analyze_strings` | `mode="refs"` |
| `find_string_usages` | `analyze_strings` | `mode="usages"` |
| `filter_strings` | `analyze_strings` | `mode="filter"` |
| `compute_embeddings` | `semantic_analysis` | `mode="embeddings"` |
| `find_similar_code` | `semantic_analysis` | `mode="similar"` |
| `semantic_search` | `semantic_analysis` | `mode="search"` |
| `detect_duplicates` | `semantic_analysis` | `mode="duplicates"` |

---

### Core Tools (Kept)

#### Tool: `analyze_deps` (formerly `analyze_python`)

**Purpose**: Extract dependency graph from Python and/or TypeScript codebase.

```python
@tool
async def analyze_deps(repo_path: str, languages: list = None) -> dict:
    """
    Analyze Python and/or TypeScript files to extract dependency graph.
    Auto-detects languages if not specified.
    
    Args:
        repo_path: Absolute path to repository root
        languages: Optional list of ["python", "typescript"]
        
    Returns:
        DependencyGraph with nodes and edges
    """
```

**Input Schema**:
```json
{
    "type": "object",
    "properties": {
        "repo_path": {
            "type": "string",
            "description": "Absolute path to the repository to analyze"
        }
    },
    "required": ["repo_path"]
}
```

**Output Schema**:
```json
{
    "nodes": [
        {
            "id": "py:src/auth.py::login",
            "type": "function",
            "file": "src/auth.py",
            "name": "login",
            "line": 42
        }
    ],
    "edges": [
        {
            "from": "py:src/api.py::handler",
            "to": "py:src/auth.py::login",
            "type": "calls"
        }
    ],
    "metadata": {
        "analyzer": "python",
        "version": "0.2.0",
        "generated_at": "2026-01-27T12:00:00Z",
        "file_count": 47
    }
}
```

---

### Tool 2: `analyze_typescript`

**Purpose**: Extract dependency graph from TypeScript/JavaScript codebase.

```python
@tool
async def analyze_typescript(repo_path: str) -> dict:
    """
    Analyze TypeScript/JavaScript files to extract imports and exports.
    
    Args:
        repo_path: Absolute path to repository root
        
    Returns:
        DependencyGraph with nodes and edges
    """
```

**Input/Output**: Same schema as `analyze_python` but with `ts:` prefix for node IDs.

---

### Tool 3: `analyze_cooccurrence`

**Purpose**: Analyze git history to find files that change together.

```python
@tool
async def analyze_cooccurrence(
    repo_path: str,
    commits: int = 500
) -> dict:
    """
    Analyze git history to identify files that frequently change together.
    
    Args:
        repo_path: Absolute path to repository root
        commits: Number of recent commits to analyze (default: 500)
        
    Returns:
        CooccurrenceGraph with file pairs and change frequency
    """
```

**Output Schema**:
```json
{
    "nodes": ["src/auth.py", "src/models/user.py"],
    "edges": [
        {
            "from": "src/auth.py",
            "to": "src/models/user.py",
            "weight": 23,
            "type": "co-occurs"
        }
    ],
    "metadata": {
        "commits_analyzed": 500,
        "generated_at": "2026-01-27T12:00:00Z"
    }
}
```

---

### Tool 4: `merge_graphs`

**Purpose**: Combine multiple dependency graphs into a unified view.

```python
@tool
async def merge_graphs(graphs: list[dict]) -> dict:
    """
    Merge multiple dependency graphs into a unified cross-language graph.
    
    Args:
        graphs: List of graph objects from analyze_* tools
        
    Returns:
        Unified graph with normalized node IDs
    """
```

---

### Tool 5: `get_impact`

**Purpose**: Determine what would be affected by changing a file or symbol.

```python
@tool
async def get_impact(
    repo_path: str,
    file_path: str,
    symbol: str | None = None,
    depth: int = 3
) -> dict:
    """
    Analyze the impact of changing a specific file or symbol.
    
    Args:
        repo_path: Absolute path to repository root
        file_path: Relative path to the file being changed
        symbol: Optional specific function/class name
        depth: How many levels of dependencies to traverse (default: 3)
        
    Returns:
        ImpactReport with upstream and downstream dependencies
    """
```

**Output Schema**:
```json
{
    "target": "src/auth.py::login",
    "upstream": [
        {"id": "src/api.py::handler", "depth": 1},
        {"id": "src/routes.py::login_route", "depth": 2}
    ],
    "downstream": [
        {"id": "src/db.py::get_user", "depth": 1}
    ],
    "co_occurring_files": ["src/models/user.py", "tests/test_auth.py"],
    "risk_assessment": {
        "direct_dependents": 5,
        "indirect_dependents": 12,
        "test_files_affected": 3
    }
}
```

---

### Tool 6: `validate_rules`

**Purpose**: Validate Cursor rule files for correctness.

```python
@tool
async def validate_rules(rules_path: str) -> dict:
    """
    Validate Cursor rule files for proper formatting and structure.
    
    Args:
        rules_path: Path to .cursor/rules/ directory or specific .mdc file
        
    Returns:
        ValidationReport with issues and suggestions
    """
```

**Output Schema**:
```json
{
    "valid": false,
    "files_checked": 5,
    "issues": [
        {
            "file": "100-python.mdc",
            "line": 15,
            "severity": "error",
            "message": "Missing required frontmatter field: description"
        },
        {
            "file": "200-api.mdc",
            "line": 42,
            "severity": "warning",
            "message": "Directive exceeds 40 token recommendation"
        }
    ],
    "summary": {
        "errors": 1,
        "warnings": 1,
        "info": 0
    }
}
```

---

### Tool 7: `detect_conflicts`

**Purpose**: Find contradictions between rules.

```python
@tool
async def detect_conflicts(rules_path: str) -> dict:
    """
    Detect conflicts and contradictions between Cursor rules.
    
    Args:
        rules_path: Path to .cursor/rules/ directory
        
    Returns:
        ConflictReport with identified contradictions
    """
```

**Output Schema**:
```json
{
    "conflicts": [
        {
            "type": "naming_convention",
            "rule_a": {"file": "100-python.mdc", "directive": "Use snake_case"},
            "rule_b": {"file": "200-api.mdc", "directive": "Use camelCase"},
            "severity": "high",
            "suggestion": "Consolidate naming convention to one standard"
        }
    ],
    "overlaps": [
        {
            "files": ["100-python.mdc", "301-testing.mdc"],
            "glob_pattern": "**/*.py",
            "note": "Both rules apply to Python files - ensure no contradictions"
        }
    ]
}
```

---

## LLM-Powered Tools (Bring Your Own Key)

These tools require an LLM API key provided by the user. The server does not store keys—they are passed via environment variables or MCP configuration.

### Supported LLM Providers

| Provider | Env Variable | Notes |
|----------|--------------|-------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude models, recommended |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4, GPT-4o |
| **Ollama** | `OLLAMA_BASE_URL` | Local models, no key needed |

### Tool 8: `generate_rules`

**Purpose**: Analyze a codebase and generate AI rules from detected conventions.

```python
@tool
async def generate_rules(
    repo_path: str,
    output_format: str = "cursor",
    focus_areas: list[str] | None = None
) -> dict:
    """
    Analyze codebase conventions and generate AI rules.
    
    Args:
        repo_path: Absolute path to repository root
        output_format: Target format - "cursor" or "claude"
        focus_areas: Optional list of areas to focus on (e.g., ["naming", "testing", "imports"])
        
    Returns:
        Generated rules in the specified format
        
    Requires:
        LLM API key in environment (ANTHROPIC_API_KEY or OPENAI_API_KEY)
    """
```

**Input Schema**:
```json
{
    "type": "object",
    "properties": {
        "repo_path": {
            "type": "string",
            "description": "Absolute path to the repository to analyze"
        },
        "output_format": {
            "type": "string",
            "enum": ["cursor", "claude"],
            "default": "cursor",
            "description": "Target rule format"
        },
        "focus_areas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Areas to analyze: naming, imports, error_handling, testing, documentation"
        }
    },
    "required": ["repo_path"]
}
```

**Output Schema**:
```json
{
    "format": "cursor",
    "rules": [
        {
            "filename": "100-python.mdc",
            "content": "---\ndescription: Python conventions\nglobs: \"**/*.py\"\nalwaysApply: false\n---\n\n# Python\n\n## Naming\n- Use snake_case for functions and variables\n- Use PascalCase for classes\n",
            "detected_from": ["src/utils.py", "src/models.py"],
            "confidence": 0.92
        }
    ],
    "analysis_summary": {
        "files_analyzed": 47,
        "patterns_detected": 12,
        "llm_provider": "anthropic",
        "tokens_used": 2340
    }
}
```

**Token Budget Awareness**:
- Each directive is compressed to <40 tokens
- Total rule content follows meta-generator guidelines
- Router rules: 150 tokens max
- Context rules: 180 tokens max

---

### Tool 9: `suggest_rule`

**Purpose**: Generate a rule from user-observed correction patterns.

```python
@tool
async def suggest_rule(
    pattern_description: str,
    examples: list[dict],
    output_format: str = "cursor"
) -> dict:
    """
    Generate a rule from user-provided correction patterns.
    
    Args:
        pattern_description: Natural language description of the pattern
        examples: List of before/after code examples showing the correction
        output_format: Target format - "cursor" or "claude"
        
    Returns:
        Generated rule based on the pattern
        
    Requires:
        LLM API key in environment
    """
```

**Input Schema**:
```json
{
    "type": "object",
    "properties": {
        "pattern_description": {
            "type": "string",
            "description": "Natural language description of the correction pattern"
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "file_type": {"type": "string"}
                }
            },
            "description": "Before/after examples of the correction"
        },
        "output_format": {
            "type": "string",
            "enum": ["cursor", "claude"],
            "default": "cursor"
        }
    },
    "required": ["pattern_description", "examples"]
}
```

**Output Schema**:
```json
{
    "rule": {
        "filename": "310-custom-pattern.mdc",
        "content": "---\ndescription: Custom pattern from user corrections\nglobs: \"**/*.py\"\nalwaysApply: false\n---\n\n# Custom Pattern\n\n## Conventions\n- Always use explicit return types\n",
        "confidence": 0.85
    },
    "reasoning": "Detected pattern: user consistently adds return type annotations to functions",
    "tokens_used": 890
}
```

---

### LLM Configuration

**MCP Config with API Key**:
```json
{
    "mcpServers": {
        "nodestradamus": {
            "command": "nodestradamus",
            "args": ["serve"],
            "env": {
                "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
            }
        }
    }
}
```

**Fallback Behavior**:
- If no API key is provided, `generate_rules` and `suggest_rule` return an error with instructions
- All other tools work without an API key

**Error Response (No Key)**:
```json
{
    "error": {
        "code": -32001,
        "message": "LLM API key required",
        "data": {
            "type": "ConfigurationError",
            "detail": "Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable",
            "docs": "https://github.com/your-org/nodestradamus#llm-configuration"
        }
    }
}
```

---

## MCP Resources Specification

### Resource 1: `graph://python/{repo_path}`

Cached Python dependency graph for a repository.

### Resource 2: `graph://typescript/{repo_path}`

Cached TypeScript dependency graph for a repository.

### Resource 3: `rules://validation/{rules_path}`

Cached validation results for rules.

---

## Remote Repository Support

Nodestradamus supports analyzing remote repositories without requiring local clones. This enables "Bring Your Own Repo" workflows.

### Repository Access Modes

| Mode | Input Format | Use Case |
|------|--------------|----------|
| **Local Path** | `/home/user/project` | Default, fastest |
| **HTTPS Clone** | `https://github.com/org/repo` | Public repos, CI/CD |
| **SSH Clone** | `git@github.com:org/repo.git` | Private repos with SSH keys |
| **GitHub API** | `github:org/repo` | PR/issue context, no full clone |

### Extended Tool Schemas

All graph analysis tools accept either `repo_path` (local) or `repo_url` (remote):

```json
{
    "type": "object",
    "properties": {
        "repo_path": {
            "type": "string",
            "description": "Absolute path to local repository"
        },
        "repo_url": {
            "type": "string",
            "description": "Remote repository URL (HTTPS, SSH, or github:org/repo)"
        },
        "branch": {
            "type": "string",
            "default": "main",
            "description": "Branch to analyze (for remote repos)"
        },
        "shallow": {
            "type": "boolean",
            "default": true,
            "description": "Use shallow clone for faster analysis"
        }
    },
    "oneOf": [
        {"required": ["repo_path"]},
        {"required": ["repo_url"]}
    ]
}
```

### Authentication Configuration

**Environment Variables**:

| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` | GitHub API access, HTTPS clone auth |
| `SSH_KEY_PATH` | Path to SSH private key (default: `~/.ssh/id_rsa`) |
| `GIT_CREDENTIALS_PATH` | Path to git-credentials file |

**MCP Config Example**:
```json
{
    "mcpServers": {
        "nodestradamus": {
            "command": "nodestradamus",
            "args": ["serve"],
            "env": {
                "GITHUB_TOKEN": "${GITHUB_TOKEN}",
                "SSH_KEY_PATH": "/home/user/.ssh/github_key"
            }
        }
    }
}
```

### Clone Strategy

1. **Shallow clone** (default): `git clone --depth=1 --single-branch`
   - Fast, minimal disk usage
   - Sufficient for static analysis
   - Co-occurrence analysis limited to available history

2. **Full clone**: When `shallow: false`
   - Required for full co-occurrence analysis
   - Cached for subsequent requests

3. **GitHub API mode** (`github:org/repo`):
   - No clone required
   - Fetches file contents via API
   - Best for single-file analysis or PR context

### Cache Management

Remote repositories are cloned to a temporary cache:

```
~/.cache/nodestradamus/repos/
├── github.com/
│   └── org/
│       └── repo/
│           ├── .git/
│           └── ...
└── gitlab.com/
    └── ...
```

**Cache behavior**:
- Repos cached for 1 hour by default
- `git fetch` on cache hit to check for updates
- Manual clear: `nodestradamus cache clear`

### Security Considerations

1. **Token scope**: Use minimal GitHub token permissions (read-only)
2. **SSH keys**: Support read-only deploy keys
3. **URL validation**: Block internal IPs and localhost URLs
4. **Temp directory**: Clone to secure temp dir, clean up after analysis

---

## Multi-Format Rule Output

Nodestradamus can generate and convert rules between different AI assistant formats, enabling portability across tools.

### Supported Formats

| Format | File Type | Target Tool | Structure |
|--------|-----------|-------------|-----------|
| **Cursor** | `.mdc` | Cursor IDE | YAML frontmatter + Markdown |
| **Claude** | `.md` | Claude Projects | Markdown with sections |

### Tool 10: `convert_rules`

**Purpose**: Transform rules between different AI assistant formats.

```python
@tool
async def convert_rules(
    source_path: str,
    source_format: str,
    target_format: str,
    output_path: str | None = None
) -> dict:
    """
    Convert rules from one format to another.
    
    Args:
        source_path: Path to source rules (file or directory)
        source_format: Source format - "cursor" or "claude"
        target_format: Target format - "cursor" or "claude"
        output_path: Optional output path (returns content if not specified)
        
    Returns:
        Converted rules content or confirmation of file write
    """
```

**Input Schema**:
```json
{
    "type": "object",
    "properties": {
        "source_path": {
            "type": "string",
            "description": "Path to source rules"
        },
        "source_format": {
            "type": "string",
            "enum": ["cursor", "claude"]
        },
        "target_format": {
            "type": "string",
            "enum": ["cursor", "claude"]
        },
        "output_path": {
            "type": "string",
            "description": "Optional output path"
        }
    },
    "required": ["source_path", "source_format", "target_format"]
}
```

**Output Schema**:
```json
{
    "converted": true,
    "source_format": "cursor",
    "target_format": "claude",
    "files": [
        {
            "source": ".cursor/rules/100-python.mdc",
            "target": "claude-project-docs/python-conventions.md",
            "content": "# Python Conventions\n\n## Naming\n- Use snake_case..."
        }
    ],
    "output_path": "claude-project-docs/",
    "warnings": []
}
```

### Format Schemas

**Cursor Format (`.mdc`)**:
```yaml
---
description: Rule description
globs: "**/*.py"
alwaysApply: false
---

# Rule Title

## Section
- Directive 1
- Directive 2
```

**Claude Projects Format (`.md`)**:
```markdown
# Project: My App

## Conventions

### Python
- Use snake_case for functions
- Use PascalCase for classes

### Testing
- Use pytest for all tests
- Mock external services
```

### Conversion Logic

**Cursor → Claude**:
1. Strip YAML frontmatter
2. Merge multiple `.mdc` files into single doc
3. Convert glob patterns to prose descriptions
4. Organize by topic (naming, testing, etc.)

**Claude → Cursor**:
1. Split sections into separate `.mdc` files
2. Generate appropriate glob patterns
3. Add frontmatter with defaults

**Round-Trip Preservation**:
- Metadata stored in HTML comments for lossless round-trips
- `<!-- nodestradamus:cursor:globs="**/*.py" -->`

---

## CLI Interface

### Commands

```bash
# Start MCP server (stdio transport)
nodestradamus serve

# Start with specific transport
nodestradamus serve --transport stdio    # Default, for Cursor/Claude
nodestradamus serve --transport sse      # Server-Sent Events

# Analyze repository (standalone, no MCP)
nodestradamus analyze /path/to/repo

# Validate rules (standalone)
nodestradamus validate-rules /path/to/.cursor/rules

# Show version
nodestradamus --version
```

### Entry Point (pyproject.toml)

```toml
[project.scripts]
nodestradamus = "nodestradamus.cli:main"
```

---

## MCP Configuration Examples

### Cursor (`.cursor/mcp.json`)

```json
{
    "mcpServers": {
        "nodestradamus": {
            "command": "python",
            "args": ["-m", "nodestradamus", "serve"],
            "env": {}
        }
    }
}
```

### Alternative: Installed Package

```json
{
    "mcpServers": {
        "nodestradamus": {
            "command": "nodestradamus",
            "args": ["serve"]
        }
    }
}
```

### Claude Desktop

```json
{
    "mcpServers": {
        "nodestradamus": {
            "command": "nodestradamus",
            "args": ["serve"]
        }
    }
}
```

---

## Implementation Phases

### Phase 1: Core MCP Server (MVP)

**Deliverables**:
- [ ] Basic MCP server with stdio transport
- [ ] `analyze_python` tool (wrap existing script)
- [ ] `analyze_cooccurrence` tool (wrap existing script)
- [ ] `get_impact` tool (graph traversal)
- [ ] `nodestradamus serve` CLI command
- [ ] Installation via `pip install .`

**Success Criteria**:
- Server starts and responds to MCP initialization
- Cursor can list available tools
- `analyze_python` returns valid graph for test repo

### Phase 2: Full Graph Tools

**Deliverables**:
- [ ] `analyze_typescript` tool
- [ ] `merge_graphs` tool
- [ ] Caching layer for graphs

### Phase 3: Rules Tools

**Deliverables**:
- [ ] `validate_rules` tool
- [ ] `detect_conflicts` tool

### Phase 4: Advanced Features

**Deliverables**:
- [ ] Hotspot analysis (high churn + high complexity)
- [ ] PageRank for code importance
- [ ] Change prediction

### Phase 5: LLM Integration (Bring Your Own Key)

**Deliverables**:
- [ ] `generate_rules` tool with LLM-powered convention detection
- [ ] `suggest_rule` tool for user-provided patterns
- [ ] Multi-provider support (Anthropic, OpenAI, Ollama)
- [ ] Token budget compression for generated rules
- [ ] API key configuration via environment variables

**Success Criteria**:
- Generate meaningful rules from cold codebase analysis
- Rules follow <40 token directive guideline
- Graceful error when no API key provided

### Phase 6: Remote Repos & Multi-Format

**Deliverables**:
- [ ] Remote repository support (HTTPS, SSH, GitHub API)
- [ ] Shallow clone optimization for large repos
- [ ] Repository caching layer
- [ ] `convert_rules` tool for format transformation
- [ ] Cursor ↔ Claude format conversion
- [ ] Round-trip metadata preservation

**Success Criteria**:
- Analyze public GitHub repo without local clone
- Convert Cursor rules to Claude Projects format
- Private repo access with SSH key or token

---

## Testing Strategy

### Unit Tests

```bash
pytest tests/unit/
```

- Test each analyzer independently
- Test graph data models
- Test MCP tool schemas

### Integration Tests

```bash
pytest tests/integration/
```

- Test MCP server initialization
- Test tool invocation via MCP protocol
- Test with sample repositories

### Manual Testing

1. Start server: `nodestradamus serve`
2. Configure in Cursor
3. Ask: "What depends on auth.py?"
4. Verify tool is called and returns expected data

---

## Error Handling

### Tool Errors

```python
class NodestradamusError(Exception):
    """Base exception for Nodestradamus errors."""
    pass

class AnalysisError(NodestradamusError):
    """Error during code analysis."""
    pass

class RuleValidationError(NodestradamusError):
    """Error validating rules."""
    pass
```

### MCP Error Responses

```python
# Return structured error in MCP format
{
    "error": {
        "code": -32000,
        "message": "Analysis failed",
        "data": {
            "type": "AnalysisError",
            "detail": "Could not parse src/broken.py: SyntaxError line 42"
        }
    }
}
```

---

## Performance Requirements

| Operation | Target Latency | Max Latency |
|-----------|----------------|-------------|
| `analyze_python` (100 files) | <500ms | 2s |
| `analyze_python` (1000 files) | <5s | 15s |
| `get_impact` | <200ms | 1s |
| `validate_rules` | <100ms | 500ms |
| `detect_conflicts` | <200ms | 1s |

---

## Security Considerations

1. **Path Validation**: All `repo_path` inputs must be validated to prevent path traversal
2. **No Code Execution**: Never execute code from analyzed files
3. **No Network**: Analysis is local-only, no external calls
4. **Subprocess Safety**: Use `subprocess.run` with explicit args, no shell=True

---

## Dependencies

### Required (requirements.txt)

```
mcp>=1.0.0
tree-sitter>=0.21.0
tree-sitter-python>=0.21.0
tree-sitter-typescript>=0.21.0
tree-sitter-javascript>=0.21.0
click>=8.0.0
pydantic>=2.0.0
```

### Optional (for advanced features)

```
networkx>=3.0.0  # For graph algorithms
```

### Optional (for LLM-powered tools)

```
anthropic>=0.40.0       # Claude API for generate_rules, suggest_rule
openai>=1.50.0          # GPT-4 API alternative
httpx>=0.27.0           # For Ollama local model support
```

### Optional (for remote repository support)

```
gitpython>=3.1.0        # Programmatic git operations
pygithub>=2.0.0         # GitHub API access
```

---

## Existing Code to Reuse

| Module / Path | Maps To |
|---------------|---------|
| `nodestradamus/analyzers/deps.py` | `analyze_deps` (multi-language) |
| `nodestradamus/analyzers/git_cooccurrence.py` | `analyze_cooccurrence` |
| `nodestradamus/analyzers/impact.py` | `get_impact` |
| `nodestradamus/analyzers/project_scout.py` | `project_scout` |
| `nodestradamus/analyzers/graph_algorithms.py` | `analyze_graph` |
| `nodestradamus/analyzers/embeddings.py` | `semantic_analysis` |
| `nodestradamus/analyzers/docs.py` | `analyze_docs` |
| `nodestradamus/mcp/tools/handlers/cache.py` | `manage_cache` |
| `nodestradamus/mcp/tools/handlers/rules_audit.py` | `compare_rules_to_codebase` |
| `.cursor/rules/002-meta-generator.mdc` | `generate_rules` / `suggest_rule` pattern detection |
| *(planned)* | `convert_rules` tool - format transformation |

---

## Success Metrics

1. **Functional**: All 10 tools work correctly with Cursor
2. **Installable**: `pip install nodestradamus` works
3. **Documented**: README with quick start guide
4. **Tested**: >80% test coverage on core analyzers
5. **Fast**: Meets latency requirements above
6. **Portable**: Rules convert between Cursor and Claude formats
7. **Extensible**: Remote repos work with user-provided credentials

---

## References

- [MCP Specification](https://modelcontextprotocol.io/specification/2024-11-05)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Cursor MCP Documentation](https://cursor.com/docs/context/mcp)

---

*Last Updated: 2026-01-27*

---

## Changelog

- **2026-01-27**: Added LLM integration (Tools 8-9), remote repo support, multi-format output (Tool 10), Phases 5-6
