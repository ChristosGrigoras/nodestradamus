# Getting Started Workflow

Optimal tool sequence for analyzing a new codebase with Nodestradamus.

## Quick Reference

```
1. project_scout          → Overview + suggested_ignores
2. analyze_deps           → Build dependency graph (pass suggested_ignores)
3. codebase_health        → Combined health check
4. semantic_analysis      → mode="embeddings" to pre-compute index
5. semantic_analysis      → mode="search" (now fast)
6. find_similar            → Structurally similar code (optional, uses parse cache)
7. get_changes_since_last  → Diff vs last run (pass workspace_path to scout/deps/health to save snapshots)
8. analyze_graph          → Importance/bottleneck analysis
9. get_impact             → Deep-dive on specific files
```

**Or use `quick_start`** to run steps 1-4 automatically (scout → deps → codebase_health → embeddings). Options: `skip_embeddings: true` to skip the slow embedding step; `health_checks: ["dead_code", "duplicates", "cycles", "bottlenecks"]` to run more checks (default is cycles + bottlenecks).

```json
{
  "tool": "quick_start",
  "arguments": {
    "repo_path": "/path/to/repo"
  }
}
```

## Why Order Matters

Running tools in the wrong order leads to surprises:

| Anti-pattern | Impact |
|--------------|--------|
| `semantic_analysis search` without pre-computing embeddings | First search takes 30-230s (embedding computation is unavoidable, but can be done upfront) |
| Skipping `project_scout` | Miss `suggested_ignores`, analyze unnecessary files |
| Not passing `suggested_ignores` to other tools | Slower analysis, noisy results from test/vendor files |

**Tip:** Use the `quick_start` tool to run the optimal setup sequence automatically.

## Step-by-Step Guide

### Step 1: project_scout (Start Here)

**Why first:** Returns repository overview, `suggested_ignores`, and intelligent analysis hints for all subsequent tools.

```json
{
  "tool": "project_scout",
  "arguments": {
    "repo_path": "/path/to/repo"
  }
}
```

**Returns:**
- `languages` — Detected languages and file counts
- `key_directories` — Main source directories
- `frameworks` — Detected frameworks (pytest, pydantic, etc.)
- `suggested_ignores` — Patterns to exclude (node_modules, venv, etc.)
- `suggested_tools` / `suggested_queries` — Which Nodestradamus tools and example queries are most relevant
- `project_type` — Classified as `app`, `lib`, `monorepo`, or `unknown`
- `readme_hints` — Extracted hints from README (e.g., "core logic in src/engine/")
- `recommended_scope` — Suggested paths for focused analysis (e.g., `["src/", "lib/"]`)
- `lazy_options` — When to use lazy loading: **LazyGraph**, **LazyEmbeddingGraph**, and **lazy embedding** (scoped + on-demand FAISS). Each entry has `option`, `when`, `description`. For monorepos or 5K+ source files, `next_steps` includes a LazyEmbeddingGraph step.

**Save this output** — use `suggested_ignores` in subsequent calls, `recommended_scope` for targeted analysis, and `lazy_options` for large codebases or monorepos.

### Step 2: analyze_deps (Build the Graph)

**Why second:** Creates the dependency graph used by impact analysis, graph algorithms, and health checks.

```json
{
  "tool": "analyze_deps",
  "arguments": {
    "repo_path": "/path/to/repo",
    "exclude": ["node_modules", "venv", "__pycache__", "*.test.*"]
  }
}
```

**Tip:** Pass `suggested_ignores` from Step 1 as the `exclude` parameter.

**Monorepos:** To analyze only one package, pass `package` (e.g. `"libs/core"`). Use `project_scout` to see `packages`; if the repo is a monorepo, it returns available package paths.

**Scoped analysis:** Use `scope` to analyze only specific paths (e.g., `["src/", "lib/"]`). Pass `recommended_scope` from `project_scout` for intelligent filtering.

**Returns:**
- Node and edge counts
- Top modules by connections
- File counts per language
- Cached to `.nodestradamus/graph.msgpack`

**Programmatic API:** Use `analyze_deps_smart()` to automatically combine `project_scout` intelligence with graph building:

```python
from nodestradamus.analyzers import analyze_deps_smart

graph, metadata = analyze_deps_smart("/path/to/repo")
# Uses project_scout to determine scope, excludes, and entry points automatically
```

### Step 3: codebase_health (Quick Health Check)

**Why third:** Uses the cached graph from Step 2. Runs multiple checks in one call.

```json
{
  "tool": "codebase_health",
  "arguments": {
    "repo_path": "/path/to/repo",
    "checks": ["dead_code", "duplicates", "cycles", "bottlenecks", "docs"]
  }
}
```

**Returns:**
- `dead_code` — Unused functions/classes
- `duplicates` — Copy-pasted code blocks
- `cycles` — Circular dependencies
- `bottlenecks` — High-centrality nodes where changes ripple widely
- `docs` — Stale doc references and undocumented exports (same logic as `analyze_docs`)

Optional: `scope` (all / source_only / tests_only), `max_items`, `exclude_external`.

### Step 4: semantic_analysis embeddings (Pre-compute Index)

**Why fourth:** Computing embeddings upfront makes all subsequent searches instant.

```json
{
  "tool": "semantic_analysis",
  "arguments": {
    "repo_path": "/path/to/repo",
    "mode": "embeddings",
    "exclude": ["node_modules", "venv", "*.test.*"]
  }
}
```

**First run takes time** (30-120 seconds for large repos), but results are cached and **updated incrementally**:
- Content hashing detects which chunks changed
- Only changed chunks are re-embedded
- Unchanged chunks reuse cached embeddings
- Subsequent runs are much faster

**Storage:** Uses SQLite for chunk metadata + FAISS for vector search. Falls back to NPZ if FAISS isn't installed.

### Step 5: semantic_analysis search (Now Fast)

With embeddings pre-computed, search is instant.

```json
{
  "tool": "semantic_analysis",
  "arguments": {
    "repo_path": "/path/to/repo",
    "mode": "search",
    "query": "how to authenticate users"
  }
}
```

### Step 6: find_similar (Structurally Similar Code, Optional)

**Why optional:** Fast structural match without embeddings. Use when you want "code that looks like this" by structure (node/edge patterns).

```json
{
  "tool": "find_similar",
  "arguments": {
    "repo_path": "/path/to/repo",
    "file_path": "src/auth/token.py",
    "top_k": 10
  }
}
```

**Optional:** `line_start`, `line_end` to restrict to a line range. Run `analyze_deps` first so the parse cache (and fingerprint index) exist.

### Step 7: get_changes_since_last (What Changed, Optional)

**When:** You ran scout/deps/health earlier with `workspace_path` set; now you want a diff of what changed.

```json
{
  "tool": "get_changes_since_last",
  "arguments": {
    "repo_path": "/path/to/repo",
    "workspace_path": "/path/to/workspace",
    "tool": "all"
  }
}
```

**To save snapshots:** When calling `project_scout`, `analyze_deps`, or `codebase_health`, pass `workspace_path` so Nodestradamus writes a summary under `workspace/.nodestradamus/snapshots/<repo_hash>/`. Then `get_changes_since_last` can diff current state vs that summary.

### Step 8: analyze_graph (Deep Analysis)

Run graph algorithms for deeper insights.

```json
{
  "tool": "analyze_graph",
  "arguments": {
    "repo_path": "/path/to/repo",
    "algorithm": "pagerank"
  }
}
```

**Algorithms:**

| Algorithm | Use Case |
|-----------|----------|
| `pagerank` | Find most critical modules (most depended upon) |
| `betweenness` | Find bottlenecks where changes ripple widely |
| `communities` | Discover module clusters |
| `cycles` | Find circular dependencies |
| `path` | Trace dependency path between two nodes |
| `hierarchy` | Collapsed view at package/module/class level |
| `layers` | Validate layered architecture (pass layer prefixes) |

### Step 9: get_impact (Deep-dive)

Once you identify a critical file, analyze its blast radius.

```json
{
  "tool": "get_impact",
  "arguments": {
    "repo_path": "/path/to/repo",
    "file_path": "src/auth/token.py",
    "symbol": "validate_token",
    "depth": 3
  }
}
```

**Returns:**
- `upstream` — What this code depends on
- `downstream` — What depends on this code
- `test_files_affected` — Tests that cover this code

## Common Workflows

### Workflow A: "I'm new to this codebase"

```
1. project_scout         → Understand structure
2. analyze_deps          → See how code connects
3. analyze_graph pagerank → Find the most important files
4. Read those files      → Start with what matters
```

### Workflow B: "I need to refactor X"

```
1. project_scout         → Get suggested_ignores
2. analyze_deps          → Build graph
3. get_impact file_path=X → See blast radius
4. Decide scope          → Safe to refactor?
```

### Workflow C: "Find similar code to X"

**Option A — Structural (no embeddings):**
```
1. analyze_deps           → Ensures parse cache is warm
2. find_similar           → file_path=X, top_k=10
```

**Option B — Semantic (embedding-based):**
```
1. semantic_analysis embeddings → Build index (once)
2. semantic_analysis similar    → Find related code
3. semantic_analysis duplicates → Find copy-paste
```

### Workflow E: "What changed since I last ran Nodestradamus?"

```
1. When running scout/deps/health, pass workspace_path → Snapshots saved under workspace/.nodestradamus/snapshots/
2. get_changes_since_last(repo_path, workspace_path) → Diffs current state vs last run
```

### Workflow D: "Health check before release"

```
1. project_scout         → Overview
2. codebase_health       → All checks in one call
3. Fix critical issues   → Dead code, cycles, duplicates
```

## Cache Management

Nodestradamus caches results for faster subsequent runs:

| Cache | Location | Updated When |
|-------|----------|--------------|
| Dependency graph | `.nodestradamus/graph.msgpack` | Files change |
| Parse cache | `.nodestradamus/parse_cache.json` | Files change |
| Fingerprints | `.nodestradamus/fingerprints.msgpack` | `find_similar` / fingerprint build |
| Chunk metadata (SQLite) | `.nodestradamus/nodestradamus.db` | `mode="embeddings"` |
| Vector index (FAISS) | `.nodestradamus/embeddings.faiss` | `mode="embeddings"` |
| Embeddings (legacy) | `.nodestradamus/embeddings.npz` | Fallback format, still supported |
| Snapshots | `workspace/.nodestradamus/snapshots/<repo_hash>/` | `project_scout` / `analyze_deps` / `codebase_health` when `workspace_path` is set |

**Incremental updates:** Embeddings use content hashing — only changed chunks are re-embedded. Subsequent runs after the initial embedding are much faster.

**Check cache status:**
```json
{
  "tool": "manage_cache",
  "arguments": {
    "mode": "info",
    "repo_path": "/path/to/repo"
  }
}
```

**Clear stale cache:**
```json
{
  "tool": "manage_cache",
  "arguments": {
    "mode": "clear",
    "repo_path": "/path/to/repo"
  }
}
```

## Passing suggested_ignores

The `suggested_ignores` from `project_scout` should be passed to other tools:

```python
# Step 1: Get ignores
scout_result = project_scout(repo_path="/path/to/repo")
ignores = scout_result["suggested_ignores"]

# Step 2-N: Pass to other tools
analyze_deps(repo_path="/path/to/repo", exclude=ignores)
semantic_analysis(repo_path="/path/to/repo", mode="embeddings", exclude=ignores)
```

**Common ignores detected by project_scout:**
- `node_modules/` — JavaScript dependencies
- `venv/`, `.venv/` — Python virtual environments
- `__pycache__/` — Python bytecode
- `dist/`, `build/` — Build outputs
- `.git/` — Git metadata
- `*.min.js`, `*.bundle.js` — Minified bundles

**Optional:** Add a `.nodestradamusignore` file in the repo (gitignore-style patterns) to exclude paths from analysis; `project_scout` returns `nodestradamusignore_exists` and ignores are merged with defaults.

## Performance Tips

1. **Run project_scout first** — It's fast and provides critical context (now extracts README hints and recommends scope)
2. **Pre-compute embeddings** — First run takes time, but incremental updates only re-embed changed chunks
3. **Use exclude patterns** — Skip vendor/test files for cleaner results
4. **Use codebase_health** — Runs multiple checks efficiently in one call
5. **Cache results** — Don't clear cache unless files have changed
6. **Install FAISS** — For large codebases (>10K chunks), `pip install faiss-cpu` enables faster similarity search

## Tool Summary

| Tool | Purpose | Depends On |
|------|---------|------------|
| `project_scout` | Quick overview | None (run first) |
| `analyze_deps` | Build dependency graph | suggested_ignores |
| `codebase_health` | Health + docs check | Cached graph |
| `semantic_analysis` | Embeddings + search | suggested_ignores |
| `find_similar` | Structurally similar code | Parse cache (run analyze_deps first) |
| `get_changes_since_last` | Diff vs last run | Snapshots (pass workspace_path to scout/deps/health) |
| `analyze_graph` | Graph algorithms | Cached graph |
| `get_impact` | Blast radius analysis | Cached graph |
| `analyze_cooccurrence` | Git history analysis | None |
| `analyze_strings` | String literal analysis | None |
| `manage_cache` | Cache management | None |
| `analyze_docs` | Stale refs and coverage | None |
| `compare_rules_to_codebase` | Rules vs hotspots | Cached graph |

## Python API (Advanced)

For programmatic usage, Nodestradamus exposes a Python API:

```python
from nodestradamus.analyzers import (
    project_scout,
    analyze_deps,
    analyze_deps_smart,
    compute_embeddings,
    find_similar_code,
    LazyEmbeddingGraph,
)

# Smart analysis: combines project_scout + analyze_deps
graph, metadata = analyze_deps_smart("/path/to/repo")
print(f"Project type: {metadata['project_type']}")
print(f"Recommended scope: {metadata['recommended_scope']}")

# Lazy loading for large codebases
lazy = LazyEmbeddingGraph("/path/to/repo")
lazy.load_scope("src/auth/")  # Load only auth module
results = lazy.find_similar("authentication logic", top_k=5)

# Incremental embeddings
result = compute_embeddings("/path/to/repo")
print(f"Chunks: {result['total_chunks']}")
print(f"New: {result.get('incremental', {}).get('new', 0)}")
print(f"Reused: {result.get('incremental', {}).get('reused', 0)}")
```

See `nodestradamus/analyzers/__init__.py` for the full list of exported functions.

For deeper documentation on `LazyEmbeddingGraph`, SQLite/FAISS storage, and rule generation workflows, see [Advanced Features](advanced-features.md).
