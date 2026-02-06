# Understanding Dependency Graphs

Dependency graphs help you and AI understand code relationships and perform **impact analysis** before making changes.

**For graph theorists:** Formal definitions, algorithm specifications (PageRank, betweenness, Louvain, cycles, path, SCC, layers), complexity, and references are in [Graph Theory Reference](graph-theory-reference.md).

## What is a Dependency Graph?

A dependency graph is a map of your codebase:

- **Nodes** are files, functions, classes, or modules
- **Edges** are relationships: "A calls B", "A imports B", "A inherits from B"

The goal is to answer questions like:
- What breaks if I change this function?
- What are the most critical files in this codebase?
- Are there circular dependencies?

---

## What Graphs Provide

When you or AI modify a function, the graph shows:

- **Downstream impact:** What calls this function? (will break if you change the signature)
- **Upstream dependencies:** What does this function call? (what it depends on)
- **Co-occurrence:** What files tend to change together in git history?

This prevents isolated changes that break dependent code.

---

## Building the Graph with Nodestradamus

Nodestradamus builds the dependency graph automatically. Use `quick_start` for the full setup, or `analyze_deps` for just the graph.

**Recommended: quick_start (does everything)**

```json
{
  "tool": "quick_start",
  "arguments": {
    "repo_path": "/path/to/repo"
  }
}
```

This runs: `project_scout` → `analyze_deps` → `codebase_health` → `semantic_analysis` (embeddings).

**Or: analyze_deps (graph only)**

```json
{
  "tool": "analyze_deps",
  "arguments": {
    "repo_path": "/path/to/repo"
  }
}
```

**Supported languages:** Python, TypeScript/JavaScript, Rust, SQL, Bash, JSON configs.

**Cache:** The graph is stored under `.nodestradamus/graph.json` (in the repo or workspace).

---

## Graph Types

| Type | What It Tracks | Nodestradamus Tool |
|------|----------------|--------------|
| **Dependency graph** | Function calls, imports, inheritance | `analyze_deps` |
| **Co-occurrence** | Files that change together in git | `analyze_cooccurrence` |

---

## How AI Uses Graphs

With Nodestradamus, AI uses the graph via MCP tools:

- **`get_impact`** — Shows upstream/downstream dependencies and affected tests for a file or symbol
- **`analyze_graph`** — Runs graph algorithms (PageRank, betweenness, cycles, etc.)
- **`codebase_health`** — Checks for dead code, cycles, bottlenecks

The `305-dependency-graph.mdc` Cursor rule tells AI to:

1. **Check for downstream impacts** before modifying functions
2. **List affected files** in responses
3. **Update related code** proactively

### Example AI Response

```
I'll update the `calculate_total()` function. Based on the dependency graph:

**Downstream impact (3 files):**
- `orders/checkout.py` - calls `calculate_total()`
- `reports/summary.py` - calls `calculate_total()`
- `tests/test_pricing.py` - tests `calculate_total()`

I'll update all affected files to match the new signature.
```

---

## Graph Algorithms

Use `analyze_graph` to run algorithms on the dependency graph:

| Algorithm | What it finds |
|-----------|---------------|
| `pagerank` | Most critical modules (most depended upon) |
| `betweenness` | Bottlenecks where changes ripple widely |
| `communities` | Clusters of related code |
| `cycles` | Circular dependencies |
| `path` | Dependency path between two nodes |
| `hierarchy` | Collapsed view at package/module/class level |
| `layers` | Validate layered architecture |

New to these terms? See [Glossary](glossary.md).

---

## CI Integration (Optional)

Keep graphs updated automatically. Example workflow using Nodestradamus CLI:

```yaml
# .github/workflows/update-graphs.yml
name: Update Dependency Graphs

on:
  push:
    branches: [main]
    paths: ['src/**', '**/*.py', '**/*.ts']

jobs:
  update-graphs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 100

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Nodestradamus
        run: pip install nodestradamus

      - name: Generate graphs
        run: nodestradamus analyze .

      - name: Commit graphs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .nodestradamus/
          git commit -m "chore: update dependency graphs" || true
          git push
```

---

## Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **Dynamic calls** | `getattr()`, reflection, runtime dispatch | Document in comments or rules |
| **Cross-repo** | Dependencies across repositories not tracked | Document in `200-project.mdc` |
| **Config files** | JSON/YAML dependencies are implicit | Use `json_patterns` in `analyze_deps` for configs |

When graphs are missing or incomplete, AI falls back to searching the codebase.

---

## Tools Reference

| Tool | Purpose |
|------|---------|
| `analyze_deps` | Build the dependency graph |
| `analyze_cooccurrence` | Find files that change together in git |
| `get_impact` | Show upstream/downstream for a file or symbol |
| `analyze_graph` | Run graph algorithms (pagerank, betweenness, etc.) |
| `codebase_health` | Health check including cycles and bottlenecks |
| `manage_cache` | Inspect or clear the `.nodestradamus/` cache |

See [Getting Started Workflow](getting-started-workflow.md) for the optimal tool sequence.
