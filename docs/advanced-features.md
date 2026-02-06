# Advanced Features

Deep documentation for Nodestradamus's advanced capabilities: lazy loading, storage architecture, and rule generation workflows.

---

## Table of Contents

1. [LazyEmbeddingGraph](#lazyembeddinggraph) — On-demand loading for large codebases
2. [Storage Architecture](#storage-architecture) — SQLite + FAISS hybrid system
3. [Incremental Embeddings](#incremental-embeddings) — Content-hash based updates
4. [Rule Generation Workflow](#rule-generation-workflow) — Using Nodestradamus to create AI rules

---

## LazyEmbeddingGraph

For large codebases (10K+ functions), building the full graph and computing all embeddings upfront is expensive. `LazyEmbeddingGraph` enables on-demand loading of only the relevant portion.

### When to Use

- **Monorepos** — Load only the package you're working on
- **Large codebases** — Avoid loading 100K nodes when you need 1K
- **Interactive exploration** — Expand scope as you discover related code

**Tip:** Run `project_scout` first; it returns `lazy_options` (LazyGraph, LazyEmbeddingGraph, lazy embedding) with `when` and `description` so you know which lazy approach to use for this repo. For monorepos or 5K+ source files, `next_steps` includes a LazyEmbeddingGraph recommendation.

### Basic Usage

```python
from nodestradamus.analyzers import LazyEmbeddingGraph

# Initialize (no data loaded yet)
lazy = LazyEmbeddingGraph("/path/to/large-repo")

# Load subgraph for a specific directory
stats = lazy.load_scope("src/auth/")
print(f"Loaded {stats['loaded']} nodes, {stats['edges']} edges")

# Compute embeddings only for loaded scope
embed_stats = lazy.compute_scoped_embeddings()

# Search within loaded scope
results = lazy.find_similar("authentication logic", top_k=5)

# Expand to include related modules
lazy.expand_from_nodes(["src/utils/crypto.py"])
```

### API Reference

#### `__init__(repo_path, workspace_path=None)`

Initialize the lazy loader. No data is loaded until you call `load_scope()` or `load_from_nodes()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo_path` | `str` | Path to the repository |
| `workspace_path` | `str \| None` | Workspace path for isolated caching |

#### `load_scope(scope: str) -> dict`

Load nodes matching a path prefix.

```python
stats = lazy.load_scope("src/auth/")
# Returns: {"loaded": 150, "edges": 320, "scope": "src/auth/"}
```

**Behavior:**
- Finds all nodes where the path contains or starts with `scope`
- Includes immediate neighbors (depth=1) for context
- Merges with any previously loaded nodes

#### `load_from_nodes(seed_nodes, max_depth=2) -> dict`

Load subgraph around specific seed nodes.

```python
stats = lazy.load_from_nodes(
    seed_nodes=["py:src/auth/token.py::validate_token"],
    max_depth=2
)
```

#### `expand_from_nodes(seed_nodes, max_depth=1) -> dict`

Expand the loaded subgraph with additional nodes. Same as `load_from_nodes()` but semantically indicates expansion.

#### `compute_scoped_embeddings(force=False) -> dict`

Compute embeddings only for the loaded scope.

```python
stats = lazy.compute_scoped_embeddings()
# Returns: {"status": "computed", "chunks": 450, "scope": "src/auth/"}
```

**Note:** Call `load_scope()` first, otherwise returns `{"status": "no_scope", "error": "Call load_scope first"}`.

#### `find_similar(query, top_k=10) -> list`

Search for similar code within the loaded scope.

```python
results = lazy.find_similar("login authentication", top_k=5)
for match in results:
    print(f"{match['file']}:{match['line']} - {match['score']:.3f}")
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `repo_path` | `str` | The repository path |
| `workspace_path` | `str \| None` | The workspace path |
| `loaded_nodes` | `set[str]` | Currently loaded node IDs (copy) |
| `is_loaded` | `bool` | Whether any data has been loaded |

---

## Storage Architecture

Nodestradamus uses a hybrid storage system: **SQLite** for metadata, **FAISS** for vector search.

### Why Hybrid?

| Concern | SQLite | FAISS |
|---------|--------|-------|
| **Chunk metadata** | ✅ Excellent (relational queries) | ❌ Not designed for this |
| **Vector similarity** | ❌ Slow at scale | ✅ Optimized (ANN search) |
| **Persistence** | ✅ Single file, ACID | ⚠️ Must rebuild from embeddings |
| **Incremental updates** | ✅ Easy (content hash lookup) | ⚠️ Append-only, rebuild for deletes |

### File Layout

```
.nodestradamus/
├── nodestradamus.db          # SQLite database (chunks, metadata)
├── embeddings.faiss    # FAISS vector index
├── graph.msgpack       # Dependency graph (msgpack format)
├── parse_cache.msgpack # Parsed AST cache
└── fingerprints.msgpack # Structural fingerprints
```

### SQLite Schema

```sql
-- Chunk metadata for embeddings
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,        -- Relative path: "src/auth/token.py"
    symbol_name TEXT,               -- Symbol: "validate_token"
    line_start INTEGER,
    line_end INTEGER,
    content_hash TEXT NOT NULL,     -- SHA256 of content (for change detection)
    snippet TEXT,                   -- First 200 chars of code
    language TEXT,                  -- "python", "typescript", etc.
    faiss_id INTEGER,               -- Index in FAISS vector store
    model_version TEXT NOT NULL,    -- Embedding model: "all-MiniLM-L6-v2"
    embedding BLOB,                 -- Raw embedding bytes (backup)
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_hash ON chunks(content_hash);
CREATE INDEX idx_chunks_symbol ON chunks(symbol_name);
CREATE INDEX idx_chunks_faiss_id ON chunks(faiss_id);

-- FAISS cache tracking
CREATE TABLE faiss_cache (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    is_stale INTEGER DEFAULT 1,     -- 1 = needs rebuild
    last_rebuilt TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

-- Schema version for migrations
CREATE TABLE schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

### FAISS Integration

FAISS provides approximate nearest neighbor (ANN) search:

```python
# Nodestradamus automatically:
# 1. Stores embeddings in SQLite (authoritative copy)
# 2. Builds FAISS index from SQLite embeddings
# 3. Marks FAISS stale when chunks change
# 4. Rebuilds FAISS on-demand before search
```

**Fallback:** If FAISS isn't installed (`pip install faiss-cpu`), Nodestradamus falls back to NumPy-based cosine similarity (slower but works).

### Database Operations

```python
from nodestradamus.utils.db import (
    get_connection,
    transaction,
    insert_chunk,
    get_chunks_by_file,
    get_chunks_by_scope,
    find_chunks_by_hash,
)

# Get a connection (cached, thread-safe)
conn = get_connection(workspace_path=None, repo_path="/path/to/repo")

# Use transaction context manager
with transaction(workspace_path=None, repo_path="/path/to/repo") as conn:
    insert_chunk(
        conn,
        file_path="src/auth/token.py",
        symbol_name="validate_token",
        line_start=10,
        line_end=25,
        content_hash="abc123...",
        snippet="def validate_token(token: str)...",
        language="python",
        faiss_id=42,
        model_version="all-MiniLM-L6-v2",
        embedding=embedding_bytes,  # numpy array.tobytes()
    )
    # Auto-commits on success, rollback on exception

# Query chunks
chunks = get_chunks_by_scope(conn, scope="src/auth/")
```

---

## Incremental Embeddings

Computing embeddings is expensive (30-120s for large repos). Nodestradamus uses content hashing to skip unchanged code.

### How It Works

1. **First run:** Hash each chunk's content, compute embedding, store both
2. **Subsequent runs:**
   - Parse files, extract chunks
   - For each chunk, compute content hash
   - If hash exists in SQLite with same model version → reuse embedding
   - If new or changed → compute new embedding
3. **Result:** Only changed code is re-embedded

### Content Hashing

```python
# Simplified version of what Nodestradamus does:
import hashlib

def content_hash(code: str, symbol_name: str, line_start: int) -> str:
    """Generate stable hash for a code chunk."""
    normalized = code.strip()
    key = f"{symbol_name}:{line_start}:{normalized}"
    return hashlib.sha256(key.encode()).hexdigest()
```

### Incremental Stats

```python
from nodestradamus.analyzers import compute_embeddings

result = compute_embeddings("/path/to/repo")
print(f"Total chunks: {result['total_chunks']}")
print(f"New chunks: {result.get('incremental', {}).get('new', 0)}")
print(f"Reused from cache: {result.get('incremental', {}).get('reused', 0)}")
print(f"Time saved: ~{result.get('incremental', {}).get('reused', 0) * 0.1:.1f}s")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NODESTRADAMUS_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Local embedding model (sentence-transformers) |
| `NODESTRADAMUS_FAISS_THRESHOLD` | `10000` | Use FAISS above this chunk count |
| `NODESTRADAMUS_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding computation |

---

## Rule Generation Workflow

Nodestradamus doesn't have a "generate rules" button — instead, it provides the intelligence to write effective rules. This workflow shows how to use Nodestradamus tools to create `.cursor/rules/`.

### The 9-Step Workflow

Based on the [LangChain case study](nodestradamus-rule-generation-case-study.md):

```
Step 1: project_scout        → Understand structure, tech stack
Step 2: analyze_deps         → Map dependencies
Step 3: analyze_graph pagerank → Find critical files
Step 4: analyze_graph betweenness → Find bottlenecks
Step 5: analyze_graph communities → Find module clusters
Step 6: analyze_graph cycles → Check for circular deps
Step 7: semantic_analysis search → Find patterns and base classes
Step 8: semantic_analysis duplicates → Find shared code
Step 9: compare_rules_to_codebase → Audit existing rules vs hotspots
```

### What Each Tool Contributes

| Tool | Rule Contribution |
|------|-------------------|
| `project_scout` | Project overview section, tech stack, key directories |
| `analyze_graph pagerank` | "Critical Files" table — what code matters most |
| `analyze_graph betweenness` | "Bottlenecks" — files that require extra review |
| `analyze_graph communities` | Module boundaries for scope-specific rules |
| `semantic_analysis search` | Pattern discovery (e.g., "find all retry logic") |
| `semantic_analysis duplicates` | Consolidation candidates |
| `compare_rules_to_codebase` | Gap analysis — which hotspots lack rules |

### Example: Creating a Project Rule

```python
from nodestradamus.analyzers import (
    project_scout,
    analyze_deps,
    pagerank,
    betweenness,
)

# Step 1: Understand the project
scout = project_scout("/path/to/repo")
print(f"Project type: {scout.project_type}")
print(f"Languages: {scout.languages}")
print(f"Frameworks: {scout.frameworks}")

# Step 2: Build dependency graph
G = analyze_deps("/path/to/repo")

# Step 3: Find critical files
pr = pagerank(G)
top_10 = sorted(pr.items(), key=lambda x: -x[1])[:10]
print("Critical files (by PageRank):")
for node, score in top_10:
    print(f"  {score:.4f} {node}")

# Step 4: Find bottlenecks
bc = betweenness(G)
bottlenecks = sorted(bc.items(), key=lambda x: -x[1])[:5]
print("\nBottlenecks (by betweenness):")
for node, score in bottlenecks:
    print(f"  {score:.4f} {node}")
```

### Rule Template

Based on analysis, create `.cursor/rules/200-project.mdc`:

```markdown
# Project Name

## Overview

{From project_scout: description, languages, frameworks}

## Key Directories

{From project_scout: key_directories}

## Critical Files

Changes to these files require extra review:

| File | Importance | Role |
|------|------------|------|
{From pagerank: top 10 files}

## Bottlenecks

High-impact nodes where changes ripple widely:

| File | Centrality | Notes |
|------|------------|-------|
{From betweenness: top 5}

## Conventions

{From semantic_analysis: discovered patterns}

## See Also

{Links to other relevant rules}
```

### MCP Tools for Rules

| Tool | Purpose |
|------|---------|
| `validate_rules` | Check rule file structure and frontmatter |
| `detect_rule_conflicts` | Find contradictions between rules |
| `compare_rules_to_codebase` | Audit rules vs actual code hotspots |

```json
{
  "tool": "validate_rules",
  "arguments": {
    "rules_path": ".cursor/rules/"
  }
}
```

---

## See Also

- [Getting Started Workflow](getting-started-workflow.md) — Optimal tool sequence
- [Glossary](glossary.md) — Term definitions
- [Graph Theory Reference](graph-theory-reference.md) — Algorithm details
- [Rule Generation Case Study](nodestradamus-rule-generation-case-study.md) — Real-world example
