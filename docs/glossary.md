# Glossary

Quick definitions for terms used in Nodestradamus documentation.

---

## Graph Concepts

**Dependency graph**
A **directed** graph of your codebase: nodes are files, functions, or classes; edges are relationships like "calls" or "imports." Edge \(u,v\) means "\(u\) depends on \(v\)." Edges are unweighted. For formal definitions and algorithm specs, see [Graph Theory Reference](graph-theory-reference.md).

**Node**
A single element in the graph: a file, function, class, or module.

**Edge**
A relationship between two nodes, such as "A calls B" or "A imports B."

**Upstream / Downstream**
- **Upstream:** What a piece of code depends on (what it calls or imports).
- **Downstream:** What depends on a piece of code (what calls or imports it).

**Blast radius**
The set of files or functions affected by a change. A large blast radius means many things could break.

**Impact analysis**
Determining what would be affected by changing a file or function. Nodestradamus's `get_impact` tool does this.

---

## Graph Algorithms

**PageRank**
An algorithm that ranks nodes by importance. High PageRank means many other nodes depend on this one (directly or indirectly). Originally used by Google to rank web pages.

**Betweenness (centrality)**
A measure of how often a node lies on the shortest path between other nodes. High betweenness means the node is a bottleneck — changes to it ripple widely.

**Bottleneck**
A node with high betweenness centrality. Changes here affect many parts of the codebase.

**Community (detection)**
Clusters of nodes that are more connected internally than to the rest of the graph. Nodestradamus uses the **Louvain** method (modularity maximization) on the graph converted to undirected. Useful for discovering module boundaries. See [Graph Theory Reference](graph-theory-reference.md).

**Cycle (circular dependency)**
When A depends on B, B depends on C, and C depends on A (or any directed loop). Nodestradamus reports **elementary cycles** (no repeated node except start/end). By default only cross-file cycles are returned. Cycles can cause import errors and tight coupling.

**Hierarchy**
A collapsed view of the graph at a higher level (package, module, or class) instead of individual functions.

**Layers (architecture)**
Validation that dependencies respect layer ordering: layers[0] = top (e.g. API), layers[-1] = bottom (e.g. domain). A **violation** is an edge from a lower layer to a higher layer. See [Graph Theory Reference](graph-theory-reference.md).

**Strongly connected component (SCC)**
A maximal set of nodes where every node can reach every other by a directed path. Used for coupling analysis; components of size ≥ 2 indicate tightly coupled modules.

---

## Semantic Search

**Semantic search**
Finding code by meaning rather than exact text. You describe what you want in natural language, and Nodestradamus finds related code.

**Embeddings**
Numeric representations of code that capture meaning. Similar code has similar embeddings. Nodestradamus uses embeddings for semantic search, finding similar code, and detecting duplicates.

**Duplicates**
Code blocks that are semantically similar (copy-pasted or nearly identical logic). Nodestradamus's `semantic_analysis` with `mode="duplicates"` finds these.

---

## Co-occurrence

**Co-occurrence**
Files that frequently change together in git history. High co-occurrence suggests the files are related, even if there's no direct import.

---

## MCP

**MCP (Model Context Protocol)**
A protocol that lets AI assistants (like Cursor or Claude) call external tools. Nodestradamus is an MCP server — AI calls Nodestradamus tools to analyze code.

---

## Documentation

**Stale reference**
A reference in documentation (e.g., a file path or function name) that no longer exists in the codebase.

**Coverage (docs)**
The percentage of important code (hotspots, exports) that is mentioned in documentation.

---

## Cache & Storage

**.nodestradamus/**
The directory where Nodestradamus stores cached analysis results:
- `graph.msgpack` — Dependency graph
- `parse_cache.msgpack` — Parsed AST cache
- `nodestradamus.db` — SQLite database (chunk metadata, embeddings)
- `embeddings.faiss` — FAISS vector index
- `fingerprints.msgpack` — Structural fingerprints

**.nodestradamusignore**
A file (gitignore-style patterns) that tells Nodestradamus which paths to exclude from analysis.

**SQLite + FAISS**
Nodestradamus's hybrid storage: SQLite stores chunk metadata (file, line, hash, embedding bytes); FAISS provides fast vector similarity search. See [Advanced Features](advanced-features.md#storage-architecture).

**Content hashing**
Each code chunk is hashed. On subsequent runs, unchanged chunks (same hash) reuse cached embeddings — only modified code is re-embedded.

---

## Lazy Loading

**LazyEmbeddingGraph**
A wrapper that loads graph and embeddings on-demand. For large codebases, load only the relevant subset instead of everything. See [Advanced Features](advanced-features.md#lazyembeddinggraph).

**Scoped embeddings**
Computing embeddings for a specific directory (scope) rather than the entire codebase. Useful for monorepos.

---

## See Also

- [Advanced Features](advanced-features.md) — LazyEmbeddingGraph, SQLite/FAISS storage, rule generation
- [Graph Theory Reference](graph-theory-reference.md) — formal model, algorithms, complexity, references
- [Understanding Dependency Graphs](dependency-graphs.md)
- [Getting Started Workflow](getting-started-workflow.md)
- [README](../README.md)
