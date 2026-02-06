# Graph Theory Reference

Formal definitions and algorithm specifications for Nodestradamus’s dependency and co-occurrence graphs. For user-facing usage, see [Understanding Dependency Graphs](dependency-graphs.md) and [Glossary](glossary.md).

---

## 1. Graph model

### 1.1 Dependency graph

- **Type:** Directed graph \( G = (V, E) \). Implemented as `networkx.DiGraph`.
- **Nodes \( V \):** Code symbols. Each node has:
  - **id:** Unique string (e.g. `py:src/auth.py::login`).
  - **type:** One of `function`, `class`, `method`, `module`, `file`, `table`, `view`, `cte`, `procedure`, `trigger`, `config`.
  - **file:** Relative file path.
  - **name:** Symbol name.
- **Edges \( E \):** Ordered pairs \((u, v)\) with **type** in `calls`, `inherits`, `imports`, `extends`.
  - **Direction:** Edge \((u, v)\) means “\(u\) depends on \(v\)” (e.g. \(u\) calls \(v\), or \(u\) imports \(v\)). So **in-neighbors** of \(v\) are dependents; **out-neighbors** are dependencies.
- **Weights:** Edges are **unweighted** in the main dependency graph (unit weight for path and centrality).
- **Multi-edges / self-loops:** Not used; at most one edge per \((u,v)\).

### 1.2 Co-occurrence graph

- **Type:** Undirected, **weighted**. Nodes are file paths; edges indicate files that change together in git history.
- **Edge weight:** Strength in \([0,1]\) (Jaccard similarity of commit sets). Optional count = number of commits where both files changed.
- **Use:** Complementary to the dependency graph; no formal algorithms in this doc (see `analyze_cooccurrence`).

---

## 2. Algorithms

### 2.1 PageRank

- **Input:** \( G = (V,E) \) (directed).
- **Output:** \( \pi\colon V \to [0,1] \), \( \sum_v \pi(v) = 1 \).
- **Model:** Random surfer with damping. At each step: with probability \( \alpha \) follow a uniform random out-edge (or stay if no out-edges); with probability \( 1-\alpha \) jump to a uniformly random node.
- **Parameters:**  
  - \( \alpha = 0.85 \) (damping factor).  
  - `max_iter = 100` (iteration limit).
- **Implementation:** Power iteration; fallback to `networkx.pagerank`; optional Rust backend.
- **Interpretation:** High \( \pi(v) \) = many nodes (directly or indirectly) depend on \( v \) → “important” or “critical” module.
- **Note:** On directed graphs, dangling nodes (no out-edges) are handled by the standard “random jump” interpretation (e.g. out-edges to all nodes with probability \( (1-\alpha)/|V| \)).

### 2.2 Betweenness centrality

- **Input:** \( G = (V,E) \) (directed), unweighted.
- **Output:** \( b\colon V \to \mathbb{R}_{\ge 0} \). Optional normalization to \([0,1]\) (default: normalized).
- **Definition (standard):**  
  \( \sigma_{st} \) = number of shortest \(s\)–\(t\) paths; \( \sigma_{st}(v) \) = number of those paths that pass through \(v\).  
  \[
  b(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}.
  \]  
  Sum over all ordered pairs \((s,t)\) with \(s \neq v \neq t\) and \( \sigma_{st} > 0 \).
- **Variant:** Unweighted shortest paths (hop count). Directed: paths follow edge direction.
- **Implementation:** Brandes-style algorithm; optional Rust backend. Normalization: scale by \(2/((|V|-1)(|V|-2))\) for directed (or equivalent so that max score is 1).
- **Complexity:** \( O(nm) \) for unweighted graphs (\(n=|V|\), \(m=|E|\)).
- **Interpretation:** High \(b(v)\) = \(v\) lies on many shortest dependency paths → “bottleneck”; changes at \(v\) can ripple widely.

### 2.3 Community detection (Louvain)

- **Input:** \( G \) converted to **undirected** (symmetric edges); multi-edges merged if present.
- **Output:** Partition of \(V\) into communities (list of sets of node IDs).
- **Algorithm:** Louvain method (maximize modularity).
- **Implementation:** `networkx.community.louvain_communities`.
- **Interpretation:** Clusters of nodes with dense internal links vs sparser links to the rest → candidate module boundaries.
- **Meta-graph:** `community_metagraph` builds a graph of communities with:
  - **Cohesion** (per community): internal_edges / total_edges (incident to the community).
  - **Afferent / efferent coupling:** incoming vs outgoing edges between communities.
  - **Instability:** efferent / (afferent + efferent) (high = many outgoing dependencies).

### 2.4 Cycles (elementary / simple cycles)

- **Input:** \( G = (V,E) \) (directed).
- **Output:** List of **elementary cycles** (cycles with no repeated node except start/end).
- **Implementation:** `networkx.simple_cycles(G)`.
- **Filtering:** By default only **cross-file** cycles are returned (at least two distinct files in the cycle); intra-file cycles can be included with `cross_file_only=False`.
- **Interpretation:** Cycles = circular dependencies; can cause import/initialization issues and tight coupling.

### 2.5 Shortest path

- **Input:** \( G = (V,E) \), source \(s\), target \(t\).
- **Output:** One shortest \(s\)–\(t\) path as a list of node IDs, or `None` if no path exists.
- **Length:** Unweighted (hop count). Directed: path follows edge direction.
- **Implementation:** `networkx.shortest_path(G, source, target)` (BFS for unweighted).
- **Interpretation:** Dependency chain from \(s\) to \(t\).

### 2.6 Strongly connected components (SCCs)

- **Input:** \( G = (V,E) \) (directed).
- **Output:** List of sets of nodes; each set is a maximal strongly connected component. Only components of size \( \ge 2 \) are returned (single-node SCCs omitted).
- **Definition:** SCC = maximal set \(S \subseteq V\) such that for every \(u,v \in S\) there is a directed path from \(u\) to \(v\) and from \(v\) to \(u\).
- **Implementation:** `networkx.strongly_connected_components(G)`; optional Rust backend.
- **Interpretation:** Tightly coupled groups; every node in the component can reach every other → high coupling. Not currently exposed as a top-level `analyze_graph` algorithm but used internally / for coupling analysis.

### 2.7 Hierarchy (aggregation)

- **Input:** \( G \) with node attributes `file`, `type`, `name`; level ∈ {package, module, class, function}.
- **Output:** Aggregated nodes and edges at the chosen level (e.g. by directory, file, or class).
- **Semantics:** Nodes are grouped by package (directory), module (file), or class; edges are summed. “function” = no aggregation.
- **Not a centrality or partition algorithm:** It is a **view** of the same graph at a coarser granularity.

### 2.8 Layers (architecture validation)

- **Input:** \( G = (V,E) \); ordered list of **layers** \( L_0, L_1, \ldots, L_{k-1} \), each a set of node IDs or path-prefix patterns. \( L_0 \) = top (e.g. API), \( L_{k-1} \) = bottom (e.g. domain/infrastructure).
- **Rule:** Upper layers may depend on lower layers; **violation** = edge \((u,v)\) with \(u\) in a lower layer than \(v\) (layer index of \(u\) > layer index of \(v\)).
- **Output:** List of violations (edge + layer indices), plus counts of valid vs classified edges.
- **Node assignment:** A node belongs to the first layer whose pattern matches (exact ID or file path prefix).

---

## 3. Limitations and design choices

| Topic | Choice / limitation |
|-------|---------------------|
| **Weights** | Dependency graph edges are unweighted; path and betweenness use hop count. |
| **Direction** | Dependency graph is directed; community detection converts to undirected. |
| **Disconnected** | Algorithms handle multiple components; PageRank and betweenness are well-defined per component. |
| **Dynamic / reflection** | Graph is static; runtime calls (e.g. `getattr`, reflection) are not reflected in \(E\). |
| **Cross-repo** | Only one repository per graph; cross-repo edges are not represented. |

---

## 4. References

- **PageRank:** Page, L., Brin, S., et al. (1998). The PageRank Citation Ranking. Stanford InfoLab.
- **Betweenness:** Brandes, U. (2001). A faster algorithm for betweenness centrality. *J. Mathematical Sociology*, 25(2), 163–177.
- **Louvain:** Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008.
- **Strongly connected components:** Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. *SIAM J. Computing*, 1(2), 146–160.
- **General:** Diestel, R. *Graph Theory* (Springer); or Cormen et al. *Introduction to Algorithms* (MIT Press), chapters on shortest paths, BFS, and SCC.

---

## 5. See also

- [Understanding Dependency Graphs](dependency-graphs.md) — usage and workflows  
- [Glossary](glossary.md) — term definitions  
- `nodestradamus/analyzers/graph_algorithms.py` — implementation  
- `nodestradamus/models/graph.py` — node/edge and graph data models  
