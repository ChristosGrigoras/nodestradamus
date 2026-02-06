# <img src="assets/nodestradamus-logo.png" alt="Nodestradamus" width="42" style="vertical-align: middle;" /> Nodestradamus

**Nodestradamus: See what breaks before you break it.** Codebase intelligence for AI and human coders—an MCP server and a Python library that give Cursor, Claude, and other AI tools (or your own scripts) deep understanding of your code through dependency graphs, semantic search, and impact analysis.

Nodestradamus predicts **what breaks** if you change something—impact before you refactor. It builds a map of your code (who calls what) so you or your AI can see that impact and find important or risky areas.

> I had determined to go as far as declaring in abstruse and puzzling utterances the future causes… Yet lest whatever human changes may be to come should scandalise delicate ears, the whole thing is written in nebulous form, rather than as a clear prophecy of any kind.  
> — *Nostradamus, 1555*

### What Nodestradamus does

- **Maps who-calls-what** — builds a dependency graph of your codebase
- **Answers "what breaks if I change this?"** — impact analysis before refactors
- **Finds code by meaning** — semantic search and duplicate detection
- **Checks docs and rules** — finds stale references and coverage gaps

New to dependency graphs or these terms? See [Understanding dependency graphs](docs/dependency-graphs.md) and [Glossary](docs/glossary.md).

### Why use Nodestradamus with cheaper models?

Nodestradamus pre-computes codebase structure (who calls what, impact, semantic index) and exposes it via MCP tools. That shifts the work from the model to the tools:

| | **Cheap model + Nodestradamus** | **Expensive model, no Nodestradamus** |
|---|--------------------------------|--------------------------------------|
| **Context** | Structured answers from tools (graph, impact, search) — small, precise inputs | Raw file dumps and long context — more tokens, more noise |
| **Cost** | Fewer tokens per task; small/cheap models can drive the same workflows | Large context and repeated reads; often need bigger, pricier models |
| **Accuracy** | Impact and dependencies come from the graph, not guesswork | Model infers structure from text; easy to miss callers or side effects |
| **Speed** | One tool call → targeted result (e.g. “what breaks if I change this?”) | Many file reads and long chains of reasoning |

Use Nodestradamus so your assistant gets **precise, graph-backed answers** instead of guessing from raw code. That makes cheaper models effective for refactors, impact analysis, and codebase navigation.

### Evaluation

Evaluations on three codebases (LangChain Python monorepo, Rich, Django) compared Nodestradamus-on vs Nodestradamus-off across 18 codebase-understanding questions (overview, impact, cycles, dead code, duplicates, health, etc.):

| Codebase | Finding |
|----------|--------|
| **LangChain** | Cheaper model (Composer) + Nodestradamus was ~40% faster and more accurate than Opus without tools (e.g. correct "0 cycles" vs inferred "many potential cycles"). Same model with Nodestradamus: 27% more concise, quantified metrics vs prose. |
| **Rich** | Haiku + Nodestradamus **42% faster** with comparable verbosity; Opus + Nodestradamus ~14% slower but more metric-rich (betweenness, cohesion, line-level analysis). Both produced more actionable answers with tools. |
| **Django** | Both models with Nodestradamus gave quantified insights (graph metrics, cycle detection, duplicate file:line refs) vs estimates; trade-off was longer time for substantiated, data-driven answers. |

Across reports: **cheaper LLM + Nodestradamus** can match or beat **expensive LLM without tools** on accuracy and actionability for structural analysis; tools provide ground truth (cycles, dead code, centrality) that models often get wrong when inferring.

**Note:** The evaluation setup and benchmarks need further tests and validation (more codebases, question sets, and baselines) before drawing stronger conclusions.

## Install

```bash
pip install nodestradamus

# Or from source
git clone https://github.com/ChristosGrigoras/nodestradamus.git
cd nodestradamus && pip install -e .

# With FAISS for faster similarity search on large codebases (optional)
pip install nodestradamus[faiss]

# With Mistral for API-based embeddings (optional; set MISTRAL_API_KEY and NODESTRADAMUS_EMBEDDING_PROVIDER=mistral)
pip install nodestradamus[mistral]

# With Rust acceleration (optional, requires Rust toolchain)
pip install maturin
maturin develop --release
```

## Quick Start

```bash
# Start the MCP server
nodestradamus serve

# Analyze a repo
nodestradamus analyze /path/to/repo
```

**Optimal tool sequence** for best results:

```
1. project_scout     → Get overview + suggested_ignores
2. analyze_deps      → Build graph (pass suggested_ignores)
3. codebase_health   → Health check
4. semantic_analysis → mode="embeddings" (pre-compute)
5. semantic_analysis → mode="search" (now fast)
```

See [docs/getting-started-workflow.md](docs/getting-started-workflow.md) for the complete guide.

Add to Cursor (`.cursor/mcp.json`):

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

## Supported Languages

| Language | Dependency Analysis | Semantic Search | String Analysis |
|----------|---------------------|-----------------|-----------------|
| **Python** | ✅ Full | ✅ Full | ✅ Full |
| **TypeScript/JavaScript** | ✅ Full | ✅ Full | ✅ Full |
| **Rust** | ✅ Full | ✅ Full | ✅ Full |
| **SQL (PostgreSQL)** | ✅ Full | ✅ Full | ✅ Full |
| **Bash** | ✅ Full | ✅ Full | ✅ Full |
| **JSON** | ✅ Configs | — | — |

## Tools

| Tool | What it does |
|------|--------------|
| `quick_start` | Runs optimal setup sequence automatically |
| `project_scout` | Reconnaissance: languages, frameworks, key dirs; lazy options for monorepos |
| `analyze_deps` | Build dependency graph (Python, TS, Rust, SQL, Bash, JSON) |
| `analyze_cooccurrence` | Files that change together in git history |
| `get_impact` | What breaks if I change this file/function? |
| `analyze_graph` | Graph algorithms on dependencies |
| `analyze_strings` | Find and trace string literals |
| `semantic_analysis` | Embedding-based search and duplicate detection |
| `find_similar` | Structurally similar code (fingerprint match) |
| `get_changes_since_last` | Diff vs last run (snapshots) |
| `codebase_health` | Health check: dead code, duplicates, cycles, docs |
| `manage_cache` | Inspect or clear `.nodestradamus/` cache |
| `analyze_docs` | Docs: stale refs and coverage |
| `compare_rules_to_codebase` | Audit rules vs hotspots; gaps and stale refs |
| `validate_rules` | Validate rule file structure and frontmatter |
| `detect_rule_conflicts` | Detect conflicts between AI rules |

### Examples

**Check impact before refactoring:**
```
get_impact(repo_path="/my/repo", file_path="src/auth.py", symbol="validate_token")
→ Shows files that call this function
```

**Semantic search:**
```
semantic_analysis(repo_path="/my/repo", mode="search", query="authentication")
→ Natural language code search over embedded chunks
```

More examples in [docs/getting-started-workflow.md](docs/getting-started-workflow.md) and [docs/creative-use-cases.md](docs/creative-use-cases.md).

### Graph Algorithms

`analyze_graph` supports pagerank, betweenness, communities, cycles, path, hierarchy, layers. Optional Rust backend for speed. See [docs/dependency-graphs.md](docs/dependency-graphs.md) and [docs/graph-theory-reference.md](docs/graph-theory-reference.md). Rust extension: [docs/installation.md](docs/installation.md).

### Rust Support

Rust analysis extracts functions, structs, enums, traits, impls, use statements. See [docs/mcp-server-spec.md](docs/mcp-server-spec.md) for details.

### Semantic Analysis

Modes: search, similar, duplicates, embeddings. See [docs/getting-started-workflow.md](docs/getting-started-workflow.md).

### Cache

Results are cached under **`.nodestradamus/`** (repo when standalone, workspace when via MCP). Use `manage_cache` (mode="info" / "clear"). Optional `.nodestradamusignore` (gitignore-style) excludes paths; `project_scout` reports if it exists. File list and incremental embedding behavior: [docs/getting-started-workflow.md](docs/getting-started-workflow.md).

### Environment

Copy `.env.example` to `.env` for embedding provider and API keys. **Embeddings:** default is local (sentence-transformers, model `jinaai/jina-embeddings-v2-base-code`); for API-based embeddings use Mistral Codestral Embed: `pip install nodestradamus[mistral]`, set `NODESTRADAMUS_EMBEDDING_PROVIDER=mistral` and `MISTRAL_API_KEY`. See [docs/installation.md](docs/installation.md).

## Cursor Rules

Nodestradamus ships `.cursor/rules/` for code quality, security, and meta-generator. See [docs/cursor-rules.md](docs/cursor-rules.md).

## Documentation

| Topic | Link |
|-------|------|
| **Getting Started** | [docs/getting-started-workflow.md](docs/getting-started-workflow.md) |
| Installation | [docs/installation.md](docs/installation.md) |
| Understanding Dependency Graphs | [docs/dependency-graphs.md](docs/dependency-graphs.md) |
| Glossary | [docs/glossary.md](docs/glossary.md) |
| MCP Server Spec | [docs/mcp-server-spec.md](docs/mcp-server-spec.md) |
| Creative Use Cases | [docs/creative-use-cases.md](docs/creative-use-cases.md) |
| Cursor Rules | [docs/cursor-rules.md](docs/cursor-rules.md) |
| GitHub Setup | [docs/github-setup.md](docs/github-setup.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Publishing to PyPI | [docs/publishing-pypi.md](docs/publishing-pypi.md) |

## Publishing to PyPI

Maintainers: see [docs/publishing-pypi.md](docs/publishing-pypi.md) for prerequisites, one-time setup (`~/.pypirc`), build with **maturin** (this project uses a Rust extension), upload with twine, TestPyPI, and a pre-publish checklist.

Quick build and upload:

```bash
pip install build twine maturin
rm -rf dist/ build/ *.egg-info/
maturin build --release --out dist --sdist
twine upload dist/*
```

Verify: `pip install nodestradamus` then `nodestradamus --version`.

## License

MIT

## Credits

- [MCP](https://modelcontextprotocol.io) — Model Context Protocol
- [NetworkX](https://networkx.org) — Graph algorithms (Python)
- [petgraph](https://github.com/petgraph/petgraph) — Graph algorithms (Rust)
- [PyO3](https://pyo3.rs) — Rust-Python bindings
- [FAISS](https://github.com/facebookresearch/faiss) — Approximate nearest neighbor search (optional)
- [sentence-transformers](https://sbert.net) — Local embeddings (default)
- [Mistral](https://mistral.ai) — Codestral Embed API (optional, for API-based embeddings)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) — Code parsing (Python, TypeScript, Rust, SQL)
