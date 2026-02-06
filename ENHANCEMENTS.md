# Nodestradamus Enhancement Backlog

Single consolidated backlog from ENHANCEMENTS.md, NODESTRADAMUS_TESTING_ENHANCEMENTS.md, and NODESTRADAMUS_USER_TESTING.md. **Validity checked against current codebase** (definitions, handlers, analyzers) on 2026-01-30.

## Executive Summary

| Status | Count | Notes |
|--------|-------|--------|
| **Done** | 16 | Implemented; verified in tool schemas/handlers |
| **Valid (open)** | 16 | Still relevant; not yet implemented or partial |
| **Fixed this pass** | 7 | quick_start deps summary, H1–H4, M1, M2 |

**Sources:** LangChain repo testing (~2.4k Python files, monorepo), MCP tool schemas, `nodestradamus/` handlers and analyzers.

---

## 1. Implemented (reference only)

Verified via tool schemas and handlers:

- **project_scout** — Monorepo detection, packages, frameworks, entry points, suggested_ignores, recommended_workflow/next_steps. Entry point suggested_queries now use only file paths (H4).
- **analyze_graph** — `scope` (source_only/tests_only/all), `exclude_external`, `exclude_tests`; communities have `module_name`, `representative`, `category`; path algorithm returns `explanation` and `suggestions` when no path found; `summary_only` for communities/hierarchy. Hierarchy now classifies "unknown" nodes as `[stdlib]` or `[external]` with `exclude_external` filtering support (M1). Communities now classify as `source`/`tests`/`stdlib`/`external` with `stdlib_modules` count in summary (M2).
- **get_impact** — `compact`, `exclude_same_file`, `exclude_external`, depth summary.
- **codebase_health** — `scope`, `exclude_external` for bottlenecks.
- **analyze_deps** — `package` for monorepo scoping.
- **semantic_analysis** — Code preview (`snippet`) in search/similar results (H1), `preview_a`/`preview_b`/`preview` in duplicates (H3), `package` for monorepo scoping with separate caches per package (H2).
- **quick_start** — Exists; deps summary now uses correct keys from `summarize_digraph` (total_files, top_callers).
- **Docs** — getting-started workflow and related docs exist.

---

## 2. Still valid (open)

### Medium priority

| ID | Enhancement | Verification |
|----|-------------|--------------|
| M3 | **Progress for long operations** — Progress or streaming for long runs (e.g. analyze_deps, codebase_health, semantic embeddings). | No progress/streaming in handlers. |
| M4 | **Cache status in responses** — Add `cache_status` / `cache_age_seconds` to analyze_deps (and similar) responses; keep manage_cache for explicit management. | Responses do not include cache hit/age. |
| M5 | **quick_start / long-run progress** — For quick_start or multi-step runs, return current_step or progress so users know it's not stuck. | quick_start runs steps without progress payload. |
| M6 | **Filter built-ins from external_imports** — In analyze_deps summary, don't list ABC, Exception, BaseModel, Protocol etc. as generic "external"; attribute or filter built-in/stdlib. | deps summarization not checked for built-in filtering. |
| M7 | **Semantic search: penalize trivial files** — Reduce rank of tiny/empty files (e.g. 2-line `__init__.py`) in search/similar. | No length/size penalty in semantic handler. |

### Lower priority / UX

| ID | Enhancement | Verification |
|----|-------------|--------------|
| L1 | **Natural language summary mode** — Optional `format: "summary"` for graph tools (e.g. pagerank) returning short prose. | Not in schema. |
| L2 | **Mermaid / diagram export** — Optional `include_diagram` for path (or other) algorithms returning Mermaid snippet. | Not in schema. |
| L3 | **Export to file** — Optional `export_path` to write large results (e.g. hierarchy, strings refs) to a file and return a short summary. | Some tools write large output; no standard export_path param. |
| L4 | **"Why important" for pagerank/betweenness** — Optional short reason per node (e.g. "Imported by 47 files"). | Not in handler. |
| L5 | **analyze_strings summary** — For refs/usages, return top N inline plus "full results in file" instead of only huge payload. | Not verified. |
| L6 | **analyze_docs false positives** — Reduce stale refs for tool names (uv, pytest, ruff, mypy), keywords (True, False), package names; add confidence/severity. | Docs analyzer not inspected for allowlists. |
| L7 | **Stale embedding warning** — Warn when embeddings are older than recent file changes; optionally list stale files. | Not in semantic handler. |
| L8 | **Symbol "did you mean"** — For unknown or typo'd symbol, suggest close matches. | Not in get_impact/semantic. |

### Rule generation (from user testing — not in codebase)

| ID | Enhancement | Verification |
|----|-------------|--------------|
| R1 | **generate-rules command** — CLI: `nodestradamus generate-rules /path/to/repo --format cursor` running scout → deps → graph (pagerank/betweenness) → semantic → rule generation. | No generate-rules in codebase. |
| R2 | **Template system** — Jinja2 (or similar) templates (e.g. python-monorepo, typescript-nextjs) with variables from Nodestradamus analysis. | No templates in repo. |
| R3 | **Diff mode (stale rules)** — `nodestradamus rules-diff` (and optional `--ci`) to detect when rules are out of date vs. code. | No rules-diff. |
| R4 | **Multi-format output** — Emit Cursor (.mdc), CLAUDE.md, AGENTS.md, optional JSON from same pipeline. | No multi-format rule output. |

---

## 3. Cross-cutting

- **Global exclude_patterns / test filtering** — Several tools have their own scope/exclude; a consistent way to "exclude tests" or "exclude external" across tools is still requested (partially done per-tool).
- **Package parameter** — analyze_deps and semantic_analysis have `package`; extend to get_impact, codebase_health for monorepos.

---

## 4. Performance (observations, no change required)

- Semantic search: first run can be slow (embeddings); subsequent runs with cache are acceptable.
- quick_start: dominated by analyze_deps; progress (M5) would improve perceived performance.

---

## 5. What works well (no change)

- Error handling: missing file → suggestions; empty query → clear error; path not found → explanation + suggestions.
- project_scout monorepo and package discovery.
- Fusion mode (get_impact) and path-finding fallbacks.
- manage_cache for cache inspection/clear.

---

## Verification method

- **Tool schemas:** `mcps/user-nodestradamus/tools/*.json` and `nodestradamus/mcp/tools/definitions.py`.
- **Handlers:** `nodestradamus/mcp/tools/handlers/*.py` (workflows, graph_algorithms, semantic, core, etc.).
- **Analyzers:** `nodestradamus/analyzers/deps.py`, `project_scout.py`, `nodestradamus/mcp/tools/utils/summarize.py`.

Consolidation and validity check: 2026-01-30.
