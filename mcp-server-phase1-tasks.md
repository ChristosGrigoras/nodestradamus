# MCP Server Phase 1 Tasks

## Overview
Implement the core Codence MCP server (Phase 1 MVP) with 4 tools: `analyze_python`, `analyze_cooccurrence`, `get_impact`, and CLI entry point.

## Context Files
- `docs/mcp-server-spec.md` - Authoritative specification
- `scripts/analyze_python_deps.py` - Existing Python analyzer
- `scripts/analyze_git_cooccurrence.py` - Existing git analyzer
- `pyproject.toml` - Project configuration

## Success Criteria
- [x] `pip install .` works without errors
- [x] `codence serve` starts MCP server on stdio
- [x] Cursor can discover and list available tools
- [x] `analyze_python` returns valid graph for test fixtures

## Blockers / Risks
- MCP SDK compatibility with Python 3.12+
- Async wrapping of sync analysis functions

---

## Design Phase

- [x] Review MCP server specification
- [x] Define package structure and architecture
- [x] Identify existing code to reuse

---

## Setup Phase

- [x] Create `codence/` package directory structure
- [x] Update `pyproject.toml` with new dependencies (mcp, click, pydantic)
- [x] Update `pyproject.toml` with `codence` entry point

---

## Implementation Phase

### Task 1: Data Models
- [x] Create `codence/models/__init__.py`
- [x] Create `codence/models/graph.py` with Pydantic models:
  - `GraphNode` - node with id, type, file, name, line
  - `GraphEdge` - edge with from, to, type, resolved
  - `GraphMetadata` - metadata with analyzer, version, timestamp
  - `DependencyGraph` - full graph with nodes, edges, metadata
  - `CooccurrenceEdge` - edge with weight/strength
  - `CooccurrenceGraph` - co-occurrence specific graph
  - `ImpactReport` - upstream/downstream dependencies

### Task 2: Refactor Existing Scripts
- [x] Refactor `scripts/analyze_python_deps.py`:
  - Extract `analyze_directory()` as importable function
  - Keep CLI `main()` for backwards compatibility
- [x] Refactor `scripts/analyze_git_cooccurrence.py`:
  - Extract core functions as importable
  - Add `repo_path` parameter to `get_commits()`
  - Keep CLI `main()` for backwards compatibility

### Task 3: Analyzers Layer
- [x] Create `codence/analyzers/__init__.py`
- [x] Create `codence/analyzers/python_deps.py`:
  - Import from refactored script
  - Return Pydantic models
- [x] Create `codence/analyzers/git_cooccurrence.py`:
  - Import from refactored script
  - Return Pydantic models

### Task 4: Impact Analysis
- [x] Create `codence/analyzers/impact.py`:
  - `get_impact(graph, file_path, symbol, depth)` function
  - BFS/DFS traversal for upstream/downstream
  - Integration with co-occurrence data

### Task 5: MCP Server Core
- [x] Create `codence/mcp/__init__.py`
- [x] Create `codence/mcp/server.py`:
  - Server initialization with name, version
  - Stdio transport setup
  - Tool registration

### Task 6: MCP Tools
- [x] Create `codence/mcp/tools/__init__.py`
- [x] Create `codence/mcp/tools/graphs.py`:
  - `analyze_python` tool
  - `analyze_cooccurrence` tool
  - `get_impact` tool

### Task 7: CLI
- [x] Create `codence/__init__.py` with version
- [x] Create `codence/__main__.py` for `python -m codence`
- [x] Create `codence/cli.py`:
  - `serve` command (starts MCP server)
  - `analyze` command (standalone analysis)
  - `--version` flag

### Task 8: Integration
- [x] Update `pyproject.toml`:
  - Add dependencies
  - Add `codence` entry point
  - Update wheel packages
- [x] Test `pip install -e .`
- [x] Test `codence serve`

---

## Testing Phase

- [x] Create `tests/test_mcp_server.py`:
  - Test server initialization
  - Test tool schema generation
- [x] Run existing tests to verify no regressions (114 passed)
- [ ] Manual test with Cursor MCP configuration

---

## Completed Tasks

All Phase 1 implementation tasks completed on 2026-01-27.

---

*Last Updated: 2026-01-27*
