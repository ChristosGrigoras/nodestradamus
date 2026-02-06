# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Lazy embedding with SQLite storage**: Embeddings are now stored in SQLite alongside chunk metadata. This enables:
  - **Scoped embedding**: Use `scope` parameter to embed only part of a codebase (e.g., `libs/auth/`). Multiple scopes accumulate - each run appends to the existing embeddings instead of replacing them.
  - **On-demand FAISS rebuild**: FAISS index is rebuilt from SQLite when needed (marked stale after scoped updates), ensuring consistent IDs across all chunks.
  - **Proper lazy loading**: Changed files are properly updated without stale embedding data.
  
- **Parallel API calls for Mistral embeddings**: Mistral provider now uses concurrent API requests (default: 4 workers) for ~3.5x faster embedding generation. Configurable via `NODESTRADAMUS_EMBEDDING_WORKERS` environment variable.

### Fixed

- **Embedding generation stalls on large codebases**: Fixed issue where the Mistral embedding provider would get stuck when encountering problematic code chunks (e.g., binary data, excessively long strings). Added binary search fallback mechanism that isolates and skips individual bad chunks while continuing to process the rest of the batch.
- **Streaming progress counter not incrementing**: Fixed `_compute_embeddings_streaming` to correctly update the total chunks counter based on successfully embedded items only.

### Changed

- `EmbeddingProvider.encode()` now returns an `EmbeddingResult` dataclass containing embeddings, success indices, and skipped chunk information, enabling graceful degradation when individual chunks fail.

## [0.1.0] - (unreleased)

- MCP server for codebase intelligence (dependency graphs, semantic search, impact analysis).
- Tools: project_scout, analyze_deps, get_impact, analyze_graph, semantic_analysis, codebase_health, and others.
- Optional Rust acceleration for graph algorithms (pagerank, strongly_connected, ancestors/descendants).
- Supported languages: Python, TypeScript/JavaScript, Rust, SQL, Bash, JSON.

[Unreleased]: https://github.com/ChristosGrigoras/nodestradamus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ChristosGrigoras/nodestradamus/releases/tag/v0.1.0
