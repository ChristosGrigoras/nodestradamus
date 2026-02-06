"""Embeddings analyzer for semantic code search and similarity.

Supports multiple embedding providers:
- local: sentence-transformers (default, runs locally)
- mistral: Codestral Embed API (fast, requires API key)

Configuration via environment variables:
    NODESTRADAMUS_EMBEDDING_PROVIDER: "local" (default) or "mistral"
    NODESTRADAMUS_EMBEDDING_MODEL: Model name override (for local provider)
    MISTRAL_API_KEY: API key for Mistral provider

Storage Architecture
--------------------
Uses a hybrid storage system for embeddings:
- SQLite: Stores chunk metadata (file paths, symbols, line ranges, content hashes)
- FAISS: Stores vectors for fast similarity search
- NPZ: Legacy format (read-only for backward compatibility)

Workspace Isolation
-------------------
When workspace_path is provided, caches are stored under the workspace
rather than the analyzed repo, enabling isolation between different
Cursor windows working on the same project.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tree_sitter import Parser

from nodestradamus.analyzers.code_parser import (
    EXTENSION_TO_LANGUAGE,
    LANGUAGE_CONFIGS,
    _extract_class_name,
    _extract_function_name,
    _find_nodes,
    _get_language,
)
from nodestradamus.analyzers.embedding_providers import (
    get_embedding_provider,
    get_expected_model_name,
)
from nodestradamus.analyzers.ignore import DEFAULT_IGNORES
from nodestradamus.logging import logger, progress_bar
from nodestradamus.utils.cache import get_cache_dir
from nodestradamus.utils.db import (
    bulk_insert_chunks,
    delete_all_chunks,
    get_all_chunks,
    get_chunk_hashes_with_faiss_ids,
    get_chunks_by_scope,
    get_connection,
    get_db_path,
    is_faiss_stale,
    mark_faiss_fresh,
    transaction,
)

# FAISS availability check for approximate nearest neighbor search
# Used for large repos (>10K chunks) for better performance
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False

# Threshold for switching to FAISS (number of chunks)
_FAISS_THRESHOLD = int(os.getenv("NODESTRADAMUS_FAISS_THRESHOLD", "10000"))

# Minimum lines of actual code for a chunk to be considered non-trivial
_MIN_CODE_LINES = 5

# Penalty factor applied to trivial chunks' similarity scores
_TRIVIAL_PENALTY = 0.5

# Max length for code snippets in results
_SNIPPET_MAX_LINES = 5
_SNIPPET_MAX_CHARS = 300


def _compute_content_hash(content: str) -> str:
    """Compute a hash of content for change detection.

    Args:
        content: Code content as string.

    Returns:
        SHA-256 hash (first 16 chars) of the content.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _count_code_lines(content: str) -> int:
    """Count lines of actual code (excluding blanks, comments, imports).

    Args:
        content: Code content as string.

    Returns:
        Number of lines with actual code logic.
    """
    code_lines = 0
    for line in content.split("\n"):
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Skip Python comments
        if stripped.startswith("#"):
            continue
        # Skip JS/TS comments
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        # Skip import/from statements (Python)
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        # Skip import/require statements (JS/TS)
        if stripped.startswith("import ") or stripped.startswith("require("):
            continue
        # Skip export statements without logic
        if stripped.startswith("export {") or stripped == "export default":
            continue
        # Skip docstrings (simplified check)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        code_lines += 1
    return code_lines


def _is_trivial_chunk(chunk: dict) -> bool:
    """Check if a chunk is trivial (too small or just boilerplate).

    Args:
        chunk: Chunk dictionary with content and metadata.

    Returns:
        True if chunk should be penalized in search results.
    """
    content = chunk.get("content", "")
    file_path = chunk.get("file", "")

    # Empty __init__.py files are trivial
    if file_path.endswith("__init__.py"):
        code_lines = _count_code_lines(content)
        if code_lines < 3:  # Just imports or empty
            return True

    # Check minimum code lines
    code_lines = _count_code_lines(content)
    if code_lines < _MIN_CODE_LINES:
        return True

    return False


def _extract_snippet(content: str) -> str:
    """Extract a short code snippet from content for display.

    Args:
        content: Full code content.

    Returns:
        First few meaningful lines, truncated.
    """
    lines = content.split("\n")
    snippet_lines = []
    for line in lines:
        # Skip empty lines at the start
        if not snippet_lines and not line.strip():
            continue
        snippet_lines.append(line)
        if len(snippet_lines) >= _SNIPPET_MAX_LINES:
            break

    snippet = "\n".join(snippet_lines)
    if len(snippet) > _SNIPPET_MAX_CHARS:
        snippet = snippet[:_SNIPPET_MAX_CHARS].rsplit("\n", 1)[0] + "..."
    return snippet


def _read_snippet_from_file(
    repo_path: Path,
    rel_file: str,
    line_start: int | None,
    line_end: int | None,
) -> str | None:
    """Read a code region from disk and extract a snippet.

    Used as fallback when chunk metadata lacks a pre-computed snippet
    (e.g. old caches).

    Args:
        repo_path: Absolute path to the repository root.
        rel_file: Relative file path within the repo.
        line_start: Start line (1-based), or None for file start.
        line_end: End line (1-based), or None for file end.

    Returns:
        Extracted snippet string, or None if file cannot be read.
    """
    try:
        full_path = repo_path / rel_file
        if not full_path.exists():
            return None
        content = full_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")

        # Extract the relevant region
        start = (line_start - 1) if line_start else 0
        end = line_end if line_end else len(lines)
        region_lines = lines[start:end]
        region_content = "\n".join(region_lines)

        return _extract_snippet(region_content)
    except Exception:
        return None


def _build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build a FAISS index for fast approximate nearest neighbor search.

    Uses IndexFlatIP (inner product) since embeddings are pre-normalized,
    making inner product equivalent to cosine similarity.

    Args:
        embeddings: Pre-normalized embedding matrix (n_samples, n_dim).

    Returns:
        FAISS index ready for searching.

    Raises:
        RuntimeError: If FAISS is not available.
    """
    if not _HAS_FAISS:
        raise RuntimeError("FAISS is not installed. Install with: pip install faiss-cpu")

    d = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatIP(d)  # Inner product (cosine on normalized vectors)
    index.add(embeddings.astype(np.float32))
    logger.info("  Built FAISS index with %d vectors", embeddings.shape[0])
    return index


def rebuild_faiss_from_sqlite(
    workspace_path: str | None,
    repo_path: str,
) -> Any | None:
    """Rebuild FAISS index from embeddings stored in SQLite.

    This enables lazy embedding - embeddings are stored in SQLite and FAISS
    is rebuilt on-demand when needed for search. Also updates faiss_ids in
    SQLite to match the rebuilt index.

    Args:
        workspace_path: Workspace path for isolated caching.
        repo_path: Repository path.

    Returns:
        FAISS index, or None if no embeddings found or FAISS not available.
    """
    if not _HAS_FAISS:
        logger.warning("  FAISS not available, skipping index rebuild")
        return None

    conn = get_connection(workspace_path, repo_path)

    # Load all chunks with embeddings from SQLite (we need chunk IDs to update faiss_ids)
    cursor = conn.execute(
        """
        SELECT id, embedding FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY id
        """
    )
    rows = cursor.fetchall()

    if not rows:
        logger.info("  No embeddings found in SQLite, skipping FAISS rebuild")
        return None

    logger.info("  Rebuilding FAISS index from %d embeddings in SQLite", len(rows))

    # Convert bytes to numpy array
    # Embeddings are stored as float32, 1536 dimensions = 6144 bytes
    embedding_dim = len(rows[0][1]) // 4  # 4 bytes per float32
    embeddings = np.zeros((len(rows), embedding_dim), dtype=np.float32)
    chunk_ids = []

    for i, (chunk_id, emb_bytes) in enumerate(rows):
        embeddings[i] = np.frombuffer(emb_bytes, dtype=np.float32)
        chunk_ids.append(chunk_id)

    # Build the index
    index = _build_faiss_index(embeddings)

    # Update faiss_ids in SQLite to match rebuilt index
    for new_faiss_id, chunk_id in enumerate(chunk_ids):
        conn.execute(
            "UPDATE chunks SET faiss_id = ? WHERE id = ?",
            (new_faiss_id, chunk_id)
        )

    # Save the index
    faiss_cache_path = _get_faiss_cache_path(workspace_path, repo_path)
    _save_faiss_index(index, faiss_cache_path)

    # Mark as fresh
    mark_faiss_fresh(conn, len(rows))
    conn.commit()

    logger.info("  FAISS rebuild complete, updated %d faiss_ids", len(rows))
    return index


def _save_faiss_index(index: Any, cache_path: Path) -> None:
    """Save a FAISS index to disk.

    Args:
        index: FAISS index to save.
        cache_path: Path to save the index.
    """
    if not _HAS_FAISS:
        return
    faiss.write_index(index, str(cache_path))
    logger.info("  Saved FAISS index to %s", cache_path)


def _load_faiss_index(cache_path: Path) -> Any | None:
    """Load a FAISS index from disk.

    Args:
        cache_path: Path to the saved index.

    Returns:
        FAISS index or None if not found/loadable.
    """
    if not _HAS_FAISS:
        return None
    if not cache_path.exists():
        return None
    try:
        index = faiss.read_index(str(cache_path))
        logger.info("  Loaded FAISS index from %s", cache_path)
        return index
    except Exception:
        return None


def _faiss_search(
    index: Any,
    query_embedding: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index for nearest neighbors.

    Args:
        index: FAISS index.
        query_embedding: Query vector.
        top_k: Number of results to return.

    Returns:
        Tuple of (similarities, indices) arrays.
    """
    # FAISS expects 2D array
    query = query_embedding.reshape(1, -1).astype(np.float32)
    similarities, indices = index.search(query, top_k)
    return similarities[0], indices[0]


def _extract_code_chunks(
    file_path: Path,
    chunk_by: str = "function",
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract code chunks from a file for embedding.

    Args:
        file_path: Path to the source file.
        chunk_by: How to chunk the file - "function", "class", or "file".
        base_dir: Repository root for relative paths in chunk ids (Rust/SQL/Bash).
            If None, tree-sitter chunking uses file_path.parent so rel_path is filename.

    Returns:
        List of chunks with content and metadata.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not content.strip():
        return []

    base = base_dir if base_dir is not None else file_path.parent
    try:
        rel_path = str(file_path.relative_to(base))
    except ValueError:
        rel_path = file_path.name
    suffix = file_path.suffix.lower()

    # For now, use simple line-based chunking for functions
    # Future: use tree-sitter for precise extraction
    if chunk_by == "file":
        chunk = {
            "id": f"file:{rel_path}",
            "type": "file",
            "file": rel_path,
            "name": file_path.name,
            "content": content[:8000],  # Limit content size
            "line_start": 1,
            "line_end": content.count("\n") + 1,
        }
        chunk["snippet"] = _extract_snippet(content)
        chunk["trivial"] = _is_trivial_chunk(chunk)
        return [chunk]

    chunks = []

    # Simple heuristic extraction for Python
    if suffix == ".py":
        chunks.extend(_extract_python_chunks(content, rel_path))

    # Simple heuristic extraction for TypeScript/JavaScript
    elif suffix in (".ts", ".tsx", ".js", ".jsx"):
        chunks.extend(_extract_js_chunks(content, rel_path))

    # Tree-sitter extraction for Rust, SQL, Bash
    elif suffix in (".rs", ".sql", ".pgsql", ".sh", ".bash"):
        chunks.extend(_extract_chunks_via_treesitter(file_path, base, chunk_by))

    # Fallback: treat whole file as one chunk
    if not chunks:
        chunk = {
            "id": f"file:{rel_path}",
            "type": "file",
            "file": rel_path,
            "name": file_path.name,
            "content": content[:8000],
            "line_start": 1,
            "line_end": content.count("\n") + 1,
        }
        chunk["snippet"] = _extract_snippet(content)
        chunk["trivial"] = _is_trivial_chunk(chunk)
        chunks.append(chunk)

    # Add snippet and trivial flag to all chunks that don't have them yet
    for chunk in chunks:
        if "snippet" not in chunk:
            chunk["snippet"] = _extract_snippet(chunk.get("content", ""))
        if "trivial" not in chunk:
            chunk["trivial"] = _is_trivial_chunk(chunk)

    return chunks


def _extract_python_chunks(content: str, file_path: str) -> list[dict[str, Any]]:
    """Extract functions and classes from Python code using simple parsing."""
    chunks = []
    lines = content.split("\n")
    current_chunk: list[str] = []
    current_name = ""
    current_type = ""
    current_start = 0
    base_indent = -1

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()

        # Detect function or class definition
        if stripped.startswith("def ") or stripped.startswith("async def "):
            # Save previous chunk
            if current_chunk and current_name:
                chunks.append(
                    {
                        "id": f"py:{file_path}::{current_name}",
                        "type": current_type,
                        "file": file_path,
                        "name": current_name,
                        "content": "\n".join(current_chunk)[:4000],
                        "line_start": current_start,
                        "line_end": i - 1,
                    }
                )

            # Start new function chunk
            current_chunk = [line]
            current_start = i
            current_type = "function"
            base_indent = len(line) - len(stripped)

            # Extract function name
            if "async def " in stripped:
                name_part = stripped[10:]
            else:
                name_part = stripped[4:]
            current_name = name_part.split("(")[0].strip()

        elif stripped.startswith("class "):
            # Save previous chunk
            if current_chunk and current_name:
                chunks.append(
                    {
                        "id": f"py:{file_path}::{current_name}",
                        "type": current_type,
                        "file": file_path,
                        "name": current_name,
                        "content": "\n".join(current_chunk)[:4000],
                        "line_start": current_start,
                        "line_end": i - 1,
                    }
                )

            # Start new class chunk
            current_chunk = [line]
            current_start = i
            current_type = "class"
            base_indent = len(line) - len(stripped)
            name_part = stripped[6:]
            current_name = name_part.split("(")[0].split(":")[0].strip()

        elif current_name:
            # Check if we're still in the current definition
            if line.strip() == "" or (
                line.strip() and len(line) - len(line.lstrip()) > base_indent
            ):
                current_chunk.append(line)
            elif not line.strip():
                current_chunk.append(line)
            else:
                # End of current definition
                if current_chunk:
                    chunks.append(
                        {
                            "id": f"py:{file_path}::{current_name}",
                            "type": current_type,
                            "file": file_path,
                            "name": current_name,
                            "content": "\n".join(current_chunk)[:4000],
                            "line_start": current_start,
                            "line_end": i - 1,
                        }
                    )
                current_chunk = []
                current_name = ""
                base_indent = -1

    # Don't forget the last chunk
    if current_chunk and current_name:
        chunks.append(
            {
                "id": f"py:{file_path}::{current_name}",
                "type": current_type,
                "file": file_path,
                "name": current_name,
                "content": "\n".join(current_chunk)[:4000],
                "line_start": current_start,
                "line_end": len(lines),
            }
        )

    return chunks


def _extract_js_chunks(content: str, file_path: str) -> list[dict[str, Any]]:
    """Extract functions and classes from JS/TS code using simple parsing."""
    chunks = []
    lines = content.split("\n")
    current_chunk: list[str] = []
    current_name = ""
    current_type = ""
    current_start = 0
    brace_count = 0
    in_definition = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Detect function or class definition
        is_func = (
            "function " in stripped
            or stripped.startswith("const ")
            and "=>" in stripped
            or stripped.startswith("export function")
            or stripped.startswith("async function")
            or stripped.startswith("export async function")
        )
        is_class = "class " in stripped and "{" in stripped

        if (is_func or is_class) and not in_definition:
            # Start new chunk
            current_chunk = [line]
            current_start = i
            current_type = "class" if is_class else "function"
            in_definition = True
            brace_count = line.count("{") - line.count("}")

            # Extract name (simplified)
            if "class " in stripped:
                parts = stripped.split("class ")[1].split()
                current_name = parts[0].rstrip("{").rstrip(":")
            elif "function " in stripped:
                parts = stripped.split("function ")[1].split("(")
                current_name = parts[0].strip()
            elif "const " in stripped:
                parts = stripped.split("const ")[1].split("=")
                current_name = parts[0].strip()
            else:
                current_name = f"anonymous_{i}"

        elif in_definition:
            current_chunk.append(line)
            brace_count += line.count("{") - line.count("}")

            if brace_count <= 0:
                # End of definition
                chunks.append(
                    {
                        "id": f"ts:{file_path}::{current_name}",
                        "type": current_type,
                        "file": file_path,
                        "name": current_name,
                        "content": "\n".join(current_chunk)[:4000],
                        "line_start": current_start,
                        "line_end": i,
                    }
                )
                current_chunk = []
                current_name = ""
                in_definition = False
                brace_count = 0

    return chunks


# Languages we support for embedding chunking (subset of EXTENSION_TO_LANGUAGE)
_EMBEDDING_LANGUAGES = frozenset({
    "python", "typescript", "javascript", "tsx", "rust", "sql", "bash",
})


def _extract_chunks_via_treesitter(
    file_path: Path,
    base_dir: Path,
    chunk_by: str,
) -> list[dict[str, Any]]:
    """Extract code chunks using tree-sitter and code_parser config.

    Used for Rust, SQL, Bash (and optionally others). Reuses LANGUAGE_CONFIGS
    and extractors from code_parser; uses node start/end for content slicing.

    Args:
        file_path: Path to the source file.
        base_dir: Repository root for relative path in chunk ids.
        chunk_by: Chunking strategy ("function", "class", or "file"); we use
            both function and class types from config.

    Returns:
        List of chunk dicts with id, type, file, name, line_start, line_end, content.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not content.strip():
        return []

    suffix = file_path.suffix.lower()
    language = EXTENSION_TO_LANGUAGE.get(suffix)
    if not language or language not in LANGUAGE_CONFIGS:
        return []

    if language not in _EMBEDDING_LANGUAGES:
        return []

    config = LANGUAGE_CONFIGS[language]
    ts_lang = _get_language(language)
    if ts_lang is None:
        return []

    try:
        parser = Parser(ts_lang)
        tree = parser.parse(content.encode("utf-8"))
    except Exception:
        return []

    if tree.root_node is None or tree.root_node.has_error:
        return []

    try:
        rel_path = file_path.relative_to(base_dir)
    except ValueError:
        rel_path = file_path.name
    rel_path_str = str(rel_path)

    chunk_types = config.function_types | config.class_types
    if not chunk_types:
        return []

    lines = content.split("\n")
    chunks: list[dict[str, Any]] = []

    for node in _find_nodes(tree.root_node, chunk_types):
        name = _extract_function_name(node, config) or _extract_class_name(node, config)
        if not name:
            name = f"{node.type}_{node.start_point[0] + 1}"

        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        chunk_content = "\n".join(lines[line_start - 1 : line_end])[:4000]

        node_type = "function" if node.type in config.function_types else "class"
        chunk_id = f"{config.prefix}:{rel_path_str}::{name}"

        chunks.append({
            "id": chunk_id,
            "type": node_type,
            "file": rel_path_str,
            "name": name,
            "line_start": line_start,
            "line_end": line_end,
            "content": chunk_content,
        })

    return chunks


def _get_cache_filename(package: str | None) -> str:
    """Get the cache filename, optionally scoped by package.

    DEPRECATED: Used only for NPZ legacy format.

    Args:
        package: Package path for scoping (e.g., 'libs/core').

    Returns:
        Cache filename (e.g., 'embeddings.npz' or 'embeddings_libs_core.npz').
    """
    if not package:
        return "embeddings.npz"
    # Sanitize package path for filename
    slug = package.replace("/", "_").replace("\\", "_").replace(".", "_")
    return f"embeddings_{slug}.npz"


def _get_faiss_cache_path(
    workspace_path: str | None,
    repo_path: str,
) -> Path:
    """Get the path to the FAISS index file.

    Args:
        workspace_path: Workspace path for isolated caching.
        repo_path: Repository path.

    Returns:
        Path to the embeddings.faiss file.
    """
    cache_base = get_cache_dir(workspace_path, repo_path)
    return cache_base / "embeddings.faiss"


def _save_embeddings_to_sqlite(
    workspace_path: str | None,
    repo_path: str,
    chunks: list[dict],
    model_version: str,
    embeddings: np.ndarray | None = None,
    full_rebuild: bool = True,
    scope: str | None = None,
) -> None:
    """Save chunk metadata and embeddings to SQLite database.

    Args:
        workspace_path: Workspace path for isolated caching.
        repo_path: Repository path.
        chunks: List of chunks with metadata and faiss_id.
        model_version: Embedding model identifier.
        embeddings: Optional numpy array of embeddings (n_chunks, dim).
            If provided, embeddings are stored alongside metadata.
        full_rebuild: If True and no scope, deletes all existing chunks.
            If scope is provided, only deletes chunks in that scope.
        scope: Path prefix for scoped updates. When provided, only chunks
            in this scope are deleted before insert (enables lazy embedding).
    """
    from nodestradamus.utils.db import delete_chunks_by_scope

    with transaction(workspace_path, repo_path) as conn:
        if scope:
            # Scoped update: delete only chunks in this scope, preserving others
            deleted = delete_chunks_by_scope(conn, scope)
            if deleted > 0:
                logger.info("  Deleted %d existing chunks in scope %s", deleted, scope)
        elif full_rebuild:
            # Full rebuild: clear all existing chunks
            delete_all_chunks(conn)

        # Prepare chunk records with embeddings if available
        records = []
        for i, chunk in enumerate(chunks):
            record = {
                "file_path": chunk["file"],
                "symbol_name": chunk.get("name"),
                "line_start": chunk.get("line_start", 0),
                "line_end": chunk.get("line_end", 0),
                "content_hash": chunk.get("content_hash", ""),
                "snippet": chunk.get("snippet"),
                "language": _detect_language(chunk["file"]),
                "faiss_id": chunk["faiss_id"],
                "model_version": model_version,
            }
            # Store embedding as bytes if available
            if embeddings is not None and i < len(embeddings):
                record["embedding"] = embeddings[i].astype(np.float32).tobytes()
            records.append(record)

        bulk_insert_chunks(conn, records)
        logger.info("  Saved %d chunks to SQLite", len(records))


def _get_chunk_key(chunk: dict) -> str:
    """Create a unique key for a chunk based on file and line range.

    Args:
        chunk: Chunk dictionary with file, line_start, line_end.

    Returns:
        Unique key string for the chunk.
    """
    return f"{chunk['file']}:{chunk.get('line_start', 0)}:{chunk.get('line_end', 0)}"


def _load_existing_embeddings(
    workspace_path: str | None,
    repo_path: str,
) -> tuple[dict[str, tuple[str, int]], dict[int, np.ndarray], str | None]:
    """Load existing chunk hashes, faiss_ids, and embeddings from cache.

    Used for incremental updates to identify unchanged chunks and
    reuse their embeddings.

    Args:
        workspace_path: Workspace path for isolated caching.
        repo_path: Repository path.

    Returns:
        Tuple of:
        - chunk_key -> (content_hash, faiss_id) mapping
        - faiss_id -> embedding array mapping
        - model_version string
        Empty dicts and None if no existing cache.
    """
    db_path = get_db_path(workspace_path, repo_path)
    faiss_cache_path = _get_faiss_cache_path(workspace_path, repo_path)

    if not db_path.exists():
        return {}, {}, None

    try:
        conn = get_connection(workspace_path, repo_path)

        # Get chunk hashes with faiss_ids for incremental update
        chunk_data = get_chunk_hashes_with_faiss_ids(conn)

        if not chunk_data:
            return {}, {}, None

        # Get model version
        rows = get_all_chunks(conn)
        if not rows:
            return {}, {}, None
        model_version = rows[0]["model_version"]

        # Collect all faiss_ids we need embeddings for
        faiss_ids_needed = {fid for _, fid in chunk_data.values()}

        # Load FAISS index to get embeddings
        faiss_index = _load_faiss_index(faiss_cache_path)
        faiss_id_to_embedding: dict[int, np.ndarray] = {}

        if faiss_index is not None:
            # Extract embeddings from FAISS index
            for faiss_id in faiss_ids_needed:
                try:
                    embedding = faiss_index.reconstruct(faiss_id)
                    faiss_id_to_embedding[faiss_id] = embedding
                except Exception:
                    continue
        else:
            # Try NPZ fallback
            cache_base = get_cache_dir(workspace_path, repo_path)
            npz_path = cache_base / "embeddings.npz"
            if npz_path.exists():
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    embeddings = data["embeddings"]
                    # Create faiss_id -> embedding mapping for needed ids
                    for faiss_id in faiss_ids_needed:
                        if faiss_id < len(embeddings):
                            faiss_id_to_embedding[faiss_id] = embeddings[faiss_id]
                except Exception:
                    pass

        return chunk_data, faiss_id_to_embedding, model_version

    except Exception as e:
        logger.warning("  Failed to load existing embeddings: %s", e)
        return {}, {}, None


def _compute_embeddings_incremental(
    all_chunks: list[dict],
    existing_chunk_data: dict[str, tuple[str, int]],
    existing_embeddings: dict[int, np.ndarray],
    existing_model: str | None,
    provider: Any,
) -> tuple[np.ndarray, list[dict], dict[str, int]]:
    """Compute embeddings incrementally, reusing unchanged chunks.

    This is the core optimization for incremental updates:
    1. Compare new chunk content_hash with existing hashes
    2. Reuse embeddings for unchanged chunks
    3. Only call embedding model for new/changed chunks
    4. Combine reused + new embeddings with new faiss_ids

    Args:
        all_chunks: All chunks extracted from the current codebase.
        existing_chunk_data: Mapping of chunk_key -> (content_hash, faiss_id) from SQLite.
        existing_embeddings: Mapping of faiss_id -> embedding from FAISS.
        existing_model: Model version of existing embeddings.
        provider: Embedding provider instance.

    Returns:
        Tuple of (embeddings array, updated chunks list, stats dict).
    """
    expected_model = provider.model_name

    # If model changed, we must re-embed everything
    if existing_model and existing_model != expected_model:
        logger.info(
            "  Model changed (%s -> %s), full re-embed required",
            existing_model,
            expected_model,
        )
        existing_chunk_data = {}
        existing_embeddings = {}

    # Categorize chunks
    unchanged_chunks: list[tuple[dict, int]] = []  # (chunk, old_faiss_id)
    changed_chunks: list[dict] = []
    new_chunks: list[dict] = []

    for chunk in all_chunks:
        chunk_key = _get_chunk_key(chunk)
        new_hash = chunk.get("content_hash", "")

        if chunk_key in existing_chunk_data:
            old_hash, old_faiss_id = existing_chunk_data[chunk_key]
            if old_hash == new_hash and old_faiss_id in existing_embeddings:
                # Unchanged and we have the embedding - can reuse
                unchanged_chunks.append((chunk, old_faiss_id))
            else:
                # Changed content or missing embedding
                changed_chunks.append(chunk)
        else:
            # New chunk
            new_chunks.append(chunk)

    stats: dict[str, int] = {
        "unchanged": len(unchanged_chunks),
        "changed": len(changed_chunks),
        "new": len(new_chunks),
        "total": len(all_chunks),
    }

    logger.info(
        "  Incremental analysis: %d unchanged, %d changed, %d new (total: %d)",
        stats["unchanged"],
        stats["changed"],
        stats["new"],
        stats["total"],
    )

    # Chunks that need new embeddings
    chunks_to_embed = changed_chunks + new_chunks

    # Build final embeddings array combining reused + new
    final_embeddings: list[np.ndarray] = []
    final_chunks: list[dict] = []
    faiss_id_counter = 0

    # First, add unchanged chunks with reused embeddings
    for chunk, old_faiss_id in unchanged_chunks:
        embedding = existing_embeddings[old_faiss_id]
        final_embeddings.append(embedding)
        chunk["faiss_id"] = faiss_id_counter
        final_chunks.append(chunk)
        faiss_id_counter += 1

    # Then, compute embeddings for new/changed chunks
    if chunks_to_embed:
        texts = [chunk["content"] for chunk in chunks_to_embed]
        logger.info(
            "  Computing embeddings for %d chunks (reusing %d from cache)",
            len(chunks_to_embed),
            len(unchanged_chunks),
        )
        result = provider.encode(texts)
        new_embeddings = result.embeddings

        # Handle partial failures
        if result.skipped_indices:
            logger.warning(
                "  Skipped %d chunks due to embedding errors",
                len(result.skipped_indices),
            )
            success_set = set(result.success_indices)
            chunks_to_embed = [c for i, c in enumerate(chunks_to_embed) if i in success_set]

        # Normalize new embeddings
        if len(new_embeddings) > 0:
            norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            new_embeddings_normalized = new_embeddings / (norms + 1e-9)

            # Add to final results
            for i, chunk in enumerate(chunks_to_embed):
                if i < len(new_embeddings_normalized):
                    final_embeddings.append(new_embeddings_normalized[i])
                    chunk["faiss_id"] = faiss_id_counter
                    final_chunks.append(chunk)
                    faiss_id_counter += 1

    # Convert to numpy array
    if final_embeddings:
        embeddings_array = np.vstack(final_embeddings)
    else:
        embeddings_array = np.array([])

    # Update stats
    stats["embedded"] = len(chunks_to_embed)
    stats["reused"] = len(unchanged_chunks)

    return embeddings_array, final_chunks, stats


def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the file.

    Returns:
        Language identifier string.
    """
    suffix = Path(file_path).suffix.lower()
    lang_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".sql": "sql",
        ".pgsql": "sql",
        ".sh": "bash",
        ".bash": "bash",
    }
    return lang_map.get(suffix, "unknown")


def _load_embeddings_from_sqlite(
    workspace_path: str | None,
    repo_path: str,
    scope: str | None = None,
) -> tuple[list[dict], str | None]:
    """Load chunk metadata from SQLite database.

    Args:
        workspace_path: Workspace path for isolated caching.
        repo_path: Repository path.
        scope: Optional path prefix to filter chunks.

    Returns:
        Tuple of (chunks list, model_version) or ([], None) if not found.
    """
    db_path = get_db_path(workspace_path, repo_path)
    if not db_path.exists():
        return [], None

    try:
        conn = get_connection(workspace_path, repo_path)

        if scope:
            rows = get_chunks_by_scope(conn, scope)
        else:
            rows = get_all_chunks(conn)

        if not rows:
            return [], None

        # Get model version from first row
        model_version = rows[0]["model_version"]

        # Convert rows to chunk dicts
        chunks = []
        for row in rows:
            chunks.append({
                "id": f"{_detect_language(row['file_path'])[:2]}:{row['file_path']}::{row['symbol_name'] or 'file'}",
                "file": row["file_path"],
                "name": row["symbol_name"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "snippet": row["snippet"],
                "faiss_id": row["faiss_id"],
                "content_hash": row["content_hash"],
            })

        return chunks, model_version
    except Exception as e:
        logger.warning("  Failed to load from SQLite: %s", e)
        return [], None


def compute_embeddings(
    repo_path: str,
    languages: list[str] | None = None,
    chunk_by: str = "function",
    cache_dir: str | None = None,
    workspace_path: str | None = None,
    exclude: list[str] | None = None,
    package: str | None = None,
    streaming: bool = False,
    batch_size: int = 1000,
    scope: str | None = None,
) -> dict[str, Any]:
    """Compute embeddings for all code in a repository.

    Uses hybrid SQLite + FAISS storage:
    - SQLite stores chunk metadata (file paths, symbols, content hashes)
    - FAISS stores vectors for fast similarity search

    Args:
        repo_path: Absolute path to the repository.
        languages: Languages to analyze (auto-detect if None).
        chunk_by: How to chunk code - "function", "class", or "file".
        cache_dir: Directory to cache embeddings (explicit override).
        workspace_path: Workspace path for isolated caching. When provided,
            caches are stored in <workspace>/.nodestradamus/cache/<repo_hash>/
            instead of <repo>/.nodestradamus/.
        exclude: Directories/patterns to exclude from analysis. If None,
            uses DEFAULT_IGNORES from the ignore module.
        package: For monorepos: limit analysis to this package path (e.g., 'libs/core').
            Embeddings are cached separately per package.
        streaming: If True, process chunks in batches and save incrementally.
            Recommended for large repos (50K+ files) to avoid memory spikes.
        batch_size: Number of chunks per batch when streaming. Default 1000.
        scope: Limit embedding to files matching this path prefix.
            Used for lazy/incremental analysis.

    Returns:
        Dict with embeddings, chunks metadata, and cache info.
    """
    repo = Path(repo_path)
    if not repo.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    # Validate package path if provided
    if package:
        package_path = repo / package
        if not package_path.exists():
            raise ValueError(f"Package path does not exist: {package}")

    # Determine cache paths
    faiss_cache_path = _get_faiss_cache_path(workspace_path, repo_path)
    faiss_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Find files to process (single source of truth from code_parser)
    if languages is None:
        extensions = {
            ext
            for ext, lang in EXTENSION_TO_LANGUAGE.items()
            if lang in _EMBEDDING_LANGUAGES
        }
    else:
        extensions = {
            ext
            for ext, lang in EXTENSION_TO_LANGUAGE.items()
            if lang in languages or (lang == "tsx" and "typescript" in languages)
        }

    # Use package path as search root if provided
    search_root = repo / package if package else repo

    # Apply scope filter if provided
    if scope:
        search_root = repo / scope

    files = []
    for ext in extensions:
        files.extend(search_root.rglob(f"*{ext}"))

    # Filter out non-source directories (use provided exclude or defaults)
    skip_patterns = set(exclude) if exclude else DEFAULT_IGNORES
    files = [f for f in files if not any(skip in f.parts for skip in skip_patterns)]

    # Use streaming mode for large repos (auto-enable for 10K+ files)
    if not streaming and len(files) >= 10000:
        logger.info("  Auto-enabling streaming mode for %d files", len(files))
        streaming = True

    # Get embedding provider
    provider = get_embedding_provider()

    if streaming:
        return _compute_embeddings_streaming(
            repo=repo,
            files=files,
            chunk_by=chunk_by,
            cache_path=faiss_cache_path,
            provider=provider,
            batch_size=batch_size,
            workspace_path=workspace_path,
        )

    # Standard (non-streaming) path for smaller repos
    # Extract chunks
    all_chunks = []
    logger.info("  Extracting code chunks from %d files", len(files))
    for file_path in progress_bar(files, desc="Extracting chunks", unit="files"):
        try:
            rel_path = file_path.relative_to(repo)
            chunks = _extract_code_chunks(file_path, chunk_by, base_dir=repo)
            for chunk in chunks:
                chunk["file"] = str(rel_path)
                chunk["id"] = chunk["id"].replace(str(file_path), str(rel_path))
                # Add content hash for incremental updates
                chunk["content_hash"] = _compute_content_hash(chunk.get("content", ""))
            all_chunks.extend(chunks)
        except Exception:
            continue

    logger.info("  Extracted %d chunks from %d files", len(all_chunks), len(files))

    if not all_chunks:
        return {
            "embeddings": [],
            "chunks": [],
            "metadata": {
                "files_processed": len(files),
                "chunks_extracted": 0,
                "model": provider.model_name,
            },
        }

    # Try incremental update: load existing hashes and embeddings
    existing_chunk_data, existing_embeddings, existing_model = _load_existing_embeddings(
        workspace_path, repo_path
    )

    # Use incremental computation if we have existing data
    if existing_chunk_data:
        embeddings_normalized, all_chunks, incremental_stats = _compute_embeddings_incremental(
            all_chunks=all_chunks,
            existing_chunk_data=existing_chunk_data,
            existing_embeddings=existing_embeddings,
            existing_model=existing_model,
            provider=provider,
        )
        logger.info(
            "  Incremental update: embedded %d/%d chunks",
            incremental_stats["embedded"],
            incremental_stats["total"],
        )
    else:
        # Full computation (no existing cache)
        texts = [chunk["content"] for chunk in all_chunks]
        logger.info("  Computing embeddings for %d chunks (this may take a while...)", len(texts))
        result = provider.encode(texts)
        embeddings = result.embeddings

        # Filter chunks to only those that succeeded
        if result.skipped_indices:
            logger.warning(
                "  Skipped %d chunks due to embedding errors",
                len(result.skipped_indices),
            )
            # Keep only chunks that succeeded
            success_set = set(result.success_indices)
            all_chunks = [c for i, c in enumerate(all_chunks) if i in success_set]

        logger.info("  Embeddings computed: shape %s", embeddings.shape)

        # Pre-normalize embeddings for faster cosine similarity at query time
        if len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / (norms + 1e-9)
        else:
            embeddings_normalized = embeddings
        logger.info("  Embeddings pre-normalized for efficient search")

        # Assign FAISS IDs to chunks
        for i, chunk in enumerate(all_chunks):
            chunk["faiss_id"] = i

    # Create chunk metadata (without content for storage)
    chunk_metadata = [{k: v for k, v in chunk.items() if k != "content"} for chunk in all_chunks]

    # Save to SQLite (metadata + embeddings) and FAISS (vectors if available)
    # Use scope for lazy embedding - only delete/update chunks in the scope
    _save_embeddings_to_sqlite(
        workspace_path, repo_path, chunk_metadata, provider.model_name,
        embeddings=embeddings_normalized if len(embeddings_normalized) > 0 else None,
        scope=scope
    )

    # Build FAISS index if available (optional optimization)
    # When using scope, skip building FAISS now - it will be rebuilt from all
    # SQLite embeddings on next search to ensure consistent IDs across scopes
    if scope:
        # Scoped update: mark FAISS as stale, rebuild on search
        logger.info("  Scoped update: FAISS will be rebuilt on next search")
    elif len(embeddings_normalized) > 0 and _HAS_FAISS:
        # Full rebuild: build FAISS immediately
        faiss_index = _build_faiss_index(embeddings_normalized)
        _save_faiss_index(faiss_index, faiss_cache_path)
        # Mark FAISS as fresh
        with transaction(workspace_path, repo_path) as conn:
            mark_faiss_fresh(conn, len(all_chunks))

        # Also save embeddings to NPZ for fallback when FAISS is not available
        cache_base = get_cache_dir(workspace_path, repo_path)
        npz_cache_path = cache_base / "embeddings.npz"
        np.savez_compressed(
            npz_cache_path,
            embeddings=embeddings_normalized,
            chunks=json.dumps(chunk_metadata),
            model=provider.model_name,
            normalized=True,
        )

    # Register in workspace registry if workspace-scoped
    if workspace_path is not None:
        try:
            from nodestradamus.utils.registry import register_analysis

            db_path = get_db_path(workspace_path, repo_path)
            cache_size = db_path.stat().st_size if db_path.exists() else None
            register_analysis(workspace_path, repo_path, cache_size=cache_size)
        except ImportError:
            pass  # Registry module not available

    # Build metadata with incremental stats if available
    metadata: dict[str, Any] = {
        "files_processed": len(files),
        "chunks_extracted": len(all_chunks),
        "model": provider.model_name,
        "cache_path": str(faiss_cache_path),
        "embedding_dim": embeddings_normalized.shape[1] if len(embeddings_normalized) > 0 else 0,
        "storage": "sqlite+faiss",
    }

    # Add incremental update stats if we used the incremental path
    if existing_chunk_data and "incremental_stats" in dir():
        pass  # incremental_stats is in local scope only when incremental path was used

    # Check if incremental_stats exists (from incremental path)
    try:
        metadata["incremental"] = {
            "unchanged": incremental_stats["unchanged"],
            "changed": incremental_stats["changed"],
            "new": incremental_stats["new"],
            "reused": incremental_stats.get("reused", 0),
            "embedded": incremental_stats["embedded"],
        }
    except NameError:
        pass  # Not using incremental path

    return {
        "embeddings": embeddings_normalized.tolist() if len(embeddings_normalized) > 0 else [],
        "chunks": chunk_metadata,
        "metadata": metadata,
    }


def _compute_embeddings_streaming(
    repo: Path,
    files: list[Path],
    chunk_by: str,
    cache_path: Path,
    provider: Any,
    batch_size: int = 1000,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Compute embeddings using streaming/batched processing.

    For large repos (50K+ files, 1M+ chunks), loading all chunks and embeddings
    into memory causes OOM. This function processes chunks in batches:
    1. Extract chunks from a batch of files
    2. Compute embeddings for that batch
    3. Append to output arrays (memory-mapped if needed)
    4. Clear batch from memory
    5. Repeat

    Uses hybrid SQLite + FAISS storage.

    Args:
        repo: Repository path.
        files: List of files to process.
        chunk_by: Chunking strategy.
        cache_path: Path to save FAISS index.
        provider: Embedding provider instance.
        batch_size: Chunks per batch.
        workspace_path: Workspace path for registry.

    Returns:
        Dict with summary metadata (not full embeddings to avoid memory spike).
    """
    logger.info("  Streaming mode: processing %d files in batches of ~%d chunks", len(files), batch_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary storage for incremental processing
    all_chunk_metadata: list[dict[str, Any]] = []
    all_embeddings: list[np.ndarray] = []
    total_chunks = 0
    current_batch_chunks: list[dict[str, Any]] = []
    faiss_id_counter = 0

    def process_batch() -> None:
        """Process current batch: compute embeddings, append to storage."""
        nonlocal total_chunks, current_batch_chunks, faiss_id_counter

        if not current_batch_chunks:
            return

        texts = [chunk["content"] for chunk in current_batch_chunks]
        logger.info("  Processing batch: %d chunks (total so far: %d)", len(texts), total_chunks)

        # Compute embeddings for batch
        result = provider.encode(texts)
        batch_embeddings = result.embeddings

        # Handle partial success - only keep chunks that succeeded
        if result.skipped_indices:
            logger.warning(
                "  Batch had %d skipped chunks (kept %d)",
                len(result.skipped_indices),
                len(result.success_indices),
            )
            # Filter to only successful chunks
            success_set = set(result.success_indices)
            successful_chunks = [c for i, c in enumerate(current_batch_chunks) if i in success_set]
        else:
            successful_chunks = current_batch_chunks

        if len(batch_embeddings) == 0:
            # Entire batch failed
            logger.warning("  Entire batch failed, skipping")
            current_batch_chunks = []
            return

        # Normalize batch
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings_normalized = batch_embeddings / (norms + 1e-9)

        # Append to storage
        all_embeddings.append(batch_embeddings_normalized)

        # Store metadata (without content) for successful chunks only
        for chunk in successful_chunks:
            metadata = {k: v for k, v in chunk.items() if k != "content"}
            metadata["faiss_id"] = faiss_id_counter
            faiss_id_counter += 1
            all_chunk_metadata.append(metadata)

        total_chunks += len(successful_chunks)
        current_batch_chunks = []

    # Process files and extract chunks
    for file_path in progress_bar(files, desc="Streaming embeddings", unit="files"):
        try:
            rel_path = file_path.relative_to(repo)
            chunks = _extract_code_chunks(file_path, chunk_by, base_dir=repo)
            for chunk in chunks:
                chunk["file"] = str(rel_path)
                chunk["id"] = chunk["id"].replace(str(file_path), str(rel_path))
                # Add content hash for incremental updates
                chunk["content_hash"] = _compute_content_hash(chunk.get("content", ""))
                current_batch_chunks.append(chunk)

                # Process batch when full
                if len(current_batch_chunks) >= batch_size:
                    process_batch()

        except Exception:
            continue

    # Process final batch
    process_batch()

    logger.info("  Streaming complete: %d chunks from %d files", total_chunks, len(files))

    if not all_embeddings:
        return {
            "embeddings": [],
            "chunks": [],
            "metadata": {
                "files_processed": len(files),
                "chunks_extracted": 0,
                "model": provider.model_name,
                "streaming": True,
            },
        }

    # Concatenate all batches
    embeddings_normalized = np.vstack(all_embeddings)
    logger.info("  Final embeddings shape: %s", embeddings_normalized.shape)

    # Save to SQLite (metadata + embeddings) and FAISS (vectors if available)
    repo_path = str(repo)
    _save_embeddings_to_sqlite(
        workspace_path, repo_path, all_chunk_metadata, provider.model_name,
        embeddings=embeddings_normalized
    )

    # Build FAISS index if available (optional optimization)
    if _HAS_FAISS:
        faiss_index = _build_faiss_index(embeddings_normalized)
        _save_faiss_index(faiss_index, cache_path)
        # Mark FAISS as fresh
        with transaction(workspace_path, repo_path) as conn:
            mark_faiss_fresh(conn, len(all_chunk_metadata))

    # Also save embeddings to NPZ for fallback when FAISS is not available
    cache_base = get_cache_dir(workspace_path, repo_path)
    npz_cache_path = cache_base / "embeddings.npz"
    np.savez_compressed(
        npz_cache_path,
        embeddings=embeddings_normalized,
        chunks=json.dumps(all_chunk_metadata),
        model=provider.model_name,
        normalized=True,
    )

    # Register in workspace registry if workspace-scoped
    if workspace_path is not None:
        try:
            from nodestradamus.utils.registry import register_analysis

            db_path = get_db_path(workspace_path, repo_path)
            cache_size = db_path.stat().st_size if db_path.exists() else None
            register_analysis(workspace_path, repo_path, cache_size=cache_size)
        except ImportError:
            pass

    return {
        # For streaming, we return summary only (not full embeddings to avoid memory spike)
        "embeddings": [],  # Empty to avoid memory duplication
        "chunks": all_chunk_metadata[:100],  # First 100 for preview
        "metadata": {
            "files_processed": len(files),
            "chunks_extracted": total_chunks,
            "model": provider.model_name,
            "cache_path": str(cache_path),
            "embedding_dim": embeddings_normalized.shape[1] if total_chunks > 0 else 0,
            "streaming": True,
            "storage": "sqlite+faiss",
            "message": f"Processed {total_chunks} chunks. Full embeddings cached at {cache_path}",
        },
    }


def load_cached_embeddings(
    repo_path: str,
    workspace_path: str | None = None,
    package: str | None = None,
    scope: str | None = None,
) -> dict[str, Any] | None:
    """Load cached embeddings if available and valid.

    Tries SQLite+FAISS first (new format), falls back to NPZ (legacy).

    Args:
        repo_path: Absolute path to the repository.
        workspace_path: Workspace path for isolated caching. When provided,
            looks for cache in <workspace>/.nodestradamus/cache/<repo_hash>/
            instead of <repo>/.nodestradamus/.
        package: For monorepos: load cache for this package path.
        scope: Optional path prefix to filter chunks (for lazy loading).

    Returns:
        Dict with embeddings, chunks, model, and normalized flag.
        Returns None if cache doesn't exist or is invalid.
    """
    # Try SQLite+FAISS first (new format)
    faiss_cache_path = _get_faiss_cache_path(workspace_path, repo_path)
    db_path = get_db_path(workspace_path, repo_path)

    if db_path.exists():
        try:
            # Load chunk metadata from SQLite
            chunks, model_version = _load_embeddings_from_sqlite(
                workspace_path, repo_path, scope=scope or package
            )

            if chunks and model_version:
                # Validate model matches expected provider
                expected_model = get_expected_model_name()
                if model_version != expected_model:
                    logger.warning(
                        "  Cached embeddings model mismatch: %s != %s, ignoring cache",
                        model_version,
                        expected_model,
                    )
                    return None

                # Check if FAISS is stale and rebuild from SQLite if needed
                conn = get_connection(workspace_path, repo_path)
                if is_faiss_stale(conn) or not faiss_cache_path.exists():
                    logger.info("  FAISS index is stale, rebuilding from SQLite embeddings")
                    faiss_index = rebuild_faiss_from_sqlite(workspace_path, repo_path)
                    if faiss_index is None:
                        # No embeddings in SQLite, fall back to NPZ
                        logger.warning("  No embeddings in SQLite, trying NPZ fallback")
                else:
                    # Load existing FAISS index
                    faiss_index = _load_faiss_index(faiss_cache_path)

                if faiss_index is None:
                    return None

                # Reconstruct embeddings from FAISS (for compatibility)
                # Note: This is less efficient but maintains API compatibility
                faiss_ids = [chunk["faiss_id"] for chunk in chunks]
                if faiss_ids:
                    embeddings = np.vstack([
                        faiss_index.reconstruct(fid) for fid in faiss_ids
                    ])
                else:
                    embeddings = np.array([])

                return {
                    "embeddings": embeddings,
                    "chunks": chunks,
                    "model": model_version,
                    "normalized": True,  # FAISS indexes use normalized vectors
                    "storage": "sqlite+faiss",
                    "faiss_index": faiss_index,
                }
        except Exception as e:
            logger.warning("  Failed to load SQLite+FAISS cache: %s", e)

    # Fall back to NPZ (legacy format)
    cache_base = get_cache_dir(workspace_path, repo_path)
    cache_filename = _get_cache_filename(package)
    cache_path = cache_base / cache_filename
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_model = str(data["model"])

        # Validate model matches expected provider
        expected_model = get_expected_model_name()
        if cached_model != expected_model:
            logger.warning(
                "  Cached embeddings model mismatch: %s != %s, ignoring cache",
                cached_model,
                expected_model,
            )
            return None

        # Check if embeddings are pre-normalized
        normalized = bool(data.get("normalized", False))

        chunks = json.loads(str(data["chunks"]))

        # Apply scope filter if requested
        if scope:
            chunks = [c for c in chunks if c["file"].startswith(scope)]

        return {
            "embeddings": data["embeddings"],
            "chunks": chunks,
            "model": cached_model,
            "normalized": normalized,
            "storage": "npz",
        }
    except Exception:
        return None


def find_similar_code(
    repo_path: str,
    query: str | None = None,
    file_path: str | None = None,
    symbol: str | None = None,
    top_k: int = 10,
    threshold: float = 0.5,
    workspace_path: str | None = None,
    filter_ids: set[str] | None = None,
    exclude: list[str] | None = None,
    package: str | None = None,
) -> list[dict[str, Any]]:
    """Find code similar to a query, file, or symbol.

    Args:
        repo_path: Absolute path to the repository.
        query: Natural language query or code snippet.
        file_path: Find code similar to this file.
        symbol: Find code similar to this function/class name.
        top_k: Number of results to return.
        threshold: Minimum similarity score (0-1).
        workspace_path: Workspace path for isolated caching.
        filter_ids: If provided, only search within chunks matching these IDs.
            This enables constrained search (e.g., within blast radius) for
            efficient fusion of graph and embedding signals.
        exclude: Directories/patterns to exclude from analysis.
        package: For monorepos: limit search to this package path.

    Returns:
        List of similar code chunks with similarity scores.
    """
    # Load or compute embeddings (package-scoped if provided)
    cached = load_cached_embeddings(repo_path, workspace_path=workspace_path, package=package)
    if cached is None:
        result = compute_embeddings(
            repo_path, workspace_path=workspace_path, exclude=exclude, package=package
        )
        embeddings = np.array(result["embeddings"])
        chunks = result["chunks"]
        embeddings_normalized = True  # compute_embeddings now pre-normalizes
    else:
        embeddings = cached["embeddings"]
        chunks = cached["chunks"]
        embeddings_normalized = cached.get("normalized", False)

    # Filter chunks by package prefix if package is set (for cross-package queries)
    if package:
        package_prefix = package.rstrip("/") + "/"
        package_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk["file"].startswith(package_prefix) or chunk["file"].startswith(package)
        ]
        if package_indices:
            embeddings = embeddings[package_indices]
            chunks = [chunks[i] for i in package_indices]

    if len(embeddings) == 0:
        return []

    # Filter to specific IDs if requested (enables constrained search within blast radius)
    if filter_ids is not None:
        filtered_indices = [i for i, chunk in enumerate(chunks) if chunk["id"] in filter_ids]
        if not filtered_indices:
            return []
        embeddings = embeddings[filtered_indices]
        chunks = [chunks[i] for i in filtered_indices]

    provider = get_embedding_provider()

    # Get query embedding
    if query:
        result = provider.encode([query])
        if len(result.embeddings) == 0:
            logger.warning("  Failed to embed query")
            return []
        query_embedding = result.embeddings[0]
    elif file_path:
        # Find the chunk for this file
        target_idx = None
        for i, chunk in enumerate(chunks):
            if chunk["file"] == file_path or file_path in chunk["file"]:
                target_idx = i
                break
        if target_idx is None:
            # Read the file and embed it
            full_path = Path(repo_path) / file_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8", errors="ignore")[:8000]
                result = provider.encode([content])
                if len(result.embeddings) == 0:
                    logger.warning("  Failed to embed file content")
                    return []
                query_embedding = result.embeddings[0]
            else:
                return []
        else:
            query_embedding = embeddings[target_idx]
    elif symbol:
        # Find the chunk for this symbol
        target_idx = None
        for i, chunk in enumerate(chunks):
            if chunk.get("name") == symbol or symbol in chunk.get("id", ""):
                target_idx = i
                break
        if target_idx is None:
            # Use symbol as query
            result = provider.encode([symbol])
            if len(result.embeddings) == 0:
                logger.warning("  Failed to embed symbol")
                return []
            query_embedding = result.embeddings[0]
        else:
            query_embedding = embeddings[target_idx]
    else:
        raise ValueError("Must provide query, file_path, or symbol")

    # Normalize query embedding for search
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

    # Use FAISS for large repos when available, otherwise brute-force
    use_faiss = (
        _HAS_FAISS
        and len(embeddings) >= _FAISS_THRESHOLD
        and embeddings_normalized
        and filter_ids is None  # FAISS not efficient for small filtered sets
    )

    if use_faiss:
        # Try to load cached FAISS index, or build one
        cache_base = get_cache_dir(workspace_path, repo_path)
        faiss_cache_path = cache_base / "embeddings.faiss"

        faiss_index = _load_faiss_index(faiss_cache_path)
        if faiss_index is None:
            faiss_index = _build_faiss_index(embeddings)
            _save_faiss_index(faiss_index, faiss_cache_path)

        # FAISS search (returns more than top_k to filter by threshold)
        similarities, top_indices = _faiss_search(faiss_index, query_norm, min(top_k * 2, len(chunks)))

        # Filter valid indices (FAISS may return -1 for empty slots)
        valid_mask = top_indices >= 0
        similarities = similarities[valid_mask]
        top_indices = top_indices[valid_mask]
    else:
        # Brute-force cosine similarity
        similarities = _cosine_similarity(query_embedding, embeddings, embeddings_normalized)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        similarities = similarities[top_indices]

    results = []
    for i, idx in enumerate(top_indices):
        score = float(similarities[i])

        chunk = chunks[idx]

        # Apply penalty for trivial chunks (E3)
        is_trivial = chunk.get("trivial", False)
        if is_trivial:
            score *= _TRIVIAL_PENALTY

        if score < threshold:
            continue
        if len(results) >= top_k:
            break

        result = {
            "id": chunk["id"],
            "type": chunk.get("type", "unknown"),
            "file": chunk["file"],
            "name": chunk.get("name", ""),
            "line_start": chunk.get("line_start"),
            "line_end": chunk.get("line_end"),
            "similarity": round(score, 4),
        }

        # Include snippet if available, with fallback to reading from disk (H1)
        snippet = chunk.get("snippet")
        if not snippet and chunk.get("file"):
            # Fallback: read from disk for old caches without snippet
            snippet = _read_snippet_from_file(
                Path(repo_path),
                chunk["file"],
                chunk.get("line_start"),
                chunk.get("line_end"),
            )
        if snippet:
            result["snippet"] = snippet

        results.append(result)

    return results


def _cosine_similarity(
    query: np.ndarray,
    embeddings: np.ndarray,
    embeddings_normalized: bool = False,
) -> np.ndarray:
    """Compute cosine similarity between query and all embeddings.

    Args:
        query: Query embedding vector.
        embeddings: Matrix of embeddings.
        embeddings_normalized: If True, embeddings are already L2-normalized,
            skipping normalization step for better performance.

    Returns:
        Array of similarity scores clamped to [0, 1].
    """
    # Always normalize query (it's a single vector, cheap operation)
    query_norm = query / (np.linalg.norm(query) + 1e-9)

    # Skip embeddings normalization if already pre-normalized
    if embeddings_normalized:
        similarities = np.dot(embeddings, query_norm)
    else:
        # Normalize embeddings (expensive for large matrices)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(embeddings_norm, query_norm)

    # Clamp to [0, 1] to handle floating point precision errors
    # (dot product of normalized vectors can exceed 1.0 slightly)
    return np.clip(similarities, 0.0, 1.0)


def semantic_search(
    repo_path: str,
    query: str,
    top_k: int = 10,
    threshold: float = 0.3,
    workspace_path: str | None = None,
    exclude: list[str] | None = None,
    package: str | None = None,
) -> list[dict[str, Any]]:
    """Search code using natural language.

    Args:
        repo_path: Absolute path to the repository.
        query: Natural language search query.
        top_k: Number of results to return.
        threshold: Minimum similarity score (0-1).
        workspace_path: Workspace path for isolated caching.
        exclude: Directories/patterns to exclude from analysis.
        package: For monorepos: limit search to this package path.

    Returns:
        List of matching code chunks with similarity scores.
    """
    return find_similar_code(
        repo_path=repo_path,
        query=query,
        top_k=top_k,
        threshold=threshold,
        workspace_path=workspace_path,
        exclude=exclude,
        package=package,
    )


def find_similar_code_hybrid(
    repo_path: str,
    query: str | None = None,
    file_path: str | None = None,
    top_k: int = 10,
    prefilter_k: int = 1000,
    threshold: float = 0.5,
    workspace_path: str | None = None,
    exclude: list[str] | None = None,
    package: str | None = None,
) -> list[dict[str, Any]]:
    """Find similar code using fingerprint pre-filter + embeddings.

    For large codebases (50K+ files), O(n) embedding comparisons are slow.
    This hybrid approach uses structural fingerprinting to pre-filter
    candidates (O(1) per hash lookup), then compares embeddings only
    among the filtered candidates (O(prefilter_k) vs O(n)).

    Expected speedup: 100-200x for large repos.

    Args:
        repo_path: Absolute path to the repository.
        query: Natural language query or code snippet.
        file_path: Find code similar to this file.
        top_k: Number of results to return.
        prefilter_k: Number of fingerprint candidates to pre-filter.
        threshold: Minimum similarity score (0-1).
        workspace_path: Workspace path for isolated caching.
        exclude: Directories/patterns to exclude from analysis.
        package: For monorepos: limit search to this package path.

    Returns:
        List of similar code chunks with similarity scores.
    """
    # Import fingerprints module (lazy import to avoid circular dependency)
    try:
        from nodestradamus.analyzers.fingerprints import find_similar as fp_find_similar
    except ImportError:
        logger.warning("  Fingerprints module not available, falling back to full search")
        return find_similar_code(
            repo_path=repo_path,
            query=query,
            file_path=file_path,
            top_k=top_k,
            threshold=threshold,
            workspace_path=workspace_path,
            exclude=exclude,
            package=package,
        )

    # Step 1: Get fingerprint candidates (fast structural pre-filter)
    # Only works when we have a file_path to fingerprint
    if file_path:
        try:
            fp_result = fp_find_similar(
                repo_path=repo_path,
                file_path=file_path,
                top_k=prefilter_k,
            )
            candidate_ids = {m["node_id"] for m in fp_result.get("matches", [])}
            logger.info(
                "  Hybrid search: fingerprint pre-filter found %d candidates",
                len(candidate_ids),
            )
        except Exception as e:
            logger.warning("  Fingerprint pre-filter failed: %s, falling back to full search", e)
            candidate_ids = None
    else:
        # For pure query-based search, we can't use fingerprint pre-filter
        # (fingerprints require structural AST, not natural language)
        candidate_ids = None

    # Step 2: Search embeddings (constrained to candidates if available)
    if candidate_ids and len(candidate_ids) >= top_k:
        # Use filter_ids to constrain embedding search
        results = find_similar_code(
            repo_path=repo_path,
            query=query,
            file_path=file_path,
            top_k=top_k,
            threshold=threshold,
            workspace_path=workspace_path,
            filter_ids=candidate_ids,
            exclude=exclude,
            package=package,
        )
    else:
        # Fallback to full search if not enough fingerprint candidates
        # or if query-only search
        results = find_similar_code(
            repo_path=repo_path,
            query=query,
            file_path=file_path,
            top_k=top_k,
            threshold=threshold,
            workspace_path=workspace_path,
            exclude=exclude,
            package=package,
        )

    return results


def detect_duplicates(
    repo_path: str,
    threshold: float = 0.9,
    max_pairs: int = 50,
    workspace_path: str | None = None,
    exclude: list[str] | None = None,
    package: str | None = None,
) -> list[dict[str, Any]]:
    """Find near-duplicate code blocks.

    Args:
        repo_path: Absolute path to the repository.
        threshold: Minimum similarity to consider as duplicate (0-1).
        max_pairs: Maximum number of duplicate pairs to return.
        workspace_path: Workspace path for isolated caching.
        exclude: Directories/patterns to exclude from analysis.
        package: For monorepos: limit detection to this package path.

    Returns:
        List of duplicate pairs with similarity scores.
    """
    # Load or compute embeddings (package-scoped if provided)
    cached = load_cached_embeddings(repo_path, workspace_path=workspace_path, package=package)
    if cached is None:
        result = compute_embeddings(
            repo_path, workspace_path=workspace_path, exclude=exclude, package=package
        )
        embeddings = np.array(result["embeddings"])
        chunks = result["chunks"]
        embeddings_normalized = True  # compute_embeddings now pre-normalizes
    else:
        embeddings = cached["embeddings"]
        chunks = cached["chunks"]
        embeddings_normalized = cached.get("normalized", False)

    # Filter chunks by package prefix if package is set
    if package:
        package_prefix = package.rstrip("/") + "/"
        package_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk["file"].startswith(package_prefix) or chunk["file"].startswith(package)
        ]
        if package_indices:
            embeddings = embeddings[package_indices]
            chunks = [chunks[i] for i in package_indices]

    if len(embeddings) < 2:
        return []

    # Normalize embeddings only if not pre-normalized
    if embeddings_normalized:
        embeddings_norm = embeddings
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-9)

    # Compute pairwise similarities (upper triangle only)
    duplicates = []
    n = len(embeddings)

    for i in progress_bar(range(n), desc="Finding duplicates", unit="chunks", total=n):
        # Compute similarities with all later chunks
        sims = np.dot(embeddings_norm[i + 1 :], embeddings_norm[i])
        high_sim_indices = np.where(sims >= threshold)[0]

        for j_offset in high_sim_indices:
            j = i + 1 + j_offset
            # Skip if same file (likely same or adjacent function)
            if chunks[i]["file"] == chunks[j]["file"]:
                continue

            # Get snippets for both chunks (H3)
            snippet_a = chunks[i].get("snippet")
            if not snippet_a and chunks[i].get("file"):
                snippet_a = _read_snippet_from_file(
                    Path(repo_path),
                    chunks[i]["file"],
                    chunks[i].get("line_start"),
                    chunks[i].get("line_end"),
                )

            snippet_b = chunks[j].get("snippet")
            if not snippet_b and chunks[j].get("file"):
                snippet_b = _read_snippet_from_file(
                    Path(repo_path),
                    chunks[j]["file"],
                    chunks[j].get("line_start"),
                    chunks[j].get("line_end"),
                )

            pair: dict[str, Any] = {
                "chunk_a": {
                    "id": chunks[i]["id"],
                    "file": chunks[i]["file"],
                    "name": chunks[i].get("name", ""),
                    "line_start": chunks[i].get("line_start"),
                },
                "chunk_b": {
                    "id": chunks[j]["id"],
                    "file": chunks[j]["file"],
                    "name": chunks[j].get("name", ""),
                    "line_start": chunks[j].get("line_start"),
                },
                "similarity": round(min(float(sims[j_offset]), 1.0), 4),
            }

            # Add preview fields (H3)
            if snippet_a:
                pair["preview_a"] = snippet_a
            if snippet_b:
                pair["preview_b"] = snippet_b
            # Add a single combined preview (first available)
            if snippet_a or snippet_b:
                pair["preview"] = snippet_a or snippet_b

            duplicates.append(pair)

            if len(duplicates) >= max_pairs:
                break

        if len(duplicates) >= max_pairs:
            break

    # Sort by similarity
    duplicates.sort(key=lambda x: x["similarity"], reverse=True)

    return duplicates[:max_pairs]
