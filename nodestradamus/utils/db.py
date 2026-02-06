"""SQLite database utilities for Nodestradamus embeddings storage.

Provides connection management and schema initialization for the hybrid
SQLite + FAISS storage system. SQLite stores chunk metadata while FAISS
handles vector similarity search.

This module is intentionally kept dependency-free from the MCP server
to avoid circular imports.
"""

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from nodestradamus.utils.cache import get_cache_dir

# Schema version for migrations
SCHEMA_VERSION = 2

# SQL schema for chunks table
SCHEMA_SQL = """
-- Chunk metadata for embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    symbol_name TEXT,
    line_start INTEGER,
    line_end INTEGER,
    content_hash TEXT NOT NULL,
    snippet TEXT,
    language TEXT,
    faiss_id INTEGER,
    model_version TEXT NOT NULL,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id);

-- FAISS cache tracking
CREATE TABLE IF NOT EXISTS faiss_cache (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    is_stale INTEGER DEFAULT 1,
    last_rebuilt TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Initialize FAISS cache tracking
INSERT OR IGNORE INTO faiss_cache (id, is_stale) VALUES (1, 1);
"""


# Thread-local storage for connections
_local = threading.local()

# Global connection cache (path -> connection)
_connection_cache: dict[str, sqlite3.Connection] = {}
_cache_lock = threading.Lock()


def get_db_path(
    workspace_path: Path | str | None,
    repo_path: Path | str,
) -> Path:
    """Get the path to the SQLite database file.

    Args:
        workspace_path: The workspace root path (from MCP roots), or None.
        repo_path: The repository being analyzed.

    Returns:
        Path to the nodestradamus.db file.
    """
    cache_dir = get_cache_dir(workspace_path, repo_path)
    return cache_dir / "nodestradamus.db"


def _init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the database schema if needed.

    Args:
        conn: SQLite connection.
    """
    cursor = conn.cursor()

    # Check if schema exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_info'"
    )
    if cursor.fetchone() is None:
        # First time setup
        conn.executescript(SCHEMA_SQL)
        cursor.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("version", str(SCHEMA_VERSION)),
        )
        conn.commit()
        return

    # Check version for future migrations
    cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
    row = cursor.fetchone()
    current_version = int(row[0]) if row else 0

    if current_version < SCHEMA_VERSION:
        # Run migrations
        if current_version < 2:
            # Migration 1 -> 2: Add embedding column and faiss_cache table
            cursor.execute(
                "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name='embedding'"
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")

            # Create faiss_cache table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faiss_cache (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    is_stale INTEGER DEFAULT 1,
                    last_rebuilt TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0
                )
            """)
            cursor.execute(
                "INSERT OR IGNORE INTO faiss_cache (id, is_stale) VALUES (1, 1)"
            )

        cursor.execute(
            "UPDATE schema_info SET value = ? WHERE key = 'version'",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()


def get_connection(
    workspace_path: Path | str | None,
    repo_path: Path | str,
) -> sqlite3.Connection:
    """Get or create a SQLite connection for the given repo.

    Connections are cached per database path. Thread-safe.

    Args:
        workspace_path: The workspace root path (from MCP roots), or None.
        repo_path: The repository being analyzed.

    Returns:
        SQLite connection with row_factory set to sqlite3.Row.
    """
    db_path = get_db_path(workspace_path, repo_path)
    db_key = str(db_path)

    with _cache_lock:
        if db_key in _connection_cache:
            return _connection_cache[db_key]

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection with WAL mode for better concurrency
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

        _init_schema(conn)

        _connection_cache[db_key] = conn
        return conn


@contextmanager
def transaction(
    workspace_path: Path | str | None,
    repo_path: Path | str,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database transactions.

    Automatically commits on success, rolls back on exception.

    Args:
        workspace_path: The workspace root path (from MCP roots), or None.
        repo_path: The repository being analyzed.

    Yields:
        SQLite connection within a transaction.
    """
    conn = get_connection(workspace_path, repo_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_connection(
    workspace_path: Path | str | None,
    repo_path: Path | str,
) -> None:
    """Close and remove a cached connection.

    Args:
        workspace_path: The workspace root path (from MCP roots), or None.
        repo_path: The repository being analyzed.
    """
    db_path = get_db_path(workspace_path, repo_path)
    db_key = str(db_path)

    with _cache_lock:
        if db_key in _connection_cache:
            _connection_cache[db_key].close()
            del _connection_cache[db_key]


def close_all_connections() -> None:
    """Close all cached connections. Useful for cleanup in tests."""
    with _cache_lock:
        for conn in _connection_cache.values():
            conn.close()
        _connection_cache.clear()


# --- Chunk CRUD operations ---


def insert_chunk(
    conn: sqlite3.Connection,
    file_path: str,
    symbol_name: str | None,
    line_start: int,
    line_end: int,
    content_hash: str,
    snippet: str | None,
    language: str | None,
    faiss_id: int,
    model_version: str,
    embedding: bytes | None = None,
) -> int:
    """Insert a new chunk and return its ID.

    Args:
        conn: SQLite connection.
        file_path: Relative path to the file.
        symbol_name: Name of the symbol (function, class, etc.).
        line_start: Starting line number.
        line_end: Ending line number.
        content_hash: Hash of the chunk content for change detection.
        snippet: Code snippet preview.
        language: Programming language.
        faiss_id: Index in the FAISS vector store.
        model_version: Embedding model identifier.
        embedding: The embedding vector as bytes (numpy array.tobytes()).

    Returns:
        The inserted row ID.
    """
    cursor = conn.execute(
        """
        INSERT INTO chunks (
            file_path, symbol_name, line_start, line_end,
            content_hash, snippet, language, faiss_id, model_version, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_path,
            symbol_name,
            line_start,
            line_end,
            content_hash,
            snippet,
            language,
            faiss_id,
            model_version,
            embedding,
        ),
    )
    # Mark FAISS as stale since we added new chunks
    mark_faiss_stale(conn)
    return cursor.lastrowid  # type: ignore[return-value]


def bulk_insert_chunks(
    conn: sqlite3.Connection,
    chunks: list[dict],
) -> list[int]:
    """Bulk insert chunks for better performance.

    Args:
        conn: SQLite connection.
        chunks: List of chunk dictionaries with keys matching insert_chunk params.
            Can include 'embedding' key with bytes data.

    Returns:
        List of inserted row IDs.
    """
    cursor = conn.cursor()
    ids = []

    for chunk in chunks:
        cursor.execute(
            """
            INSERT INTO chunks (
                file_path, symbol_name, line_start, line_end,
                content_hash, snippet, language, faiss_id, model_version, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["file_path"],
                chunk.get("symbol_name"),
                chunk["line_start"],
                chunk["line_end"],
                chunk["content_hash"],
                chunk.get("snippet"),
                chunk.get("language"),
                chunk["faiss_id"],
                chunk["model_version"],
                chunk.get("embedding"),
            ),
        )
        ids.append(cursor.lastrowid)

    # Mark FAISS as stale since we added new chunks
    if chunks:
        mark_faiss_stale(conn)

    return ids  # type: ignore[return-value]


def get_chunks_by_file(
    conn: sqlite3.Connection,
    file_path: str,
) -> list[sqlite3.Row]:
    """Get all chunks for a specific file.

    Args:
        conn: SQLite connection.
        file_path: Relative path to the file.

    Returns:
        List of chunk rows.
    """
    cursor = conn.execute(
        "SELECT * FROM chunks WHERE file_path = ?",
        (file_path,),
    )
    return cursor.fetchall()


def get_chunks_by_scope(
    conn: sqlite3.Connection,
    scope: str,
) -> list[sqlite3.Row]:
    """Get all chunks matching a path prefix (scope).

    Args:
        conn: SQLite connection.
        scope: Path prefix to filter by (e.g., "src/auth/").

    Returns:
        List of chunk rows.
    """
    cursor = conn.execute(
        "SELECT * FROM chunks WHERE file_path LIKE ?",
        (f"{scope}%",),
    )
    return cursor.fetchall()


def get_chunk_by_faiss_id(
    conn: sqlite3.Connection,
    faiss_id: int,
) -> sqlite3.Row | None:
    """Get a chunk by its FAISS index ID.

    Args:
        conn: SQLite connection.
        faiss_id: The FAISS vector index.

    Returns:
        The chunk row, or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM chunks WHERE faiss_id = ?",
        (faiss_id,),
    )
    return cursor.fetchone()


def get_chunks_by_faiss_ids(
    conn: sqlite3.Connection,
    faiss_ids: list[int],
) -> list[sqlite3.Row]:
    """Get multiple chunks by their FAISS index IDs.

    Args:
        conn: SQLite connection.
        faiss_ids: List of FAISS vector indices.

    Returns:
        List of chunk rows (order not guaranteed).
    """
    if not faiss_ids:
        return []

    placeholders = ",".join("?" * len(faiss_ids))
    cursor = conn.execute(
        f"SELECT * FROM chunks WHERE faiss_id IN ({placeholders})",
        faiss_ids,
    )
    return cursor.fetchall()


def get_all_chunks(
    conn: sqlite3.Connection,
    model_version: str | None = None,
) -> list[sqlite3.Row]:
    """Get all chunks, optionally filtered by model version.

    Args:
        conn: SQLite connection.
        model_version: Optional model version filter.

    Returns:
        List of all chunk rows.
    """
    if model_version:
        cursor = conn.execute(
            "SELECT * FROM chunks WHERE model_version = ? ORDER BY faiss_id",
            (model_version,),
        )
    else:
        cursor = conn.execute("SELECT * FROM chunks ORDER BY faiss_id")
    return cursor.fetchall()


def get_chunk_hashes(
    conn: sqlite3.Connection,
) -> dict[str, str]:
    """Get a mapping of (file_path, line_start, line_end) -> content_hash.

    Used for incremental updates to detect changed chunks.

    Args:
        conn: SQLite connection.

    Returns:
        Dict mapping chunk key to content hash.
    """
    cursor = conn.execute(
        "SELECT file_path, line_start, line_end, content_hash FROM chunks"
    )
    return {
        f"{row['file_path']}:{row['line_start']}:{row['line_end']}": row["content_hash"]
        for row in cursor.fetchall()
    }


def get_chunk_hashes_with_faiss_ids(
    conn: sqlite3.Connection,
) -> dict[str, tuple[str, int]]:
    """Get a mapping of chunk_key -> (content_hash, faiss_id).

    Extended version of get_chunk_hashes that also returns the faiss_id,
    enabling reuse of embeddings for unchanged chunks.

    Args:
        conn: SQLite connection.

    Returns:
        Dict mapping chunk key to (content_hash, faiss_id) tuple.
    """
    cursor = conn.execute(
        "SELECT file_path, line_start, line_end, content_hash, faiss_id FROM chunks"
    )
    return {
        f"{row['file_path']}:{row['line_start']}:{row['line_end']}": (
            row["content_hash"],
            row["faiss_id"],
        )
        for row in cursor.fetchall()
    }


def delete_chunks_by_file(
    conn: sqlite3.Connection,
    file_path: str,
) -> int:
    """Delete all chunks for a specific file.

    Args:
        conn: SQLite connection.
        file_path: Relative path to the file.

    Returns:
        Number of rows deleted.
    """
    cursor = conn.execute(
        "DELETE FROM chunks WHERE file_path = ?",
        (file_path,),
    )
    return cursor.rowcount


def delete_all_chunks(conn: sqlite3.Connection) -> int:
    """Delete all chunks (used for full rebuild).

    Args:
        conn: SQLite connection.

    Returns:
        Number of rows deleted.
    """
    cursor = conn.execute("DELETE FROM chunks")
    return cursor.rowcount


def get_chunk_count(conn: sqlite3.Connection) -> int:
    """Get the total number of chunks.

    Args:
        conn: SQLite connection.

    Returns:
        Total chunk count.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
    return cursor.fetchone()[0]


def get_max_faiss_id(conn: sqlite3.Connection) -> int:
    """Get the maximum FAISS ID, or -1 if no chunks exist.

    Args:
        conn: SQLite connection.

    Returns:
        Maximum faiss_id, or -1 if table is empty.
    """
    cursor = conn.execute("SELECT MAX(faiss_id) FROM chunks")
    result = cursor.fetchone()[0]
    return result if result is not None else -1


def get_files_with_chunks(conn: sqlite3.Connection) -> set[str]:
    """Get the set of all file paths that have chunks.

    Args:
        conn: SQLite connection.

    Returns:
        Set of file paths.
    """
    cursor = conn.execute("SELECT DISTINCT file_path FROM chunks")
    return {row[0] for row in cursor.fetchall()}


# --- FAISS cache management ---


def mark_faiss_stale(conn: sqlite3.Connection) -> None:
    """Mark the FAISS cache as stale (needs rebuild).

    Called automatically when chunks are inserted/deleted.

    Args:
        conn: SQLite connection.
    """
    conn.execute("UPDATE faiss_cache SET is_stale = 1 WHERE id = 1")


def mark_faiss_fresh(conn: sqlite3.Connection, chunk_count: int) -> None:
    """Mark the FAISS cache as fresh after rebuild.

    Args:
        conn: SQLite connection.
        chunk_count: Number of chunks in the rebuilt index.
    """
    conn.execute(
        """
        UPDATE faiss_cache
        SET is_stale = 0, last_rebuilt = CURRENT_TIMESTAMP, chunk_count = ?
        WHERE id = 1
        """,
        (chunk_count,),
    )


def is_faiss_stale(conn: sqlite3.Connection) -> bool:
    """Check if the FAISS cache needs to be rebuilt.

    Args:
        conn: SQLite connection.

    Returns:
        True if FAISS needs rebuild, False if fresh.
    """
    cursor = conn.execute("SELECT is_stale FROM faiss_cache WHERE id = 1")
    row = cursor.fetchone()
    return row is None or row[0] == 1


def get_faiss_cache_info(conn: sqlite3.Connection) -> dict:
    """Get FAISS cache status information.

    Args:
        conn: SQLite connection.

    Returns:
        Dict with is_stale, last_rebuilt, chunk_count.
    """
    cursor = conn.execute(
        "SELECT is_stale, last_rebuilt, chunk_count FROM faiss_cache WHERE id = 1"
    )
    row = cursor.fetchone()
    if row is None:
        return {"is_stale": True, "last_rebuilt": None, "chunk_count": 0}
    return {
        "is_stale": bool(row[0]),
        "last_rebuilt": row[1],
        "chunk_count": row[2] or 0,
    }


# --- Embedding storage and retrieval ---


def get_all_embeddings(
    conn: sqlite3.Connection,
    model_version: str | None = None,
) -> list[tuple[int, bytes]]:
    """Get all embeddings from the database.

    Args:
        conn: SQLite connection.
        model_version: Optional model version filter.

    Returns:
        List of (faiss_id, embedding_bytes) tuples, ordered by faiss_id.
    """
    if model_version:
        cursor = conn.execute(
            """
            SELECT faiss_id, embedding FROM chunks
            WHERE model_version = ? AND embedding IS NOT NULL
            ORDER BY faiss_id
            """,
            (model_version,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT faiss_id, embedding FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY faiss_id
            """
        )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def get_embeddings_by_scope(
    conn: sqlite3.Connection,
    scope: str,
) -> list[tuple[int, bytes]]:
    """Get embeddings for files matching a path prefix.

    Args:
        conn: SQLite connection.
        scope: Path prefix to filter by.

    Returns:
        List of (faiss_id, embedding_bytes) tuples.
    """
    cursor = conn.execute(
        """
        SELECT faiss_id, embedding FROM chunks
        WHERE file_path LIKE ? AND embedding IS NOT NULL
        ORDER BY faiss_id
        """,
        (f"{scope}%",),
    )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def update_chunk_embedding(
    conn: sqlite3.Connection,
    chunk_id: int,
    embedding: bytes,
    faiss_id: int,
) -> None:
    """Update the embedding for an existing chunk.

    Args:
        conn: SQLite connection.
        chunk_id: The chunk's primary key ID.
        embedding: The embedding vector as bytes.
        faiss_id: The new FAISS index position.
    """
    conn.execute(
        """
        UPDATE chunks
        SET embedding = ?, faiss_id = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (embedding, faiss_id, chunk_id),
    )
    mark_faiss_stale(conn)


def delete_chunks_by_scope(
    conn: sqlite3.Connection,
    scope: str,
) -> int:
    """Delete all chunks matching a path prefix.

    Args:
        conn: SQLite connection.
        scope: Path prefix to delete.

    Returns:
        Number of rows deleted.
    """
    cursor = conn.execute(
        "DELETE FROM chunks WHERE file_path LIKE ?",
        (f"{scope}%",),
    )
    if cursor.rowcount > 0:
        mark_faiss_stale(conn)
    return cursor.rowcount


def get_chunks_without_embeddings(
    conn: sqlite3.Connection,
    scope: str | None = None,
) -> list[sqlite3.Row]:
    """Get chunks that don't have embeddings yet.

    Args:
        conn: SQLite connection.
        scope: Optional path prefix filter.

    Returns:
        List of chunk rows missing embeddings.
    """
    if scope:
        cursor = conn.execute(
            """
            SELECT * FROM chunks
            WHERE embedding IS NULL AND file_path LIKE ?
            """,
            (f"{scope}%",),
        )
    else:
        cursor = conn.execute("SELECT * FROM chunks WHERE embedding IS NULL")
    return cursor.fetchall()
