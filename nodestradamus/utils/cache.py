"""Cache directory utilities for Nodestradamus.

Provides functions to compute workspace-scoped cache directories and
repository hashes. These utilities are used by both the MCP server and
the analyzers for consistent cache management.

This module is intentionally kept dependency-free from the MCP server
to avoid circular imports.
"""

import hashlib
from pathlib import Path
from urllib.parse import urlparse


def get_cache_dir(workspace_path: Path | str | None, repo_path: Path | str) -> Path:
    """Compute the cache directory for a repository.

    When workspace_path is provided, returns a workspace-scoped cache directory.
    When workspace_path is None, falls back to cache in the analyzed repo itself.

    Args:
        workspace_path: The workspace root path (from MCP roots), or None.
        repo_path: The repository being analyzed.

    Returns:
        Path to the cache directory.

    Examples:
        >>> get_cache_dir("/home/user/workspace", "/home/user/project")
        PosixPath('/home/user/workspace/.nodestradamus/cache/a1b2c3d4')

        >>> get_cache_dir(None, "/home/user/project")
        PosixPath('/home/user/project/.nodestradamus')
    """
    repo = Path(repo_path).resolve()

    if workspace_path is None:
        # Fallback: cache in the analyzed repo itself
        return repo / ".nodestradamus"

    workspace = Path(workspace_path).resolve()

    # Hash the repo path to create a unique subdirectory
    repo_hash = hashlib.sha256(str(repo).encode()).hexdigest()[:16]

    return workspace / ".nodestradamus" / "cache" / repo_hash


def get_repo_hash(repo_path: Path | str) -> str:
    """Get the hash identifier for a repository path.

    Args:
        repo_path: The repository path.

    Returns:
        A 16-character hex hash of the absolute path.
    """
    repo = Path(repo_path).resolve()
    return hashlib.sha256(str(repo).encode()).hexdigest()[:16]


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


def parse_file_uri(uri: str) -> Path | None:
    """Parse a file:// URI and return the path.

    Args:
        uri: A file:// URI string.

    Returns:
        The path as a Path object, or None if not a file URI.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None

    # Handle file:///path/to/dir format
    return Path(parsed.path)
