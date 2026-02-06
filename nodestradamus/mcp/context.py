"""Workspace context utilities for Nodestradamus MCP server.

Provides functions to get workspace roots from MCP clients and compute
cache directories for workspace-isolated caching.

Note: Cache utility functions are defined in nodestradamus.utils.cache to avoid
circular imports. They are re-exported here for convenience.
"""

from pathlib import Path

from nodestradamus.logging import logger
from nodestradamus.utils.cache import get_cache_dir, get_repo_hash, parse_file_uri

# Re-export for backward compatibility
__all__ = ["get_cache_dir", "get_repo_hash", "parse_file_uri", "get_workspace_from_session"]


async def get_workspace_from_session(session: object) -> Path | None:
    """Get the workspace root from an MCP session.

    Requests roots from the client and returns the first root path.
    If the client doesn't support roots or returns none, returns None.

    Args:
        session: An MCP ServerSession object.

    Returns:
        The workspace root path, or None if not available.
    """
    # Check if session has list_roots method (ServerSession does)
    if not hasattr(session, "list_roots"):
        logger.debug("Session does not support list_roots")
        return None

    try:
        result = await session.list_roots()

        if not result.roots:
            logger.debug("Client returned no roots")
            return None

        # Use the first root as the workspace
        first_root = result.roots[0]

        # Parse the URI (should be file://)
        workspace = parse_file_uri(str(first_root.uri))
        if workspace:
            logger.debug("Workspace root: %s", workspace)
            return workspace

        logger.warning("First root is not a file URI: %s", first_root.uri)
        return None

    except Exception as e:
        # Client may not support roots capability
        logger.debug("Failed to get roots from client: %s", e)
        return None
