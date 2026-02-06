"""Workspace registry for Nodestradamus.

Provides a lightweight JSON-based registry to track which repositories
have been analyzed from a workspace. This helps users understand which
cache directories correspond to which repositories.

The registry is stored at `<workspace>/.nodestradamus/registry.json`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from nodestradamus.logging import logger
from nodestradamus.utils.cache import get_repo_hash


def get_registry_path(workspace_path: Path | str) -> Path:
    """Get the path to the workspace registry file.

    Args:
        workspace_path: The workspace root path.

    Returns:
        Path to registry.json.
    """
    workspace = Path(workspace_path).resolve()
    return workspace / ".nodestradamus" / "registry.json"


def load_registry(workspace_path: Path | str) -> dict[str, Any]:
    """Load the workspace registry.

    Args:
        workspace_path: The workspace root path.

    Returns:
        Registry data dict. Empty dict if registry doesn't exist.
    """
    registry_path = get_registry_path(workspace_path)

    if not registry_path.exists():
        return {"repos": {}, "version": 1}

    try:
        with open(registry_path) as f:
            data = json.load(f)
        # Ensure required keys exist
        if "repos" not in data:
            data["repos"] = {}
        if "version" not in data:
            data["version"] = 1
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load registry: %s", e)
        return {"repos": {}, "version": 1}


def save_registry(workspace_path: Path | str, registry: dict[str, Any]) -> bool:
    """Save the workspace registry.

    Args:
        workspace_path: The workspace root path.
        registry: Registry data to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    registry_path = get_registry_path(workspace_path)

    try:
        # Ensure parent directory exists
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)
        return True
    except OSError as e:
        logger.warning("Failed to save registry: %s", e)
        return False


def register_analysis(
    workspace_path: Path | str,
    repo_path: Path | str,
    cache_size: int | None = None,
) -> None:
    """Register a repository analysis in the workspace registry.

    Updates the registry with the analyzed repo's path, hash, and timestamp.
    Called after computing embeddings.

    Args:
        workspace_path: The workspace root path.
        repo_path: The repository that was analyzed.
        cache_size: Optional size of the cache in bytes.
    """
    repo = Path(repo_path).resolve()
    repo_hash = get_repo_hash(repo)

    registry = load_registry(workspace_path)

    registry["repos"][repo_hash] = {
        "path": str(repo),
        "hash": repo_hash,
        "last_analyzed": datetime.now().isoformat(),
        "cache_size": cache_size,
    }

    save_registry(workspace_path, registry)
    logger.debug("Registered analysis: %s -> %s", repo, repo_hash)


def unregister_analysis(workspace_path: Path | str, repo_path: Path | str) -> bool:
    """Remove a repository from the workspace registry.

    Called when clearing cache for a repository.

    Args:
        workspace_path: The workspace root path.
        repo_path: The repository to unregister.

    Returns:
        True if the repo was found and removed, False otherwise.
    """
    repo = Path(repo_path).resolve()
    repo_hash = get_repo_hash(repo)

    registry = load_registry(workspace_path)

    if repo_hash in registry["repos"]:
        del registry["repos"][repo_hash]
        save_registry(workspace_path, registry)
        logger.debug("Unregistered analysis: %s", repo)
        return True

    return False


def get_repo_for_hash(workspace_path: Path | str, repo_hash: str) -> str | None:
    """Look up the repository path for a cache hash.

    Args:
        workspace_path: The workspace root path.
        repo_hash: The 16-character hash identifier.

    Returns:
        The repository path, or None if not found in registry.
    """
    registry = load_registry(workspace_path)

    entry = registry["repos"].get(repo_hash)
    if entry:
        return entry.get("path")

    return None


def list_registered_repos(workspace_path: Path | str) -> list[dict[str, Any]]:
    """List all registered repositories in the workspace.

    Args:
        workspace_path: The workspace root path.

    Returns:
        List of repo entries with path, hash, last_analyzed, and cache_size.
    """
    registry = load_registry(workspace_path)
    return list(registry["repos"].values())
