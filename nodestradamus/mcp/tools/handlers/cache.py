"""Cache management handlers.

Handler for manage_cache tool with all modes:
- info: Show cache details for a specific repo
- clear: Delete cache for a specific repo
- list: Show all caches in the workspace
"""

import json
import shutil
from pathlib import Path
from typing import Any

from nodestradamus.utils.cache import get_cache_dir, get_repo_hash


async def handle_manage_cache(arguments: dict[str, Any]) -> str:
    """Handle manage_cache tool call.

    Manages Nodestradamus embedding caches with three modes:
    - info: Show cache details for a specific repo
    - clear: Delete cache for a specific repo
    - list: Show all caches in the workspace

    Args:
        arguments: Tool arguments with mode, repo_path, workspace_path.

    Returns:
        JSON string with cache information or operation results.

    Raises:
        ValueError: If required parameters are missing or mode is unknown.
    """
    mode = arguments.get("mode")
    repo_path = arguments.get("repo_path")
    workspace_path = arguments.get("workspace_path")

    if not mode:
        raise ValueError("mode is required")

    if mode == "info":
        if not repo_path:
            raise ValueError("repo_path is required for 'info' mode")
        return await _run_info(repo_path, workspace_path)

    elif mode == "clear":
        if not repo_path:
            raise ValueError("repo_path is required for 'clear' mode")
        return await _run_clear(repo_path, workspace_path)

    elif mode == "list":
        if not workspace_path:
            raise ValueError("workspace_path is required for 'list' mode")
        return await _run_list(workspace_path)

    else:
        raise ValueError(f"Unknown mode: {mode}")


async def _run_info(repo_path: str, workspace_path: str | None) -> str:
    """Get cache information for a repository.

    Args:
        repo_path: Path to the repository.
        workspace_path: Optional workspace path for scoped cache.

    Returns:
        JSON string with cache locations and sizes.
    """
    repo = Path(repo_path).resolve()

    # Check both workspace-scoped and repo-local cache locations
    cache_locations: list[dict[str, Any]] = []

    # Workspace-scoped cache
    if workspace_path:
        ws_cache_dir = get_cache_dir(workspace_path, repo_path)
        ws_embeddings = ws_cache_dir / "embeddings.npz"
        if ws_embeddings.exists():
            stat = ws_embeddings.stat()
            cache_locations.append({
                "location": "workspace",
                "path": str(ws_embeddings),
                "size_bytes": stat.st_size,
                "size_human": _format_size(stat.st_size),
                "modified": stat.st_mtime,
                "repo_hash": get_repo_hash(repo_path),
            })

    # Repo-local cache (fallback location)
    local_cache_dir = repo / ".nodestradamus"
    local_embeddings = local_cache_dir / "embeddings.npz"
    if local_embeddings.exists():
        stat = local_embeddings.stat()
        cache_locations.append({
            "location": "repo-local",
            "path": str(local_embeddings),
            "size_bytes": stat.st_size,
            "size_human": _format_size(stat.st_size),
            "modified": stat.st_mtime,
        })

    if not cache_locations:
        return json.dumps({
            "status": "no_cache",
            "repo_path": str(repo),
            "message": "No embedding cache found for this repository",
            "checked_locations": {
                "workspace": str(get_cache_dir(workspace_path, repo_path)) if workspace_path else None,
                "repo_local": str(local_embeddings),
            },
        }, indent=2)

    return json.dumps({
        "status": "found",
        "repo_path": str(repo),
        "caches": cache_locations,
    }, indent=2)


async def _run_clear(repo_path: str, workspace_path: str | None) -> str:
    """Clear cache for a repository.

    Args:
        repo_path: Path to the repository.
        workspace_path: Optional workspace path for scoped cache.

    Returns:
        JSON string with cleared cache information.
    """
    repo = Path(repo_path).resolve()
    cleared: list[dict[str, Any]] = []
    errors: list[str] = []

    # Clear workspace-scoped cache
    if workspace_path:
        ws_cache_dir = get_cache_dir(workspace_path, repo_path)
        if ws_cache_dir.exists():
            try:
                size = _get_dir_size(ws_cache_dir)
                shutil.rmtree(ws_cache_dir)
                cleared.append({
                    "location": "workspace",
                    "path": str(ws_cache_dir),
                    "size_cleared": _format_size(size),
                })
                # Unregister from workspace registry
                try:
                    from nodestradamus.utils.registry import unregister_analysis
                    unregister_analysis(workspace_path, repo_path)
                except ImportError:
                    pass
            except Exception as e:
                errors.append(f"Failed to clear workspace cache: {e}")

    # Clear repo-local cache
    local_cache_dir = repo / ".nodestradamus"
    if local_cache_dir.exists():
        try:
            size = _get_dir_size(local_cache_dir)
            shutil.rmtree(local_cache_dir)
            cleared.append({
                "location": "repo-local",
                "path": str(local_cache_dir),
                "size_cleared": _format_size(size),
            })
        except Exception as e:
            errors.append(f"Failed to clear repo-local cache: {e}")

    if not cleared and not errors:
        return json.dumps({
            "status": "no_cache",
            "repo_path": str(repo),
            "message": "No cache found to clear",
        }, indent=2)

    result: dict[str, Any] = {
        "status": "cleared" if cleared else "error",
        "repo_path": str(repo),
        "cleared": cleared,
    }
    if errors:
        result["errors"] = errors

    return json.dumps(result, indent=2)


async def _run_list(workspace_path: str) -> str:
    """List all caches in a workspace.

    Args:
        workspace_path: Path to the workspace.

    Returns:
        JSON string with all caches in the workspace.
    """
    workspace = Path(workspace_path).resolve()
    cache_root = workspace / ".nodestradamus" / "cache"

    # Try to load registry for repo path lookups
    registry_repos: dict[str, str] = {}
    try:
        from nodestradamus.utils.registry import load_registry
        registry = load_registry(workspace_path)
        for repo_hash, entry in registry.get("repos", {}).items():
            registry_repos[repo_hash] = entry.get("path", "unknown")
    except ImportError:
        pass

    if not cache_root.exists():
        return json.dumps({
            "status": "empty",
            "workspace": str(workspace),
            "message": "No caches found in this workspace",
            "cache_directory": str(cache_root),
        }, indent=2)

    caches: list[dict[str, Any]] = []

    # Each subdirectory in cache/ is a repo hash
    for cache_dir in cache_root.iterdir():
        if cache_dir.is_dir():
            embeddings_file = cache_dir / "embeddings.npz"
            if embeddings_file.exists():
                stat = embeddings_file.stat()
                repo_hash = cache_dir.name
                cache_entry: dict[str, Any] = {
                    "repo_hash": repo_hash,
                    "path": str(cache_dir),
                    "embeddings_size": _format_size(stat.st_size),
                    "embeddings_size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
                # Add repo path from registry if available
                if repo_hash in registry_repos:
                    cache_entry["repo_path"] = registry_repos[repo_hash]
                caches.append(cache_entry)

    # Sort by modification time (most recent first)
    caches.sort(key=lambda x: x["modified"], reverse=True)

    total_size = sum(c["embeddings_size_bytes"] for c in caches)

    result: dict[str, Any] = {
        "status": "found",
        "workspace": str(workspace),
        "cache_count": len(caches),
        "total_size": _format_size(total_size),
        "caches": caches,
    }

    # Add hint if some caches don't have registry entries
    missing_registry = [c for c in caches if "repo_path" not in c]
    if missing_registry:
        result["hint"] = (
            f"{len(missing_registry)} cache(s) without registry entries. "
            "Use 'info' mode with a specific repo_path to identify which cache belongs to which repo."
        )

    return json.dumps(result, indent=2)


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory.

    Args:
        path: Directory path.

    Returns:
        Total size in bytes.
    """
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total
