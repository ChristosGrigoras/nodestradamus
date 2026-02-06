"""Snapshot utilities for Nodestradamus tool output.

Saves and loads diff-friendly summaries of tool runs under
workspace/.nodestradamus/snapshots/<repo_hash>/ so "changed since last run"
can be computed. Uses get_repo_hash from cache for directory naming.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nodestradamus.utils.cache import get_repo_hash


def get_snapshot_dir(workspace_path: Path | str, repo_path: Path | str) -> Path:
    """Directory for snapshots for a repo in a workspace.

    Args:
        workspace_path: Workspace root path.
        repo_path: Repository path.

    Returns:
        Path to workspace/.nodestradamus/snapshots/<repo_hash>/.
    """
    workspace = Path(workspace_path).resolve()
    repo_hash = get_repo_hash(repo_path)
    return workspace / ".nodestradamus" / "snapshots" / repo_hash


def save_snapshot(
    workspace_path: Path | str,
    repo_path: Path | str,
    tool_name: str,
    summary: dict[str, Any],
) -> None:
    """Save a diff-friendly summary for a tool run.

    Writes to workspace/.nodestradamus/snapshots/<repo_hash>/<tool_name>.json
    with timestamp and summary. Overwrites any previous snapshot for that tool.

    Args:
        workspace_path: Workspace root path.
        repo_path: Repository path.
        tool_name: Tool identifier (e.g. project_scout, analyze_deps, codebase_health).
        summary: Diff-friendly summary dict (no full payloads).
    """
    snapshot_dir = get_snapshot_dir(workspace_path, repo_path)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{tool_name}.json"
    payload = {
        "tool": tool_name,
        "saved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "summary": summary,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_snapshot(
    workspace_path: Path | str,
    repo_path: Path | str,
    tool_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load the last saved summary for a tool.

    Args:
        workspace_path: Workspace root path.
        repo_path: Repository path.
        tool_name: Tool identifier.

    Returns:
        (summary dict, saved_at ISO string) or (None, None) if missing/invalid.
    """
    snapshot_dir = get_snapshot_dir(workspace_path, repo_path)
    path = snapshot_dir / f"{tool_name}.json"
    if not path.exists():
        return None, None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary"), data.get("saved_at")
    except (json.JSONDecodeError, OSError):
        return None, None


def diff_summaries(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Compute a simple diff between two summary dicts.

    Produces added keys, removed keys, and for list-like values
    (that are comparable as sets) produces added/removed items.
    Other value changes are recorded as changed[key] = (old, new).

    Args:
        current: Latest summary.
        previous: Previous summary.

    Returns:
        Dict with added_keys, removed_keys, added_items (per key), removed_items (per key), changed (per key).
    """
    delta: dict[str, Any] = {
        "added_keys": [k for k in current if k not in previous],
        "removed_keys": [k for k in previous if k not in current],
        "added_items": {},
        "removed_items": {},
        "changed": {},
    }
    for key in set(current) & set(previous):
        cur = current[key]
        prev = previous[key]
        if cur == prev:
            continue
        if isinstance(cur, list) and isinstance(prev, list):
            set_cur = {_hashable(x) for x in cur if _hashable(x) is not None}
            set_prev = {_hashable(x) for x in prev if _hashable(x) is not None}
            added = set_cur - set_prev
            removed = set_prev - set_cur
            if added or removed:
                delta["added_items"][key] = list(added)
                delta["removed_items"][key] = list(removed)
        elif isinstance(cur, dict) and isinstance(prev, dict):
            sub = diff_summaries(cur, prev)
            if sub["added_keys"] or sub["removed_keys"] or sub["changed"]:
                delta["changed"][key] = sub
        else:
            delta["changed"][key] = (prev, cur)
    return delta


def _hashable(x: Any) -> Any:
    """Convert to hashable for set comparison where possible."""
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        try:
            return tuple(_hashable(e) for e in x)
        except TypeError:
            return None
    if isinstance(x, dict):
        try:
            return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
        except TypeError:
            return None
    return None
