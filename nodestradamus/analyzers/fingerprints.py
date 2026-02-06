"""Structural code fingerprinting for fast similar-code lookup.

Builds Shazam-like constellations from parse cache (node type + edge type pairs),
hashes them, and supports lookup by file/range to find structurally similar
locations. Uses normalized node IDs: {lang}:{relative_path}::{symbol}.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack

from nodestradamus.analyzers.code_parser import parse_directory
from nodestradamus.analyzers.deps import _detect_languages
from nodestradamus.analyzers.ignore import load_ignore_patterns
from nodestradamus.logging import logger

# 2.0: Switched to MessagePack binary format
FINGERPRINT_CACHE_VERSION = "2.0"


def _get_fingerprint_cache_path(repo_path: Path) -> Path:
    """Path to the fingerprint cache file (repo-local)."""
    return repo_path / ".nodestradamus" / "fingerprints.msgpack"


def _get_legacy_fingerprint_cache_path(repo_path: Path) -> Path:
    """Path to the legacy JSON fingerprint cache (for migration)."""
    return repo_path / ".nodestradamus" / "fingerprints.json"


def _descriptor_key(
    type_a: str,
    type_b: str,
    edge_type: str,
) -> str:
    """Normalize (type_a, type_b, edge_type) into a canonical string for hashing."""
    parts = sorted([type_a, type_b]) + [edge_type]
    return "|".join(parts)


def _hash_descriptor(descriptor: str) -> str:
    """Hash a descriptor to a short hex string."""
    return hashlib.sha256(descriptor.encode()).hexdigest()[:16]


def build_fingerprint_index(
    repo_path: str | Path,
    languages: list[str] | None = None,
    use_cache: bool = True,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Build the structural fingerprint index from parsed nodes/edges.

    For each same-file edge, forms a descriptor (node_type_a, node_type_b, edge_type),
    hashes it, and records (file, node_id) as a location for that hash. Uses
    normalized node IDs from the parser.

    Args:
        repo_path: Repository root path.
        languages: Languages to include (None = auto-detect).
        use_cache: Use parse cache when calling parse_directory.
        exclude: Patterns to exclude; None uses load_ignore_patterns.

    Returns:
        Dict with hash_to_locations, file_to_hashes, version, created_at, and metadata.
    """
    repo_path = Path(repo_path).resolve()
    if languages is None:
        languages = _detect_languages(repo_path)
    if exclude is None:
        exclude = list(load_ignore_patterns(repo_path, languages=dict.fromkeys(languages, 1)))

    result = parse_directory(repo_path, languages=languages, use_cache=use_cache, exclude=exclude)
    nodes: list[dict[str, Any]] = result["nodes"]
    edges: list[dict[str, Any]] = result["edges"]

    node_by_id: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}

    hash_to_locations: dict[str, list[dict[str, str]]] = {}
    file_to_hashes: dict[str, set[str]] = {}

    for edge in edges:
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))
        edge_type = edge.get("type", "unknown")
        if not source or not target:
            continue
        na = node_by_id.get(source)
        nb = node_by_id.get(target)
        if not na or not nb:
            continue
        file_a = na.get("file", "")
        file_b = nb.get("file", "")
        if file_a != file_b:
            continue
        type_a = na.get("type", "unknown")
        type_b = nb.get("type", "unknown")
        descriptor = _descriptor_key(type_a, type_b, edge_type)
        h = _hash_descriptor(descriptor)
        loc_a = {"file": file_a, "node_id": source}
        loc_b = {"file": file_b, "node_id": target}
        if h not in hash_to_locations:
            hash_to_locations[h] = []
        hash_to_locations[h].append(loc_a)
        hash_to_locations[h].append(loc_b)
        file_to_hashes.setdefault(file_a, set()).add(h)

    created_at = datetime.now(UTC).isoformat()
    index = {
        "version": FINGERPRINT_CACHE_VERSION,
        "created_at": created_at,
        "hash_to_locations": dict(hash_to_locations.items()),
        "file_to_hashes": {f: list(s) for f, s in file_to_hashes.items()},
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "hash_count": len(hash_to_locations),
        },
    }

    if use_cache:
        _save_fingerprint_index(repo_path, index)
    return index


def _save_fingerprint_index(repo_path: Path, index: dict[str, Any]) -> None:
    """Write fingerprint index using MessagePack format."""
    path = _get_fingerprint_cache_path(repo_path)
    legacy_path = _get_legacy_fingerprint_cache_path(repo_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            msgpack.pack(index, f)
        logger.info(
            "  Saved fingerprint index with %d hashes (msgpack)",
            len(index.get("hash_to_locations", {})),
        )

        # Clean up legacy JSON cache if it exists
        if legacy_path.exists():
            try:
                legacy_path.unlink()
                logger.info("  Removed legacy JSON fingerprint cache")
            except OSError:
                pass

    except OSError as e:
        logger.warning("  Failed to save fingerprint index: %s", e)


def load_fingerprint_index(repo_path: str | Path) -> dict[str, Any] | None:
    """Load fingerprint index from cache if valid.

    Tries MessagePack first, falls back to legacy JSON.
    """
    repo_path = Path(repo_path).resolve()
    path = _get_fingerprint_cache_path(repo_path)
    legacy_path = _get_legacy_fingerprint_cache_path(repo_path)

    # Try MessagePack first
    if path.exists():
        try:
            with path.open("rb") as f:
                data = msgpack.unpack(f, raw=False)
            if data.get("version") != FINGERPRINT_CACHE_VERSION:
                logger.info("  Fingerprint cache version mismatch, ignoring")
                return None
            return data
        except (msgpack.UnpackException, msgpack.ExtraData, OSError, KeyError, ValueError) as e:
            logger.warning("  Failed to load fingerprint index: %s", e)
            return None

    # Fall back to legacy JSON (for migration info)
    if legacy_path.exists():
        logger.info("  Found legacy JSON fingerprint cache, will rebuild")
        return None

    return None


def find_similar(
    repo_path: str | Path,
    file_path: str,
    line_start: int | None = None,
    line_end: int | None = None,
    top_k: int = 15,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Find structurally similar locations to the given file (and optional line range).

    Builds or loads the fingerprint index, extracts hashes for nodes in the
    given file/range, looks up all locations for those hashes, and returns
    the top_k locations by overlap count (excluding the query location).

    Args:
        repo_path: Repository root path.
        file_path: Relative path to the file (anchor for similarity).
        line_start: Optional start line (1-based); None = whole file.
        line_end: Optional end line (1-based); None = whole file.
        top_k: Maximum number of similar locations to return.
        use_cache: Use and update fingerprint cache.

    Returns:
        Dict with summary (match_count, top_k, file, range) and matches
        list of {file, node_id, overlap_count}.
    """
    repo_path = Path(repo_path).resolve()
    index = load_fingerprint_index(repo_path)
    if index is None:
        index = build_fingerprint_index(repo_path, use_cache=use_cache)
    hash_to_locations = index.get("hash_to_locations", {})
    if not hash_to_locations:
        return {
            "summary": {
                "match_count": 0,
                "top_k": top_k,
                "file": file_path,
                "line_range": [line_start, line_end],
            },
            "matches": [],
        }

    result = parse_directory(repo_path, use_cache=True)
    nodes = result["nodes"]
    edges = result["edges"]
    node_by_id = {n["id"]: n for n in nodes}

    file_nodes = [n for n in nodes if n.get("file") == file_path]
    if line_start is not None or line_end is not None:
        file_nodes = [
            n
            for n in file_nodes
            if n.get("line") is not None
            and (line_start is None or n["line"] >= line_start)
            and (line_end is None or n["line"] <= line_end)
        ]
    node_ids_in_scope = {n["id"] for n in file_nodes}

    hashes_in_scope: set[str] = set()
    for edge in edges:
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))
        if source not in node_ids_in_scope and target not in node_ids_in_scope:
            continue
        na = node_by_id.get(source)
        nb = node_by_id.get(target)
        if not na or not nb:
            continue
        if na.get("file") != nb.get("file"):
            continue
        type_a = na.get("type", "unknown")
        type_b = nb.get("type", "unknown")
        edge_type = edge.get("type", "unknown")
        descriptor = _descriptor_key(type_a, type_b, edge_type)
        hashes_in_scope.add(_hash_descriptor(descriptor))

    overlap_count: dict[tuple[str, str], int] = {}
    for h in hashes_in_scope:
        for loc in hash_to_locations.get(h, []):
            loc_file = loc.get("file", "")
            loc_id = loc.get("node_id", "")
            if loc_file == file_path and loc_id in node_ids_in_scope:
                continue
            key = (loc_file, loc_id)
            overlap_count[key] = overlap_count.get(key, 0) + 1

    ranked = sorted(overlap_count.items(), key=lambda x: -x[1])[:top_k]
    matches = [
        {"file": f, "node_id": nid, "overlap_count": c}
        for (f, nid), c in ranked
    ]

    return {
        "summary": {
            "match_count": len(overlap_count),
            "top_k": top_k,
            "file": file_path,
            "line_range": [line_start, line_end],
        },
        "matches": matches,
    }
