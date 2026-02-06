"""Handler for find_similar (structural fingerprint matching)."""

import asyncio
import json
from typing import Any

from nodestradamus.analyzers.fingerprints import find_similar


async def handle_find_similar(arguments: dict[str, Any]) -> str:
    """Handle find_similar tool call.

    Finds structurally similar code to a file or region using fingerprint index.
    Returns summary (match_count, top_k, file, range) and matches list.

    Args:
        arguments: Tool arguments with repo_path, file_path, optional line_start,
                   line_end, top_k.

    Returns:
        JSON string with summary and matches.

    Raises:
        ValueError: If repo_path or file_path is missing.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")
    file_path = arguments.get("file_path")
    if not file_path:
        raise ValueError("file_path is required")
    line_start = arguments.get("line_start")
    line_end = arguments.get("line_end")
    top_k = arguments.get("top_k", 15)

    result = await asyncio.to_thread(
        find_similar,
        repo_path,
        file_path,
        line_start=line_start,
        line_end=line_end,
        top_k=top_k,
    )
    return json.dumps(result, indent=2)
