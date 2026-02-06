"""Semantic analysis handlers.

Consolidated handler for semantic_analysis tool with all modes inlined:
- search: Natural language code search
- similar: Find related code to a query/file/symbol
- duplicates: Find copy-pasted code
- embeddings: Compute/refresh embedding cache
"""

import asyncio
import json
from typing import Any

from nodestradamus.analyzers import (
    compute_embeddings,
    detect_duplicates,
    find_similar_code,
    semantic_search,
)


async def handle_semantic_analysis(arguments: dict[str, Any]) -> str:
    """Handle semantic_analysis consolidated tool call.

    Dispatches to appropriate handler based on 'mode' parameter.

    Args:
        arguments: Tool arguments with repo_path, mode, and mode-specific params.

    Returns:
        JSON string with analysis results.

    Raises:
        ValueError: If required parameters are missing or mode is unknown.
    """
    repo_path = arguments.get("repo_path")
    mode = arguments.get("mode")
    workspace_path = arguments.get("workspace_path")
    exclude = arguments.get("exclude")
    package = arguments.get("package")

    if not repo_path:
        raise ValueError("repo_path is required")
    if not mode:
        raise ValueError("mode is required")

    if mode == "search":
        return await _run_search(repo_path, arguments, workspace_path, exclude, package)
    elif mode == "similar":
        return await _run_similar(repo_path, arguments, workspace_path, exclude, package)
    elif mode == "duplicates":
        return await _run_duplicates(repo_path, arguments, workspace_path, exclude, package)
    elif mode == "embeddings":
        return await _run_embeddings(repo_path, arguments, workspace_path, exclude, package)
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def _run_search(
    repo_path: str,
    arguments: dict[str, Any],
    workspace_path: str | None,
    exclude: list[str] | None,
    package: str | None,
) -> str:
    """Run semantic search with natural language query.

    Args:
        repo_path: Path to repository.
        arguments: Tool arguments with query, top_k, threshold.
        workspace_path: Optional workspace path for cache isolation.
        exclude: Directories/patterns to exclude from analysis.
        package: Optional package path for monorepo scoping.

    Returns:
        JSON string with search results.

    Raises:
        ValueError: If query is not provided.
    """
    query = arguments.get("query")
    if not query:
        raise ValueError("query is required for 'search' mode")

    top_k = arguments.get("top_k", 10)
    threshold = arguments.get("threshold", 0.3)

    results = await asyncio.to_thread(
        semantic_search,
        repo_path,
        query=query,
        top_k=top_k,
        threshold=threshold,
        workspace_path=workspace_path,
        exclude=exclude,
        package=package,
    )

    response: dict[str, Any] = {
        "query": query,
        "results": results,
        "count": len(results),
    }
    if package:
        response["package"] = package

    return json.dumps(response, indent=2)


async def _run_similar(
    repo_path: str,
    arguments: dict[str, Any],
    workspace_path: str | None,
    exclude: list[str] | None,
    package: str | None,
) -> str:
    """Find code similar to a query, file, or symbol.

    Args:
        repo_path: Path to repository.
        arguments: Tool arguments with query/file_path/symbol, top_k, threshold.
        workspace_path: Optional workspace path for cache isolation.
        exclude: Directories/patterns to exclude from analysis.
        package: Optional package path for monorepo scoping.

    Returns:
        JSON string with similar code results.

    Raises:
        ValueError: If none of query, file_path, or symbol is provided.
    """
    query = arguments.get("query")
    file_path = arguments.get("file_path")
    symbol = arguments.get("symbol")
    top_k = arguments.get("top_k", 10)
    threshold = arguments.get("threshold", 0.5)

    if not any([query, file_path, symbol]):
        raise ValueError("query, file_path, or symbol is required for 'similar' mode")

    results = await asyncio.to_thread(
        find_similar_code,
        repo_path,
        query=query,
        file_path=file_path,
        symbol=symbol,
        top_k=top_k,
        threshold=threshold,
        workspace_path=workspace_path,
        exclude=exclude,
        package=package,
    )

    response: dict[str, Any] = {
        "similar_code": results,
        "count": len(results),
        "query_type": "text" if query else ("file" if file_path else "symbol"),
    }
    if package:
        response["package"] = package

    return json.dumps(response, indent=2)


async def _run_duplicates(
    repo_path: str,
    arguments: dict[str, Any],
    workspace_path: str | None,
    exclude: list[str] | None,
    package: str | None,
) -> str:
    """Detect duplicate/copy-pasted code.

    Args:
        repo_path: Path to repository.
        arguments: Tool arguments with threshold, max_pairs.
        workspace_path: Optional workspace path for cache isolation.
        exclude: Directories/patterns to exclude from analysis.
        package: Optional package path for monorepo scoping.

    Returns:
        JSON string with duplicate code pairs.
    """
    threshold = arguments.get("threshold", 0.9)
    max_pairs = arguments.get("max_pairs", 50)

    duplicates = await asyncio.to_thread(
        detect_duplicates,
        repo_path,
        threshold=threshold,
        max_pairs=max_pairs,
        workspace_path=workspace_path,
        exclude=exclude,
        package=package,
    )

    response: dict[str, Any] = {
        "duplicates": duplicates,
        "count": len(duplicates),
        "threshold": threshold,
    }
    if package:
        response["package"] = package

    return json.dumps(response, indent=2)


async def _run_embeddings(
    repo_path: str,
    arguments: dict[str, Any],
    workspace_path: str | None,
    exclude: list[str] | None,
    package: str | None,
) -> str:
    """Compute or refresh embedding cache.

    Args:
        repo_path: Path to repository.
        arguments: Tool arguments with languages, chunk_by.
        workspace_path: Optional workspace path for cache isolation.
        exclude: Directories/patterns to exclude from analysis.
        package: Optional package path for monorepo scoping.

    Returns:
        JSON string with embedding computation results.
    """
    languages = arguments.get("languages")
    chunk_by = arguments.get("chunk_by", "function")

    result = await asyncio.to_thread(
        compute_embeddings,
        repo_path,
        languages=languages,
        chunk_by=chunk_by,
        workspace_path=workspace_path,
        exclude=exclude,
        package=package,
    )

    # Don't return the actual embeddings (too large), just metadata
    response: dict[str, Any] = {
        "status": "success",
        "metadata": result["metadata"],
        "sample_chunks": result["chunks"][:10],
    }
    if package:
        response["package"] = package

    return json.dumps(response, indent=2)
