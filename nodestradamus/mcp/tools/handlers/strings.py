"""String analysis handlers.

Consolidated handler for analyze_strings tool with all modes inlined:
- refs: Find shared strings across files
- usages: Find where a specific string is used
- filter: Clean noisy results from refs output
"""

import asyncio
import json
from typing import Any

from nodestradamus.analyzers import (
    analyze_string_topology,
    find_string_usages,
)
from nodestradamus.mcp.tools.utils.string_filters import (
    TYPE_ANNOTATIONS,
    is_css_class,
    is_import_path,
)


async def handle_analyze_strings(arguments: dict[str, Any]) -> str:
    """Handle analyze_strings consolidated tool call.

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

    if not repo_path:
        raise ValueError("repo_path is required")
    if not mode:
        raise ValueError("mode is required")

    if mode == "refs":
        return await _run_refs(repo_path, arguments)
    elif mode == "usages":
        return await _run_usages(repo_path, arguments)
    elif mode == "filter":
        return _run_filter(arguments)
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def _run_refs(repo_path: str, arguments: dict[str, Any]) -> str:
    """Find shared strings across files.

    Args:
        repo_path: Path to repository.
        arguments: Tool arguments with min_files, top_n, summary_only.

    Returns:
        JSON string with significant strings.
        When summary_only=true, returns concise summary with top 5 strings.
    """
    min_files = arguments.get("min_files", 2)
    include_single_use = arguments.get("include_single_use", False)
    top_n_count = arguments.get("top_n", 50)
    summary_only = arguments.get("summary_only", False)

    result = await asyncio.to_thread(
        analyze_string_topology,
        repo_path,
        min_files=min_files,
        include_single_use=include_single_use,
    )

    # Format for compact output
    strings = result.significant_strings[:top_n_count]
    formatted = []
    for s in strings:
        call_sites = set()
        for ctx in s.contexts:
            if ctx.call_site:
                call_sites.add(ctx.call_site)
        formatted.append(
            {
                "value": s.value,
                "referenced_by": s.referenced_by,
                "reference_count": s.reference_count,
                "importance_score": round(s.importance_score, 3),
                "call_sites": sorted(call_sites)[:5],
            }
        )

    # Detect noise and add hint for LLM to chain filter_strings
    noise_hint = None
    if formatted:
        noise_signals = []
        top_val = formatted[0]["value"]
        top_count = formatted[0]["reference_count"]

        # Check for type annotations at top
        if top_val in TYPE_ANNOTATIONS:
            noise_signals.append(f"top result '{top_val}' is a type annotation")

        # Check for statistical outlier
        if len(formatted) >= 2:
            second_count = formatted[1]["reference_count"]
            if top_count > second_count * 2:
                noise_signals.append(
                    f"top result ({top_count} refs) is >2x second ({second_count})"
                )

        # Check for imports/CSS in top 10
        imports_in_top = sum(1 for s in formatted[:10] if is_import_path(s["value"]))
        css_in_top = sum(1 for s in formatted[:10] if is_css_class(s["value"]))
        if imports_in_top >= 2:
            noise_signals.append(f"{imports_in_top} import paths in top 10")
        if css_in_top >= 2:
            noise_signals.append(f"{css_in_top} CSS classes in top 10")

        if noise_signals:
            noise_hint = (
                f"⚠️ Noise detected: {'; '.join(noise_signals)}. "
                "Use filter_strings tool to clean results."
            )

    # Summary mode (E7): return concise output with top 5 strings
    if summary_only:
        top_strings = []
        for s in formatted[:5]:
            top_strings.append({
                "value": s["value"][:50] + ("..." if len(s["value"]) > 50 else ""),
                "reference_count": s["reference_count"],
            })

        return json.dumps(
            {
                "summary": {
                    "total_strings": result.metadata.get("total_strings", len(formatted)),
                    "significant_strings": len(formatted),
                },
                "top_strings": top_strings,
                "truncated": True,
                "message": f"Showing top 5 of {len(formatted)} strings. Use summary_only=false for full results.",
            },
            indent=2,
        )

    output: dict[str, Any] = {
        "significant_strings": formatted,
        "metadata": result.metadata,
    }
    if noise_hint:
        output["noise_hint"] = noise_hint

    return json.dumps(output, indent=2)


async def _run_usages(repo_path: str, arguments: dict[str, Any]) -> str:
    """Find where a specific string is used.

    Args:
        repo_path: Path to repository.
        arguments: Tool arguments with target_string.

    Returns:
        JSON string with usage locations.

    Raises:
        ValueError: If target_string is not provided.
    """
    target_string = arguments.get("target_string")
    if not target_string:
        raise ValueError("target_string is required for 'usages' mode")

    result = await asyncio.to_thread(
        find_string_usages,
        repo_path,
        target_string,
    )

    return json.dumps(result, indent=2)


def _run_filter(arguments: dict[str, Any]) -> str:
    """Filter noisy strings from refs output.

    Args:
        arguments: Tool arguments with strings, exclude_types, exclude_imports,
                   exclude_css, min_length, max_frequency_percentile.

    Returns:
        JSON string with filtered results.

    Raises:
        ValueError: If strings is not provided.
    """
    strings = arguments.get("strings")
    if not strings:
        raise ValueError("strings is required for 'filter' mode")

    if len(strings) == 0:
        return json.dumps({"filtered_strings": [], "removed_count": 0}, indent=2)

    exclude_types = arguments.get("exclude_types", True)
    exclude_imports = arguments.get("exclude_imports", True)
    exclude_css = arguments.get("exclude_css", True)
    min_length = arguments.get("min_length", 3)
    max_percentile = arguments.get("max_frequency_percentile", 95)

    # Calculate percentile threshold
    counts = [s.get("reference_count", 1) for s in strings]
    if counts and max_percentile < 100:
        sorted_counts = sorted(counts)
        percentile_idx = int(len(sorted_counts) * max_percentile / 100)
        percentile_threshold = sorted_counts[min(percentile_idx, len(sorted_counts) - 1)]
    else:
        percentile_threshold = float("inf")

    filtered = []
    removed = {"type_annotations": 0, "imports": 0, "css": 0, "short": 0, "outliers": 0}

    for s in strings:
        value = s.get("value", "")
        count = s.get("reference_count", 1)

        # Length filter
        if len(value) < min_length:
            removed["short"] += 1
            continue

        # Type annotation filter
        if exclude_types and value in TYPE_ANNOTATIONS:
            removed["type_annotations"] += 1
            continue

        # Import path filter
        if exclude_imports and is_import_path(value):
            removed["imports"] += 1
            continue

        # CSS class filter
        if exclude_css and is_css_class(value):
            removed["css"] += 1
            continue

        # Statistical outlier filter
        if count > percentile_threshold:
            removed["outliers"] += 1
            continue

        filtered.append(s)

    return json.dumps(
        {
            "filtered_strings": filtered,
            "original_count": len(strings),
            "filtered_count": len(filtered),
            "removed_breakdown": removed,
            "percentile_threshold": (
                percentile_threshold if percentile_threshold != float("inf") else None
            ),
        },
        indent=2,
    )
