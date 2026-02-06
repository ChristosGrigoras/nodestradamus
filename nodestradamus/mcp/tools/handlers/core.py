"""Core MCP tool handlers.

Handlers for the main analysis tools:
- analyze_deps: Dependency graph analysis
- analyze_cooccurrence: Git co-occurrence analysis
- get_impact: Impact analysis for file/symbol changes
- project_scout: Repository reconnaissance
- analyze_docs: Documentation analysis
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from nodestradamus.analyzers import (
    analyze_docs,
    analyze_git_cooccurrence,
    project_scout,
)
from nodestradamus.analyzers.clustering import cluster_functions_in_file
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.duplicates import find_exact_duplicates
from nodestradamus.analyzers.ignore import load_ignore_patterns
from nodestradamus.analyzers.impact import (
    AmbiguousMatchError,
    NoMatchError,
    _get_breaking_changes,
    _get_symbol_usage,
    get_impact,
)
from nodestradamus.mcp.tools.utils.summarize import summarize_cooccurrence, summarize_digraph
from nodestradamus.models.graph import RefactorAnalysis, digraph_to_json, graph_to_json
from nodestradamus.utils.snapshots import save_snapshot


async def handle_project_scout(arguments: dict[str, Any]) -> str:
    """Handle project_scout tool call.

    Args:
        arguments: Tool arguments with repo_path.

    Returns:
        JSON string with project metadata.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")

    metadata = await asyncio.to_thread(project_scout, repo_path)
    workspace_path = arguments.get("workspace_path")
    if workspace_path:
        try:
            save_snapshot(workspace_path, repo_path, "project_scout", metadata.model_dump())
        except Exception:
            pass
    return metadata.model_dump_json(indent=2)


async def handle_analyze_docs(arguments: dict[str, Any]) -> str:
    """Handle analyze_docs tool call.

    Args:
        arguments: Tool arguments with repo_path, docs_path, include_readme.

    Returns:
        JSON string with documentation analysis results.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")

    docs_path = arguments.get("docs_path")
    include_readme = arguments.get("include_readme", True)

    result = await asyncio.to_thread(
        analyze_docs,
        repo_path,
        docs_path=docs_path,
        include_readme=include_readme,
    )

    # Format stale references for output, sorted by confidence (E14)
    high_confidence = []
    medium_confidence = []
    low_confidence = []

    for ref in result.stale_references:
        ref_dict = {
            "doc_file": ref.doc_file,
            "line": ref.line,
            "ref_type": ref.ref_type,
            "raw_text": ref.raw_text,
            "confidence": ref.confidence,
        }
        if ref.confidence == "high":
            high_confidence.append(ref_dict)
        elif ref.confidence == "medium":
            medium_confidence.append(ref_dict)
        else:
            low_confidence.append(ref_dict)

    output = {
        "summary": {
            "total_docs": result.total_docs,
            "total_references": result.total_references,
            "valid_references": result.valid_references,
            "stale_references": len(result.stale_references),
            "stale_by_confidence": {
                "high": len(high_confidence),
                "medium": len(medium_confidence),
                "low": len(low_confidence),
            },
            "coverage_percent": result.coverage,
        },
        # Show high-confidence stale refs first (most likely actual issues)
        "stale_high_confidence": high_confidence[:20],
        "stale_medium_confidence": medium_confidence[:20],
        "stale_low_confidence": low_confidence[:10],
        "undocumented_exports": result.undocumented_exports[:30],
        "metadata": result.metadata,
    }

    return json.dumps(output, indent=2)


async def handle_analyze_deps(arguments: dict[str, Any]) -> str:
    """Handle analyze_deps tool call.

    Args:
        arguments: Tool arguments with repo_path, languages, full_graph, top_n,
                   include_fields, exclude, package.

    Returns:
        JSON string with dependency graph or summary.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")

    languages = arguments.get("languages")
    full_graph = arguments.get("full_graph", False)
    top_n_count = arguments.get("top_n", 15)
    include_fields = arguments.get("include_fields", False)
    package = arguments.get("package")

    # Load exclude patterns: use provided list, or auto-load from .nodestradamusignore + defaults
    exclude = arguments.get("exclude")
    if exclude is None:
        exclude = list(load_ignore_patterns(Path(repo_path)))

    # Time the graph building
    start = time.perf_counter()
    G = await asyncio.to_thread(
        analyze_deps, repo_path, languages, exclude=exclude, package=package
    )
    graph_build_ms = (time.perf_counter() - start) * 1000

    # Count files processed
    files_processed = len({
        attrs.get("file", "")
        for _, attrs in G.nodes(data=True)
        if attrs.get("file")
    })

    if full_graph:
        graph_json = digraph_to_json(G)
        # Include fields if requested and present
        if include_fields:
            for node in graph_json.get("nodes", []):
                node_id = node.get("id", "")
                if node_id in G:
                    fields = G.nodes[node_id].get("fields")
                    if fields:
                        node["fields"] = fields
        # Add detailed timing
        graph_json["_timing_detail"] = {
            "graph_build_ms": round(graph_build_ms, 1),
            "files_processed": files_processed,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
        return json.dumps(graph_json, indent=2)

    summary = summarize_digraph(G, top_n_count=top_n_count, include_fields=include_fields)
    # Add detailed timing to summary
    summary["_timing_detail"] = {
        "graph_build_ms": round(graph_build_ms, 1),
        "files_processed": files_processed,
    }
    workspace_path = arguments.get("workspace_path")
    if workspace_path:
        try:
            snapshot_summary = {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "files": sorted(
                    {attrs.get("file", "") for _, attrs in G.nodes(data=True) if attrs.get("file")}
                ),
            }
            save_snapshot(workspace_path, repo_path, "analyze_deps", snapshot_summary)
        except Exception:
            pass
    return json.dumps(summary, indent=2)


async def handle_analyze_cooccurrence(arguments: dict[str, Any]) -> str:
    """Handle analyze_cooccurrence tool call.

    Args:
        arguments: Tool arguments with repo_path, commits, full_graph, top_n.

    Returns:
        JSON string with co-occurrence analysis results.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")

    commits = arguments.get("commits", 500)
    full_graph = arguments.get("full_graph", False)
    top_n_count = arguments.get("top_n", 20)

    G = await asyncio.to_thread(
        analyze_git_cooccurrence,
        repo_path,
        commits=commits,
    )

    if full_graph:
        return json.dumps(graph_to_json(G), indent=2)

    summary = summarize_cooccurrence(G, top_n_count=top_n_count)
    return json.dumps(summary, indent=2)


async def handle_get_impact(arguments: dict[str, Any]) -> str:
    """Handle get_impact tool call.

    Args:
        arguments: Tool arguments with repo_path, file_path, symbol, depth,
                   include_semantic, fusion_mode, refactor_mode, compact,
                   exclude_same_file, exclude_external.

    Returns:
        JSON string with impact analysis report.
    """
    repo_path = arguments.get("repo_path")
    file_path = arguments.get("file_path")

    if not repo_path:
        raise ValueError("repo_path is required")
    if not file_path:
        raise ValueError("file_path is required")

    symbol = arguments.get("symbol")
    depth = arguments.get("depth", 3)
    include_semantic = arguments.get("include_semantic", False)
    fusion_mode = arguments.get("fusion_mode", False)
    refactor_mode = arguments.get("refactor_mode", False)
    compact = arguments.get("compact", True)
    exclude_same_file = arguments.get("exclude_same_file", True)
    exclude_external = arguments.get("exclude_external", True)

    try:
        report = await asyncio.to_thread(
            get_impact,
            repo_path,
            file_path,
            symbol=symbol,
            depth=depth,
            include_semantic=include_semantic,
            fusion_mode=fusion_mode,
            compact=compact,
            exclude_same_file=exclude_same_file,
            exclude_external=exclude_external,
        )

        # If refactor_mode is enabled, compute additional analysis
        if refactor_mode:
            # Build the graph for refactor analysis
            G = await asyncio.to_thread(analyze_deps, repo_path)

            # Normalize file_path (strip leading ./)
            target_file = file_path.lstrip("./")

            # Get symbol usage
            symbol_usage = await asyncio.to_thread(
                _get_symbol_usage, G, target_file
            )

            # Get breaking changes
            breaking_changes = await asyncio.to_thread(
                _get_breaking_changes, G, target_file
            )

            # Find duplicate code blocks
            duplicates = await asyncio.to_thread(
                find_exact_duplicates,
                Path(repo_path),
                target_file,
            )

            # Cluster functions
            clusters = await asyncio.to_thread(
                cluster_functions_in_file,
                Path(repo_path),
                target_file,
            )

            # Create RefactorAnalysis
            refactor_analysis = RefactorAnalysis(
                symbol_usage=symbol_usage,
                duplicates=duplicates,
                clusters=clusters,
                breaking_changes=breaking_changes,
            )

            # Attach to report
            report.refactor_analysis = refactor_analysis

        return report.model_dump_json(indent=2)

    except AmbiguousMatchError as e:
        # Return structured error with candidates for disambiguation
        error_response = {
            "error": "ambiguous_match",
            "message": str(e),
            "query": {
                "file_path": file_path,
                "symbol": symbol,
            },
            "candidates": e.candidates,
            "hint": (
                "Multiple nodes match your query. "
                "Provide a more specific file_path (e.g., 'src/api/request.py' instead of 'request.py') "
                "or specify the full path from the repository root."
            ),
        }
        return json.dumps(error_response, indent=2)

    except NoMatchError as e:
        # Return structured error with suggestions
        error_response = {
            "error": "no_match",
            "message": str(e),
            "query": {
                "file_path": file_path,
                "symbol": symbol,
            },
            "suggestions": e.suggestions,
            "hint": (
                "No nodes found matching your query. "
                "Check the file path spelling and ensure the file is part of the analyzed codebase. "
                "Use analyze_deps first to see what files are in the graph."
            ),
        }
        return json.dumps(error_response, indent=2)
