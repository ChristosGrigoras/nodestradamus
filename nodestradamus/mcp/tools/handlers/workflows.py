"""Workflow handlers.

Handlers for workflow tools:
- codebase_health: Run multiple analyses and return a unified health report
- quick_start: Run optimal setup sequence for a new codebase
- get_changes_since_last: Diff current state vs last saved snapshots
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from nodestradamus.analyzers import (
    analyze_docs,
    compute_embeddings,
    detect_duplicates,
    find_dead_code,
    project_scout,
)
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.graph_algorithms import betweenness, find_cycles, top_n
from nodestradamus.analyzers.ignore import load_ignore_patterns
from nodestradamus.mcp.tools.handlers.graph_algorithms import _filter_graph_for_ranking
from nodestradamus.mcp.tools.utils.summarize import short_name, summarize_digraph
from nodestradamus.utils.snapshots import (
    diff_summaries,
    load_snapshot,
    save_snapshot,
)


async def handle_codebase_health(arguments: dict[str, Any]) -> str:
    """Handle codebase_health workflow tool call.

    Runs multiple analyses and returns a unified health report:
    - dead_code: Unused functions/classes
    - duplicates: Copy-pasted code
    - cycles: Circular dependencies
    - bottlenecks: High-impact nodes
    - docs: Stale documentation references

    Args:
        arguments: Tool arguments with repo_path, checks, max_items, scope,
                   exclude_external.

    Returns:
        JSON string with unified health report.

    Raises:
        ValueError: If repo_path is not provided.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")

    checks = arguments.get("checks", ["dead_code", "duplicates", "cycles", "bottlenecks", "docs"])
    max_items = arguments.get("max_items", 20)
    # Scope filtering for bottleneck analysis (E2)
    scope = arguments.get("scope", "source_only")
    exclude_external = arguments.get("exclude_external", True)

    # Load exclude patterns: use provided list, or load from .nodestradamusignore + defaults
    exclude = arguments.get("exclude")
    if exclude is None:
        # Auto-load patterns from .nodestradamusignore and defaults
        exclude = list(load_ignore_patterns(Path(repo_path)))

    report: dict[str, Any] = {
        "summary": {},
        "findings": {},
    }

    # Build dependency graph once (reused by multiple checks)
    G = await asyncio.to_thread(analyze_deps, repo_path, exclude=exclude)

    if G.number_of_nodes() == 0:
        return json.dumps(
            {
                "error": "No nodes found in dependency graph",
                "summary": {"status": "empty"},
                "findings": {},
            },
            indent=2,
        )

    # Run requested checks
    if "dead_code" in checks:
        dead = await asyncio.to_thread(find_dead_code, G)
        report["findings"]["dead_code"] = dead[:max_items]
        report["summary"]["dead_code_count"] = len(dead)

    if "duplicates" in checks:
        duplicates = await asyncio.to_thread(
            detect_duplicates,
            repo_path,
            threshold=0.9,
            max_pairs=max_items,
            exclude=exclude,
        )
        report["findings"]["duplicates"] = duplicates
        report["summary"]["duplicate_pairs"] = len(duplicates)

    if "cycles" in checks:
        cycles = await asyncio.to_thread(find_cycles, G, cross_file_only=True)
        formatted_cycles = []
        for cycle in cycles[:max_items]:
            files_in_cycle = set()
            for node in cycle:
                file_path = G.nodes[node].get("file", "")
                if file_path:
                    files_in_cycle.add(file_path)
            formatted_cycles.append(
                {
                    "length": len(cycle),
                    "files": sorted(files_in_cycle),
                    "cycle": [short_name(node) for node in cycle],
                }
            )
        report["findings"]["cycles"] = formatted_cycles
        report["summary"]["cycle_count"] = len(cycles)

    if "bottlenecks" in checks:
        # Apply filtering to focus on source code (E2)
        G_filtered = _filter_graph_for_ranking(G, scope=scope, exclude_external=exclude_external)
        scores = await asyncio.to_thread(betweenness, G_filtered)
        ranked = top_n(scores, n=max_items)
        bottlenecks = [
            {"node": short_name(node), "full_id": node, "betweenness": round(score, 6)}
            for node, score in ranked
        ]
        report["findings"]["bottlenecks"] = bottlenecks
        report["summary"]["top_bottleneck"] = bottlenecks[0]["node"] if bottlenecks else None
        report["summary"]["bottleneck_filter"] = {"scope": scope, "exclude_external": exclude_external}

    if "docs" in checks:
        doc_result = await asyncio.to_thread(analyze_docs, repo_path)
        stale_refs = [
            {
                "doc_file": ref.doc_file,
                "line": ref.line,
                "raw_text": ref.raw_text,
            }
            for ref in doc_result.stale_references[:max_items]
        ]
        report["findings"]["docs"] = {
            "stale_references": stale_refs,
            "undocumented_exports": doc_result.undocumented_exports[:max_items],
        }
        report["summary"]["stale_doc_refs"] = len(doc_result.stale_references)
        report["summary"]["doc_coverage_percent"] = doc_result.coverage

    # Calculate overall health score (simple heuristic)
    issues = (
        report["summary"].get("dead_code_count", 0)
        + report["summary"].get("duplicate_pairs", 0) * 2
        + report["summary"].get("cycle_count", 0) * 3
        + report["summary"].get("stale_doc_refs", 0)
    )
    if issues == 0:
        report["summary"]["health"] = "excellent"
    elif issues < 5:
        report["summary"]["health"] = "good"
    elif issues < 15:
        report["summary"]["health"] = "fair"
    else:
        report["summary"]["health"] = "needs_attention"

    report["summary"]["total_nodes"] = G.number_of_nodes()
    report["summary"]["checks_run"] = checks

    workspace_path = arguments.get("workspace_path")
    if workspace_path:
        try:
            save_snapshot(workspace_path, repo_path, "codebase_health", report["summary"])
        except Exception:
            pass
    return json.dumps(report, indent=2)


async def handle_quick_start(arguments: dict[str, Any]) -> str:
    """Handle quick_start workflow tool call.

    Runs the optimal setup sequence for a new codebase:
    1. project_scout — Get overview and suggested_ignores
    2. analyze_deps — Build dependency graph
    3. codebase_health — Run health checks (subset)
    4. semantic_analysis embeddings — Pre-compute for fast search

    Args:
        arguments: Tool arguments with repo_path, skip_embeddings, health_checks.

    Returns:
        JSON string with combined setup report.

    Raises:
        ValueError: If repo_path is not provided.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")

    skip_embeddings = arguments.get("skip_embeddings", False)
    health_checks = arguments.get("health_checks", ["cycles", "bottlenecks"])

    report: dict[str, Any] = {
        "status": "success",
        "steps_completed": [],
        "timings": {},
    }

    # Step 1: project_scout
    start = time.perf_counter()
    metadata = await asyncio.to_thread(project_scout, repo_path)
    scout_ms = (time.perf_counter() - start) * 1000
    report["timings"]["project_scout_ms"] = round(scout_ms, 1)
    report["steps_completed"].append("project_scout")

    # Extract key info from scout
    suggested_ignores = metadata.suggested_ignores
    report["scout"] = {
        "primary_language": metadata.primary_language,
        "languages": metadata.languages,
        "frameworks": metadata.frameworks,
        "key_directories": metadata.key_directories,
        "has_tests": metadata.has_tests,
        "suggested_ignores": suggested_ignores,
        "lazy_options": metadata.lazy_options,
    }

    # Step 2: analyze_deps
    start = time.perf_counter()
    G = await asyncio.to_thread(analyze_deps, repo_path, exclude=suggested_ignores)
    deps_ms = (time.perf_counter() - start) * 1000
    report["timings"]["analyze_deps_ms"] = round(deps_ms, 1)
    report["steps_completed"].append("analyze_deps")

    if G.number_of_nodes() == 0:
        report["status"] = "warning"
        report["warning"] = "No nodes found in dependency graph"
        report["deps"] = {"nodes": 0, "edges": 0}
    else:
        # Get summary (summarize_digraph returns summary.* and top_callers/top_called)
        summary = summarize_digraph(G, top_n_count=10, include_fields=False)
        nested = summary.get("summary", {})
        report["deps"] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "files": nested.get("total_files", 0),
            "top_modules": summary.get("top_callers", [])[:5],
        }

    # Step 3: codebase_health (subset of checks for speed)
    if G.number_of_nodes() > 0:
        start = time.perf_counter()
        health: dict[str, Any] = {"checks_run": health_checks}

        if "cycles" in health_checks:
            cycles = await asyncio.to_thread(find_cycles, G, cross_file_only=True)
            health["cycle_count"] = len(cycles)
            if cycles:
                # Show first few cycles
                sample_cycles = []
                for cycle in cycles[:3]:
                    files_in_cycle = set()
                    for node in cycle:
                        file_path = G.nodes[node].get("file", "")
                        if file_path:
                            files_in_cycle.add(file_path)
                    sample_cycles.append(sorted(files_in_cycle))
                health["sample_cycles"] = sample_cycles

        if "bottlenecks" in health_checks:
            # Apply filtering to focus on source code (E2)
            G_filtered = _filter_graph_for_ranking(G, scope="source_only", exclude_external=True)
            scores = await asyncio.to_thread(betweenness, G_filtered)
            ranked = top_n(scores, n=5)
            health["top_bottlenecks"] = [
                {"node": short_name(node), "score": round(score, 4)}
                for node, score in ranked
            ]

        if "dead_code" in health_checks:
            dead = await asyncio.to_thread(find_dead_code, G)
            health["dead_code_count"] = len(dead)

        if "duplicates" in health_checks:
            duplicates = await asyncio.to_thread(
                detect_duplicates, repo_path, threshold=0.9, max_pairs=10
            )
            health["duplicate_pairs"] = len(duplicates)

        health_ms = (time.perf_counter() - start) * 1000
        report["timings"]["health_check_ms"] = round(health_ms, 1)
        report["steps_completed"].append("codebase_health")
        report["health"] = health

    # Step 4: semantic_analysis embeddings (optional)
    if not skip_embeddings:
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                compute_embeddings,
                repo_path,
                chunk_by="function",
                exclude=suggested_ignores,
            )
            embeddings_ms = (time.perf_counter() - start) * 1000
            report["timings"]["embeddings_ms"] = round(embeddings_ms, 1)
            report["steps_completed"].append("semantic_analysis_embeddings")
            report["embeddings"] = {
                "chunks_indexed": len(result.get("chunks", [])),
                "ready_for_search": True,
            }
        except Exception as e:
            report["embeddings"] = {
                "error": str(e),
                "ready_for_search": False,
            }
    else:
        report["embeddings"] = {
            "skipped": True,
            "ready_for_search": False,
            "note": "Run semantic_analysis mode=embeddings before searching",
        }

    # Calculate total time
    total_ms = sum(
        v for k, v in report["timings"].items() if k.endswith("_ms")
    )
    report["timings"]["total_ms"] = round(total_ms, 1)

    # Add next steps guidance
    report["next_steps"] = [
        "semantic_analysis mode=search — Fast semantic code search (embeddings ready)"
        if report.get("embeddings", {}).get("ready_for_search")
        else "semantic_analysis mode=embeddings — Pre-compute embeddings for fast search",
        "analyze_graph algorithm=pagerank — Find most critical code",
        "get_impact file_path=<file> — Analyze blast radius of a file",
    ]

    return json.dumps(report, indent=2)


def _analyze_deps_summary_for_snapshot(G: Any) -> dict[str, Any]:
    """Build a diff-friendly summary from the dependency graph for snapshot/diff."""
    files = sorted(
        {attrs.get("file", "") for _, attrs in G.nodes(data=True) if attrs.get("file")}
    )
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "files": files,
    }


async def handle_get_changes_since_last(arguments: dict[str, Any]) -> str:
    """Handle get_changes_since_last tool call.

    Loads snapshots saved when project_scout, analyze_deps, or codebase_health
    were called with workspace_path. Re-runs the selected tool(s) using caches
    and returns a diff (added/removed/changed) per tool.

    Args:
        arguments: Tool arguments with repo_path, workspace_path, optional tool.

    Returns:
        JSON string with summary and delta per tool.

    Raises:
        ValueError: If repo_path or workspace_path is missing.
    """
    repo_path = arguments.get("repo_path")
    if not repo_path:
        raise ValueError("repo_path is required")
    workspace_path = arguments.get("workspace_path")
    if not workspace_path:
        raise ValueError("workspace_path is required")
    tool_param = arguments.get("tool", "all")
    tools_to_run = (
        ["project_scout", "analyze_deps", "codebase_health"]
        if tool_param == "all"
        else [tool_param]
    )

    # Load exclude patterns from .nodestradamusignore + defaults
    exclude = list(load_ignore_patterns(Path(repo_path)))

    result: dict[str, Any] = {
        "summary": {"repo_path": repo_path, "workspace_path": workspace_path, "tools": tools_to_run},
        "deltas": {},
    }

    for tool_name in tools_to_run:
        if tool_name not in ("project_scout", "analyze_deps", "codebase_health"):
            result["deltas"][tool_name] = {"error": "unsupported tool for snapshot diff"}
            continue
        previous_summary, saved_at = load_snapshot(workspace_path, repo_path, tool_name)
        if previous_summary is None:
            result["deltas"][tool_name] = {
                "status": "no_snapshot",
                "message": "No previous snapshot; run this tool with workspace_path to save one",
            }
            continue

        if tool_name == "project_scout":
            metadata = await asyncio.to_thread(project_scout, repo_path)
            current_summary = metadata.model_dump()
        elif tool_name == "analyze_deps":
            G = await asyncio.to_thread(analyze_deps, repo_path, exclude=exclude)
            current_summary = _analyze_deps_summary_for_snapshot(G)
        else:
            # codebase_health: build same summary shape as handle_codebase_health
            report_summary: dict[str, Any] = {"total_nodes": 0, "checks_run": []}
            G = await asyncio.to_thread(analyze_deps, repo_path, exclude=exclude)
            report_summary["total_nodes"] = G.number_of_nodes()
            if G.number_of_nodes() > 0:
                dead = await asyncio.to_thread(find_dead_code, G)
                report_summary["dead_code_count"] = len(dead)
                cycles = await asyncio.to_thread(find_cycles, G, cross_file_only=True)
                report_summary["cycle_count"] = len(cycles)
                duplicates = await asyncio.to_thread(
                    detect_duplicates, repo_path, threshold=0.9, max_pairs=20, exclude=exclude
                )
                report_summary["duplicate_pairs"] = len(duplicates)
                G_filtered = _filter_graph_for_ranking(
                    G, scope="source_only", exclude_external=True
                )
                scores = await asyncio.to_thread(betweenness, G_filtered)
                ranked = top_n(scores, n=20)
                report_summary["top_bottleneck"] = (
                    short_name(ranked[0][0]) if ranked else None
                )
                doc_result = await asyncio.to_thread(analyze_docs, repo_path)
                report_summary["stale_doc_refs"] = len(doc_result.stale_references)
                report_summary["doc_coverage_percent"] = doc_result.coverage
            issues = (
                report_summary.get("dead_code_count", 0)
                + report_summary.get("duplicate_pairs", 0) * 2
                + report_summary.get("cycle_count", 0) * 3
                + report_summary.get("stale_doc_refs", 0)
            )
            report_summary["health"] = (
                "excellent"
                if issues == 0
                else "good"
                if issues < 5
                else "fair"
                if issues < 15
                else "needs_attention"
            )
            report_summary["checks_run"] = [
                "dead_code",
                "duplicates",
                "cycles",
                "bottlenecks",
                "docs",
            ]
            current_summary = report_summary

        delta = diff_summaries(current_summary, previous_summary)
        result["deltas"][tool_name] = {
            "status": "ok",
            "previous_saved_at": saved_at,
            "delta": delta,
        }

    return json.dumps(result, indent=2)
