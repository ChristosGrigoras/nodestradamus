"""Rules audit handler.

Handler for compare_rules_to_codebase tool:
- Discovers rules from Cursor, OpenCode, or Claude Code formats
- Derives inferred facets from codebase analysis
- Compares existing rules to codebase hotspots
- Returns coverage, gaps, stale references, and recommendations
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from nodestradamus.analyzers import detect_duplicates, project_scout
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.graph_algorithms import (
    betweenness,
    find_cycles,
    pagerank,
    top_n,
)
from nodestradamus.mcp.tools.handlers.graph_algorithms import _filter_graph_for_ranking
from nodestradamus.mcp.tools.utils.rule_parser import (
    RuleSource,
    check_path_coverage,
    discover_rules,
    find_stale_references,
)
from nodestradamus.mcp.tools.utils.summarize import short_name


async def handle_compare_rules(arguments: dict[str, Any]) -> str:
    """Handle compare_rules_to_codebase tool call.

    Compares existing rules (Cursor, OpenCode, Claude Code) with codebase analysis.
    Returns coverage, gaps, stale references, and recommendations.

    Args:
        arguments: Tool arguments with repo_path, rules_sources, custom_rules_path,
                   top_n, include_duplicates, include_cycles.

    Returns:
        JSON string with comparison report.

    Raises:
        ValueError: If repo_path is not provided.
    """
    repo_path_str = arguments.get("repo_path")
    if not repo_path_str:
        raise ValueError("repo_path is required")

    repo_path = Path(repo_path_str)
    if not repo_path.is_dir():
        raise ValueError(f"repo_path is not a directory: {repo_path}")

    # Parse arguments
    rules_sources_arg = arguments.get("rules_sources")
    custom_rules_path = arguments.get("custom_rules_path")
    top_n_count = arguments.get("top_n", 15)
    include_duplicates = arguments.get("include_duplicates", False)
    include_cycles = arguments.get("include_cycles", False)

    # Convert rules_sources to proper type
    rules_sources: list[RuleSource] | None = None
    if rules_sources_arg:
        rules_sources = [s for s in rules_sources_arg if s in ("cursor", "opencode", "claude")]

    report: dict[str, Any] = {}

    # Step 1: Discover and parse existing rules
    discovery_result = await asyncio.to_thread(
        discover_rules,
        repo_path,
        sources=rules_sources,
        custom_path=custom_rules_path,
    )

    existing_rules = discovery_result.rules
    report["existing_rules_summary"] = {
        "count": len(existing_rules),
        "sources_checked": discovery_result.sources_checked,
        "sources_found": discovery_result.sources_found,
        "rules": [
            {
                "file": rule.file,
                "source": rule.source,
                "code_paths_count": len(rule.code_paths),
            }
            for rule in existing_rules
        ],
    }

    if not existing_rules:
        report["existing_rules_summary"]["message"] = "No rules found"

    # Step 2: Run codebase analysis to derive inferred facets
    # Run project_scout for structure
    scout_result = await asyncio.to_thread(project_scout, str(repo_path))
    report["inferred_facets"] = {
        "structure": {
            "primary_language": scout_result.primary_language,
            "languages": scout_result.languages,
            "frameworks": scout_result.frameworks,
            "is_monorepo": scout_result.is_monorepo,
            "packages": [
                {"name": p.name, "path": p.path}
                for p in scout_result.packages[:10]
            ] if scout_result.packages else [],
            "key_directories": scout_result.key_directories,
        }
    }

    # Build dependency graph
    G = await asyncio.to_thread(
        analyze_deps,
        str(repo_path),
        exclude=scout_result.suggested_ignores,
    )

    if G.number_of_nodes() == 0:
        report["inferred_facets"]["critical_files"] = []
        report["inferred_facets"]["bottlenecks"] = []
        report["coverage"] = {}
        report["gaps"] = []
        report["stale"] = []
        report["recommendations"] = [
            "No code found to analyze. Check repo_path points to a code repository."
        ]
        return json.dumps(report, indent=2)

    # Filter graph for source code ranking
    G_filtered = _filter_graph_for_ranking(G, scope="source_only", exclude_external=True)

    # Run pagerank for critical files
    pr_scores = pagerank(G_filtered)
    pr_ranked = top_n(pr_scores, n=top_n_count)
    critical_files = []
    critical_paths = []
    for node, score in pr_ranked:
        file_path = G.nodes[node].get("file", "")
        if file_path:
            critical_files.append({
                "node": short_name(node),
                "file": file_path,
                "importance": round(score, 6),
            })
            if file_path not in critical_paths:
                critical_paths.append(file_path)
    report["inferred_facets"]["critical_files"] = critical_files

    # Run betweenness for bottlenecks
    bw_scores = betweenness(G_filtered)
    bw_ranked = top_n(bw_scores, n=top_n_count)
    bottlenecks = []
    bottleneck_paths = []
    for node, score in bw_ranked:
        file_path = G.nodes[node].get("file", "")
        if file_path:
            bottlenecks.append({
                "node": short_name(node),
                "file": file_path,
                "betweenness": round(score, 6),
            })
            if file_path not in bottleneck_paths:
                bottleneck_paths.append(file_path)
    report["inferred_facets"]["bottlenecks"] = bottlenecks

    # Optional: Run cycle detection
    if include_cycles:
        cycles = find_cycles(G, cross_file_only=True)
        formatted_cycles = []
        for cycle in cycles[:10]:
            files_in_cycle = set()
            for node in cycle:
                file_path = G.nodes[node].get("file", "")
                if file_path:
                    files_in_cycle.add(file_path)
            formatted_cycles.append({
                "length": len(cycle),
                "files": sorted(files_in_cycle),
            })
        report["inferred_facets"]["cycles"] = {
            "count": len(cycles),
            "samples": formatted_cycles,
        }

    # Optional: Run duplicate detection
    if include_duplicates:
        duplicates = await asyncio.to_thread(
            detect_duplicates,
            str(repo_path),
            threshold=0.9,
            max_pairs=20,
        )
        report["inferred_facets"]["duplicates"] = {
            "count": len(duplicates),
            "pairs": duplicates[:10],
        }

    # Step 3: Compare rules to inferred facets
    # Combine critical and bottleneck paths for coverage check
    all_hotspot_paths = list(dict.fromkeys(critical_paths + bottleneck_paths))

    if existing_rules:
        # Check coverage: which hotspots are mentioned in rules
        coverage = check_path_coverage(existing_rules, all_hotspot_paths)
        report["coverage"] = {
            path: {
                "mentioned_in": rules,
                "is_covered": len(rules) > 0,
            }
            for path, rules in coverage.items()
        }

        # Find gaps: hotspots not mentioned in any rule
        gaps = [
            {
                "path": path,
                "type": "critical" if path in critical_paths else "bottleneck",
                "suggestion": f"Consider adding guidance for {path}",
            }
            for path, rules in coverage.items()
            if not rules
        ]
        report["gaps"] = gaps

        # Find stale references: rule paths that don't exist
        stale_refs = find_stale_references(existing_rules, repo_path)
        report["stale"] = stale_refs
    else:
        # No rules exist - all hotspots are gaps
        report["coverage"] = {}
        report["gaps"] = [
            {
                "path": path,
                "type": "critical" if path in critical_paths else "bottleneck",
                "suggestion": f"Consider documenting {path}",
            }
            for path in all_hotspot_paths
        ]
        report["stale"] = []

    # Step 4: Generate recommendations
    recommendations = []

    if not existing_rules:
        recommendations.append(
            "No rules found. Consider creating rules to document codebase conventions."
        )
        if critical_paths:
            recommendations.append(
                f"Start by documenting the most critical files: {', '.join(critical_paths[:3])}"
            )

    covered_count = sum(
        1 for v in report.get("coverage", {}).values()
        if v.get("is_covered")
    )
    total_hotspots = len(all_hotspot_paths)

    if existing_rules and total_hotspots > 0:
        coverage_pct = covered_count / total_hotspots * 100
        if coverage_pct < 50:
            recommendations.append(
                f"Low hotspot coverage ({coverage_pct:.0f}%). "
                f"Consider adding guidance for undocumented critical files."
            )
        elif coverage_pct >= 80:
            recommendations.append(
                f"Good hotspot coverage ({coverage_pct:.0f}%). Rules document most critical code."
            )

    gaps = report.get("gaps", [])
    if gaps:
        top_gaps = [g["path"] for g in gaps[:5]]
        recommendations.append(
            f"Top undocumented hotspots: {', '.join(top_gaps)}"
        )

    stale = report.get("stale", [])
    if stale:
        stale_paths = [s["path"] for s in stale[:3]]
        recommendations.append(
            f"Fix stale references: {', '.join(stale_paths)}"
        )

    if include_cycles:
        cycle_info = report["inferred_facets"].get("cycles", {})
        if cycle_info.get("count", 0) > 0:
            recommendations.append(
                f"Found {cycle_info['count']} circular dependencies. "
                "Consider adding import guidelines to rules."
            )
        else:
            recommendations.append("Clean import structure (no circular dependencies).")

    if include_duplicates:
        dup_info = report["inferred_facets"].get("duplicates", {})
        if dup_info.get("count", 0) > 5:
            recommendations.append(
                f"Found {dup_info['count']} duplicate code pairs. "
                "Consider documenting shared patterns or refactoring."
            )

    report["recommendations"] = recommendations

    # Add summary stats
    report["summary"] = {
        "rules_count": len(existing_rules),
        "hotspots_count": total_hotspots,
        "covered_count": covered_count,
        "gaps_count": len(gaps),
        "stale_count": len(stale),
        "coverage_percent": round(covered_count / total_hotspots * 100, 1) if total_hotspots > 0 else 0,
    }

    return json.dumps(report, indent=2)
