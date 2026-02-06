"""CLI interface for Nodestradamus.

Provides commands for running the MCP server and standalone analysis.
"""

import json
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env before importing other nodestradamus modules
# This ensures env vars are set before module-level code reads them
load_dotenv()

from nodestradamus import __version__  # noqa: E402


@click.group()
@click.version_option(version=__version__, prog_name="nodestradamus")
def cli() -> None:
    """Nodestradamus - MCP server for codebase intelligence."""
    pass


@cli.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport protocol (default: stdio)",
)
def serve(transport: str) -> None:
    """Start the MCP server.

    Runs the Nodestradamus MCP server using the specified transport.
    Default is stdio for use with Cursor and Claude Desktop.
    """
    if transport == "sse":
        click.echo("SSE transport not yet implemented", err=True)
        sys.exit(1)

    # Import here to avoid circular imports and slow startup
    from nodestradamus import run_server

    run_server()


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option(
    "--type",
    "analysis_type",
    type=click.Choice(["deps", "cooccurrence", "all"]),
    default="all",
    help="Type of analysis to run (default: all)",
)
@click.option(
    "--language",
    type=click.Choice(["python", "typescript", "rust", "auto"]),
    default="auto",
    help="Language to analyze (default: auto-detect)",
)
@click.option(
    "--commits",
    type=int,
    default=500,
    help="Number of commits for co-occurrence analysis (default: 500)",
)
def analyze(repo_path: str, analysis_type: str, language: str, commits: int) -> None:
    """Analyze a repository and output dependency graph.

    REPO_PATH: Path to the repository to analyze.
    """
    from nodestradamus.analyzers import analyze_deps, analyze_git_cooccurrence, graph_metadata

    repo = Path(repo_path)
    results: dict = {}

    if analysis_type in ("deps", "all"):
        try:
            languages = None if language == "auto" else [language]
            graph = analyze_deps(repo, languages=languages)
            metadata = graph_metadata(graph)
            results["deps"] = {
                "node_count": metadata["node_count"],
                "edge_count": metadata["edge_count"],
                "node_types": metadata["node_types"],
                "edge_types": metadata["edge_types"],
            }
        except Exception as e:
            click.echo(f"Dependency analysis failed: {e}", err=True)
            if analysis_type == "deps":
                sys.exit(1)

    if analysis_type in ("cooccurrence", "all"):
        try:
            graph = analyze_git_cooccurrence(repo, commits=commits)
            results["cooccurrence"] = json.loads(graph.model_dump_json(by_alias=True))
        except Exception as e:
            click.echo(f"Co-occurrence analysis failed: {e}", err=True)
            if analysis_type == "cooccurrence":
                sys.exit(1)

    click.echo(json.dumps(results, indent=2))


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
def scout(repo_path: str) -> None:
    """Quick reconnaissance of a repository.

    REPO_PATH: Path to the repository to analyze.

    Returns language distribution, key directories, frameworks,
    and recommended Nodestradamus tools.
    """
    from nodestradamus.analyzers import project_scout

    try:
        metadata = project_scout(repo_path)
        click.echo(metadata.model_dump_json(indent=2))
    except Exception as e:
        click.echo(f"Scout failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument("file_path")
@click.option("--symbol", help="Specific function/class name")
@click.option("--depth", type=int, default=3, help="Traversal depth (default: 3)")
def impact(repo_path: str, file_path: str, symbol: str | None, depth: int) -> None:
    """Analyze impact of changing a file.

    REPO_PATH: Path to the repository root.
    FILE_PATH: Relative path to the file being changed.
    """
    from nodestradamus.analyzers.impact import get_impact

    try:
        report = get_impact(repo_path, file_path, symbol=symbol, depth=depth)
        click.echo(report.model_dump_json(indent=2))
    except Exception as e:
        click.echo(f"Impact analysis failed: {e}", err=True)
        sys.exit(1)


def _short_name(node_id: str) -> str:
    """Extract short name from node ID like 'py:path/file.py::func'."""
    if "::" in node_id:
        parts = node_id.split("::")
        file_part = parts[0].replace("py:", "").replace("ts:", "").split("/")[-1]
        return f"{file_part}::{parts[-1]}"
    return node_id


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option(
    "--checks",
    type=click.Choice(["dead_code", "duplicates", "cycles", "bottlenecks", "all"]),
    multiple=True,
    default=["all"],
    help="Checks to run (default: all). Can specify multiple.",
)
@click.option(
    "--max-items",
    type=int,
    default=20,
    help="Maximum items per check (default: 20)",
)
@click.option(
    "--skip-duplicates",
    is_flag=True,
    help="Skip duplicate detection (slow for large repos)",
)
@click.option(
    "--skip-bottlenecks",
    is_flag=True,
    help="Skip bottleneck detection (slow for large repos)",
)
def health(
    repo_path: str,
    checks: tuple[str, ...],
    max_items: int,
    skip_duplicates: bool,
    skip_bottlenecks: bool,
) -> None:
    """Comprehensive codebase health check.

    REPO_PATH: Path to the repository to analyze.

    Runs multiple analyses and returns a unified health report including:
    - dead_code: Unused functions/classes
    - duplicates: Copy-pasted code (slow)
    - cycles: Circular dependencies
    - bottlenecks: High-impact nodes (slow)
    """
    from nodestradamus.analyzers import analyze_deps
    from nodestradamus.analyzers.dead_code import find_dead_code
    from nodestradamus.analyzers.embeddings import detect_duplicates
    from nodestradamus.analyzers.graph_algorithms import betweenness, find_cycles, top_n

    # Determine which checks to run
    if "all" in checks:
        check_list = ["dead_code", "duplicates", "cycles", "bottlenecks"]
    else:
        check_list = list(checks)

    # Apply skip flags
    if skip_duplicates and "duplicates" in check_list:
        check_list.remove("duplicates")
    if skip_bottlenecks and "bottlenecks" in check_list:
        check_list.remove("bottlenecks")

    click.echo(f"Running health checks: {', '.join(check_list)}", err=True)

    report: dict = {
        "summary": {},
        "findings": {},
    }

    # Build dependency graph
    click.echo("Building dependency graph...", err=True)
    try:
        G = analyze_deps(repo_path)
    except Exception as e:
        click.echo(f"Failed to build dependency graph: {e}", err=True)
        sys.exit(1)

    if G.number_of_nodes() == 0:
        click.echo(json.dumps({
            "error": "No nodes found in dependency graph",
            "summary": {"status": "empty"},
            "findings": {},
        }, indent=2))
        return

    click.echo(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", err=True)

    # Run requested checks
    if "dead_code" in check_list:
        click.echo("Checking for dead code...", err=True)
        try:
            dead = find_dead_code(G)
            report["findings"]["dead_code"] = dead[:max_items]
            report["summary"]["dead_code_count"] = len(dead)
        except Exception as e:
            click.echo(f"Dead code check failed: {e}", err=True)

    if "duplicates" in check_list:
        nodes = G.number_of_nodes()
        # Rough estimate: embedding + pairwise comparison
        estimated_secs = nodes * 0.05  # ~50ms per function to embed + compare
        if estimated_secs > 60:
            click.echo(
                f"Detecting duplicates (~{estimated_secs / 60:.1f} min for {nodes:,} code chunks)...",
                err=True,
            )
        else:
            click.echo(
                f"Detecting duplicates (~{estimated_secs:.0f}s for {nodes:,} code chunks)...",
                err=True,
            )
        try:
            start = time.time()
            duplicates = detect_duplicates(repo_path, threshold=0.9, max_pairs=max_items)
            elapsed = time.time() - start
            click.echo(f"Duplicate detection completed in {elapsed:.1f}s", err=True)
            report["findings"]["duplicates"] = duplicates
            report["summary"]["duplicate_pairs"] = len(duplicates)
        except Exception as e:
            click.echo(f"Duplicate detection failed: {e}", err=True)

    if "cycles" in check_list:
        click.echo("Finding circular dependencies...", err=True)
        try:
            cycles = find_cycles(G, cross_file_only=True)
            formatted_cycles = []
            for cycle in cycles[:max_items]:
                files_in_cycle = set()
                for node in cycle:
                    file_path = G.nodes[node].get("file", "")
                    if file_path:
                        files_in_cycle.add(file_path)
                formatted_cycles.append({
                    "length": len(cycle),
                    "files": sorted(files_in_cycle),
                    "cycle": [_short_name(node) for node in cycle],
                })
            report["findings"]["cycles"] = formatted_cycles
            report["summary"]["cycle_count"] = len(cycles)
        except Exception as e:
            click.echo(f"Cycle detection failed: {e}", err=True)

    if "bottlenecks" in check_list:
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        # Rough estimate: O(V*E) complexity, ~1M ops/sec on typical hardware
        estimated_ops = nodes * edges
        estimated_secs = estimated_ops / 1_000_000
        if estimated_secs > 60:
            click.echo(
                f"Calculating bottlenecks (~{estimated_secs / 60:.1f} min for {nodes:,} nodes × {edges:,} edges)...",
                err=True,
            )
        else:
            click.echo(
                f"Calculating bottlenecks (~{estimated_secs:.0f}s for {nodes:,} nodes × {edges:,} edges)...",
                err=True,
            )
        try:
            start = time.time()
            scores = betweenness(G)
            elapsed = time.time() - start
            click.echo(f"Bottleneck calculation completed in {elapsed:.1f}s", err=True)
            ranked = top_n(scores, n=max_items)
            bottlenecks = [
                {"node": _short_name(node), "full_id": node, "betweenness": round(score, 6)}
                for node, score in ranked
            ]
            report["findings"]["bottlenecks"] = bottlenecks
            report["summary"]["top_bottleneck"] = bottlenecks[0]["node"] if bottlenecks else None
        except Exception as e:
            click.echo(f"Bottleneck detection failed: {e}", err=True)

    # Calculate overall health score
    issues = (
        report["summary"].get("dead_code_count", 0)
        + report["summary"].get("duplicate_pairs", 0) * 2
        + report["summary"].get("cycle_count", 0) * 3
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
    report["summary"]["checks_run"] = check_list

    click.echo(json.dumps(report, indent=2))


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
