#!/usr/bin/env python3
"""Benchmark MCP tools vs direct library calls.

Measures:
1. Direct library function call time
2. MCP handler overhead (asyncio wrapping, JSON serialization)
3. Identifies which tools have the most overhead

Usage:
    python scripts/benchmark_mcp_tools.py /path/to/repo
    python scripts/benchmark_mcp_tools.py .  # Current directory
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodestradamus.analyzers import analyze_git_cooccurrence, project_scout
from nodestradamus.analyzers.deps import analyze_deps
from nodestradamus.analyzers.impact import get_impact
from nodestradamus.mcp.tools.handlers import (
    handle_analyze_cooccurrence,
    handle_analyze_deps,
    handle_get_impact,
    handle_project_scout,
)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def format_overhead(direct: float, mcp: float) -> str:
    """Format overhead percentage."""
    if direct == 0:
        return "N/A"
    overhead = ((mcp - direct) / direct) * 100
    return f"+{overhead:.1f}%"


async def benchmark_project_scout(repo_path: str, iterations: int = 3) -> dict:
    """Benchmark project_scout: direct vs MCP handler."""
    results = {"direct_times": [], "mcp_times": []}

    for _ in range(iterations):
        # Direct call
        start = time.perf_counter()
        _ = project_scout(repo_path)
        results["direct_times"].append(time.perf_counter() - start)

        # MCP handler call
        start = time.perf_counter()
        _ = await handle_project_scout({"repo_path": repo_path})
        results["mcp_times"].append(time.perf_counter() - start)

    results["direct_avg"] = sum(results["direct_times"]) / len(results["direct_times"])
    results["mcp_avg"] = sum(results["mcp_times"]) / len(results["mcp_times"])
    results["overhead"] = results["mcp_avg"] - results["direct_avg"]
    return results


async def benchmark_analyze_deps(repo_path: str, iterations: int = 3) -> dict:
    """Benchmark analyze_deps: direct vs MCP handler."""
    results = {"direct_times": [], "mcp_times": []}

    for _ in range(iterations):
        # Direct call
        start = time.perf_counter()
        _ = analyze_deps(repo_path)
        results["direct_times"].append(time.perf_counter() - start)

        # MCP handler call
        start = time.perf_counter()
        _ = await handle_analyze_deps({"repo_path": repo_path})
        results["mcp_times"].append(time.perf_counter() - start)

    results["direct_avg"] = sum(results["direct_times"]) / len(results["direct_times"])
    results["mcp_avg"] = sum(results["mcp_times"]) / len(results["mcp_times"])
    results["overhead"] = results["mcp_avg"] - results["direct_avg"]
    return results


async def benchmark_analyze_cooccurrence(repo_path: str, iterations: int = 3) -> dict:
    """Benchmark analyze_cooccurrence: direct vs MCP handler.

    Note: MCP handler does additional summarization work, so overhead
    represents the cost of summarization (which adds value), not protocol overhead.
    """
    results = {"direct_times": [], "mcp_times": []}

    for _ in range(iterations):
        # Direct call (limited commits for speed)
        start = time.perf_counter()
        _ = analyze_git_cooccurrence(repo_path, commits=50)
        results["direct_times"].append(time.perf_counter() - start)

        # MCP handler call (includes summarization)
        start = time.perf_counter()
        _ = await handle_analyze_cooccurrence({"repo_path": repo_path, "commits": 50})
        results["mcp_times"].append(time.perf_counter() - start)

    results["direct_avg"] = sum(results["direct_times"]) / len(results["direct_times"])
    results["mcp_avg"] = sum(results["mcp_times"]) / len(results["mcp_times"])
    results["overhead"] = results["mcp_avg"] - results["direct_avg"]
    results["note"] = "MCP overhead includes summarization work (adds value)"
    return results


async def benchmark_get_impact(repo_path: str, iterations: int = 3) -> dict:
    """Benchmark get_impact: direct vs MCP handler."""
    results = {"direct_times": [], "mcp_times": []}

    # Find a Python file to analyze
    repo = Path(repo_path)
    py_files = list(repo.glob("**/*.py"))
    if not py_files:
        return {"skipped": True, "reason": "No Python files found"}

    # Use relative path from repo root
    target_file = str(py_files[0].relative_to(repo))

    for _ in range(iterations):
        # Direct call
        start = time.perf_counter()
        try:
            _ = get_impact(repo_path, target_file, depth=2)
        except Exception:
            pass  # File might not be in graph
        results["direct_times"].append(time.perf_counter() - start)

        # MCP handler call
        start = time.perf_counter()
        _ = await handle_get_impact(
            {"repo_path": repo_path, "file_path": target_file, "depth": 2}
        )
        results["mcp_times"].append(time.perf_counter() - start)

    results["direct_avg"] = sum(results["direct_times"]) / len(results["direct_times"])
    results["mcp_avg"] = sum(results["mcp_times"]) / len(results["mcp_times"])
    results["overhead"] = results["mcp_avg"] - results["direct_avg"]
    return results


async def benchmark_json_serialization(repo_path: str) -> dict:
    """Benchmark JSON serialization overhead separately."""
    # Build a graph once
    G = analyze_deps(repo_path)

    results = {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "serialization_times": [],
    }

    # Test JSON serialization time
    from nodestradamus.models.graph import digraph_to_json

    for _ in range(5):
        start = time.perf_counter()
        data = digraph_to_json(G)
        _ = json.dumps(data, indent=2)
        results["serialization_times"].append(time.perf_counter() - start)

    results["serialization_avg"] = (
        sum(results["serialization_times"]) / len(results["serialization_times"])
    )
    return results


async def benchmark_asyncio_overhead() -> dict:
    """Benchmark asyncio.to_thread overhead for trivial operations."""
    results = {"sync_times": [], "async_times": []}

    def trivial_sync():
        return sum(range(1000))

    for _ in range(10):
        # Direct sync call
        start = time.perf_counter()
        _ = trivial_sync()
        results["sync_times"].append(time.perf_counter() - start)

        # Wrapped in asyncio.to_thread
        start = time.perf_counter()
        _ = await asyncio.to_thread(trivial_sync)
        results["async_times"].append(time.perf_counter() - start)

    results["sync_avg"] = sum(results["sync_times"]) / len(results["sync_times"])
    results["async_avg"] = sum(results["async_times"]) / len(results["async_times"])
    results["overhead"] = results["async_avg"] - results["sync_avg"]
    return results


async def run_benchmarks(repo_path: str, iterations: int = 3) -> dict:
    """Run all benchmarks."""
    print(f"\n{'='*70}")
    print("MCP Tool Performance Benchmark")
    print(f"Repository: {repo_path}")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}\n")

    results = {"repo_path": repo_path, "iterations": iterations, "benchmarks": {}}

    # Asyncio overhead (baseline)
    print("Measuring asyncio.to_thread overhead...")
    async_overhead = await benchmark_asyncio_overhead()
    results["asyncio_overhead_us"] = async_overhead["overhead"] * 1_000_000
    print(f"  asyncio.to_thread overhead: {format_time(async_overhead['overhead'])}")
    print()

    # Tool benchmarks
    benchmarks = [
        ("project_scout", benchmark_project_scout),
        ("analyze_deps", benchmark_analyze_deps),
        ("analyze_cooccurrence", benchmark_analyze_cooccurrence),
        ("get_impact", benchmark_get_impact),
    ]

    print(f"{'Tool':<25} {'Direct':<12} {'MCP':<12} {'Overhead':<12} {'%':<10}")
    print("-" * 70)

    for name, bench_func in benchmarks:
        try:
            result = await bench_func(repo_path, iterations)

            if result.get("skipped"):
                print(f"{name:<25} SKIPPED: {result.get('reason', 'Unknown')}")
                continue

            direct_str = format_time(result["direct_avg"])
            mcp_str = format_time(result["mcp_avg"])
            overhead_str = format_time(result["overhead"])
            pct_str = format_overhead(result["direct_avg"], result["mcp_avg"])

            print(f"{name:<25} {direct_str:<12} {mcp_str:<12} {overhead_str:<12} {pct_str:<10}")

            results["benchmarks"][name] = {
                "direct_avg_ms": result["direct_avg"] * 1000,
                "mcp_avg_ms": result["mcp_avg"] * 1000,
                "overhead_ms": result["overhead"] * 1000,
                "overhead_pct": (
                    ((result["mcp_avg"] - result["direct_avg"]) / result["direct_avg"]) * 100
                    if result["direct_avg"] > 0
                    else 0
                ),
            }
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
            results["benchmarks"][name] = {"error": str(e)}

    # JSON serialization benchmark
    print()
    print("Measuring JSON serialization overhead...")
    json_result = await benchmark_json_serialization(repo_path)
    print(f"  Graph size: {json_result['graph_nodes']} nodes, {json_result['graph_edges']} edges")
    print(f"  JSON serialization: {format_time(json_result['serialization_avg'])}")
    results["json_serialization_ms"] = json_result["serialization_avg"] * 1000

    # Summary
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    total_overhead = sum(
        b.get("overhead_ms", 0)
        for b in results["benchmarks"].values()
        if isinstance(b, dict) and "overhead_ms" in b
    )
    avg_overhead_pct = sum(
        b.get("overhead_pct", 0)
        for b in results["benchmarks"].values()
        if isinstance(b, dict) and "overhead_pct" in b
    ) / max(len([b for b in results["benchmarks"].values() if isinstance(b, dict) and "overhead_pct" in b]), 1)

    print(f"Total MCP overhead across tools: {total_overhead:.1f}ms")
    print(f"Average overhead percentage: {avg_overhead_pct:.1f}%")
    print()

    # Identify bottlenecks
    print("Potential bottlenecks (overhead > 20%):")
    for name, data in results["benchmarks"].items():
        if isinstance(data, dict) and data.get("overhead_pct", 0) > 20:
            print(f"  - {name}: +{data['overhead_pct']:.1f}% overhead ({data['overhead_ms']:.1f}ms)")

    if json_result["serialization_avg"] > 0.1:  # More than 100ms
        print(f"  - JSON serialization: {format_time(json_result['serialization_avg'])} (consider streaming)")

    print()
    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark MCP tools vs direct library calls"
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=".",
        help="Path to repository to analyze (default: current directory)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for timing (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    repo_path = str(Path(args.repo_path).resolve())
    if not Path(repo_path).exists():
        print(f"Error: {repo_path} does not exist")
        sys.exit(1)

    results = asyncio.run(run_benchmarks(repo_path, args.iterations))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
