#!/usr/bin/env python3
"""Benchmark graph algorithms: Rust vs NetworkX.

Compares performance of Rust-accelerated graph algorithms against
pure NetworkX implementations on real codebases.

Usage:
    python scripts/benchmark_graph.py /path/to/repo
    python scripts/benchmark_graph.py /tmp/test-repos/langchain
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodestradamus.analyzers import graph_algorithms
from nodestradamus.analyzers.deps import analyze_deps


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def benchmark_function(func, *args, **kwargs) -> tuple[float, any]:
    """Run a function and return (elapsed_time, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def run_benchmarks(repo_path: str, iterations: int = 3) -> dict:
    """Run benchmarks on a repository.

    Args:
        repo_path: Path to the repository to analyze.
        iterations: Number of iterations for timing (default 3).

    Returns:
        Dict with benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {repo_path}")
    print(f"{'='*60}\n")

    # Check Rust availability
    rust_available = graph_algorithms._HAS_RUST
    print(f"Rust backend available: {rust_available}")
    if not rust_available:
        print("⚠️  Install Rust extension with: maturin develop --release")
        print()

    # Step 1: Build dependency graph
    print("Building dependency graph...")
    start = time.perf_counter()
    graph = analyze_deps(repo_path)
    build_time = time.perf_counter() - start
    print(f"  Graph built in {format_time(build_time)}")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print()

    results = {
        "repo_path": repo_path,
        "rust_available": rust_available,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "graph_build_time": build_time,
        "benchmarks": {},
    }

    # Define algorithms to benchmark (Rust-accelerated when available)
    algorithms = [
        ("pagerank", lambda g: graph_algorithms.pagerank(g)),
        ("strongly_connected", lambda g: graph_algorithms.strongly_connected(g)),
        ("ancestors_at_depth", lambda g: graph_algorithms.ancestors_at_depth(
            g, list(g.nodes())[0] if g.nodes() else "", max_depth=3
        )),
        ("descendants_at_depth", lambda g: graph_algorithms.descendants_at_depth(
            g, list(g.nodes())[0] if g.nodes() else "", max_depth=3
        )),
        ("betweenness", lambda g: graph_algorithms.betweenness(g)),
    ]

    # NetworkX-only algorithms (not accelerated)
    networkx_only = [
        ("detect_communities", lambda g: graph_algorithms.detect_communities(g)),
        ("find_cycles", lambda g: graph_algorithms.find_cycles(g)),
    ]

    print("Running benchmarks...")
    print(f"{'Algorithm':<25} {'Rust':<12} {'NetworkX':<12} {'Speedup':<10}")
    print("-" * 60)

    for name, func in algorithms:
        # Benchmark with Rust (if available)
        rust_times = []
        if rust_available:
            graph_algorithms._HAS_RUST = True
            for _ in range(iterations):
                elapsed, _ = benchmark_function(func, graph)
                rust_times.append(elapsed)
            rust_avg = sum(rust_times) / len(rust_times)
        else:
            rust_avg = None

        # Benchmark with NetworkX
        graph_algorithms._HAS_RUST = False
        nx_times = []
        for _ in range(iterations):
            elapsed, _ = benchmark_function(func, graph)
            nx_times.append(elapsed)
        nx_avg = sum(nx_times) / len(nx_times)

        # Restore Rust state
        graph_algorithms._HAS_RUST = rust_available

        # Calculate speedup
        if rust_avg and nx_avg > 0:
            speedup = nx_avg / rust_avg
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup = None
            speedup_str = "N/A"

        rust_str = format_time(rust_avg) if rust_avg else "N/A"
        nx_str = format_time(nx_avg)

        print(f"{name:<25} {rust_str:<12} {nx_str:<12} {speedup_str:<10}")

        results["benchmarks"][name] = {
            "rust_time": rust_avg,
            "networkx_time": nx_avg,
            "speedup": speedup,
            "accelerated": True,
        }

    print()
    print("NetworkX-only algorithms (not Rust-accelerated):")
    print(f"{'Algorithm':<25} {'Time':<12}")
    print("-" * 40)

    for name, func in networkx_only:
        times = []
        for _ in range(iterations):
            elapsed, _ = benchmark_function(func, graph)
            times.append(elapsed)
        avg = sum(times) / len(times)
        print(f"{name:<25} {format_time(avg):<12}")

        results["benchmarks"][name] = {
            "rust_time": None,
            "networkx_time": avg,
            "speedup": None,
            "accelerated": False,
        }

    print()

    # Summary
    if rust_available:
        accelerated = [b for b in results["benchmarks"].values() if b["accelerated"] and b["speedup"]]
        if accelerated:
            avg_speedup = sum(b["speedup"] for b in accelerated) / len(accelerated)
            print(f"Average speedup (Rust vs NetworkX): {avg_speedup:.1f}x")
            print()

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark graph algorithms: Rust vs NetworkX"
    )
    parser.add_argument(
        "repo_path",
        help="Path to repository to analyze",
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=3,
        help="Number of iterations for timing (default: 3)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    if not Path(args.repo_path).exists():
        print(f"Error: {args.repo_path} does not exist")
        sys.exit(1)

    results = run_benchmarks(args.repo_path, args.iterations)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
