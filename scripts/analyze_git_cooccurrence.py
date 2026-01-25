#!/usr/bin/env python3
"""
Analyze git history to find files that frequently change together.

Outputs a JSON co-occurrence graph for AI impact analysis.

Usage:
    python analyze_git_cooccurrence.py
    python analyze_git_cooccurrence.py --commits 200 > .cursor/graph/co-occurrence.json
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations


def get_commits(num_commits: int = 100) -> list[list[str]]:
    """Get list of files changed in each commit."""
    result = subprocess.run(
        ["git", "log", f"-{num_commits}", "--name-only", "--pretty=format:---COMMIT---"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error running git log: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    commits = []
    current_files = []

    for line in result.stdout.strip().split("\n"):
        if line == "---COMMIT---":
            if current_files:
                commits.append(current_files)
            current_files = []
        elif line.strip():
            current_files.append(line.strip())

    if current_files:
        commits.append(current_files)

    return commits


def calculate_cooccurrence(commits: list[list[str]], min_occurrences: int = 2) -> dict:
    """Calculate how often files change together."""
    pair_counts = defaultdict(int)
    file_counts = defaultdict(int)

    for files in commits:
        # Count individual file occurrences
        for f in files:
            file_counts[f] += 1

        # Count pairs (order doesn't matter)
        for pair in combinations(sorted(set(files)), 2):
            pair_counts[pair] += 1

    # Filter to pairs that occur together at least min_occurrences times
    significant_pairs = {
        pair: count
        for pair, count in pair_counts.items()
        if count >= min_occurrences
    }

    return {
        "pair_counts": significant_pairs,
        "file_counts": dict(file_counts),
    }


def build_graph(cooccurrence: dict, min_strength: float = 0.3) -> dict:
    """Build graph from co-occurrence data."""
    pair_counts = cooccurrence["pair_counts"]
    file_counts = cooccurrence["file_counts"]

    nodes = set()
    edges = []

    for (file_a, file_b), count in pair_counts.items():
        # Calculate strength: how often they change together vs separately
        # Jaccard-like: intersection / union
        total = file_counts[file_a] + file_counts[file_b] - count
        strength = count / total if total > 0 else 0

        if strength >= min_strength:
            nodes.add(file_a)
            nodes.add(file_b)
            edges.append({
                "from": file_a,
                "to": file_b,
                "type": "co-occurs",
                "count": count,
                "strength": round(strength, 3),
            })

    # Sort edges by strength
    edges.sort(key=lambda e: e["strength"], reverse=True)

    return {
        "nodes": sorted(nodes),
        "edges": edges,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "analyze_git_cooccurrence.py",
            "total_commits_analyzed": len(cooccurrence.get("commits", [])),
            "min_strength_threshold": min_strength,
        },
    }


def main():
    num_commits = 100
    min_strength = 0.3

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--commits" and i + 1 < len(args):
            num_commits = int(args[i + 1])
            i += 2
        elif args[i] == "--min-strength" and i + 1 < len(args):
            min_strength = float(args[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            print("Usage: python analyze_git_cooccurrence.py [--commits N] [--min-strength F]", file=sys.stderr)
            sys.exit(1)

    commits = get_commits(num_commits)
    cooccurrence = calculate_cooccurrence(commits)
    cooccurrence["commits"] = commits
    graph = build_graph(cooccurrence, min_strength)

    print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    main()
