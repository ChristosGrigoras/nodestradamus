#!/usr/bin/env python3
"""
Analyze git history to find files that frequently change together.

Outputs a JSON co-occurrence graph for AI impact analysis.

Usage:
    python analyze_git_cooccurrence.py
    python analyze_git_cooccurrence.py --commits 200 > .cursor/graph/co-occurrence.json

This module can also be imported and used programmatically:
    from scripts.analyze_git_cooccurrence import analyze_cooccurrence
    result = analyze_cooccurrence(Path("/path/to/repo"), num_commits=500)
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any


class GitError(Exception):
    """Error running git commands."""

    pass


def get_commits(repo_path: Path, num_commits: int = 100) -> list[list[str]]:
    """Get list of files changed in each commit.

    Args:
        repo_path: Path to the git repository.
        num_commits: Number of recent commits to analyze.

    Returns:
        List of lists, where each inner list contains file paths changed in a commit.

    Raises:
        GitError: If git command fails.
    """
    result = subprocess.run(
        ["git", "log", f"-{num_commits}", "--name-only", "--pretty=format:---COMMIT---"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )

    if result.returncode != 0:
        raise GitError(f"git log failed: {result.stderr}")

    commits: list[list[str]] = []
    current_files: list[str] = []

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


def calculate_cooccurrence(commits: list[list[str]], min_occurrences: int = 2) -> dict[str, Any]:
    """Calculate how often files change together.

    Args:
        commits: List of commits, each containing list of changed files.
        min_occurrences: Minimum times a pair must co-occur to be included.

    Returns:
        Dictionary with pair_counts and file_counts.
    """
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    file_counts: defaultdict[str, int] = defaultdict(int)

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


def build_graph(
    cooccurrence: dict[str, Any],
    min_strength: float = 0.3,
    commits_analyzed: int = 0,
) -> dict[str, Any]:
    """Build graph from co-occurrence data.

    Args:
        cooccurrence: Output from calculate_cooccurrence.
        min_strength: Minimum Jaccard strength to include an edge.
        commits_analyzed: Number of commits that were analyzed.

    Returns:
        Graph dictionary with nodes, edges, and metadata.
    """
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
            "analyzer": "git_cooccurrence",
            "version": "0.2.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "commits_analyzed": commits_analyzed,
            "min_strength_threshold": min_strength,
        },
    }


def analyze_cooccurrence(
    repo_path: Path,
    num_commits: int = 500,
    min_strength: float = 0.3,
    min_occurrences: int = 2,
) -> dict[str, Any]:
    """Analyze git history to find files that change together.

    This is the main entry point for programmatic use.

    Args:
        repo_path: Path to the git repository.
        num_commits: Number of recent commits to analyze.
        min_strength: Minimum Jaccard strength to include an edge.
        min_occurrences: Minimum times a pair must co-occur.

    Returns:
        Co-occurrence graph with nodes, edges, and metadata.

    Raises:
        GitError: If git commands fail.
    """
    repo_path = Path(repo_path).resolve()
    commits = get_commits(repo_path, num_commits)
    cooccurrence = calculate_cooccurrence(commits, min_occurrences)
    return build_graph(cooccurrence, min_strength, commits_analyzed=len(commits))


def main() -> None:
    """CLI entry point for standalone usage."""
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

    try:
        graph = analyze_cooccurrence(Path.cwd(), num_commits, min_strength)
        print(json.dumps(graph, indent=2))
    except GitError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
