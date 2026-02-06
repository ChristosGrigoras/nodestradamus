"""Git co-occurrence analyzer.

Analyzes git history to find files that frequently change together.
Returns a NetworkX undirected weighted graph.
"""

from pathlib import Path

import networkx as nx

from scripts.analyze_git_cooccurrence import GitError, analyze_cooccurrence


class AnalysisError(Exception):
    """Error during code analysis."""

    pass


def analyze_git_cooccurrence(
    repo_path: str | Path,
    commits: int = 500,
    min_strength: float = 0.3,
) -> nx.Graph:
    """Analyze git history to find files that change together.

    Args:
        repo_path: Absolute path to repository root.
        commits: Number of recent commits to analyze.
        min_strength: Minimum Jaccard strength to include an edge.

    Returns:
        NetworkX undirected Graph with weighted edges.

    Raises:
        AnalysisError: If git analysis fails.
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.is_dir():
        raise ValueError(f"Not a directory: {repo_path}")

    try:
        raw_result = analyze_cooccurrence(
            repo_path,
            num_commits=commits,
            min_strength=min_strength,
        )
    except GitError as e:
        raise AnalysisError(f"Git analysis failed: {e}") from e

    return _build_graph(raw_result["nodes"], raw_result["edges"])


def _build_graph(
    nodes: list[str],
    edges: list[dict],
) -> nx.Graph:
    """Convert raw co-occurrence data to NetworkX graph.

    Args:
        nodes: List of file paths.
        edges: List of edge dicts with from, to, count, strength.

    Returns:
        NetworkX undirected Graph with weighted edges.
    """
    G = nx.Graph()

    for node in nodes:
        G.add_node(node, type="file")

    for edge in edges:
        source = edge.get("from", edge.get("source", ""))
        target = edge.get("to", edge.get("target", ""))

        if source and target:
            G.add_edge(
                source,
                target,
                count=edge.get("count", 1),
                strength=edge.get("strength", 0.0),
            )

    return G


def cooccurrence_metadata(G: nx.Graph) -> dict:
    """Extract summary metadata from co-occurrence graph.

    Args:
        G: Co-occurrence graph.

    Returns:
        Dict with file counts and edge statistics.
    """
    if G.number_of_edges() == 0:
        return {
            "file_count": G.number_of_nodes(),
            "edge_count": 0,
            "avg_strength": 0.0,
            "max_strength": 0.0,
        }

    strengths = [d.get("strength", 0.0) for _, _, d in G.edges(data=True)]

    return {
        "file_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "avg_strength": sum(strengths) / len(strengths),
        "max_strength": max(strengths),
    }
