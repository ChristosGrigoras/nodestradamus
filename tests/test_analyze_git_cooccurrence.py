"""Tests for analyze_git_cooccurrence.py."""


from scripts.analyze_git_cooccurrence import (
    build_graph,
    calculate_cooccurrence,
)


class TestCalculateCooccurrence:
    """Tests for calculate_cooccurrence function."""

    def test_empty_commits(self):
        """Test with no commits."""
        result = calculate_cooccurrence([])

        assert result["pair_counts"] == {}
        assert result["file_counts"] == {}

    def test_single_file_commits(self):
        """Test with single-file commits."""
        commits = [
            ["file1.py"],
            ["file2.py"],
            ["file1.py"],
        ]

        result = calculate_cooccurrence(commits, min_occurrences=1)

        # No pairs since each commit has only one file
        assert result["pair_counts"] == {}
        assert result["file_counts"]["file1.py"] == 2
        assert result["file_counts"]["file2.py"] == 1

    def test_pair_counting(self):
        """Test counting file pairs that change together."""
        commits = [
            ["file1.py", "file2.py"],
            ["file1.py", "file2.py"],
            ["file1.py", "file3.py"],
        ]

        result = calculate_cooccurrence(commits, min_occurrences=1)

        # file1 and file2 appear together twice
        assert ("file1.py", "file2.py") in result["pair_counts"]
        assert result["pair_counts"][("file1.py", "file2.py")] == 2

        # file1 and file3 appear together once
        assert ("file1.py", "file3.py") in result["pair_counts"]
        assert result["pair_counts"][("file1.py", "file3.py")] == 1

    def test_min_occurrences_filter(self):
        """Test filtering by minimum occurrences."""
        commits = [
            ["file1.py", "file2.py"],
            ["file1.py", "file2.py"],
            ["file1.py", "file3.py"],  # Only once with file3
        ]

        result = calculate_cooccurrence(commits, min_occurrences=2)

        # Only file1-file2 pair meets threshold
        assert ("file1.py", "file2.py") in result["pair_counts"]
        assert ("file1.py", "file3.py") not in result["pair_counts"]

    def test_handles_duplicate_files_in_commit(self):
        """Test handling of duplicate files in a single commit."""
        commits = [
            ["file1.py", "file1.py", "file2.py"],
        ]

        result = calculate_cooccurrence(commits, min_occurrences=1)

        # File appears twice in list, gets counted twice by the current implementation
        # This is expected behavior - each line in git output is counted
        assert result["file_counts"]["file1.py"] == 2


class TestBuildGraph:
    """Tests for build_graph function."""

    def test_empty_cooccurrence(self):
        """Test building graph with empty data."""
        cooccurrence = {
            "pair_counts": {},
            "file_counts": {},
            "commits": [],
        }

        graph = build_graph(cooccurrence, min_strength=0.0)

        assert graph["nodes"] == []
        assert graph["edges"] == []

    def test_graph_structure(self):
        """Test that graph has correct structure."""
        cooccurrence = {
            "pair_counts": {("file1.py", "file2.py"): 5},
            "file_counts": {"file1.py": 5, "file2.py": 5},
            "commits": [],
        }

        graph = build_graph(cooccurrence, min_strength=0.0)

        assert "nodes" in graph
        assert "edges" in graph
        assert "metadata" in graph
        assert "file1.py" in graph["nodes"]
        assert "file2.py" in graph["nodes"]

    def test_strength_calculation(self):
        """Test edge strength calculation."""
        # Files always change together: strength = 1.0
        cooccurrence = {
            "pair_counts": {("file1.py", "file2.py"): 5},
            "file_counts": {"file1.py": 5, "file2.py": 5},
            "commits": [],
        }

        graph = build_graph(cooccurrence, min_strength=0.0)

        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["strength"] == 1.0

    def test_strength_filtering(self):
        """Test filtering edges by minimum strength."""
        # Low strength: files rarely change together
        cooccurrence = {
            "pair_counts": {("file1.py", "file2.py"): 1},
            "file_counts": {"file1.py": 10, "file2.py": 10},
            "commits": [],
        }

        # High threshold should filter out
        graph = build_graph(cooccurrence, min_strength=0.5)
        assert len(graph["edges"]) == 0

        # Low threshold should include
        graph = build_graph(cooccurrence, min_strength=0.0)
        assert len(graph["edges"]) == 1

    def test_edges_sorted_by_strength(self):
        """Test that edges are sorted by strength descending."""
        cooccurrence = {
            "pair_counts": {
                ("a.py", "b.py"): 1,
                ("c.py", "d.py"): 10,
            },
            "file_counts": {
                "a.py": 10,
                "b.py": 10,
                "c.py": 10,
                "d.py": 10,
            },
            "commits": [],
        }

        graph = build_graph(cooccurrence, min_strength=0.0)

        assert len(graph["edges"]) == 2
        # Higher strength edge should be first
        assert graph["edges"][0]["strength"] >= graph["edges"][1]["strength"]

    def test_metadata_included(self):
        """Test that metadata is included in output."""
        cooccurrence = {
            "pair_counts": {},
            "file_counts": {},
            "commits": [["a.py"], ["b.py"]],
        }

        graph = build_graph(cooccurrence, min_strength=0.0)

        assert "generated_at" in graph["metadata"]
        assert graph["metadata"]["analyzer"] == "git_cooccurrence"
