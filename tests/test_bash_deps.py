"""Tests for Bash/Shell dependency analysis."""

from pathlib import Path

from nodestradamus.analyzers.code_parser import BASH_CONFIG, parse_file
from nodestradamus.analyzers.deps import BASH_EXTENSIONS, _detect_languages, analyze_deps

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_bash"


class TestBashLanguageDetection:
    """Tests for Bash language detection."""

    def test_detects_sh_files(self, tmp_path: Path) -> None:
        """Should detect .sh files as Bash."""
        (tmp_path / "script.sh").write_text("#!/bin/bash\necho hello")
        languages = _detect_languages(tmp_path)
        assert "bash" in languages

    def test_detects_bash_files(self, tmp_path: Path) -> None:
        """Should detect .bash files as Bash."""
        (tmp_path / "script.bash").write_text("#!/bin/bash\necho hello")
        languages = _detect_languages(tmp_path)
        assert "bash" in languages

    def test_bash_extensions_defined(self) -> None:
        """BASH_EXTENSIONS should include .sh and .bash."""
        assert ".sh" in BASH_EXTENSIONS
        assert ".bash" in BASH_EXTENSIONS

    def test_bash_config_exists(self) -> None:
        """BASH_CONFIG should be properly defined."""
        assert BASH_CONFIG.name == "bash"
        assert BASH_CONFIG.prefix == "sh"
        assert "function_definition" in BASH_CONFIG.function_types


class TestBashParsing:
    """Tests for Bash file parsing."""

    def test_parses_bash_functions(self) -> None:
        """Should extract function definitions from Bash scripts."""
        result = parse_file(FIXTURE_DIR / "main.sh", FIXTURE_DIR)
        function_nodes = [n.name for n in result.nodes if n.type == "function"]
        assert "setup_environment" in function_nodes
        assert "main" in function_nodes

    def test_parses_utils_functions(self) -> None:
        """Should extract multiple functions from utils.sh."""
        result = parse_file(FIXTURE_DIR / "utils.sh", FIXTURE_DIR)
        function_nodes = [n.name for n in result.nodes if n.type == "function"]
        assert "log_message" in function_nodes
        assert "check_dependency" in function_nodes
        assert "get_script_dir" in function_nodes

    def test_extracts_source_imports(self) -> None:
        """Should extract source command imports."""
        result = parse_file(FIXTURE_DIR / "main.sh", FIXTURE_DIR)
        import_edges = [e for e in result.edges if e.type == "imports"]
        import_sources = [e.target for e in import_edges]

        # Should find the resolved utils.sh import
        assert any("utils.sh" in src for src in import_sources)

    def test_extracts_dot_imports(self) -> None:
        """Should extract . (dot) command imports."""
        result = parse_file(FIXTURE_DIR / "main.sh", FIXTURE_DIR)
        import_edges = [e for e in result.edges if e.type == "imports"]
        import_sources = [e.target for e in import_edges]

        # Should find the lib/helpers.sh import (may be unresolved if file doesn't exist)
        assert any("helpers.sh" in src for src in import_sources)

    def test_creates_module_node(self) -> None:
        """Should create a module node for the script file."""
        result = parse_file(FIXTURE_DIR / "main.sh", FIXTURE_DIR)
        module_nodes = [n for n in result.nodes if n.type == "module"]
        assert len(module_nodes) == 1
        assert module_nodes[0].language == "bash"

    def test_function_containment_edges(self) -> None:
        """Should create containment edges for functions."""
        result = parse_file(FIXTURE_DIR / "utils.sh", FIXTURE_DIR)
        contains_edges = [e for e in result.edges if e.type == "contains"]
        defined_in_edges = [e for e in result.edges if e.type == "defined_in"]

        # Should have containment edges for each function
        assert len(contains_edges) >= 3  # At least 3 functions
        assert len(defined_in_edges) >= 3


class TestBashAnalyzeDeps:
    """Integration tests for Bash dependency analysis."""

    def test_analyzes_bash_repository(self) -> None:
        """Should build dependency graph from Bash files."""
        graph = analyze_deps(FIXTURE_DIR, languages=["bash"])
        node_names = {data.get("name") for _, data in graph.nodes(data=True)}

        # Should contain functions from our fixtures
        assert "log_message" in node_names
        assert "setup_environment" in node_names

    def test_bash_graph_has_functions(self) -> None:
        """Graph should contain function nodes."""
        graph = analyze_deps(FIXTURE_DIR, languages=["bash"])
        function_nodes = [
            n for n, data in graph.nodes(data=True)
            if data.get("type") == "function"
        ]
        assert len(function_nodes) >= 5  # Multiple functions across files

    def test_bash_graph_has_imports(self) -> None:
        """Graph should contain import edges."""
        graph = analyze_deps(FIXTURE_DIR, languages=["bash"])
        import_edges = [
            (u, v) for u, v, data in graph.edges(data=True)
            if data.get("type") == "imports"
        ]
        # Should have at least some import edges from source commands
        assert len(import_edges) >= 1
