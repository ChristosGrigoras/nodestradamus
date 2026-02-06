"""Tests for analyze_python_deps.py."""

from pathlib import Path

import pytest

from scripts.analyze_python_deps import (
    analyze_directory,
    analyze_file,
    extract_calls,
)


class TestExtractCalls:
    """Tests for extract_calls function."""

    def test_extract_simple_function_call(self):
        """Test extracting simple function calls."""
        import ast
        code = "print('hello')"
        tree = ast.parse(code)
        calls = extract_calls(tree)
        assert "print" in calls

    def test_extract_method_call(self):
        """Test extracting method calls."""
        import ast
        code = "obj.method()"
        tree = ast.parse(code)
        calls = extract_calls(tree)
        assert "method" in calls

    def test_extract_multiple_calls(self):
        """Test extracting multiple function calls."""
        import ast
        code = """
def foo():
    bar()
    baz()
    obj.method()
"""
        tree = ast.parse(code)
        calls = extract_calls(tree)
        assert "bar" in calls
        assert "baz" in calls
        assert "method" in calls


class TestAnalyzeFile:
    """Tests for analyze_file function."""

    def test_analyze_simple_file(self, sample_python_file: Path):
        """Test analyzing a simple Python file."""
        result = analyze_file(sample_python_file)

        assert "definitions" in result
        assert "edges" in result
        assert "error" not in result

        # Check that functions were found
        func_names = [d["name"] for d in result["definitions"]]
        assert any("hello_world" in name for name in func_names)
        assert any("greet" in name for name in func_names)
        assert any("main" in name for name in func_names)

        # Check that class was found
        assert any("UserService" in name for name in func_names)

    def test_analyze_file_with_syntax_error(self, temp_dir: Path):
        """Test handling files with syntax errors."""
        bad_file = temp_dir / "bad.py"
        bad_file.write_text("def broken(")

        result = analyze_file(bad_file)
        assert "error" in result

    def test_analyze_nonexistent_file(self, temp_dir: Path):
        """Test handling nonexistent files."""
        import pytest
        with pytest.raises(FileNotFoundError):
            analyze_file(temp_dir / "nonexistent.py")


class TestAnalyzeDirectory:
    """Tests for analyze_directory function."""

    def test_analyze_sample_directory(self):
        """Test analyzing the sample Python fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "sample_python"
        if not fixtures_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(fixtures_dir)

        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result
        assert len(result["nodes"]) > 0

    def test_analyze_empty_directory(self, temp_dir: Path):
        """Test analyzing an empty directory."""
        result = analyze_directory(temp_dir)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_skips_pycache(self, temp_dir: Path):
        """Test that __pycache__ is skipped."""
        pycache = temp_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "module.py").write_text("def foo(): pass")

        result = analyze_directory(temp_dir)

        # Should not find any files in __pycache__
        assert all("__pycache__" not in node for node in result["nodes"])

    def test_metadata_includes_generator(self, temp_dir: Path):
        """Test that metadata includes generator info."""
        (temp_dir / "test.py").write_text("def foo(): pass")

        result = analyze_directory(temp_dir)

        assert "metadata" in result
        assert result["metadata"]["analyzer"] == "python"
        assert "generated_at" in result["metadata"]


class TestEdgeResolution:
    """Tests for edge resolution between definitions."""

    def test_resolves_internal_calls(self):
        """Test that calls to internal functions are resolved."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "sample_python"
        if not fixtures_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(fixtures_dir)

        # Check that at least some edges are resolved
        resolved_edges = [e for e in result["edges"] if e.get("resolved")]
        assert len(resolved_edges) > 0

    def test_unresolved_external_calls(self, temp_dir: Path):
        """Test that calls to external functions are not resolved."""
        (temp_dir / "test.py").write_text("""
import os

def foo():
    os.path.exists("test")
""")

        result = analyze_directory(temp_dir)

        # 'exists' is external, should not be resolved
        external_edges = [e for e in result["edges"] if not e.get("resolved")]
        assert len(external_edges) > 0
