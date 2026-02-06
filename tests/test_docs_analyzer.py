"""Tests for documentation analyzer."""

from pathlib import Path

import pytest

from nodestradamus.analyzers.docs import (
    DocReference,
    analyze_docs,
    extract_doc_references,
    validate_references,
)


@pytest.fixture
def sample_docs_dir() -> Path:
    """Return path to sample docs fixture."""
    return Path(__file__).parent / "fixtures" / "sample_docs"


class TestExtractDocReferences:
    """Tests for extract_doc_references function."""

    def test_extracts_inline_code(self, sample_docs_dir: Path) -> None:
        """Test extraction of inline code references."""
        readme = sample_docs_dir / "README.md"
        refs = extract_doc_references(readme, sample_docs_dir)

        # Should find inline code like `main.py`, `process_data`
        inline_refs = [r for r in refs if r.ref_type == "inline_code"]
        assert len(inline_refs) > 0

    def test_extracts_file_links(self, sample_docs_dir: Path) -> None:
        """Test extraction of markdown file links."""
        readme = sample_docs_dir / "README.md"
        refs = extract_doc_references(readme, sample_docs_dir)

        # Should find [utils.py](utils.py)
        file_links = [r for r in refs if r.ref_type == "file_link"]
        assert any("utils.py" in r.raw_text for r in file_links)

    def test_extracts_code_blocks_with_paths(self, sample_docs_dir: Path) -> None:
        """Test extraction of code block file references."""
        api_doc = sample_docs_dir / "api.md"
        refs = extract_doc_references(api_doc, sample_docs_dir)

        # Should find ```python:main.py and ```1:10:main.py
        code_blocks = [r for r in refs if r.ref_type == "code_block"]
        assert any("main.py" in r.raw_text for r in code_blocks)

    def test_extracts_function_refs(self, sample_docs_dir: Path) -> None:
        """Test extraction of function call references."""
        readme = sample_docs_dir / "README.md"
        refs = extract_doc_references(readme, sample_docs_dir)

        # Should find `process_data()`
        func_refs = [r for r in refs if r.ref_type == "function_ref"]
        assert any("process_data" in r.raw_text for r in func_refs)


class TestValidateReferences:
    """Tests for validate_references function."""

    def test_validates_existing_files(self, sample_docs_dir: Path) -> None:
        """Test that existing files are marked as valid."""
        refs = [
            DocReference(
                doc_file="README.md",
                line=10,
                ref_type="file_link",
                raw_text="utils.py",
            ),
            DocReference(
                doc_file="README.md",
                line=15,
                ref_type="file_link",
                raw_text="services/user.py",
            ),
        ]

        validated = validate_references(refs, sample_docs_dir)

        assert all(r.exists for r in validated)
        assert validated[0].resolved_to == "utils.py"
        assert validated[1].resolved_to == "services/user.py"

    def test_marks_missing_files_as_stale(self, sample_docs_dir: Path) -> None:
        """Test that missing files are marked as not existing."""
        refs = [
            DocReference(
                doc_file="stale.md",
                line=5,
                ref_type="file_link",
                raw_text="deleted_file.py",
            ),
            DocReference(
                doc_file="stale.md",
                line=10,
                ref_type="file_link",
                raw_text="nonexistent.rs",
            ),
        ]

        validated = validate_references(refs, sample_docs_dir)

        assert not any(r.exists for r in validated)

    def test_validates_symbols_against_set(self, sample_docs_dir: Path) -> None:
        """Test symbol validation against known symbols."""
        refs = [
            DocReference(
                doc_file="README.md",
                line=10,
                ref_type="function_ref",
                raw_text="process_data()",
            ),
            DocReference(
                doc_file="README.md",
                line=15,
                ref_type="class_ref",
                raw_text="DataProcessor",
            ),
        ]

        symbols = {"process_data", "DataProcessor", "validate_input"}
        validated = validate_references(refs, sample_docs_dir, code_symbols=symbols)

        assert all(r.exists for r in validated)

    def test_marks_unknown_symbols_as_invalid(self, sample_docs_dir: Path) -> None:
        """Test that unknown symbols are marked as not existing."""
        refs = [
            DocReference(
                doc_file="stale.md",
                line=10,
                ref_type="function_ref",
                raw_text="deleted_function()",
            ),
        ]

        symbols = {"process_data", "validate_input"}
        validated = validate_references(refs, sample_docs_dir, code_symbols=symbols)

        assert not validated[0].exists


class TestAnalyzeDocs:
    """Tests for analyze_docs function."""

    def test_finds_stale_references(self, sample_docs_dir: Path) -> None:
        """Test that stale references are detected."""
        result = analyze_docs(sample_docs_dir)

        # stale.md has broken references
        assert result.total_references > 0
        assert len(result.stale_references) > 0

        # Should find the deleted_file.py reference
        stale_files = [r.raw_text for r in result.stale_references]
        assert any("deleted" in f or "nonexistent" in f for f in stale_files)

    def test_counts_valid_references(self, sample_docs_dir: Path) -> None:
        """Test that valid references are counted."""
        result = analyze_docs(sample_docs_dir)

        # Should have some valid references (utils.py, main.py, etc.)
        assert result.valid_references > 0
        assert result.valid_references <= result.total_references

    def test_includes_metadata(self, sample_docs_dir: Path) -> None:
        """Test that metadata is included in result."""
        result = analyze_docs(sample_docs_dir)

        assert "doc_files" in result.metadata
        assert len(result.metadata["doc_files"]) > 0

    def test_respects_include_readme_flag(self, sample_docs_dir: Path) -> None:
        """Test that include_readme flag is respected."""
        # With README
        result_with = analyze_docs(sample_docs_dir, include_readme=True)

        # Without README (still includes docs in sample_docs/)
        result_without = analyze_docs(sample_docs_dir, include_readme=False)

        # Both should find docs since sample_docs has markdown files
        assert result_with.total_docs > 0
        assert result_without.total_docs > 0


class TestIntegration:
    """Integration tests for the docs analyzer."""

    def test_full_analysis_on_sample_docs(self, sample_docs_dir: Path) -> None:
        """Run full analysis on sample docs fixture."""
        result = analyze_docs(sample_docs_dir)

        # Should analyze multiple doc files
        assert result.total_docs >= 3  # README.md, api.md, stale.md

        # Should find references
        assert result.total_references > 5

        # Should detect stale refs from stale.md
        stale_count = len(result.stale_references)
        assert stale_count > 0

        # Coverage should be calculated
        assert 0 <= result.coverage <= 100

    def test_handles_empty_docs_directory(self, tmp_path: Path) -> None:
        """Test handling of directory with no docs."""
        result = analyze_docs(tmp_path)

        assert result.total_docs == 0
        assert result.total_references == 0
        assert len(result.stale_references) == 0

    def test_handles_docs_with_only_valid_refs(self, tmp_path: Path) -> None:
        """Test docs with all valid references."""
        # Create a simple doc with valid reference
        (tmp_path / "main.py").write_text("def hello(): pass")
        (tmp_path / "README.md").write_text("See [main.py](main.py)")

        result = analyze_docs(tmp_path)

        assert result.total_docs == 1
        assert result.total_references >= 1
        assert result.valid_references >= 1
