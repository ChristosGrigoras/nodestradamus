"""Tests for validate_rules.py."""

from pathlib import Path

import pytest

from nodestradamus.mcp.tools.handlers.rules_validation import (
    _estimate_tokens as estimate_tokens,
)
from nodestradamus.mcp.tools.handlers.rules_validation import (
    _extract_rule_number as extract_rule_number,
)
from nodestradamus.mcp.tools.handlers.rules_validation import (
    _parse_frontmatter as parse_frontmatter,
)
from nodestradamus.mcp.tools.handlers.rules_validation import (
    _validate_rule_file as validate_rule_file,
)
from nodestradamus.mcp.tools.handlers.rules_validation import (
    _validate_rules_directory as validate_rules_directory,
)


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_short_text(self):
        """Test token estimation for short text."""
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text)
        assert tokens == 2  # 11 // 4 = 2

    def test_longer_text(self):
        """Test token estimation for longer text."""
        text = "This is a longer piece of text for testing."
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestExtractRuleNumber:
    """Tests for extract_rule_number function."""

    def test_standard_format(self):
        """Test extracting number from standard format."""
        assert extract_rule_number("100-python") == 100
        assert extract_rule_number("001-router") == 1
        assert extract_rule_number("305-dependency-graph") == 305

    def test_no_number(self):
        """Test handling files without numbers."""
        assert extract_rule_number("readme") is None
        assert extract_rule_number("python-rules") is None

    def test_number_only(self):
        """Test files with only numbers."""
        assert extract_rule_number("100") == 100


class TestParseFrontmatter:
    """Tests for parse_frontmatter function."""

    def test_valid_frontmatter(self):
        """Test parsing valid frontmatter."""
        content = '''---
description: Test rule
globs: "**/*.py"
alwaysApply: false
---

# Test Rule

Content here.
'''
        frontmatter, body, errors = parse_frontmatter(content)

        assert frontmatter is not None
        assert frontmatter["description"] == "Test rule"
        assert frontmatter["globs"] == "**/*.py"
        assert frontmatter["alwaysApply"] is False
        assert "# Test Rule" in body
        assert len(errors) == 0

    def test_missing_frontmatter(self):
        """Test handling missing frontmatter."""
        content = "# Just content\n\nNo frontmatter here."

        frontmatter, body, errors = parse_frontmatter(content)

        assert frontmatter is None
        assert len(errors) > 0
        assert any("frontmatter" in e.lower() for e in errors)

    def test_missing_description(self):
        """Test handling missing description field."""
        content = '''---
globs: "**/*"
---

# Content
'''
        frontmatter, body, errors = parse_frontmatter(content)

        assert any("description" in e.lower() for e in errors)

    def test_boolean_parsing(self):
        """Test parsing boolean values."""
        content = '''---
description: Test
alwaysApply: true
---

Content
'''
        frontmatter, body, errors = parse_frontmatter(content)

        assert frontmatter["alwaysApply"] is True


class TestValidateRuleFile:
    """Tests for validate_rule_file function."""

    def test_valid_rule_file(self, sample_rules_dir: Path):
        """Test validating a valid rule file."""
        rule_file = sample_rules_dir / "100-python.mdc"
        base_dir = sample_rules_dir.parent.parent

        result = validate_rule_file(rule_file, sample_rules_dir, base_dir)

        assert result["file"] == "100-python.mdc"
        assert len(result["errors"]) == 0
        assert result["info"]["rule_number"] == 100

    def test_missing_frontmatter(self, temp_dir: Path):
        """Test validating file without frontmatter."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        bad_file = rules_dir / "100-bad.mdc"
        bad_file.write_text("# No frontmatter\n\nJust content.")

        result = validate_rule_file(bad_file, rules_dir, temp_dir)

        assert len(result["errors"]) > 0

    def test_token_estimation(self, sample_rules_dir: Path):
        """Test that token count is included in info."""
        rule_file = sample_rules_dir / "100-python.mdc"
        base_dir = sample_rules_dir.parent.parent

        result = validate_rule_file(rule_file, sample_rules_dir, base_dir)

        assert "estimated_tokens" in result["info"]
        assert result["info"]["estimated_tokens"] > 0


class TestValidateRulesDirectory:
    """Tests for validate_rules_directory function."""

    def test_validate_sample_rules(self, sample_rules_dir: Path):
        """Test validating sample rules directory."""
        results = validate_rules_directory(sample_rules_dir)

        assert results["total_files"] == 2
        assert "files" in results
        assert "summary" in results

    def test_duplicate_detection(self, temp_dir: Path):
        """Test detection of duplicate rule numbers."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        # Create two files with same rule number
        for name in ["100-python.mdc", "100-javascript.mdc"]:
            (rules_dir / name).write_text('''---
description: Test
globs: "**/*"
---

# Test
''')

        results = validate_rules_directory(rules_dir)

        # Should detect duplicate
        assert results["errors"] > 0

        # Check for duplicate error message
        all_errors = []
        for f in results["files"]:
            all_errors.extend(f["errors"])
        assert any("duplicate" in e.lower() for e in all_errors)

    def test_empty_directory(self, temp_dir: Path):
        """Test validating empty rules directory."""
        rules_dir = temp_dir / "empty_rules"
        rules_dir.mkdir()

        results = validate_rules_directory(rules_dir)

        assert results["total_files"] == 0
        assert results["valid"] == 0

    def test_summary_statistics(self, sample_rules_dir: Path):
        """Test that summary includes proper statistics."""
        results = validate_rules_directory(sample_rules_dir)

        summary = results["summary"]
        assert "total" in summary
        assert "valid" in summary
        assert "with_warnings" in summary
        assert "with_errors" in summary
        assert "rule_number_range" in summary


class TestIntegration:
    """Integration tests for the validation system."""

    def test_validate_real_rules(self):
        """Test validating the actual project rules."""
        rules_dir = Path(__file__).parent.parent / ".cursor" / "rules"
        if not rules_dir.exists():
            pytest.skip("Project rules directory not found")

        results = validate_rules_directory(rules_dir)

        # Should find some rules
        assert results["total_files"] > 0

        # Should have proper structure
        assert "files" in results
        assert "summary" in results
