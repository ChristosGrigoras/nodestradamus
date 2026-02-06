"""Tests for detect_rule_conflicts.py."""

from pathlib import Path

import pytest

from nodestradamus.mcp.tools.handlers.rules_validation import (
    _check_glob_overlap as check_glob_overlap,
    _detect_category_conflicts as detect_conflicts,
    _detect_directive_conflicts as analyze_directive_conflicts,
    _find_category_mentions as find_category_mentions,
    _parse_rule_for_conflicts as parse_rule_file,
)


class TestParseRuleFile:
    """Tests for parse_rule_file function."""

    def test_parse_valid_rule(self, temp_dir: Path) -> None:
        """Test parsing a valid rule file."""
        rule_file = temp_dir / "100-test.mdc"
        rule_file.write_text('''---
description: Test rule
globs: "**/*.py"
---

# Test Rule

- Use snake_case for functions
- Add docstrings
''')
        result = parse_rule_file(rule_file)

        assert result is not None
        assert result["file"] == "100-test.mdc"
        assert result["frontmatter"]["description"] == "Test rule"
        assert result["globs"] == "**/*.py"
        assert len(result["directives"]) == 2

    def test_parse_missing_frontmatter(self, temp_dir: Path) -> None:
        """Test parsing file without frontmatter returns None."""
        rule_file = temp_dir / "bad.mdc"
        rule_file.write_text("# No frontmatter\n\nJust content.")

        result = parse_rule_file(rule_file)
        assert result is None

    def test_parse_nonexistent_file(self, temp_dir: Path) -> None:
        """Test parsing nonexistent file returns None."""
        result = parse_rule_file(temp_dir / "nonexistent.mdc")
        assert result is None

    def test_parse_default_globs(self, temp_dir: Path) -> None:
        """Test that missing globs defaults to **/*."""
        rule_file = temp_dir / "100-test.mdc"
        rule_file.write_text('''---
description: Test rule
---

# Test
''')
        result = parse_rule_file(rule_file)

        assert result is not None
        assert result["globs"] == "**/*"


class TestFindCategoryMentions:
    """Tests for find_category_mentions function."""

    def test_finds_naming_conventions(self) -> None:
        """Test finding naming convention keywords."""
        # Body must be lowercased (as parse_rule_file does)
        rule = {"body": "use snake_case for functions and pascalcase for classes"}

        mentions = find_category_mentions(rule)

        assert "naming_convention" in mentions
        assert "snake_case" in mentions["naming_convention"]
        assert "PascalCase" in mentions["naming_convention"]  # Original keyword preserved

    def test_finds_testing_keywords(self) -> None:
        """Test finding testing framework keywords."""
        rule = {"body": "use pytest for all tests"}

        mentions = find_category_mentions(rule)

        assert "testing" in mentions
        assert "pytest" in mentions["testing"]

    def test_no_mentions(self) -> None:
        """Test when no categories are mentioned."""
        rule = {"body": "generic content without keywords"}

        mentions = find_category_mentions(rule)

        assert mentions == {}

    def test_multiple_categories(self) -> None:
        """Test finding multiple category mentions."""
        rule = {"body": "use snake_case naming, pytest testing, and type hints"}

        mentions = find_category_mentions(rule)

        assert "naming_convention" in mentions
        assert "testing" in mentions
        assert "type_hints" in mentions


class TestCheckGlobOverlap:
    """Tests for check_glob_overlap function."""

    def test_universal_globs_overlap(self) -> None:
        """Test that universal globs always overlap."""
        assert check_glob_overlap("**/*", "*.py") is True
        assert check_glob_overlap("*.ts", "**/*") is True

    def test_same_extension_overlaps(self) -> None:
        """Test that same extensions overlap."""
        assert check_glob_overlap("*.py", "*.py") is True
        assert check_glob_overlap("src/*.py", "tests/*.py") is True

    def test_different_extensions_no_overlap(self) -> None:
        """Test that different extensions don't overlap."""
        assert check_glob_overlap("*.py", "*.ts") is False
        assert check_glob_overlap("*.js", "*.py") is False

    def test_no_extension_assumes_overlap(self) -> None:
        """Test that missing extension filter assumes overlap."""
        assert check_glob_overlap("src/*", "*.py") is True


class TestDetectConflicts:
    """Tests for detect_conflicts function."""

    def test_no_conflicts_different_categories(self) -> None:
        """Test no conflicts when rules address different categories."""
        rules = [
            {"file": "100-python.mdc", "body": "use snake_case", "globs": "**/*.py"},
            {"file": "101-testing.mdc", "body": "use pytest", "globs": "**/*.py"},
        ]

        conflicts = detect_conflicts(rules)
        assert len(conflicts) == 0

    def test_detects_naming_conflict(self) -> None:
        """Test detection of naming convention conflicts."""
        # Bodies must be lowercased (as parse_rule_file does)
        rules = [
            {"file": "100-python.mdc", "body": "use snake_case", "globs": "**/*"},
            {"file": "101-js.mdc", "body": "use camelcase", "globs": "**/*"},
        ]

        conflicts = detect_conflicts(rules)

        assert len(conflicts) == 1
        assert conflicts[0]["category"] == "naming_convention"
        assert conflicts[0]["severity"] == "warning"

    def test_no_conflict_non_overlapping_globs(self) -> None:
        """Test no conflict when globs don't overlap."""
        rules = [
            {"file": "100-python.mdc", "body": "use snake_case", "globs": "*.py"},
            {"file": "101-js.mdc", "body": "use camelCase", "globs": "*.js"},
        ]

        conflicts = detect_conflicts(rules)
        assert len(conflicts) == 0


class TestAnalyzeDirectiveConflicts:
    """Tests for analyze_directive_conflicts function."""

    def test_no_directive_conflicts(self) -> None:
        """Test when there are no directive conflicts."""
        rules = [
            {"file": "100-python.mdc", "body": "use snake_case for naming"},
            {"file": "101-js.mdc", "body": "follow coding standards"},
        ]

        conflicts = analyze_directive_conflicts(rules)
        assert len(conflicts) == 0

    def test_detects_naming_mismatch(self) -> None:
        """Test detection of snake_case vs camelCase mismatch."""
        rules = [
            {"file": "100-python.mdc", "body": "use snake_case for all functions"},
            {"file": "101-js.mdc", "body": "use camelCase for variables"},
        ]

        conflicts = analyze_directive_conflicts(rules)

        assert len(conflicts) == 1
        assert conflicts[0]["description"] == "Naming convention mismatch"
        assert conflicts[0]["severity"] == "error"

    def test_detects_test_framework_mismatch(self) -> None:
        """Test detection of pytest vs unittest mismatch."""
        rules = [
            {"file": "300-testing.mdc", "body": "use pytest for all tests"},
            {"file": "301-legacy.mdc", "body": "use unittest for integration"},
        ]

        conflicts = analyze_directive_conflicts(rules)

        assert len(conflicts) == 1
        assert conflicts[0]["description"] == "Test framework mismatch"

    def test_detects_import_style_mismatch(self) -> None:
        """Test detection of absolute vs relative import mismatch."""
        rules = [
            {"file": "100-imports.mdc", "body": "always use absolute import"},
            {"file": "101-local.mdc", "body": "prefer relative import for local"},
        ]

        conflicts = analyze_directive_conflicts(rules)

        assert len(conflicts) == 1
        assert conflicts[0]["description"] == "Import style mismatch"


class TestIntegration:
    """Integration tests for the conflict detection system."""

    def test_analyze_sample_rules(self, sample_rules_dir: Path) -> None:
        """Test analyzing sample rules directory."""
        rules = []
        for filepath in sorted(sample_rules_dir.glob("*.mdc")):
            parsed = parse_rule_file(filepath)
            if parsed:
                rules.append(parsed)

        # Should find at least the sample rules
        assert len(rules) >= 2

        # Run conflict detection
        category_conflicts = detect_conflicts(rules)
        directive_conflicts = analyze_directive_conflicts(rules)

        # Both should return lists (may or may not have conflicts)
        assert isinstance(category_conflicts, list)
        assert isinstance(directive_conflicts, list)

    def test_analyze_real_rules(self) -> None:
        """Test analyzing the actual project rules."""
        rules_dir = Path(__file__).parent.parent / ".cursor" / "rules"
        if not rules_dir.exists():
            pytest.skip("Project rules directory not found")

        rules = []
        for filepath in sorted(rules_dir.glob("*.mdc")):
            parsed = parse_rule_file(filepath)
            if parsed:
                rules.append(parsed)

        # Should find some rules
        assert len(rules) > 0

        # Run analysis - just verify it doesn't crash
        category_conflicts = detect_conflicts(rules)
        directive_conflicts = analyze_directive_conflicts(rules)

        assert isinstance(category_conflicts, list)
        assert isinstance(directive_conflicts, list)
