"""Tests for rule parser and compare_rules_to_codebase tool."""

import json
from pathlib import Path

import pytest

from nodestradamus.mcp.tools.utils.rule_parser import (
    ParsedRule,
    check_path_coverage,
    discover_rules,
    extract_code_paths,
    find_stale_references,
    parse_frontmatter,
)


class TestParseFrontmatter:
    """Tests for parse_frontmatter function."""

    def test_valid_cursor_frontmatter(self):
        """Test parsing valid Cursor-style frontmatter."""
        content = '''---
description: Python coding conventions
globs: "**/*.py"
alwaysApply: true
---

# Python

Use snake_case for functions.
'''
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["description"] == "Python coding conventions"
        assert frontmatter["globs"] == "**/*.py"
        assert frontmatter["alwaysApply"] is True
        assert "# Python" in body

    def test_claude_paths_frontmatter(self):
        """Test parsing Claude Code paths frontmatter."""
        content = '''---
paths: src/api/**/*.ts
---

# API Rules

Follow REST conventions.
'''
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["paths"] == "src/api/**/*.ts"
        assert "# API Rules" in body

    def test_no_frontmatter(self):
        """Test handling markdown with no frontmatter."""
        content = '''# AGENTS.md

This is an OpenCode rules file.
'''
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter == {}
        assert "# AGENTS.md" in body

    def test_boolean_values(self):
        """Test boolean value parsing."""
        content = '''---
alwaysApply: false
enabled: true
---

Content
'''
        frontmatter, _ = parse_frontmatter(content)

        assert frontmatter["alwaysApply"] is False
        assert frontmatter["enabled"] is True


class TestExtractCodePaths:
    """Tests for extract_code_paths function."""

    def test_backtick_paths(self):
        """Test extracting paths from backticks."""
        body = '''
Check `nodestradamus/analyzers/deps.py` for dependency analysis.
Also see `src/utils.ts` for utilities.
'''
        paths = extract_code_paths(body)

        assert "nodestradamus/analyzers/deps.py" in paths
        assert "src/utils.ts" in paths

    def test_directory_paths(self):
        """Test extracting directory paths."""
        body = '''
Files in `nodestradamus/` are the main package.
Tests are in `tests/`.
'''
        paths = extract_code_paths(body)

        assert "nodestradamus/" in paths
        assert "tests/" in paths

    def test_table_paths(self):
        """Test extracting paths from markdown tables."""
        body = '''
| File | Description |
|------|-------------|
| `nodestradamus/mcp/server.py` | MCP server |
| nodestradamus/analyzers/deps.py | Dependencies |
'''
        paths = extract_code_paths(body)

        assert "nodestradamus/mcp/server.py" in paths
        assert "nodestradamus/analyzers/deps.py" in paths

    def test_markdown_link_paths(self):
        """Test extracting paths from markdown links."""
        body = '''
See [handler](nodestradamus/mcp/tools/handlers/core.py) for details.
'''
        paths = extract_code_paths(body)

        assert "nodestradamus/mcp/tools/handlers/core.py" in paths

    def test_skips_urls(self):
        """Test that URLs are not extracted as paths."""
        body = '''
Visit https://example.com/path/file.py for docs.
See www.example.com for more.
'''
        paths = extract_code_paths(body)

        assert not any("example.com" in p for p in paths)

    def test_skips_version_numbers(self):
        """Test that version numbers are not extracted."""
        body = '''
Requires Python 3.12 or higher.
'''
        paths = extract_code_paths(body)

        assert "3.12" not in paths


class TestDiscoverRules:
    """Tests for discover_rules function."""

    def test_discover_cursor_rules(self, temp_dir: Path):
        """Test discovering Cursor-style rules."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        (rules_dir / "100-python.mdc").write_text('''---
description: Python rules
globs: "**/*.py"
---

# Python

Use snake_case.
Reference `src/utils.py` for utilities.
''')

        result = discover_rules(temp_dir)

        assert "cursor" in result.sources_checked
        assert "cursor" in result.sources_found
        assert len(result.rules) == 1
        assert result.rules[0].source == "cursor"
        assert result.rules[0].file == ".cursor/rules/100-python.mdc"
        assert "src/utils.py" in result.rules[0].code_paths

    def test_discover_opencode_rules(self, temp_dir: Path):
        """Test discovering OpenCode-style rules."""
        (temp_dir / "AGENTS.md").write_text('''# AGENTS.md

Project conventions for this repo.
Check `src/main.py` for entry point.
''')

        result = discover_rules(temp_dir, sources=["opencode"])

        assert "opencode" in result.sources_checked
        assert "opencode" in result.sources_found
        assert len(result.rules) == 1
        assert result.rules[0].source == "opencode"
        assert result.rules[0].file == "AGENTS.md"

    def test_discover_claude_rules(self, temp_dir: Path):
        """Test discovering Claude Code-style rules."""
        claude_dir = temp_dir / ".claude"
        claude_dir.mkdir()
        rules_dir = claude_dir / "rules"
        rules_dir.mkdir()

        (claude_dir / "CLAUDE.md").write_text('''# Project Rules

Main project rules.
''')

        (rules_dir / "code-style.md").write_text('''---
paths: src/**/*.ts
---

# Code Style

Use TypeScript.
''')

        result = discover_rules(temp_dir, sources=["claude"])

        assert "claude" in result.sources_checked
        assert "claude" in result.sources_found
        assert len(result.rules) == 2

    def test_no_rules_found(self, temp_dir: Path):
        """Test handling repos with no rules."""
        result = discover_rules(temp_dir)

        assert result.sources_checked == ["cursor", "opencode", "claude"]
        assert result.sources_found == []
        assert len(result.rules) == 0

    def test_custom_path(self, temp_dir: Path):
        """Test using custom rules path."""
        custom_dir = temp_dir / "my-rules"
        custom_dir.mkdir()

        (custom_dir / "rules.md").write_text('''# Custom Rules

My custom rules.
''')

        result = discover_rules(temp_dir, custom_path=str(custom_dir))

        assert len(result.rules) == 1
        assert "my-rules/rules.md" in result.rules[0].file


class TestCheckPathCoverage:
    """Tests for check_path_coverage function."""

    def test_path_in_body(self):
        """Test finding path mentioned in rule body."""
        rules = [
            ParsedRule(
                source="cursor",
                file="100-python.mdc",
                body="see nodestradamus/analyzers/deps.py for details",
                original_body="See nodestradamus/analyzers/deps.py for details",
                code_paths=["nodestradamus/analyzers/deps.py"],
            )
        ]

        coverage = check_path_coverage(rules, ["nodestradamus/analyzers/deps.py"])

        assert "nodestradamus/analyzers/deps.py" in coverage
        assert "100-python.mdc" in coverage["nodestradamus/analyzers/deps.py"]

    def test_parent_directory_coverage(self):
        """Test that parent directory mention covers child paths."""
        rules = [
            ParsedRule(
                source="cursor",
                file="100-project.mdc",
                body="all code in nodestradamus/ directory",
                original_body="All code in nodestradamus/ directory",
                code_paths=["nodestradamus/"],
            )
        ]

        coverage = check_path_coverage(rules, ["nodestradamus/analyzers/deps.py"])

        assert "nodestradamus/analyzers/deps.py" in coverage
        assert "100-project.mdc" in coverage["nodestradamus/analyzers/deps.py"]

    def test_path_not_covered(self):
        """Test finding uncovered paths."""
        rules = [
            ParsedRule(
                source="cursor",
                file="100-python.mdc",
                body="python conventions",
                original_body="Python conventions",
                code_paths=[],
            )
        ]

        coverage = check_path_coverage(rules, ["src/new_file.py"])

        assert "src/new_file.py" in coverage
        assert coverage["src/new_file.py"] == []


class TestFindStaleReferences:
    """Tests for find_stale_references function."""

    def test_finds_missing_file(self, temp_dir: Path):
        """Test finding references to non-existent files."""
        rules = [
            ParsedRule(
                source="cursor",
                file="100-python.mdc",
                body="see old_module.py for details",
                original_body="See old_module.py for details",
                code_paths=["old_module.py"],
            )
        ]

        stale = find_stale_references(rules, temp_dir)

        assert len(stale) == 1
        assert stale[0]["path"] == "old_module.py"
        assert stale[0]["rule"] == "100-python.mdc"

    def test_existing_file_not_stale(self, temp_dir: Path):
        """Test that existing files are not marked stale."""
        (temp_dir / "existing.py").write_text("# exists")

        rules = [
            ParsedRule(
                source="cursor",
                file="100-python.mdc",
                body="see existing.py",
                original_body="See existing.py",
                code_paths=["existing.py"],
            )
        ]

        stale = find_stale_references(rules, temp_dir)

        assert len(stale) == 0


class TestCompareRulesToCodebase:
    """Integration tests for compare_rules_to_codebase tool."""

    @pytest.mark.asyncio
    async def test_no_rules_returns_inferred_facets(self, temp_dir: Path):
        """Test that tool works when no rules exist."""
        from nodestradamus.mcp.tools.handlers.rules_audit import handle_compare_rules

        # Create a minimal Python file for analysis
        (temp_dir / "main.py").write_text('''
def hello():
    print("hello")

if __name__ == "__main__":
    hello()
''')

        result = await handle_compare_rules({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert "existing_rules_summary" in data
        assert data["existing_rules_summary"]["count"] == 0
        assert "message" in data["existing_rules_summary"]
        assert "inferred_facets" in data
        assert "recommendations" in data

    @pytest.mark.asyncio
    async def test_with_cursor_rules(self, temp_dir: Path):
        """Test tool with Cursor-style rules."""
        from nodestradamus.mcp.tools.handlers.rules_audit import handle_compare_rules

        # Create rules
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "100-project.mdc").write_text('''---
description: Project rules
globs: "**/*"
alwaysApply: true
---

# Project

Main module is `main.py`.
''')

        # Create code
        (temp_dir / "main.py").write_text('''
def main():
    print("main")

if __name__ == "__main__":
    main()
''')

        result = await handle_compare_rules({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert data["existing_rules_summary"]["count"] == 1
        assert "cursor" in data["existing_rules_summary"]["sources_found"]
        assert "summary" in data

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_nodestradamus_repo_integration(self):
        """Test tool on Nodestradamus repo itself.

        Note: This test is marked as slow because it analyzes the full Nodestradamus
        repo. Run with `pytest -m slow` to include it, or skip with default run.
        """
        from nodestradamus.mcp.tools.handlers.rules_audit import handle_compare_rules

        # Get repo path (parent of tests directory)
        repo_path = Path(__file__).parent.parent

        # Skip if not in expected location
        if not (repo_path / ".cursor" / "rules").exists():
            pytest.skip("Nodestradamus .cursor/rules not found")

        result = await handle_compare_rules({
            "repo_path": str(repo_path),
            "top_n": 10,
        })
        data = json.loads(result)

        # Should find Cursor rules
        assert data["existing_rules_summary"]["count"] > 0
        assert "cursor" in data["existing_rules_summary"]["sources_found"]

        # Should have inferred facets
        assert "inferred_facets" in data
        assert "critical_files" in data["inferred_facets"]
        assert "bottlenecks" in data["inferred_facets"]

        # Should have coverage analysis
        assert "coverage" in data
        assert "gaps" in data
        assert "summary" in data
