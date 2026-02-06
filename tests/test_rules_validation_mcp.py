"""Tests for validate_rules and detect_rule_conflicts MCP tools."""

import json
from pathlib import Path

import pytest

from nodestradamus.mcp.tools.handlers.rules_validation import (
    handle_detect_conflicts,
    handle_validate_rules,
)


class TestValidateRulesMCP:
    """Tests for validate_rules MCP tool handler."""

    @pytest.mark.asyncio
    async def test_no_rules_found(self, temp_dir: Path):
        """Test handling repos with no rules."""
        result = await handle_validate_rules({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "No rules found"
        assert "searched" in data
        assert "recommendation" in data

    @pytest.mark.asyncio
    async def test_validates_cursor_rules(self, temp_dir: Path):
        """Test validating Cursor-style rules."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        (rules_dir / "100-python.mdc").write_text('''---
description: Python rules
globs: "**/*.py"
---

# Python Coding Standards

Use snake_case for all function and variable names.
Use PascalCase for class names.
Add docstrings to all public functions and classes.
''')

        result = await handle_validate_rules({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert data["total_files"] == 1
        assert data["valid"] == 1
        assert data["errors"] == 0
        assert len(data["files"]) == 1
        assert data["files"][0]["file"] == "100-python.mdc"
        assert data["files"][0]["info"]["rule_number"] == 100

    @pytest.mark.asyncio
    async def test_detects_missing_frontmatter(self, temp_dir: Path):
        """Test detection of missing frontmatter."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        (rules_dir / "100-bad.mdc").write_text("# No frontmatter\n\nJust content.")

        result = await handle_validate_rules({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert data["errors"] > 0
        assert any("frontmatter" in e.lower() for e in data["files"][0]["errors"])

    @pytest.mark.asyncio
    async def test_detects_duplicate_rule_numbers(self, temp_dir: Path):
        """Test detection of duplicate rule numbers."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        for name in ["100-python.mdc", "100-javascript.mdc"]:
            (rules_dir / name).write_text('''---
description: Test
globs: "**/*"
---

# Test Rule
''')

        result = await handle_validate_rules({"repo_path": str(temp_dir)})
        data = json.loads(result)

        # Should detect duplicate
        all_errors = []
        for f in data["files"]:
            all_errors.extend(f["errors"])
        assert any("duplicate" in e.lower() for e in all_errors)

    @pytest.mark.asyncio
    async def test_custom_rules_path(self, temp_dir: Path):
        """Test using custom rules path."""
        custom_dir = temp_dir / "my-rules"
        custom_dir.mkdir()

        (custom_dir / "100-custom.mdc").write_text('''---
description: Custom rules
globs: "**/*"
---

# Custom Coding Standards

These are my custom rules for the project.
Follow consistent naming conventions and add proper documentation.
''')

        result = await handle_validate_rules({
            "repo_path": str(temp_dir),
            "custom_rules_path": str(custom_dir),
        })
        data = json.loads(result)

        assert data["total_files"] == 1
        assert data["valid"] == 1

    @pytest.mark.asyncio
    async def test_missing_repo_path(self):
        """Test error when repo_path is missing."""
        with pytest.raises(ValueError, match="repo_path is required"):
            await handle_validate_rules({})

    @pytest.mark.asyncio
    async def test_validates_real_rules(self):
        """Test validating actual project rules."""
        repo_path = Path(__file__).parent.parent
        if not (repo_path / ".cursor" / "rules").exists():
            pytest.skip("Project rules directory not found")

        result = await handle_validate_rules({"repo_path": str(repo_path)})
        data = json.loads(result)

        assert data["total_files"] > 0
        assert "summary" in data
        assert "files" in data


class TestDetectRuleConflictsMCP:
    """Tests for detect_rule_conflicts MCP tool handler."""

    @pytest.mark.asyncio
    async def test_no_rules_found(self, temp_dir: Path):
        """Test handling repos with no rules."""
        result = await handle_detect_conflicts({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "No rules found"

    @pytest.mark.asyncio
    async def test_no_conflicts_different_categories(self, temp_dir: Path):
        """Test no conflicts when rules address different categories."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        (rules_dir / "100-python.mdc").write_text('''---
description: Python naming
globs: "**/*.py"
---

# Python

Use snake_case for functions.
''')

        (rules_dir / "101-testing.mdc").write_text('''---
description: Testing
globs: "**/*.py"
---

# Testing

Use pytest for all tests.
''')

        result = await handle_detect_conflicts({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert data["rules_analyzed"] == 2
        # No conflicts expected since they address different categories
        assert data["summary"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_detects_naming_conflict(self, temp_dir: Path):
        """Test detection of naming convention conflicts."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        (rules_dir / "100-python.mdc").write_text('''---
description: Python naming
globs: "**/*"
---

# Python

Use snake_case for all functions.
''')

        (rules_dir / "101-js.mdc").write_text('''---
description: JS naming
globs: "**/*"
---

# JavaScript

Use camelCase for all functions.
''')

        result = await handle_detect_conflicts({"repo_path": str(temp_dir)})
        data = json.loads(result)

        assert data["rules_analyzed"] == 2
        # Should detect naming convention conflict
        assert len(data["conflicts"]) > 0 or data["summary"]["warnings"] > 0

    @pytest.mark.asyncio
    async def test_no_conflict_non_overlapping_globs(self, temp_dir: Path):
        """Test no category conflict when globs don't overlap."""
        rules_dir = temp_dir / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)

        # Use different keywords that would conflict if globs overlapped
        # but naming_convention category conflict checks globs
        (rules_dir / "100-python.mdc").write_text('''---
description: Python documentation
globs: "*.py"
---

# Python Documentation

Always add Google style docstrings.
''')

        (rules_dir / "101-js.mdc").write_text('''---
description: JS documentation
globs: "*.js"
---

# JavaScript Documentation

Always add JSDoc comments.
''')

        result = await handle_detect_conflicts({"repo_path": str(temp_dir)})
        data = json.loads(result)

        # Category conflicts (documentation) should be 0 due to non-overlapping globs
        # (directive conflicts don't check globs)
        category_conflicts = [
            c for c in data["conflicts"] if c["category"] != "directive_conflict"
        ]
        assert len(category_conflicts) == 0

    @pytest.mark.asyncio
    async def test_missing_repo_path(self):
        """Test error when repo_path is missing."""
        with pytest.raises(ValueError, match="repo_path is required"):
            await handle_detect_conflicts({})

    @pytest.mark.asyncio
    async def test_single_file_rules(self, temp_dir: Path):
        """Test handling single-file rules (AGENTS.md)."""
        (temp_dir / "AGENTS.md").write_text("# Project Rules\n\nConventions here.")

        result = await handle_detect_conflicts({
            "repo_path": str(temp_dir),
            "rules_source": "opencode",
        })
        data = json.loads(result)

        assert "message" in data
        assert "single-file" in data["message"].lower()
        assert data["rules_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_detects_real_conflicts(self):
        """Test analyzing actual project rules."""
        repo_path = Path(__file__).parent.parent
        if not (repo_path / ".cursor" / "rules").exists():
            pytest.skip("Project rules directory not found")

        result = await handle_detect_conflicts({"repo_path": str(repo_path)})
        data = json.loads(result)

        assert data["rules_analyzed"] > 0
        assert "conflicts" in data
        assert "summary" in data
