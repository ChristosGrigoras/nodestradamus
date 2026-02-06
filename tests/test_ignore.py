"""Tests for the centralized ignore module."""

from pathlib import Path

from nodestradamus.analyzers.ignore import (
    DEFAULT_IGNORES,
    FRAMEWORK_IGNORES,
    LANGUAGE_IGNORES,
    NODESTRADAMUSIGNORE_FILENAME,
    create_should_ignore_func,
    generate_nodestradamusignore_content,
    generate_suggested_ignores,
    get_default_ignores,
    get_framework_ignores,
    get_language_ignores,
    load_ignore_patterns,
    nodestradamusignore_exists,
    parse_nodestradamusignore,
    should_ignore,
)


class TestDefaultIgnores:
    """Tests for DEFAULT_IGNORES constant."""

    def test_includes_node_modules(self):
        """node_modules should always be ignored."""
        assert "node_modules" in DEFAULT_IGNORES

    def test_includes_git(self):
        """.git should always be ignored."""
        assert ".git" in DEFAULT_IGNORES

    def test_includes_pycache(self):
        """__pycache__ should always be ignored."""
        assert "__pycache__" in DEFAULT_IGNORES

    def test_includes_venv(self):
        """Virtual environment directories should be ignored."""
        assert "venv" in DEFAULT_IGNORES
        assert ".venv" in DEFAULT_IGNORES

    def test_includes_build_dirs(self):
        """Common build directories should be ignored."""
        assert "dist" in DEFAULT_IGNORES
        assert "build" in DEFAULT_IGNORES

    def test_is_frozenset(self):
        """DEFAULT_IGNORES should be immutable."""
        assert isinstance(DEFAULT_IGNORES, frozenset)


class TestFrameworkIgnores:
    """Tests for FRAMEWORK_IGNORES mapping."""

    def test_next_framework(self):
        """Next.js should have .next and out patterns."""
        assert "next" in FRAMEWORK_IGNORES
        assert ".next" in FRAMEWORK_IGNORES["next"]
        assert "out" in FRAMEWORK_IGNORES["next"]

    def test_rust_framework(self):
        """Rust should have target pattern."""
        assert "rust" in FRAMEWORK_IGNORES
        assert "target" in FRAMEWORK_IGNORES["rust"]

    def test_pytest_framework(self):
        """pytest should have .pytest_cache pattern."""
        assert "pytest" in FRAMEWORK_IGNORES
        assert ".pytest_cache" in FRAMEWORK_IGNORES["pytest"]


class TestLanguageIgnores:
    """Tests for LANGUAGE_IGNORES mapping."""

    def test_rust_language(self):
        """Rust language should have target pattern."""
        assert "rust" in LANGUAGE_IGNORES
        assert "target" in LANGUAGE_IGNORES["rust"]

    def test_java_language(self):
        """Java language should have appropriate patterns."""
        assert "java" in LANGUAGE_IGNORES
        assert "target" in LANGUAGE_IGNORES["java"]


class TestGetDefaultIgnores:
    """Tests for get_default_ignores function."""

    def test_returns_set(self):
        """Should return a mutable set."""
        result = get_default_ignores()
        assert isinstance(result, set)

    def test_returns_copy(self):
        """Should return a copy, not the original."""
        result = get_default_ignores()
        result.add("custom_dir")
        assert "custom_dir" not in DEFAULT_IGNORES


class TestGetFrameworkIgnores:
    """Tests for get_framework_ignores function."""

    def test_returns_patterns_for_known_framework(self):
        """Should return patterns for known frameworks."""
        result = get_framework_ignores(["next"])
        assert ".next" in result
        assert "out" in result

    def test_returns_empty_for_unknown_framework(self):
        """Should return empty set for unknown frameworks."""
        result = get_framework_ignores(["unknown_framework"])
        assert result == set()

    def test_combines_multiple_frameworks(self):
        """Should combine patterns from multiple frameworks."""
        result = get_framework_ignores(["next", "rust"])
        assert ".next" in result
        assert "target" in result

    def test_case_insensitive(self):
        """Should match frameworks case-insensitively."""
        result = get_framework_ignores(["Next", "RUST"])
        assert ".next" in result
        assert "target" in result


class TestGetLanguageIgnores:
    """Tests for get_language_ignores function."""

    def test_returns_patterns_for_known_language(self):
        """Should return patterns for known languages."""
        result = get_language_ignores({"rust": 10})
        assert "target" in result

    def test_returns_empty_for_unknown_language(self):
        """Should return empty set for unknown languages."""
        result = get_language_ignores({"unknown": 5})
        assert result == set()


class TestGenerateSuggestedIgnores:
    """Tests for generate_suggested_ignores function."""

    def test_includes_defaults(self):
        """Should include default patterns."""
        result = generate_suggested_ignores([], {})
        assert "node_modules" in result
        assert ".git" in result

    def test_includes_framework_patterns(self):
        """Should include framework-specific patterns."""
        result = generate_suggested_ignores(["next"], {})
        assert ".next" in result
        assert "out" in result

    def test_includes_language_patterns(self):
        """Should include language-specific patterns."""
        result = generate_suggested_ignores([], {"rust": 10})
        assert "target" in result

    def test_returns_sorted_list(self):
        """Should return a sorted list."""
        result = generate_suggested_ignores(["next"], {"rust": 10})
        assert result == sorted(result)


class TestParseNodestradamusignore:
    """Tests for parse_nodestradamusignore function."""

    def test_returns_empty_for_missing_file(self, tmp_path):
        """Should return empty set if file doesn't exist."""
        result = parse_nodestradamusignore(tmp_path)
        assert result == set()

    def test_parses_patterns(self, tmp_path):
        """Should parse patterns from file."""
        ignore_file = tmp_path / NODESTRADAMUSIGNORE_FILENAME
        ignore_file.write_text("custom_dir\nother_dir\n")

        result = parse_nodestradamusignore(tmp_path)
        assert "custom_dir" in result
        assert "other_dir" in result

    def test_ignores_comments(self, tmp_path):
        """Should ignore comment lines."""
        ignore_file = tmp_path / NODESTRADAMUSIGNORE_FILENAME
        ignore_file.write_text("# This is a comment\ncustom_dir\n")

        result = parse_nodestradamusignore(tmp_path)
        assert "# This is a comment" not in result
        assert "custom_dir" in result

    def test_ignores_empty_lines(self, tmp_path):
        """Should ignore empty lines."""
        ignore_file = tmp_path / NODESTRADAMUSIGNORE_FILENAME
        ignore_file.write_text("custom_dir\n\nother_dir\n")

        result = parse_nodestradamusignore(tmp_path)
        assert "" not in result
        assert len(result) == 2

    def test_strips_trailing_slashes(self, tmp_path):
        """Should strip trailing slashes from patterns."""
        ignore_file = tmp_path / NODESTRADAMUSIGNORE_FILENAME
        ignore_file.write_text("custom_dir/\n")

        result = parse_nodestradamusignore(tmp_path)
        assert "custom_dir" in result
        assert "custom_dir/" not in result


class TestNodestradamusignoreExists:
    """Tests for nodestradamusignore_exists function."""

    def test_returns_false_when_missing(self, tmp_path):
        """Should return False if file doesn't exist."""
        result = nodestradamusignore_exists(tmp_path)
        assert result is False

    def test_returns_true_when_exists(self, tmp_path):
        """Should return True if file exists."""
        ignore_file = tmp_path / NODESTRADAMUSIGNORE_FILENAME
        ignore_file.write_text("custom_dir\n")

        result = nodestradamusignore_exists(tmp_path)
        assert result is True


class TestLoadIgnorePatterns:
    """Tests for load_ignore_patterns function."""

    def test_includes_defaults(self, tmp_path):
        """Should include default patterns."""
        result = load_ignore_patterns(tmp_path)
        assert "node_modules" in result
        assert ".git" in result

    def test_includes_framework_patterns(self, tmp_path):
        """Should include framework patterns when provided."""
        result = load_ignore_patterns(tmp_path, frameworks=["next"])
        assert ".next" in result

    def test_includes_language_patterns(self, tmp_path):
        """Should include language patterns when provided."""
        result = load_ignore_patterns(tmp_path, languages={"rust": 10})
        assert "target" in result

    def test_includes_nodestradamusignore_patterns(self, tmp_path):
        """Should include patterns from .nodestradamusignore."""
        ignore_file = tmp_path / NODESTRADAMUSIGNORE_FILENAME
        ignore_file.write_text("custom_dir\n")

        result = load_ignore_patterns(tmp_path)
        assert "custom_dir" in result


class TestShouldIgnore:
    """Tests for should_ignore function."""

    def test_ignores_node_modules(self, tmp_path):
        """Should ignore paths in node_modules."""
        path = tmp_path / "node_modules" / "package" / "index.js"
        patterns = {"node_modules"}

        result = should_ignore(path, tmp_path, patterns)
        assert result is True

    def test_ignores_nested_pattern(self, tmp_path):
        """Should ignore nested paths matching pattern."""
        path = tmp_path / "src" / ".next" / "cache" / "file.js"
        patterns = {".next"}

        result = should_ignore(path, tmp_path, patterns)
        assert result is True

    def test_allows_non_matching_paths(self, tmp_path):
        """Should allow paths not matching any pattern."""
        path = tmp_path / "src" / "components" / "Button.tsx"
        patterns = {"node_modules", ".next"}

        result = should_ignore(path, tmp_path, patterns)
        assert result is False

    def test_ignores_paths_outside_repo(self, tmp_path):
        """Should ignore paths outside repo root."""
        other_path = Path("/some/other/path/file.py")
        patterns = set()

        result = should_ignore(other_path, tmp_path, patterns)
        assert result is True


class TestCreateShouldIgnoreFunc:
    """Tests for create_should_ignore_func function."""

    def test_returns_callable(self, tmp_path):
        """Should return a callable function."""
        func = create_should_ignore_func(tmp_path)
        assert callable(func)

    def test_callable_ignores_correctly(self, tmp_path):
        """Returned function should correctly identify ignored paths."""
        func = create_should_ignore_func(tmp_path, patterns={"node_modules"})

        ignored_path = tmp_path / "node_modules" / "package.json"
        allowed_path = tmp_path / "src" / "index.ts"

        assert func(ignored_path) is True
        assert func(allowed_path) is False

    def test_loads_patterns_if_not_provided(self, tmp_path):
        """Should load patterns if none provided."""
        func = create_should_ignore_func(tmp_path)

        # Should use DEFAULT_IGNORES
        ignored_path = tmp_path / "node_modules" / "package.json"
        assert func(ignored_path) is True


class TestGenerateNodestradamusignoreContent:
    """Tests for generate_nodestradamusignore_content function."""

    def test_includes_header(self):
        """Should include header comment."""
        result = generate_nodestradamusignore_content([], {})
        assert "# .nodestradamusignore" in result

    def test_includes_detected_info(self):
        """Should include detected frameworks/languages."""
        result = generate_nodestradamusignore_content(["next", "express"], {"typescript": 50})
        assert "next" in result
        assert "express" in result
        assert "typescript" in result

    def test_includes_framework_patterns(self):
        """Should include framework-specific patterns."""
        result = generate_nodestradamusignore_content(["next"], {})
        assert ".next/" in result

    def test_includes_universal_patterns(self):
        """Should include common patterns."""
        result = generate_nodestradamusignore_content([], {})
        assert "node_modules/" in result
        assert "__pycache__/" in result

    def test_ends_with_newline(self):
        """File content should end with newline."""
        result = generate_nodestradamusignore_content([], {})
        assert result.endswith("\n")
