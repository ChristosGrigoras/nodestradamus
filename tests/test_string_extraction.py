"""Tests for consolidated string extraction module."""

from pathlib import Path

import pytest

from nodestradamus.analyzers.string_extraction import (
    # Constants
    NOISE_PATTERNS,
    PYTHON_NOISE_PATTERNS,
    SKIP_DIRS,
    SQL_NOISE_PATTERNS,
    TYPESCRIPT_NOISE_PATTERNS,
    extract_bash_strings,
    extract_bash_strings_from_file,
    extract_python_strings,
    extract_rust_strings,
    extract_rust_strings_from_file,
    # Python extraction
    extract_python_strings_from_file,
    extract_sql_strings,
    # SQL extraction
    extract_sql_strings_from_file,
    extract_typescript_strings,
    # TypeScript extraction
    extract_typescript_strings_from_file,
    group_strings_by_value_and_file,
    # Utilities
    is_noise,
    should_skip_path,
)


class TestIsNoise:
    """Tests for the is_noise utility function."""

    def test_empty_string_is_noise(self) -> None:
        """Empty strings should be noise."""
        assert is_noise("") is True

    def test_single_char_is_noise(self) -> None:
        """Single character strings should be noise."""
        assert is_noise("a") is True
        assert is_noise("x") is True

    def test_whitespace_only_is_noise(self) -> None:
        """Whitespace-only strings should be noise."""
        assert is_noise("   ") is True
        assert is_noise("\t") is True
        assert is_noise("\n") is True

    def test_common_noise_patterns(self) -> None:
        """Common noise patterns should be detected."""
        assert is_noise("true") is True
        assert is_noise("false") is True
        assert is_noise("null") is True
        assert is_noise("utf-8") is True

    def test_real_strings_are_not_noise(self) -> None:
        """Real strings should not be noise."""
        assert is_noise("config/settings.yaml") is False
        assert is_noise("postgresql://localhost/db") is False
        assert is_noise("important_value") is False

    def test_custom_min_length(self) -> None:
        """Custom minimum length should be respected."""
        assert is_noise("ab", min_length=3) is True
        assert is_noise("abc", min_length=3) is False

    def test_custom_noise_patterns(self) -> None:
        """Custom noise patterns should be respected."""
        custom_patterns = frozenset({"custom_noise"})
        assert is_noise("custom_noise", noise_patterns=custom_patterns) is True
        assert is_noise("not_custom", noise_patterns=custom_patterns) is False


class TestShouldSkipPath:
    """Tests for the should_skip_path utility function."""

    def test_skips_hidden_directories(self) -> None:
        """Hidden directories should be skipped."""
        assert should_skip_path(Path("/project/.git/config")) is True
        assert should_skip_path(Path("/project/.venv/lib/python")) is True

    def test_skips_node_modules(self) -> None:
        """node_modules should be skipped."""
        assert should_skip_path(Path("/project/node_modules/package/index.js")) is True

    def test_skips_cache_directories(self) -> None:
        """Cache directories should be skipped."""
        assert should_skip_path(Path("/project/__pycache__/module.pyc")) is True
        assert should_skip_path(Path("/project/.pytest_cache/v/cache")) is True

    def test_allows_source_directories(self) -> None:
        """Source directories should not be skipped."""
        assert should_skip_path(Path("/project/src/main.py")) is False
        assert should_skip_path(Path("/project/lib/utils.ts")) is False


class TestGroupStringsByValueAndFile:
    """Tests for the grouping utility function."""

    def test_groups_same_strings(self) -> None:
        """Same strings in same file should be grouped."""
        strings = [
            {"value": "test", "file": "a.py", "context": {"line": 1}},
            {"value": "test", "file": "a.py", "context": {"line": 5}},
        ]
        result = group_strings_by_value_and_file(strings)

        assert len(result) == 1
        assert result[0]["value"] == "test"
        assert len(result[0]["contexts"]) == 2

    def test_separates_different_files(self) -> None:
        """Same strings in different files should be separate."""
        strings = [
            {"value": "test", "file": "a.py", "context": {"line": 1}},
            {"value": "test", "file": "b.py", "context": {"line": 1}},
        ]
        result = group_strings_by_value_and_file(strings)

        assert len(result) == 2

    def test_separates_different_values(self) -> None:
        """Different string values should be separate."""
        strings = [
            {"value": "one", "file": "a.py", "context": {"line": 1}},
            {"value": "two", "file": "a.py", "context": {"line": 2}},
        ]
        result = group_strings_by_value_and_file(strings)

        assert len(result) == 2


class TestPythonStringExtraction:
    """Tests for Python string extraction."""

    def test_extracts_function_call_strings(self, tmp_path: Path) -> None:
        """Test extraction of strings from function calls."""
        code = '''
def main():
    config = open("config/settings.yaml")
    data = load("data.json")
'''
        py_file = tmp_path / "test.py"
        py_file.write_text(code)

        result = extract_python_strings_from_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "config/settings.yaml" in values
        assert "data.json" in values

    def test_extracts_variable_assignments(self, tmp_path: Path) -> None:
        """Test extraction of strings from variable assignments."""
        code = '''
REDIS_CHANNEL = "events/notifications"
DATABASE_URL = "postgresql://localhost/db"
'''
        py_file = tmp_path / "config.py"
        py_file.write_text(code)

        result = extract_python_strings_from_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "events/notifications" in values
        assert "postgresql://localhost/db" in values

    def test_captures_context(self, tmp_path: Path) -> None:
        """Test that context is captured correctly."""
        code = '''
class DatabaseService:
    def connect(self):
        return self.pool.get("db://production")
'''
        py_file = tmp_path / "service.py"
        py_file.write_text(code)

        result = extract_python_strings_from_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        db_str = next(s for s in strings if s["value"] == "db://production")
        assert db_str["context"]["enclosing_class"] == "DatabaseService"
        assert db_str["context"]["enclosing_function"] == "connect"
        assert db_str["context"]["call_site"] == "self.pool.get"

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from multiple files in a directory."""
        (tmp_path / "app.py").write_text('CONFIG = "app.yaml"')
        (tmp_path / "utils.py").write_text('LOG_PATH = "logs/app.log"')

        result = extract_python_strings(tmp_path)

        assert result["file_count"] == 2
        assert len(result["strings"]) >= 2

        values = [s["value"] for s in result["strings"]]
        assert "app.yaml" in values
        assert "logs/app.log" in values

    def test_filters_noise(self, tmp_path: Path) -> None:
        """Test that noise strings are filtered."""
        code = '''
x = ""
y = "utf-8"
real = "important_config"
'''
        py_file = tmp_path / "test.py"
        py_file.write_text(code)

        result = extract_python_strings_from_file(py_file, tmp_path)

        values = [s["value"] for s in result["strings"]]
        assert "" not in values
        assert "utf-8" not in values
        assert "important_config" in values

    def test_handles_syntax_errors(self, tmp_path: Path) -> None:
        """Test graceful handling of syntax errors."""
        py_file = tmp_path / "bad.py"
        py_file.write_text("def broken(")

        result = extract_python_strings_from_file(py_file, tmp_path)

        assert "error" in result


class TestTypeScriptStringExtraction:
    """Tests for TypeScript/JavaScript string extraction."""

    def test_extracts_import_strings(self, tmp_path: Path) -> None:
        """Test extraction from import statements."""
        code = '''
const config = require("./config.json");
const API_URL = "https://api.example.com";
'''
        ts_file = tmp_path / "app.js"
        ts_file.write_text(code)

        result = extract_typescript_strings_from_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "./config.json" in values
        assert "https://api.example.com" in values

    def test_extracts_function_call_arguments(self, tmp_path: Path) -> None:
        """Test extraction from function call arguments."""
        code = '''
function setup() {
    redis.subscribe("channel/updates");
}
'''
        ts_file = tmp_path / "setup.ts"
        ts_file.write_text(code)

        result = extract_typescript_strings_from_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "channel/updates" in values

        channel_str = next(s for s in strings if s["value"] == "channel/updates")
        assert channel_str["context"]["call_site"] == "redis.subscribe"
        assert channel_str["context"]["enclosing_function"] == "setup"

    def test_captures_class_context(self, tmp_path: Path) -> None:
        """Test that class context is captured."""
        code = '''
class ApiClient {
    async fetch() {
        return await fetch("https://api.example.com/data");
    }
}
'''
        ts_file = tmp_path / "client.ts"
        ts_file.write_text(code)

        result = extract_typescript_strings_from_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        api_str = next(s for s in strings if "api.example.com" in s["value"])
        assert api_str["context"]["enclosing_class"] == "ApiClient"
        assert api_str["context"]["enclosing_function"] == "fetch"

    def test_handles_template_strings(self, tmp_path: Path) -> None:
        """Test extraction from template strings."""
        code = '''
const greeting = `Hello, World!`;
const message = `Important message here`;
'''
        ts_file = tmp_path / "messages.ts"
        ts_file.write_text(code)

        result = extract_typescript_strings_from_file(ts_file, tmp_path)

        assert "error" not in result
        values = [s["value"] for s in result["strings"]]
        assert "Hello, World!" in values
        assert "Important message here" in values

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from multiple files in a directory."""
        (tmp_path / "app.ts").write_text('const APP = "my-app";')
        (tmp_path / "config.js").write_text('const DB = "mongo://db";')

        result = extract_typescript_strings(tmp_path)

        assert result["file_count"] == 2
        values = [s["value"] for s in result["strings"]]
        assert "my-app" in values
        assert "mongo://db" in values

    def test_filters_noise(self, tmp_path: Path) -> None:
        """Test that noise strings are filtered."""
        code = '''
const method = "GET";
const realValue = "important_path/config";
'''
        ts_file = tmp_path / "test.ts"
        ts_file.write_text(code)

        result = extract_typescript_strings_from_file(ts_file, tmp_path)

        values = [s["value"] for s in result["strings"]]
        assert "GET" not in values
        assert "important_path/config" in values


class TestSQLStringExtraction:
    """Tests for SQL string extraction."""

    def test_extracts_string_literals(self, tmp_path: Path) -> None:
        """Test extraction of SQL string literals."""
        code = """
INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com');
"""
        sql_file = tmp_path / "insert.sql"
        sql_file.write_text(code)

        result = extract_sql_strings_from_file(sql_file, tmp_path)

        # Note: tree-sitter-sql may not be available in all environments
        if "errors" in result and result["errors"]:
            if "tree-sitter-sql not available" in str(result["errors"]):
                pytest.skip("tree-sitter-sql not available")

        strings = result["strings"]
        values = [s["value"] for s in strings]

        assert "John Doe" in values
        assert "john@example.com" in values

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from SQL files in a directory."""
        (tmp_path / "schema.sql").write_text(
            "CREATE TABLE users (name VARCHAR(100) DEFAULT 'Anonymous');"
        )
        (tmp_path / "data.sql").write_text(
            "INSERT INTO config (key, value) VALUES ('app_name', 'MyApp');"
        )

        result = extract_sql_strings(tmp_path)

        # Skip if SQL parser not available
        if result.get("errors"):
            if any("tree-sitter-sql not available" in str(e) for e in result["errors"]):
                pytest.skip("tree-sitter-sql not available")

        assert result["file_count"] == 2


class TestRustStringExtraction:
    """Tests for Rust string extraction."""

    def test_extracts_string_literals(self, tmp_path: Path) -> None:
        """Test extraction of Rust string literals."""
        code = r'''
fn main() {
    let msg = "Hello, World!";
    println!("{}", msg);
}
'''
        rs_file = tmp_path / "main.rs"
        rs_file.write_text(code)

        result = extract_rust_strings_from_file(rs_file, tmp_path)

        if result.get("errors") and any(
            "tree-sitter-rust not available" in str(e) for e in result["errors"]
        ):
            pytest.skip("tree-sitter-rust not available")

        values = [s["value"] for s in result["strings"]]
        assert "Hello, World!" in values

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from Rust files in a directory."""
        (tmp_path / "lib.rs").write_text(r'let x = "config";')
        (tmp_path / "main.rs").write_text(r'println!("start");')

        result = extract_rust_strings(tmp_path)

        if result.get("errors") and any(
            "tree-sitter-rust not available" in str(e) for e in result["errors"]
        ):
            pytest.skip("tree-sitter-rust not available")

        assert result["file_count"] >= 1
        values = [s["value"] for s in result["strings"]]
        assert "config" in values or "start" in values


class TestBashStringExtraction:
    """Tests for Bash string extraction."""

    def test_extracts_string_literals(self, tmp_path: Path) -> None:
        """Test extraction of Bash string literals."""
        code = '''
main() {
    echo "hello from bash"
}
'''
        sh_file = tmp_path / "script.sh"
        sh_file.write_text(code)

        result = extract_bash_strings_from_file(sh_file, tmp_path)

        if result.get("errors") and any(
            "tree-sitter-bash not available" in str(e) for e in result["errors"]
        ):
            pytest.skip("tree-sitter-bash not available")

        values = [s["value"] for s in result["strings"]]
        assert "hello from bash" in values

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from Bash files in a directory."""
        (tmp_path / "a.sh").write_text('echo "first"')
        (tmp_path / "b.sh").write_text('echo "second"')

        result = extract_bash_strings(tmp_path)

        if result.get("errors") and any(
            "tree-sitter-bash not available" in str(e) for e in result["errors"]
        ):
            pytest.skip("tree-sitter-bash not available")

        assert result["file_count"] == 2
        values = [s["value"] for s in result["strings"]]
        assert "first" in values
        assert "second" in values


class TestIntegration:
    """Integration tests for string extraction."""

    def test_python_extraction_matches_fixtures(self) -> None:
        """Test Python extraction against fixture files."""
        fixtures = Path(__file__).parent / "fixtures" / "sample_python"
        if not fixtures.exists():
            pytest.skip("Fixture directory not found")

        result = extract_python_strings(fixtures)

        assert result["file_count"] > 0
        assert len(result["strings"]) > 0
        assert "metadata" in result

    def test_typescript_extraction_matches_fixtures(self) -> None:
        """Test TypeScript extraction against fixture files."""
        fixtures = Path(__file__).parent / "fixtures" / "sample_typescript"
        if not fixtures.exists():
            pytest.skip("Fixture directory not found")

        result = extract_typescript_strings(fixtures)

        assert result["file_count"] > 0
        assert len(result["strings"]) > 0
        assert "metadata" in result


class TestConstantsExported:
    """Tests to verify constants are properly exported."""

    def test_noise_patterns_include_base(self) -> None:
        """NOISE_PATTERNS should include base patterns."""
        assert "true" in NOISE_PATTERNS
        assert "false" in NOISE_PATTERNS
        assert "utf-8" in NOISE_PATTERNS

    def test_skip_dirs_include_common(self) -> None:
        """SKIP_DIRS should include common directories."""
        assert "node_modules" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert ".git" in SKIP_DIRS
        assert "venv" in SKIP_DIRS

    def test_language_specific_patterns(self) -> None:
        """Language-specific patterns should be available."""
        assert "rb" in PYTHON_NOISE_PATTERNS
        assert "GET" in TYPESCRIPT_NOISE_PATTERNS
        assert "plpgsql" in SQL_NOISE_PATTERNS
