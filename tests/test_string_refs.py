"""Tests for string reference extraction."""

from pathlib import Path

from nodestradamus.analyzers.string_extraction import (
    extract_python_strings,
)
from nodestradamus.analyzers.string_extraction import (
    extract_python_strings_from_file as extract_python_file,
)
from nodestradamus.analyzers.string_extraction import (
    extract_typescript_strings as extract_ts_strings,
)
from nodestradamus.analyzers.string_extraction import (
    extract_typescript_strings_from_file as extract_ts_file,
)


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

        result = extract_python_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        # Should extract the file path strings
        values = [s["value"] for s in strings]
        assert "config/settings.yaml" in values
        assert "data.json" in values

        # Check context for config path
        config_str = next(s for s in strings if s["value"] == "config/settings.yaml")
        assert config_str["context"]["call_site"] == "open"
        assert config_str["context"]["enclosing_function"] == "main"

    def test_extracts_variable_assignments(self, tmp_path: Path) -> None:
        """Test extraction of strings from variable assignments."""
        code = '''
REDIS_CHANNEL = "events/notifications"
DATABASE_URL = "postgresql://localhost/db"
'''
        py_file = tmp_path / "config.py"
        py_file.write_text(code)

        result = extract_python_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "events/notifications" in values
        assert "postgresql://localhost/db" in values

        # Check variable name context
        redis_str = next(s for s in strings if s["value"] == "events/notifications")
        assert redis_str["context"]["variable_name"] == "REDIS_CHANNEL"

    def test_extracts_dictionary_values(self, tmp_path: Path) -> None:
        """Test extraction of strings from dictionaries."""
        code = '''
config = {
    "host": "localhost",
    "channel": "pubsub/events",
}
'''
        py_file = tmp_path / "settings.py"
        py_file.write_text(code)

        result = extract_python_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "localhost" in values
        assert "pubsub/events" in values
        assert "host" in values
        assert "channel" in values

    def test_filters_noise_strings(self, tmp_path: Path) -> None:
        """Test that short/noise strings are filtered out."""
        code = '''
x = ""
y = " "
z = "a"
mode = "r"
encoding = "utf-8"
real_value = "important_config_path"
'''
        py_file = tmp_path / "test.py"
        py_file.write_text(code)

        result = extract_python_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        # Short/noise strings should be filtered
        assert "" not in values
        assert " " not in values
        assert "a" not in values
        assert "r" not in values
        assert "utf-8" not in values
        # Real value should be kept
        assert "important_config_path" in values

    def test_captures_class_context(self, tmp_path: Path) -> None:
        """Test that class context is captured."""
        code = '''
class DatabaseService:
    def connect(self):
        return self.pool.get("db://production")
'''
        py_file = tmp_path / "service.py"
        py_file.write_text(code)

        result = extract_python_file(py_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        db_str = next(s for s in strings if s["value"] == "db://production")
        assert db_str["context"]["enclosing_class"] == "DatabaseService"
        assert db_str["context"]["enclosing_function"] == "connect"

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from multiple files in a directory."""
        # Create multiple Python files
        (tmp_path / "app.py").write_text('CONFIG = "app.yaml"')
        (tmp_path / "utils.py").write_text('LOG_PATH = "logs/app.log"')

        result = extract_python_strings(tmp_path)

        assert result["file_count"] == 2
        assert len(result["strings"]) >= 2

        values = [s["value"] for s in result["strings"]]
        assert "app.yaml" in values
        assert "logs/app.log" in values


class TestTypeScriptStringExtraction:
    """Tests for TypeScript/JavaScript string extraction."""

    def test_extracts_import_strings(self, tmp_path: Path) -> None:
        """Test extraction of strings from various locations."""
        code = '''
const config = require("./config.json");
const API_URL = "https://api.example.com";

function connect() {
    return fetch("https://api.example.com/users");
}
'''
        ts_file = tmp_path / "app.js"
        ts_file.write_text(code)

        result = extract_ts_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "./config.json" in values
        assert "https://api.example.com" in values
        assert "https://api.example.com/users" in values

    def test_extracts_variable_assignments(self, tmp_path: Path) -> None:
        """Test extraction of strings from variable assignments."""
        code = '''
const CHANNEL_NAME = "events/notifications";
let dbUrl = "mongodb://localhost:27017";
'''
        ts_file = tmp_path / "config.ts"
        ts_file.write_text(code)

        result = extract_ts_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "events/notifications" in values
        assert "mongodb://localhost:27017" in values

        # Check variable name context
        channel_str = next(s for s in strings if s["value"] == "events/notifications")
        assert channel_str["context"]["variable_name"] == "CHANNEL_NAME"

    def test_extracts_function_call_arguments(self, tmp_path: Path) -> None:
        """Test extraction of strings from function call arguments."""
        code = '''
function setup() {
    redis.subscribe("channel/updates");
    db.connect("postgresql://localhost/app");
}
'''
        ts_file = tmp_path / "setup.ts"
        ts_file.write_text(code)

        result = extract_ts_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "channel/updates" in values
        assert "postgresql://localhost/app" in values

        # Check call site context
        channel_str = next(s for s in strings if s["value"] == "channel/updates")
        assert channel_str["context"]["call_site"] == "redis.subscribe"
        assert channel_str["context"]["enclosing_function"] == "setup"

    def test_filters_noise_strings(self, tmp_path: Path) -> None:
        """Test that short/noise strings are filtered out."""
        code = '''
const x = "";
const y = " ";
const method = "GET";
const realValue = "important_path/config";
'''
        ts_file = tmp_path / "test.ts"
        ts_file.write_text(code)

        result = extract_ts_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        # Noise should be filtered
        assert "" not in values
        assert " " not in values
        assert "GET" not in values
        # Real value should be kept
        assert "important_path/config" in values

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

        result = extract_ts_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        api_str = next(s for s in strings if "api.example.com" in s["value"])
        assert api_str["context"]["enclosing_class"] == "ApiClient"
        assert api_str["context"]["enclosing_function"] == "fetch"

    def test_directory_extraction(self, tmp_path: Path) -> None:
        """Test extraction from multiple files in a directory."""
        # Create multiple TS files
        (tmp_path / "app.ts").write_text('const APP = "my-app";')
        (tmp_path / "config.js").write_text('const DB = "mongo://db";')

        result = extract_ts_strings(tmp_path)

        assert result["file_count"] == 2
        assert len(result["strings"]) >= 2

        values = [s["value"] for s in result["strings"]]
        assert "my-app" in values
        assert "mongo://db" in values

    def test_handles_template_strings(self, tmp_path: Path) -> None:
        """Test extraction from template strings."""
        code = '''
const greeting = `Hello, World!`;
const message = `Important message here`;
'''
        ts_file = tmp_path / "messages.ts"
        ts_file.write_text(code)

        result = extract_ts_file(ts_file, tmp_path)

        assert "error" not in result
        strings = result["strings"]

        values = [s["value"] for s in strings]
        assert "Hello, World!" in values
        assert "Important message here" in values
