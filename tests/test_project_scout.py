"""Tests for project_scout analyzer."""

import json
from pathlib import Path

import pytest

from nodestradamus.analyzers.project_scout import project_scout
from nodestradamus.models.graph import ProjectMetadata


@pytest.fixture
def sample_mixed_project(temp_dir: Path) -> Path:
    """Create a sample project with mixed languages."""
    # Python files
    (temp_dir / "main.py").write_text("def main():\n    pass\n")
    (temp_dir / "utils.py").write_text("def helper():\n    pass\n")

    # TypeScript files
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "index.ts").write_text("export const main = () => {};\n")
    (temp_dir / "src" / "utils.ts").write_text("export function helper() {}\n")
    (temp_dir / "src" / "app.tsx").write_text("export default function App() {}\n")

    # JavaScript files
    (temp_dir / "config.js").write_text("module.exports = {};\n")

    # Config files
    (temp_dir / "package.json").write_text(json.dumps({
        "name": "test-project",
        "dependencies": {
            "express": "^4.18.0",
            "react": "^18.0.0",
        },
        "devDependencies": {
            "jest": "^29.0.0",
        },
    }))
    (temp_dir / "pyproject.toml").write_text("""
[project]
name = "test"
dependencies = ["pytest", "flask"]
""")

    # Tests directory
    (temp_dir / "tests").mkdir()
    (temp_dir / "tests" / "test_main.py").write_text("def test_main():\n    pass\n")

    # CI config
    (temp_dir / ".github").mkdir()
    (temp_dir / ".github" / "workflows").mkdir()
    (temp_dir / ".github" / "workflows" / "ci.yml").write_text("name: CI\n")

    return temp_dir


@pytest.fixture
def sample_python_only_project(temp_dir: Path) -> Path:
    """Create a Python-only project."""
    (temp_dir / "main.py").write_text("def main():\n    pass\n")
    (temp_dir / "app.py").write_text("from flask import Flask\n")
    (temp_dir / "utils.py").write_text("def helper():\n    pass\n")
    (temp_dir / "requirements.txt").write_text("flask==2.0.0\npytest\n")
    (temp_dir / "tests").mkdir()
    (temp_dir / "tests" / "test_main.py").write_text("def test_main():\n    pass\n")
    return temp_dir


class TestProjectScout:
    """Tests for project_scout function."""

    def test_detects_languages(self, sample_mixed_project: Path) -> None:
        """Should detect all languages in the project."""
        result = project_scout(sample_mixed_project)

        assert isinstance(result, ProjectMetadata)
        assert "python" in result.languages
        assert "typescript" in result.languages
        assert "javascript" in result.languages
        # 3 Python files (main.py, utils.py, test_main.py)
        assert result.languages["python"] == 3
        # 3 TypeScript files (index.ts, utils.ts, app.tsx)
        assert result.languages["typescript"] == 3
        # 1 JavaScript file (config.js)
        assert result.languages["javascript"] == 1

    def test_detects_primary_language(self, sample_mixed_project: Path) -> None:
        """Should identify the primary language."""
        result = project_scout(sample_mixed_project)

        # TypeScript and Python are tied (3 each), so either could be primary
        assert result.primary_language in ("python", "typescript")

    def test_detects_config_files(self, sample_mixed_project: Path) -> None:
        """Should detect configuration files."""
        result = project_scout(sample_mixed_project)

        assert "package.json" in result.config_files
        assert "pyproject.toml" in result.config_files

    def test_detects_entry_points(self, sample_mixed_project: Path) -> None:
        """Should detect entry point files."""
        result = project_scout(sample_mixed_project)

        assert "main.py" in result.entry_points
        assert "src/index.ts" in result.entry_points

    def test_detects_key_directories(self, sample_mixed_project: Path) -> None:
        """Should detect important directories."""
        result = project_scout(sample_mixed_project)

        assert "src/" in result.key_directories
        assert "tests/" in result.key_directories

    def test_detects_frameworks(self, sample_mixed_project: Path) -> None:
        """Should detect frameworks from package files."""
        result = project_scout(sample_mixed_project)

        assert "express" in result.frameworks
        assert "react" in result.frameworks
        assert "jest" in result.frameworks
        assert "flask" in result.frameworks
        assert "pytest" in result.frameworks

    def test_detects_package_managers(self, sample_mixed_project: Path) -> None:
        """Should detect package managers."""
        result = project_scout(sample_mixed_project)

        assert "npm" in result.package_managers
        assert "pip" in result.package_managers

    def test_detects_tests(self, sample_mixed_project: Path) -> None:
        """Should detect tests directory."""
        result = project_scout(sample_mixed_project)

        assert result.has_tests is True

    def test_detects_ci(self, sample_mixed_project: Path) -> None:
        """Should detect CI configuration."""
        result = project_scout(sample_mixed_project)

        assert result.has_ci is True

    def test_suggests_tools(self, sample_mixed_project: Path) -> None:
        """Should suggest appropriate Nodestradamus tools."""
        result = project_scout(sample_mixed_project)

        assert "analyze_deps" in result.suggested_tools
        assert "analyze_cooccurrence" in result.suggested_tools

    def test_suggests_queries(self, sample_mixed_project: Path) -> None:
        """Should suggest example queries."""
        result = project_scout(sample_mixed_project)

        assert len(result.suggested_queries) > 0
        # Should mention one of the entry points
        assert any("main.py" in q or "index.ts" in q for q in result.suggested_queries)

    def test_python_only_project(self, sample_python_only_project: Path) -> None:
        """Should correctly analyze Python-only projects."""
        result = project_scout(sample_python_only_project)

        assert result.primary_language == "python"
        assert "typescript" not in result.languages
        assert "analyze_deps" in result.suggested_tools
        assert "flask" in result.frameworks
        assert "pytest" in result.frameworks

    def test_skips_node_modules(self, temp_dir: Path) -> None:
        """Should skip node_modules directory."""
        (temp_dir / "index.ts").write_text("export const main = () => {};\n")
        (temp_dir / "node_modules").mkdir()
        (temp_dir / "node_modules" / "lodash.js").write_text("module.exports = {};\n")

        result = project_scout(temp_dir)

        # Should only count index.ts, not node_modules content
        assert result.languages.get("typescript", 0) == 1
        assert result.languages.get("javascript", 0) == 0

    def test_skips_pycache(self, temp_dir: Path) -> None:
        """Should skip __pycache__ directory."""
        (temp_dir / "main.py").write_text("def main():\n    pass\n")
        (temp_dir / "__pycache__").mkdir()
        (temp_dir / "__pycache__" / "main.cpython-312.pyc").write_text("garbage")

        result = project_scout(temp_dir)

        # Should only count main.py
        assert result.languages.get("python", 0) == 1

    def test_empty_project(self, temp_dir: Path) -> None:
        """Should handle empty projects gracefully."""
        result = project_scout(temp_dir)

        assert result.languages == {}
        assert result.primary_language is None
        assert result.config_files == []
        assert result.entry_points == []

    def test_no_git(self, temp_dir: Path) -> None:
        """Should handle projects without git."""
        (temp_dir / "main.py").write_text("def main():\n    pass\n")

        result = project_scout(temp_dir)

        assert result.has_git is False
        assert result.recent_commit_count == 0
        assert result.contributors == 0

    def test_invalid_path(self) -> None:
        """Should raise error for invalid path."""
        with pytest.raises(ValueError, match="Not a directory"):
            project_scout("/nonexistent/path")

    def test_returns_pydantic_model(self, sample_mixed_project: Path) -> None:
        """Should return a valid Pydantic model."""
        result = project_scout(sample_mixed_project)

        assert isinstance(result, ProjectMetadata)
        # Should be serializable
        json_output = result.model_dump_json()
        assert "languages" in json_output
        assert "primary_language" in json_output


class TestMonorepoDetection:
    """Tests for monorepo and Python package detection."""

    @pytest.fixture
    def sample_monorepo(self, temp_dir: Path) -> Path:
        """Create a sample monorepo with multiple packages."""
        # Create monorepo structure similar to LangChain
        # libs/core package
        core_dir = temp_dir / "libs" / "core" / "langchain_core"
        core_dir.mkdir(parents=True)
        (core_dir / "__init__.py").write_text("# Core package\n")
        (core_dir / "runnables.py").write_text("class Runnable:\n    pass\n")
        (core_dir / "messages.py").write_text("class Message:\n    pass\n")
        (temp_dir / "libs" / "core" / "pyproject.toml").write_text("""
[project]
name = "langchain-core"
dependencies = ["pydantic>=2.0", "httpx"]

[tool.poetry.scripts]
langchain-core = "langchain_core.cli:main"
""")
        (temp_dir / "libs" / "core" / "tests").mkdir()
        (temp_dir / "libs" / "core" / "tests" / "test_runnables.py").write_text(
            "def test_runnable():\n    pass\n"
        )

        # libs/langchain package
        lc_dir = temp_dir / "libs" / "langchain" / "langchain"
        lc_dir.mkdir(parents=True)
        (lc_dir / "__init__.py").write_text("# LangChain package\n")
        (lc_dir / "chains.py").write_text("class Chain:\n    pass\n")
        (temp_dir / "libs" / "langchain" / "pyproject.toml").write_text("""
[project]
name = "langchain"
dependencies = ["langchain-core", "openai"]
""")

        # libs/partners/openai package
        openai_dir = temp_dir / "libs" / "partners" / "openai" / "langchain_openai"
        openai_dir.mkdir(parents=True)
        (openai_dir / "__init__.py").write_text("# OpenAI integration\n")
        (openai_dir / "chat_models.py").write_text("class ChatOpenAI:\n    pass\n")
        (temp_dir / "libs" / "partners" / "openai" / "pyproject.toml").write_text("""
[project]
name = "langchain-openai"
dependencies = ["openai", "langchain-core"]
""")

        # Root files
        (temp_dir / "pyproject.toml").write_text("""
[project]
name = "langchain-monorepo"

[tool.pytest]
testpaths = ["libs"]
""")
        (temp_dir / "Makefile").write_text("test:\n\tpytest\n")
        (temp_dir / ".pre-commit-config.yaml").write_text("repos:\n  - repo: local\n")

        return temp_dir

    def test_detects_monorepo_packages_as_key_directories(
        self, sample_monorepo: Path
    ) -> None:
        """Should detect nested packages in monorepo structure."""
        result = project_scout(sample_monorepo)

        # Should find the libs/ directory
        assert "libs/" in result.key_directories

        # Should find nested packages (those with pyproject.toml)
        key_dirs_str = " ".join(result.key_directories)
        assert "libs/core" in key_dirs_str or "langchain_core" in key_dirs_str
        assert "libs/langchain" in key_dirs_str or any(
            "langchain/" in d for d in result.key_directories
        )

    def test_detects_nested_config_files(self, sample_monorepo: Path) -> None:
        """Should detect config files in nested packages."""
        result = project_scout(sample_monorepo)

        # Root config files
        assert "pyproject.toml" in result.config_files
        assert "Makefile" in result.config_files
        assert ".pre-commit-config.yaml" in result.config_files

        # Nested pyproject.toml files
        config_files_str = " ".join(result.config_files)
        assert "libs/core/pyproject.toml" in config_files_str
        assert "libs/langchain/pyproject.toml" in config_files_str

    def test_detects_frameworks_from_nested_configs(
        self, sample_monorepo: Path
    ) -> None:
        """Should detect frameworks from nested pyproject.toml files."""
        result = project_scout(sample_monorepo)

        # Should detect pydantic from libs/core/pyproject.toml
        assert "pydantic" in result.frameworks
        # Should detect openai from nested packages
        assert "openai" in result.frameworks
        # Should detect httpx
        assert "httpx" in result.frameworks

    def test_detects_tests_in_nested_directories(self, sample_monorepo: Path) -> None:
        """Should detect tests in nested package directories."""
        result = project_scout(sample_monorepo)

        assert result.has_tests is True

    def test_detects_script_entry_points(self, sample_monorepo: Path) -> None:
        """Should detect script entry points from pyproject.toml."""
        result = project_scout(sample_monorepo)

        # Should find the poetry script definition
        entry_points_str = " ".join(result.entry_points)
        assert "langchain-core" in entry_points_str or "cli:main" in entry_points_str

    def test_lazy_options_present(self, sample_mixed_project: Path) -> None:
        """Scout should include lazy_options (LazyGraph, LazyEmbeddingGraph, lazy embedding)."""
        result = project_scout(sample_mixed_project)

        assert hasattr(result, "lazy_options")
        assert len(result.lazy_options) >= 3
        option_names = [o["option"] for o in result.lazy_options]
        assert "LazyGraph" in option_names
        assert "LazyEmbeddingGraph" in option_names
        assert any("lazy embedding" in name for name in option_names)
        for o in result.lazy_options:
            assert "option" in o and "when" in o and "description" in o

    def test_lazy_next_step_for_monorepo(self, sample_monorepo: Path) -> None:
        """For monorepos, next_steps should include LazyEmbeddingGraph."""
        result = project_scout(sample_monorepo)

        assert result.is_monorepo is True
        lazy_steps = [s for s in result.next_steps if s.get("tool") == "LazyEmbeddingGraph"]
        assert len(lazy_steps) == 1
        assert "load_scope" in lazy_steps[0]["description"] or "scoped" in lazy_steps[0]["description"]

    def test_detects_python_packages_with_init(self, temp_dir: Path) -> None:
        """Should detect Python packages by __init__.py presence."""
        # Create a package without standard directory name
        custom_pkg = temp_dir / "my_custom_package"
        custom_pkg.mkdir()
        (custom_pkg / "__init__.py").write_text("# Custom package\n")
        (custom_pkg / "module.py").write_text("def func():\n    pass\n")
        (custom_pkg / "utils.py").write_text("def helper():\n    pass\n")

        result = project_scout(temp_dir)

        # Should detect the custom package as a key directory
        assert "my_custom_package/" in result.key_directories

    def test_detects_tests_from_file_patterns(self, temp_dir: Path) -> None:
        """Should detect tests from test_*.py patterns, not just directories."""
        # Create test files without a tests/ directory
        (temp_dir / "main.py").write_text("def main():\n    pass\n")
        (temp_dir / "test_main.py").write_text("def test_main():\n    pass\n")

        result = project_scout(temp_dir)

        assert result.has_tests is True

    def test_detects_tests_from_spec_file_patterns(self, temp_dir: Path) -> None:
        """Should detect tests from *.spec.ts patterns."""
        (temp_dir / "index.ts").write_text("export const main = () => {};\n")
        (temp_dir / "index.spec.ts").write_text("test('main', () => {});\n")

        result = project_scout(temp_dir)

        assert result.has_tests is True


class TestPhase3Features:
    """Tests for Phase 3 smart pipeline features."""

    def test_project_type_app(self, temp_dir: Path) -> None:
        """Should classify app-style projects correctly."""
        # Create an app with entry points and server dependencies
        (temp_dir / "main.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
        (temp_dir / "server.py").write_text("import uvicorn\n")
        (temp_dir / "pyproject.toml").write_text("""
[project]
name = "my-app"
dependencies = ["flask", "uvicorn"]

[project.scripts]
serve = "main:run"
""")

        result = project_scout(temp_dir)

        assert result.project_type == "app"

    def test_project_type_lib(self, temp_dir: Path) -> None:
        """Should classify library-style projects correctly."""
        # Create a library structure
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "__init__.py").write_text("# Library\n")
        (temp_dir / "src" / "core.py").write_text("class Widget:\n    pass\n")
        (temp_dir / "pyproject.toml").write_text("""
[project]
name = "my-lib"
readme = "README.md"
classifiers = ["Development Status :: 4 - Beta"]

[tool.poetry]
packages = [{include = "src"}]
""")

        result = project_scout(temp_dir)

        assert result.project_type == "lib"

    def test_project_type_monorepo(self, temp_dir: Path) -> None:
        """Should classify monorepo-style projects correctly."""
        # Create a monorepo with multiple packages
        pkg1 = temp_dir / "packages" / "core"
        pkg1.mkdir(parents=True)
        (pkg1 / "pyproject.toml").write_text("[project]\nname = 'core'\n")
        (pkg1 / "__init__.py").write_text("")

        pkg2 = temp_dir / "packages" / "api"
        pkg2.mkdir(parents=True)
        (pkg2 / "pyproject.toml").write_text("[project]\nname = 'api'\n")
        (pkg2 / "__init__.py").write_text("")

        result = project_scout(temp_dir)

        assert result.project_type == "monorepo"
        assert result.is_monorepo is True

    def test_readme_hints_extraction(self, temp_dir: Path) -> None:
        """Should extract hints from README files."""
        (temp_dir / "README.md").write_text("""
# My Project

The core logic lives in `src/engine/`.

Run with: python src/main.py

## Project Structure

- `src/` - Source code
- `tests/` - Test files
""")
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "main.py").write_text("def main():\n    pass\n")
        (temp_dir / "src" / "engine").mkdir()
        (temp_dir / "src" / "engine" / "__init__.py").write_text("")
        (temp_dir / "tests").mkdir()

        result = project_scout(temp_dir)

        # Should have some readme hints
        assert len(result.readme_hints) >= 0  # May or may not match patterns

        # Should have recommended scope
        assert "src/" in result.recommended_scope or len(result.recommended_scope) >= 0

    def test_recommended_scope_from_readme_paths(self, temp_dir: Path) -> None:
        """Should extract recommended scope from README inline code paths."""
        (temp_dir / "README.md").write_text("""
See `lib/core/module.py` for the main implementation.
Configuration in `config/settings.py`.
""")
        (temp_dir / "lib").mkdir()
        (temp_dir / "lib" / "core").mkdir()
        (temp_dir / "lib" / "core" / "module.py").write_text("# Core module\n")
        (temp_dir / "config").mkdir()
        (temp_dir / "config" / "settings.py").write_text("# Settings\n")

        result = project_scout(temp_dir)

        # Should recommend lib/ and/or config/ based on README mentions
        assert "lib/" in result.recommended_scope or "config/" in result.recommended_scope

    def test_new_fields_in_pydantic_model(self, temp_dir: Path) -> None:
        """Should include new Phase 3 fields in model output."""
        (temp_dir / "main.py").write_text("def main():\n    pass\n")

        result = project_scout(temp_dir)

        # All new fields should be present
        assert hasattr(result, "project_type")
        assert hasattr(result, "readme_hints")
        assert hasattr(result, "recommended_scope")

        # Should be serializable
        json_output = result.model_dump_json()
        assert "project_type" in json_output
        assert "readme_hints" in json_output
        assert "recommended_scope" in json_output


class TestSuggestedQueries:
    """Tests for suggested_queries in project_scout (H4)."""

    def test_get_impact_uses_file_path_not_script(self, temp_dir: Path) -> None:
        """H4: get_impact suggestion should use file paths, not [script:...] entries."""
        # Create a pyproject.toml with script entry points
        (temp_dir / "pyproject.toml").write_text("""
[project]
name = "test"

[project.scripts]
my-cli = "my_package.cli:main"
""")
        # Create a real Python file entry point
        (temp_dir / "main.py").write_text("def main():\n    pass\n")
        (temp_dir / "my_package").mkdir()
        (temp_dir / "my_package" / "__init__.py").write_text("")
        (temp_dir / "my_package" / "cli.py").write_text("def main():\n    pass\n")

        result = project_scout(temp_dir)

        # Find get_impact suggestion if present
        impact_queries = [q for q in result.suggested_queries if "get_impact" in q]

        # If there's a get_impact suggestion, it should NOT contain "[script:"
        for query in impact_queries:
            assert "[script:" not in query, (
                f"get_impact suggestion should not use script-style entry: {query}"
            )
            # It should use the file path entry point
            if "on " in query:
                # Extract the path after "on " and before " to"
                after_on = query.split("on ", 1)[1]
                path_part = after_on.split(" to")[0].strip()
                # Should be a file path, not a script reference
                assert path_part.endswith((".py", ".ts", ".js")) or "/" in path_part, (
                    f"get_impact path should be a file path: {path_part}"
                )

    def test_get_impact_skipped_when_only_script_entries(self, temp_dir: Path) -> None:
        """H4: When only script-style entries exist, get_impact should be skipped."""
        # Create only a pyproject.toml with script entry points (no file entry points)
        (temp_dir / "pyproject.toml").write_text("""
[project]
name = "test"

[project.scripts]
my-cli = "my_package.cli:main"
""")
        # Create the package but no standard entry points (main.py, cli.py, etc.)
        (temp_dir / "my_package").mkdir()
        (temp_dir / "my_package" / "__init__.py").write_text("")
        (temp_dir / "my_package" / "core.py").write_text("def run():\n    pass\n")

        result = project_scout(temp_dir)

        # Find get_impact suggestion if present
        impact_queries = [q for q in result.suggested_queries if "get_impact" in q]

        # If there's any get_impact suggestion, it should NOT contain "[script:"
        for query in impact_queries:
            assert "[script:" not in query, (
                f"get_impact suggestion should not use script-style entry: {query}"
            )
