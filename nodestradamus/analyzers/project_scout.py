"""Project scout analyzer for repository reconnaissance.

Provides quick metadata about a repository's structure, languages,
frameworks, and recommended Nodestradamus tools.
"""

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Literal

from nodestradamus.analyzers.ignore import (
    DEFAULT_IGNORES,
    generate_suggested_ignores,
    nodestradamusignore_exists,
    parse_nodestradamusignore,
)
from nodestradamus.models.graph import PackageInfo, ProjectMetadata

# Language detection by file extension
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".sql": "sql",
    ".pgsql": "sql",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
}

# Config files to detect (root-level)
CONFIG_FILES: list[str] = [
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
    "webpack.config.js",
    "next.config.js",
    "next.config.mjs",
    ".eslintrc.json",
    ".eslintrc.js",
    ".prettierrc",
    ".prettierrc.json",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".pre-commit-config.yaml",
    "tox.ini",
    "pytest.ini",
    ".env.example",
    "renovate.json",
    ".nvmrc",
    ".python-version",
    ".tool-versions",
    "justfile",
    "Taskfile.yml",
]

# Entry point patterns
ENTRY_POINTS: list[str] = [
    "main.py",
    "app.py",
    "server.py",
    "cli.py",
    "run.py",
    "__main__.py",
    "index.ts",
    "index.js",
    "main.ts",
    "main.js",
    "app.ts",
    "app.js",
    "server.ts",
    "server.js",
    "cli.ts",
    "cli.js",
    "main.go",
    "main.rs",
    "lib.rs",
    "Main.java",
    "Application.java",
]

# Note: SKIP_DIRS now imported from nodestradamus.analyzers.ignore as DEFAULT_IGNORES


def project_scout(repo_path: str | Path) -> ProjectMetadata:
    """Perform reconnaissance on a repository.

    Args:
        repo_path: Absolute path to the repository.

    Returns:
        ProjectMetadata with language distribution, structure, and recommendations.
    """
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise ValueError(f"Not a directory: {repo}")

    # Collect data
    languages = _count_languages(repo)
    config_files = _find_config_files(repo)
    entry_points = _find_entry_points(repo)
    key_dirs = _find_key_directories(repo)
    git_info = _get_git_info(repo)
    frameworks = _detect_frameworks(repo, config_files)
    package_managers = _detect_package_managers(config_files)

    # Determine primary language
    primary = max(languages, key=languages.get) if languages else None

    # Check for tests and CI
    has_tests = _has_tests(repo)
    has_ci = _has_ci(repo)

    # Generate tool recommendations
    suggested_tools, suggested_queries = _generate_recommendations(
        languages, primary, entry_points, has_tests
    )

    # Generate suggested ignore patterns based on detected frameworks/languages
    suggested_ignores = generate_suggested_ignores(frameworks, dict(languages))

    # Check if .nodestradamusignore exists and merge its patterns
    has_nodestradamusignore = nodestradamusignore_exists(repo)
    if has_nodestradamusignore:
        custom_ignores = parse_nodestradamusignore(repo)
        # Merge custom patterns into suggested_ignores (as sorted list)
        all_patterns = set(suggested_ignores) | custom_ignores
        suggested_ignores = sorted(all_patterns)

    # Generate workflow guidance
    recommended_workflow, next_steps = _generate_workflow_guidance(suggested_ignores)

    # Detect monorepo packages
    packages = _detect_packages(repo)
    is_monorepo = len(packages) >= 2  # Monorepo has 2+ packages

    # Parse README for hints (Phase 3)
    readme_hints, recommended_scope = _parse_readme_hints(repo)

    # Classify project type (Phase 3)
    project_type = _classify_project_type(
        repo, packages, config_files, entry_points, key_dirs
    )

    # Override project_type if monorepo detected
    if is_monorepo:
        project_type = "monorepo"

    # Lazy loading recommendations (LazyGraph, LazyEmbeddingGraph, lazy embedding)
    total_source_files = sum(languages.values())
    lazy_options = _generate_lazy_options(total_source_files, is_monorepo)

    # Add lazy workflow next step when relevant (monorepo or large repo)
    if is_monorepo or total_source_files >= 5_000:
        next_steps = list(next_steps)
        next_steps.append(
            {
                "tool": "LazyEmbeddingGraph",
                "description": "For scoped analysis: load_scope() then compute_scoped_embeddings() and find_similar(). See lazy_options for details.",
            }
        )

    return ProjectMetadata(
        languages=dict(languages),
        primary_language=primary,
        key_directories=key_dirs,
        entry_points=entry_points,
        config_files=config_files,
        has_git=git_info["has_git"],
        recent_commit_count=git_info["recent_commits"],
        contributors=git_info["contributors"],
        frameworks=frameworks,
        package_managers=package_managers,
        has_tests=has_tests,
        has_ci=has_ci,
        suggested_tools=suggested_tools,
        suggested_queries=suggested_queries,
        suggested_ignores=suggested_ignores,
        nodestradamusignore_exists=has_nodestradamusignore,
        recommended_workflow=recommended_workflow,
        next_steps=next_steps,
        is_monorepo=is_monorepo,
        packages=packages,
        project_type=project_type,
        readme_hints=readme_hints,
        recommended_scope=recommended_scope,
        lazy_options=lazy_options,
    )


def _count_languages(repo: Path) -> Counter[str]:
    """Count files by language."""
    counts: Counter[str] = Counter()

    for path in repo.rglob("*"):
        if path.is_file() and not _should_skip(path, repo):
            ext = path.suffix.lower()
            if ext in LANGUAGE_EXTENSIONS:
                counts[LANGUAGE_EXTENSIONS[ext]] += 1

    return counts


def _should_skip(path: Path, repo: Path) -> bool:
    """Check if path should be skipped.

    Uses DEFAULT_IGNORES from the central ignore module.
    """
    try:
        rel = path.relative_to(repo)
        parts = rel.parts
        return any(part in DEFAULT_IGNORES or part.startswith(".") for part in parts)
    except ValueError:
        return True


def _find_config_files(repo: Path) -> list[str]:
    """Find configuration files in repo root and monorepo packages.

    Returns root config files first, then nested config files with relative paths.
    """
    found: list[str] = []

    # Check root-level config files
    for name in CONFIG_FILES:
        if (repo / name).exists():
            found.append(name)

    # For monorepos: find nested pyproject.toml and package.json files
    nested_configs: list[str] = []

    for config_file in repo.rglob("pyproject.toml"):
        if _should_skip(config_file, repo):
            continue
        if config_file.parent != repo:  # Not root
            try:
                rel_path = str(config_file.relative_to(repo))
                nested_configs.append(rel_path)
            except ValueError:
                continue

    for config_file in repo.rglob("package.json"):
        if _should_skip(config_file, repo):
            continue
        if config_file.parent != repo:
            try:
                rel_path = str(config_file.relative_to(repo))
                nested_configs.append(rel_path)
            except ValueError:
                continue

    # Sort nested configs by depth (shallower first), then alphabetically
    nested_configs.sort(key=lambda p: (p.count("/"), p))

    # Add nested configs, limit to avoid overwhelming output
    found.extend(nested_configs[:15])

    return found


def _find_entry_points(repo: Path) -> list[str]:
    """Find likely entry point files.

    Checks:
    1. Standard locations (root, src/, app/, bin/)
    2. __main__.py files in packages
    3. Script definitions in pyproject.toml [tool.poetry.scripts] or [project.scripts]
    4. package.json bin/main fields
    """
    found: list[str] = []

    # Check root
    for name in ENTRY_POINTS:
        if (repo / name).exists():
            found.append(name)

    # Check common source directories
    source_dirs = ["src", "app", "bin", "scripts"]
    for dir_name in source_dirs:
        src_dir = repo / dir_name
        if src_dir.is_dir():
            for name in ENTRY_POINTS:
                if (src_dir / name).exists():
                    entry = f"{dir_name}/{name}"
                    if entry not in found:
                        found.append(entry)

    # Find __main__.py files in packages (for python -m package invocation)
    for main_file in repo.rglob("__main__.py"):
        if _should_skip(main_file, repo):
            continue
        try:
            rel_path = str(main_file.relative_to(repo))
            if rel_path not in found:
                found.append(rel_path)
        except ValueError:
            continue

    # Parse pyproject.toml for script entry points (root)
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        found.extend(_parse_pyproject_scripts(pyproject, found))

    # Parse nested pyproject.toml files for script entry points (monorepo support)
    for toml_file in repo.rglob("pyproject.toml"):
        if _should_skip(toml_file, repo):
            continue
        if toml_file != pyproject:  # Skip root, already processed
            found.extend(_parse_pyproject_scripts(toml_file, found))

    # Parse package.json for entry points
    package_json = repo / "package.json"
    if package_json.exists():
        found.extend(_parse_package_json_entries(package_json, found))

    return found[:15]  # Limit to 15


def _parse_pyproject_scripts(pyproject: Path, existing: list[str]) -> list[str]:
    """Parse pyproject.toml for script entry points.

    Looks for:
    - [tool.poetry.scripts]
    - [project.scripts]
    - [project.gui-scripts]
    """
    import re

    scripts: list[str] = []
    try:
        content = pyproject.read_text()

        # Simple TOML parsing for scripts sections
        # Match lines like: my-command = "my_package.cli:main"
        # Script names can contain letters, numbers, underscores, and hyphens
        script_pattern = re.compile(r'^\s*([\w-]+)\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)

        in_scripts_section = False
        for line in content.split("\n"):
            if re.match(r'\[(?:tool\.poetry\.scripts|project\.scripts|project\.gui-scripts)\]', line):
                in_scripts_section = True
                continue
            elif line.startswith("["):
                in_scripts_section = False
                continue

            if in_scripts_section:
                match = script_pattern.match(line)
                if match:
                    script_name = match.group(1)
                    script_path = match.group(2)
                    # Convert "package.module:func" to note about entry point
                    entry = f"[script:{script_name}] {script_path}"
                    if entry not in existing and entry not in scripts:
                        scripts.append(entry)

    except OSError:
        pass

    return scripts[:5]  # Limit


def _parse_package_json_entries(package_json: Path, existing: list[str]) -> list[str]:
    """Parse package.json for entry points (main, bin fields)."""
    entries: list[str] = []
    try:
        data = json.loads(package_json.read_text())

        # Check main field
        main = data.get("main")
        if main and main not in existing and main not in entries:
            entries.append(main)

        # Check bin field (can be string or object)
        bin_field = data.get("bin")
        if isinstance(bin_field, str):
            if bin_field not in existing and bin_field not in entries:
                entries.append(bin_field)
        elif isinstance(bin_field, dict):
            for _name, path in bin_field.items():
                if path not in existing and path not in entries:
                    entries.append(path)

    except (json.JSONDecodeError, OSError):
        pass

    return entries[:5]  # Limit


def _parse_readme_hints(repo: Path) -> tuple[list[str], list[str]]:
    """Parse README files to extract hints about project structure.

    Looks for:
    - Mentioned file paths and directories
    - Commands that reference source files
    - Descriptions of core modules/packages

    Args:
        repo: Path to the repository root.

    Returns:
        Tuple of (readme_hints, recommended_scope) where:
        - readme_hints: Natural language hints about the project
        - recommended_scope: Extracted path prefixes for focused analysis
    """
    readme_hints: list[str] = []
    recommended_scope: set[str] = set()

    # Find README files
    readme_files = [
        repo / "README.md",
        repo / "README.rst",
        repo / "readme.md",
        repo / "Readme.md",
    ]

    content = ""
    for readme_path in readme_files:
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8", errors="replace")
                break
            except OSError:
                continue

    if not content:
        return [], []

    # Pattern 1: Extract inline code references to paths
    # Matches `src/foo.py`, `lib/core/`, etc.
    path_pattern = re.compile(r"`([a-zA-Z0-9_\-./]+(?:\.[a-zA-Z]+)?)`")
    for match in path_pattern.finditer(content):
        text = match.group(1)
        # Only consider if it looks like a path
        if "/" in text or text.endswith((".py", ".ts", ".js", ".rs", ".go")):
            # Extract top-level directory
            parts = text.split("/")
            if len(parts) >= 1 and parts[0]:
                top_dir = parts[0].rstrip("/")
                # Check if it exists
                if (repo / top_dir).exists():
                    recommended_scope.add(f"{top_dir}/")

    # Pattern 2: Look for "core logic in X" or "main code in X" phrases
    location_pattern = re.compile(
        r"(?:core|main|source|primary|implementation)\s+(?:code|logic|functionality|module)s?\s+"
        r"(?:is\s+)?(?:in|at|under|lives?\s+in)\s+[`\"']?([a-zA-Z0-9_\-./]+)[`\"']?",
        re.IGNORECASE,
    )
    for match in location_pattern.finditer(content):
        path = match.group(1).strip("`\"'")
        if path:
            readme_hints.append(f"Core logic in {path}")
            parts = path.split("/")
            if parts[0] and (repo / parts[0]).exists():
                recommended_scope.add(f"{parts[0]}/")

    # Pattern 3: Look for run commands that reference files
    # e.g., "python src/main.py" or "node dist/index.js"
    run_pattern = re.compile(
        r"(?:python|python3|node|npm\s+run|cargo\s+run|go\s+run)\s+([a-zA-Z0-9_\-./]+)",
        re.IGNORECASE,
    )
    for match in run_pattern.finditer(content):
        path = match.group(1)
        if "/" in path:
            parts = path.split("/")
            if parts[0] and (repo / parts[0]).exists():
                recommended_scope.add(f"{parts[0]}/")
                readme_hints.append(f"Entry point: {path}")

    # Pattern 4: Look for installation/usage sections mentioning import paths
    import_pattern = re.compile(
        r"(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)",
    )
    for match in import_pattern.finditer(content):
        module = match.group(1).split(".")[0]
        if module and (repo / module).is_dir():
            recommended_scope.add(f"{module}/")

    # Pattern 5: Look for directory structure in markdown
    # Lines like "- `src/` - Source code" or "* lib/ — Libraries"
    dir_desc_pattern = re.compile(
        r"^[\s\-\*]*[`\"']?([a-zA-Z0-9_\-]+/?)[`\"']?\s*[-–—:]\s*(.{10,60})",
        re.MULTILINE,
    )
    for match in dir_desc_pattern.finditer(content):
        dir_name = match.group(1).rstrip("/")
        description = match.group(2).strip()
        if (repo / dir_name).is_dir():
            recommended_scope.add(f"{dir_name}/")
            # Only add hints for significant descriptions
            if any(
                word in description.lower()
                for word in ["main", "core", "source", "primary", "entry"]
            ):
                readme_hints.append(f"{dir_name}: {description[:50]}")

    # Sort and limit
    recommended_scope_list = sorted(recommended_scope)[:10]
    readme_hints = readme_hints[:10]

    return readme_hints, recommended_scope_list


def _classify_project_type(
    repo: Path,
    packages: list[PackageInfo],
    config_files: list[str],
    entry_points: list[str],
    key_dirs: list[str],
) -> Literal["app", "lib", "monorepo", "unknown"]:
    """Classify the project type based on structure analysis.

    Args:
        repo: Path to repository root.
        packages: Detected monorepo packages.
        config_files: List of config files found.
        entry_points: List of entry point files.
        key_dirs: List of key directories.

    Returns:
        Project type: "app", "lib", "monorepo", or "unknown".
    """
    # Monorepo: Multiple packages detected
    if len(packages) >= 2:
        return "monorepo"

    # Check for indicators in config files
    pyproject = repo / "pyproject.toml"
    package_json = repo / "package.json"

    is_lib_like = False
    is_app_like = False

    # Check pyproject.toml for library indicators
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8").lower()
            # Library indicators
            if any(
                indicator in content
                for indicator in [
                    "[tool.poetry]",
                    "[project]",
                    "classifiers",
                    "readme =",
                ]
            ):
                # Check if it's publishable (library)
                if "packages" in content or "find:" in content:
                    is_lib_like = True
            # App indicators
            if any(
                indicator in content
                for indicator in [
                    "[tool.poetry.scripts]",
                    "[project.scripts]",
                    "entry-points",
                    "uvicorn",
                    "gunicorn",
                    "flask",
                    "fastapi",
                    "django",
                ]
            ):
                is_app_like = True
        except OSError:
            pass

    # Check package.json for library vs app indicators
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text())
            # Library indicators
            if data.get("main") or data.get("exports") or data.get("types"):
                is_lib_like = True
            # App indicators (has scripts like start, dev, serve)
            scripts = data.get("scripts", {})
            if any(
                script in scripts for script in ["start", "dev", "serve", "build:app"]
            ):
                is_app_like = True
            # Framework indicators
            deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
            }
            app_frameworks = ["next", "nuxt", "express", "fastify", "nestjs", "koa"]
            if any(fw in deps for fw in app_frameworks):
                is_app_like = True
        except (json.JSONDecodeError, OSError):
            pass

    # Entry point heuristics
    app_entry_points = ["main.py", "app.py", "server.py", "cli.py", "run.py"]
    if any(ep in entry_points or ep.endswith(f"/{ep}") for ep in app_entry_points for ep in entry_points):
        is_app_like = True

    # Presence of key directories
    app_dirs = ["api/", "server/", "backend/", "frontend/", "routes/", "handlers/"]
    lib_dirs = ["src/", "lib/"]

    if any(d in key_dirs for d in app_dirs):
        is_app_like = True
    if any(d in key_dirs for d in lib_dirs) and not is_app_like:
        is_lib_like = True

    # Decision logic
    if is_app_like and not is_lib_like:
        return "app"
    elif is_lib_like and not is_app_like:
        return "lib"
    elif is_app_like and is_lib_like:
        # Both indicators present - prefer app classification
        return "app"
    else:
        return "unknown"


def _find_key_directories(repo: Path) -> list[str]:
    """Find important directories including Python packages and monorepo packages.

    Discovers directories by:
    1. Looking for standard directory names at root (src/, lib/, etc.)
    2. Finding Python packages (directories with __init__.py)
    3. Finding monorepo packages (directories with pyproject.toml/package.json)
    4. Ranking by file count for importance
    """
    key_dirs: list[str] = []
    dir_file_counts: dict[str, int] = {}

    # Standard important directory names
    important = [
        "src",
        "lib",
        "libs",
        "packages",
        "app",
        "api",
        "server",
        "client",
        "frontend",
        "backend",
        "services",
        "components",
        "pages",
        "routes",
        "handlers",
        "controllers",
        "models",
        "utils",
        "tests",
        "test",
        "spec",
        "__tests__",
        "scripts",
        "docs",
        "config",
        "core",
    ]

    # Check standard directories at root
    for name in important:
        if (repo / name).is_dir():
            key_dirs.append(f"{name}/")

    # Find Python packages (directories with __init__.py)
    # This catches monorepo structures like libs/core, libs/langchain
    python_packages = _find_python_packages(repo)
    for pkg_path in python_packages:
        try:
            rel_path = pkg_path.relative_to(repo)
            dir_str = str(rel_path) + "/"
            if dir_str not in key_dirs:
                # Count files to rank importance
                file_count = sum(1 for _ in pkg_path.rglob("*.py") if not _should_skip(_, repo))
                dir_file_counts[dir_str] = file_count
        except ValueError:
            continue

    # Find monorepo packages (directories with their own pyproject.toml/package.json)
    for config_file in repo.rglob("pyproject.toml"):
        if _should_skip(config_file, repo):
            continue
        parent = config_file.parent
        if parent != repo:  # Not the root
            try:
                rel_path = parent.relative_to(repo)
                dir_str = str(rel_path) + "/"
                if dir_str not in key_dirs and dir_str not in dir_file_counts:
                    file_count = sum(1 for _ in parent.rglob("*.py") if not _should_skip(_, repo))
                    dir_file_counts[dir_str] = file_count
            except ValueError:
                continue

    for config_file in repo.rglob("package.json"):
        if _should_skip(config_file, repo):
            continue
        parent = config_file.parent
        if parent != repo:
            try:
                rel_path = parent.relative_to(repo)
                dir_str = str(rel_path) + "/"
                if dir_str not in key_dirs and dir_str not in dir_file_counts:
                    file_count = sum(
                        1 for _ in parent.rglob("*")
                        if _.suffix in (".ts", ".tsx", ".js", ".jsx") and not _should_skip(_, repo)
                    )
                    dir_file_counts[dir_str] = file_count
            except ValueError:
                continue

    # Add discovered directories, sorted by file count (most files first)
    sorted_dirs = sorted(dir_file_counts.items(), key=lambda x: -x[1])
    for dir_str, _ in sorted_dirs:
        if dir_str not in key_dirs:
            key_dirs.append(dir_str)

    return key_dirs[:20]  # Limit to top 20


def _find_python_packages(repo: Path, max_depth: int = 4) -> list[Path]:
    """Find Python package directories (those containing __init__.py).

    Args:
        repo: Repository root path.
        max_depth: Maximum directory depth to search.

    Returns:
        List of paths to Python package directories, sorted by depth then name.
    """
    packages: list[Path] = []

    for init_file in repo.rglob("__init__.py"):
        if _should_skip(init_file, repo):
            continue
        pkg_dir = init_file.parent
        try:
            rel_path = pkg_dir.relative_to(repo)
            # Check depth
            if len(rel_path.parts) <= max_depth:
                # Only include top-level packages (not nested subpackages)
                # e.g., include libs/core/langchain_core but not libs/core/langchain_core/runnables
                is_top_level = True
                for existing in packages:
                    try:
                        pkg_dir.relative_to(existing)
                        is_top_level = False
                        break
                    except ValueError:
                        pass
                if is_top_level:
                    packages.append(pkg_dir)
        except ValueError:
            continue

    # Sort by depth (shallower first), then alphabetically
    packages.sort(key=lambda p: (len(p.relative_to(repo).parts), str(p)))
    return packages[:30]  # Limit


def _get_git_info(repo: Path) -> dict:
    """Get git repository information."""
    git_dir = repo / ".git"
    if not git_dir.exists():
        return {"has_git": False, "recent_commits": 0, "contributors": 0}

    result = {"has_git": True, "recent_commits": 0, "contributors": 0}

    try:
        # Recent commits (last 30 days)
        out = subprocess.run(
            ["git", "log", "--oneline", "--since=30 days ago"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            result["recent_commits"] = (
                len(out.stdout.strip().split("\n")) if out.stdout.strip() else 0
            )

        # Contributors
        out = subprocess.run(
            ["git", "shortlog", "-sn", "--all"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            result["contributors"] = (
                len(out.stdout.strip().split("\n")) if out.stdout.strip() else 0
            )

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return result


def _detect_packages(repo: Path) -> list[PackageInfo]:
    """Detect packages in a monorepo.

    Finds packages by looking for:
    1. Directories with pyproject.toml (Python packages)
    2. Directories with package.json (JS/TS packages)

    Excludes root-level config files (those aren't sub-packages).

    Args:
        repo: Path to repository root.

    Returns:
        List of PackageInfo for each detected package.
    """
    packages: list[PackageInfo] = []
    seen_paths: set[str] = set()

    # Find Python packages (directories with pyproject.toml)
    for pyproject in repo.rglob("pyproject.toml"):
        if _should_skip(pyproject, repo):
            continue

        # Skip root-level pyproject.toml
        if pyproject.parent == repo:
            continue

        pkg_dir = pyproject.parent
        rel_path = str(pkg_dir.relative_to(repo))

        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)

        # Try to extract package name from pyproject.toml
        name = _extract_python_package_name(pyproject)
        if not name:
            name = pkg_dir.name

        packages.append(PackageInfo(
            name=name,
            path=rel_path,
            language="python",
        ))

    # Find JS/TS packages (directories with package.json)
    for pkg_json in repo.rglob("package.json"):
        if _should_skip(pkg_json, repo):
            continue

        # Skip root-level package.json
        if pkg_json.parent == repo:
            continue

        pkg_dir = pkg_json.parent
        rel_path = str(pkg_dir.relative_to(repo))

        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)

        # Try to extract package name from package.json
        name = _extract_js_package_name(pkg_json)
        if not name:
            name = pkg_dir.name

        # Determine if TypeScript or JavaScript
        lang = "typescript" if (pkg_dir / "tsconfig.json").exists() else "javascript"

        packages.append(PackageInfo(
            name=name,
            path=rel_path,
            language=lang,
        ))

    # Sort by path for consistent ordering
    packages.sort(key=lambda p: p.path)
    return packages


def _extract_python_package_name(pyproject_path: Path) -> str | None:
    """Extract package name from pyproject.toml.

    Looks for:
    - [project] name = "..."
    - [tool.poetry] name = "..."
    """
    try:
        content = pyproject_path.read_text(encoding="utf-8")

        # Look for [project] name or [tool.poetry] name
        import re
        # Match name = "package-name" or name = 'package-name'
        match = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def _extract_js_package_name(pkg_json_path: Path) -> str | None:
    """Extract package name from package.json."""
    try:
        content = pkg_json_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return data.get("name")
    except Exception:
        pass
    return None


def _detect_frameworks(repo: Path, config_files: list[str]) -> list[str]:
    """Detect frameworks from package files.

    Checks:
    1. Root package.json dependencies
    2. Root pyproject.toml/requirements.txt dependencies
    3. Nested pyproject.toml files (for monorepos)
    """
    frameworks: list[str] = []
    python_deps: set[str] = set()

    # Python framework keywords to detect
    python_framework_keywords = [
        "flask",
        "django",
        "fastapi",
        "pytest",
        "redis",
        "celery",
        "sqlalchemy",
        "pydantic",
        "aiohttp",
        "tornado",
        "starlette",
        "uvicorn",
        "gunicorn",
        "httpx",
        "requests",
        "click",
        "typer",
        "rich",
        "numpy",
        "pandas",
        "scipy",
        "tensorflow",
        "pytorch",
        "torch",
        "transformers",
        "langchain",
        "openai",
        "anthropic",
    ]

    # Check package.json (root)
    if "package.json" in config_files:
        frameworks.extend(_detect_js_frameworks(repo / "package.json"))

    # Check all nested package.json files
    for config in config_files:
        if config.endswith("package.json") and config != "package.json":
            pkg_path = repo / config
            if pkg_path.exists():
                frameworks.extend(_detect_js_frameworks(pkg_path))

    # Check pyproject.toml (root)
    if "pyproject.toml" in config_files:
        python_deps.update(_detect_python_deps_from_toml(repo / "pyproject.toml", python_framework_keywords))

    # Check all nested pyproject.toml files (monorepo support)
    for config in config_files:
        if config.endswith("pyproject.toml") and config != "pyproject.toml":
            toml_path = repo / config
            if toml_path.exists():
                python_deps.update(_detect_python_deps_from_toml(toml_path, python_framework_keywords))

    # Also scan for any pyproject.toml not in config_files list (in case they were truncated)
    for toml_file in repo.rglob("pyproject.toml"):
        if not _should_skip(toml_file, repo):
            python_deps.update(_detect_python_deps_from_toml(toml_file, python_framework_keywords))

    # Check requirements.txt
    if "requirements.txt" in config_files:
        python_deps.update(_detect_python_deps_from_requirements(repo / "requirements.txt", python_framework_keywords))

    # Add Python frameworks, sorted
    frameworks.extend(sorted(python_deps))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_frameworks: list[str] = []
    for fw in frameworks:
        if fw not in seen:
            seen.add(fw)
            unique_frameworks.append(fw)

    return unique_frameworks[:20]  # Limit


def _detect_js_frameworks(package_json: Path) -> list[str]:
    """Detect JavaScript/TypeScript frameworks from package.json."""
    frameworks: list[str] = []
    try:
        pkg = json.loads(package_json.read_text())
        deps = {
            **pkg.get("dependencies", {}),
            **pkg.get("devDependencies", {}),
            **pkg.get("peerDependencies", {}),
        }

        framework_map = {
            "react": "react",
            "vue": "vue",
            "angular": "angular",
            "svelte": "svelte",
            "next": "nextjs",
            "nuxt": "nuxt",
            "express": "express",
            "fastify": "fastify",
            "koa": "koa",
            "nest": "nestjs",
            "hono": "hono",
            "jest": "jest",
            "mocha": "mocha",
            "vitest": "vitest",
            "playwright": "playwright",
            "cypress": "cypress",
            "tailwindcss": "tailwind",
            "redis": "redis",
            "ioredis": "redis",
            "prisma": "prisma",
            "drizzle": "drizzle",
            "sequelize": "sequelize",
            "typeorm": "typeorm",
            "mongoose": "mongoose",
            "graphql": "graphql",
            "apollo": "apollo",
            "trpc": "trpc",
        }

        for dep in deps:
            for key, name in framework_map.items():
                if key in dep.lower() and name not in frameworks:
                    frameworks.append(name)

    except (json.JSONDecodeError, OSError):
        pass

    return frameworks


def _detect_python_deps_from_toml(toml_path: Path, keywords: list[str]) -> set[str]:
    """Detect Python dependencies from a pyproject.toml file."""
    deps: set[str] = set()
    try:
        content = toml_path.read_text().lower()
        for keyword in keywords:
            if keyword in content:
                deps.add(keyword)
    except OSError:
        pass
    return deps


def _detect_python_deps_from_requirements(req_path: Path, keywords: list[str]) -> set[str]:
    """Detect Python dependencies from a requirements.txt file."""
    deps: set[str] = set()
    try:
        content = req_path.read_text().lower()
        for keyword in keywords:
            if keyword in content:
                deps.add(keyword)
    except OSError:
        pass
    return deps


def _detect_package_managers(config_files: list[str]) -> list[str]:
    """Detect package managers from config files."""
    managers = []

    if "package.json" in config_files:
        managers.append("npm")
    if (
        "pyproject.toml" in config_files
        or "requirements.txt" in config_files
        or "setup.py" in config_files
    ):
        managers.append("pip")
    if "Cargo.toml" in config_files:
        managers.append("cargo")
    if "go.mod" in config_files:
        managers.append("go")
    if "pom.xml" in config_files:
        managers.append("maven")
    if "build.gradle" in config_files:
        managers.append("gradle")
    if "Gemfile" in config_files:
        managers.append("bundler")
    if "composer.json" in config_files:
        managers.append("composer")

    return managers


def _has_tests(repo: Path) -> bool:
    """Check if repository has tests.

    Checks for:
    1. Test directories (tests/, test/, spec/, __tests__/)
    2. Test file patterns (test_*.py, *_test.py, *.test.ts, *.spec.ts)
    """
    # Check for test directories at root
    test_dirs = ["tests", "test", "spec", "__tests__"]
    if any((repo / d).is_dir() for d in test_dirs):
        return True

    # Check for test directories anywhere in the repo (for monorepos)
    for d in test_dirs:
        for path in repo.rglob(d):
            if path.is_dir() and not _should_skip(path, repo):
                return True

    # Check for Python test files (test_*.py, *_test.py)
    for pattern in ["test_*.py", "*_test.py"]:
        for path in repo.rglob(pattern):
            if not _should_skip(path, repo):
                return True

    # Check for JS/TS test files (*.test.ts, *.spec.ts, *.test.js, *.spec.js)
    for pattern in ["*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx",
                    "*.test.js", "*.spec.js", "*.test.jsx", "*.spec.jsx"]:
        for path in repo.rglob(pattern):
            if not _should_skip(path, repo):
                return True

    return False


def _has_ci(repo: Path) -> bool:
    """Check if repository has CI configuration."""
    ci_paths = [
        repo / ".github" / "workflows",
        repo / ".gitlab-ci.yml",
        repo / "Jenkinsfile",
        repo / ".circleci",
        repo / ".travis.yml",
        repo / "azure-pipelines.yml",
    ]
    return any(p.exists() for p in ci_paths)


def _generate_recommendations(
    languages: Counter[str],
    primary: str | None,
    entry_points: list[str],
    has_tests: bool,
) -> tuple[list[str], list[str]]:
    """Generate tool and query recommendations."""
    tools = []
    queries = []

    # Language-based recommendations
    if "python" in languages:
        tools.append("analyze_deps")
        queries.append("analyze_deps to map Python dependencies")

    if "typescript" in languages or "javascript" in languages:
        tools.append("analyze_deps")
        queries.append("analyze_deps to map TS/JS dependencies")

    if "rust" in languages:
        tools.append("analyze_deps")
        queries.append("analyze_deps to map Rust dependencies")
    if "sql" in languages:
        tools.append("analyze_deps")
        queries.append("analyze_deps to map SQL dependencies")

    # Always recommend cooccurrence if git exists
    tools.append("analyze_cooccurrence")
    queries.append("analyze_cooccurrence to find coupled files")

    # Entry point impact analysis (H4: only use file-path entries, not script-style)
    if entry_points:
        # Filter to entries that look like file paths (not [script:...] style)
        file_path_entries = [
            e for e in entry_points
            if not e.startswith("[script:")
            and (e.endswith((".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".sh")) or "/" in e)
        ]
        if file_path_entries:
            entry = file_path_entries[0]
            queries.append(f"get_impact on {entry} to find integration points")

    # Deduplicate tools while preserving order
    seen = set()
    tools = [t for t in tools if not (t in seen or seen.add(t))]

    return tools, queries


def _generate_lazy_options(
    total_source_files: int,
    is_monorepo: bool,
) -> list[dict[str, str]]:
    """Generate lazy/on-demand loading recommendations for scout.

    Surfaces LazyGraph, LazyEmbeddingGraph, and lazy embedding so users know
    when to use them (large codebases, monorepos) instead of full quick_start.
    """
    options: list[dict[str, str]] = []

    # LazyGraph: subgraph from cache on demand (50K+ files scale)
    when_lazy_graph = []
    if total_source_files >= 50_000:
        when_lazy_graph.append("very large codebase (50K+ source files)")
    if when_lazy_graph:
        options.append(
            {
                "option": "LazyGraph",
                "when": "; ".join(when_lazy_graph),
                "description": "Load subgraphs on demand from cache (load_around, load_file). Use instead of loading the full dependency graph.",
            }
        )
    else:
        options.append(
            {
                "option": "LazyGraph",
                "when": "50K+ source files",
                "description": "Load subgraphs on demand from cache. For smaller repos, analyze_deps (full graph) is fine.",
            }
        )

    # LazyEmbeddingGraph: scope-based graph + embeddings
    when_lazy_embed = []
    if is_monorepo:
        when_lazy_embed.append("monorepo")
    if total_source_files >= 5_000:
        when_lazy_embed.append("large codebase (5K+ source files)")
    if when_lazy_embed:
        options.append(
            {
                "option": "LazyEmbeddingGraph",
                "when": "; ".join(when_lazy_embed),
                "description": "Load scope with load_scope() or load_from_nodes(), compute_scoped_embeddings(), then find_similar(). Expand with expand_from_nodes().",
            }
        )
    else:
        options.append(
            {
                "option": "LazyEmbeddingGraph",
                "when": "monorepo or 5K+ source files",
                "description": "On-demand graph + embeddings by scope. For smaller repos, quick_start then semantic_analysis is simpler.",
            }
        )

    # Lazy embedding (SQLite + on-demand FAISS)
    when_faiss = []
    if is_monorepo:
        when_faiss.append("monorepo")
    if total_source_files >= 2_000:
        when_faiss.append("many source files")
    if when_faiss:
        options.append(
            {
                "option": "lazy embedding (scoped + on-demand FAISS)",
                "when": "; ".join(when_faiss),
                "description": "Embed specific dirs; embeddings stored in SQLite; FAISS rebuilt on-demand for search. Use scope/package params in semantic_analysis mode=embeddings.",
            }
        )
    else:
        options.append(
            {
                "option": "lazy embedding (scoped + on-demand FAISS)",
                "when": "monorepo or when you want incremental/scoped embedding",
                "description": "Embed by scope; FAISS rebuilt from SQLite when needed. Enables incremental updates and consistent chunk IDs.",
            }
        )

    return options


def _generate_workflow_guidance(
    suggested_ignores: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Generate workflow guidance for next steps.

    Returns:
        Tuple of (recommended_workflow, next_steps).
    """
    # Standard workflow sequence
    recommended_workflow = [
        "1. analyze_deps — Build dependency graph (pass suggested_ignores as exclude)",
        "2. codebase_health — Run health checks (dead code, cycles, duplicates)",
        "3. semantic_analysis mode=embeddings — Pre-compute embeddings (takes time, but makes search fast)",
        "4. semantic_analysis mode=search — Now searches are instant",
        "5. analyze_graph algorithm=pagerank — Find most critical code",
        "6. get_impact — Deep-dive on specific files",
    ]

    # Actionable next steps with tool names
    next_steps = [
        {
            "tool": "quick_start",
            "description": "Run steps 1-3 automatically (recommended for first-time analysis)",
        },
        {
            "tool": "analyze_deps",
            "description": f"Build dependency graph. Pass exclude={suggested_ignores[:3]}... to filter noise",
        },
        {
            "tool": "codebase_health",
            "description": "Quick health check: dead code, duplicates, cycles, bottlenecks",
        },
    ]

    return recommended_workflow, next_steps
