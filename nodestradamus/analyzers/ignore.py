"""Centralized ignore pattern management for Nodestradamus analyzers.

Provides consistent file/directory filtering across all analyzers with support
for .nodestradamusignore files and framework-aware defaults.

Configuration:
    - DEFAULT_IGNORES: Universal patterns (node_modules, .git, etc.)
    - FRAMEWORK_IGNORES: Framework-specific patterns based on project_scout detection
    - .nodestradamusignore: Per-repo customization using gitignore syntax
"""

from collections.abc import Callable
from pathlib import Path

from nodestradamus.logging import logger

# Universal ignore patterns - always excluded
DEFAULT_IGNORES: frozenset[str] = frozenset({
    # Version control
    ".git",
    ".svn",
    ".hg",
    # Dependencies
    "node_modules",
    "vendor",
    "bower_components",
    # Python
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "*.egg-info",
    ".eggs",
    # Virtual environments
    "venv",
    ".venv",
    "env",
    ".env",
    # Build outputs (generic)
    "dist",
    "build",
    # IDE
    ".idea",
    ".vscode",
    # Test coverage
    "coverage",
    ".coverage",
    "htmlcov",
    ".nyc_output",
})

# Framework-specific ignore patterns
# Key: framework name (as detected by project_scout)
# Value: list of additional patterns to ignore
FRAMEWORK_IGNORES: dict[str, list[str]] = {
    # JavaScript/TypeScript frameworks
    "next": [".next", "out"],
    "nuxt": [".nuxt", ".output"],
    "vite": [".vite"],
    "gatsby": [".cache", "public"],
    "remix": [".cache", "build"],
    "astro": [".astro"],
    "sveltekit": [".svelte-kit"],
    # Build tools
    "webpack": ["dist"],
    "parcel": [".parcel-cache"],
    "turbo": [".turbo"],
    "nx": [".nx"],
    # Python frameworks
    "pytest": [".pytest_cache"],
    "django": ["staticfiles", "media"],
    "flask": ["instance"],
    # Rust
    "rust": ["target"],
    "cargo": ["target"],
    # Other
    "docker": [".docker"],
}

# Language-specific patterns (based on detected languages)
LANGUAGE_IGNORES: dict[str, list[str]] = {
    "rust": ["target"],
    "go": ["bin", "pkg"],
    "java": ["target", "out", ".gradle"],
    "csharp": ["bin", "obj"],
}

NODESTRADAMUSIGNORE_FILENAME = ".nodestradamusignore"


def get_default_ignores() -> set[str]:
    """Get a mutable copy of default ignore patterns.

    Returns:
        Set of directory/file patterns to ignore.
    """
    return set(DEFAULT_IGNORES)


def get_framework_ignores(frameworks: list[str]) -> set[str]:
    """Get ignore patterns for detected frameworks.

    Args:
        frameworks: List of framework names from project_scout.

    Returns:
        Set of additional patterns based on frameworks.
    """
    patterns: set[str] = set()
    for framework in frameworks:
        framework_lower = framework.lower()
        if framework_lower in FRAMEWORK_IGNORES:
            patterns.update(FRAMEWORK_IGNORES[framework_lower])
    return patterns


def get_language_ignores(languages: dict[str, int]) -> set[str]:
    """Get ignore patterns for detected languages.

    Args:
        languages: Dict mapping language name to file count.

    Returns:
        Set of additional patterns based on languages.
    """
    patterns: set[str] = set()
    for language in languages:
        language_lower = language.lower()
        if language_lower in LANGUAGE_IGNORES:
            patterns.update(LANGUAGE_IGNORES[language_lower])
    return patterns


def generate_suggested_ignores(
    frameworks: list[str],
    languages: dict[str, int],
) -> list[str]:
    """Generate suggested ignore patterns based on project analysis.

    Combines default, framework, and language patterns into a deduplicated
    list suitable for LLM-generated .nodestradamusignore files.

    Args:
        frameworks: Detected frameworks from project_scout.
        languages: Language distribution from project_scout.

    Returns:
        Sorted list of suggested ignore patterns.
    """
    patterns = get_default_ignores()
    patterns.update(get_framework_ignores(frameworks))
    patterns.update(get_language_ignores(languages))
    return sorted(patterns)


def parse_nodestradamusignore(repo_path: Path) -> set[str]:
    """Parse .nodestradamusignore file if it exists.

    Supports gitignore-style syntax:
    - Lines starting with # are comments
    - Empty lines are ignored
    - Patterns are directory/file names or glob patterns
    - Lines starting with ! are negations (not yet implemented)

    Args:
        repo_path: Path to repository root.

    Returns:
        Set of patterns from .nodestradamusignore, empty if file doesn't exist.
    """
    ignore_file = repo_path / NODESTRADAMUSIGNORE_FILENAME
    if not ignore_file.exists():
        return set()

    patterns: set[str] = set()
    try:
        content = ignore_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Skip negation patterns for now (could implement later)
            if line.startswith("!"):
                logger.debug("  Negation patterns not yet supported: %s", line)
                continue
            # Remove trailing slashes for consistency
            patterns.add(line.rstrip("/"))
    except OSError as e:
        logger.warning("  Failed to read %s: %s", ignore_file, e)

    return patterns


def nodestradamusignore_exists(repo_path: Path) -> bool:
    """Check if .nodestradamusignore file exists in repository.

    Args:
        repo_path: Path to repository root.

    Returns:
        True if .nodestradamusignore exists.
    """
    return (repo_path / NODESTRADAMUSIGNORE_FILENAME).exists()


def load_ignore_patterns(
    repo_path: Path,
    frameworks: list[str] | None = None,
    languages: dict[str, int] | None = None,
) -> set[str]:
    """Load all applicable ignore patterns for a repository.

    Combines patterns from:
    1. Default ignores (always applied)
    2. Framework-specific ignores (if frameworks provided)
    3. Language-specific ignores (if languages provided)
    4. .nodestradamusignore file (if exists)

    Args:
        repo_path: Path to repository root.
        frameworks: Optional list of detected frameworks.
        languages: Optional dict of detected languages.

    Returns:
        Combined set of all ignore patterns.
    """
    patterns = get_default_ignores()

    if frameworks:
        patterns.update(get_framework_ignores(frameworks))

    if languages:
        patterns.update(get_language_ignores(languages))

    # Override/extend with .nodestradamusignore
    custom_patterns = parse_nodestradamusignore(repo_path)
    if custom_patterns:
        patterns.update(custom_patterns)
        logger.debug("  Loaded %d patterns from .nodestradamusignore", len(custom_patterns))

    return patterns


def should_ignore(path: Path, repo_path: Path, patterns: set[str]) -> bool:
    """Check if a path should be ignored.

    Matches against directory names in the path and handles both
    exact matches and simple glob patterns.

    Args:
        path: Path to check.
        repo_path: Repository root for relative path calculation.
        patterns: Set of ignore patterns.

    Returns:
        True if path should be ignored.
    """
    try:
        rel_path = path.relative_to(repo_path)
    except ValueError:
        # Path is outside repo
        return True

    parts = rel_path.parts

    for part in parts:
        # Exact match
        if part in patterns:
            return True
        # Hidden files/directories (starting with .)
        if part.startswith(".") and part not in {"."}:
            # Check if this specific hidden dir is in patterns
            if part in patterns:
                return True

    return False


def create_should_ignore_func(
    repo_path: Path,
    patterns: set[str] | None = None,
    frameworks: list[str] | None = None,
    languages: dict[str, int] | None = None,
) -> Callable[[Path], bool]:
    """Create a callable for checking if paths should be ignored.

    Useful for passing to filter() or iterating over files.

    Args:
        repo_path: Repository root path.
        patterns: Pre-computed patterns (if None, will load).
        frameworks: Optional frameworks for pattern loading.
        languages: Optional languages for pattern loading.

    Returns:
        Function that takes a Path and returns True if it should be ignored.
    """
    if patterns is None:
        patterns = load_ignore_patterns(repo_path, frameworks, languages)

    def _should_ignore(path: Path) -> bool:
        return should_ignore(path, repo_path, patterns)

    return _should_ignore


def generate_nodestradamusignore_content(
    frameworks: list[str],
    languages: dict[str, int],
) -> str:
    """Generate .nodestradamusignore file content with comments.

    Creates a well-documented ignore file suitable for LLM generation.

    Args:
        frameworks: Detected frameworks.
        languages: Detected languages.

    Returns:
        Formatted .nodestradamusignore file content.
    """
    lines = [
        "# .nodestradamusignore - Auto-generated by Nodestradamus",
        f"# Detected frameworks: {', '.join(frameworks) if frameworks else 'none'}",
        f"# Detected languages: {', '.join(languages.keys()) if languages else 'none'}",
        "",
        "# Dependencies",
        "node_modules/",
        "vendor/",
        "",
        "# Build outputs",
        "dist/",
        "build/",
    ]

    # Add framework-specific patterns
    framework_patterns = get_framework_ignores(frameworks)
    if framework_patterns:
        lines.append("")
        lines.append("# Framework-specific")
        for pattern in sorted(framework_patterns):
            lines.append(f"{pattern}/")

    # Add language-specific patterns
    language_patterns = get_language_ignores(languages)
    if language_patterns:
        lines.append("")
        lines.append("# Language-specific")
        for pattern in sorted(language_patterns):
            lines.append(f"{pattern}/")

    lines.extend([
        "",
        "# Caches",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        "coverage/",
        "",
        "# Virtual environments",
        "venv/",
        ".venv/",
        "",
        "# IDE",
        ".idea/",
        ".vscode/",
    ])

    return "\n".join(lines) + "\n"
