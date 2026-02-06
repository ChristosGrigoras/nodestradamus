"""Documentation analyzer for detecting stale references and coverage gaps.

Parses markdown files to extract code references (file paths, function names,
code blocks) and validates them against the actual codebase.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nodestradamus.analyzers.code_parser import (
    EXTENSION_TO_LANGUAGE,
    SKIP_DIRS,
)
from nodestradamus.logging import logger

# =============================================================================
# ALLOWLISTS FOR REDUCING FALSE POSITIVES (E14)
# =============================================================================

# Common developer tools and CLI commands that appear in docs
TOOL_ALLOWLIST = frozenset({
    # Python tools
    "pip", "pip3", "pipx", "uv", "poetry", "pdm", "hatch", "flit",
    "pytest", "mypy", "ruff", "black", "isort", "flake8", "pylint",
    "python", "python3", "ipython", "jupyter", "uvicorn", "gunicorn",
    "pydantic", "fastapi", "flask", "django", "celery", "alembic",
    # JavaScript/Node tools
    "npm", "npx", "yarn", "pnpm", "bun", "deno", "node", "nodejs",
    "eslint", "prettier", "tsc", "tsx", "vite", "webpack", "esbuild",
    "react", "vue", "angular", "next", "nuxt", "remix", "astro",
    # Rust tools
    "cargo", "rustc", "rustup", "clippy", "rustfmt",
    # General dev tools
    "git", "docker", "make", "cmake", "gradle", "maven",
    "curl", "wget", "jq", "yq", "sed", "awk", "grep", "find",
    "ssh", "scp", "rsync", "tar", "zip", "unzip", "gzip",
    # Editors and IDEs
    "vim", "nvim", "emacs", "code", "cursor", "subl",
    # Cloud/infra
    "aws", "gcloud", "az", "kubectl", "helm", "terraform", "pulumi",
})

# Python keywords and builtins that shouldn't be flagged as stale
PYTHON_KEYWORDS = frozenset({
    # Keywords
    "True", "False", "None", "and", "or", "not", "is", "in",
    "if", "elif", "else", "for", "while", "break", "continue",
    "def", "class", "return", "yield", "raise", "try", "except",
    "finally", "with", "as", "import", "from", "pass", "lambda",
    "global", "nonlocal", "assert", "del", "async", "await",
    # Common builtins
    "print", "len", "range", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "type", "isinstance", "hasattr", "getattr",
    "setattr", "open", "input", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "min", "max", "sum", "any", "all", "abs",
    "round", "format", "repr", "id", "hash", "callable", "iter", "next",
    "super", "object", "property", "staticmethod", "classmethod",
    # Common type hints
    "Optional", "Union", "List", "Dict", "Set", "Tuple", "Callable",
    "Any", "TypeVar", "Generic", "Literal", "Final", "ClassVar",
})

# Shell commands commonly shown in documentation
SHELL_COMMANDS = frozenset({
    "cd", "ls", "pwd", "mkdir", "rm", "cp", "mv", "cat", "head", "tail",
    "echo", "export", "source", "alias", "which", "whereis", "man",
    "ps", "kill", "top", "htop", "df", "du", "free", "chmod", "chown",
    "ln", "touch", "nano", "vi", "less", "more", "xargs", "tee",
    "sort", "uniq", "wc", "cut", "tr", "diff", "patch",
    "env", "printenv", "exit", "clear", "history", "sudo",
})

# Common generic strings that are not code references
GENERIC_STRINGS = frozenset({
    "example", "Example", "test", "Test", "TODO", "FIXME", "NOTE",
    "error", "Error", "warning", "Warning", "info", "Info", "debug", "Debug",
    "name", "value", "key", "data", "result", "output", "input",
    "config", "Config", "options", "Options", "settings", "Settings",
    "foo", "bar", "baz", "qux", "hello", "world",
})

# Combined allowlist for quick lookup
DOC_ALLOWLIST = TOOL_ALLOWLIST | PYTHON_KEYWORDS | SHELL_COMMANDS | GENERIC_STRINGS


@dataclass
class DocReference:
    """A reference from documentation to code."""

    doc_file: str  # Relative path to the doc file
    line: int  # Line number in the doc file
    ref_type: str  # "code_block", "inline_code", "file_link", "function_ref"
    raw_text: str  # The original text from the doc
    resolved_to: str | None = None  # Resolved target (file path or symbol)
    exists: bool = False  # Whether the reference is valid
    confidence: str = "medium"  # "high", "medium", "low" - how sure we are it's stale


@dataclass
class DocAnalysisResult:
    """Result of documentation analysis."""

    total_docs: int
    total_references: int
    valid_references: int
    stale_references: list[DocReference]
    undocumented_exports: list[str]  # Exported symbols without doc references
    coverage: float  # Percentage of exports that are documented
    metadata: dict[str, Any] = field(default_factory=dict)


def _find_markdown_files(directory: Path) -> list[Path]:
    """Find all markdown files in a directory.

    Args:
        directory: Root directory to search.

    Returns:
        List of markdown file paths.
    """
    markdown_extensions = {".md", ".mdx", ".mdc"}
    files = []

    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in markdown_extensions:
            continue
        if any(part in SKIP_DIRS for part in filepath.parts):
            continue
        files.append(filepath)

    return files


def _is_in_allowlist(text: str) -> bool:
    """Check if a text string is in the documentation allowlists.

    Args:
        text: The text to check.

    Returns:
        True if the text should be ignored (in allowlist).
    """
    # Direct match
    if text in DOC_ALLOWLIST:
        return True

    # Check base word (before parentheses for function calls)
    base = text.split("(")[0].strip()
    if base in DOC_ALLOWLIST:
        return True

    # Check if it's a command with arguments (e.g., "pip install xyz")
    first_word = text.split()[0] if " " in text else text
    if first_word in TOOL_ALLOWLIST or first_word in SHELL_COMMANDS:
        return True

    # Check for package names with dashes/underscores (e.g., "langchain-core")
    # These are typically not code references in the analyzed repo
    if "-" in text and text.replace("-", "").replace("_", "").isalnum():
        # Skip common package name patterns like "package-name"
        if not text.endswith((".py", ".ts", ".js", ".rs")):
            return True

    return False


def _extract_inline_code(content: str, doc_file: str) -> list[DocReference]:
    """Extract inline code references from markdown content.

    Matches patterns like `function_name`, `path/to/file.py`, `ClassName`.
    Filters out common tools, keywords, and commands (E14).

    Args:
        content: Markdown file content.
        doc_file: Relative path to the doc file.

    Returns:
        List of DocReference objects for inline code.
    """
    refs = []
    # Match inline code: `code`
    pattern = re.compile(r"`([^`\n]+)`")

    for i, line in enumerate(content.split("\n"), 1):
        for match in pattern.finditer(line):
            raw_text = match.group(1).strip()
            # Skip if it looks like a command or shell snippet
            if raw_text.startswith(("$", "#", "//", "/*", "<!--")):
                continue
            # Skip if it's a very short string (likely not a reference)
            if len(raw_text) < 2:
                continue

            # Skip if in allowlist (E14: reduce false positives)
            if _is_in_allowlist(raw_text):
                continue

            # Determine reference type
            if "/" in raw_text or raw_text.endswith((".py", ".ts", ".js", ".rs", ".sql")):
                ref_type = "file_link"
            elif "(" in raw_text and ")" in raw_text:
                ref_type = "function_ref"
            elif raw_text[0].isupper() and "_" not in raw_text:
                ref_type = "class_ref"
            else:
                ref_type = "inline_code"

            refs.append(
                DocReference(
                    doc_file=doc_file,
                    line=i,
                    ref_type=ref_type,
                    raw_text=raw_text,
                )
            )

    return refs


def _extract_code_blocks(content: str, doc_file: str) -> list[DocReference]:
    """Extract code block references from markdown content.

    Matches fenced code blocks with file path annotations like:
    ```python:path/to/file.py
    or
    ```typescript:src/utils.ts

    Args:
        content: Markdown file content.
        doc_file: Relative path to the doc file.

    Returns:
        List of DocReference objects for code blocks with file paths.
    """
    refs = []
    # Match code blocks with file paths: ```lang:path/file.ext or ```startLine:endLine:path
    pattern = re.compile(
        r"```(?:(\w+):)?(?:(\d+):(\d+):)?([^\s`]+\.(?:py|ts|tsx|js|jsx|rs|sql|sh|json|md))",
        re.MULTILINE,
    )

    for i, line in enumerate(content.split("\n"), 1):
        for match in pattern.finditer(line):
            file_path = match.group(4)
            refs.append(
                DocReference(
                    doc_file=doc_file,
                    line=i,
                    ref_type="code_block",
                    raw_text=file_path,
                )
            )

    return refs


def _extract_markdown_links(content: str, doc_file: str) -> list[DocReference]:
    """Extract markdown file links from content.

    Matches patterns like [text](path/to/file.py) or [text](./relative/path.ts).

    Args:
        content: Markdown file content.
        doc_file: Relative path to the doc file.

    Returns:
        List of DocReference objects for file links.
    """
    refs = []
    # Match markdown links to code files
    code_extensions = "|".join(ext.lstrip(".") for ext in EXTENSION_TO_LANGUAGE.keys())
    pattern = re.compile(rf"\[([^\]]+)\]\(([^)]+\.(?:{code_extensions}))\)")

    for i, line in enumerate(content.split("\n"), 1):
        for match in pattern.finditer(line):
            file_path = match.group(2)
            # Skip external URLs
            if file_path.startswith(("http://", "https://", "ftp://")):
                continue
            refs.append(
                DocReference(
                    doc_file=doc_file,
                    line=i,
                    ref_type="file_link",
                    raw_text=file_path,
                )
            )

    return refs


def extract_doc_references(doc_path: Path, base_dir: Path) -> list[DocReference]:
    """Extract all code references from a markdown file.

    Args:
        doc_path: Path to the markdown file.
        base_dir: Repository root for relative paths.

    Returns:
        List of DocReference objects.
    """
    try:
        content = doc_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Failed to read doc file %s: %s", doc_path, e)
        return []

    try:
        rel_path = str(doc_path.relative_to(base_dir))
    except ValueError:
        rel_path = str(doc_path)

    refs = []
    refs.extend(_extract_inline_code(content, rel_path))
    refs.extend(_extract_code_blocks(content, rel_path))
    refs.extend(_extract_markdown_links(content, rel_path))

    return refs


def validate_references(
    refs: list[DocReference],
    base_dir: Path,
    code_symbols: set[str] | None = None,
) -> list[DocReference]:
    """Validate documentation references against the codebase.

    Args:
        refs: List of DocReference objects to validate.
        base_dir: Repository root for resolving file paths.
        code_symbols: Optional set of known symbols (function/class names).

    Returns:
        List of DocReference objects with validation results and confidence.
    """
    validated = []

    for ref in refs:
        ref_copy = DocReference(
            doc_file=ref.doc_file,
            line=ref.line,
            ref_type=ref.ref_type,
            raw_text=ref.raw_text,
        )

        if ref.ref_type in ("file_link", "code_block"):
            # Normalize path
            raw_path = ref.raw_text.lstrip("./")
            # Handle startLine:endLine:path format
            if ":" in raw_path and not raw_path.startswith(("http:", "https:")):
                parts = raw_path.split(":")
                if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                    raw_path = parts[2]
                elif len(parts) == 2 and not parts[0].isdigit():
                    # lang:path format
                    raw_path = parts[1]

            candidate = base_dir / raw_path
            if candidate.exists():
                ref_copy.exists = True
                ref_copy.resolved_to = raw_path
                ref_copy.confidence = "high"
            else:
                ref_copy.exists = False
                ref_copy.resolved_to = None
                # High confidence stale: explicit file path doesn't exist
                ref_copy.confidence = "high"

        elif ref.ref_type in ("function_ref", "class_ref", "inline_code"):
            # Extract symbol name (strip parentheses for function calls)
            symbol = ref.raw_text.split("(")[0].strip()

            if code_symbols:
                if symbol in code_symbols:
                    ref_copy.exists = True
                    ref_copy.resolved_to = symbol
                    ref_copy.confidence = "high"
                else:
                    ref_copy.exists = False
                    # Set confidence based on ref_type
                    if ref.ref_type == "function_ref":
                        ref_copy.confidence = "medium"  # Explicit function call
                    elif ref.ref_type == "class_ref":
                        ref_copy.confidence = "medium"  # Explicit class reference
                    else:
                        ref_copy.confidence = "low"  # Generic inline code
            else:
                # Without symbol info, mark as unknown
                ref_copy.exists = True  # Assume valid if we can't verify
                ref_copy.resolved_to = symbol
                ref_copy.confidence = "low"

        validated.append(ref_copy)

    return validated


def analyze_docs(
    repo_path: str | Path,
    docs_path: str | None = None,
    include_readme: bool = True,
) -> DocAnalysisResult:
    """Analyze documentation for stale references and coverage.

    Args:
        repo_path: Path to the repository root.
        docs_path: Optional subdirectory for docs (default: looks for docs/).
        include_readme: Whether to include README.md files.

    Returns:
        DocAnalysisResult with findings.
    """
    base_dir = Path(repo_path).resolve()

    # Find markdown files
    doc_files: list[Path] = []

    # Check docs subdirectory
    if docs_path:
        docs_dir = base_dir / docs_path
        if docs_dir.exists():
            doc_files.extend(_find_markdown_files(docs_dir))
    else:
        # Try common doc directories
        found_docs_dir = False
        for dir_name in ("docs", "doc", "documentation"):
            docs_dir = base_dir / dir_name
            if docs_dir.exists():
                doc_files.extend(_find_markdown_files(docs_dir))
                found_docs_dir = True

        # If no docs/ directory, scan root for markdown files (excluding README)
        if not found_docs_dir:
            for md_file in base_dir.glob("*.md"):
                if md_file.name.lower() != "readme.md" and md_file not in doc_files:
                    doc_files.append(md_file)

    # Include README files
    if include_readme:
        for readme in base_dir.glob("**/README.md"):
            if any(part in SKIP_DIRS for part in readme.parts):
                continue
            if readme not in doc_files:
                doc_files.append(readme)

    # Also include .cursor/rules/ if present
    cursor_rules = base_dir / ".cursor" / "rules"
    if cursor_rules.exists():
        doc_files.extend(_find_markdown_files(cursor_rules))

    logger.info("  analyze_docs: found %d doc files", len(doc_files))

    # Extract all references
    all_refs: list[DocReference] = []
    for doc_file in doc_files:
        all_refs.extend(extract_doc_references(doc_file, base_dir))

    logger.info("  analyze_docs: extracted %d references", len(all_refs))

    # Build set of known symbols from the codebase
    code_symbols: set[str] = set()
    try:
        from nodestradamus.analyzers.deps import analyze_deps

        G = analyze_deps(str(base_dir))
        for _node_id, attrs in G.nodes(data=True):
            name = attrs.get("name", "")
            if name:
                code_symbols.add(name)
    except Exception as e:
        logger.warning("  Failed to build symbol set: %s", e)

    # Validate references
    validated = validate_references(all_refs, base_dir, code_symbols)

    # Find stale references
    stale = [ref for ref in validated if not ref.exists]
    valid_count = len(validated) - len(stale)

    # Find undocumented exports
    documented_symbols = {
        ref.resolved_to for ref in validated if ref.resolved_to and ref.exists
    }
    undocumented = [
        symbol for symbol in code_symbols if symbol not in documented_symbols
    ]
    # Filter to likely important symbols (exported functions/classes)
    # This is a heuristic - we include symbols that look like public APIs
    important_undocumented = [
        s for s in undocumented
        if not s.startswith("_") and not s.startswith("test_")
    ][:50]  # Limit to avoid overwhelming output

    # Calculate coverage
    if code_symbols:
        coverage = len(documented_symbols) / len(code_symbols) * 100
    else:
        coverage = 0.0

    return DocAnalysisResult(
        total_docs=len(doc_files),
        total_references=len(validated),
        valid_references=valid_count,
        stale_references=stale,
        undocumented_exports=important_undocumented,
        coverage=round(coverage, 1),
        metadata={
            "doc_files": [str(f.relative_to(base_dir)) for f in doc_files],
            "docs_path": docs_path,
            "include_readme": include_readme,
        },
    )
