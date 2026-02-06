"""Rule file parsing for Cursor, OpenCode, and Claude Code formats.

Supports:
- Cursor: .cursor/rules/*.mdc with YAML frontmatter (description, globs, alwaysApply)
- OpenCode: AGENTS.md or CLAUDE.md in project root (plain markdown)
- Claude Code: .claude/CLAUDE.md + .claude/rules/**/*.md with optional paths frontmatter

All formats are normalized to a common ParsedRule structure for comparison.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Frontmatter pattern (YAML between --- delimiters)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Patterns to extract code paths from rule bodies
CODE_PATH_PATTERNS = [
    # Backtick paths: `nodestradamus/analyzers/deps.py`, `src/utils.ts`
    re.compile(r"`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`"),
    # Backtick directories: `nodestradamus/`, `src/components/`
    re.compile(r"`([a-zA-Z0-9_\-./]+/)`"),
    # Table cell paths (markdown tables): | `path/file.py` | or | path/file.py |
    re.compile(r"\|\s*`?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`?\s*\|"),
    # Markdown links to files: [text](path/file.py)
    re.compile(r"\[.*?\]\(([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)\)"),
]

# Paths to skip (common false positives)
SKIP_PATH_PATTERNS = [
    re.compile(r"^https?://"),  # URLs
    re.compile(r"^www\."),  # Websites
    re.compile(r"^\d+\.\d+"),  # Version numbers like 3.12
    re.compile(r"^[a-z]+\.com$"),  # Domains
    re.compile(r"^\.\w+$"),  # Just extensions like .py
]

RuleSource = Literal["cursor", "opencode", "claude"]


@dataclass
class ParsedRule:
    """Normalized rule from any supported format."""

    source: RuleSource
    file: str  # Relative path from repo root
    body: str  # Full markdown body (lowercase for matching)
    original_body: str  # Original body preserving case
    frontmatter: dict[str, str | bool] = field(default_factory=dict)
    path_globs: list[str] = field(default_factory=list)  # Globs from frontmatter
    code_paths: list[str] = field(default_factory=list)  # Paths extracted from body


@dataclass
class RuleDiscoveryResult:
    """Result of discovering rules in a repository."""

    rules: list[ParsedRule]
    sources_checked: list[RuleSource]
    sources_found: list[RuleSource]


def parse_frontmatter(content: str) -> tuple[dict[str, str | bool], str]:
    """Parse YAML frontmatter from content.

    Args:
        content: Full file content.

    Returns:
        Tuple of (frontmatter_dict, body_without_frontmatter).
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = content[match.end() :]

    # Simple YAML parsing
    frontmatter: dict[str, str | bool] = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value_str = value.strip().strip('"').strip("'")

            # Handle boolean values
            if value_str.lower() == "true":
                frontmatter[key] = True
            elif value_str.lower() == "false":
                frontmatter[key] = False
            else:
                frontmatter[key] = value_str

    return frontmatter, body


def extract_code_paths(body: str) -> list[str]:
    """Extract code paths mentioned in rule body.

    Extracts all path-like references without validation. Path validation
    is done separately by find_stale_references().

    Args:
        body: Rule body text.

    Returns:
        List of unique code paths found.
    """
    paths: set[str] = set()

    for pattern in CODE_PATH_PATTERNS:
        for match in pattern.finditer(body):
            path = match.group(1)

            # Skip false positives
            if any(skip.match(path) for skip in SKIP_PATH_PATTERNS):
                continue

            # Skip very short paths (likely not real paths)
            if len(path) < 4:
                continue

            # Normalize path (remove leading ./ or /)
            path = path.lstrip("./")

            paths.add(path)

    return sorted(paths)


def parse_cursor_rule(filepath: Path, repo_path: Path) -> ParsedRule | None:
    """Parse a Cursor .mdc rule file.

    Args:
        filepath: Path to the .mdc file.
        repo_path: Repository root for relative path calculation.

    Returns:
        ParsedRule or None if parsing fails.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    frontmatter, body = parse_frontmatter(content)

    # Cursor rules require frontmatter
    if not frontmatter:
        # Still parse but note it's incomplete
        pass

    # Extract globs from frontmatter
    globs_value = frontmatter.get("globs", "")
    path_globs = []
    if isinstance(globs_value, str) and globs_value:
        # Handle comma-separated or single glob
        path_globs = [g.strip() for g in globs_value.split(",") if g.strip()]

    relative_path = str(filepath.relative_to(repo_path))
    code_paths = extract_code_paths(body)

    return ParsedRule(
        source="cursor",
        file=relative_path,
        body=body.lower(),
        original_body=body,
        frontmatter=frontmatter,
        path_globs=path_globs,
        code_paths=code_paths,
    )


def parse_opencode_rule(filepath: Path, repo_path: Path) -> ParsedRule | None:
    """Parse an OpenCode AGENTS.md or CLAUDE.md file.

    Args:
        filepath: Path to the markdown file.
        repo_path: Repository root for relative path calculation.

    Returns:
        ParsedRule or None if parsing fails.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # OpenCode files typically don't have frontmatter, but check anyway
    frontmatter, body = parse_frontmatter(content)

    relative_path = str(filepath.relative_to(repo_path))
    code_paths = extract_code_paths(body)

    return ParsedRule(
        source="opencode",
        file=relative_path,
        body=body.lower(),
        original_body=body,
        frontmatter=frontmatter,
        path_globs=[],  # OpenCode doesn't use globs
        code_paths=code_paths,
    )


def parse_claude_rule(filepath: Path, repo_path: Path) -> ParsedRule | None:
    """Parse a Claude Code rule file (.claude/CLAUDE.md or .claude/rules/*.md).

    Args:
        filepath: Path to the markdown file.
        repo_path: Repository root for relative path calculation.

    Returns:
        ParsedRule or None if parsing fails.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    frontmatter, body = parse_frontmatter(content)

    # Claude Code uses 'paths' in frontmatter for conditional rules
    paths_value = frontmatter.get("paths", "")
    path_globs = []
    if isinstance(paths_value, str) and paths_value:
        path_globs = [g.strip() for g in paths_value.split(",") if g.strip()]

    relative_path = str(filepath.relative_to(repo_path))
    code_paths = extract_code_paths(body)

    return ParsedRule(
        source="claude",
        file=relative_path,
        body=body.lower(),
        original_body=body,
        frontmatter=frontmatter,
        path_globs=path_globs,
        code_paths=code_paths,
    )


def discover_cursor_rules(repo_path: Path) -> list[ParsedRule]:
    """Discover and parse Cursor rules from .cursor/rules/*.mdc.

    Args:
        repo_path: Repository root.

    Returns:
        List of parsed rules.
    """
    rules: list[ParsedRule] = []
    rules_dir = repo_path / ".cursor" / "rules"

    if not rules_dir.is_dir():
        return rules

    for filepath in sorted(rules_dir.glob("*.mdc")):
        rule = parse_cursor_rule(filepath, repo_path)
        if rule:
            rules.append(rule)

    return rules


def discover_opencode_rules(repo_path: Path) -> list[ParsedRule]:
    """Discover and parse OpenCode rules (AGENTS.md, CLAUDE.md fallback).

    Args:
        repo_path: Repository root.

    Returns:
        List of parsed rules.
    """
    rules: list[ParsedRule] = []

    # Check AGENTS.md first (OpenCode primary)
    agents_file = repo_path / "AGENTS.md"
    if agents_file.is_file():
        rule = parse_opencode_rule(agents_file, repo_path)
        if rule:
            rules.append(rule)

    # Check CLAUDE.md as fallback (OpenCode compatibility)
    claude_file = repo_path / "CLAUDE.md"
    if claude_file.is_file():
        rule = parse_opencode_rule(claude_file, repo_path)
        if rule:
            rules.append(rule)

    return rules


def discover_claude_rules(repo_path: Path) -> list[ParsedRule]:
    """Discover and parse Claude Code rules (.claude/CLAUDE.md + .claude/rules/**/*.md).

    Args:
        repo_path: Repository root.

    Returns:
        List of parsed rules.
    """
    rules: list[ParsedRule] = []
    claude_dir = repo_path / ".claude"

    if not claude_dir.is_dir():
        return rules

    # Main CLAUDE.md in .claude/
    main_file = claude_dir / "CLAUDE.md"
    if main_file.is_file():
        rule = parse_claude_rule(main_file, repo_path)
        if rule:
            rules.append(rule)

    # Rules in .claude/rules/**/*.md
    rules_dir = claude_dir / "rules"
    if rules_dir.is_dir():
        for filepath in sorted(rules_dir.rglob("*.md")):
            rule = parse_claude_rule(filepath, repo_path)
            if rule:
                rules.append(rule)

    return rules


def discover_rules(
    repo_path: Path,
    sources: list[RuleSource] | None = None,
    custom_path: str | None = None,
) -> RuleDiscoveryResult:
    """Discover and parse rules from all configured sources.

    Args:
        repo_path: Repository root.
        sources: Which sources to check. Default: ["cursor", "opencode", "claude"].
        custom_path: Optional custom path (file or directory) to use instead of sources.

    Returns:
        RuleDiscoveryResult with all parsed rules and metadata.
    """
    if custom_path:
        # Use custom path only
        custom = Path(custom_path)
        if not custom.is_absolute():
            custom = repo_path / custom_path

        rules: list[ParsedRule] = []
        if custom.is_file():
            # Single file - try to parse based on extension
            if custom.suffix == ".mdc":
                rule = parse_cursor_rule(custom, repo_path)
            else:
                rule = parse_opencode_rule(custom, repo_path)
            if rule:
                rules.append(rule)
        elif custom.is_dir():
            # Directory - parse all markdown/mdc files
            for ext in ["*.mdc", "*.md"]:
                for filepath in sorted(custom.rglob(ext)):
                    if filepath.suffix == ".mdc":
                        rule = parse_cursor_rule(filepath, repo_path)
                    else:
                        rule = parse_opencode_rule(filepath, repo_path)
                    if rule:
                        rules.append(rule)

        return RuleDiscoveryResult(
            rules=rules,
            sources_checked=["cursor"],  # Custom path counts as generic
            sources_found=["cursor"] if rules else [],
        )

    # Default sources
    if sources is None:
        sources = ["cursor", "opencode", "claude"]

    all_rules: list[ParsedRule] = []
    sources_found: list[RuleSource] = []

    if "cursor" in sources:
        cursor_rules = discover_cursor_rules(repo_path)
        if cursor_rules:
            all_rules.extend(cursor_rules)
            sources_found.append("cursor")

    if "opencode" in sources:
        opencode_rules = discover_opencode_rules(repo_path)
        if opencode_rules:
            all_rules.extend(opencode_rules)
            sources_found.append("opencode")

    if "claude" in sources:
        claude_rules = discover_claude_rules(repo_path)
        if claude_rules:
            all_rules.extend(claude_rules)
            sources_found.append("claude")

    return RuleDiscoveryResult(
        rules=all_rules,
        sources_checked=sources,
        sources_found=sources_found,
    )


def check_path_coverage(
    rules: list[ParsedRule],
    target_paths: list[str],
) -> dict[str, list[str]]:
    """Check which target paths are mentioned in rules.

    Args:
        rules: List of parsed rules.
        target_paths: Paths to check coverage for.

    Returns:
        Dict mapping each target path to list of rule files that mention it.
    """
    coverage: dict[str, list[str]] = {}

    for path in target_paths:
        mentioning_rules = []
        path_lower = path.lower()

        for rule in rules:
            # Check if path appears in rule body
            if path_lower in rule.body:
                mentioning_rules.append(rule.file)
            # Check if path is in extracted code paths
            elif path in rule.code_paths:
                mentioning_rules.append(rule.file)
            # Check if parent directory is mentioned
            else:
                path_parts = path.split("/")
                for i in range(len(path_parts)):
                    parent = "/".join(path_parts[: i + 1])
                    if parent.lower() in rule.body:
                        mentioning_rules.append(rule.file)
                        break

        coverage[path] = mentioning_rules

    return coverage


def find_stale_references(
    rules: list[ParsedRule],
    repo_path: Path,
) -> list[dict[str, str]]:
    """Find code paths in rules that don't exist in the repo.

    Args:
        rules: List of parsed rules.
        repo_path: Repository root.

    Returns:
        List of stale references with rule file and path.
    """
    stale: list[dict[str, str]] = []

    for rule in rules:
        for code_path in rule.code_paths:
            full_path = repo_path / code_path
            # Check if file or directory exists
            if not full_path.exists():
                # Also check if it's a prefix of any existing path
                parent = full_path.parent
                if parent.exists():
                    # Check if any sibling starts with this name
                    name = full_path.name
                    if not any(p.name.startswith(name.split(".")[0]) for p in parent.iterdir()):
                        stale.append({
                            "rule": rule.file,
                            "path": code_path,
                            "source": rule.source,
                        })
                else:
                    stale.append({
                        "rule": rule.file,
                        "path": code_path,
                        "source": rule.source,
                    })

    return stale
