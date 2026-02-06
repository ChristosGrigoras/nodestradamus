"""Rules validation and conflict detection handlers.

Handlers for:
- validate_rules: Validate rule file structure, frontmatter, and references
- detect_rule_conflicts: Detect conflicts between rules (naming, testing, imports)
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

# Token budgets per rule type (based on meta-generator guidance)
TOKEN_BUDGETS = {
    "router": 150,
    "meta": 200,
    "default": 180,
}

# Frontmatter pattern
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Known conflict categories and their keywords
CONFLICT_CATEGORIES = {
    "naming_convention": {
        "keywords": [
            "snake_case",
            "camelCase",
            "PascalCase",
            "UPPER_SNAKE",
            "kebab-case",
        ],
        "description": "Naming convention conflicts",
    },
    "import_style": {
        "keywords": ["absolute import", "relative import", "import order"],
        "description": "Import style conflicts",
    },
    "error_handling": {
        "keywords": ["exception", "Result type", "error code", "try/except", "raise"],
        "description": "Error handling approach conflicts",
    },
    "documentation": {
        "keywords": ["docstring", "JSDoc", "comment", "Google style", "NumPy style"],
        "description": "Documentation style conflicts",
    },
    "testing": {
        "keywords": ["pytest", "unittest", "jest", "mocha", "test framework"],
        "description": "Testing framework conflicts",
    },
    "type_hints": {
        "keywords": ["type hint", "typing", "TypeScript", "strict mode"],
        "description": "Type annotation conflicts",
    },
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)."""
    return len(text) // 4


def _parse_frontmatter(content: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Parse YAML frontmatter from rule file.

    Returns:
        Tuple of (frontmatter_dict, body, errors)
    """
    errors = []
    match = FRONTMATTER_PATTERN.match(content)

    if not match:
        return None, content, ["Missing or invalid frontmatter (must start with ---)"]

    frontmatter_text = match.group(1)
    body = content[match.end() :]

    # Simple YAML parsing for the fields we care about
    frontmatter = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Handle boolean values
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False

            frontmatter[key] = value

    # Validate required fields
    if "description" not in frontmatter:
        errors.append("Missing 'description' in frontmatter")

    return frontmatter, body, errors


def _extract_rule_number(filename: str) -> int | None:
    """Extract the numeric prefix from a rule filename."""
    match = re.match(r"^(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def _check_file_references(body: str, rules_dir: Path, base_dir: Path) -> list[str]:
    """Check if referenced files exist."""
    errors = []

    # Pattern for file references in markdown
    file_patterns = [
        re.compile(r"\[.*?\]\((\.cursor/rules/[^)]+)\)"),  # Markdown links
        re.compile(r"`(\.cursor/rules/[^`]+)`"),  # Backtick references
        re.compile(r"For.*?:\s*`([^`]+\.mdc)`"),  # References like "For X: `file.mdc`"
    ]

    for pattern in file_patterns:
        for match in pattern.finditer(body):
            ref_path = match.group(1)
            # Normalize the path
            if ref_path.startswith(".cursor/rules/"):
                full_path = base_dir / ref_path
            else:
                full_path = rules_dir / ref_path

            if not full_path.exists():
                errors.append(f"Referenced file not found: {ref_path}")

    return errors


def _validate_rule_file(
    filepath: Path, rules_dir: Path, base_dir: Path
) -> dict[str, Any]:
    """Validate a single rule file."""
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}
    result: dict[str, Any] = {
        "file": filepath.name,
        "path": str(filepath),
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        errors.append(f"Cannot read file: {e}")
        return result

    # Parse frontmatter
    frontmatter, body, parse_errors = _parse_frontmatter(content)
    errors.extend(parse_errors)

    if frontmatter:
        info["frontmatter"] = frontmatter

    # Extract rule number
    rule_number = _extract_rule_number(filepath.stem)
    if rule_number is not None:
        info["rule_number"] = rule_number
    else:
        warnings.append("Filename doesn't start with a number")

    # Check token budget
    tokens = _estimate_tokens(content)
    info["estimated_tokens"] = tokens

    # Determine budget based on rule type
    budget = TOKEN_BUDGETS["default"]
    if "router" in filepath.stem.lower():
        budget = TOKEN_BUDGETS["router"]
    elif "meta" in filepath.stem.lower():
        budget = TOKEN_BUDGETS["meta"]

    if tokens > budget * 10:  # Allow 10x for full content (budget is per directive)
        warnings.append(f"High token count ({tokens}). Consider condensing content.")

    # Check for required sections
    if "# " not in body:
        warnings.append("Missing markdown header (# title)")

    # Check file references
    ref_errors = _check_file_references(body, rules_dir, base_dir)
    errors.extend(ref_errors)

    # Check for empty body
    if len(body.strip()) < 50:
        warnings.append("Rule body is very short")

    return result


def _validate_rules_directory(rules_dir: Path) -> dict[str, Any]:
    """Validate all rules in a directory."""
    total_files = 0
    valid_count = 0
    warning_count = 0
    error_count = 0
    files: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    base_dir = rules_dir.parent.parent  # .cursor/rules -> project root

    rule_files = sorted(rules_dir.glob("*.mdc"))
    total_files = len(rule_files)

    # Track rule numbers for duplicate detection
    rule_numbers: dict[int, list[str]] = {}

    for filepath in rule_files:
        file_result = _validate_rule_file(filepath, rules_dir, base_dir)
        files.append(file_result)

        # Track rule numbers
        if "rule_number" in file_result["info"]:
            num = file_result["info"]["rule_number"]
            if num not in rule_numbers:
                rule_numbers[num] = []
            rule_numbers[num].append(filepath.name)

        # Count results
        if file_result["errors"]:
            error_count += 1
        elif file_result["warnings"]:
            warning_count += 1
        else:
            valid_count += 1

    # Check for duplicate rule numbers
    for num, dup_files in rule_numbers.items():
        if len(dup_files) > 1:
            for file_result in files:
                if file_result["file"] in dup_files:
                    file_result["errors"].append(
                        f"Duplicate rule number {num}: {', '.join(dup_files)}"
                    )
                    error_count += 1
                    valid_count -= 1

    # Generate summary
    summary = {
        "total": total_files,
        "valid": valid_count,
        "with_warnings": warning_count,
        "with_errors": error_count,
        "rule_number_range": {
            "min": min(rule_numbers.keys()) if rule_numbers else None,
            "max": max(rule_numbers.keys()) if rule_numbers else None,
        },
    }

    return {
        "total_files": total_files,
        "valid": valid_count,
        "warnings": warning_count,
        "errors": error_count,
        "files": files,
        "summary": summary,
    }


def _parse_rule_for_conflicts(filepath: Path) -> dict[str, Any] | None:
    """Parse a rule file and extract relevant information for conflict detection."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Parse frontmatter
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return None

    frontmatter_text = match.group(1)
    body = content[match.end() :]

    # Extract frontmatter fields
    frontmatter = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")

    # Extract directives (lines starting with - in body)
    directives = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- ") and len(stripped) > 5:
            directives.append(stripped[2:])

    return {
        "file": filepath.name,
        "path": str(filepath),
        "frontmatter": frontmatter,
        "body": body.lower(),  # Lowercase for keyword matching
        "directives": directives,
        "globs": frontmatter.get("globs", "**/*"),
    }


def _find_category_mentions(rule: dict[str, Any]) -> dict[str, list[str]]:
    """Find which conflict categories are mentioned in a rule."""
    mentions: dict[str, list[str]] = {}
    body = rule["body"]

    for category, info in CONFLICT_CATEGORIES.items():
        found_keywords = []
        for keyword in info["keywords"]:
            if keyword.lower() in body:
                found_keywords.append(keyword)

        if found_keywords:
            mentions[category] = found_keywords

    return mentions


def _check_glob_overlap(glob1: str, glob2: str) -> bool:
    """Check if two glob patterns might overlap (simplified check)."""
    # If either is universal, they overlap
    if glob1 == "**/*" or glob2 == "**/*":
        return True

    # Check for common extensions
    ext_pattern = re.compile(r"\*\.(\w+)")
    exts1 = set(ext_pattern.findall(glob1))
    exts2 = set(ext_pattern.findall(glob2))

    if exts1 and exts2:
        return bool(exts1 & exts2)  # Overlap if shared extensions

    # If one has no extension filter, assume potential overlap
    return True


def _detect_category_conflicts(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect potential conflicts between rules based on category mentions."""
    conflicts = []

    # Build category mentions for each rule
    rule_categories: dict[str, dict[str, list[str]]] = {}
    for rule in rules:
        rule_categories[rule["file"]] = _find_category_mentions(rule)

    # Compare each pair of rules
    for i, rule1 in enumerate(rules):
        for rule2 in rules[i + 1 :]:
            # Check if globs overlap
            if not _check_glob_overlap(rule1["globs"], rule2["globs"]):
                continue

            # Check for shared categories with different keywords
            cats1 = rule_categories[rule1["file"]]
            cats2 = rule_categories[rule2["file"]]

            shared_categories = set(cats1.keys()) & set(cats2.keys())

            for category in shared_categories:
                keywords1 = set(cats1[category])
                keywords2 = set(cats2[category])

                # If they mention different keywords in same category, potential conflict
                different_keywords = keywords1.symmetric_difference(keywords2)
                if different_keywords:
                    conflicts.append(
                        {
                            "category": category,
                            "description": CONFLICT_CATEGORIES[category]["description"],
                            "severity": "warning",
                            "rule1": {
                                "file": rule1["file"],
                                "keywords": list(keywords1),
                            },
                            "rule2": {
                                "file": rule2["file"],
                                "keywords": list(keywords2),
                            },
                            "message": (
                                f"Both rules address {category} "
                                "but mention different approaches"
                            ),
                        }
                    )

    return conflicts


def _detect_directive_conflicts(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze specific directive conflicts."""
    conflicts = []

    # Known conflicting directive patterns
    directive_conflicts = [
        (r"use\s+snake_case", r"use\s+camelCase", "Naming convention mismatch"),
        (
            r"always\s+use\s+docstring",
            r"docstring.*optional",
            "Documentation requirement mismatch",
        ),
        (r"use\s+pytest", r"use\s+unittest", "Test framework mismatch"),
        (r"absolute\s+import", r"relative\s+import", "Import style mismatch"),
    ]

    for pattern1, pattern2, description in directive_conflicts:
        regex1 = re.compile(pattern1, re.IGNORECASE)
        regex2 = re.compile(pattern2, re.IGNORECASE)

        rules_match1 = []
        rules_match2 = []

        for rule in rules:
            if regex1.search(rule["body"]):
                rules_match1.append(rule["file"])
            if regex2.search(rule["body"]):
                rules_match2.append(rule["file"])

        if rules_match1 and rules_match2:
            conflicts.append(
                {
                    "category": "directive_conflict",
                    "description": description,
                    "severity": "error",
                    "rules_pattern1": rules_match1,
                    "rules_pattern2": rules_match2,
                    "message": f"Direct conflict: {description}",
                }
            )

    return conflicts


def _detect_all_conflicts(rules_dir: Path) -> dict[str, Any]:
    """Detect all conflicts in a rules directory."""
    # Parse all rule files
    rules = []
    for filepath in sorted(rules_dir.glob("*.mdc")):
        parsed = _parse_rule_for_conflicts(filepath)
        if parsed:
            rules.append(parsed)

    if not rules:
        return {
            "rules_analyzed": 0,
            "conflicts": [],
            "summary": {"total": 0, "errors": 0, "warnings": 0},
            "message": "No valid rules found to analyze",
        }

    # Detect conflicts
    category_conflicts = _detect_category_conflicts(rules)
    directive_conflicts = _detect_directive_conflicts(rules)
    all_conflicts = category_conflicts + directive_conflicts

    return {
        "rules_analyzed": len(rules),
        "conflicts": all_conflicts,
        "summary": {
            "total": len(all_conflicts),
            "errors": len([c for c in all_conflicts if c["severity"] == "error"]),
            "warnings": len([c for c in all_conflicts if c["severity"] == "warning"]),
        },
    }


def _discover_rules_path(repo_path: Path, rules_source: str | None) -> Path | None:
    """Discover rules directory based on source preference."""
    if rules_source == "cursor":
        path = repo_path / ".cursor" / "rules"
        return path if path.is_dir() else None
    elif rules_source == "claude":
        path = repo_path / ".claude"
        return path if path.is_dir() else None
    elif rules_source == "opencode":
        path = repo_path / "AGENTS.md"
        return path if path.is_file() else None

    # Auto-detect: try cursor, then claude
    cursor_path = repo_path / ".cursor" / "rules"
    if cursor_path.is_dir():
        return cursor_path

    claude_path = repo_path / ".claude"
    if claude_path.is_dir():
        return claude_path

    return None


async def handle_validate_rules(arguments: dict[str, Any]) -> str:
    """Handle validate_rules tool call.

    Validates rule file structure, frontmatter, token budgets, and references.

    Args:
        arguments: Tool arguments with repo_path and optional rules_source.

    Returns:
        JSON string with validation report.

    Raises:
        ValueError: If repo_path is not provided or rules not found.
    """
    repo_path_str = arguments.get("repo_path")
    if not repo_path_str:
        raise ValueError("repo_path is required")

    repo_path = Path(repo_path_str)
    if not repo_path.is_dir():
        raise ValueError(f"repo_path is not a directory: {repo_path}")

    # Get rules path
    rules_source = arguments.get("rules_source")
    custom_path = arguments.get("custom_rules_path")

    if custom_path:
        rules_dir = Path(custom_path)
    else:
        rules_dir = _discover_rules_path(repo_path, rules_source)

    if not rules_dir or not rules_dir.exists():
        return json.dumps(
            {
                "error": "No rules found",
                "searched": {
                    "cursor": str(repo_path / ".cursor" / "rules"),
                    "claude": str(repo_path / ".claude"),
                },
                "recommendation": "Create rules in .cursor/rules/ or .claude/",
            },
            indent=2,
        )

    # Handle single file (AGENTS.md)
    if rules_dir.is_file():
        return json.dumps(
            {
                "total_files": 1,
                "valid": 1,
                "warnings": 0,
                "errors": 0,
                "files": [
                    {
                        "file": rules_dir.name,
                        "path": str(rules_dir),
                        "errors": [],
                        "warnings": [],
                        "info": {"format": "opencode"},
                    }
                ],
                "summary": {
                    "total": 1,
                    "valid": 1,
                    "with_warnings": 0,
                    "with_errors": 0,
                    "format": "opencode",
                },
            },
            indent=2,
        )

    # Run validation
    result = await asyncio.to_thread(_validate_rules_directory, rules_dir)

    return json.dumps(result, indent=2)


async def handle_detect_conflicts(arguments: dict[str, Any]) -> str:
    """Handle detect_rule_conflicts tool call.

    Detects potential conflicts between rules (naming, testing, imports, etc.).

    Args:
        arguments: Tool arguments with repo_path and optional rules_source.

    Returns:
        JSON string with conflict report.

    Raises:
        ValueError: If repo_path is not provided or rules not found.
    """
    repo_path_str = arguments.get("repo_path")
    if not repo_path_str:
        raise ValueError("repo_path is required")

    repo_path = Path(repo_path_str)
    if not repo_path.is_dir():
        raise ValueError(f"repo_path is not a directory: {repo_path}")

    # Get rules path
    rules_source = arguments.get("rules_source")
    custom_path = arguments.get("custom_rules_path")

    if custom_path:
        rules_dir = Path(custom_path)
    else:
        rules_dir = _discover_rules_path(repo_path, rules_source)

    if not rules_dir or not rules_dir.exists():
        return json.dumps(
            {
                "error": "No rules found",
                "searched": {
                    "cursor": str(repo_path / ".cursor" / "rules"),
                    "claude": str(repo_path / ".claude"),
                },
                "recommendation": "Create rules in .cursor/rules/ or .claude/",
            },
            indent=2,
        )

    # Handle single file (can't detect conflicts with just one file)
    if rules_dir.is_file():
        return json.dumps(
            {
                "rules_analyzed": 1,
                "conflicts": [],
                "summary": {"total": 0, "errors": 0, "warnings": 0},
                "message": "Single-file rules (AGENTS.md) - conflict detection requires multiple rule files",
            },
            indent=2,
        )

    # Run conflict detection
    result = await asyncio.to_thread(_detect_all_conflicts, rules_dir)

    return json.dumps(result, indent=2)
