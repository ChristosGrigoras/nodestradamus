"""Duplicate code detection with exact line locations.

Finds identical and near-identical code blocks across the codebase.
"""

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from nodestradamus.analyzers.code_parser import EXTENSION_TO_LANGUAGE, SKIP_DIRS
from nodestradamus.models.graph import DuplicateBlock, DuplicateLocation


def _normalize_code(content: str) -> str:
    """Normalize code for comparison.

    Removes comments, normalizes whitespace, and strips empty lines.

    Args:
        content: Raw code content.

    Returns:
        Normalized code string.
    """
    lines = content.split("\n")
    normalized_lines = []

    in_multiline_comment = False

    for line in lines:
        # Strip leading/trailing whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle multiline comments /* ... */
        if "/*" in line:
            in_multiline_comment = True
            # Check if it ends on same line
            if "*/" in line:
                in_multiline_comment = False
                # Extract parts outside comment
                line = re.sub(r"/\*.*?\*/", "", line).strip()
                if not line:
                    continue
            else:
                continue

        if in_multiline_comment:
            if "*/" in line:
                in_multiline_comment = False
            continue

        # Strip inline comments (# for Python/Bash, // for JS/TS/Rust)
        # Be careful not to strip strings containing # or //
        # Simple heuristic: only strip if # or // is not inside quotes
        if "#" in line and not ('"' in line or "'" in line):
            line = line.split("#")[0].strip()
        if "//" in line and not ('"' in line or "'" in line):
            line = line.split("//")[0].strip()

        # Skip full-line comments (now empty after stripping)
        if not line:
            continue

        # Normalize whitespace within line
        line = re.sub(r"\s+", " ", line)

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of normalized content.

    Args:
        content: Normalized code content.

    Returns:
        First 16 characters of the hex digest.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _extract_code_blocks(
    filepath: Path,
    min_lines: int = 5,
) -> list[dict]:
    """Extract meaningful code blocks from a file.

    Extracts functions, classes, and multi-line constant definitions
    as potential duplicate candidates.

    Args:
        filepath: Path to the source file.
        min_lines: Minimum lines for a block to be considered.

    Returns:
        List of dicts with content, line_start, line_end, normalized, hash.
    """
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = content.split("\n")
    blocks = []

    # Simple heuristic: look for indented blocks starting with def/class/const/function
    # This is a simplified approach - the full version would use tree-sitter

    block_starters = {
        "python": ["def ", "class "],
        "typescript": ["function ", "const ", "class ", "interface ", "type ", "export function ", "export const ", "export class "],
        "javascript": ["function ", "const ", "class ", "export function ", "export const "],
        "rust": ["fn ", "struct ", "enum ", "impl ", "const ", "static ", "pub fn ", "pub struct "],
        "bash": ["function ", "() {"],  # Bash function patterns
    }

    ext = filepath.suffix.lower()
    language = EXTENSION_TO_LANGUAGE.get(ext, "")
    starters = block_starters.get(language, [])

    if not starters:
        return []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Check if this line starts a block
        is_block_start = any(stripped.startswith(s) for s in starters)

        if is_block_start:
            # Find the end of this block
            block_lines = [line]
            start_indent = len(line) - len(stripped)
            j = i + 1

            # For Python/Bash: track indentation
            # For others: track braces
            if language in ("python", "bash"):
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.strip():  # Non-empty line
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= start_indent and not next_line.strip().startswith(("#", "//")):
                            break
                    block_lines.append(next_line)
                    j += 1
            else:
                # Track brace depth
                brace_depth = line.count("{") - line.count("}")
                while j < len(lines) and brace_depth > 0:
                    next_line = lines[j]
                    brace_depth += next_line.count("{") - next_line.count("}")
                    block_lines.append(next_line)
                    j += 1
                # Include the closing brace line
                if j < len(lines):
                    block_lines.append(lines[j])
                    j += 1

            # Only include blocks with enough lines
            if len(block_lines) >= min_lines:
                block_content = "\n".join(block_lines)
                normalized = _normalize_code(block_content)

                # Only include if normalized version is substantial
                if normalized.count("\n") >= min_lines - 2:
                    blocks.append({
                        "content": block_content,
                        "line_start": i + 1,  # 1-based
                        "line_end": i + len(block_lines),
                        "normalized": normalized,
                        "hash": _compute_content_hash(normalized),
                    })

            i = j
        else:
            i += 1

    return blocks


def find_exact_duplicates(
    repo_path: Path,
    target_file: str | None = None,
    min_lines: int = 5,
    exclude: list[str] | None = None,
) -> list[DuplicateBlock]:
    """Find identical code blocks across the codebase.

    Extracts code blocks (functions, classes, constants), normalizes them,
    and groups by content hash to find exact duplicates.

    Args:
        repo_path: Path to the repository root.
        target_file: If provided, only find duplicates of blocks in this file.
        min_lines: Minimum lines for a block to be considered (default: 5).
        exclude: Directories to exclude (uses SKIP_DIRS if None).

    Returns:
        List of DuplicateBlock objects with all locations.
    """
    repo_path = Path(repo_path).resolve()
    skip_patterns = set(exclude) if exclude else SKIP_DIRS

    # Collect all code blocks from all files
    all_blocks: dict[str, list[dict]] = defaultdict(list)  # hash -> list of locations

    # Get all source files
    extensions = set(EXTENSION_TO_LANGUAGE.keys())
    files_to_process: list[Path] = []

    for filepath in repo_path.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in extensions:
            continue
        if any(part in skip_patterns for part in filepath.parts):
            continue
        files_to_process.append(filepath)

    # If target_file specified, we want to find duplicates OF blocks in that file
    target_hashes: set[str] | None = None
    if target_file:
        target_path = repo_path / target_file
        if target_path.exists():
            target_blocks = _extract_code_blocks(target_path, min_lines)
            target_hashes = {b["hash"] for b in target_blocks}

    # Process all files
    for filepath in files_to_process:
        try:
            rel_path = str(filepath.relative_to(repo_path))
        except ValueError:
            rel_path = str(filepath)

        blocks = _extract_code_blocks(filepath, min_lines)

        for block in blocks:
            # If target specified, only include blocks matching target hashes
            if target_hashes is not None and block["hash"] not in target_hashes:
                continue

            all_blocks[block["hash"]].append({
                "file": rel_path,
                "line_start": block["line_start"],
                "line_end": block["line_end"],
                "content": block["content"],
            })

    # Convert to DuplicateBlock objects (only include hashes with 2+ locations)
    results: list[DuplicateBlock] = []

    for content_hash, locations in all_blocks.items():
        if len(locations) < 2:
            continue

        # Create preview (first 3 lines)
        first_content = locations[0]["content"]
        preview_lines = first_content.split("\n")[:3]
        preview = "\n".join(preview_lines)
        if len(first_content.split("\n")) > 3:
            preview += "\n..."

        duplicate_locations = [
            DuplicateLocation(
                file=loc["file"],
                line_start=loc["line_start"],
                line_end=loc["line_end"],
                preview=preview,
            )
            for loc in locations
        ]

        results.append(
            DuplicateBlock(
                content_hash=content_hash[:8],
                similarity=1.0,  # Exact match
                locations=duplicate_locations,
            )
        )

    # Sort by number of locations (most duplicated first)
    results.sort(key=lambda x: len(x.locations), reverse=True)

    return results
