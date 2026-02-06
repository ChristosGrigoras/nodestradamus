#!/usr/bin/env python3
"""
Analyze TypeScript/JavaScript code dependencies using regex-based parsing.

Outputs a JSON graph of imports and exports for AI impact analysis.

Usage:
    python analyze_ts_deps.py src/
    python analyze_ts_deps.py src/ > .cursor/graph/ts-deps.json

This module can also be imported and used programmatically:
    from scripts.analyze_ts_deps import analyze_directory
    result = analyze_directory(Path("/path/to/repo"))
"""

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Regex patterns for TypeScript/JavaScript
IMPORT_PATTERNS = [
    # ES modules: import { x } from 'module'
    re.compile(r"""import\s+(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)?\s*(?:,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))?\s*from\s*['"]([^'"]+)['"]""", re.MULTILINE),
    # Side-effect imports: import 'module'
    re.compile(r"""import\s+['"]([^'"]+)['"]""", re.MULTILINE),
    # CommonJS require: require('module')
    re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE),
    # Dynamic imports: import('module')
    re.compile(r"""import\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE),
]

# Pattern for function/class definitions
FUNCTION_PATTERN = re.compile(
    r"""(?:export\s+)?(?:async\s+)?function\s+(\w+)""", re.MULTILINE
)
CLASS_PATTERN = re.compile(
    r"""(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?""", re.MULTILINE
)
ARROW_FUNCTION_PATTERN = re.compile(
    r"""(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>""", re.MULTILINE
)

EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}


def extract_imports(source: str) -> list[str]:
    """Extract all import paths from source code."""
    imports = []
    for pattern in IMPORT_PATTERNS:
        imports.extend(pattern.findall(source))
    return list(set(imports))


def extract_definitions(source: str, filepath: Path) -> list[dict[str, Any]]:
    """Extract function and class definitions from source code."""
    definitions = []

    # Find functions
    for match in FUNCTION_PATTERN.finditer(source):
        line_no = source[:match.start()].count("\n") + 1
        definitions.append({
            "name": f"{filepath}::{match.group(1)}",
            "type": "function",
            "line": line_no,
        })

    # Find arrow functions
    for match in ARROW_FUNCTION_PATTERN.finditer(source):
        line_no = source[:match.start()].count("\n") + 1
        definitions.append({
            "name": f"{filepath}::{match.group(1)}",
            "type": "function",
            "line": line_no,
        })

    # Find classes
    for match in CLASS_PATTERN.finditer(source):
        line_no = source[:match.start()].count("\n") + 1
        definitions.append({
            "name": f"{filepath}::{match.group(1)}",
            "type": "class",
            "line": line_no,
            "extends": match.group(2) if match.group(2) else None,
        })

    return definitions


def resolve_import(import_path: str, filepath: Path, base_dir: Path) -> str | None:
    """Attempt to resolve a relative import to a file path."""
    if import_path.startswith("."):
        # Relative import
        parent = filepath.parent
        resolved = (parent / import_path).resolve()

        # Try with extensions
        for ext in EXTENSIONS:
            candidate = resolved.with_suffix(ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(base_dir))
                except ValueError:
                    return str(candidate)

        # Try as directory with index file
        if resolved.is_dir():
            for ext in EXTENSIONS:
                index = resolved / f"index{ext}"
                if index.exists():
                    try:
                        return str(index.relative_to(base_dir))
                    except ValueError:
                        return str(index)

    # External package or unresolved
    return None


def analyze_file(filepath: Path, base_dir: Path) -> dict[str, Any]:
    """Analyze a single TypeScript/JavaScript file for dependencies."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(filepath)}

    try:
        rel_path = filepath.relative_to(base_dir)
    except ValueError:
        rel_path = filepath

    definitions = extract_definitions(source, rel_path)
    imports = extract_imports(source)

    edges = []
    for import_path in imports:
        resolved = resolve_import(import_path, filepath, base_dir)
        edges.append({
            "from": str(rel_path),
            "to": resolved if resolved else import_path,
            "type": "imports",
            "resolved": resolved is not None,
        })

    return {
        "definitions": definitions,
        "edges": edges,
        "imports": imports,
    }


def analyze_directory(directory: Path) -> dict[str, Any]:
    """Analyze all TypeScript/JavaScript files in a directory.

    Args:
        directory: Path to the directory to analyze.

    Returns:
        Dictionary containing nodes, edges, files, errors, and metadata.
        This is the core function for programmatic use.
    """
    directory = Path(directory).resolve()
    nodes = []
    edges = []
    file_nodes = []
    errors = []

    for filepath in directory.rglob("*"):
        # Skip non-files and non-matching extensions
        if not filepath.is_file():
            continue
        if filepath.suffix not in EXTENSIONS:
            continue

        # Skip common non-source directories
        skip_dirs = {".git", "node_modules", "dist", "build", ".next", "coverage", "__pycache__"}
        if any(part in skip_dirs for part in filepath.parts):
            continue

        try:
            rel_path = filepath.relative_to(directory)
        except ValueError:
            rel_path = filepath

        file_nodes.append(str(rel_path))

        result = analyze_file(filepath, directory)
        if "error" in result:
            errors.append(result)
        else:
            # Add ts: prefix and proper structure to definitions
            for defn in result["definitions"]:
                defn["id"] = f"ts:{defn['name']}"
                defn["file"] = str(rel_path)
                # Extract just the symbol name
                if "::" in defn["name"]:
                    defn["name"] = defn["name"].split("::")[-1]
            nodes.extend(result["definitions"])
            edges.extend(result["edges"])

    # Build node name lookup for resolution
    defined_names = {n["name"]: n["id"] for n in nodes}

    # Add inheritance edges for classes
    for node in nodes:
        if node["type"] == "class" and node.get("extends"):
            parent = node["extends"]
            edges.append({
                "from": node["id"],
                "to": defined_names.get(parent, parent),
                "type": "extends",
                "resolved": parent in defined_names,
            })

    # Update edges to use ts: prefix for file-level imports
    updated_edges = []
    for edge in edges:
        updated_edge = {
            "from": f"ts:{edge['from']}" if not edge["from"].startswith("ts:") else edge["from"],
            "to": edge["to"],
            "type": edge["type"],
            "resolved": edge["resolved"],
        }
        updated_edges.append(updated_edge)

    return {
        "nodes": nodes,
        "files": sorted(file_nodes),
        "edges": updated_edges,
        "errors": errors,
        "metadata": {
            "analyzer": "typescript",
            "version": "0.2.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
            "file_count": len(file_nodes),
        },
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_ts_deps.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = analyze_directory(directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
