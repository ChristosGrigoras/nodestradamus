#!/usr/bin/env python3
"""
Analyze Python code dependencies using the ast module.

Outputs a JSON graph of function/class calls for AI impact analysis.

Usage:
    python analyze_python_deps.py src/
    python analyze_python_deps.py src/ > .cursor/graph/python-deps.json

This module can also be imported and used programmatically:
    from scripts.analyze_python_deps import analyze_directory
    result = analyze_directory(Path("/path/to/repo"))
"""

import ast
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Directories to skip during analysis
SKIP_DIRS = frozenset({"venv", "node_modules", "__pycache__", ".git", ".venv", "env", ".env"})


def extract_calls(node: ast.AST) -> list[str]:
    """Extract all function/method calls from an AST node."""
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return calls


def analyze_file(filepath: Path, base_dir: Path | None = None) -> dict[str, Any]:
    """Analyze a single Python file for dependencies.

    Args:
        filepath: Path to the Python file to analyze.
        base_dir: Optional base directory for relative path calculation.

    Returns:
        Dictionary with 'definitions' and 'edges', or 'error' if parsing failed.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(filepath)}

    # Use relative path if base_dir provided
    rel_path = filepath.relative_to(base_dir) if base_dir else filepath

    definitions = []
    edges = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"py:{rel_path}::{node.name}"
            definitions.append({
                "id": func_id,
                "name": node.name,
                "type": "function",
                "file": str(rel_path),
                "line": node.lineno,
            })
            for call in extract_calls(node):
                edges.append({
                    "from": func_id,
                    "to": call,
                    "type": "calls",
                })

        elif isinstance(node, ast.ClassDef):
            class_id = f"py:{rel_path}::{node.name}"
            definitions.append({
                "id": class_id,
                "name": node.name,
                "type": "class",
                "file": str(rel_path),
                "line": node.lineno,
            })
            for base in node.bases:
                if isinstance(base, ast.Name):
                    edges.append({
                        "from": class_id,
                        "to": base.id,
                        "type": "inherits",
                    })

    return {"definitions": definitions, "edges": edges}


def analyze_directory(directory: Path) -> dict[str, Any]:
    """Analyze all Python files in a directory.

    Args:
        directory: Path to the directory to analyze.

    Returns:
        Dictionary containing nodes, edges, errors, and metadata.
        This is the core function for programmatic use.
    """
    directory = Path(directory).resolve()
    nodes = []
    edges = []
    errors = []
    file_count = 0

    for filepath in directory.rglob("*.py"):
        # Skip common non-source directories
        if any(part.startswith(".") or part in SKIP_DIRS for part in filepath.parts):
            continue

        file_count += 1
        result = analyze_file(filepath, base_dir=directory)
        if "error" in result:
            errors.append(result)
        else:
            nodes.extend(result["definitions"])
            edges.extend(result["edges"])

    # Resolve internal edges (match call names to definitions)
    # Build lookup: symbol name -> full node ID
    defined_names: dict[str, str] = {}
    for n in nodes:
        symbol_name = n["name"]
        # If multiple definitions, last one wins (simple heuristic)
        defined_names[symbol_name] = n["id"]

    resolved_edges = []
    for edge in edges:
        target = edge["to"]
        if target in defined_names:
            edge["to"] = defined_names[target]
            edge["resolved"] = True
        else:
            edge["resolved"] = False
        resolved_edges.append(edge)

    return {
        "nodes": nodes,
        "edges": resolved_edges,
        "errors": errors,
        "metadata": {
            "analyzer": "python",
            "version": "0.2.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
            "file_count": file_count,
        },
    }


def main() -> None:
    """CLI entry point for standalone usage."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_python_deps.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = analyze_directory(directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
