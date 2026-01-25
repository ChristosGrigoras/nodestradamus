#!/usr/bin/env python3
"""
Analyze Python code dependencies using the ast module.

Outputs a JSON graph of function/class calls for AI impact analysis.

Usage:
    python analyze_python_deps.py src/
    python analyze_python_deps.py src/ > .cursor/graph/python-deps.json
"""

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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


def analyze_file(filepath: Path) -> dict:
    """Analyze a single Python file for dependencies."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(filepath)}

    definitions = []
    edges = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = f"{filepath}::{node.name}"
            definitions.append({
                "name": func_name,
                "type": "function",
                "line": node.lineno,
            })
            for call in extract_calls(node):
                edges.append({
                    "from": func_name,
                    "to": call,
                    "type": "calls",
                })

        elif isinstance(node, ast.ClassDef):
            class_name = f"{filepath}::{node.name}"
            definitions.append({
                "name": class_name,
                "type": "class",
                "line": node.lineno,
            })
            for base in node.bases:
                if isinstance(base, ast.Name):
                    edges.append({
                        "from": class_name,
                        "to": base.id,
                        "type": "inherits",
                    })

    return {"definitions": definitions, "edges": edges}


def analyze_directory(directory: Path) -> dict:
    """Analyze all Python files in a directory."""
    nodes = []
    edges = []
    errors = []

    for filepath in directory.rglob("*.py"):
        # Skip common non-source directories
        if any(part.startswith(".") or part in ("venv", "node_modules", "__pycache__", ".git") 
               for part in filepath.parts):
            continue

        result = analyze_file(filepath)
        if "error" in result:
            errors.append(result)
        else:
            nodes.extend(result["definitions"])
            edges.extend(result["edges"])

    # Resolve internal edges (match call names to definitions)
    defined_names = {n["name"].split("::")[-1]: n["name"] for n in nodes}
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
        "nodes": [n["name"] for n in nodes],
        "node_details": nodes,
        "edges": resolved_edges,
        "errors": errors,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "analyze_python_deps.py",
            "source_directory": str(directory),
        },
    }


def main():
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
