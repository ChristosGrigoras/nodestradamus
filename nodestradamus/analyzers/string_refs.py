"""String reference analyzer wrapper.

Combines Python, TypeScript, and SQL string extraction into a unified interface.
Returns Pydantic models compatible with the rest of Nodestradamus.
"""

from pathlib import Path
from typing import Any

# Import consolidated extraction functions
from nodestradamus.analyzers.string_extraction import (
    # Re-export extension constants for backward compatibility
    extract_python_strings,
    extract_sql_strings,
    extract_typescript_strings,
)
from nodestradamus.models.graph import (
    StringContext,
    StringRefGraph,
    StringRefMetadata,
    StringRefNode,
)


def _convert_raw_to_models(raw_strings: list[dict[str, Any]]) -> list[StringRefNode]:
    """Convert raw string dicts to Pydantic models.

    Args:
        raw_strings: List of raw string dicts from extractors.

    Returns:
        List of StringRefNode models.
    """
    nodes = []
    for s in raw_strings:
        contexts = [
            StringContext(
                call_site=c.get("call_site"),
                variable_name=c.get("variable_name"),
                enclosing_function=c.get("enclosing_function"),
                enclosing_class=c.get("enclosing_class"),
                line=c["line"],
            )
            for c in s.get("contexts", [])
        ]

        node = StringRefNode(
            value=s["value"],
            file=s["file"],
            contexts=contexts,
        )
        nodes.append(node)

    return nodes


def analyze_string_refs(
    repo_path: str | Path,
    include_python: bool = True,
    include_typescript: bool = True,
    include_sql: bool = True,
) -> StringRefGraph:
    """Analyze a repository to extract all string references.

    Runs Python, TypeScript, and SQL extraction, then combines results.

    Args:
        repo_path: Absolute path to repository root.
        include_python: Whether to analyze Python files.
        include_typescript: Whether to analyze TypeScript/JavaScript files.
        include_sql: Whether to analyze SQL files.

    Returns:
        StringRefGraph with all extracted string references.
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.is_dir():
        raise ValueError(f"Not a directory: {repo_path}")

    all_strings: list[StringRefNode] = []
    all_errors: list[dict] = []
    total_files = 0

    # Extract Python strings
    if include_python:
        try:
            py_result = extract_python_strings(repo_path)
            py_nodes = _convert_raw_to_models(py_result["strings"])
            all_strings.extend(py_nodes)
            all_errors.extend(py_result.get("errors", []))
            total_files += py_result.get("file_count", 0)
        except Exception as e:
            all_errors.append({"analyzer": "python", "error": str(e)})

    # Extract TypeScript/JavaScript strings
    if include_typescript:
        try:
            ts_result = extract_typescript_strings(repo_path)
            ts_nodes = _convert_raw_to_models(ts_result["strings"])
            all_strings.extend(ts_nodes)
            all_errors.extend(ts_result.get("errors", []))
            total_files += ts_result.get("file_count", 0)
        except Exception as e:
            all_errors.append({"analyzer": "typescript", "error": str(e)})

    # Extract SQL strings
    if include_sql:
        try:
            sql_result = extract_sql_strings(repo_path)
            sql_nodes = _convert_raw_to_models(sql_result["strings"])
            all_strings.extend(sql_nodes)
            all_errors.extend(sql_result.get("errors", []))
            total_files += sql_result.get("file_count", 0)
        except Exception as e:
            all_errors.append({"analyzer": "sql", "error": str(e)})

    metadata = StringRefMetadata(
        analyzer="string_refs",
        version="0.1.0",
        file_count=total_files,
        total_strings=len(all_strings),
        source_directory=str(repo_path),
    )

    return StringRefGraph(
        strings=all_strings,
        file_count=total_files,
        metadata=metadata,
        errors=all_errors,
    )


def analyze_python_string_refs(repo_path: str | Path) -> StringRefGraph:
    """Analyze only Python files for string references.

    Args:
        repo_path: Absolute path to repository root.

    Returns:
        StringRefGraph with Python string references.
    """
    return analyze_string_refs(
        repo_path, include_python=True, include_typescript=False, include_sql=False
    )


def analyze_typescript_string_refs(repo_path: str | Path) -> StringRefGraph:
    """Analyze only TypeScript/JavaScript files for string references.

    Args:
        repo_path: Absolute path to repository root.

    Returns:
        StringRefGraph with TypeScript/JavaScript string references.
    """
    return analyze_string_refs(
        repo_path, include_python=False, include_typescript=True, include_sql=False
    )


def analyze_sql_string_refs(repo_path: str | Path) -> StringRefGraph:
    """Analyze only SQL files for string references.

    Args:
        repo_path: Absolute path to repository root.

    Returns:
        StringRefGraph with SQL string references.
    """
    return analyze_string_refs(
        repo_path, include_python=False, include_typescript=False, include_sql=True
    )
