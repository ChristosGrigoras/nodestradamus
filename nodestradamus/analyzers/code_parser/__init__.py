"""Unified code parser using tree-sitter for all languages.

Language-agnostic dependency extraction with consistent graph structure.
Supports Python, TypeScript, JavaScript, Rust, SQL, Bash, JSON, and C/C++.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use 'spawn' context to avoid deadlocks when forking from threads.
# The default 'fork' inherits lock state, causing issues with asyncio.to_thread().
_MP_CONTEXT = multiprocessing.get_context("spawn")
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tree_sitter import Parser

# Re-export all base classes, configs, and helpers
from nodestradamus.analyzers.code_parser.base import (
    BASH_CONFIG,
    CPP_CONFIG,
    EXTENSION_TO_LANGUAGE,
    JAVASCRIPT_CONFIG,
    JSON_CONFIG,
    LANGUAGE_CONFIGS,
    PARSE_CACHE_VERSION,
    PYTHON_CONFIG,
    RUST_CONFIG,
    SKIP_DIRS,
    SQL_CONFIG,
    TYPESCRIPT_CONFIG,
    CodeEdge,
    CodeNode,
    FieldInfo,
    FileCacheEntry,
    LanguageConfig,
    ParseCache,
    ParseResult,
    _extract_class_name,
    _extract_function_name,
    _extract_string_value,
    _find_nodes,
    _get_child_by_field,
    _get_child_by_type,
    _get_language,
    _is_exported,
    compute_file_hash,
    is_file_stale,
    load_parse_cache,
    save_parse_cache,
)
from nodestradamus.analyzers.code_parser.bash import (
    _extract_bash_constants,
    _extract_bash_imports,
    _extract_bash_internal_calls,
    _resolve_bash_import_path,
)
from nodestradamus.analyzers.code_parser.cpp import (
    _extract_cpp_constants,
    _extract_cpp_includes,
    _extract_cpp_internal_calls,
    _extract_cpp_namespaces,
    _extract_cpp_templates,
    _resolve_cpp_include_path,
)
from nodestradamus.analyzers.code_parser.fields import (
    _extract_json_fields,
    _extract_python_fields,
    _extract_rust_fields,
    _extract_schema_fields,
    _extract_ts_properties,
)

# Import language-specific functions
from nodestradamus.analyzers.code_parser.python import (
    _extract_class_parent_python,
    _extract_python_constants,
    _extract_python_imports,
    _extract_python_internal_calls,
)
from nodestradamus.analyzers.code_parser.rust import (
    _extract_rust_constants,
    _extract_rust_imports,
    _extract_rust_internal_calls,
    _resolve_rust_import_path,
)
from nodestradamus.analyzers.code_parser.sql import (
    SQL_TYPE_NODES,
    SqlObjectDefinition,
    _extract_sql_columns,
    _normalize_sql_name,
    _parse_sql_file,
    _parse_sql_file_treesitter,
)
from nodestradamus.analyzers.code_parser.typescript import (
    _extract_class_parent_js,
    _extract_js_imports,
    _extract_ts_constants,
    _extract_ts_internal_calls,
)
from nodestradamus.logging import ProgressBar, logger, progress_bar


def _resolve_import_path(
    import_source: str,
    filepath: Path,
    base_dir: Path,
    language: str,
) -> str | None:
    """Resolve an import path to a file path.

    Returns relative path if resolvable, None if external.
    """
    if language == "python":
        # Python imports: convert dots to path
        if import_source.startswith("."):
            # Relative import
            parts = import_source.lstrip(".").split(".")
            levels = len(import_source) - len(import_source.lstrip("."))

            current = filepath.parent
            for _ in range(levels - 1):
                current = current.parent

            for part in parts:
                current = current / part

            # Try as file or package
            for candidate in [
                current.with_suffix(".py"),
                current / "__init__.py",
            ]:
                if candidate.exists():
                    try:
                        return str(candidate.relative_to(base_dir))
                    except ValueError:
                        return str(candidate)
            return None
        else:
            # Absolute import - check if it's internal
            parts = import_source.split(".")
            candidate = base_dir / "/".join(parts)

            for path in [
                candidate.with_suffix(".py"),
                candidate / "__init__.py",
            ]:
                if path.exists():
                    try:
                        return str(path.relative_to(base_dir))
                    except ValueError:
                        return str(path)
            return None

    else:  # JavaScript/TypeScript
        if not import_source.startswith("."):
            return None  # External package

        parent = filepath.parent
        resolved = (parent / import_source).resolve()

        # Try with extensions
        extensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
        for ext in extensions:
            candidate = resolved.with_suffix(ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(base_dir))
                except ValueError:
                    return str(candidate)

        # Try as directory with index file
        if resolved.is_dir():
            for ext in extensions:
                index = resolved / f"index{ext}"
                if index.exists():
                    try:
                        return str(index.relative_to(base_dir))
                    except ValueError:
                        return str(index)

        return None


def _add_import_edges(
    edges: list[CodeEdge],
    file_id: str,
    prefix: str,
    imp: dict,
    resolved_path: str | None,
) -> None:
    """Add import edges to the edge list.

    Args:
        edges: List to append edges to.
        file_id: Source file node ID.
        prefix: Language prefix (py, ts, rs).
        imp: Import info dict with source and names.
        resolved_path: Resolved file path or None if external.
    """
    if resolved_path:
        target_file = f"{prefix}:{resolved_path}"

        # File-level import
        edges.append(
            CodeEdge(
                source=file_id,
                target=target_file,
                type="imports",
                resolved=True,
                names=imp.get("names", []),
            )
        )

        # Symbol-level imports
        for name in imp.get("names", []):
            edges.append(
                CodeEdge(
                    source=file_id,
                    target=f"{prefix}:{resolved_path}::{name}",
                    type="imports_symbol",
                    resolved=True,
                )
            )
    else:
        # External import
        edges.append(
            CodeEdge(
                source=file_id,
                target=imp["source"],
                type="imports",
                resolved=False,
            )
        )


def parse_file(
    filepath: Path,
    base_dir: Path,
    language: str | None = None,
) -> ParseResult:
    """Parse a single source file and extract nodes/edges.

    Args:
        filepath: Path to the file to parse.
        base_dir: Repository root for relative paths.
        language: Language override (auto-detected if None).

    Returns:
        ParseResult with nodes and edges.
    """
    # Determine language
    if language is None:
        ext = filepath.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        if not language:
            return ParseResult(
                [], [], [{"error": f"Unknown extension: {ext}", "file": str(filepath)}]
            )

    # Get language config and parser
    config = LANGUAGE_CONFIGS.get(language)
    if not config:
        return ParseResult(
            [], [], [{"error": f"No config for language: {language}", "file": str(filepath)}]
        )

    try:
        rel_path = filepath.relative_to(base_dir)
    except ValueError:
        rel_path = filepath

    nodes: list[CodeNode] = []
    edges: list[CodeEdge] = []

    # Create file-level module node
    file_id = f"{config.prefix}:{rel_path}"
    nodes.append(
        CodeNode(
            id=file_id,
            name=str(rel_path),
            type="module",
            file=str(rel_path),
            line=1,
            language=language,
        )
    )

    # For SQL, try tree-sitter first, fall back to regex if unavailable
    if language == "sql":
        try:
            source = filepath.read_bytes()
            source_text = source.decode(errors="replace")
        except OSError as e:
            return ParseResult([], [], [{"error": str(e), "file": str(filepath)}])

        ts_language = _get_language("sql")
        if ts_language:
            # Use tree-sitter-sql for parsing
            sql_nodes, sql_edges = _parse_sql_file_treesitter(
                source, source_text, rel_path, file_id, config, ts_language
            )
        else:
            # Fall back to regex-based parsing
            sql_nodes, sql_edges = _parse_sql_file(source_text, rel_path, file_id, config)
        nodes.extend(sql_nodes)
        edges.extend(sql_edges)
        return ParseResult(nodes=nodes, edges=edges)

    # For other languages, use tree-sitter
    ts_language = _get_language(language)
    if not ts_language:
        return ParseResult(
            [],
            [],
            [{"error": f"tree-sitter binding not available: {language}", "file": str(filepath)}],
        )

    parser = Parser(ts_language)

    # Parse file
    try:
        source = filepath.read_bytes()
        tree = parser.parse(source)
    except (OSError, UnicodeDecodeError) as e:
        return ParseResult([], [], [{"error": str(e), "file": str(filepath)}])

    # Extract functions
    for node in _find_nodes(tree.root_node, config.function_types):
        name = _extract_function_name(node, config)
        if not name:
            continue

        func_id = f"{config.prefix}:{rel_path}::{name}"
        nodes.append(
            CodeNode(
                id=func_id,
                name=name,
                type="function",
                file=str(rel_path),
                line=node.start_point[0] + 1,
                exported=_is_exported(node),
                language=language,
            )
        )

        # Bidirectional containment edges
        edges.append(CodeEdge(source=func_id, target=file_id, type="defined_in"))
        edges.append(CodeEdge(source=file_id, target=func_id, type="contains"))

    # Extract classes (and JSON objects as configs)
    for node in _find_nodes(tree.root_node, config.class_types):
        name = _extract_class_name(node, config)

        # For JSON, use filename as config name and extract fields from root object
        if language == "json" and node.type == "object":
            name = filepath.stem  # Use filename without extension
            node_type = "config"
        else:
            node_type = "class"

        if not name:
            continue

        class_id = f"{config.prefix}:{rel_path}::{name}"

        # Extract schema fields for schema-defining constructs
        extracted_fields = _extract_schema_fields(node, language)

        nodes.append(
            CodeNode(
                id=class_id,
                name=name,
                type=node_type,
                file=str(rel_path),
                line=node.start_point[0] + 1,
                exported=_is_exported(node),
                language=language,
                fields=extracted_fields if extracted_fields else None,
            )
        )

        # Containment edges
        edges.append(CodeEdge(source=class_id, target=file_id, type="defined_in"))
        edges.append(CodeEdge(source=file_id, target=class_id, type="contains"))

        # Inheritance edge (not applicable to JSON)
        if language != "json":
            if language == "python":
                parent = _extract_class_parent_python(node)
            elif language == "rust":
                parent = None  # Rust uses traits differently, handled via impl blocks
            else:
                parent = _extract_class_parent_js(node)

            if parent:
                edges.append(CodeEdge(source=class_id, target=parent, type="extends", resolved=False))

    # Extract constants (module-level UPPER_CASE assignments)
    constants: list[dict] = []
    if language == "python":
        constants = _extract_python_constants(tree.root_node)
    elif language in ("typescript", "javascript", "tsx"):
        constants = _extract_ts_constants(tree.root_node)
    elif language == "rust":
        constants = _extract_rust_constants(tree.root_node)
    elif language == "bash":
        constants = _extract_bash_constants(tree.root_node)
    elif language == "cpp":
        constants = _extract_cpp_constants(tree.root_node)

    for const in constants:
        const_id = f"{config.prefix}:{rel_path}::{const['name']}"
        nodes.append(
            CodeNode(
                id=const_id,
                name=const["name"],
                type="constant",
                file=str(rel_path),
                line=const["line"],
                exported=True,  # Module-level constants are typically exported
                language=language,
            )
        )
        edges.append(CodeEdge(source=const_id, target=file_id, type="defined_in"))
        edges.append(CodeEdge(source=file_id, target=const_id, type="contains"))

    # Extract internal calls (function-to-function calls within this file)
    # Build set of defined function names first
    defined_functions = {
        node.name for node in nodes if node.type == "function"
    }

    internal_calls: list[dict] = []
    if language == "python":
        internal_calls = _extract_python_internal_calls(tree.root_node, defined_functions)
    elif language in ("typescript", "javascript", "tsx"):
        internal_calls = _extract_ts_internal_calls(tree.root_node, defined_functions)
    elif language == "rust":
        internal_calls = _extract_rust_internal_calls(tree.root_node, defined_functions)
    elif language == "bash":
        internal_calls = _extract_bash_internal_calls(tree.root_node, defined_functions)
    elif language == "cpp":
        internal_calls = _extract_cpp_internal_calls(tree.root_node, defined_functions)

    for call in internal_calls:
        caller_id = f"{config.prefix}:{rel_path}::{call['caller']}"
        callee_id = f"{config.prefix}:{rel_path}::{call['callee']}"
        edges.append(
            CodeEdge(
                source=caller_id,
                target=callee_id,
                type="calls",
                resolved=True,
            )
        )

    # Extract imports
    if language == "python":
        imports = _extract_python_imports(tree.root_node, filepath, base_dir)
        for imp in imports:
            resolved_path = _resolve_import_path(imp["source"], filepath, base_dir, language)
            _add_import_edges(edges, file_id, config.prefix, imp, resolved_path)
    elif language == "rust":
        imports = _extract_rust_imports(tree.root_node, filepath, base_dir)
        for imp in imports:
            resolved_path = _resolve_rust_import_path(imp, filepath, base_dir)
            _add_import_edges(edges, file_id, config.prefix, imp, resolved_path)
    elif language == "bash":
        imports = _extract_bash_imports(tree.root_node, filepath, base_dir)
        for imp in imports:
            resolved_path = _resolve_bash_import_path(imp, filepath, base_dir)
            _add_import_edges(edges, file_id, config.prefix, imp, resolved_path)
    elif language == "cpp":
        imports = _extract_cpp_includes(tree.root_node, filepath, base_dir)
        for imp in imports:
            resolved_path = _resolve_cpp_include_path(imp, filepath, base_dir)
            _add_import_edges(edges, file_id, config.prefix, imp, resolved_path)
    elif language == "json":
        pass  # JSON doesn't have imports
    else:
        imports = _extract_js_imports(tree.root_node, filepath, base_dir)
        for imp in imports:
            resolved_path = _resolve_import_path(imp["source"], filepath, base_dir, language)
            _add_import_edges(edges, file_id, config.prefix, imp, resolved_path)

    return ParseResult(nodes=nodes, edges=edges)


def _parse_file_to_dicts(
    filepath: Path,
    base_dir: Path,
) -> dict[str, Any]:
    """Parse a file and convert result to dicts (for parallel processing).

    This function is designed to be called in worker processes.
    It returns plain dicts that can be pickled/transferred.

    Args:
        filepath: Path to the file to parse.
        base_dir: Repository root for relative paths.

    Returns:
        Dict with nodes, edges, errors, rel_path, and file stats.
    """
    try:
        rel_path = filepath.relative_to(base_dir)
    except ValueError:
        rel_path = filepath
    rel_path_str = str(rel_path)

    # Get file stats and content hash for cache update
    try:
        stat = filepath.stat()
        mtime = stat.st_mtime
        size = stat.st_size
        content_hash = compute_file_hash(filepath)
    except OSError:
        mtime = 0.0
        size = 0
        content_hash = ""

    result = parse_file(filepath, base_dir)

    # Convert to dicts for serialization
    file_nodes: list[dict] = []
    file_edges: list[dict] = []

    for node in result.nodes:
        node_dict: dict[str, Any] = {
            "id": node.id,
            "name": node.name,
            "type": node.type,
            "file": node.file,
            "line": node.line,
            "exported": node.exported,
            "language": node.language,
        }
        if node.fields:
            node_dict["fields"] = [
                {
                    "name": f.name,
                    "type": f.type,
                    "nullable": f.nullable,
                    "references": f.references,
                }
                for f in node.fields
            ]
        file_nodes.append(node_dict)

    for edge in result.edges:
        file_edges.append(
            {
                "from": edge.source,
                "to": edge.target,
                "type": edge.type,
                "resolved": edge.resolved,
            }
        )

    return {
        "rel_path": rel_path_str,
        "nodes": file_nodes,
        "edges": file_edges,
        "errors": result.errors,
        "mtime": mtime,
        "size": size,
        "content_hash": content_hash,
    }


# Minimum files to trigger parallel parsing (sequential is faster for small sets)
PARALLEL_THRESHOLD = 50


def parse_directory(
    directory: Path,
    languages: list[str] | None = None,
    use_cache: bool = True,
    exclude: list[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Parse all source files in a directory.

    Uses mtime-based caching to skip re-parsing unchanged files.
    For large codebases (>50 stale files), uses parallel parsing.

    Args:
        directory: Path to the directory to analyze.
        languages: Languages to include (None = auto-detect all).
        use_cache: Whether to use the parse cache (default: True).
        exclude: Directories/patterns to exclude. If None, uses SKIP_DIRS.
        workers: Number of parallel workers. None = auto (cpu_count).
                 Set to 1 to disable parallel parsing.

    Returns:
        Dictionary with nodes, edges, errors, and metadata.
    """
    directory = Path(directory).resolve()
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    errors: list[dict] = []
    cached_count = 0
    parsed_count = 0

    # Determine which extensions to look for
    if languages:
        extensions = {
            ext
            for ext, lang in EXTENSION_TO_LANGUAGE.items()
            if lang in languages or (lang == "tsx" and "typescript" in languages)
        }
    else:
        extensions = set(EXTENSION_TO_LANGUAGE.keys())

    # First pass: collect all files to process
    # Use provided exclude patterns or fall back to SKIP_DIRS
    skip_patterns = set(exclude) if exclude else SKIP_DIRS
    files_to_process: list[Path] = []
    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in extensions:
            continue
        if any(part in skip_patterns for part in filepath.parts):
            continue
        files_to_process.append(filepath)

    total_files = len(files_to_process)
    logger.info("  parse_directory: found %d files to analyze", total_files)

    # Load existing cache
    cache: ParseCache | None = None
    if use_cache:
        cache = load_parse_cache(directory)
        if cache:
            logger.info("  Loaded parse cache with %d entries", len(cache.files))

    # Create new cache if needed
    if cache is None:
        cache = ParseCache(
            version=PARSE_CACHE_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            files={},
        )

    # Track current files to detect deleted files later
    current_files: set[str] = set()

    # Separate files into cached (fresh) vs stale (need parsing)
    cached_files: list[tuple[Path, str]] = []  # (filepath, rel_path_str)
    stale_files: list[Path] = []

    for filepath in files_to_process:
        try:
            rel_path = filepath.relative_to(directory)
        except ValueError:
            rel_path = filepath
        rel_path_str = str(rel_path)
        current_files.add(rel_path_str)

        if use_cache and not is_file_stale(filepath, rel_path_str, cache):
            cached_files.append((filepath, rel_path_str))
        else:
            stale_files.append(filepath)

    # Process cached files (fast path - no parsing needed)
    for _, rel_path_str in cached_files:
        entry = cache.files[rel_path_str]
        all_nodes.extend(entry.nodes)
        all_edges.extend(entry.edges)
        cached_count += 1

    # Process stale files (need parsing)
    stale_count = len(stale_files)
    if stale_count == 0:
        logger.info("  All %d files cached, no parsing needed", cached_count)
    else:
        # Determine whether to use parallel parsing
        effective_workers = workers if workers is not None else multiprocessing.cpu_count()
        use_parallel = effective_workers > 1 and stale_count >= PARALLEL_THRESHOLD

        if use_parallel:
            logger.info(
                "  Parsing %d stale files in parallel (%d workers)",
                stale_count,
                effective_workers,
            )

            with ProcessPoolExecutor(max_workers=effective_workers, mp_context=_MP_CONTEXT) as executor:
                # Submit all parsing tasks
                future_to_path = {
                    executor.submit(_parse_file_to_dicts, fp, directory): fp
                    for fp in stale_files
                }

                # Collect results as they complete with progress bar
                with ProgressBar(total=stale_count, desc="Parsing files", unit="files") as pbar:
                    for future in as_completed(future_to_path):
                        pbar.update()
                        filepath = future_to_path[future]
                        try:
                            result = future.result()
                            parsed_count += 1

                            all_nodes.extend(result["nodes"])
                            all_edges.extend(result["edges"])
                            if result["errors"]:
                                errors.extend(result["errors"])

                            # Update cache
                            if use_cache and result["mtime"] > 0:
                                cache.files[result["rel_path"]] = FileCacheEntry(
                                    mtime=result["mtime"],
                                    size=result["size"],
                                    nodes=result["nodes"],
                                    edges=result["edges"],
                                    content_hash=result["content_hash"],
                                )
                        except Exception as e:
                            errors.append({"error": str(e), "file": str(filepath)})
        else:
            # Sequential parsing for small sets or when workers=1
            logger.info("  Parsing %d stale files sequentially", stale_count)

            for filepath in progress_bar(stale_files, desc="Parsing files", unit="files"):
                result = _parse_file_to_dicts(filepath, directory)
                parsed_count += 1

                all_nodes.extend(result["nodes"])
                all_edges.extend(result["edges"])
                if result["errors"]:
                    errors.extend(result["errors"])

                # Update cache
                if use_cache and result["mtime"] > 0:
                    cache.files[result["rel_path"]] = FileCacheEntry(
                        mtime=result["mtime"],
                        size=result["size"],
                        nodes=result["nodes"],
                        edges=result["edges"],
                        content_hash=result["content_hash"],
                    )

    file_count = cached_count + parsed_count

    # Remove deleted files from cache
    if use_cache:
        deleted_files = set(cache.files.keys()) - current_files
        for deleted in deleted_files:
            del cache.files[deleted]
        if deleted_files:
            logger.info("  Removed %d deleted files from cache", len(deleted_files))

        # Save updated cache
        cache.created_at = datetime.now(UTC).isoformat()
        save_parse_cache(directory, cache)

    logger.info(
        "  parse_directory: %d cached, %d parsed, %d total",
        cached_count,
        parsed_count,
        file_count,
    )

    # Resolve symbolic edges (inheritance, calls)
    defined_names: dict[str, str] = {n["name"]: n["id"] for n in all_nodes}

    for edge in all_edges:
        if not edge["resolved"] and edge["type"] == "extends":
            target = edge["to"]
            if target in defined_names:
                edge["to"] = defined_names[target]
                edge["resolved"] = True

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "errors": errors,
        "metadata": {
            "analyzer": "unified_treesitter",
            "version": "0.4.0",  # Bumped for parallel parsing
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
            "file_count": file_count,
            "cached_count": cached_count,
            "parsed_count": parsed_count,
            "parallel": parsed_count >= PARALLEL_THRESHOLD,
        },
    }


# =============================================================================
# Public API (Backward Compatibility)
# =============================================================================

__all__ = [
    # Main functions
    "parse_file",
    "parse_directory",
    # Parallel parsing
    "PARALLEL_THRESHOLD",
    "_parse_file_to_dicts",
    # Cache management
    "PARSE_CACHE_VERSION",
    "FileCacheEntry",
    "ParseCache",
    "load_parse_cache",
    "save_parse_cache",
    "is_file_stale",
    "compute_file_hash",
    # Core types
    "FieldInfo",
    "CodeNode",
    "CodeEdge",
    "ParseResult",
    "LanguageConfig",
    # Language configs
    "PYTHON_CONFIG",
    "TYPESCRIPT_CONFIG",
    "JAVASCRIPT_CONFIG",
    "RUST_CONFIG",
    "SQL_CONFIG",
    "BASH_CONFIG",
    "JSON_CONFIG",
    "CPP_CONFIG",
    "LANGUAGE_CONFIGS",
    # Mappings
    "EXTENSION_TO_LANGUAGE",
    "SKIP_DIRS",
    # Tree-sitter helpers (used by string_extraction.py)
    "_find_nodes",
    "_get_child_by_field",
    "_get_child_by_type",
    "_extract_string_value",
    "_is_exported",
    "_extract_function_name",
    "_extract_class_name",
    "_get_language",
    # SQL-specific (for tests)
    "SQL_TYPE_NODES",
    "SqlObjectDefinition",
    "_extract_sql_columns",
    "_normalize_sql_name",
    "_parse_sql_file",
    "_parse_sql_file_treesitter",
    # Field extraction
    "_extract_schema_fields",
    "_extract_ts_properties",
    "_extract_python_fields",
    "_extract_rust_fields",
    "_extract_json_fields",
    # Constants extraction (refactor analysis)
    "_extract_python_constants",
    "_extract_ts_constants",
    "_extract_rust_constants",
    "_extract_bash_constants",
    # Internal call extraction (refactor analysis)
    "_extract_python_internal_calls",
    "_extract_ts_internal_calls",
    "_extract_rust_internal_calls",
    "_extract_bash_internal_calls",
    "_extract_cpp_internal_calls",
    # C++ specific
    "_extract_cpp_constants",
    "_extract_cpp_includes",
    "_extract_cpp_namespaces",
    "_extract_cpp_templates",
    "_resolve_cpp_include_path",
]
