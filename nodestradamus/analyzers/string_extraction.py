"""Consolidated string extraction utilities.

This module provides shared utilities and language-specific extractors for
finding string literals in Python, TypeScript/JavaScript, and SQL files.

The extraction functions return dictionaries with:
- strings: List of string references with context
- file_count: Number of files processed
- errors: List of any errors encountered
- metadata: Extraction metadata
"""

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tree_sitter import Node, Parser

# Re-export tree-sitter utilities from code_parser
from nodestradamus.analyzers.code_parser import (
    _extract_string_value as extract_string_value,
)
from nodestradamus.analyzers.code_parser import (
    _find_nodes as find_nodes,
)
from nodestradamus.analyzers.code_parser import (
    _get_child_by_field as get_child_by_field,
)
from nodestradamus.analyzers.code_parser import (
    _get_child_by_type as get_child_by_type,
)

# =============================================================================
# Shared Constants
# =============================================================================

# Directories to skip during analysis (merged from all extractors)
SKIP_DIRS = frozenset({
    # Python
    "venv", ".venv", "env", ".env", "__pycache__",
    ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    # Node/JS
    "node_modules", ".next", ".turbo", "coverage", ".cache",
    # Build outputs
    "dist", "build", "out",
    # VCS
    ".git",
})

# Minimum string length to consider (filters out single chars, empty strings)
MIN_STRING_LENGTH = 2

# Base noise patterns common to all languages
_BASE_NOISE_PATTERNS = frozenset({
    "", " ", "\n", "\t", "\r\n", ",", ".", ":", ";", "-", "_", "/", "\\",
    "true", "false", "null", "none", "yes", "no", "on", "off",
    "utf-8", "utf8", "ascii",
})

# Python-specific noise patterns
PYTHON_NOISE_PATTERNS = _BASE_NOISE_PATTERNS | frozenset({
    "latin-1", "r", "w", "rb", "wb", "a", "ab",
})

# TypeScript/JavaScript-specific noise patterns
TYPESCRIPT_NOISE_PATTERNS = _BASE_NOISE_PATTERNS | frozenset({
    "undefined", "GET", "POST", "PUT", "DELETE", "PATCH",
    "div", "span", "button", "input", "form", "label", "a", "p", "h1", "h2", "h3",
})

# SQL-specific noise patterns
SQL_NOISE_PATTERNS = _BASE_NOISE_PATTERNS | frozenset({
    "plpgsql", "sql", "c", "internal",
})

# Combined noise patterns for generic use
NOISE_PATTERNS = _BASE_NOISE_PATTERNS | PYTHON_NOISE_PATTERNS | TYPESCRIPT_NOISE_PATTERNS | SQL_NOISE_PATTERNS

# File extensions by language
PYTHON_EXTENSIONS = frozenset({".py", ".pyw"})
TYPESCRIPT_EXTENSIONS = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
SQL_EXTENSIONS = frozenset({".sql", ".pgsql"})


# =============================================================================
# Shared Utilities
# =============================================================================

def is_noise(
    value: str,
    min_length: int = MIN_STRING_LENGTH,
    noise_patterns: frozenset[str] = NOISE_PATTERNS,
) -> bool:
    """Check if a string is likely noise.

    Args:
        value: The string value to check.
        min_length: Minimum length for non-noise strings.
        noise_patterns: Set of known noise patterns.

    Returns:
        True if the string is likely noise, False otherwise.
    """
    if len(value) < min_length:
        return True
    if value.lower() in noise_patterns:
        return True
    # Skip strings that are just whitespace
    if not value.strip():
        return True
    return False


def should_skip_path(filepath: Path, skip_dirs: frozenset[str] = SKIP_DIRS) -> bool:
    """Check if a file path should be skipped.

    Args:
        filepath: Path to check.
        skip_dirs: Set of directory names to skip.

    Returns:
        True if the path should be skipped.
    """
    return any(part.startswith(".") or part in skip_dirs for part in filepath.parts)


def group_strings_by_value_and_file(
    strings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group strings by (value, file) and aggregate contexts.

    Args:
        strings: List of string dicts with 'value', 'file', 'context' keys.

    Returns:
        List of grouped strings with 'contexts' list.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for s in strings:
        key = (s["value"], s["file"])
        if key not in grouped:
            grouped[key] = {
                "value": s["value"],
                "file": s["file"],
                "contexts": [],
            }
        grouped[key]["contexts"].append(s["context"])
    return list(grouped.values())


# =============================================================================
# Python String Extraction
# =============================================================================

class PythonStringVisitor(ast.NodeVisitor):
    """AST visitor that extracts string literals with context."""

    def __init__(self, filepath: Path, base_dir: Path | None = None):
        self.filepath = filepath
        self.base_dir = base_dir
        self.rel_path = str(filepath.relative_to(base_dir) if base_dir else filepath)
        self.strings: list[dict[str, Any]] = []

        # Context stack for tracking enclosing scopes
        self._class_stack: list[str] = []
        self._function_stack: list[str] = []

    def _make_context(
        self,
        line: int,
        call_site: str | None = None,
        variable_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a context dict for a string occurrence."""
        return {
            "call_site": call_site,
            "variable_name": variable_name,
            "enclosing_function": self._function_stack[-1] if self._function_stack else None,
            "enclosing_class": self._class_stack[-1] if self._class_stack else None,
            "line": line,
        }

    def _is_noise(self, value: str) -> bool:
        """Check if a string is likely noise."""
        return is_noise(value, noise_patterns=PYTHON_NOISE_PATTERNS)

    def _extract_call_name(self, node: ast.expr) -> str | None:
        """Extract the function name from a Call node's func attribute."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle chained attributes like os.path.join
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def _add_string(
        self,
        value: str,
        line: int,
        call_site: str | None = None,
        variable_name: str | None = None,
    ) -> None:
        """Add a string with its context."""
        if self._is_noise(value):
            return

        context = self._make_context(line, call_site, variable_name)
        self.strings.append({
            "value": value,
            "file": self.rel_path,
            "context": context,
        })

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class scope."""
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function scope."""
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function scope."""
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Extract strings from function call arguments."""
        call_name = self._extract_call_name(node.func)

        # Check positional arguments
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self._add_string(arg.value, arg.lineno, call_site=call_name)
            elif isinstance(arg, ast.JoinedStr):
                # f-string - extract static parts
                for part in arg.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        self._add_string(
                            part.value, part.lineno, call_site=call_name
                        )

        # Check keyword arguments
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                self._add_string(kw.value.value, kw.value.lineno, call_site=call_name)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract strings from variable assignments."""
        # Get variable name if single target
        var_name = None
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
            elif isinstance(target, ast.Attribute):
                var_name = target.attr

        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            self._add_string(
                node.value.value, node.value.lineno, variable_name=var_name
            )
        elif isinstance(node.value, ast.JoinedStr):
            # f-string assignment
            for part in node.value.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    self._add_string(
                        part.value, part.lineno, variable_name=var_name
                    )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Extract strings from annotated assignments."""
        var_name = None
        if isinstance(node.target, ast.Name):
            var_name = node.target.id

        if node.value:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self._add_string(
                    node.value.value, node.value.lineno, variable_name=var_name
                )

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Extract strings from dictionary literals."""
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self._add_string(key.value, key.lineno)

        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                self._add_string(value.value, value.lineno)

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Extract strings from return statements."""
        if node.value:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self._add_string(node.value.value, node.value.lineno)

        self.generic_visit(node)


def extract_python_strings_from_file(
    filepath: Path, base_dir: Path | None = None
) -> dict[str, Any]:
    """Extract all string literals from a Python file.

    Args:
        filepath: Path to the Python file.
        base_dir: Optional base directory for relative path calculation.

    Returns:
        Dictionary with 'strings' list or 'error' if parsing failed.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(filepath)}

    visitor = PythonStringVisitor(filepath, base_dir)
    visitor.visit(tree)

    return {"strings": visitor.strings}


def extract_python_strings(directory: Path) -> dict[str, Any]:
    """Extract string literals from all Python files in a directory.

    Args:
        directory: Path to the directory to analyze.

    Returns:
        Dictionary containing strings, file_count, errors, and metadata.
    """
    directory = Path(directory).resolve()
    all_strings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    file_count = 0

    for filepath in directory.rglob("*.py"):
        if should_skip_path(filepath):
            continue

        file_count += 1
        result = extract_python_strings_from_file(filepath, base_dir=directory)

        if "error" in result:
            errors.append(result)
        else:
            all_strings.extend(result["strings"])

    strings = group_strings_by_value_and_file(all_strings)

    return {
        "strings": strings,
        "file_count": file_count,
        "errors": errors,
        "metadata": {
            "analyzer": "python_strings",
            "version": "0.1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
            "total_strings": len(strings),
        },
    }


# =============================================================================
# TypeScript/JavaScript String Extraction
# =============================================================================

def _get_ts_parsers() -> dict[str, Parser]:
    """Lazy-load TypeScript/JavaScript parsers."""
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_typescript as ts_typescript
    from tree_sitter import Language

    ts_lang = Language(ts_typescript.language_typescript())
    tsx_lang = Language(ts_typescript.language_tsx())
    js_lang = Language(ts_javascript.language())

    return {
        ".ts": Parser(ts_lang),
        ".tsx": Parser(tsx_lang),
        ".js": Parser(js_lang),
        ".jsx": Parser(js_lang),
        ".mjs": Parser(js_lang),
        ".cjs": Parser(js_lang),
    }


# Cached parsers
_TS_PARSERS: dict[str, Parser] | None = None


def _get_ts_parser(filepath: Path) -> Parser | None:
    """Get the appropriate parser for a TypeScript/JavaScript file."""
    global _TS_PARSERS
    if _TS_PARSERS is None:
        try:
            _TS_PARSERS = _get_ts_parsers()
        except ImportError:
            return None
    return _TS_PARSERS.get(filepath.suffix.lower())


def _get_ts_enclosing_scope(node: Node) -> tuple[str | None, str | None]:
    """Find the enclosing function and class names for a TypeScript node.

    Returns:
        Tuple of (enclosing_function, enclosing_class)
    """
    enclosing_function = None
    enclosing_class = None
    current = node.parent

    while current:
        if current.type in ("function_declaration", "method_definition", "arrow_function"):
            if enclosing_function is None:
                # Get function name
                name_node = get_child_by_field(current, "name")
                if name_node:
                    enclosing_function = name_node.text.decode()
                elif current.type == "arrow_function":
                    # For arrow functions, try to get the variable name
                    parent = current.parent
                    if parent and parent.type == "variable_declarator":
                        name_node = get_child_by_field(parent, "name")
                        if name_node:
                            enclosing_function = name_node.text.decode()

        elif current.type == "class_declaration":
            if enclosing_class is None:
                name_node = get_child_by_field(current, "name")
                if name_node:
                    enclosing_class = name_node.text.decode()

        current = current.parent

    return enclosing_function, enclosing_class


def _get_ts_call_site(node: Node) -> str | None:
    """Get the call site if this string is an argument to a function call."""
    current = node.parent

    while current:
        if current.type == "arguments":
            # Parent of arguments should be call_expression
            call_expr = current.parent
            if call_expr and call_expr.type == "call_expression":
                func = get_child_by_field(call_expr, "function")
                if func:
                    if func.type == "identifier":
                        return func.text.decode()
                    elif func.type == "member_expression":
                        # Get the full expression like console.log or path.join
                        return func.text.decode()
            break

        # Don't traverse too far up
        if current.type in ("program", "function_declaration", "class_declaration"):
            break

        current = current.parent

    return None


def _get_ts_variable_name(node: Node) -> str | None:
    """Get the variable name if this string is assigned to a variable."""
    current = node.parent

    while current:
        if current.type == "variable_declarator":
            name_node = get_child_by_field(current, "name")
            if name_node:
                return name_node.text.decode()
            break

        elif current.type == "assignment_expression":
            left = get_child_by_field(current, "left")
            if left:
                if left.type == "identifier":
                    return left.text.decode()
                elif left.type == "member_expression":
                    # Get property name for obj.prop = "value"
                    prop = get_child_by_field(left, "property")
                    if prop:
                        return prop.text.decode()
            break

        # Don't traverse too far up
        if current.type in ("program", "function_declaration", "call_expression"):
            break

        current = current.parent

    return None


def _is_ts_noise(value: str) -> bool:
    """Check if a TypeScript/JavaScript string is likely noise."""
    if is_noise(value, noise_patterns=TYPESCRIPT_NOISE_PATTERNS):
        return True
    # Skip very short CSS class-like strings
    if len(value) <= 3 and value.replace("-", "").replace("_", "").isalpha():
        return True
    return False


def extract_typescript_strings_from_file(
    filepath: Path, base_dir: Path
) -> dict[str, Any]:
    """Extract all string literals from a TypeScript/JavaScript file.

    Args:
        filepath: Path to the file.
        base_dir: Base directory for relative paths.

    Returns:
        Dictionary with 'strings' list or 'error' if parsing failed.
    """
    parser = _get_ts_parser(filepath)
    if not parser:
        return {"error": f"Unsupported file type: {filepath.suffix}", "file": str(filepath)}

    try:
        source = filepath.read_bytes()
        tree = parser.parse(source)
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(filepath)}

    try:
        rel_path = str(filepath.relative_to(base_dir))
    except ValueError:
        rel_path = str(filepath)

    strings: list[dict[str, Any]] = []

    # Find all string nodes
    string_types = {"string", "template_string"}
    string_nodes = find_nodes(tree.root_node, string_types)

    for node in string_nodes:
        value = extract_string_value(node)

        if _is_ts_noise(value):
            continue

        # Get context
        enclosing_function, enclosing_class = _get_ts_enclosing_scope(node)
        call_site = _get_ts_call_site(node)
        variable_name = _get_ts_variable_name(node)

        line = node.start_point[0] + 1  # tree-sitter is 0-indexed

        context = {
            "call_site": call_site,
            "variable_name": variable_name,
            "enclosing_function": enclosing_function,
            "enclosing_class": enclosing_class,
            "line": line,
        }

        strings.append({
            "value": value,
            "file": rel_path,
            "context": context,
        })

    return {"strings": strings}


def extract_typescript_strings(directory: Path) -> dict[str, Any]:
    """Extract string literals from all TypeScript/JavaScript files in a directory.

    Args:
        directory: Path to the directory to analyze.

    Returns:
        Dictionary containing strings, file_count, errors, and metadata.
    """
    directory = Path(directory).resolve()
    all_strings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    file_count = 0

    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in TYPESCRIPT_EXTENSIONS:
            continue
        if should_skip_path(filepath):
            continue

        file_count += 1
        result = extract_typescript_strings_from_file(filepath, directory)

        if "error" in result:
            errors.append(result)
        else:
            all_strings.extend(result["strings"])

    strings = group_strings_by_value_and_file(all_strings)

    return {
        "strings": strings,
        "file_count": file_count,
        "errors": errors,
        "metadata": {
            "analyzer": "typescript_strings",
            "version": "0.1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
            "total_strings": len(strings),
        },
    }


# =============================================================================
# SQL String Extraction
# =============================================================================

def _get_sql_parser() -> tuple[Parser | None, Any]:
    """Lazy-load tree-sitter-sql parser."""
    try:
        import tree_sitter_sql as ts_sql
        from tree_sitter import Language

        lang = Language(ts_sql.language())
        parser = Parser(lang)
        return parser, lang
    except ImportError:
        return None, None


# Cached SQL parser
_SQL_PARSER: Parser | None = None
_SQL_PARSER_LOADED = False


def _get_cached_sql_parser() -> Parser | None:
    """Get cached SQL parser."""
    global _SQL_PARSER, _SQL_PARSER_LOADED
    if not _SQL_PARSER_LOADED:
        _SQL_PARSER, _ = _get_sql_parser()
        _SQL_PARSER_LOADED = True
    return _SQL_PARSER


def _clean_sql_string(value: str) -> str:
    """Clean SQL string literal by removing quotes and escape sequences."""
    # Remove outer quotes
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    elif value.startswith("E'") and value.endswith("'"):
        value = value[2:-1]
    elif value.startswith("$$") and value.endswith("$$"):
        # Dollar-quoted string - extract content
        return ""  # Skip dollar-quoted strings (usually code blocks)
    elif value.startswith("$") and "$" in value[1:]:
        # Tagged dollar quote like $tag$...$tag$
        return ""  # Skip dollar-quoted strings

    # Unescape doubled single quotes
    value = value.replace("''", "'")

    return value


def _get_sql_enclosing_definition(node: Node) -> tuple[str | None, str | None]:
    """Find the enclosing function/procedure/view definition."""
    current = node.parent
    definition_types = {
        "create_function": "function",
        "create_procedure": "procedure",
        "create_trigger": "trigger",
        "create_view": "view",
        "create_materialized_view": "view",
        "create_table": "table",
    }

    while current:
        if current.type in definition_types:
            # Find the object_reference child for the name
            for child in current.children:
                if child.type == "object_reference":
                    return definition_types[current.type], child.text.decode()
            return definition_types[current.type], None
        current = current.parent

    return None, None


def _get_sql_statement_type(node: Node) -> str | None:
    """Find the statement type containing this node."""
    current = node.parent
    statement_types = {
        "select", "insert", "update", "delete", "create_table",
        "create_view", "create_function", "create_procedure",
        "create_trigger", "alter_table", "drop_table",
    }

    while current:
        if current.type in statement_types:
            return current.type
        if current.type == "statement":
            # Get the first child type
            for child in current.children:
                if child.type in statement_types:
                    return child.type
        current = current.parent

    return None


def extract_sql_strings_from_file(
    filepath: Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract all string literals from a SQL file.

    Args:
        filepath: Path to the SQL file.
        base_dir: Base directory for relative paths.

    Returns:
        Dict with 'strings' list and 'errors' list.
    """
    parser = _get_cached_sql_parser()
    if parser is None:
        return {
            "strings": [],
            "errors": [{"error": "tree-sitter-sql not available", "file": str(filepath)}],
        }

    try:
        source = filepath.read_bytes()
        tree = parser.parse(source)
    except (OSError, UnicodeDecodeError) as e:
        return {"strings": [], "errors": [{"error": str(e), "file": str(filepath)}]}

    rel_path = str(filepath.relative_to(base_dir) if base_dir else filepath)
    strings: list[dict[str, Any]] = []

    # Find all literal nodes (string literals in tree-sitter-sql)
    literal_nodes = find_nodes(tree.root_node, {"literal"})

    for literal_node in literal_nodes:
        raw_value = literal_node.text.decode()

        # Skip non-string literals (numbers, etc.)
        if not (raw_value.startswith("'") or raw_value.startswith("E'") or
                raw_value.startswith("$")):
            continue

        value = _clean_sql_string(raw_value)

        if is_noise(value, noise_patterns=SQL_NOISE_PATTERNS):
            continue

        # Get context
        def_type, def_name = _get_sql_enclosing_definition(literal_node)
        stmt_type = _get_sql_statement_type(literal_node)

        context = {
            "call_site": stmt_type,  # Use statement type as "call site"
            "variable_name": None,
            "enclosing_function": def_name,
            "enclosing_class": def_type,  # Use definition type as "class"
            "line": literal_node.start_point[0] + 1,
        }

        strings.append({
            "value": value,
            "file": rel_path,
            "context": context,
        })

    return {"strings": strings, "errors": []}


def extract_sql_strings(directory: Path) -> dict[str, Any]:
    """Extract all string literals from SQL files in a directory.

    Args:
        directory: Path to the directory.

    Returns:
        Dict with 'strings' list, 'errors' list, and 'file_count'.
    """
    directory = Path(directory).resolve()

    if not directory.is_dir():
        return {
            "strings": [],
            "errors": [{"error": f"Not a directory: {directory}"}],
            "file_count": 0,
        }

    all_strings: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    file_count = 0

    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() not in SQL_EXTENSIONS:
            continue

        if should_skip_path(filepath):
            continue

        file_count += 1
        result = extract_sql_strings_from_file(filepath, directory)
        all_strings.extend(result["strings"])
        all_errors.extend(result.get("errors", []))

    # Group strings by value (SQL grouping is slightly different)
    grouped: dict[str, dict[str, Any]] = {}
    for s in all_strings:
        value = s["value"]
        if value not in grouped:
            grouped[value] = {
                "value": value,
                "file": s["file"],
                "contexts": [],
            }
        grouped[value]["contexts"].append(s["context"])

    return {
        "strings": list(grouped.values()),
        "errors": all_errors,
        "file_count": file_count,
        "metadata": {
            "analyzer": "sql_strings",
            "version": "0.1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
        },
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "SKIP_DIRS",
    "MIN_STRING_LENGTH",
    "NOISE_PATTERNS",
    "PYTHON_NOISE_PATTERNS",
    "TYPESCRIPT_NOISE_PATTERNS",
    "SQL_NOISE_PATTERNS",
    "PYTHON_EXTENSIONS",
    "TYPESCRIPT_EXTENSIONS",
    "SQL_EXTENSIONS",
    # Utilities
    "is_noise",
    "should_skip_path",
    "group_strings_by_value_and_file",
    # Tree-sitter utilities (re-exported)
    "find_nodes",
    "extract_string_value",
    "get_child_by_field",
    "get_child_by_type",
    # Python extraction
    "PythonStringVisitor",
    "extract_python_strings_from_file",
    "extract_python_strings",
    # TypeScript extraction
    "extract_typescript_strings_from_file",
    "extract_typescript_strings",
    # SQL extraction
    "extract_sql_strings_from_file",
    "extract_sql_strings",
]
