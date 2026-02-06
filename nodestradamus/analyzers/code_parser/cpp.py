"""C/C++ specific parsing functions.

Handles C/C++ include extraction, path resolution, constants, namespaces, and internal calls.
"""

from pathlib import Path

from tree_sitter import Node

from nodestradamus.analyzers.code_parser.base import _find_nodes, _get_child_by_field


def _extract_cpp_constants(tree: Node) -> list[dict]:
    """Extract const and #define constants from C/C++ AST.

    Detects:
    - #define UPPER_CASE value
    - const TYPE UPPER_CASE = value;
    - constexpr TYPE UPPER_CASE = value;
    - static const TYPE UPPER_CASE = value;

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and value_preview.
    """
    constants = []

    # Extract #define macros
    for node in _find_nodes(tree, {"preproc_def", "preproc_function_def"}):
        name_node = _get_child_by_field(node, "name")
        if not name_node:
            continue

        name = name_node.text.decode()

        # Check if UPPER_CASE (C convention for macros)
        if name.isupper() or ("_" in name and name == name.upper()):
            value_node = _get_child_by_field(node, "value")
            value_preview = ""
            if value_node:
                value_text = value_node.text.decode().strip()
                value_preview = value_text[:50] + ("..." if len(value_text) > 50 else "")

            constants.append({
                "name": name,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "value_preview": value_preview,
            })

    # Extract const/constexpr declarations
    for node in _find_nodes(tree, {"declaration"}):
        # Check for const or constexpr in type specifiers
        has_const = False
        for child in node.children:
            if child.type in ("type_qualifier", "storage_class_specifier"):
                text = child.text.decode()
                if text in ("const", "constexpr"):
                    has_const = True
                    break

        if not has_const:
            continue

        # Get declarator and extract name
        declarator = _get_child_by_field(node, "declarator")
        if not declarator:
            continue

        name = None
        # Handle init_declarator
        if declarator.type == "init_declarator":
            inner_declarator = _get_child_by_field(declarator, "declarator")
            if inner_declarator and inner_declarator.type == "identifier":
                name = inner_declarator.text.decode()
        elif declarator.type == "identifier":
            name = declarator.text.decode()

        if not name:
            continue

        # Check if UPPER_CASE
        if name.isupper() or ("_" in name and name == name.upper()):
            node_text = node.text.decode()
            value_preview = ""
            if "=" in node_text:
                value_part = node_text.split("=", 1)[1].strip().rstrip(";")
                value_preview = value_part[:50] + ("..." if len(value_part) > 50 else "")

            constants.append({
                "name": name,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "value_preview": value_preview,
            })

    return constants


def _extract_cpp_internal_calls(
    tree: Node,
    defined_functions: set[str],
) -> list[dict]:
    """Extract internal function calls within the same file.

    Finds all function calls where the callee is a function defined in this file.
    Handles direct calls, method calls, and namespace-qualified calls.

    Args:
        tree: Root node of the parsed AST.
        defined_functions: Set of function names defined in this file.

    Returns:
        List of dicts with caller, callee, and line.
    """
    calls = []

    # Find all function definitions
    for func_node in _find_nodes(tree, {"function_definition"}):
        # Get function name from declarator
        caller_name = _get_function_name_from_declarator(func_node)
        if not caller_name:
            continue

        # Find all call expressions within this function
        for call_node in _find_nodes(func_node, {"call_expression"}):
            func = _get_child_by_field(call_node, "function")
            if not func:
                continue

            callee_name = None
            if func.type == "identifier":
                callee_name = func.text.decode()
            elif func.type == "qualified_identifier":
                # Handle namespace::func() or Class::method()
                for child in reversed(func.children):
                    if child.type == "identifier":
                        callee_name = child.text.decode()
                        break
            elif func.type == "field_expression":
                # Handle obj.method() or obj->method()
                field = _get_child_by_field(func, "field")
                if field and field.type == "field_identifier":
                    callee_name = field.text.decode()

            if callee_name and callee_name in defined_functions and callee_name != caller_name:
                calls.append({
                    "caller": caller_name,
                    "callee": callee_name,
                    "line": call_node.start_point[0] + 1,
                })

    return calls


def _get_function_name_from_declarator(func_node: Node) -> str | None:
    """Extract function name from a function_definition node's declarator."""
    declarator = _get_child_by_field(func_node, "declarator")
    if not declarator:
        return None

    # Navigate through pointer/reference declarators
    while declarator and declarator.type in (
        "pointer_declarator",
        "reference_declarator",
        "parenthesized_declarator",
    ):
        declarator = _get_child_by_field(declarator, "declarator")

    if not declarator:
        return None

    if declarator.type == "function_declarator":
        inner = _get_child_by_field(declarator, "declarator")
        if inner:
            if inner.type == "identifier":
                return inner.text.decode()
            if inner.type == "qualified_identifier":
                # Get the last identifier (method name)
                for child in reversed(inner.children):
                    if child.type == "identifier":
                        return child.text.decode()

    return None


def _extract_cpp_includes(
    tree: Node,
    filepath: Path,
    base_dir: Path,
) -> list[dict]:
    """Extract #include directives from C/C++ AST.

    Handles:
    - #include <system/header.h>  (system headers)
    - #include "local/header.h"   (local headers)
    - #include "header.hpp"       (C++ headers)

    Args:
        tree: Root node of the parsed AST.
        filepath: Current file path.
        base_dir: Repository root.

    Returns:
        List of dicts with source, names, line, and is_system_header.
    """
    includes = []

    for node in _find_nodes(tree, {"preproc_include"}):
        path_node = _get_child_by_field(node, "path")
        if not path_node:
            continue

        path_text = path_node.text.decode()

        # Determine if system header (<...>) or local header ("...")
        is_system_header = path_text.startswith("<")

        # Extract the actual path (remove quotes or angle brackets)
        if path_text.startswith("<") and path_text.endswith(">"):
            include_path = path_text[1:-1]
        elif path_text.startswith('"') and path_text.endswith('"'):
            include_path = path_text[1:-1]
        else:
            include_path = path_text

        # Extract the header name (last part of path)
        header_name = Path(include_path).stem

        includes.append({
            "source": include_path,
            "names": [header_name],
            "line": node.start_point[0] + 1,
            "is_system_header": is_system_header,
            "raw_path": path_text,
        })

    return includes


def _resolve_cpp_include_path(
    import_info: dict,
    filepath: Path,
    base_dir: Path,
    include_paths: list[Path] | None = None,
) -> str | None:
    """Resolve a C/C++ include path to a file path.

    Resolution order:
    1. For "..." includes: relative to current file first
    2. For both: check include_paths if provided
    3. For <...> system headers: return None (external)

    Args:
        import_info: Include info dict from _extract_cpp_includes.
        filepath: Current file path.
        base_dir: Repository root.
        include_paths: Optional list of additional include directories.

    Returns:
        Relative path if resolvable, None if external/system.
    """
    include_path = import_info["source"]
    is_system_header = import_info["is_system_header"]

    # Skip system headers entirely unless they happen to be in our codebase
    if is_system_header:
        # Still try to resolve in case it's a local header with wrong syntax
        pass

    # For local includes, try relative to current file first
    if not is_system_header:
        current_dir = filepath.parent
        candidate = current_dir / include_path
        if candidate.exists():
            try:
                return str(candidate.relative_to(base_dir))
            except ValueError:
                return str(candidate)

    # Try include paths
    if include_paths:
        for include_dir in include_paths:
            candidate = include_dir / include_path
            if candidate.exists():
                try:
                    return str(candidate.relative_to(base_dir))
                except ValueError:
                    return str(candidate)

    # Try common include directory patterns within the repo
    common_include_dirs = [
        base_dir / "include",
        base_dir / "src",
        base_dir / "inc",
        base_dir / "headers",
        # Also try parent directories
        filepath.parent.parent / "include",
        filepath.parent.parent,
    ]

    for include_dir in common_include_dirs:
        if not include_dir.exists():
            continue
        candidate = include_dir / include_path
        if candidate.exists():
            try:
                return str(candidate.relative_to(base_dir))
            except ValueError:
                return str(candidate)

    # Try as relative to base_dir
    candidate = base_dir / include_path
    if candidate.exists():
        try:
            return str(candidate.relative_to(base_dir))
        except ValueError:
            return str(candidate)

    # System header or unresolvable local header
    return None


def _extract_cpp_namespaces(tree: Node) -> list[dict]:
    """Extract namespace definitions from C/C++ AST.

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and nested info.
    """
    namespaces = []

    for node in _find_nodes(tree, {"namespace_definition"}):
        name_node = _get_child_by_field(node, "name")
        name = name_node.text.decode() if name_node else "(anonymous)"

        namespaces.append({
            "name": name,
            "line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        })

    return namespaces


def _extract_cpp_templates(tree: Node) -> list[dict]:
    """Extract template declarations from C/C++ AST.

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and template parameters.
    """
    templates = []

    for node in _find_nodes(tree, {"template_declaration"}):
        # Get template parameters
        params_node = _get_child_by_field(node, "parameters")
        params = []
        if params_node:
            for child in params_node.children:
                if child.type in (
                    "type_parameter_declaration",
                    "parameter_declaration",
                ):
                    params.append(child.text.decode())

        # Get the templated declaration (class or function)
        templated = None
        for child in node.children:
            if child.type in (
                "function_definition",
                "class_specifier",
                "struct_specifier",
            ):
                templated = child
                break

        if templated:
            name = None
            if templated.type == "function_definition":
                name = _get_function_name_from_declarator(templated)
            elif templated.type in ("class_specifier", "struct_specifier"):
                for c in templated.children:
                    if c.type == "type_identifier":
                        name = c.text.decode()
                        break

            if name:
                templates.append({
                    "name": name,
                    "line": node.start_point[0] + 1,
                    "parameters": params,
                    "kind": templated.type,
                })

    return templates
