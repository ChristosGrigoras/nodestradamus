"""TypeScript/JavaScript-specific parsing functions.

Handles TypeScript and JavaScript import extraction, class inheritance,
constants, and internal calls.
"""

from pathlib import Path

from tree_sitter import Node

from nodestradamus.analyzers.code_parser.base import (
    _extract_string_value,
    _find_nodes,
    _get_child_by_field,
    _get_child_by_type,
)


def _extract_ts_constants(tree: Node) -> list[dict]:
    """Extract top-level const declarations from TypeScript/JavaScript AST.

    Detects:
    - const UPPER_CASE = ...
    - export const UPPER_CASE = ...

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and value_preview.
    """
    constants = []

    # Find lexical_declaration (const/let) at top level or in export_statement
    for node in _find_nodes(tree, {"lexical_declaration"}):
        # Check if this is a const declaration
        is_const = False
        for child in node.children:
            if child.type == "const":
                is_const = True
                break

        if not is_const:
            continue

        # Only consider top-level (parent is program or export_statement)
        if node.parent and node.parent.type not in ("program", "export_statement"):
            continue

        # Extract variable declarators
        for declarator in _find_nodes(node, {"variable_declarator"}):
            name_node = _get_child_by_field(declarator, "name")
            if name_node and name_node.type == "identifier":
                name = name_node.text.decode()
                # Check if UPPER_CASE
                if name.isupper() or ("_" in name and name == name.upper()):
                    value_node = _get_child_by_field(declarator, "value")
                    value_preview = ""
                    if value_node:
                        value_text = value_node.text.decode()
                        value_preview = value_text[:50] + ("..." if len(value_text) > 50 else "")
                    constants.append({
                        "name": name,
                        "line": declarator.start_point[0] + 1,
                        "end_line": declarator.end_point[0] + 1,
                        "value_preview": value_preview,
                    })

    return constants


def _extract_ts_internal_calls(
    tree: Node,
    defined_functions: set[str],
) -> list[dict]:
    """Extract internal function calls within the same file.

    Finds all function calls where the callee is a function defined in this file.

    Args:
        tree: Root node of the parsed AST.
        defined_functions: Set of function names defined in this file.

    Returns:
        List of dicts with caller, callee, and line.
    """
    calls = []

    # Find all function definitions (function_declaration and arrow_function)
    func_types = {"function_declaration", "arrow_function", "method_definition"}

    for func_node in _find_nodes(tree, func_types):
        # Get function name
        caller_name = None
        if func_node.type == "function_declaration":
            name_node = _get_child_by_field(func_node, "name")
            if name_node:
                caller_name = name_node.text.decode()
        elif func_node.type == "arrow_function":
            # Arrow functions get their name from parent variable_declarator
            if func_node.parent and func_node.parent.type == "variable_declarator":
                name_node = _get_child_by_field(func_node.parent, "name")
                if name_node:
                    caller_name = name_node.text.decode()
        elif func_node.type == "method_definition":
            name_node = _get_child_by_field(func_node, "name")
            if name_node:
                caller_name = name_node.text.decode()

        if not caller_name:
            continue

        # Find all call expressions within this function
        for call_node in _find_nodes(func_node, {"call_expression"}):
            func = _get_child_by_field(call_node, "function")
            if func and func.type == "identifier":
                callee_name = func.text.decode()
                if callee_name in defined_functions and callee_name != caller_name:
                    calls.append({
                        "caller": caller_name,
                        "callee": callee_name,
                        "line": call_node.start_point[0] + 1,
                    })

    return calls


def _extract_js_imports(
    tree: Node,
    filepath: Path,
    base_dir: Path,
) -> list[dict]:
    """Extract imports from JavaScript/TypeScript AST."""
    imports = []

    for node in _find_nodes(tree, {"import_statement"}):
        source_node = _get_child_by_type(node, "string")
        if source_node:
            source = _extract_string_value(source_node)

            names = []
            for ident in _find_nodes(node, {"identifier"}):
                if ident.parent and ident.parent.type != "string":
                    names.append(ident.text.decode())

            imports.append(
                {
                    "source": source,
                    "names": names,
                    "line": node.start_point[0] + 1,
                }
            )

    # Also find require() calls
    for node in _find_nodes(tree, {"call_expression"}):
        func = _get_child_by_field(node, "function")
        if func and func.text == b"require":
            args = _get_child_by_field(node, "arguments")
            if args:
                string_node = _get_child_by_type(args, "string")
                if string_node:
                    imports.append(
                        {
                            "source": _extract_string_value(string_node),
                            "names": [],
                            "line": node.start_point[0] + 1,
                            "type": "require",
                        }
                    )

    return imports


def _extract_class_parent_js(node: Node) -> str | None:
    """Extract parent class name for JavaScript/TypeScript."""
    # Check for extends_clause or class_heritage
    heritage = _get_child_by_type(node, "class_heritage")
    if heritage:
        extends = _get_child_by_type(heritage, "extends_clause")
        if extends:
            ident = _get_child_by_type(extends, "identifier")
            if ident:
                return ident.text.decode()

    extends = _get_child_by_type(node, "extends_clause")
    if extends:
        ident = _get_child_by_type(extends, "identifier")
        if ident:
            return ident.text.decode()

    return None
