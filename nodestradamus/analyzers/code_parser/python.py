"""Python-specific parsing functions.

Handles Python import extraction, class inheritance, constants, and internal calls.
"""

from pathlib import Path

from tree_sitter import Node

from nodestradamus.analyzers.code_parser.base import (
    _find_nodes,
    _get_child_by_field,
    _get_child_by_type,
)


def _extract_python_constants(tree: Node) -> list[dict]:
    """Extract top-level constant assignments from Python AST.

    Detects module-level assignments with UPPER_CASE names (convention for constants).
    Also extracts type-annotated module-level variables.

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and value_preview.
    """
    constants = []

    # Only look at direct children of the module (top-level)
    for child in tree.children:
        if child.type == "expression_statement":
            # Check for assignment
            for expr in child.children:
                if expr.type == "assignment":
                    # Get the left-hand side (target)
                    left = _get_child_by_field(expr, "left")
                    if left and left.type == "identifier":
                        name = left.text.decode()
                        # Check if it's UPPER_CASE (constant convention)
                        if name.isupper() or "_" in name and name == name.upper():
                            # Get value preview (first 50 chars)
                            right = _get_child_by_field(expr, "right")
                            value_preview = ""
                            if right:
                                value_text = right.text.decode()
                                value_preview = value_text[:50] + ("..." if len(value_text) > 50 else "")
                            constants.append({
                                "name": name,
                                "line": child.start_point[0] + 1,
                                "end_line": child.end_point[0] + 1,
                                "value_preview": value_preview,
                            })

        # Also check for annotated assignments (e.g., CONST: str = "value")
        elif child.type == "annotated_assignment":
            left = None
            for sub in child.children:
                if sub.type == "identifier":
                    left = sub
                    break
            if left:
                name = left.text.decode()
                if name.isupper() or "_" in name and name == name.upper():
                    constants.append({
                        "name": name,
                        "line": child.start_point[0] + 1,
                        "end_line": child.end_point[0] + 1,
                        "value_preview": "",
                    })

    return constants


def _extract_python_internal_calls(
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

    # Find all function definitions and their bodies
    for func_node in _find_nodes(tree, {"function_definition"}):
        # Get function name
        name_node = _get_child_by_field(func_node, "name")
        if not name_node:
            continue
        caller_name = name_node.text.decode()

        # Find all call expressions within this function
        for call_node in _find_nodes(func_node, {"call"}):
            # Get the function being called
            func = _get_child_by_field(call_node, "function")
            if func and func.type == "identifier":
                callee_name = func.text.decode()
                # Check if callee is defined in this file
                if callee_name in defined_functions and callee_name != caller_name:
                    calls.append({
                        "caller": caller_name,
                        "callee": callee_name,
                        "line": call_node.start_point[0] + 1,
                    })

    return calls


def _extract_python_imports(
    tree: Node,
    filepath: Path,
    base_dir: Path,
) -> list[dict]:
    """Extract imports from Python AST."""
    imports = []

    for node in _find_nodes(tree, {"import_statement", "import_from_statement"}):
        if node.type == "import_statement":
            # import foo, bar
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(
                        {
                            "source": child.text.decode(),
                            "names": [],
                            "line": node.start_point[0] + 1,
                        }
                    )
        elif node.type == "import_from_statement":
            # from foo import bar, baz
            # The first dotted_name is the module, subsequent ones are import names
            module = ""
            names = []
            found_import_keyword = False

            for child in node.children:
                if child.type == "dotted_name":
                    if not found_import_keyword:
                        # This is the module name
                        module = child.text.decode()
                    else:
                        # This is an imported name
                        names.append(child.text.decode())
                elif child.type == "import":
                    found_import_keyword = True
                elif child.type == "aliased_import":
                    # Handle "from x import y as z" - get the original name
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            names.append(sub.text.decode())
                            break

            imports.append(
                {
                    "source": module,
                    "names": names,
                    "line": node.start_point[0] + 1,
                }
            )

    return imports


def _extract_class_parent_python(node: Node) -> str | None:
    """Extract parent class name for Python."""
    args = _get_child_by_type(node, "argument_list")
    if args:
        for child in args.children:
            if child.type == "identifier":
                return child.text.decode()
    return None
