"""Rust-specific parsing functions.

Handles Rust import (use) extraction, path resolution, constants, and internal calls.
"""

from pathlib import Path

from tree_sitter import Node

from nodestradamus.analyzers.code_parser.base import _find_nodes, _get_child_by_field


def _extract_rust_constants(tree: Node) -> list[dict]:
    """Extract const and static declarations from Rust AST.

    Detects:
    - const UPPER_CASE: Type = value;
    - static UPPER_CASE: Type = value;
    - pub const/static variants

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and value_preview.
    """
    constants = []

    for node in _find_nodes(tree, {"const_item", "static_item"}):
        # Get the identifier (name)
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode()
                break

        if not name:
            continue

        # Check if UPPER_CASE (Rust convention for constants)
        if name.isupper() or ("_" in name and name == name.upper()):
            # Get value preview from the entire node text
            node_text = node.text.decode()
            # Extract just the value part after the =
            if "=" in node_text:
                value_part = node_text.split("=", 1)[1].strip().rstrip(";")
                value_preview = value_part[:50] + ("..." if len(value_part) > 50 else "")
            else:
                value_preview = ""

            constants.append({
                "name": name,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "value_preview": value_preview,
            })

    return constants


def _extract_rust_internal_calls(
    tree: Node,
    defined_functions: set[str],
) -> list[dict]:
    """Extract internal function calls within the same file.

    Finds all function calls where the callee is a function defined in this file.
    Handles both simple calls (func()) and method calls (Self::func()).

    Args:
        tree: Root node of the parsed AST.
        defined_functions: Set of function names defined in this file.

    Returns:
        List of dicts with caller, callee, and line.
    """
    calls = []

    # Find all function definitions
    for func_node in _find_nodes(tree, {"function_item"}):
        # Get function name
        caller_name = None
        for child in func_node.children:
            if child.type == "identifier":
                caller_name = child.text.decode()
                break

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
            elif func.type == "scoped_identifier":
                # Handle Self::func() or module::func()
                # Get the last identifier in the path
                for child in reversed(func.children):
                    if child.type == "identifier":
                        callee_name = child.text.decode()
                        break

            if callee_name and callee_name in defined_functions and callee_name != caller_name:
                calls.append({
                    "caller": caller_name,
                    "callee": callee_name,
                    "line": call_node.start_point[0] + 1,
                })

    return calls


def _extract_rust_imports(
    tree: Node,
    filepath: Path,
    base_dir: Path,
) -> list[dict]:
    """Extract imports from Rust AST (use declarations).

    Handles:
    - use crate::module::item;
    - use super::module;
    - use self::module;
    - use std::collections::HashMap;
    - use module::{item1, item2};
    """
    imports = []

    for node in _find_nodes(tree, {"use_declaration"}):
        # Get the use tree (the path after 'use')
        use_tree = None
        for child in node.children:
            if child.type in ("use_tree", "scoped_identifier", "identifier", "use_as_clause"):
                use_tree = child
                break

        if not use_tree:
            continue

        # Extract the full path text
        path_text = use_tree.text.decode()

        # Parse the path to get module and names
        # Handle use lists like use module::{a, b}
        if "::{" in path_text:
            base_path = path_text.split("::{")[0]
            names_part = path_text.split("::{")[1].rstrip("}")
            names = [n.strip() for n in names_part.split(",")]
        else:
            # Single import
            parts = path_text.split("::")
            base_path = "::".join(parts[:-1]) if len(parts) > 1 else ""
            names = [parts[-1]] if parts else []

        imports.append(
            {
                "source": path_text,
                "base_path": base_path,
                "names": names,
                "line": node.start_point[0] + 1,
                "is_crate": path_text.startswith("crate::"),
                "is_super": path_text.startswith("super::"),
                "is_self": path_text.startswith("self::"),
            }
        )

    return imports


def _resolve_rust_import_path(
    import_info: dict,
    filepath: Path,
    base_dir: Path,
) -> str | None:
    """Resolve a Rust import path to a file path.

    Handles crate::, super::, self::, and external crates.
    Returns relative path if resolvable, None if external.
    """
    source = import_info["source"]

    # External crate (std, etc.) - not resolvable
    if not (import_info["is_crate"] or import_info["is_super"] or import_info["is_self"]):
        # Check if it could be a local module in the same crate
        first_part = source.split("::")[0]
        # Try to find it as a sibling module
        parent = filepath.parent
        for candidate in [parent / f"{first_part}.rs", parent / first_part / "mod.rs"]:
            if candidate.exists():
                try:
                    return str(candidate.relative_to(base_dir))
                except ValueError:
                    return str(candidate)
        return None

    # Remove the prefix and get path parts
    if import_info["is_crate"]:
        # crate:: refers to crate root
        path_without_prefix = source.replace("crate::", "")
        # Find crate root (look for Cargo.toml or src/lib.rs)
        current = filepath.parent
        while current != base_dir and current != current.parent:
            if (current / "Cargo.toml").exists():
                crate_root = current / "src"
                break
            current = current.parent
        else:
            crate_root = base_dir / "src"

        parts = path_without_prefix.split("::")
        target = crate_root
        for part in parts[:-1]:  # All but last (which might be a symbol)
            target = target / part

    elif import_info["is_super"]:
        # super:: goes up one module level
        path_without_prefix = source.replace("super::", "")
        target = filepath.parent.parent
        parts = path_without_prefix.split("::")
        for part in parts[:-1]:
            target = target / part

    elif import_info["is_self"]:
        # self:: refers to current module
        path_without_prefix = source.replace("self::", "")
        target = filepath.parent
        parts = path_without_prefix.split("::")
        for part in parts[:-1]:
            target = target / part

    else:
        return None

    # Try to find the file
    for candidate in [target.with_suffix(".rs"), target / "mod.rs"]:
        if candidate.exists():
            try:
                return str(candidate.relative_to(base_dir))
            except ValueError:
                return str(candidate)

    return None
