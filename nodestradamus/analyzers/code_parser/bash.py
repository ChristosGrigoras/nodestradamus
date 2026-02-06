"""Bash-specific parsing functions.

Handles Bash source/dot command extraction, path resolution, constants,
and internal calls.
"""

from pathlib import Path

from tree_sitter import Node

from nodestradamus.analyzers.code_parser.base import _extract_string_value, _find_nodes


def _extract_bash_constants(tree: Node) -> list[dict]:
    """Extract top-level variable assignments from Bash AST.

    Detects module-level UPPER_CASE variable assignments.

    Args:
        tree: Root node of the parsed AST.

    Returns:
        List of dicts with name, line, and value_preview.
    """
    constants = []

    # Only look at direct children of the program (top-level)
    for child in tree.children:
        if child.type == "variable_assignment":
            # Get the variable name
            name_node = None
            value_node = None
            for sub in child.children:
                if sub.type == "variable_name":
                    name_node = sub
                elif sub.type in ("word", "string", "concatenation", "expansion"):
                    value_node = sub

            if name_node:
                name = name_node.text.decode()
                # Check if UPPER_CASE
                if name.isupper() or ("_" in name and name == name.upper()):
                    value_preview = ""
                    if value_node:
                        value_text = value_node.text.decode()
                        value_preview = value_text[:50] + ("..." if len(value_text) > 50 else "")
                    constants.append({
                        "name": name,
                        "line": child.start_point[0] + 1,
                        "end_line": child.end_point[0] + 1,
                        "value_preview": value_preview,
                    })

    return constants


def _extract_bash_internal_calls(
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

    # Find all function definitions
    for func_node in _find_nodes(tree, {"function_definition"}):
        # Get function name
        caller_name = None
        for child in func_node.children:
            if child.type == "word":
                caller_name = child.text.decode()
                break

        if not caller_name:
            continue

        # Find all command nodes within this function
        for cmd_node in _find_nodes(func_node, {"command"}):
            # Get the command name
            cmd_name = None
            for child in cmd_node.children:
                if child.type == "command_name":
                    for word in child.children:
                        if word.type == "word":
                            cmd_name = word.text.decode()
                            break
                    break

            if cmd_name and cmd_name in defined_functions and cmd_name != caller_name:
                calls.append({
                    "caller": caller_name,
                    "callee": cmd_name,
                    "line": cmd_node.start_point[0] + 1,
                })

    return calls


def _extract_bash_imports(
    tree: Node,
    filepath: Path,
    base_dir: Path,
) -> list[dict]:
    """Extract source/dot commands and script calls from Bash AST.

    Handles:
    - source script.sh / . script.sh
    - ./other-script.sh (direct script calls)
    - $SCRIPT_DIR/util.sh (variable-based paths)
    """
    imports = []

    for node in _find_nodes(tree, {"command"}):
        # Get the command name (first word)
        command_name = None
        args = []

        for child in node.children:
            if child.type == "command_name":
                # Get the actual word inside command_name
                for word in child.children:
                    if word.type == "word":
                        command_name = word.text.decode()
                        break
            elif child.type == "word":
                args.append(child.text.decode())
            elif child.type == "string":
                # Handle quoted strings
                args.append(_extract_string_value(child))
            elif child.type == "simple_expansion" or child.type == "expansion":
                # Variable expansion like $SCRIPT_DIR
                args.append(child.text.decode())
            elif child.type == "concatenation":
                # Concatenated path like ${SCRIPT_DIR}/file.sh
                args.append(child.text.decode())

        # Check for source or . command
        if command_name in ("source", ".") and args:
            source_path = args[0]
            imports.append(
                {
                    "source": source_path,
                    "names": [],
                    "line": node.start_point[0] + 1,
                    "type": "source",
                }
            )
        # Check for direct script execution (./script.sh or path/script.sh)
        elif command_name and (
            command_name.endswith(".sh")
            or command_name.endswith(".bash")
            or command_name.startswith("./")
            or command_name.startswith("../")
        ):
            imports.append(
                {
                    "source": command_name,
                    "names": [],
                    "line": node.start_point[0] + 1,
                    "type": "exec",
                }
            )

    return imports


def _resolve_bash_import_path(
    import_info: dict,
    filepath: Path,
    base_dir: Path,
) -> str | None:
    """Resolve a Bash source/exec path to a file path.

    Args:
        import_info: Import info with source path.
        filepath: Path to the current file.
        base_dir: Repository base directory.

    Returns:
        Relative path if resolvable, None if not found or contains variables.
    """
    source = import_info["source"]

    # Skip paths with unresolved variables
    if "$" in source:
        return None

    # Ensure base_dir is absolute for proper relative_to comparison
    base_dir = base_dir.resolve()

    # Handle relative paths
    if source.startswith("./") or source.startswith("../"):
        resolved = (filepath.parent / source).resolve()
    elif source.startswith("/"):
        # Absolute path - check if it's within base_dir
        resolved = Path(source)
    else:
        # Relative to current directory
        resolved = (filepath.parent / source).resolve()

    # Check if file exists
    if resolved.exists() and resolved.is_file():
        try:
            return str(resolved.relative_to(base_dir))
        except ValueError:
            return str(resolved)

    # Try adding .sh extension if not present
    if not source.endswith((".sh", ".bash")):
        for ext in (".sh", ".bash"):
            candidate = resolved.with_suffix(ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(base_dir))
                except ValueError:
                    return str(candidate)

    return None
