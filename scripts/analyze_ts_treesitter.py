#!/usr/bin/env python3
"""
Analyze TypeScript/JavaScript code dependencies using tree-sitter.

More accurate than regex-based parsing - uses actual AST.

Usage:
    python analyze_ts_treesitter.py src/
    python analyze_ts_treesitter.py src/ > .cursor/graph/ts-deps.json
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tree_sitter_javascript as ts_javascript
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Node, Parser


class PathAliasResolver:
    """Resolves TypeScript path aliases from tsconfig.json."""

    def __init__(self, base_dir: Path, tsconfig_path: Path | None = None):
        """Initialize resolver.

        Args:
            base_dir: Project root directory for relative path resolution.
            tsconfig_path: Optional explicit path to tsconfig.json.
                          If not provided, searches in base_dir.
        """
        self.base_dir = base_dir.resolve()
        self.tsconfig_dir: Path | None = None
        self.base_url: Path | None = None
        self.paths: dict[str, list[str]] = {}

        if tsconfig_path:
            self._load_from_path(tsconfig_path)
        else:
            self._load_tsconfig()

    def _load_from_path(self, tsconfig_path: Path) -> None:
        """Load from a specific tsconfig.json path."""
        if tsconfig_path.exists():
            try:
                self.tsconfig_dir = tsconfig_path.parent
                self._parse_tsconfig(tsconfig_path)
            except (json.JSONDecodeError, OSError):
                pass

    def _load_tsconfig(self) -> None:
        """Load and parse tsconfig.json, following extends if needed."""
        tsconfig_path = self.base_dir / "tsconfig.json"
        if not tsconfig_path.exists():
            # Try jsconfig.json for JavaScript projects
            tsconfig_path = self.base_dir / "jsconfig.json"
            if not tsconfig_path.exists():
                return

        try:
            self.tsconfig_dir = tsconfig_path.parent
            self._parse_tsconfig(tsconfig_path)
        except (json.JSONDecodeError, OSError):
            pass

    @classmethod
    def find_nearest(cls, filepath: Path, repo_root: Path) -> "PathAliasResolver | None":
        """Find the nearest tsconfig.json by walking up from filepath.

        Args:
            filepath: The source file to find tsconfig for.
            repo_root: Stop searching at this directory.

        Returns:
            PathAliasResolver if tsconfig found, None otherwise.
        """
        current = filepath.parent.resolve()
        repo_root = repo_root.resolve()

        while current >= repo_root:
            for config_name in ("tsconfig.json", "jsconfig.json"):
                config_path = current / config_name
                if config_path.exists():
                    return cls(repo_root, tsconfig_path=config_path)

            if current == repo_root:
                break
            current = current.parent

        return None

    @staticmethod
    def _strip_json_comments(content: str) -> str:
        """Remove JSON5-style comments from tsconfig content.

        Handles:
        - // line comments (not inside strings)
        - /* block comments */ (not inside strings)
        """
        result = []
        i = 0
        in_string = False
        string_char = None

        while i < len(content):
            char = content[i]

            # Handle string boundaries
            if char in ('"', "'") and (i == 0 or content[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                result.append(char)
                i += 1
                continue

            # Skip comments only when not in string
            if not in_string:
                # Line comment
                if content[i : i + 2] == "//":
                    # Skip to end of line
                    while i < len(content) and content[i] != "\n":
                        i += 1
                    continue

                # Block comment
                if content[i : i + 2] == "/*":
                    i += 2
                    while i < len(content) - 1 and content[i : i + 2] != "*/":
                        i += 1
                    i += 2  # Skip */
                    continue

            result.append(char)
            i += 1

        return "".join(result)

    def _parse_tsconfig(self, config_path: Path) -> None:
        """Parse a single tsconfig file."""
        content = config_path.read_text()
        # Remove JSON comments (tsconfig allows them)
        content = self._strip_json_comments(content)

        config = json.loads(content)

        # Handle extends
        if "extends" in config:
            extends_path = config_path.parent / config["extends"]
            # Add .json if not present
            if not extends_path.suffix:
                extends_path = extends_path.with_suffix(".json")
            if extends_path.exists():
                self._parse_tsconfig(extends_path)

        compiler_options = config.get("compilerOptions", {})

        # Get baseUrl (relative to config file location)
        if "baseUrl" in compiler_options:
            base_url = compiler_options["baseUrl"]
            self.base_url = (config_path.parent / base_url).resolve()

        # Get paths
        if "paths" in compiler_options:
            self.paths.update(compiler_options["paths"])

    def resolve(self, import_path: str) -> str | None:
        """Resolve a path alias to a relative file path.

        Args:
            import_path: The import path (e.g., "@/lib/utils")

        Returns:
            Resolved relative path or None if not an alias.
        """
        if not self.paths:
            return None

        for pattern, targets in self.paths.items():
            # Convert glob pattern to regex
            # "@/*" -> match "@/" prefix
            if pattern.endswith("/*"):
                prefix = pattern[:-2]  # Remove "/*"
                if import_path.startswith(prefix + "/"):
                    # Get the rest after the prefix
                    rest = import_path[len(prefix) + 1 :]

                    # Try each target
                    for target in targets:
                        if target.endswith("/*"):
                            target_base = target[:-2]  # Remove "/*"
                        else:
                            target_base = target

                        # Resolve relative to baseUrl, tsconfig dir, or project root
                        # Priority: baseUrl > tsconfig location > repo root
                        base = self.base_url or self.tsconfig_dir or self.base_dir
                        resolved = (base / target_base / rest).resolve()

                        # Try with extensions
                        for ext in PARSERS.keys():
                            candidate = resolved.with_suffix(ext)
                            if candidate.exists():
                                try:
                                    return str(candidate.relative_to(self.base_dir))
                                except ValueError:
                                    return str(candidate)

                        # Try as directory with index
                        if resolved.is_dir():
                            for ext in PARSERS.keys():
                                index = resolved / f"index{ext}"
                                if index.exists():
                                    try:
                                        return str(index.relative_to(self.base_dir))
                                    except ValueError:
                                        return str(index)

                        # Try without extension (might be added later)
                        if resolved.exists():
                            try:
                                return str(resolved.relative_to(self.base_dir))
                            except ValueError:
                                return str(resolved)

            # Exact match (e.g., "@utils": ["./src/utils"])
            elif import_path == pattern:
                for target in targets:
                    base = self.base_url or self.tsconfig_dir or self.base_dir
                    resolved = (base / target).resolve()

                    for ext in PARSERS.keys():
                        candidate = resolved.with_suffix(ext)
                        if candidate.exists():
                            try:
                                return str(candidate.relative_to(self.base_dir))
                            except ValueError:
                                return str(candidate)

        return None

# Setup parsers
TS_LANGUAGE = Language(ts_typescript.language_typescript())
TSX_LANGUAGE = Language(ts_typescript.language_tsx())
JS_LANGUAGE = Language(ts_javascript.language())

PARSERS = {
    ".ts": Parser(TS_LANGUAGE),
    ".tsx": Parser(TSX_LANGUAGE),
    ".js": Parser(JS_LANGUAGE),
    ".jsx": Parser(JS_LANGUAGE),  # JSX uses JS parser with some extensions
    ".mjs": Parser(JS_LANGUAGE),
    ".cjs": Parser(JS_LANGUAGE),
}

SKIP_DIRS = frozenset({".git", "node_modules", "dist", "build", ".next", "coverage", "__pycache__"})


def get_parser(filepath: Path) -> Parser | None:
    """Get the appropriate parser for a file extension."""
    return PARSERS.get(filepath.suffix.lower())


def find_nodes(node: Node, types: set[str]) -> list[Node]:
    """Recursively find all nodes of given types."""
    results = []
    if node.type in types:
        results.append(node)
    for child in node.children:
        results.extend(find_nodes(child, types))
    return results


def get_child_by_type(node: Node, type_name: str) -> Node | None:
    """Get first child of a specific type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def get_child_by_field(node: Node, field_name: str) -> Node | None:
    """Get child by field name."""
    return node.child_by_field_name(field_name)


def extract_string_value(node: Node) -> str:
    """Extract string value, removing quotes."""
    text = node.text.decode()
    if text.startswith(("'", '"', "`")):
        return text[1:-1]
    return text


def extract_imports(tree: Node) -> list[dict[str, Any]]:
    """Extract all imports from the AST."""
    imports = []

    for node in find_nodes(tree, {"import_statement"}):
        # Get the source string
        source_node = get_child_by_type(node, "string")
        if source_node:
            source = extract_string_value(source_node)

            # Get imported names
            names = []
            for ident in find_nodes(node, {"identifier"}):
                # Skip if it's the source path
                if ident.parent and ident.parent.type != "string":
                    names.append(ident.text.decode())

            imports.append({
                "source": source,
                "names": names,
                "line": node.start_point[0] + 1,
            })

    # Also find require() calls
    for node in find_nodes(tree, {"call_expression"}):
        func = get_child_by_field(node, "function")
        if func and func.text == b"require":
            args = get_child_by_field(node, "arguments")
            if args:
                string_node = get_child_by_type(args, "string")
                if string_node:
                    imports.append({
                        "source": extract_string_value(string_node),
                        "names": [],
                        "line": node.start_point[0] + 1,
                        "type": "require",
                    })

    return imports


def extract_exports(tree: Node) -> list[dict[str, Any]]:
    """Extract all exports from the AST."""
    exports = []

    for node in find_nodes(tree, {"export_statement"}):
        # Check what's being exported
        for child in node.children:
            if child.type == "function_declaration":
                name_node = get_child_by_field(child, "name")
                if name_node:
                    exports.append({
                        "name": name_node.text.decode(),
                        "type": "function",
                        "line": child.start_point[0] + 1,
                    })
            elif child.type == "class_declaration":
                name_node = get_child_by_field(child, "name")
                extends_clause = get_child_by_type(child, "extends_clause")
                extends_name = None
                if extends_clause:
                    extends_ident = get_child_by_type(extends_clause, "identifier")
                    if extends_ident:
                        extends_name = extends_ident.text.decode()

                if name_node:
                    exports.append({
                        "name": name_node.text.decode(),
                        "type": "class",
                        "line": child.start_point[0] + 1,
                        "extends": extends_name,
                    })
            elif child.type == "lexical_declaration":
                # const/let exports (including arrow functions)
                for decl in find_nodes(child, {"variable_declarator"}):
                    name_node = get_child_by_field(decl, "name")
                    value_node = get_child_by_field(decl, "value")

                    if name_node:
                        export_type = "variable"
                        if value_node and value_node.type == "arrow_function":
                            export_type = "function"

                        exports.append({
                            "name": name_node.text.decode(),
                            "type": export_type,
                            "line": decl.start_point[0] + 1,
                        })

    return exports


def extract_definitions(tree: Node, filepath: Path) -> list[dict[str, Any]]:
    """Extract all function and class definitions."""
    definitions = []

    # Functions (including async)
    for node in find_nodes(tree, {"function_declaration"}):
        name_node = get_child_by_field(node, "name")
        if name_node:
            definitions.append({
                "id": f"ts:{filepath}::{name_node.text.decode()}",
                "name": name_node.text.decode(),
                "type": "function",
                "file": str(filepath),
                "line": node.start_point[0] + 1,
                "exported": node.parent and node.parent.type == "export_statement",
            })

    # Arrow functions assigned to variables
    for node in find_nodes(tree, {"variable_declarator"}):
        name_node = get_child_by_field(node, "name")
        value_node = get_child_by_field(node, "value")

        if name_node and value_node and value_node.type == "arrow_function":
            # Check if exported
            parent = node.parent
            exported = False
            while parent:
                if parent.type == "export_statement":
                    exported = True
                    break
                parent = parent.parent

            definitions.append({
                "id": f"ts:{filepath}::{name_node.text.decode()}",
                "name": name_node.text.decode(),
                "type": "function",
                "file": str(filepath),
                "line": node.start_point[0] + 1,
                "exported": exported,
            })

    # Classes
    for node in find_nodes(tree, {"class_declaration"}):
        # Class name can be 'name' field or 'type_identifier' child
        name_node = get_child_by_field(node, "name")
        if not name_node:
            name_node = get_child_by_type(node, "type_identifier")

        # Find extends clause (may be nested in class_heritage)
        extends_name = None
        class_heritage = get_child_by_type(node, "class_heritage")
        if class_heritage:
            extends_clause = get_child_by_type(class_heritage, "extends_clause")
            if extends_clause:
                extends_ident = get_child_by_type(extends_clause, "identifier")
                if extends_ident:
                    extends_name = extends_ident.text.decode()
        else:
            # Direct extends_clause (some grammars)
            extends_clause = get_child_by_type(node, "extends_clause")
            if extends_clause:
                extends_ident = get_child_by_type(extends_clause, "identifier")
                if extends_ident:
                    extends_name = extends_ident.text.decode()

        if name_node:
            definitions.append({
                "id": f"ts:{filepath}::{name_node.text.decode()}",
                "name": name_node.text.decode(),
                "type": "class",
                "file": str(filepath),
                "line": node.start_point[0] + 1,
                "extends": extends_name,
                "exported": node.parent and node.parent.type == "export_statement",
            })

    # Methods inside classes
    for node in find_nodes(tree, {"method_definition"}):
        name_node = get_child_by_field(node, "name")
        if name_node:
            # Find parent class
            parent = node.parent
            class_name = None
            while parent:
                if parent.type == "class_declaration":
                    class_name_node = get_child_by_field(parent, "name")
                    if class_name_node:
                        class_name = class_name_node.text.decode()
                    break
                parent = parent.parent

            method_name = name_node.text.decode()
            full_name = f"{class_name}.{method_name}" if class_name else method_name

            definitions.append({
                "id": f"ts:{filepath}::{full_name}",
                "name": full_name,
                "type": "method",
                "file": str(filepath),
                "line": node.start_point[0] + 1,
            })

    return definitions


def extract_calls(tree: Node, filepath: Path) -> list[dict[str, Any]]:
    """Extract function calls."""
    calls = []

    for node in find_nodes(tree, {"call_expression"}):
        func = get_child_by_field(node, "function")
        if func:
            if func.type == "identifier":
                calls.append({
                    "name": func.text.decode(),
                    "line": node.start_point[0] + 1,
                })
            elif func.type == "member_expression":
                # obj.method() calls
                prop = get_child_by_field(func, "property")
                if prop:
                    calls.append({
                        "name": prop.text.decode(),
                        "line": node.start_point[0] + 1,
                        "member_access": True,
                    })

    return calls


def resolve_import_path(
    import_path: str,
    filepath: Path,
    base_dir: Path,
    alias_resolver: PathAliasResolver | None = None,
) -> str | None:
    """Attempt to resolve an import to a file path.

    Handles:
    - Relative imports (./foo, ../bar)
    - Path aliases (@/lib/utils, ~/components)
    - Index files in directories

    Args:
        import_path: The import string from the source code.
        filepath: The file containing the import.
        base_dir: The project root directory.
        alias_resolver: Optional PathAliasResolver for tsconfig paths.

    Returns:
        Resolved relative path or None if external/unresolved.
    """
    # Try path alias first
    if alias_resolver:
        resolved = alias_resolver.resolve(import_path)
        if resolved:
            return resolved

    # Handle relative imports
    if not import_path.startswith("."):
        return None  # External package

    parent = filepath.parent
    resolved = (parent / import_path).resolve()

    # Try with extensions
    for ext in PARSERS.keys():
        candidate = resolved.with_suffix(ext)
        if candidate.exists():
            try:
                return str(candidate.relative_to(base_dir))
            except ValueError:
                return str(candidate)

    # Try as directory with index file
    if resolved.is_dir():
        for ext in PARSERS.keys():
            index = resolved / f"index{ext}"
            if index.exists():
                try:
                    return str(index.relative_to(base_dir))
                except ValueError:
                    return str(index)

    return None


def analyze_file(
    filepath: Path,
    base_dir: Path,
    alias_resolver: PathAliasResolver | None = None,
) -> dict[str, Any]:
    """Analyze a single TypeScript/JavaScript file.

    Args:
        filepath: Path to the file to analyze.
        base_dir: Project root directory.
        alias_resolver: Optional resolver for tsconfig path aliases.

    Returns:
        Dictionary with definitions, imports, calls, and edges.
    """
    parser = get_parser(filepath)
    if not parser:
        return {"error": f"Unsupported file type: {filepath.suffix}", "file": str(filepath)}

    try:
        source = filepath.read_bytes()
        tree = parser.parse(source)
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(filepath)}

    try:
        rel_path = filepath.relative_to(base_dir)
    except ValueError:
        rel_path = filepath

    imports = extract_imports(tree.root_node)
    definitions = extract_definitions(tree.root_node, rel_path)
    calls = extract_calls(tree.root_node, rel_path)

    # Add a file-level node (module node) so imports can connect to it
    file_node = {
        "id": f"ts:{rel_path}",
        "name": rel_path.stem,
        "type": "module",
        "file": str(rel_path),
        "line": 1,
        "exported": False,
    }
    definitions.insert(0, file_node)  # Add file node first

    # Build edges from imports
    edges = []

    # Create a file-level node ID
    file_node_id = f"ts:{rel_path}"

    # Add containment edges: bidirectional links between file and its symbols
    # This enables graph traversal in both directions
    for defn in definitions:
        # Skip the file node itself
        if defn["id"] == file_node_id:
            continue

        # Symbol -> File (for "what file is this symbol in?")
        edges.append({
            "from": defn["id"],
            "to": file_node_id,
            "type": "defined_in",
            "resolved": True,
        })
        # File -> Symbol (for traversing from file to its contents)
        edges.append({
            "from": file_node_id,
            "to": defn["id"],
            "type": "contains",
            "resolved": True,
        })

    for imp in imports:
        resolved = resolve_import_path(imp["source"], filepath, base_dir, alias_resolver)
        imported_names = imp.get("names", [])

        if resolved:
            target_file = f"ts:{resolved}"

            # Create file-level import edge
            edges.append({
                "from": file_node_id,
                "to": target_file,
                "type": "imports",
                "resolved": True,
                "names": imported_names,
            })

            # Also create symbol-level import edges for named imports
            # This connects the importing file to the specific symbols it imports
            for name in imported_names:
                edges.append({
                    "from": file_node_id,
                    "to": f"ts:{resolved}::{name}",
                    "type": "imports_symbol",
                    "resolved": True,
                })
        else:
            # External/unresolved import - just create file-level edge
            edges.append({
                "from": file_node_id,
                "to": imp["source"],
                "type": "imports",
                "resolved": False,
                "names": imported_names,
            })

    return {
        "definitions": definitions,
        "imports": imports,
        "calls": calls,
        "edges": edges,
    }


def analyze_directory(directory: Path) -> dict[str, Any]:
    """Analyze all TypeScript/JavaScript files in a directory.

    Supports monorepos by finding the nearest tsconfig.json for each file.

    Args:
        directory: Path to the directory to analyze.

    Returns:
        Dictionary containing nodes, edges, files, errors, and metadata.
    """
    directory = Path(directory).resolve()
    nodes = []
    edges = []
    files = []
    errors = []

    # Pre-scan for all tsconfig.json files and cache resolvers
    tsconfig_resolvers: dict[Path, PathAliasResolver] = {}

    for tsconfig_path in directory.rglob("tsconfig.json"):
        if any(part in SKIP_DIRS for part in tsconfig_path.parts):
            continue
        try:
            resolver = PathAliasResolver(directory, tsconfig_path=tsconfig_path)
            tsconfig_resolvers[tsconfig_path.parent] = resolver
        except Exception:
            pass  # Skip invalid tsconfigs

    # Also check for jsconfig.json
    for jsconfig_path in directory.rglob("jsconfig.json"):
        if any(part in SKIP_DIRS for part in jsconfig_path.parts):
            continue
        if jsconfig_path.parent not in tsconfig_resolvers:  # tsconfig takes priority
            try:
                resolver = PathAliasResolver(directory, tsconfig_path=jsconfig_path)
                tsconfig_resolvers[jsconfig_path.parent] = resolver
            except Exception:
                pass

    # Fallback resolver for root (may have no paths)
    if directory not in tsconfig_resolvers:
        tsconfig_resolvers[directory] = PathAliasResolver(directory)

    def get_resolver_for_file(filepath: Path) -> PathAliasResolver:
        """Find the nearest resolver by walking up from filepath."""
        current = filepath.parent
        while current >= directory:
            if current in tsconfig_resolvers:
                return tsconfig_resolvers[current]
            current = current.parent
        return tsconfig_resolvers[directory]

    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in PARSERS:
            continue
        if any(part in SKIP_DIRS for part in filepath.parts):
            continue

        try:
            rel_path = filepath.relative_to(directory)
        except ValueError:
            rel_path = filepath

        files.append(str(rel_path))

        # Get the appropriate resolver for this file's location
        alias_resolver = get_resolver_for_file(filepath)
        result = analyze_file(filepath, directory, alias_resolver)
        if "error" in result:
            errors.append(result)
        else:
            nodes.extend(result["definitions"])
            edges.extend(result["edges"])

    # Build lookup for inheritance edges
    defined_classes = {n["name"]: n["id"] for n in nodes if n["type"] == "class"}

    # Add inheritance edges
    for node in nodes:
        if node["type"] == "class" and node.get("extends"):
            parent = node["extends"]
            edges.append({
                "from": node["id"],
                "to": defined_classes.get(parent, parent),
                "type": "extends",
                "resolved": parent in defined_classes,
            })

    return {
        "nodes": nodes,
        "files": sorted(files),
        "edges": edges,
        "errors": errors,
        "metadata": {
            "analyzer": "typescript_treesitter",
            "version": "0.2.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_directory": str(directory),
            "file_count": len(files),
            "parser": "tree-sitter",
        },
    }


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_ts_treesitter.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = analyze_directory(directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
