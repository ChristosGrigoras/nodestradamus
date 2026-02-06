"""Core classes, cache management, and tree-sitter helpers.

Provides the foundational components for multi-language code parsing.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import msgpack
from tree_sitter import Language, Node

from nodestradamus.analyzers.ignore import DEFAULT_IGNORES
from nodestradamus.logging import logger

# Parse cache version - bump when cache format changes
# 2.0: Switched to MessagePack binary format for faster I/O
# 2.1: Added content_hash for more reliable change detection
PARSE_CACHE_VERSION = "2.1"


@dataclass
class FileCacheEntry:
    """Cache entry for a single parsed file."""

    mtime: float  # File modification time
    size: int  # File size in bytes
    nodes: list[dict]  # Parsed nodes as dicts
    edges: list[dict]  # Parsed edges as dicts
    content_hash: str = ""  # SHA256 hash of file contents (optional for backward compat)


@dataclass
class ParseCache:
    """Cache for parsed files to avoid re-parsing unchanged files."""

    version: str  # Cache format version
    created_at: str  # ISO timestamp
    files: dict[str, FileCacheEntry]  # path -> cache entry


def _get_parse_cache_path(directory: Path) -> Path:
    """Get the path to the parse cache file for a directory.

    Args:
        directory: The directory being parsed.

    Returns:
        Path to the cache file (MessagePack format).
    """
    return directory / ".nodestradamus" / "parse_cache.msgpack"


def _get_legacy_cache_path(directory: Path) -> Path:
    """Get the path to the legacy JSON cache file (for migration)."""
    return directory / ".nodestradamus" / "parse_cache.json"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file's contents.

    Args:
        filepath: Path to the file to hash.

    Returns:
        Hex digest of the SHA256 hash, or empty string on error.
    """
    try:
        with filepath.open("rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def load_parse_cache(directory: Path) -> ParseCache | None:
    """Load cached parse results if valid.

    Tries MessagePack format first, falls back to legacy JSON for migration.

    Args:
        directory: The directory being parsed.

    Returns:
        ParseCache if cache exists and is valid, None otherwise.
    """
    cache_path = _get_parse_cache_path(directory)
    legacy_path = _get_legacy_cache_path(directory)

    # Try MessagePack first
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                data = msgpack.unpack(f, raw=False)

            if data.get("version") != PARSE_CACHE_VERSION:
                logger.info("  Parse cache version mismatch, ignoring cache")
                return None

            files = {}
            for path, entry in data.get("files", {}).items():
                files[path] = FileCacheEntry(
                    mtime=entry["mtime"],
                    size=entry["size"],
                    nodes=entry["nodes"],
                    edges=entry["edges"],
                    content_hash=entry.get("content_hash", ""),  # Optional for backward compat
                )

            return ParseCache(
                version=data["version"],
                created_at=data["created_at"],
                files=files,
            )

        except (msgpack.UnpackException, msgpack.ExtraData, KeyError, TypeError, ValueError) as e:
            logger.warning("  Failed to load parse cache: %s", e)
            return None

    # Fall back to legacy JSON (for migration)
    if legacy_path.exists():
        try:
            with legacy_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Version will mismatch (1.0 vs 2.0), but we can migrate the data
            # by treating it as stale and re-parsing on next save
            logger.info("  Found legacy JSON cache, will migrate to MessagePack")
            return None  # Force re-parse and save as MessagePack

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return None


def save_parse_cache(directory: Path, cache: ParseCache) -> None:
    """Save parse results to cache using MessagePack format.

    Args:
        directory: The directory being parsed.
        cache: The cache to save.
    """
    cache_path = _get_parse_cache_path(directory)
    legacy_path = _get_legacy_cache_path(directory)

    try:
        # Ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for serialization
        data = {
            "version": cache.version,
            "created_at": cache.created_at,
            "files": {
                path: {
                    "mtime": entry.mtime,
                    "size": entry.size,
                    "nodes": entry.nodes,
                    "edges": entry.edges,
                    "content_hash": entry.content_hash,
                }
                for path, entry in cache.files.items()
            },
        }

        # Write MessagePack (binary, compact)
        with cache_path.open("wb") as f:
            msgpack.pack(data, f)

        logger.info("  Saved parse cache with %d files (msgpack)", len(cache.files))

        # Clean up legacy JSON cache if it exists
        if legacy_path.exists():
            try:
                legacy_path.unlink()
                logger.info("  Removed legacy JSON cache")
            except OSError:
                pass

    except OSError as e:
        logger.warning("  Failed to save parse cache: %s", e)


def is_file_stale(
    file_path: Path,
    cache_key: str,
    cache: ParseCache,
) -> bool:
    """Check if a file needs to be re-parsed based on mtime/size and content hash.

    Uses a two-level check:
    1. Fast path: If mtime and size match, file is not stale.
    2. Content hash: If mtime/size differ but content hash matches, file is not stale.
       This handles cases like git checkout where mtime changes but content doesn't.

    Args:
        file_path: The absolute file path to stat.
        cache_key: The relative path string used as cache key.
        cache: The current parse cache.

    Returns:
        True if file is stale (needs re-parsing), False if cache is valid.
    """
    # Not in cache = stale
    if cache_key not in cache.files:
        return True

    entry = cache.files[cache_key]

    try:
        stat = file_path.stat()

        # Fast path: mtime and size match = not stale
        if stat.st_mtime == entry.mtime and stat.st_size == entry.size:
            return False

        # Size changed = definitely stale (skip hash computation)
        if stat.st_size != entry.size:
            return True

        # Size matches but mtime differs: check content hash if available
        # This handles git checkout, touch, copy operations where content is unchanged
        if entry.content_hash:
            current_hash = compute_file_hash(file_path)
            if current_hash and current_hash == entry.content_hash:
                # Content unchanged, update mtime in cache entry to avoid future recomputation
                entry.mtime = stat.st_mtime
                return False

        # No hash available or hash differs = stale
        return True

    except OSError:
        # File doesn't exist or can't be stat'd = stale
        return True


# Lazy imports for tree-sitter language bindings
_LANGUAGES: dict[str, Language] = {}


def _get_language(name: str) -> Language | None:
    """Lazily load tree-sitter language bindings."""
    if name in _LANGUAGES:
        return _LANGUAGES[name]

    try:
        if name == "python":
            import tree_sitter_python as ts_python

            _LANGUAGES[name] = Language(ts_python.language())
        elif name == "typescript":
            import tree_sitter_typescript as ts_typescript

            _LANGUAGES[name] = Language(ts_typescript.language_typescript())
        elif name == "tsx":
            import tree_sitter_typescript as ts_typescript

            _LANGUAGES[name] = Language(ts_typescript.language_tsx())
        elif name == "javascript":
            import tree_sitter_javascript as ts_javascript

            _LANGUAGES[name] = Language(ts_javascript.language())
        elif name == "rust":
            import tree_sitter_rust as ts_rust

            _LANGUAGES[name] = Language(ts_rust.language())
        elif name == "sql":
            import tree_sitter_sql as ts_sql

            _LANGUAGES[name] = Language(ts_sql.language())
        elif name == "bash":
            import tree_sitter_bash as ts_bash

            _LANGUAGES[name] = Language(ts_bash.language())
        elif name == "json":
            import tree_sitter_json as ts_json

            _LANGUAGES[name] = Language(ts_json.language())
        elif name == "markdown":
            import tree_sitter_markdown as ts_markdown

            _LANGUAGES[name] = Language(ts_markdown.language())
        elif name == "cpp":
            import tree_sitter_cpp as ts_cpp

            _LANGUAGES[name] = Language(ts_cpp.language())
        else:
            return None
    except ImportError:
        return None

    return _LANGUAGES.get(name)


# File extension to language mapping
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".sql": "sql",
    ".pgsql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".json": "json",
    ".md": "markdown",
    ".mdx": "markdown",
    ".mdc": "markdown",
    # C/C++/Objective-C
    ".c": "cpp",
    ".h": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".h++": "cpp",
    ".m": "cpp",  # Objective-C (tree-sitter-cpp handles basic parsing)
    ".mm": "cpp",  # Objective-C++
}

# Directories to skip - using central ignore module
# Adding .nodestradamus (our cache directory) to the set
SKIP_DIRS = DEFAULT_IGNORES | {".nodestradamus"}


@dataclass
class FieldInfo:
    """A field/column in a schema-defining construct."""

    name: str
    type: str  # Normalized or raw type string
    nullable: bool = True
    references: str | None = None  # FK target for SQL (e.g., 'users.id')


@dataclass
class CodeNode:
    """A node in the dependency graph (function, class, module, etc.)."""

    id: str
    name: str
    type: str  # module, function, class, method, table, config
    file: str
    line: int
    exported: bool = False
    language: str = ""
    fields: list[FieldInfo] | None = None  # Schema fields for classes/tables/interfaces


@dataclass
class CodeEdge:
    """An edge in the dependency graph."""

    source: str
    target: str
    type: str  # contains, defined_in, imports, calls, extends
    resolved: bool = True
    names: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing a single file."""

    nodes: list[CodeNode]
    edges: list[CodeEdge]
    errors: list[dict] = field(default_factory=list)


@dataclass
class LanguageConfig:
    """Configuration for parsing a specific language."""

    name: str
    prefix: str  # Node ID prefix (py, ts, etc.)

    # Node types to look for
    function_types: set[str]
    class_types: set[str]
    import_types: set[str]

    # Field names for extracting info
    name_field: str = "name"

    # Custom extractors (optional)
    extract_imports: Callable[[Node, Path, Path], list[dict]] | None = None
    extract_class_parent: Callable[[Node], str | None] | None = None


# Language configurations
PYTHON_CONFIG = LanguageConfig(
    name="python",
    prefix="py",
    function_types={"function_definition"},
    class_types={"class_definition"},
    import_types={"import_statement", "import_from_statement"},
)

TYPESCRIPT_CONFIG = LanguageConfig(
    name="typescript",
    prefix="ts",
    function_types={"function_declaration", "arrow_function"},
    class_types={"class_declaration", "interface_declaration", "type_alias_declaration"},
    import_types={"import_statement"},
)

JAVASCRIPT_CONFIG = LanguageConfig(
    name="javascript",
    prefix="ts",  # Use same prefix as TS for unified JS/TS graph
    function_types={"function_declaration", "arrow_function"},
    class_types={"class_declaration"},
    import_types={"import_statement"},
)

RUST_CONFIG = LanguageConfig(
    name="rust",
    prefix="rs",
    function_types={"function_item"},
    class_types={"struct_item", "enum_item", "trait_item", "impl_item"},
    import_types={"use_declaration"},
)

SQL_CONFIG = LanguageConfig(
    name="sql",
    prefix="sql",
    function_types={"create_function_statement", "create_procedure_statement"},
    class_types={"create_table_statement", "create_view_statement", "create_trigger_statement"},
    import_types=set(),
)

BASH_CONFIG = LanguageConfig(
    name="bash",
    prefix="sh",
    function_types={"function_definition"},
    class_types=set(),  # Bash doesn't have classes
    import_types={"command"},  # source/. commands are parsed as commands
)

JSON_CONFIG = LanguageConfig(
    name="json",
    prefix="json",
    function_types=set(),  # JSON doesn't have functions
    class_types={"object"},  # Top-level objects are treated as configs
    import_types=set(),  # JSON doesn't have imports
)

CPP_CONFIG = LanguageConfig(
    name="cpp",
    prefix="cpp",
    function_types={"function_definition", "template_function"},
    class_types={
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "union_specifier",
    },
    import_types={"preproc_include"},
)

LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": PYTHON_CONFIG,
    "typescript": TYPESCRIPT_CONFIG,
    "tsx": TYPESCRIPT_CONFIG,  # TSX uses same config as TS
    "javascript": JAVASCRIPT_CONFIG,
    "rust": RUST_CONFIG,
    "sql": SQL_CONFIG,
    "bash": BASH_CONFIG,
    "json": JSON_CONFIG,
    "cpp": CPP_CONFIG,
}


# =============================================================================
# Tree-sitter Helper Functions
# =============================================================================


def _find_nodes(node: Node, types: set[str]) -> list[Node]:
    """Recursively find all nodes of given types."""
    results = []
    if node.type in types:
        results.append(node)
    for child in node.children:
        results.extend(_find_nodes(child, types))
    return results


def _get_child_by_field(node: Node, field_name: str) -> Node | None:
    """Get child by field name."""
    return node.child_by_field_name(field_name)


def _get_child_by_type(node: Node, type_name: str) -> Node | None:
    """Get first child of a specific type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _extract_string_value(node: Node) -> str:
    """Extract string value, removing quotes."""
    text = node.text.decode()
    if text.startswith(("'", '"', "`")):
        return text[1:-1]
    return text


def _is_exported(node: Node) -> bool:
    """Check if a definition is exported."""
    parent = node.parent
    while parent:
        if parent.type in ("export_statement", "decorated_definition"):
            return True
        if parent.type in ("module", "program"):
            break
        parent = parent.parent
    return False


def _extract_function_name(node: Node, config: LanguageConfig) -> str | None:
    """Extract function name from a function node."""
    # Try field name first
    name_node = _get_child_by_field(node, config.name_field)
    if name_node:
        return name_node.text.decode()

    # For arrow functions, look at parent variable_declarator
    if node.type == "arrow_function" and node.parent:
        if node.parent.type == "variable_declarator":
            name_node = _get_child_by_field(node.parent, "name")
            if name_node:
                return name_node.text.decode()

    # For Rust function_item, the name is an identifier child
    if node.type == "function_item":
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()

    # For Bash function_definition, the name is a 'word' child (first one before '(')
    if node.type == "function_definition":
        for child in node.children:
            if child.type == "word":
                return child.text.decode()

    # For C/C++ function_definition, find the declarator
    if node.type in ("function_definition", "template_function"):
        declarator = _get_child_by_field(node, "declarator")
        if declarator:
            # Handle function_declarator or pointer_declarator
            while declarator and declarator.type in (
                "pointer_declarator",
                "reference_declarator",
                "parenthesized_declarator",
            ):
                declarator = _get_child_by_field(declarator, "declarator")

            if declarator and declarator.type == "function_declarator":
                # Get the identifier from the function declarator
                func_declarator = _get_child_by_field(declarator, "declarator")
                if func_declarator:
                    # Handle qualified names like ClassName::methodName
                    if func_declarator.type == "qualified_identifier":
                        # Get the last identifier (method name)
                        for child in reversed(func_declarator.children):
                            if child.type == "identifier":
                                return child.text.decode()
                            if child.type == "destructor_name":
                                return child.text.decode()
                    elif func_declarator.type == "identifier":
                        return func_declarator.text.decode()

    return None


def _extract_class_name(node: Node, config: LanguageConfig) -> str | None:
    """Extract class name from a class node."""
    name_node = _get_child_by_field(node, config.name_field)
    if name_node:
        return name_node.text.decode()

    # TypeScript might use type_identifier
    name_node = _get_child_by_type(node, "type_identifier")
    if name_node:
        return name_node.text.decode()

    # TypeScript interface/type alias - name is an identifier
    if node.type in ("interface_declaration", "type_alias_declaration"):
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode()

    # Rust struct/enum/trait - look for type_identifier child
    if node.type in ("struct_item", "enum_item", "trait_item"):
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode()

    # Rust impl - extract the type being implemented
    if node.type == "impl_item":
        # Look for the type being implemented
        # impl Trait for Type or impl Type
        for child in node.children:
            if child.type == "type_identifier":
                return f"impl_{child.text.decode()}"
            if child.type == "generic_type":
                # Handle impl<T> Type<T>
                type_id = _get_child_by_type(child, "type_identifier")
                if type_id:
                    return f"impl_{type_id.text.decode()}"

    # C/C++ class/struct/enum/union specifiers
    if node.type in ("class_specifier", "struct_specifier", "enum_specifier", "union_specifier"):
        # Name is the type_identifier child
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode()

    return None
