"""SQL-specific parsing functions.

Handles PostgreSQL, MySQL, and standard SQL parsing using tree-sitter
with regex fallback for constructs tree-sitter doesn't handle well.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser

from nodestradamus.analyzers.code_parser.base import (
    CodeEdge,
    CodeNode,
    FieldInfo,
    LanguageConfig,
)

# =============================================================================
# SQL Regex Patterns
# =============================================================================

_SQL_IDENTIFIER_PATTERN = r'"[^"]+"|[A-Za-z_][\w$]*'
_SQL_QUALIFIED_NAME_PATTERN = rf"(?:{_SQL_IDENTIFIER_PATTERN})(?:\s*\.\s*(?:{_SQL_IDENTIFIER_PATTERN}))?"

_SQL_CREATE_TABLE_RE = re.compile(
    rf"^\s*create\s+(?:or\s+replace\s+)?(?:temporary\s+|temp\s+)?"
    rf"table\s+(?:if\s+not\s+exists\s+)?(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)
_SQL_CREATE_VIEW_RE = re.compile(
    rf"^\s*create\s+(?:or\s+replace\s+)?(?:materialized\s+)?view\s+"
    rf"(?:if\s+not\s+exists\s+)?(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)
_SQL_CREATE_FUNCTION_RE = re.compile(
    rf"^\s*create\s+(?:or\s+replace\s+)?function\s+"
    rf"(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)
_SQL_CREATE_PROCEDURE_RE = re.compile(
    rf"^\s*create\s+(?:or\s+replace\s+)?procedure\s+"
    rf"(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)
_SQL_CREATE_TRIGGER_RE = re.compile(
    rf"^\s*create\s+(?:constraint\s+)?trigger\s+"
    rf"(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)
_SQL_TRIGGER_ON_RE = re.compile(
    rf"\bon\s+(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE,
)
_SQL_TABLE_REF_RE = re.compile(
    rf"\b(from|join|update|into|delete\s+from|insert\s+into|references)\s+"
    rf"(?:only\s+)?(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})",
    re.IGNORECASE,
)
_SQL_FUNCTION_CALL_RE = re.compile(
    rf"\b(?P<name>{_SQL_QUALIFIED_NAME_PATTERN})\s*\(",
    re.IGNORECASE,
)
_SQL_AS_RE = re.compile(r"\bas\b", re.IGNORECASE)

_SQL_FUNCTION_STOPWORDS = {
    "select",
    "from",
    "where",
    "join",
    "on",
    "case",
    "when",
    "then",
    "else",
    "end",
    "cast",
    "nullif",
    "coalesce",
    "exists",
    "distinct",
    "create",
    "table",
    "view",
    "function",
    "procedure",
    "trigger",
    "language",
    "returns",
    "return",
    "begin",
}


# SQL type node patterns - tree-sitter-sql uses specific node types for each SQL type
SQL_TYPE_NODES = frozenset({
    # Numeric types (both keyword_ and short forms)
    "int", "keyword_int", "keyword_integer", "keyword_smallint", "keyword_bigint",
    "keyword_serial", "keyword_bigserial", "keyword_smallserial",
    "decimal", "keyword_decimal", "keyword_numeric", "keyword_real", "keyword_float",
    "keyword_double", "keyword_money",
    # String types
    "varchar", "keyword_varchar", "char", "keyword_char", "keyword_text", "text",
    "keyword_uuid", "keyword_json", "keyword_jsonb",
    # Date/time types
    "keyword_date", "keyword_time", "keyword_timestamp", "keyword_timestamptz",
    "keyword_interval", "timestamp",
    # Boolean
    "keyword_boolean", "keyword_bool", "boolean",
    # Binary
    "keyword_bytea", "keyword_blob", "bytea",
    # Other common types
    "keyword_array", "keyword_enum",
})


@dataclass(frozen=True)
class SqlObjectDefinition:
    """SQL object definition extracted from a statement."""

    object_type: str
    name: str
    line: int
    statement: str
    start_index: int


# =============================================================================
# SQL Helper Functions
# =============================================================================


def _line_number_from_index(source: str, index: int) -> int:
    """Return 1-based line number for a byte index."""
    return source.count("\n", 0, index) + 1


def _normalize_sql_identifier(identifier: str) -> str:
    """Normalize a SQL identifier, stripping quotes and folding case."""
    identifier = identifier.strip()
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier[1:-1].replace('""', '"')
    return identifier.lower()


def _normalize_sql_name(name: str) -> str:
    """Normalize a qualified SQL name (schema.object)."""
    parts = [p for p in re.split(r"\s*\.\s*", name.strip()) if p]
    normalized = [_normalize_sql_identifier(part) for part in parts]
    return ".".join(normalized)


def _find_dollar_tag(source: str, index: int) -> str | None:
    """Find a dollar-quote tag like $$ or $tag$ starting at index."""
    if source[index] != "$":
        return None
    end = source.find("$", index + 1)
    if end == -1:
        return None
    tag = source[index : end + 1]
    if re.fullmatch(r"\$[A-Za-z0-9_]*\$", tag):
        return tag
    return None


def _split_sql_statements(source: str) -> list[tuple[str, int]]:
    """Split SQL source into statements, preserving start indices."""
    statements: list[tuple[str, int]] = []
    start = 0
    i = 0
    in_single = False
    in_double = False
    dollar_tag: str | None = None
    in_line_comment = False
    in_block_comment = False

    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if dollar_tag:
            if source.startswith(dollar_tag, i):
                tag_length = len(dollar_tag)
                dollar_tag = None
                i += tag_length
                continue
            i += 1
            continue

        if in_single:
            if ch == "'" and nxt == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            if ch == '"' and nxt == '"':
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == "$":
            tag = _find_dollar_tag(source, i)
            if tag:
                dollar_tag = tag
                i += len(tag)
                continue
        if ch == ";":
            statement = source[start:i].strip()
            if statement:
                statements.append((statement, start))
            start = i + 1
        i += 1

    tail = source[start:].strip()
    if tail:
        statements.append((tail, start))

    return statements


def _select_cte_section(statement: str) -> tuple[str | None, int]:
    """Select the CTE section from a SQL statement, if present."""
    stripped = statement.lstrip()
    if stripped.lower().startswith("with"):
        offset = len(statement) - len(stripped)
        return stripped, offset

    match = _SQL_AS_RE.search(statement)
    if not match:
        return None, 0

    tail = statement[match.end() :]
    tail_stripped = tail.lstrip()
    if tail_stripped.lower().startswith("with"):
        offset = match.end() + (len(tail) - len(tail_stripped))
        return tail_stripped, offset
    return None, 0


def _extract_cte_definitions(
    section: str, base_index: int, source: str
) -> list[tuple[str, int]]:
    """Extract CTE names from a WITH section."""
    ctes: list[tuple[str, int]] = []
    i = 0

    with_match = re.match(r"\s*with\s+(recursive\s+)?", section, re.IGNORECASE)
    if with_match:
        i = with_match.end()

    while i < len(section):
        while i < len(section) and section[i] in (" ", "\n", "\t", ","):
            i += 1
        if i >= len(section):
            break

        name_match = re.match(_SQL_QUALIFIED_NAME_PATTERN, section[i:], re.IGNORECASE)
        if not name_match:
            break
        raw_name = name_match.group(0)
        name_start = i
        i += name_match.end()

        while i < len(section) and section[i].isspace():
            i += 1

        if i < len(section) and section[i] == "(":
            depth = 1
            i += 1
            while i < len(section) and depth:
                if section[i] == "(":
                    depth += 1
                elif section[i] == ")":
                    depth -= 1
                i += 1

        as_match = re.match(r"\s*as\s*\(", section[i:], re.IGNORECASE)
        if not as_match:
            break
        i += as_match.end()

        depth = 1
        while i < len(section) and depth:
            if section[i] == "(":
                depth += 1
            elif section[i] == ")":
                depth -= 1
            i += 1

        normalized = _normalize_sql_name(raw_name)
        line = _line_number_from_index(source, base_index + name_start)
        ctes.append((normalized, line))

        while i < len(section) and section[i].isspace():
            i += 1
        if i < len(section) and section[i] == ",":
            i += 1
            continue
        break

    return ctes


def _extract_sql_definition(
    statement: str, start_index: int, source: str
) -> SqlObjectDefinition | None:
    """Extract a SQL object definition from a statement."""
    patterns = [
        ("table", _SQL_CREATE_TABLE_RE),
        ("view", _SQL_CREATE_VIEW_RE),
        ("function", _SQL_CREATE_FUNCTION_RE),
        ("procedure", _SQL_CREATE_PROCEDURE_RE),
        ("trigger", _SQL_CREATE_TRIGGER_RE),
    ]
    for object_type, pattern in patterns:
        match = pattern.match(statement)
        if match:
            raw_name = match.group("name")
            name = _normalize_sql_name(raw_name)
            line = _line_number_from_index(source, start_index)
            return SqlObjectDefinition(
                object_type=object_type,
                name=name,
                line=line,
                statement=statement,
                start_index=start_index,
            )
    return None


def _sql_reference_context(statement: str, object_type: str | None) -> str:
    """Select statement slice for extracting SQL references."""
    if object_type in {"view", "table"}:
        match = _SQL_AS_RE.search(statement)
        if match:
            return statement[match.end() :]

    if object_type in {"function", "procedure", "trigger"}:
        match = _SQL_AS_RE.search(statement)
        if match:
            return statement[match.end() :]
        begin_match = re.search(r"\bbegin\b", statement, re.IGNORECASE)
        if begin_match:
            return statement[begin_match.end() :]

    return statement


def _extract_sql_table_refs(text: str) -> set[str]:
    """Extract table-like references from SQL text."""
    refs = set()
    for match in _SQL_TABLE_REF_RE.finditer(text):
        refs.add(_normalize_sql_name(match.group("name")))
    return refs


def _extract_sql_function_calls(text: str) -> set[str]:
    """Extract function call references from SQL text."""
    calls = set()
    for match in _SQL_FUNCTION_CALL_RE.finditer(text):
        raw_name = match.group("name")
        normalized = _normalize_sql_name(raw_name)
        leaf = normalized.split(".")[-1].lower()
        if leaf in _SQL_FUNCTION_STOPWORDS:
            continue
        calls.add(normalized)
    return calls


def _resolve_sql_reference(
    name: str, object_map: dict[str, dict[str, str]], object_types: tuple[str, ...]
) -> tuple[str, bool]:
    """Resolve a SQL reference to a known object ID."""
    for obj_type in object_types:
        target_id = object_map.get(obj_type, {}).get(name)
        if target_id:
            return target_id, True
    return name, False


def _sql_node_id(rel_path: Path, object_type: str, name: str, prefix: str) -> str:
    """Build a node ID for a SQL object."""
    return f"{prefix}:{rel_path}::{object_type}::{name}"


def _sql_cte_node_id(rel_path: Path, statement_index: int, name: str, prefix: str) -> str:
    """Build a node ID for a SQL CTE."""
    return f"{prefix}:{rel_path}::cte::{statement_index}::{name}"


def _extract_trigger_target(statement: str) -> str | None:
    """Extract trigger target table from CREATE TRIGGER statement."""
    match = _SQL_TRIGGER_ON_RE.search(statement)
    if match:
        return _normalize_sql_name(match.group("name"))
    return None


# =============================================================================
# SQL Column/Field Extraction
# =============================================================================


def _extract_sql_columns(node: Node) -> list[FieldInfo]:
    """Extract columns from CREATE TABLE statement.

    Args:
        node: Tree-sitter node for a create_table statement.

    Returns:
        List of FieldInfo for each column definition.
    """
    fields: list[FieldInfo] = []

    # Find column definitions within the table body
    for child in node.children:
        if child.type in ("column_definitions", "column_definition_list"):
            for col_def in child.children:
                if col_def.type == "column_definition":
                    field = _parse_sql_column_definition(col_def)
                    if field:
                        fields.append(field)
        elif child.type == "column_definition":
            field = _parse_sql_column_definition(child)
            if field:
                fields.append(field)

    return fields


def _parse_sql_column_definition(node: Node) -> FieldInfo | None:
    """Parse a single SQL column definition node.

    Args:
        node: Tree-sitter column_definition node.

    Returns:
        FieldInfo or None if parsing fails.
    """
    col_name = None
    col_type = "unknown"
    nullable = True
    references = None

    for child in node.children:
        if child.type == "identifier" and col_name is None:
            col_name = child.text.decode()
        elif child.type in SQL_TYPE_NODES or child.type in ("column_type", "data_type", "type"):
            # Extract the full type text (e.g., "VARCHAR(255)")
            col_type = child.text.decode()
        elif child.type == "column_constraint":
            constraint_text = child.text.decode().upper()
            if "NOT NULL" in constraint_text:
                nullable = False
            if "REFERENCES" in constraint_text:
                # Extract FK target
                ref_match = re.search(r"REFERENCES\s+(\w+)(?:\s*\(\s*(\w+)\s*\))?", constraint_text)
                if ref_match:
                    table = ref_match.group(1)
                    col = ref_match.group(2) or "id"
                    references = f"{table}.{col}"
        elif child.type in ("keyword_not", "not_null"):
            # Check if followed by NULL keyword
            nullable = False
        elif child.type == "references_constraint":
            # Handle explicit REFERENCES constraint node
            for ref_child in child.children:
                if ref_child.type == "object_reference":
                    references = ref_child.text.decode()
    # Handle inline REFERENCES (keyword_references followed by object_reference)
    children_list = list(node.children)
    for i, child in enumerate(children_list):
        if child.type == "keyword_references":
            # Look for object_reference and column identifier
            ref_table = None
            ref_col = "id"  # Default to id
            for j in range(i + 1, len(children_list)):
                next_child = children_list[j]
                if next_child.type == "object_reference":
                    ref_table = next_child.text.decode()
                elif next_child.type == "identifier" and ref_table is not None:
                    ref_col = next_child.text.decode()
                    break
                elif next_child.type in (",", ")"):
                    break
            if ref_table:
                references = f"{ref_table}.{ref_col}"

    # If we still have unknown type, try to find it from any child that looks like a type
    if col_type == "unknown":
        for child in node.children:
            # Skip known non-type nodes
            if child.type in ("identifier", ",", "(", ")", "keyword_primary", "keyword_key",
                              "keyword_not", "keyword_null", "keyword_default", "keyword_references"):
                continue
            # If it starts with keyword_ and we don't have a type yet, it might be the type
            if child.type.startswith("keyword_") and col_name is not None:
                type_name = child.type.replace("keyword_", "").upper()
                if type_name not in ("PRIMARY", "KEY", "NOT", "NULL", "DEFAULT", "REFERENCES", "UNIQUE"):
                    col_type = type_name
                    break

    if col_name:
        return FieldInfo(name=col_name, type=col_type, nullable=nullable, references=references)
    return None


# =============================================================================
# SQL File Parsing (Tree-sitter)
# =============================================================================


def _parse_sql_file_treesitter(
    source: bytes,
    source_text: str,
    rel_path: Path,
    file_id: str,
    config: LanguageConfig,
    ts_language: "Language",
) -> tuple[list[CodeNode], list[CodeEdge]]:
    """Parse SQL file using tree-sitter-sql for accurate AST parsing.

    Args:
        source: Raw file bytes.
        source_text: Decoded file content.
        rel_path: Relative path for node IDs.
        file_id: File-level node ID.
        config: Language configuration.
        ts_language: tree-sitter Language object.

    Returns:
        Tuple of (nodes, edges) extracted from the SQL file.
    """
    nodes: list[CodeNode] = []
    edges: list[CodeEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    object_map: dict[str, dict[str, str]] = {
        "table": {},
        "view": {},
        "function": {},
        "procedure": {},
        "trigger": {},
    }

    parser = Parser(ts_language)
    tree = parser.parse(source)

    # Map tree-sitter node types to our object types
    ts_type_map = {
        "create_table": "table",
        "create_view": "view",
        "create_materialized_view": "view",  # Treat materialized views as views
        "create_function": "function",
        "create_procedure": "procedure",
        "create_trigger": "trigger",
    }

    # Supplement with regex for constructs tree-sitter-sql doesn't handle well
    # (e.g., CREATE PROCEDURE in PostgreSQL syntax)
    for statement, start_index in _split_sql_statements(source_text):
        # Check for procedures (tree-sitter-sql often fails to parse these)
        proc_match = _SQL_CREATE_PROCEDURE_RE.match(statement)
        if proc_match:
            name = _normalize_sql_name(proc_match.group("name"))
            if name not in object_map["procedure"]:
                obj_id = _sql_node_id(rel_path, "procedure", name, config.prefix)
                object_map["procedure"][name] = obj_id
                line = _line_number_from_index(source_text, start_index)
                nodes.append(
                    CodeNode(
                        id=obj_id,
                        name=name,
                        type="procedure",
                        file=str(rel_path),
                        line=line,
                        exported=False,
                        language=config.name,
                    )
                )
                edges.append(CodeEdge(source=obj_id, target=file_id, type="defined_in"))
                edges.append(CodeEdge(source=file_id, target=obj_id, type="contains"))

    def find_nodes_by_types(node: Node, types: set[str]) -> list[Node]:
        """Recursively find all nodes of given types."""
        results = []
        if node.type in types:
            results.append(node)
        for child in node.children:
            results.extend(find_nodes_by_types(child, types))
        return results

    def get_first_object_reference(node: Node) -> str | None:
        """Get the first object_reference child's text."""
        for child in node.children:
            if child.type == "object_reference":
                return child.text.decode()
        return None

    def get_all_object_references(node: Node) -> list[tuple[str, int]]:
        """Get all object_reference nodes with their line numbers."""
        results = []
        if node.type == "object_reference":
            results.append((node.text.decode(), node.start_point[0] + 1))
        for child in node.children:
            results.extend(get_all_object_references(child))
        return results

    def get_all_invocations(node: Node) -> list[tuple[str, int]]:
        """Get all function invocation nodes."""
        results = []
        if node.type == "invocation":
            # Get the function name (first child before parenthesis)
            for child in node.children:
                if child.type == "object_reference":
                    results.append((child.text.decode(), node.start_point[0] + 1))
                    break
                elif child.type == "identifier":
                    results.append((child.text.decode(), node.start_point[0] + 1))
                    break
        for child in node.children:
            results.extend(get_all_invocations(child))
        return results

    # Global CTE map for all statements (populated in second pass)
    all_cte_ids: dict[str, str] = {}

    # First pass: extract all definitions
    definition_types = set(ts_type_map.keys())
    for stmt_node in find_nodes_by_types(tree.root_node, definition_types):
        obj_type = ts_type_map[stmt_node.type]
        name = get_first_object_reference(stmt_node)
        if not name:
            continue

        name = _normalize_sql_name(name)
        if name in object_map[obj_type]:
            continue

        obj_id = _sql_node_id(rel_path, obj_type, name, config.prefix)
        object_map[obj_type][name] = obj_id

        # Extract fields for tables
        extracted_fields: list[FieldInfo] | None = None
        if obj_type == "table":
            extracted_fields = _extract_sql_columns(stmt_node)

        nodes.append(
            CodeNode(
                id=obj_id,
                name=name,
                type=obj_type,
                file=str(rel_path),
                line=stmt_node.start_point[0] + 1,
                exported=False,
                language=config.name,
                fields=extracted_fields if extracted_fields else None,
            )
        )
        edges.append(CodeEdge(source=obj_id, target=file_id, type="defined_in"))
        edges.append(CodeEdge(source=file_id, target=obj_id, type="contains"))

        # Create FK edges for REFERENCES constraints
        if extracted_fields:
            for field in extracted_fields:
                if field.references:
                    # Parse the reference (e.g., "users.id" -> table "users")
                    ref_parts = field.references.split(".")
                    ref_table = ref_parts[0] if ref_parts else field.references
                    # Try to resolve to a known table
                    ref_table_id = object_map.get("table", {}).get(ref_table, ref_table)
                    fk_edge_key = (obj_id, ref_table_id, "references_fk")
                    if fk_edge_key not in edge_keys:
                        edge_keys.add(fk_edge_key)
                        edges.append(
                            CodeEdge(
                                source=obj_id,
                                target=ref_table_id,
                                type="references_fk",
                                resolved=ref_table in object_map.get("table", {}),
                            )
                        )

    # Second pass: extract references, calls, and CTEs
    statement_index = 0
    for stmt_node in find_nodes_by_types(tree.root_node, {"statement"}):
        # Extract CTEs for this statement
        stmt_cte_map: dict[str, str] = {}
        cte_nodes = find_nodes_by_types(stmt_node, {"cte"})
        for cte_node in cte_nodes:
            for child in cte_node.children:
                if child.type == "identifier":
                    cte_name = child.text.decode()
                    if cte_name not in stmt_cte_map:
                        cte_id = _sql_cte_node_id(
                            rel_path, statement_index, cte_name, config.prefix
                        )
                        stmt_cte_map[cte_name] = cte_id
                        all_cte_ids[cte_name] = cte_id
                        nodes.append(
                            CodeNode(
                                id=cte_id,
                                name=cte_name,
                                type="cte",
                                file=str(rel_path),
                                line=cte_node.start_point[0] + 1,
                                exported=False,
                                language=config.name,
                            )
                        )
                        edges.append(
                            CodeEdge(source=cte_id, target=file_id, type="defined_in")
                        )
                        edges.append(
                            CodeEdge(source=file_id, target=cte_id, type="contains")
                        )
                    break
        statement_index += 1
        # Determine owner (the definition this statement belongs to)
        owner_id = file_id
        for def_type in definition_types:
            def_nodes = find_nodes_by_types(stmt_node, {def_type})
            if def_nodes:
                name = get_first_object_reference(def_nodes[0])
                if name:
                    name = _normalize_sql_name(name)
                    obj_type = ts_type_map[def_type]
                    if name in object_map[obj_type]:
                        owner_id = object_map[obj_type][name]
                break

        # Find table references in FROM, JOIN, etc.
        from_nodes = find_nodes_by_types(stmt_node, {"from", "join", "relation"})
        for from_node in from_nodes:
            for ref, _line in get_all_object_references(from_node):
                ref = _normalize_sql_name(ref)
                # Skip if it's the definition itself (exact name match, not substring)
                if owner_id != file_id and owner_id.endswith(f"::{ref}"):
                    continue
                # Check if it's a CTE reference first (statement-local)
                if ref in stmt_cte_map:
                    target_id = stmt_cte_map[ref]
                    resolved = True
                else:
                    target_id, resolved = _resolve_sql_reference(
                        ref, object_map, ("table", "view")
                    )
                key = (owner_id, target_id, "references")
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append(
                        CodeEdge(
                            source=owner_id,
                            target=target_id,
                            type="references",
                            resolved=resolved,
                        )
                    )

        # Find function calls using tree-sitter
        for call_name, _line in get_all_invocations(stmt_node):
            call_name = _normalize_sql_name(call_name)
            # Skip SQL built-in functions
            if call_name.lower() in _SQL_FUNCTION_STOPWORDS:
                continue
            target_id, resolved = _resolve_sql_reference(
                call_name, object_map, ("function", "procedure")
            )
            key = (owner_id, target_id, "calls")
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    CodeEdge(
                        source=owner_id,
                        target=target_id,
                        type="calls",
                        resolved=resolved,
                    )
                )

        # Supplement with regex for function calls in PL/pgSQL bodies
        # (tree-sitter-sql doesn't fully parse PL/pgSQL constructs like PERFORM)
        stmt_text = stmt_node.text.decode()
        for match in _SQL_FUNCTION_CALL_RE.finditer(stmt_text):
            call_name = _normalize_sql_name(match.group("name"))
            if call_name.lower() in _SQL_FUNCTION_STOPWORDS:
                continue
            target_id, resolved = _resolve_sql_reference(
                call_name, object_map, ("function", "procedure")
            )
            key = (owner_id, target_id, "calls")
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    CodeEdge(
                        source=owner_id,
                        target=target_id,
                        type="calls",
                        resolved=resolved,
                    )
                )

        # For triggers, extract the target table
        trigger_nodes = find_nodes_by_types(stmt_node, {"create_trigger"})
        for trigger_node in trigger_nodes:
            trigger_name = get_first_object_reference(trigger_node)
            if not trigger_name:
                continue
            trigger_name = _normalize_sql_name(trigger_name)
            trigger_id = object_map.get("trigger", {}).get(trigger_name, owner_id)

            # Find ON keyword and the table after it
            for i, child in enumerate(trigger_node.children):
                if child.type == "keyword_on" and i + 1 < len(trigger_node.children):
                    next_child = trigger_node.children[i + 1]
                    if next_child.type == "object_reference":
                        table_name = _normalize_sql_name(next_child.text.decode())
                        target_id, resolved = _resolve_sql_reference(
                            table_name, object_map, ("table", "view")
                        )
                        key = (trigger_id, target_id, "references")
                        if key not in edge_keys:
                            edge_keys.add(key)
                            edges.append(
                                CodeEdge(
                                    source=trigger_id,
                                    target=target_id,
                                    type="references",
                                    resolved=resolved,
                                )
                            )
                    break

    return nodes, edges


# =============================================================================
# SQL File Parsing (Regex Fallback)
# =============================================================================


def _parse_sql_file(
    source_text: str,
    rel_path: Path,
    file_id: str,
    config: LanguageConfig,
) -> tuple[list[CodeNode], list[CodeEdge]]:
    """Parse SQL file content for object definitions and references."""
    statements = _split_sql_statements(source_text)
    nodes: list[CodeNode] = []
    edges: list[CodeEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    object_map: dict[str, dict[str, str]] = {
        "table": {},
        "view": {},
        "function": {},
        "procedure": {},
        "trigger": {},
    }

    for statement, start_index in statements:
        definition = _extract_sql_definition(statement, start_index, source_text)
        if not definition:
            continue
        obj_id = _sql_node_id(rel_path, definition.object_type, definition.name, config.prefix)
        if definition.name in object_map[definition.object_type]:
            continue
        object_map[definition.object_type][definition.name] = obj_id
        nodes.append(
            CodeNode(
                id=obj_id,
                name=definition.name,
                type=definition.object_type,
                file=str(rel_path),
                line=definition.line,
                exported=False,
                language=config.name,
            )
        )
        edges.append(CodeEdge(source=obj_id, target=file_id, type="defined_in"))
        edges.append(CodeEdge(source=file_id, target=obj_id, type="contains"))

    for statement_index, (statement, start_index) in enumerate(statements):
        cte_map: dict[str, str] = {}
        cte_section, cte_offset = _select_cte_section(statement)
        if cte_section:
            for cte_name, line in _extract_cte_definitions(
                cte_section, start_index + cte_offset, source_text
            ):
                if cte_name in cte_map:
                    continue
                cte_id = _sql_cte_node_id(rel_path, statement_index, cte_name, config.prefix)
                cte_map[cte_name] = cte_id
                nodes.append(
                    CodeNode(
                        id=cte_id,
                        name=cte_name,
                        type="cte",
                        file=str(rel_path),
                        line=line,
                        exported=False,
                        language=config.name,
                    )
                )
                edges.append(CodeEdge(source=cte_id, target=file_id, type="defined_in"))
                edges.append(CodeEdge(source=file_id, target=cte_id, type="contains"))

        definition = _extract_sql_definition(statement, start_index, source_text)
        owner_id = file_id
        if definition:
            owner_id = object_map.get(definition.object_type, {}).get(definition.name, file_id)

        context = _sql_reference_context(
            statement, definition.object_type if definition else None
        )
        table_refs = _extract_sql_table_refs(context)
        for ref in table_refs:
            if ref in cte_map:
                target_id = cte_map[ref]
                resolved = True
            else:
                target_id, resolved = _resolve_sql_reference(
                    ref, object_map, ("table", "view")
                )
            key = (owner_id, target_id, "references")
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    CodeEdge(
                        source=owner_id,
                        target=target_id,
                        type="references",
                        resolved=resolved,
                    )
                )

        function_calls = _extract_sql_function_calls(context)
        for call in function_calls:
            target_id, resolved = _resolve_sql_reference(
                call, object_map, ("function", "procedure")
            )
            key = (owner_id, target_id, "calls")
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    CodeEdge(
                        source=owner_id,
                        target=target_id,
                        type="calls",
                        resolved=resolved,
                    )
                )

        if definition and definition.object_type == "trigger":
            trigger_target = _extract_trigger_target(statement)
            if trigger_target:
                target_id, resolved = _resolve_sql_reference(
                    trigger_target, object_map, ("table", "view")
                )
                key = (owner_id, target_id, "references")
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append(
                        CodeEdge(
                            source=owner_id,
                            target=target_id,
                            type="references",
                            resolved=resolved,
                        )
                    )

    return nodes, edges
