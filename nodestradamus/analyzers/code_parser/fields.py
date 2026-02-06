"""Schema field extraction for all languages.

Extracts fields/properties from classes, interfaces, structs, tables, and JSON.
"""

from typing import Any

from tree_sitter import Node

from nodestradamus.analyzers.code_parser.base import FieldInfo


def _extract_schema_fields(node: Node, language: str) -> list[FieldInfo]:
    """Extract fields from any schema-defining construct.

    Dispatches to language-specific extractors based on the language.

    Args:
        node: Tree-sitter AST node (class, interface, struct, table, etc.)
        language: Language identifier (python, typescript, sql, rust, json)

    Returns:
        List of FieldInfo objects extracted from the node.
    """
    extractors: dict[str, Any] = {
        "typescript": _extract_ts_properties,
        "tsx": _extract_ts_properties,
        "javascript": _extract_ts_properties,
        "python": _extract_python_fields,
        "rust": _extract_rust_fields,
        "json": _extract_json_fields,
    }
    extractor = extractors.get(language)
    return extractor(node) if extractor else []


# =============================================================================
# TypeScript/JavaScript Field Extraction
# =============================================================================


def _extract_ts_properties(node: Node) -> list[FieldInfo]:
    """Extract properties from TypeScript interface or type alias.

    Args:
        node: Tree-sitter node for interface_declaration or type_alias_declaration.

    Returns:
        List of FieldInfo for each property.
    """
    fields: list[FieldInfo] = []

    # Find the object type or interface body
    for child in node.children:
        if child.type in ("object_type", "interface_body"):
            for prop in child.children:
                if prop.type == "property_signature":
                    field = _parse_ts_property_signature(prop)
                    if field:
                        fields.append(field)

    return fields


def _parse_ts_property_signature(node: Node) -> FieldInfo | None:
    """Parse a TypeScript property signature.

    Args:
        node: Tree-sitter property_signature node.

    Returns:
        FieldInfo or None if parsing fails.
    """
    prop_name = None
    prop_type = "unknown"
    nullable = True  # Properties are nullable unless explicitly required

    for child in node.children:
        if child.type == "property_identifier":
            prop_name = child.text.decode()
        elif child.type == "type_annotation":
            # Get the type from the annotation
            for type_child in child.children:
                if type_child.type not in (":", "?"):
                    prop_type = type_child.text.decode()
        elif child.type == "?":
            nullable = True

    # Check if property name ends with ? (optional)
    if prop_name and not node.text.decode().replace(prop_name, "").startswith("?"):
        # No ? after name means required (not nullable in strict sense)
        nullable = False

    if prop_name:
        return FieldInfo(name=prop_name, type=prop_type, nullable=nullable)
    return None


# =============================================================================
# Python Field Extraction
# =============================================================================


def _extract_python_fields(node: Node) -> list[FieldInfo]:
    """Extract typed attributes from Python class body.

    Handles BaseModel, dataclass, and TypedDict patterns.

    Args:
        node: Tree-sitter node for class_definition.

    Returns:
        List of FieldInfo for each typed attribute.
    """
    fields: list[FieldInfo] = []

    # Find the class body (block node)
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    # Look for typed assignments: name: type or name: type = default
                    for expr_child in stmt.children:
                        if expr_child.type == "assignment":
                            field = _parse_python_typed_assignment(expr_child)
                            if field:
                                fields.append(field)
                        elif expr_child.type == "type":
                            # Bare type annotation: name: type
                            field = _parse_python_type_annotation(stmt)
                            if field:
                                fields.append(field)
                elif stmt.type == "typed_parameter":
                    # Handle TypedDict style
                    field = _parse_python_typed_parameter(stmt)
                    if field:
                        fields.append(field)

    return fields


def _parse_python_typed_assignment(node: Node) -> FieldInfo | None:
    """Parse a Python typed assignment (name: type = value).

    Args:
        node: Tree-sitter assignment node.

    Returns:
        FieldInfo or None.
    """
    # Looking for pattern: left: type = right
    left = node.child_by_field_name("left")
    type_node = node.child_by_field_name("type")

    if left and left.type == "identifier":
        name = left.text.decode()
        type_str = type_node.text.decode() if type_node else "unknown"
        nullable = "None" in type_str or "Optional" in type_str or "|" in type_str
        return FieldInfo(name=name, type=type_str, nullable=nullable)

    return None


def _parse_python_type_annotation(node: Node) -> FieldInfo | None:
    """Parse a Python bare type annotation (name: type).

    Args:
        node: Tree-sitter expression_statement containing type annotation.

    Returns:
        FieldInfo or None.
    """
    # Pattern: identifier: type
    for child in node.children:
        if child.type == "type":
            # Find identifier and type
            identifier = None
            type_str = "unknown"
            for c in child.children:
                if c.type == "identifier" and identifier is None:
                    identifier = c.text.decode()
                elif c.type == "type":
                    type_str = c.text.decode()
            if identifier:
                nullable = "None" in type_str or "Optional" in type_str
                return FieldInfo(name=identifier, type=type_str, nullable=nullable)
    return None


def _parse_python_typed_parameter(node: Node) -> FieldInfo | None:
    """Parse a Python typed parameter.

    Args:
        node: Tree-sitter typed_parameter node.

    Returns:
        FieldInfo or None.
    """
    name = None
    type_str = "unknown"

    for child in node.children:
        if child.type == "identifier" and name is None:
            name = child.text.decode()
        elif child.type == "type":
            type_str = child.text.decode()

    if name:
        nullable = "None" in type_str or "Optional" in type_str
        return FieldInfo(name=name, type=type_str, nullable=nullable)
    return None


# =============================================================================
# Rust Field Extraction
# =============================================================================


def _extract_rust_fields(node: Node) -> list[FieldInfo]:
    """Extract fields from Rust struct.

    Args:
        node: Tree-sitter node for struct_item.

    Returns:
        List of FieldInfo for each struct field.
    """
    fields: list[FieldInfo] = []

    # Find the field declaration list
    for child in node.children:
        if child.type == "field_declaration_list":
            for field_node in child.children:
                if field_node.type == "field_declaration":
                    field = _parse_rust_field_declaration(field_node)
                    if field:
                        fields.append(field)

    return fields


def _parse_rust_field_declaration(node: Node) -> FieldInfo | None:
    """Parse a Rust struct field declaration.

    Args:
        node: Tree-sitter field_declaration node.

    Returns:
        FieldInfo or None.
    """
    name = None
    type_str = "unknown"

    for child in node.children:
        if child.type == "field_identifier":
            name = child.text.decode()
        elif child.type == "type_identifier" or child.type.endswith("_type"):
            type_str = child.text.decode()

    if name:
        nullable = "Option" in type_str
        return FieldInfo(name=name, type=type_str, nullable=nullable)
    return None


# =============================================================================
# JSON Field Extraction
# =============================================================================


def _extract_json_fields(node: Node) -> list[FieldInfo]:
    """Infer field types from JSON object.

    Args:
        node: Tree-sitter node for a JSON object.

    Returns:
        List of FieldInfo for each key-value pair.
    """
    fields: list[FieldInfo] = []

    for child in node.children:
        if child.type == "pair":
            field = _parse_json_pair(child)
            if field:
                fields.append(field)

    return fields


def _parse_json_pair(node: Node) -> FieldInfo | None:
    """Parse a JSON key-value pair.

    Args:
        node: Tree-sitter pair node.

    Returns:
        FieldInfo or None.
    """
    key = None
    value_type = "unknown"

    for child in node.children:
        if child.type == "string":
            if key is None:
                # First string is the key
                key = child.text.decode().strip('"\'')
            else:
                # String value
                value_type = "string"
        elif child.type == ":":
            # Skip the colon separator
            continue
        elif key is not None:
            # Infer type from value
            value_type = _infer_json_type(child)

    if key:
        nullable = value_type == "null"
        return FieldInfo(name=key, type=value_type, nullable=nullable)
    return None


def _infer_json_type(node: Node) -> str:
    """Infer the type of a JSON value node.

    Args:
        node: Tree-sitter value node.

    Returns:
        Inferred type string.
    """
    type_map = {
        "string": "string",
        "number": "number",
        "true": "boolean",
        "false": "boolean",
        "null": "null",
        "object": "object",
        "array": "array",
    }
    return type_map.get(node.type, node.type)
