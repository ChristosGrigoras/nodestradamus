"""String filtering utilities for noise detection.

Identifies and filters common noise patterns in string analysis:
- Type annotations (string, number, boolean, etc.)
- Import paths (./relative, @scoped/package)
- CSS classes (Tailwind utilities, responsive prefixes)
"""

# Common type annotation strings that appear frequently but aren't meaningful
TYPE_ANNOTATIONS = frozenset(
    {
        "string",
        "number",
        "boolean",
        "object",
        "any",
        "void",
        "null",
        "undefined",
        "String",
        "Number",
        "Boolean",
        "Object",
        "Array",
        "Function",
        "true",
        "false",
        "int",
        "float",
        "str",
        "bool",
        "dict",
        "list",
    }
)

# Common import package names
COMMON_IMPORTS = frozenset(
    {
        "react",
        "express",
        "next",
        "vue",
        "angular",
        "lodash",
        "axios",
        "ioredis",
        "redis",
        "mongoose",
        "sequelize",
        "prisma",
        "typeorm",
        "lucide-react",
        "tailwindcss",
        "styled-components",
        "@emotion",
        "use client",
        "use server",
        "use strict",
    }
)

# Tailwind CSS utility patterns
_TAILWIND_PATTERNS = frozenset(
    {
        "flex",
        "grid",
        "block",
        "inline",
        "hidden",
        "items-",
        "justify-",
        "content-",
        "self-",
        "px-",
        "py-",
        "mx-",
        "my-",
        "p-",
        "m-",
        "w-",
        "h-",
        "min-",
        "max-",
        "text-",
        "font-",
        "leading-",
        "tracking-",
        "bg-",
        "border-",
        "rounded-",
        "shadow-",
        "space-",
        "gap-",
        "divide-",
        "animate-",
        "transition-",
        "duration-",
        "ease-",
        "hover:",
        "focus:",
        "active:",
        "disabled:",
        "sm:",
        "md:",
        "lg:",
        "xl:",
        "2xl:",
    }
)


def is_css_class(s: str) -> bool:
    """Check if string looks like a CSS class string (especially Tailwind).

    Args:
        s: String to check.

    Returns:
        True if the string appears to be CSS classes.
    """
    s_lower = s.lower()

    # Single Tailwind class (e.g., "animate-spin", "font-medium")
    if "-" in s and not s.startswith(("/", "@", ".")):
        if any(s_lower.startswith(p) or p in s_lower for p in _TAILWIND_PATTERNS):
            return True

    # Multiple classes (e.g., "flex items-center gap-2")
    if " " in s:
        parts = s_lower.split()
        tailwind_matches = sum(
            1 for part in parts if any(part.startswith(p) or p in part for p in _TAILWIND_PATTERNS)
        )
        # If >50% of parts look like Tailwind, it's CSS
        if tailwind_matches > len(parts) * 0.5:
            return True

    return False


def is_import_path(s: str) -> bool:
    """Check if string looks like an import path.

    Args:
        s: String to check.

    Returns:
        True if the string appears to be an import path.
    """
    # Relative imports
    if s.startswith("./") or s.startswith("../"):
        return True
    # Scoped packages
    if s.startswith("@") and "/" in s:
        return True
    # Known package names
    if s.lower() in COMMON_IMPORTS:
        return True
    return False
