#!/usr/bin/env python3
"""
Validate Cursor AI rules in .cursor/rules/ directory.

Thin wrapper around nodestradamus.mcp.tools.handlers.rules_validation.

Checks:
- YAML frontmatter syntax
- Unique rule numbering
- Referenced file existence
- Token budget compliance
- Rule format consistency

Usage:
    python validate_rules.py
    python validate_rules.py .cursor/rules/
    python validate_rules.py --strict  # Fail on warnings
"""

import json
import sys
from pathlib import Path
from typing import Any

# Import the core validation logic from the main package
from nodestradamus.mcp.tools.handlers.rules_validation import (
    _validate_rules_directory as validate_rules_directory,
)

# Keep _validate_rules_directory for internal use
_validate_rules_directory = validate_rules_directory


def print_results(results: dict[str, Any], verbose: bool = True) -> None:
    """Print validation results in a readable format."""
    print("\n" + "=" * 60)
    print("RULE VALIDATION REPORT")
    print("=" * 60)

    summary = results["summary"]
    print(f"\nTotal files: {summary['total']}")
    print(f"  ✓ Valid: {summary['valid']}")
    print(f"  ⚠ Warnings: {summary['with_warnings']}")
    print(f"  ✗ Errors: {summary['with_errors']}")

    if verbose:
        print("\n" + "-" * 60)
        print("FILE DETAILS")
        print("-" * 60)

        for file_result in results["files"]:
            status = "✓" if not file_result["errors"] and not file_result["warnings"] else ""
            if file_result["errors"]:
                status = "✗"
            elif file_result["warnings"]:
                status = "⚠"

            print(f"\n{status} {file_result['file']}")

            if file_result["errors"]:
                for error in file_result["errors"]:
                    print(f"    ERROR: {error}")

            if file_result["warnings"]:
                for warning in file_result["warnings"]:
                    print(f"    WARNING: {warning}")

    print("\n" + "=" * 60)


def main() -> None:
    # Parse arguments
    args = sys.argv[1:]
    strict = "--strict" in args
    json_output = "--json" in args
    args = [a for a in args if not a.startswith("--")]

    # Determine rules directory
    if args:
        rules_dir = Path(args[0])
    else:
        # Default to .cursor/rules/ in current directory
        rules_dir = Path(".cursor/rules")

    if not rules_dir.is_dir():
        print(f"Error: {rules_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Run validation using the core function from the main package
    results = _validate_rules_directory(rules_dir)

    # Output results
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)

    # Exit code
    if results["errors"] > 0:
        sys.exit(1)
    if strict and results["warnings"] > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
