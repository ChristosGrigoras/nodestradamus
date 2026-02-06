#!/usr/bin/env python3
"""
Detect potential conflicts between Cursor AI rules.

Thin wrapper around nodestradamus.mcp.tools.handlers.rules_validation.

Analyzes rules for contradictory directives and reports potential conflicts.

Usage:
    python detect_rule_conflicts.py
    python detect_rule_conflicts.py .cursor/rules/
    python detect_rule_conflicts.py --json
"""

import json
import sys
from pathlib import Path
from typing import Any

# Import the core conflict detection logic from the main package
from nodestradamus.mcp.tools.handlers.rules_validation import (
    _detect_all_conflicts,
)


def print_report(result: dict[str, Any]) -> None:
    """Print a human-readable conflict report."""
    conflicts = result["conflicts"]
    rules_count = result["rules_analyzed"]

    print("\n" + "=" * 60)
    print("RULE CONFLICT DETECTION REPORT")
    print("=" * 60)

    print(f"\nRules analyzed: {rules_count}")
    print(f"Potential conflicts found: {len(conflicts)}")

    if not conflicts:
        print("\n✓ No conflicts detected!")
        print("=" * 60)
        return

    # Group by severity
    errors = [c for c in conflicts if c["severity"] == "error"]
    warnings = [c for c in conflicts if c["severity"] == "warning"]

    if errors:
        print("\n" + "-" * 60)
        print("ERRORS (Direct Conflicts)")
        print("-" * 60)
        for conflict in errors:
            print(f"\n✗ {conflict['description']}")
            print(f"  Message: {conflict['message']}")
            if "rules_pattern1" in conflict:
                print(f"  Pattern 1 in: {', '.join(conflict['rules_pattern1'])}")
                print(f"  Pattern 2 in: {', '.join(conflict['rules_pattern2'])}")

    if warnings:
        print("\n" + "-" * 60)
        print("WARNINGS (Potential Conflicts)")
        print("-" * 60)
        for conflict in warnings:
            print(f"\n⚠ {conflict['description']}")
            print(f"  {conflict['rule1']['file']}: {', '.join(conflict['rule1']['keywords'])}")
            print(f"  {conflict['rule2']['file']}: {', '.join(conflict['rule2']['keywords'])}")

    print("\n" + "=" * 60)
    print("\nRecommendations:")
    print("1. Review flagged rules for intentional vs accidental differences")
    print("2. Use 200-project.mdc for project-specific overrides")
    print("3. Ensure higher-priority rules (80-100) override lower ones")
    print("=" * 60)


def main() -> None:
    # Parse arguments
    args = sys.argv[1:]
    json_output = "--json" in args
    args = [a for a in args if not a.startswith("--")]

    # Determine rules directory
    if args:
        rules_dir = Path(args[0])
    else:
        rules_dir = Path(".cursor/rules")

    if not rules_dir.is_dir():
        print(f"Error: {rules_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Detect conflicts using the core function from the main package
    result = _detect_all_conflicts(rules_dir)

    # Output results
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)

    # Exit with error if there are error-level conflicts
    if result["summary"]["errors"] > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
