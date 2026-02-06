"""Tests for TypeScript path alias resolution."""

import json
from pathlib import Path

import pytest

from scripts.analyze_ts_treesitter import (
    PathAliasResolver,
    analyze_directory,
    resolve_import_path,
)


@pytest.fixture
def sample_ts_aliases_dir() -> Path:
    """Return path to sample TypeScript fixtures with path aliases."""
    return Path(__file__).parent / "fixtures" / "sample_ts_aliases"


@pytest.fixture
def alias_project(temp_dir: Path) -> Path:
    """Create a minimal project with path aliases."""
    # Create tsconfig.json
    (temp_dir / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                        "@/*": ["./src/*"],
                        "@lib/*": ["./src/lib/*"],
                        "@utils": ["./src/utils/index"],
                    },
                }
            }
        )
    )

    # Create source files
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "lib").mkdir()
    (temp_dir / "src" / "utils").mkdir()

    (temp_dir / "src" / "lib" / "helpers.ts").write_text(
        "export function helper() {}"
    )
    (temp_dir / "src" / "utils" / "index.ts").write_text(
        "export function util() {}"
    )
    (temp_dir / "src" / "app.ts").write_text(
        """
import { helper } from "@/lib/helpers";
import { util } from "@utils";
import { another } from "@lib/helpers";

export function main() {
    helper();
    util();
}
"""
    )

    return temp_dir


class TestPathAliasResolver:
    """Tests for PathAliasResolver class."""

    def test_loads_tsconfig(self, alias_project: Path) -> None:
        """Should load tsconfig.json and parse paths."""
        resolver = PathAliasResolver(alias_project)

        assert resolver.base_url is not None
        assert "@/*" in resolver.paths
        assert "@lib/*" in resolver.paths
        assert "@utils" in resolver.paths

    def test_resolves_wildcard_alias(self, alias_project: Path) -> None:
        """Should resolve @/* style aliases."""
        resolver = PathAliasResolver(alias_project)

        resolved = resolver.resolve("@/lib/helpers")
        assert resolved is not None
        assert "lib/helpers" in resolved
        assert resolved.endswith(".ts")

    def test_resolves_nested_alias(self, alias_project: Path) -> None:
        """Should resolve @lib/* style aliases."""
        resolver = PathAliasResolver(alias_project)

        resolved = resolver.resolve("@lib/helpers")
        assert resolved is not None
        assert "lib/helpers" in resolved

    def test_resolves_exact_alias(self, alias_project: Path) -> None:
        """Should resolve exact match aliases like @utils."""
        resolver = PathAliasResolver(alias_project)

        resolved = resolver.resolve("@utils")
        assert resolved is not None
        assert "utils" in resolved

    def test_returns_none_for_external_packages(self, alias_project: Path) -> None:
        """Should return None for external packages."""
        resolver = PathAliasResolver(alias_project)

        assert resolver.resolve("react") is None
        assert resolver.resolve("lodash") is None
        assert resolver.resolve("@types/node") is None

    def test_returns_none_for_relative_imports(self, alias_project: Path) -> None:
        """Should return None for relative imports (handled elsewhere)."""
        resolver = PathAliasResolver(alias_project)

        assert resolver.resolve("./utils") is None
        assert resolver.resolve("../lib/helpers") is None

    def test_handles_missing_tsconfig(self, temp_dir: Path) -> None:
        """Should handle projects without tsconfig.json."""
        resolver = PathAliasResolver(temp_dir)

        assert resolver.paths == {}
        assert resolver.resolve("@/lib/utils") is None

    def test_handles_jsconfig(self, temp_dir: Path) -> None:
        """Should fall back to jsconfig.json for JS projects."""
        (temp_dir / "jsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "baseUrl": ".",
                        "paths": {"@/*": ["./src/*"]},
                    }
                }
            )
        )
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "utils.js").write_text("export function util() {}")

        resolver = PathAliasResolver(temp_dir)

        assert "@/*" in resolver.paths
        resolved = resolver.resolve("@/utils")
        assert resolved is not None


class TestResolveImportPath:
    """Tests for resolve_import_path function with alias support."""

    def test_resolves_relative_import(self, alias_project: Path) -> None:
        """Should still resolve relative imports."""
        filepath = alias_project / "src" / "app.ts"
        resolver = PathAliasResolver(alias_project)

        resolved = resolve_import_path(
            "./lib/helpers", filepath, alias_project, resolver
        )
        assert resolved is not None
        assert "lib/helpers" in resolved

    def test_resolves_alias_import(self, alias_project: Path) -> None:
        """Should resolve path alias imports."""
        filepath = alias_project / "src" / "app.ts"
        resolver = PathAliasResolver(alias_project)

        resolved = resolve_import_path(
            "@/lib/helpers", filepath, alias_project, resolver
        )
        assert resolved is not None
        assert "lib/helpers" in resolved
        assert resolved.endswith(".ts")

    def test_returns_none_for_external(self, alias_project: Path) -> None:
        """Should return None for external packages."""
        filepath = alias_project / "src" / "app.ts"
        resolver = PathAliasResolver(alias_project)

        resolved = resolve_import_path("react", filepath, alias_project, resolver)
        assert resolved is None


class TestMonorepoSupport:
    """Tests for monorepo tsconfig detection."""

    def test_finds_nested_tsconfig(self, temp_dir: Path) -> None:
        """Should find tsconfig in nested directory."""
        # Create monorepo structure
        frontend = temp_dir / "packages" / "frontend"
        frontend.mkdir(parents=True)

        (frontend / "tsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "baseUrl": ".",
                        "paths": {"@/*": ["./src/*"]},
                    }
                }
            )
        )

        (frontend / "src").mkdir()
        (frontend / "src" / "utils.ts").write_text("export function util() {}")
        (frontend / "src" / "app.ts").write_text(
            'import { util } from "@/utils";\nexport function main() { util(); }'
        )

        # Analyze from repo root
        result = analyze_directory(temp_dir)

        # Should resolve the @/ import
        app_edges = [e for e in result["edges"] if "app.ts" in e.get("from", "")]
        resolved = [e for e in app_edges if e.get("resolved")]

        assert len(resolved) > 0, f"Expected resolved imports in app.ts. Edges: {app_edges}"

    def test_multiple_tsconfigs(self, temp_dir: Path) -> None:
        """Should handle multiple packages with different tsconfigs."""
        # Package A
        pkg_a = temp_dir / "packages" / "pkg-a"
        pkg_a.mkdir(parents=True)
        (pkg_a / "tsconfig.json").write_text(
            json.dumps({"compilerOptions": {"paths": {"@a/*": ["./src/*"]}}})
        )
        (pkg_a / "src").mkdir()
        (pkg_a / "src" / "index.ts").write_text("export const a = 1;")

        # Package B
        pkg_b = temp_dir / "packages" / "pkg-b"
        pkg_b.mkdir(parents=True)
        (pkg_b / "tsconfig.json").write_text(
            json.dumps({"compilerOptions": {"paths": {"@b/*": ["./src/*"]}}})
        )
        (pkg_b / "src").mkdir()
        (pkg_b / "src" / "index.ts").write_text("export const b = 2;")
        (pkg_b / "src" / "user.ts").write_text(
            'import { b } from "@b/index";\nexport const user = b;'
        )

        result = analyze_directory(temp_dir)

        # Should resolve @b/ imports in package B
        user_edges = [e for e in result["edges"] if "user.ts" in e.get("from", "")]
        resolved = [e for e in user_edges if e.get("resolved")]

        assert len(resolved) > 0, f"Expected resolved @b/ import. Edges: {user_edges}"


class TestAnalyzeDirectoryWithAliases:
    """Integration tests for analyze_directory with path aliases."""

    def test_resolves_alias_edges(self, sample_ts_aliases_dir: Path) -> None:
        """Should create resolved edges for alias imports."""
        if not sample_ts_aliases_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(sample_ts_aliases_dir)

        # Find edges from Button.tsx
        button_edges = [
            e for e in result["edges"] if "Button" in e.get("from", "")
        ]

        # Should have a resolved edge to lib/utils.ts
        resolved_utils = [
            e for e in button_edges if e.get("resolved") and "utils" in e.get("to", "")
        ]
        assert len(resolved_utils) > 0, (
            f"Expected resolved edge to utils.ts from Button.tsx. "
            f"Button edges: {button_edges}"
        )

    def test_all_internal_imports_resolved(self, sample_ts_aliases_dir: Path) -> None:
        """All @/ imports should be resolved to actual files."""
        if not sample_ts_aliases_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(sample_ts_aliases_dir)

        # Count resolved vs unresolved internal edges
        internal_edges = [e for e in result["edges"] if e["type"] == "imports"]
        [e for e in internal_edges if e.get("resolved")]
        unresolved = [e for e in internal_edges if not e.get("resolved")]

        # Filter out truly external packages
        unresolved_internal = [
            e
            for e in unresolved
            if "@" in e.get("to", "") and not e.get("to", "").startswith("@types")
        ]

        assert len(unresolved_internal) == 0, (
            f"Expected all @/ imports to be resolved. "
            f"Unresolved: {unresolved_internal}"
        )

    def test_utils_has_dependents(self, sample_ts_aliases_dir: Path) -> None:
        """utils.ts should have multiple files depending on it."""
        if not sample_ts_aliases_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(sample_ts_aliases_dir)

        # Find edges pointing to utils.ts
        utils_dependents = [
            e
            for e in result["edges"]
            if "utils" in e.get("to", "") and e.get("resolved")
        ]

        # Button.tsx, Card.tsx, and app.tsx all import utils
        assert len(utils_dependents) >= 3, (
            f"Expected at least 3 files importing utils.ts. "
            f"Found: {utils_dependents}"
        )

    def test_metadata_shows_resolved_config(self, sample_ts_aliases_dir: Path) -> None:
        """Metadata should indicate tsconfig was found."""
        if not sample_ts_aliases_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(sample_ts_aliases_dir)

        assert result["metadata"]["analyzer"] == "typescript_treesitter"
        assert result["metadata"]["file_count"] >= 4  # utils, Button, Card, app
