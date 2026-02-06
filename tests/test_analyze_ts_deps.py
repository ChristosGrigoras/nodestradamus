"""Tests for analyze_ts_deps.py."""

from pathlib import Path

import pytest

from scripts.analyze_ts_deps import (
    analyze_directory,
    analyze_file,
    extract_definitions,
    extract_imports,
)


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_es_module_import(self):
        """Test extracting ES module imports."""
        source = "import { foo } from './module';"
        imports = extract_imports(source)
        assert "./module" in imports

    def test_default_import(self):
        """Test extracting default imports."""
        source = "import React from 'react';"
        imports = extract_imports(source)
        assert "react" in imports

    def test_namespace_import(self):
        """Test extracting namespace imports."""
        source = "import * as utils from './utils';"
        imports = extract_imports(source)
        assert "./utils" in imports

    def test_side_effect_import(self):
        """Test extracting side-effect imports."""
        source = "import './styles.css';"
        imports = extract_imports(source)
        assert "./styles.css" in imports

    def test_commonjs_require(self):
        """Test extracting CommonJS require."""
        source = "const fs = require('fs');"
        imports = extract_imports(source)
        assert "fs" in imports

    def test_dynamic_import(self):
        """Test extracting dynamic imports."""
        source = "const module = await import('./dynamic');"
        imports = extract_imports(source)
        assert "./dynamic" in imports

    def test_multiple_imports(self):
        """Test extracting multiple imports."""
        source = """
import { a } from './a';
import b from './b';
const c = require('./c');
"""
        imports = extract_imports(source)
        assert "./a" in imports
        assert "./b" in imports
        assert "./c" in imports


class TestExtractDefinitions:
    """Tests for extract_definitions function."""

    def test_function_definition(self):
        """Test extracting function definitions."""
        source = """
function hello() {
    console.log('hello');
}
"""
        filepath = Path("test.ts")
        defs = extract_definitions(source, filepath)

        func_names = [d["name"] for d in defs if d["type"] == "function"]
        assert any("hello" in name for name in func_names)

    def test_async_function(self):
        """Test extracting async function definitions."""
        source = """
async function fetchData() {
    return await fetch('/api');
}
"""
        filepath = Path("test.ts")
        defs = extract_definitions(source, filepath)

        func_names = [d["name"] for d in defs if d["type"] == "function"]
        assert any("fetchData" in name for name in func_names)

    def test_arrow_function(self):
        """Test extracting arrow function definitions."""
        source = """
const add = (a, b) => a + b;
const multiply = (a, b) => {
    return a * b;
};
"""
        filepath = Path("test.ts")
        defs = extract_definitions(source, filepath)

        func_names = [d["name"] for d in defs if d["type"] == "function"]
        assert any("add" in name for name in func_names)
        assert any("multiply" in name for name in func_names)

    def test_class_definition(self):
        """Test extracting class definitions."""
        source = """
class UserService {
    getUser(id) {
        return this.users[id];
    }
}
"""
        filepath = Path("test.ts")
        defs = extract_definitions(source, filepath)

        class_defs = [d for d in defs if d["type"] == "class"]
        assert len(class_defs) >= 1
        assert any("UserService" in d["name"] for d in class_defs)

    def test_class_with_extends(self):
        """Test extracting class inheritance."""
        source = """
class AdminService extends UserService {
    deleteUser(id) {}
}
"""
        filepath = Path("test.ts")
        defs = extract_definitions(source, filepath)

        class_defs = [d for d in defs if d["type"] == "class"]
        admin_class = next(d for d in class_defs if "AdminService" in d["name"])
        assert admin_class["extends"] == "UserService"

    def test_exported_definitions(self):
        """Test extracting exported definitions."""
        source = """
export function publicFunc() {}
export class PublicClass {}
export const publicArrow = () => {};
"""
        filepath = Path("test.ts")
        defs = extract_definitions(source, filepath)

        names = [d["name"] for d in defs]
        assert any("publicFunc" in name for name in names)
        assert any("PublicClass" in name for name in names)
        assert any("publicArrow" in name for name in names)


class TestAnalyzeFile:
    """Tests for analyze_file function."""

    def test_analyze_simple_file(self, sample_typescript_file: Path):
        """Test analyzing a simple TypeScript file."""
        result = analyze_file(sample_typescript_file, sample_typescript_file.parent)

        assert "definitions" in result
        assert "edges" in result
        assert "error" not in result

    def test_analyze_file_with_imports(self, temp_dir: Path):
        """Test analyzing file with imports."""
        ts_file = temp_dir / "test.ts"
        ts_file.write_text("""
import { foo } from './utils';
import bar from 'external-pkg';

export function test() {
    foo();
}
""")

        result = analyze_file(ts_file, temp_dir)

        assert len(result["imports"]) >= 2
        assert any("./utils" in imp for imp in result["imports"])
        assert any("external-pkg" in imp for imp in result["imports"])


class TestAnalyzeDirectory:
    """Tests for analyze_directory function."""

    def test_analyze_sample_directory(self):
        """Test analyzing the sample TypeScript fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "sample_typescript"
        if not fixtures_dir.exists():
            pytest.skip("Sample fixtures not found")

        result = analyze_directory(fixtures_dir)

        assert "nodes" in result
        assert "edges" in result
        assert "files" in result
        assert "metadata" in result
        assert len(result["files"]) > 0

    def test_analyze_empty_directory(self, temp_dir: Path):
        """Test analyzing an empty directory."""
        result = analyze_directory(temp_dir)

        assert result["nodes"] == []
        assert result["files"] == []

    def test_skips_node_modules(self, temp_dir: Path):
        """Test that node_modules is skipped."""
        node_modules = temp_dir / "node_modules"
        node_modules.mkdir()
        (node_modules / "package" / "index.js").parent.mkdir(parents=True)
        (node_modules / "package" / "index.js").write_text("export function foo() {}")

        result = analyze_directory(temp_dir)

        assert all("node_modules" not in f for f in result["files"])

    def test_handles_multiple_extensions(self, temp_dir: Path):
        """Test handling various file extensions."""
        (temp_dir / "file.ts").write_text("export const a = 1;")
        (temp_dir / "file.tsx").write_text("export const b = 2;")
        (temp_dir / "file.js").write_text("export const c = 3;")
        (temp_dir / "file.jsx").write_text("export const d = 4;")

        result = analyze_directory(temp_dir)

        assert len(result["files"]) == 4

    def test_metadata_includes_stats(self, temp_dir: Path):
        """Test that metadata includes file count."""
        (temp_dir / "test.ts").write_text("export const a = 1;")

        result = analyze_directory(temp_dir)

        assert result["metadata"]["file_count"] == 1
        assert result["metadata"]["analyzer"] == "typescript"
