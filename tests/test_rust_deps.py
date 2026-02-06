"""Tests for Rust dependency analysis."""

from pathlib import Path

from nodestradamus.analyzers.code_parser import RUST_CONFIG, parse_file
from nodestradamus.analyzers.deps import _detect_languages, analyze_deps


class TestRustLanguageDetection:
    """Tests for Rust language detection."""

    def test_detects_rust_files(self, tmp_path: Path) -> None:
        """Should detect .rs files as Rust."""
        (tmp_path / "main.rs").write_text("fn main() {}")
        langs = _detect_languages(tmp_path)
        assert "rust" in langs

    def test_rust_config_exists(self) -> None:
        """RUST_CONFIG should be properly defined."""
        assert RUST_CONFIG.name == "rust"
        assert RUST_CONFIG.prefix == "rs"
        assert "function_item" in RUST_CONFIG.function_types
        assert "struct_item" in RUST_CONFIG.class_types


class TestRustParsing:
    """Tests for Rust file parsing."""

    def test_parses_rust_functions(self, tmp_path: Path) -> None:
        """Should extract function definitions."""
        rust_code = '''
fn main() {
    println!("Hello");
}

pub fn helper(x: i32) -> i32 {
    x + 1
}

fn internal() {}
'''
        rust_file = tmp_path / "main.rs"
        rust_file.write_text(rust_code)

        result = parse_file(rust_file, tmp_path)

        assert len(result.errors) == 0
        func_nodes = [n for n in result.nodes if n.type == "function"]
        func_names = {n.name for n in func_nodes}
        assert "main" in func_names
        assert "helper" in func_names
        assert "internal" in func_names

    def test_parses_rust_structs(self, tmp_path: Path) -> None:
        """Should extract struct definitions."""
        rust_code = '''
pub struct User {
    name: String,
    age: u32,
}

struct Config {
    debug: bool,
}
'''
        rust_file = tmp_path / "types.rs"
        rust_file.write_text(rust_code)

        result = parse_file(rust_file, tmp_path)

        class_nodes = [n for n in result.nodes if n.type == "class"]
        names = {n.name for n in class_nodes}
        assert "User" in names
        assert "Config" in names

    def test_parses_rust_enums(self, tmp_path: Path) -> None:
        """Should extract enum definitions."""
        rust_code = '''
pub enum Status {
    Active,
    Inactive,
    Pending,
}
'''
        rust_file = tmp_path / "status.rs"
        rust_file.write_text(rust_code)

        result = parse_file(rust_file, tmp_path)

        class_nodes = [n for n in result.nodes if n.type == "class"]
        names = {n.name for n in class_nodes}
        assert "Status" in names

    def test_parses_rust_traits(self, tmp_path: Path) -> None:
        """Should extract trait definitions."""
        rust_code = '''
pub trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
'''
        rust_file = tmp_path / "traits.rs"
        rust_file.write_text(rust_code)

        result = parse_file(rust_file, tmp_path)

        class_nodes = [n for n in result.nodes if n.type == "class"]
        names = {n.name for n in class_nodes}
        assert "Drawable" in names

    def test_parses_rust_use_statements(self, tmp_path: Path) -> None:
        """Should extract use declarations as imports."""
        rust_code = '''
use std::collections::HashMap;
use crate::utils::helper;

fn main() {}
'''
        rust_file = tmp_path / "main.rs"
        rust_file.write_text(rust_code)

        result = parse_file(rust_file, tmp_path)

        import_edges = [e for e in result.edges if e.type == "imports"]
        sources = {e.target for e in import_edges}
        assert "std::collections::HashMap" in sources


class TestRustAnalyzeDeps:
    """Integration tests for Rust dependency analysis."""

    def test_analyzes_rust_repository(self, tmp_path: Path) -> None:
        """Should build dependency graph from Rust files."""
        src = tmp_path / "src"
        src.mkdir()

        (src / "main.rs").write_text('''
mod utils;

fn main() {
    utils::greet();
}
''')
        (src / "utils.rs").write_text('''
pub fn greet() {
    println!("Hello!");
}

pub fn helper() -> i32 {
    42
}
''')

        graph = analyze_deps(tmp_path, languages=["rust"])

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Check we have the expected functions
        node_names = {data.get("name") for _, data in graph.nodes(data=True)}
        assert "main" in node_names
        assert "greet" in node_names
        assert "helper" in node_names
