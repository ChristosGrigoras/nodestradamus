"""Tests for SQL dependency analysis."""

from pathlib import Path

from nodestradamus.analyzers.code_parser import SQL_CONFIG, parse_file
from nodestradamus.analyzers.deps import _detect_languages, analyze_deps

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_sql"


class TestSqlLanguageDetection:
    """Tests for SQL language detection."""

    def test_detects_sql_files(self, tmp_path: Path) -> None:
        """Should detect .sql files as SQL."""
        (tmp_path / "schema.sql").write_text("CREATE TABLE users (id int);")
        languages = _detect_languages(tmp_path)
        assert "sql" in languages

    def test_sql_config_exists(self) -> None:
        """SQL_CONFIG should be properly defined."""
        assert SQL_CONFIG.name == "sql"
        assert SQL_CONFIG.prefix == "sql"


class TestSqlParsing:
    """Tests for SQL file parsing."""

    def test_parses_sql_tables(self) -> None:
        """Should extract table definitions from SQL schema files."""
        result = parse_file(FIXTURE_DIR / "schema.sql", FIXTURE_DIR)
        table_nodes = [n.name for n in result.nodes if n.type == "table"]
        assert "public.users" in table_nodes
        assert "public.orders" in table_nodes

    def test_parses_sql_views_and_ctes(self) -> None:
        """Should extract view definitions and CTEs."""
        result = parse_file(FIXTURE_DIR / "views.sql", FIXTURE_DIR)
        view_nodes = [n.name for n in result.nodes if n.type == "view"]
        cte_nodes = [n.name for n in result.nodes if n.type == "cte"]
        assert "public.user_orders" in view_nodes
        assert "public.recent_orders" in view_nodes
        assert "recent" in cte_nodes

    def test_parses_sql_functions_procedures_triggers(self) -> None:
        """Should extract functions, procedures, and triggers."""
        result = parse_file(FIXTURE_DIR / "functions.sql", FIXTURE_DIR)
        function_nodes = [n.name for n in result.nodes if n.type == "function"]
        procedure_nodes = [n.name for n in result.nodes if n.type == "procedure"]
        trigger_nodes = [n.name for n in result.nodes if n.type == "trigger"]
        assert "public.get_user_orders" in function_nodes
        assert "public.log_access" in function_nodes
        assert "public.cleanup_orders" in procedure_nodes
        assert "audit_orders" in trigger_nodes

    def test_extracts_sql_references(self) -> None:
        """Should extract table and CTE references from views."""
        result = parse_file(FIXTURE_DIR / "views.sql", FIXTURE_DIR)
        reference_targets = {e.target for e in result.edges if e.type == "references"}
        assert "public.users" in reference_targets
        assert "public.orders" in reference_targets
        assert any(target.endswith("::recent") for target in reference_targets)

    def test_extracts_sql_function_calls(self) -> None:
        """Should extract function call dependencies."""
        result = parse_file(FIXTURE_DIR / "functions.sql", FIXTURE_DIR)
        call_targets = {e.target for e in result.edges if e.type == "calls"}
        assert any("public.log_access" in target for target in call_targets)


class TestSqlAnalyzeDeps:
    """Integration tests for SQL dependency analysis."""

    def test_analyzes_sql_repository(self) -> None:
        """Should build dependency graph from SQL files."""
        graph = analyze_deps(FIXTURE_DIR, languages=["sql"])
        node_names = {data.get("name") for _, data in graph.nodes(data=True)}
        assert "public.users" in node_names
        assert "public.orders" in node_names
