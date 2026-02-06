"""Tests for embeddings analyzer."""

import tempfile
from pathlib import Path

import pytest

from nodestradamus.analyzers.embeddings import (
    _extract_js_chunks,
    _extract_python_chunks,
    compute_embeddings,
    detect_duplicates,
    find_similar_code,
    semantic_search,
)


class TestPythonChunkExtraction:
    """Tests for Python code chunk extraction."""

    def test_extracts_function(self) -> None:
        """Test extraction of a simple function."""
        code = '''def hello():
    """Say hello."""
    print("Hello, World!")

x = 1
'''
        chunks = _extract_python_chunks(code, "test.py")
        assert len(chunks) == 1
        assert chunks[0]["name"] == "hello"
        assert chunks[0]["type"] == "function"
        assert "Hello, World!" in chunks[0]["content"]

    def test_extracts_async_function(self) -> None:
        """Test extraction of async function."""
        code = '''async def fetch_data(url):
    response = await client.get(url)
    return response.json()
'''
        chunks = _extract_python_chunks(code, "test.py")
        assert len(chunks) == 1
        assert chunks[0]["name"] == "fetch_data"
        assert chunks[0]["type"] == "function"

    def test_extracts_class(self) -> None:
        """Test extraction of a class."""
        code = '''class MyClass:
    """A test class."""

    def method(self):
        pass
'''
        chunks = _extract_python_chunks(code, "test.py")
        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if c["name"] == "MyClass"), None)
        assert class_chunk is not None
        assert class_chunk["type"] == "class"

    def test_extracts_multiple_functions(self) -> None:
        """Test extraction of multiple functions."""
        code = '''def foo():
    return 1

def bar():
    return 2

def baz():
    return 3
'''
        chunks = _extract_python_chunks(code, "test.py")
        names = {c["name"] for c in chunks}
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names

    def test_empty_file(self) -> None:
        """Test handling of empty file."""
        chunks = _extract_python_chunks("", "test.py")
        assert chunks == []

    def test_no_functions(self) -> None:
        """Test file with no functions or classes."""
        code = '''x = 1
y = 2
print(x + y)
'''
        chunks = _extract_python_chunks(code, "test.py")
        assert chunks == []


class TestJSChunkExtraction:
    """Tests for JavaScript/TypeScript code chunk extraction."""

    def test_extracts_function(self) -> None:
        """Test extraction of a function declaration."""
        code = '''function greet(name) {
    console.log("Hello, " + name);
}
'''
        chunks = _extract_js_chunks(code, "test.js")
        assert len(chunks) == 1
        assert chunks[0]["name"] == "greet"
        assert chunks[0]["type"] == "function"

    def test_extracts_async_function(self) -> None:
        """Test extraction of async function."""
        code = '''async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}
'''
        chunks = _extract_js_chunks(code, "test.js")
        assert len(chunks) == 1
        assert chunks[0]["name"] == "fetchData"

    def test_extracts_class(self) -> None:
        """Test extraction of a class."""
        code = '''class User {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return "Hello, " + this.name;
    }
}
'''
        chunks = _extract_js_chunks(code, "test.js")
        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if c["name"] == "User"), None)
        assert class_chunk is not None
        assert class_chunk["type"] == "class"

    def test_extracts_arrow_function(self) -> None:
        """Test extraction of const arrow function."""
        code = '''const add = (a, b) => {
    return a + b;
};
'''
        chunks = _extract_js_chunks(code, "test.js")
        assert len(chunks) == 1
        assert chunks[0]["name"] == "add"

    def test_export_function(self) -> None:
        """Test extraction of exported function."""
        code = '''export function helper() {
    return true;
}
'''
        chunks = _extract_js_chunks(code, "test.js")
        assert len(chunks) == 1
        assert chunks[0]["name"] == "helper"


class TestComputeEmbeddings:
    """Tests for compute_embeddings function."""

    def test_empty_directory(self) -> None:
        """Test handling of empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compute_embeddings(tmpdir)
            assert result["metadata"]["chunks_extracted"] == 0
            assert result["chunks"] == []

    def test_computes_embeddings_for_python(self) -> None:
        """Test computing embeddings for Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text('''def greet(name):
    """Greet someone."""
    return f"Hello, {name}!"

def farewell(name):
    """Say goodbye."""
    return f"Goodbye, {name}!"
''')

            result = compute_embeddings(tmpdir)

            assert result["metadata"]["files_processed"] >= 1
            assert result["metadata"]["chunks_extracted"] >= 2
            assert len(result["embeddings"]) == len(result["chunks"])

            # Check chunk structure
            for chunk in result["chunks"]:
                assert "id" in chunk
                assert "file" in chunk
                assert "type" in chunk

    def test_caches_embeddings(self) -> None:
        """Test that embeddings are cached (SQLite + FAISS or NPZ fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("def foo(): pass")

            compute_embeddings(tmpdir)

            # Check that some cache exists (FAISS or NPZ depending on availability)
            cache_dir = Path(tmpdir) / ".nodestradamus"
            has_faiss = (cache_dir / "embeddings.faiss").exists()
            has_npz = (cache_dir / "embeddings.npz").exists()
            has_sqlite = (cache_dir / "nodestradamus.db").exists()
            assert has_sqlite, "SQLite database should exist"
            assert has_faiss or has_npz, "FAISS or NPZ cache should exist"

    def test_skips_node_modules(self) -> None:
        """Test that node_modules is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in node_modules
            nm_dir = Path(tmpdir) / "node_modules" / "pkg"
            nm_dir.mkdir(parents=True)
            (nm_dir / "index.js").write_text("function test() {}")

            # Create a file outside node_modules
            (Path(tmpdir) / "app.js").write_text("function main() {}")

            result = compute_embeddings(tmpdir)

            # Should only find app.js, not node_modules
            files = {c["file"] for c in result["chunks"]}
            assert not any("node_modules" in f for f in files)

    def test_invalid_path(self) -> None:
        """Test handling of invalid path."""
        with pytest.raises(ValueError, match="does not exist"):
            compute_embeddings("/nonexistent/path/to/repo")

    def test_computes_embeddings_for_rust_and_bash(self) -> None:
        """Test that Rust and Bash files produce AST-based chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "main.rs").write_text("""
fn main() {
    println!("hello");
}
fn other() -> i32 { 42 }
""")
            (root / "script.sh").write_text("""
main() {
    echo "world"
}
""")
            result = compute_embeddings(tmpdir)
            chunks = result.get("chunks", [])
            ids = [c["id"] for c in chunks]
            rust_chunks = [c for c in chunks if c["id"].startswith("rs:")]
            bash_chunks = [c for c in chunks if c["id"].startswith("sh:")]
            assert (
                len(rust_chunks) + len(bash_chunks) >= 1
            ), f"Expected some rs: or sh: chunks, got ids: {ids}"
            for c in rust_chunks + bash_chunks:
                assert "name" in c
                assert "line_start" in c
                assert "line_end" in c


class TestFindSimilarCode:
    """Tests for find_similar_code function."""

    def test_find_similar_by_query(self) -> None:
        """Test finding similar code by text query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with related content
            Path(tmpdir, "auth.py").write_text('''def authenticate_user(username, password):
    """Authenticate a user with username and password."""
    user = find_user(username)
    if user and verify_password(password, user.password_hash):
        return create_session(user)
    return None
''')
            Path(tmpdir, "utils.py").write_text('''def format_date(date):
    """Format a date as a string."""
    return date.strftime("%Y-%m-%d")
''')

            # Pre-compute embeddings
            compute_embeddings(tmpdir)

            # Search for authentication-related code
            results = find_similar_code(
                tmpdir,
                query="login user validation",
                top_k=5,
                threshold=0.0,  # Low threshold for testing
            )

            assert len(results) > 0
            # Auth function should be more similar to login query
            auth_result = next((r for r in results if "authenticate" in r["name"]), None)
            assert auth_result is not None

    def test_find_similar_by_symbol(self) -> None:
        """Test finding similar code by symbol name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "handlers.py").write_text('''def handle_request(req):
    """Handle incoming request."""
    return process(req)

def handle_response(res):
    """Handle outgoing response."""
    return format(res)

def calculate_total(items):
    """Calculate total price."""
    return sum(item.price for item in items)
''')

            compute_embeddings(tmpdir)

            results = find_similar_code(
                tmpdir,
                symbol="handle_request",
                top_k=5,
                threshold=0.0,
            )

            assert len(results) > 0
            # handle_response should be similar to handle_request
            names = [r["name"] for r in results]
            assert "handle_response" in names or "handle_request" in names

    def test_requires_query_or_file_or_symbol(self) -> None:
        """Test that at least one search parameter is required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("def foo(): pass")
            compute_embeddings(tmpdir)

            with pytest.raises(ValueError, match="Must provide"):
                find_similar_code(tmpdir)


class TestSemanticSearch:
    """Tests for semantic_search function."""

    def test_search_by_description(self) -> None:
        """Test searching code by natural language description."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "email.py").write_text(r'''import re

def validate_email(email):
    """Check if an email address is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def send_notification(user, message):
    """Send a notification to a user."""
    pass
''')

            compute_embeddings(tmpdir)

            results = semantic_search(
                tmpdir,
                query="function that checks if email address is valid",
                top_k=5,
                threshold=0.0,
            )

            assert len(results) > 0
            # validate_email should be the top result
            top_names = [r["name"] for r in results[:2]]
            assert "validate_email" in top_names


class TestDetectDuplicates:
    """Tests for detect_duplicates function."""

    def test_finds_similar_functions(self) -> None:
        """Test detection of similar code blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two files with very similar functions
            Path(tmpdir, "module_a.py").write_text('''def process_user_data(user):
    """Process user data and return formatted result."""
    name = user.get("name", "Unknown")
    email = user.get("email", "")
    age = user.get("age", 0)
    return {"name": name, "email": email, "age": age}
''')
            Path(tmpdir, "module_b.py").write_text('''def format_user_info(user):
    """Format user information into a dict."""
    name = user.get("name", "Unknown")
    email = user.get("email", "")
    age = user.get("age", 0)
    return {"name": name, "email": email, "age": age}
''')

            compute_embeddings(tmpdir)

            duplicates = detect_duplicates(
                tmpdir,
                threshold=0.8,  # High similarity
                max_pairs=10,
            )

            # Should find the similar functions
            assert len(duplicates) >= 1
            for dup in duplicates:
                assert "chunk_a" in dup
                assert "chunk_b" in dup
                assert "similarity" in dup
                assert dup["similarity"] >= 0.8

    def test_threshold_filters_results(self) -> None:
        """Test that threshold parameter correctly filters duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two functions with similar structure
            Path(tmpdir, "module_a.py").write_text('''def process_data(data):
    """Process incoming data."""
    result = transform(data)
    return result
''')
            Path(tmpdir, "module_b.py").write_text('''def handle_request(request):
    """Handle incoming request."""
    response = process(request)
    return response
''')

            compute_embeddings(tmpdir)

            # With a very high threshold (> 1.0), nothing should be returned
            # since similarity is clamped to [0, 1]
            duplicates_strict = detect_duplicates(
                tmpdir,
                threshold=1.1,  # Impossible threshold
                max_pairs=10,
            )
            assert len(duplicates_strict) == 0

            # With threshold=0, everything should be returned as duplicates
            duplicates_all = detect_duplicates(
                tmpdir,
                threshold=0.0,
                max_pairs=10,
            )
            # Should find at least 1 pair (there are 2 functions)
            assert len(duplicates_all) >= 1
            # All returned pairs should have similarity >= 0
            for dup in duplicates_all:
                assert dup["similarity"] >= 0.0
                assert dup["similarity"] <= 1.0

    def test_max_pairs_limit(self) -> None:
        """Test that max_pairs limits results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many similar functions
            for i in range(10):
                Path(tmpdir, f"module_{i}.py").write_text(f'''def handler_{i}(request):
    """Handle request and return response."""
    data = request.get("data")
    return {{"status": "ok", "data": data}}
''')

            compute_embeddings(tmpdir)

            duplicates = detect_duplicates(
                tmpdir,
                threshold=0.7,
                max_pairs=3,
            )

            assert len(duplicates) <= 3


class TestSnippetInResults:
    """Tests for code snippets in search/similar results (H1)."""

    def test_search_results_include_snippet(self) -> None:
        """H1: Search results should include a snippet field with code preview."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with a function
            Path(tmpdir, "utils.py").write_text('''def calculate_total(items):
    """Calculate total price of items."""
    total = 0
    for item in items:
        total += item.price
    return total
''')

            compute_embeddings(tmpdir)

            results = semantic_search(
                tmpdir,
                query="calculate total price",
                top_k=5,
                threshold=0.0,  # Low threshold to ensure results
            )

            assert len(results) > 0, "Should have at least one result"

            # At least one result should have a snippet
            snippets_found = [r for r in results if r.get("snippet")]
            assert len(snippets_found) > 0, "At least one result should have a snippet"

            # The snippet should contain relevant code
            snippet = snippets_found[0]["snippet"]
            assert "def" in snippet or "calculate" in snippet, (
                f"Snippet should contain code: {snippet}"
            )

    def test_similar_results_include_snippet(self) -> None:
        """H1: Similar results should include a snippet field with code preview."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with similar functions
            Path(tmpdir, "auth.py").write_text('''def authenticate_user(username, password):
    """Authenticate a user with credentials."""
    user = db.find_user(username)
    if user and verify_password(password, user.hash):
        return create_session(user)
    return None
''')
            Path(tmpdir, "login.py").write_text('''def login(email, secret):
    """Log in a user by email."""
    account = db.get_account(email)
    if account and check_secret(secret, account.key):
        return new_session(account)
    return None
''')

            compute_embeddings(tmpdir)

            results = find_similar_code(
                tmpdir,
                query="user authentication login",
                top_k=5,
                threshold=0.0,
            )

            assert len(results) > 0, "Should have at least one result"

            # At least one result should have a snippet
            snippets_found = [r for r in results if r.get("snippet")]
            assert len(snippets_found) > 0, "At least one result should have a snippet"

    def test_snippet_fallback_reads_from_disk(self) -> None:
        """H1: When chunk lacks snippet, fallback should read from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file
            Path(tmpdir, "service.py").write_text('''def process_request(data):
    """Process incoming request data."""
    validated = validate(data)
    result = transform(validated)
    return result
''')

            compute_embeddings(tmpdir)

            # Search should return results with snippets
            results = semantic_search(
                tmpdir,
                query="process data request",
                top_k=3,
                threshold=0.0,
            )

            assert len(results) > 0
            # Check that file path and line info are present
            result = results[0]
            assert "file" in result
            assert result["file"] == "service.py"
            # Snippet should be present (either from cache or fallback)
            assert "snippet" in result, "Result should have snippet from cache or fallback"


class TestDuplicatePreview:
    """Tests for preview fields in duplicate detection (H3)."""

    def test_duplicates_include_preview_fields(self) -> None:
        """H3: Duplicate pairs should include preview_a, preview_b, and preview fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create very similar functions in different files
            Path(tmpdir, "handler_a.py").write_text('''def process_data(input_data):
    """Process the input data and return result."""
    validated = validate_input(input_data)
    transformed = apply_transformation(validated)
    return format_output(transformed)
''')
            Path(tmpdir, "handler_b.py").write_text('''def handle_data(input_data):
    """Handle the input data and return result."""
    validated = validate_input(input_data)
    transformed = apply_transformation(validated)
    return format_output(transformed)
''')

            compute_embeddings(tmpdir)

            duplicates = detect_duplicates(
                tmpdir,
                threshold=0.7,  # Moderate threshold
                max_pairs=10,
            )

            # Should find at least one duplicate pair
            assert len(duplicates) >= 1, "Should find at least one duplicate pair"

            # Each pair should have preview fields
            for dup in duplicates:
                # Should have at least one preview
                has_preview = (
                    "preview_a" in dup
                    or "preview_b" in dup
                    or "preview" in dup
                )
                assert has_preview, f"Duplicate pair should have preview: {dup}"

                # The combined preview should be set if either individual is set
                if "preview_a" in dup or "preview_b" in dup:
                    assert "preview" in dup, "Should have combined preview when individual exists"

    def test_duplicate_preview_contains_code(self) -> None:
        """H3: Preview fields should contain actual code content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with recognizable code
            Path(tmpdir, "utils_a.py").write_text('''def calculate_score(values):
    """Calculate the score from values."""
    total = sum(values)
    average = total / len(values)
    return average * 100
''')
            Path(tmpdir, "utils_b.py").write_text('''def compute_score(values):
    """Compute the score from values."""
    total = sum(values)
    average = total / len(values)
    return average * 100
''')

            compute_embeddings(tmpdir)

            duplicates = detect_duplicates(
                tmpdir,
                threshold=0.7,
                max_pairs=10,
            )

            assert len(duplicates) >= 1

            # Check that preview contains code
            dup = duplicates[0]
            preview = dup.get("preview") or dup.get("preview_a") or dup.get("preview_b")
            assert preview is not None, "Should have a preview"
            assert "def " in preview, f"Preview should contain function definition: {preview}"


class TestPackageScoping:
    """Tests for package scoping in semantic analysis (H2)."""

    def test_search_with_package_scoping(self) -> None:
        """H2: Search should be limited to files in the specified package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a monorepo-like structure
            pkg_a = Path(tmpdir) / "packages" / "core"
            pkg_a.mkdir(parents=True)
            (pkg_a / "utils.py").write_text('''def core_function():
    """A function in the core package."""
    return "core"
''')

            pkg_b = Path(tmpdir) / "packages" / "api"
            pkg_b.mkdir(parents=True)
            (pkg_b / "routes.py").write_text('''def api_handler():
    """A function in the api package."""
    return "api"
''')

            # Search with package scoping to core
            results = semantic_search(
                tmpdir,
                query="function",
                top_k=10,
                threshold=0.0,
                package="packages/core",
            )

            # All results should be from the core package
            for result in results:
                assert "packages/core" in result["file"], (
                    f"Result file should be in packages/core: {result['file']}"
                )
                assert "packages/api" not in result["file"], (
                    f"Result file should not be in packages/api: {result['file']}"
                )

    def test_similar_with_package_scoping(self) -> None:
        """H2: Similar should be limited to files in the specified package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create monorepo structure
            libs = Path(tmpdir) / "libs"
            (libs / "alpha").mkdir(parents=True)
            (libs / "alpha" / "helpers.py").write_text('''def alpha_helper():
    """Helper function in alpha."""
    return "alpha"
''')

            (libs / "beta").mkdir(parents=True)
            (libs / "beta" / "helpers.py").write_text('''def beta_helper():
    """Helper function in beta."""
    return "beta"
''')

            # Find similar code scoped to alpha
            results = find_similar_code(
                tmpdir,
                query="helper function",
                top_k=10,
                threshold=0.0,
                package="libs/alpha",
            )

            # All results should be from libs/alpha
            for result in results:
                assert "libs/alpha" in result["file"], (
                    f"Result should be in libs/alpha: {result['file']}"
                )

    def test_duplicates_with_package_scoping(self) -> None:
        """H2: Duplicates detection should be limited to specified package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure with duplicates in different packages
            pkg1 = Path(tmpdir) / "pkg1"
            pkg1.mkdir()
            (pkg1 / "util_a.py").write_text('''def common_pattern():
    """Process data and return result."""
    data = get_input()
    result = transform(data)
    return output(result)
''')
            (pkg1 / "util_b.py").write_text('''def similar_pattern():
    """Process data and return result."""
    data = get_input()
    result = transform(data)
    return output(result)
''')

            pkg2 = Path(tmpdir) / "pkg2"
            pkg2.mkdir()
            (pkg2 / "handler.py").write_text('''def handler_pattern():
    """Handle request and return response."""
    request = get_request()
    response = process(request)
    return send(response)
''')

            # Detect duplicates scoped to pkg1 only
            duplicates = detect_duplicates(
                tmpdir,
                threshold=0.7,
                max_pairs=10,
                package="pkg1",
            )

            # All duplicates should be from pkg1
            for dup in duplicates:
                assert "pkg1" in dup["chunk_a"]["file"], (
                    f"chunk_a should be in pkg1: {dup['chunk_a']['file']}"
                )
                assert "pkg1" in dup["chunk_b"]["file"], (
                    f"chunk_b should be in pkg1: {dup['chunk_b']['file']}"
                )

    def test_package_scoped_cache_is_separate(self) -> None:
        """H2: Package-scoped embeddings use SQLite with scope filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create monorepo structure
            core = Path(tmpdir) / "core"
            core.mkdir()
            (core / "main.py").write_text("def core_main():\n    pass\n")

            api = Path(tmpdir) / "api"
            api.mkdir()
            (api / "main.py").write_text("def api_main():\n    pass\n")

            # Compute embeddings for full repo (stores all chunks)
            result = compute_embeddings(tmpdir)

            # Check that SQLite cache exists with all chunks
            cache_dir = Path(tmpdir) / ".nodestradamus"
            assert (cache_dir / "nodestradamus.db").exists(), "SQLite database should exist"

            # Verify chunks from both packages are stored
            assert result["metadata"]["chunks_extracted"] >= 2, "Should have chunks from both packages"

            # Test that scope filtering works in find_similar_code
            # (package parameter filters results, not separate cache files)
            core_results = find_similar_code(
                tmpdir, query="main function", package="core", threshold=0.0
            )
            api_results = find_similar_code(
                tmpdir, query="main function", package="api", threshold=0.0
            )

            # Results should be scoped to their packages
            for r in core_results:
                assert "core" in r["file"], f"Core results should be from core/: {r['file']}"
            for r in api_results:
                assert "api" in r["file"], f"API results should be from api/: {r['file']}"
