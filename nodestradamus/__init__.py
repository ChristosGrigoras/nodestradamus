"""Nodestradamus - MCP server for codebase intelligence."""

# Load .env so NODESTRADAMUS_EMBEDDING_PROVIDER, MISTRAL_API_KEY, etc. are set
# for any entry point (CLI, pytest, scripts) that imports nodestradamus.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Keep in sync with pyproject.toml [project] version (CI checks this).
__version__ = "0.1.0"


def run_server() -> None:
    """Run the Nodestradamus MCP server (blocking).

    This is the main entry point for starting the MCP server.
    Uses stdio transport for communication with Cursor/Claude Desktop.
    """
    from nodestradamus.mcp.server import run_server as _run_server
    _run_server()
