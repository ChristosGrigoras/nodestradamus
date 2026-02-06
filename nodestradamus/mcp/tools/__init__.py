"""MCP tools registration.

Provides tool definitions, handlers, and registration functions for
the Nodestradamus MCP server.
"""

from mcp.server import Server

from nodestradamus.mcp.tools.definitions import (
    ALL_TOOLS,
    ANALYZE_COOCCURRENCE_TOOL,
    ANALYZE_DEPS_TOOL,
    ANALYZE_DOCS_TOOL,
    ANALYZE_GRAPH_TOOL,
    ANALYZE_STRINGS_TOOL,
    CODEBASE_HEALTH_TOOL,
    GET_IMPACT_TOOL,
    MANAGE_CACHE_TOOL,
    PROJECT_SCOUT_TOOL,
    SEMANTIC_ANALYSIS_TOOL,
)
from nodestradamus.mcp.tools.dispatch import (
    dispatch_tool,
    inject_timing,
    register_graph_tools,
)


def register_tools(server: Server) -> None:
    """Register all MCP tools with the server.

    Args:
        server: The MCP server instance.
    """
    register_graph_tools(server)


__all__ = [
    # Registration
    "register_tools",
    "register_graph_tools",
    # Dispatch
    "dispatch_tool",
    "inject_timing",
    # Tool definitions
    "ALL_TOOLS",
    "ANALYZE_DEPS_TOOL",
    "ANALYZE_COOCCURRENCE_TOOL",
    "GET_IMPACT_TOOL",
    "PROJECT_SCOUT_TOOL",
    "ANALYZE_DOCS_TOOL",
    "ANALYZE_GRAPH_TOOL",
    "ANALYZE_STRINGS_TOOL",
    "SEMANTIC_ANALYSIS_TOOL",
    "CODEBASE_HEALTH_TOOL",
    "MANAGE_CACHE_TOOL",
]
