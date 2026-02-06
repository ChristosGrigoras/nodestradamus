"""MCP tool registration and dispatch.

Registers all Nodestradamus tools with the MCP server and handles
dispatching tool calls to the appropriate handlers.
"""

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from nodestradamus.logging import log_operation
from nodestradamus.mcp.tools.definitions import ALL_TOOLS
from nodestradamus.mcp.tools.handlers import (
    handle_analyze_cooccurrence,
    handle_analyze_deps,
    handle_analyze_docs,
    handle_analyze_graph,
    handle_analyze_strings,
    handle_codebase_health,
    handle_compare_rules,
    handle_detect_conflicts,
    handle_find_similar,
    handle_get_changes_since_last,
    handle_get_impact,
    handle_manage_cache,
    handle_project_scout,
    handle_quick_start,
    handle_semantic_analysis,
    handle_validate_rules,
)


def register_graph_tools(server: Server) -> None:
    """Register graph analysis tools with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available graph analysis tools."""
        return ALL_TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        try:
            result = await dispatch_tool(name, arguments)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error: {type(e).__name__}: {e}",
                )
            ]


# Handler dispatch table
_HANDLERS = {
    # Core tools
    "project_scout": handle_project_scout,
    "analyze_deps": handle_analyze_deps,
    "analyze_cooccurrence": handle_analyze_cooccurrence,
    "get_impact": handle_get_impact,
    "analyze_docs": handle_analyze_docs,
    # Consolidated tools
    "analyze_graph": handle_analyze_graph,
    "analyze_strings": handle_analyze_strings,
    "semantic_analysis": handle_semantic_analysis,
    # Workflow tools
    "quick_start": handle_quick_start,
    "codebase_health": handle_codebase_health,
    "compare_rules_to_codebase": handle_compare_rules,
    "find_similar": handle_find_similar,
    "get_changes_since_last": handle_get_changes_since_last,
    # Rules validation tools
    "validate_rules": handle_validate_rules,
    "detect_rule_conflicts": handle_detect_conflicts,
    # Cache management
    "manage_cache": handle_manage_cache,
}


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch tool call to appropriate handler.

    All tool responses include a 'timing' field with performance metrics.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        JSON string with tool response and timing.

    Raises:
        ValueError: If tool name is unknown.
    """
    handler = _HANDLERS.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")

    # Remove legacy timeout argument if passed (no longer used)
    arguments.pop("timeout", None)

    # Log tool invocation with relevant details
    log_details: dict[str, Any] = {"repo": arguments.get("repo_path", "N/A")}
    if "algorithm" in arguments:
        log_details["algorithm"] = arguments["algorithm"]
    if "mode" in arguments:
        log_details["mode"] = arguments["mode"]

    with log_operation(f"tool:{name}", log_details) as timing:
        result_str = await handler(arguments)

    # Inject timing into the response
    return inject_timing(result_str, timing.elapsed_ms)


def inject_timing(result_str: str, elapsed_ms: float) -> str:
    """Inject timing information into a JSON response.

    Args:
        result_str: JSON string from handler.
        elapsed_ms: Elapsed time in milliseconds.

    Returns:
        JSON string with timing field added.
    """
    try:
        result = json.loads(result_str)
        if isinstance(result, dict):
            result["timing"] = {
                "total_ms": round(elapsed_ms, 1),
            }
            return json.dumps(result, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass
    return result_str
