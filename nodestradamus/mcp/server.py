"""MCP server implementation for Nodestradamus.

Provides codebase intelligence tools via the Model Context Protocol.

Refresh after edits
------------------
Analysis reflects the current state of the repo. After the user or assistant
saves new or edited files, re-running tools (e.g. analyze_deps, get_impact,
codebase_health) refreshes the graph and reports for the new changes and
adds value to the next step.

Workspace Isolation
-------------------
Tools that create caches (like semantic_analysis) support an optional
`workspace_path` parameter. When provided, caches are stored in the
workspace rather than the analyzed repo, enabling isolation between
different Cursor windows working on the same project.

Cache locations:
- With workspace_path: <workspace>/.nodestradamus/cache/<repo_hash>/
- Without workspace_path: <repo>/.nodestradamus/ (fallback)

Orphan Prevention
-----------------
Cursor doesn't always send SIGTERM when closing. This server monitors stdin
for POLLHUP (pipe closed) and parent PID changes to detect orphaning and
self-terminate, preventing zombie processes.
"""

import asyncio
import os
import select
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server

from nodestradamus import __version__
from nodestradamus.logging import logger
from nodestradamus.mcp.tools import register_tools

# Server configuration
SERVER_NAME = "nodestradamus"
SERVER_VERSION = __version__
PROTOCOL_VERSION = "2024-11-05"

# Orphan detection interval (seconds)
_ORPHAN_CHECK_INTERVAL = 2.0


def create_server() -> Server:
    """Create and configure the MCP server.

    Returns:
        Configured MCP Server instance with all tools registered.
    """
    server = Server(SERVER_NAME)

    # Register all tools
    register_tools(server)

    return server


def _is_orphan_parent(ppid: int) -> bool:
    """Check if parent PID indicates an orphaned process.

    On Linux, orphaned processes get reparented to:
    - PID 1 (init/systemd)
    - systemd --user session manager (varies by system)

    We detect orphaning by checking if the parent is init/systemd.
    """
    if ppid == 1:
        return True

    # Check if parent is systemd (user session manager)
    try:
        with open(f"/proc/{ppid}/comm") as f:
            parent_comm = f.read().strip()
            if parent_comm in ("systemd", "init", "launchd"):
                return True
    except OSError:
        pass

    return False


async def _orphan_watchdog(shutdown_event: asyncio.Event) -> None:
    """Monitor for orphaning conditions and signal shutdown.

    Detects conditions that indicate Cursor has closed:
    1. stdin POLLHUP - the pipe was closed (Cursor exited)
    2. Parent PID changed - reparented (orphaned)
    3. Parent is init/systemd - orphaned and reparented

    This prevents zombie MCP server processes when Cursor doesn't
    send SIGTERM on exit (known Cursor bug).
    """
    original_ppid = os.getppid()

    while not shutdown_event.is_set():
        try:
            # Check 1: stdin closed (POLLHUP)
            # Use select with timeout=0 to check for exceptional conditions
            if hasattr(select, "poll"):
                # Linux: use poll() for proper POLLHUP detection
                poll = select.poll()
                poll.register(sys.stdin.fileno(), select.POLLHUP | select.POLLERR)
                events = poll.poll(0)  # Non-blocking
                if events:
                    for _, event in events:
                        if event & (select.POLLHUP | select.POLLERR):
                            logger.info("Stdin closed (POLLHUP), shutting down")
                            shutdown_event.set()
                            return
            else:
                # macOS/Windows fallback: check if stdin is readable
                # When pipe closes, select returns it as "readable" but read returns empty
                readable, _, exceptional = select.select(
                    [sys.stdin], [], [sys.stdin], 0
                )
                if exceptional:
                    logger.info("Stdin exceptional condition, shutting down")
                    shutdown_event.set()
                    return

            # Check 2: parent PID changed (orphaned)
            current_ppid = os.getppid()
            if current_ppid != original_ppid:
                logger.info(
                    "Parent PID changed (%d -> %d), orphaned, shutting down",
                    original_ppid,
                    current_ppid,
                )
                shutdown_event.set()
                return

            # Check 3: parent is init/systemd (reparented after orphaning)
            if _is_orphan_parent(current_ppid):
                logger.info(
                    "Parent is init/systemd (PID %d), orphaned, shutting down",
                    current_ppid,
                )
                shutdown_event.set()
                return

            # Sleep before next check
            await asyncio.sleep(_ORPHAN_CHECK_INTERVAL)

        except (OSError, ValueError):
            # stdin file descriptor invalid - Cursor closed
            logger.info("Stdin invalid, shutting down")
            shutdown_event.set()
            return


async def run_server_async() -> None:
    """Run the MCP server with stdio transport and orphan detection."""
    server = create_server()
    shutdown_event = asyncio.Event()

    # Start orphan watchdog
    watchdog_task = asyncio.create_task(_orphan_watchdog(shutdown_event))

    try:
        async with stdio_server() as (read_stream, write_stream):
            # Run server until shutdown or stream closes
            server_task = asyncio.create_task(
                server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
            )

            # Wait for either server to finish or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, asyncio.create_task(shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    finally:
        # Clean up watchdog
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

        logger.info("MCP server shutdown complete")


def run_server() -> None:
    """Run the MCP server (blocking)."""
    asyncio.run(run_server_async())
