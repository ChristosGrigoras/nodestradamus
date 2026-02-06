#!/bin/bash
# Run the Nodestradamus MCP server
# Usage: ./run-mcp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
exec python -m nodestradamus serve
