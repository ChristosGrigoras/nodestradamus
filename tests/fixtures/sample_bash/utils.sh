#!/bin/bash
# Utility functions

log_message() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg"
}

check_dependency() {
    local dep="$1"
    if ! command -v "$dep" &> /dev/null; then
        echo "Error: $dep not found"
        return 1
    fi
}

get_script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}
