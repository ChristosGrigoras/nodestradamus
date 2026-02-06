#!/bin/bash
# Main entry point script

# Source helper utilities
source ./utils.sh
. ./lib/helpers.sh

# Define a function
setup_environment() {
    echo "Setting up environment"
    export PATH="$PATH:/usr/local/bin"
}

# Call another script
./scripts/deploy.sh

# Function that uses sourced utilities
main() {
    setup_environment
    log_message "Starting main process"
    
    # Direct script call
    ../shared/common.sh
}

main "$@"
