"""Utility functions for sample docs testing."""


def validate_input(data: dict) -> bool:
    """Validate input data.

    Args:
        data: The data to validate.

    Returns:
        True if valid, False otherwise.
    """
    return isinstance(data, dict) and len(data) > 0


def format_output(result: dict) -> str:
    """Format output for display.

    Args:
        result: The result to format.

    Returns:
        Formatted string.
    """
    return str(result)
