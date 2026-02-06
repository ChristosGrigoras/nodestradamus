"""Sample utility module for testing Python dependency analysis."""


def helper_function():
    """A simple helper function."""
    print("Helper called")


def another_helper(value: str) -> str:
    """Another helper function with type hints."""
    return value.upper()
