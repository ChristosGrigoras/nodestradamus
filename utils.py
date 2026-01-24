def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


def format_list(items: list) -> str:
    """Format a list of items as a comma-separated string."""
    return ", ".join(items)


def capitalize_words(text: str) -> str:
    """Capitalize the first letter of each word in a string."""
    return text.title()