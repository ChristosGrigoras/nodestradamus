def greet(name):
    """Greet a user by name."""
    return f"Hello, {name}!"


def format_list(items):
    """Format a list of items as a comma-separated string."""
    return ", ".join(items)


def capitalize_words(text):
    """Capitalize the first letter of each word in a string."""
    return text.title()


def calculate_discount(price: float, discount_percentage: float) -> float:
    """
    Calculate the discounted price.

    Args:
        price: Original price of the item.
        discount_percentage: Discount percentage (0-100).

    Returns:
        The price after applying the discount.

    Raises:
        ValueError: If discount_percentage is not between 0 and 100.
    """
    if not 0 <= discount_percentage <= 100:
        raise ValueError(f"Discount must be between 0 and 100, got {discount_percentage}")
    
    discount_amount = price * (discount_percentage / 100)
    return price - discount_amount