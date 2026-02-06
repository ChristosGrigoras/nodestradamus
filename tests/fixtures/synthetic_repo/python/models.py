"""Models module - 3 classes with methods.

Expected counts:
- Classes: 3 (User, Product, Order)
- Methods: 9 (3 per class: __init__, save, delete)
- Standalone functions: 0
"""


class User:
    """User model."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def save(self) -> bool:
        """Save user to database."""
        return True

    def delete(self) -> bool:
        """Delete user from database."""
        return True


class Product:
    """Product model."""

    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

    def save(self) -> bool:
        """Save product to database."""
        return True

    def delete(self) -> bool:
        """Delete product from database."""
        return True


class Order:
    """Order model."""

    def __init__(self, user: User, products: list):
        self.user = user
        self.products = products

    def save(self) -> bool:
        """Save order to database."""
        return True

    def delete(self) -> bool:
        """Delete order from database."""
        return True
