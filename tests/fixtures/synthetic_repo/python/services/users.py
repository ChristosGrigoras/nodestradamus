"""User service - 1 class + 2 standalone functions.

Expected counts:
- Classes: 1 (UserService)
- Methods: 4 (__init__, get_user, create_user, list_users)
- Standalone functions: 2 (get_user_by_email, count_users)
"""

from python.models import User
from python.utils import validate_email


class UserService:
    """Service for user operations."""

    def __init__(self):
        self.users: dict[str, User] = {}

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.users.get(user_id)

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        if not validate_email(email):
            raise ValueError("Invalid email")
        user = User(name, email)
        self.users[email] = user
        return user

    def list_users(self) -> list[User]:
        """List all users."""
        return list(self.users.values())


def get_user_by_email(email: str) -> User | None:
    """Get user by email address."""
    # This would query the database
    return None


def count_users() -> int:
    """Count total users."""
    return 0
