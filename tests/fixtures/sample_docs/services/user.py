"""User service module."""


class UserService:
    """Service for user operations."""

    def get_user(self, user_id: str) -> dict:
        """Get user by ID."""
        return {"id": user_id, "name": "Test User"}

    def create_user(self, name: str) -> dict:
        """Create a new user."""
        return {"id": "new", "name": name}
