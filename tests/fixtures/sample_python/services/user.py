"""Sample user service for testing Python dependency analysis."""

from utils import helper_function


class UserService:
    """Service for user operations."""

    def __init__(self):
        """Initialize the user service."""
        self.users = {}

    def get_user(self, user_id: str):
        """Get a user by ID."""
        helper_function()
        return self.users.get(user_id)

    def create_user(self, user_id: str, name: str):
        """Create a new user."""
        self.users[user_id] = {"id": user_id, "name": name}
        return self.users[user_id]


class AdminService(UserService):
    """Extended service for admin operations."""

    def delete_user(self, user_id: str):
        """Delete a user."""
        if user_id in self.users:
            del self.users[user_id]
