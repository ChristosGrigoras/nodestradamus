"""Authentication service - 1 class + 3 standalone functions.

Expected counts:
- Classes: 1 (AuthService)
- Methods: 3 (__init__, login, logout)
- Standalone functions: 3 (create_token, verify_token, refresh_token)
"""

from python.utils import hash_password


class AuthService:
    """Authentication service."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def login(self, email: str, password: str) -> str | None:
        """Authenticate user and return token."""
        hashed = hash_password(password)
        return create_token(email) if hashed else None

    def logout(self, token: str) -> bool:
        """Invalidate a token."""
        return True


def create_token(user_id: str) -> str:
    """Create a new authentication token."""
    return f"token_{user_id}"


def verify_token(token: str) -> bool:
    """Verify if token is valid."""
    return token.startswith("token_")


def refresh_token(token: str) -> str:
    """Refresh an existing token."""
    if verify_token(token):
        return f"refreshed_{token}"
    raise ValueError("Invalid token")
