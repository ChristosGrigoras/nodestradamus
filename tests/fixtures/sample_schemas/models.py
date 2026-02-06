"""Sample Python models for testing field extraction."""

from dataclasses import dataclass

from pydantic import BaseModel


class User(BaseModel):
    """User model."""
    id: int
    email: str
    name: str | None = None
    is_active: bool = True


@dataclass
class Order:
    """Order model."""
    id: int
    user_id: int
    total: float
    status: str = "pending"
