"""Utility functions module - 5 standalone functions.

Expected counts:
- Classes: 0
- Functions: 5 (format_date, parse_json, slugify, hash_password, validate_email)
"""

import hashlib
import json
import re
from datetime import datetime


def format_date(dt: datetime) -> str:
    """Format a datetime as ISO string."""
    return dt.isoformat()


def parse_json(data: str) -> dict:
    """Parse JSON string to dict."""
    return json.loads(data)


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))
