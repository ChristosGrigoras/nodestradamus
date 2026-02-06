"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    filepath = temp_dir / "sample.py"
    filepath.write_text('''
def hello_world():
    """Say hello."""
    print("Hello, World!")


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


class UserService:
    """Service for user operations."""

    def __init__(self):
        self.users = {}

    def get_user(self, user_id: str):
        """Get a user by ID."""
        return self.users.get(user_id)

    def create_user(self, user_id: str, name: str):
        """Create a new user."""
        self.users[user_id] = {"id": user_id, "name": name}
        return self.users[user_id]


def main():
    """Main entry point."""
    hello_world()
    service = UserService()
    service.create_user("1", "Alice")
    print(greet("Bob"))


if __name__ == "__main__":
    main()
''')
    return filepath


@pytest.fixture
def sample_typescript_file(temp_dir: Path) -> Path:
    """Create a sample TypeScript file for testing."""
    filepath = temp_dir / "sample.ts"
    filepath.write_text('''
import { User } from './models/user';
import * as utils from './utils';

export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export const add = (a: number, b: number): number => {
    return a + b;
};

export class UserService {
    private users: Map<string, User> = new Map();

    getUser(id: string): User | undefined {
        return this.users.get(id);
    }

    createUser(id: string, name: string): User {
        const user: User = { id, name };
        this.users.set(id, user);
        return user;
    }
}

async function fetchData(url: string): Promise<any> {
    const response = await fetch(url);
    return response.json();
}
''')
    return filepath


@pytest.fixture
def sample_rules_dir(temp_dir: Path) -> Path:
    """Create a sample rules directory for testing."""
    rules_dir = temp_dir / ".cursor" / "rules"
    rules_dir.mkdir(parents=True)

    # Create a valid rule file
    (rules_dir / "100-python.mdc").write_text('''---
description: Python coding conventions
globs: "**/*.py"
alwaysApply: false
---

# Python

## Conventions
- Use snake_case for functions
- Use PascalCase for classes
- Add type hints to all functions
''')

    # Create another rule file
    (rules_dir / "200-project.mdc").write_text('''---
description: Project-specific conventions
globs: "**/*"
alwaysApply: true
---

# My Project

## Overview
This is a sample project.

## Conventions
- Follow the team style guide
''')

    return rules_dir


@pytest.fixture
def sample_graph() -> dict:
    """Create a sample dependency graph for testing."""
    return {
        "nodes": ["main.py::main", "utils.py::helper"],
        "node_details": [
            {"name": "main.py::main", "type": "function", "line": 10},
            {"name": "utils.py::helper", "type": "function", "line": 5},
        ],
        "edges": [
            {"from": "main.py::main", "to": "utils.py::helper", "type": "calls", "resolved": True},
        ],
        "errors": [],
        "metadata": {
            "generated_at": "2026-01-26T00:00:00+00:00",
            "generator": "test",
        },
    }


@pytest.fixture
def sample_python_dir() -> str:
    """Return path to sample Python fixtures directory."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "sample_python"
    return str(fixtures_dir)
