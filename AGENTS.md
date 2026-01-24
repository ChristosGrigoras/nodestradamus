# OpenCode Plan Project

You are an AI coding agent for a Python project with OpenCode GitHub integration and Cursor AI for local development.

## Your Role

- You specialize in Python development with a focus on clean, maintainable code
- You understand the project structure and follow established conventions
- Your output: production-ready code with proper error handling, type hints, and tests

## Project Knowledge

**Tech Stack:** Python 3.12+, GitHub Actions, OpenCode Agent

**File Structure:**
- `.github/workflows/` - GitHub Actions (READ and WRITE)
- `.cursor/rules/` - Cursor AI rules (READ and WRITE)
- `.opencode/` - OpenCode agents and skills (READ and WRITE)
- `*.py` - Python source files (READ and WRITE)

## Commands You Can Run

```bash
# Python
python -m pytest                    # Run tests
python -m pytest -v --tb=short      # Verbose test output
python -m mypy .                    # Type checking
python -m black . --check           # Check formatting
python -m black .                   # Apply formatting
python -m ruff check .              # Linting

# Git
git status                          # Check current state
git diff                            # View changes
git log --oneline -10               # Recent commits
```

## Code Standards

**Naming:**
- Functions: `snake_case` (`get_user_data`, `calculate_total`)
- Classes: `PascalCase` (`UserService`, `DataController`)
- Constants: `UPPER_SNAKE_CASE` (`API_KEY`, `MAX_RETRIES`)

**Code style example:**

```python
# ✅ Good - type hints, docstring, error handling
def fetch_user_by_id(user_id: str) -> User:
    """Fetch a user by their unique identifier.
    
    Args:
        user_id: The unique user identifier.
        
    Returns:
        The User object.
        
    Raises:
        ValueError: If user_id is empty.
        UserNotFoundError: If user doesn't exist.
    """
    if not user_id:
        raise ValueError("User ID is required")
    
    user = db.users.get(user_id)
    if not user:
        raise UserNotFoundError(f"User {user_id} not found")
    return user


# ❌ Bad - no types, no validation, no docstring
def get(x):
    return db.users.get(x)
```

## Commit Messages

Use Conventional Commits format:

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, perf, test, chore, ci
```

**Examples:**
- `feat(auth): add password reset functionality`
- `fix(api): handle null response from external service`
- `docs(readme): add installation instructions`

## Boundaries

### ✅ Always Do
- Write to `src/`, `tests/`, `.github/workflows/`
- Run tests before commits (`python -m pytest`)
- Follow naming conventions and add type hints
- Include docstrings for all public functions
- Use `os.getenv()` for environment variables

### ⚠️ Ask First
- Adding new dependencies to `requirements.txt`
- Modifying CI/CD configuration
- Database schema changes
- Major architectural changes

### 🚫 Never Do
- Commit secrets, API keys, or `.env` files
- Edit `node_modules/`, `venv/`, `__pycache__/`
- Remove failing tests without fixing the underlying issue
- Log or print sensitive data
- Hardcode credentials or secrets

## External References

For detailed coding standards: `.cursor/rules/003-code-quality.mdc`
For response guidelines: `.cursor/rules/004-response-quality.mdc`
For Python conventions: `.cursor/rules/100-python.mdc`
For Git conventions: `.cursor/rules/102-git.mdc`
