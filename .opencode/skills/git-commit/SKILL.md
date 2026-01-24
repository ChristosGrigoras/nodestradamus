---
name: git-commit
description: Generate clear, conventional commit messages
license: MIT
---

## What I Do

- Analyze staged changes
- Generate descriptive commit messages
- Follow conventional commit format
- Keep messages focused and clear

## Commit Format

```
<type>(<scope>): <subject>

<body>
```

## Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **refactor**: Code restructuring
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

## Guidelines

- Subject line: max 50 chars, imperative mood
- Body: explain WHAT and WHY, not HOW
- Reference issues when applicable
- One logical change per commit

## Examples

```
feat(auth): add JWT token refresh endpoint

Implements automatic token refresh to improve UX.
Tokens now refresh 5 minutes before expiration.

Closes #42
```

```
fix(utils): handle empty list in format_list()

Previously raised IndexError on empty input.
Now returns empty string gracefully.
```
