# Git Commit Skill

## Purpose

Write clear, consistent commit messages following Conventional Commits.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

| Type | When to Use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `chore` | Maintenance, dependencies |
| `ci` | CI/CD changes |

## Rules

### Subject Line
- Imperative mood: "add" not "added" or "adds"
- No period at end
- Max 50 characters
- Lowercase

### Body
- Explain what and why, not how
- Wrap at 72 characters
- Separate from subject with blank line

### Footer
- Reference issues: `Fixes #123`, `Closes #456`
- Breaking changes: `BREAKING CHANGE: description`

## Examples

```
feat(auth): add password reset functionality

Users can now reset their password via email link.
Token expires after 24 hours.

Closes #42
```

```
fix(api): handle null response from payment service

The Stripe API occasionally returns null instead of
an error object. Added defensive check.

Fixes #78
```

## Process

1. Stage changes: `git add -A`
2. Review diff: `git diff --staged`
3. Write message following format
4. Commit: `git commit -m "..."`
