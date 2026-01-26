# OpenCode Configuration

## File: `AGENTS.md`

This is what OpenCode reads first. It should contain:

```markdown
# Project Name

Brief description of what this project does.

## Project Structure
- `src/` - Main source code
- `tests/` - Test files
- `docs/` - Documentation

## Code Standards
- Your specific conventions
- Technologies used

## Important Notes
- Any context OpenCode needs
```

---

## File: `opencode.json`

Configuration file that controls OpenCode's behavior:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": [
    ".cursor/rules/003-code-quality.mdc",
    ".cursor/rules/004-response-quality.mdc"
  ],
  "agent": {
    "build": {
      "model": "opencode/big-pickle",
      "temperature": 0.3
    }
  },
  "permission": {
    "edit": "allow",
    "bash": {
      "git status": "allow",
      "git diff": "allow"
    }
  }
}
```

**Key insight:** The `instructions` array points to Cursor's rule files. This means both AIs follow the same standards!

---

## Custom Agents

Create specialized AI personas in `.opencode/agents/`:

### `reviewer.md`

```markdown
# Code Reviewer Agent

You are a senior code reviewer. Focus on:
- Code quality and readability
- Security vulnerabilities
- Performance issues
- Best practices

Be constructive but thorough. Always explain why something should change.
```

### `documenter.md`

```markdown
# Documentation Agent

You generate comprehensive documentation. Focus on:
- Clear explanations
- Usage examples
- API references
- Getting started guides
```

### `housekeeper.md`

```markdown
# Housekeeper Agent

You identify cleanup opportunities. Look for:
- Dead code and unused imports
- Outdated dependencies
- Files that could be archived
- Refactoring opportunities
```

**Usage:**
```
/opencode @reviewer check this PR for security issues
/opencode @documenter generate API docs for the routes module
/opencode @housekeeper scan for dead code
```

---

## Skills

Reusable behaviors in `.opencode/skills/*/SKILL.md`:

### `code-review/SKILL.md`

```markdown
# Code Review Skill

## Checklist
1. Check for security issues (SQL injection, XSS, etc.)
2. Verify error handling
3. Look for performance bottlenecks
4. Ensure tests exist
5. Check documentation

## Output Format
- 🔴 Critical: Must fix
- 🟡 Warning: Should fix
- 🟢 Suggestion: Nice to have
```

### `git-commit/SKILL.md`

```markdown
# Git Commit Skill

## Commit Message Format
Use Conventional Commits:

```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, perf, test, chore, ci
```

## Examples
- `feat(auth): add password reset functionality`
- `fix(api): handle null response from external service`
- `docs(readme): add installation instructions`
```

### `housekeeping/SKILL.md`

```markdown
# Housekeeping Skill

## What to Look For
1. Unused imports and dead code
2. TODO/FIXME comments older than 6 months
3. Deprecated dependencies
4. Large files that could be split
5. Duplicate code patterns

## Report Format
Group findings by severity and effort to fix.
```

---

## Usage Examples

### Using OpenCode via GitHub Comments

On any issue or pull request, comment:

```
/opencode create a user authentication module with login and logout functions
```

```
/oc fix the TypeError in utils.py line 45
```

```
/opencode add comprehensive docstrings to all functions in api.py
```

```
/opencode refactor the database connection to use connection pooling
```

### Using Custom Agents

```
/opencode @reviewer check the code in this PR for security issues
```

```
/opencode @documenter generate API documentation for the routes module
```

```
/opencode @housekeeper scan the codebase for dead code and unused imports
```

### Using Cursor AI Locally

Just open your project in Cursor IDE. The rules in `.cursor/rules/` are automatically applied.

**Meta-commands you can use in Cursor chat:**

```
@rules              # Show active rules
@context python     # Force Python context
@clean              # Minimal context mode
```
