# Customization Guide

## Adding a New Language

Create `.cursor/rules/101-javascript.mdc`:

```markdown
---
description: JavaScript coding conventions
globs: "**/*.{js,jsx,ts,tsx}"
alwaysApply: false
---

# JavaScript

## Conventions
- Use const/let, never var
- Prefer arrow functions
- Use async/await over .then()
- Destructure when possible

## Naming
- camelCase for variables and functions
- PascalCase for components and classes
- UPPER_SNAKE for constants
```

### More Language Examples

**Rust** (`102-rust.mdc`):
```markdown
---
description: Rust coding conventions
globs: "**/*.rs"
alwaysApply: false
---

# Rust

## Conventions
- Use `Result` for fallible operations
- Prefer `?` operator over `unwrap()`
- Document public APIs with `///`
- Use `clippy` suggestions

## Naming
- snake_case for functions and variables
- PascalCase for types and traits
- SCREAMING_SNAKE for constants
```

**Go** (`103-go.mdc`):
```markdown
---
description: Go coding conventions
globs: "**/*.go"
alwaysApply: false
---

# Go

## Conventions
- Use `gofmt` formatting
- Handle errors explicitly
- Prefer composition over inheritance
- Keep interfaces small

## Naming
- camelCase for private
- PascalCase for exported
- Short variable names in small scopes
```

---

## Adding a New OpenCode Agent

Create `.opencode/agents/security-auditor.md`:

```markdown
# Security Auditor Agent

You are a security-focused code auditor. Your job is to find vulnerabilities.

## Focus Areas
- SQL injection
- XSS vulnerabilities
- Authentication weaknesses
- Secrets in code
- Insecure dependencies

## Response Format
Always provide:
1. Severity (Critical/High/Medium/Low)
2. Location (file and line)
3. Description of the issue
4. Recommended fix with code example
```

**Usage:** `/opencode @security-auditor audit the authentication module`

### More Agent Examples

**Performance Analyzer:**
```markdown
# Performance Analyzer Agent

You analyze code for performance issues.

## Look For
- N+1 queries
- Unnecessary loops
- Memory leaks
- Blocking operations
- Missing caching opportunities

## Output
Provide benchmarks or estimates when possible.
```

**Migration Helper:**
```markdown
# Migration Helper Agent

You assist with code migrations and upgrades.

## Capabilities
- Upgrade dependency versions
- Migrate deprecated APIs
- Convert between frameworks
- Update syntax for new language versions

## Approach
1. Identify all affected files
2. Create migration plan
3. Apply changes incrementally
4. Verify each step
```

---

## Adding a New Skill

Create `.opencode/skills/database-review/SKILL.md`:

```markdown
# Database Review Skill

## Checklist
1. Check for SQL injection vulnerabilities
2. Verify indexes exist for queried columns
3. Look for N+1 query patterns
4. Ensure transactions are used appropriately
5. Check connection pooling configuration

## Output Format
- 🔴 Critical: Security or data integrity issue
- 🟡 Warning: Performance issue
- 🟢 Suggestion: Optimization opportunity
```

---

## Modifying Workflows

### Change Auto-Update Frequency

The `update-cursorrules.yml` workflow triggers on every push. To change to weekly:

```yaml
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday midnight
```

### Add New Workflow Trigger

To trigger on specific file types:

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'src/**/*.py'
      - 'tests/**/*.py'
```

### Disable a Workflow

Either delete the file or add a condition that's always false:

```yaml
jobs:
  example:
    if: false  # Disabled
    runs-on: ubuntu-latest
```

---

## Rule Priority System

Assign priorities (1–100) to prevent conflicts:

| Priority Range | Purpose | Examples |
|----------------|---------|----------|
| `10-20` | Global rules | Code quality, response quality |
| `30-60` | Domain rules | Python, JavaScript, frontend |
| `80-100` | Override/security rules | Security requirements |

**Example:**
```yaml
---
description: Security Requirements
globs: "**/*"
alwaysApply: true
priority: 100
---
```

Higher priority rules override lower priority rules. Security rules (priority 100) always win.

---

## Project-Specific Overrides

Use `200-project.mdc` to override defaults:

```markdown
---
description: Project-specific conventions
globs: "**/*"
alwaysApply: true
priority: 50
---

# My Project

## Override Python Rules
- Use camelCase (not snake_case) for this legacy codebase
- Docstrings are optional for private functions

## Domain Terms
- "Widget" = our core product entity
- "Flow" = user journey through the app
```
