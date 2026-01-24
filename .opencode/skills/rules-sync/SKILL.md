---
name: rules-sync
description: Synchronize and update AI rules based on codebase patterns
license: MIT
---

## What I Do

- Analyze codebase for coding patterns and conventions
- Update `.cursor/rules/*.mdc` files to match reality
- Keep `AGENTS.md` in sync with project structure
- Suggest new rules based on repeated corrections

## When to Use Me

Use this skill when:
- After major refactoring
- When adding new file types or patterns
- Monthly rule review
- After onboarding feedback

## Analysis Steps

1. **Scan file structure** - Update project directories in rules
2. **Analyze code patterns** - Extract naming conventions, import styles
3. **Check for consistency** - Rules match actual code
4. **Identify gaps** - Patterns not covered by rules

## Files to Update

| File | Purpose |
|------|---------|
| `.cursor/rules/200-project.mdc` | Project structure, key directories |
| `AGENTS.md` | Project overview, external references |
| `opencode.json` | Shared instructions list |

## Rule Generation Template

```markdown
---
description: [What this rule covers]
globs: ["**/*.ext"]
alwaysApply: false
---

# [Context Name]

## Conventions
- [Pattern 1 observed in codebase]
- [Pattern 2 observed in codebase]

## Avoid
- [Anti-pattern seen in corrections]
```
