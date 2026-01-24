# Housekeeping Skill

## Purpose

Find and report cleanup opportunities in the codebase.

## Scan Categories

### 1. Dead Code
- Unreachable code branches
- Unused functions and classes
- Commented-out code blocks
- Unused variables

**Commands:**
```bash
# Python
vulture .
ruff check . --select=F401,F841

# JavaScript/TypeScript
npx ts-prune
```

### 2. Unused Imports
- Imports never referenced
- Wildcard imports that could be specific

### 3. Outdated Dependencies
- Security vulnerabilities
- Major version updates available
- Deprecated packages

**Commands:**
```bash
# Python
pip list --outdated
pip-audit

# JavaScript
npm outdated
npm audit
```

### 4. Code Duplication
- Similar logic in multiple files
- Copy-pasted functions
- Patterns that could be abstracted

### 5. Large Files
- Files over 500 lines
- Classes with too many methods
- Functions over 50 lines

### 6. TODO/FIXME Items
- Forgotten tasks
- Temporary workarounds
- Technical debt markers

**Commands:**
```bash
grep -r "TODO\|FIXME\|HACK\|XXX" --include="*.py" .
```

## Priority Matrix

| Finding | Effort | Priority |
|---------|--------|----------|
| Security vulnerability | Any | 🔴 High |
| Dead code (large) | Low | 🟡 Medium |
| Unused imports | Low | 🟢 Low |
| Code duplication | Medium | 🟡 Medium |
| Outdated deps (non-security) | Low | 🟢 Low |

## Output Template

```markdown
## Housekeeping Report

**Date:** YYYY-MM-DD

### Summary
| Category | Count | Action |
|----------|-------|--------|
| ... | ... | ... |

### Findings
[Detailed list with file paths and recommendations]

### Quick Wins
[Low-effort, high-impact items]
```
