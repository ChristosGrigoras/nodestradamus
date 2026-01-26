# Rule Governance

Long-term success with AI rules relies on governance, not just writing rules.

## Rule Priority System

Assign priorities (1–100) to prevent conflicts:

| Priority Range | Purpose | Examples |
|----------------|---------|----------|
| `10-20` | Global rules | Code quality, response quality |
| `30-60` | Domain rules | Python, JavaScript, frontend |
| `80-100` | Override/security rules | Security requirements, critical constraints |

**How it works:**
- Higher priority rules override lower priority rules
- Security rules (priority 100) always win
- Domain rules (30-60) override global defaults

**Example in rule file:**
```yaml
---
description: Security Requirements
globs: "**/*"
alwaysApply: true
priority: 100
---
```

---

## Rule Ownership

Every rule file should have a maintainer responsible for:
- Updates when conventions change
- Enforcement of the rule
- Reviewing proposed changes
- Resolving conflicts with other rules

Consider adding a comment at the top of each rule:
```markdown
# Owner: @username
# Last reviewed: 2026-01
```

---

## Quarterly Rule Review

Set calendar reminders to revisit rules:

**Review checklist:**
- [ ] Code style changes since last review?
- [ ] Architectural updates to document?
- [ ] Outdated instructions to remove?
- [ ] New patterns to add?
- [ ] Rules that are ignored frequently?

---

## Versioning for Major Rule Updates

When rules change significantly:
1. Create versioned backups: `100-python-v1.mdc`, `100-python-v2.mdc`
2. Maintain a changelog in the rule file
3. Communicate changes to your team

**Example changelog header:**
```markdown
# Changelog
# v2 (2026-01): Added async/await patterns
# v1 (2025-10): Initial Python rules
```

---

## Deprecation Strategy

Mark outdated rules before removing:

```markdown
# ⚠️ DEPRECATED: Will be removed in Q2 2026
# Replaced by: 101-typescript.mdc
```

This prevents confusion during transitions.

---

## Treat Rule Changes as Code Changes

All rule edits should pass through:
1. Pull request with clear description
2. Review by rule owner
3. Visual test of generation quality
4. Merge only after approval

A minor rule error can influence hundreds of files — treat rules seriously.

---

## The Meta-Generator

The meta-generator in `002-meta-generator.mdc` observes your corrections and suggests new rules.

### How Pattern Detection Works

1. **What you change:** When you modify AI-generated code
2. **How often:** Counting similar corrections
3. **The pattern:** What the correction has in common

### Threshold

**3+ similar corrections = potential rule**

### What Gets Detected

| Your Corrections | Detected Pattern | Suggested Rule |
|------------------|------------------|----------------|
| Adding `try/except` to 3 functions | Missing error handling | "Always include error handling for external calls" |
| Adding type hints to 3 functions | Missing type hints | "Use type hints for all function parameters" |
| Changing `print()` to `logger.info()` 3 times | Using print for logging | "Use logging module instead of print statements" |

### Triggering the Meta-Generator

In Cursor, ask:
```
Review the recent changes to [file] and tell me what patterns you notice in how I've been modifying the code.
```

If patterns are detected, Cursor will offer to create a rule.
