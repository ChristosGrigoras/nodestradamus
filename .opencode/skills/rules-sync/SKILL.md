# Rules Sync Skill

## Purpose

Analyze codebase changes and update `.cursor/rules/` to reflect new patterns.

## When to Trigger

- After significant code changes pushed to main
- When new frameworks or libraries are added
- When coding patterns change
- Monthly review cycle

## Analysis Process

1. **Review recent commits**
   ```bash
   git log -10 --oneline --no-merges
   git diff HEAD~10..HEAD --stat
   ```

2. **Identify new patterns**
   - New file types or directories
   - New frameworks or libraries
   - Changed naming conventions
   - New architectural patterns

3. **Check existing rules**
   - Do current rules cover new patterns?
   - Are any rules outdated?
   - Are there conflicts between rules?

4. **Propose updates**
   - Add new patterns to appropriate rule files
   - Update outdated instructions
   - Remove obsolete rules

## Rule File Structure

| Range | Purpose |
|-------|---------|
| 001-099 | Core system (router, meta-gen, security) |
| 100-199 | Language-specific (Python, JS, Go) |
| 200-299 | Project-specific context |
| 300-399 | Capability-specific (testing, API, arch) |

## Update Checklist

- [ ] Router triggers updated for new contexts
- [ ] Language rules reflect new frameworks
- [ ] Project context updated with new directories
- [ ] Capability rules cover new patterns
- [ ] No conflicting instructions between rules

## Output Template

```markdown
## Rules Sync Report

**Analyzed commits:** X
**Period:** YYYY-MM-DD to YYYY-MM-DD

### Detected Changes
- [List of significant changes]

### Proposed Rule Updates

#### [Rule file name]
**Add:**
- [New patterns to add]

**Update:**
- [Existing patterns to modify]

**Remove:**
- [Outdated patterns to remove]

### No Action Needed
- [Rules that are still accurate]
```

## Reminder

After updating rules, remind user:
> ⚠️ **Reload required:** `Cmd/Ctrl+Shift+P` → "Developer: Reload Window"
