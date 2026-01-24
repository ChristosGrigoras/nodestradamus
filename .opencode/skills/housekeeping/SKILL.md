---
name: housekeeping
description: Analyze codebase for cleanup opportunities - dead code, unused files, organization issues
license: MIT
---

## What I Do

- Scan for backup/temp files that should be deleted
- Find dead code and unused imports
- Identify files not touched in months
- Check for duplicate or versioned files (`v1`, `final`, `fixed`)
- Suggest organization improvements

## When to Use Me

Use this skill when:
- Before major releases (cleanup sprint)
- Monthly maintenance reviews
- Onboarding new team members (clean house first)
- Repository is feeling "cluttered"

## Scan Commands

```bash
# Find backup/temp files
find . -name "*.bak" -o -name "*.tmp" -o -name "*~" -o -name "*.old"

# Find files not modified in 6 months
find . -type f -mtime +180 -not -path "./.git/*"

# Find potential duplicates (similar names)
find . -type f -name "*final*" -o -name "*v[0-9]*" -o -name "*fixed*" -o -name "*backup*"

# Find empty files
find . -type f -empty -not -path "./.git/*"

# Find large files (>1MB)
find . -type f -size +1M -not -path "./.git/*"
```

## Cleanup Checklist

1. **Immediate Delete**: `.bak`, `.tmp`, `~` files, `.DS_Store`
2. **Review Before Delete**: `*final*`, `*v2*`, `*fixed*` files
3. **Archive to Branch**: Old features, deprecated code
4. **Update .gitignore**: Prevent future clutter

## Archive Strategy

For files worth keeping but not in main:

```bash
# Create archive branch
git checkout -b archive/old-features
git add archived/
git commit -m "chore: archive deprecated features"
git push origin archive/old-features
git checkout main
```
