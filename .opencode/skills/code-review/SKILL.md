# Code Review Skill

## Purpose

Systematically review code changes for quality, security, and maintainability.

## Checklist

### Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation on all external data
- [ ] SQL queries use parameterized statements
- [ ] No path traversal vulnerabilities
- [ ] Authentication/authorization checks present

### Code Quality
- [ ] Functions have single responsibility
- [ ] No code duplication (DRY)
- [ ] Meaningful variable/function names
- [ ] Error handling is explicit
- [ ] No god classes or methods

### Testing
- [ ] Critical paths have tests
- [ ] Edge cases are covered
- [ ] Tests are isolated and independent
- [ ] No flaky tests introduced

### Performance
- [ ] No N+1 query patterns
- [ ] Pagination for large datasets
- [ ] No blocking operations in async code
- [ ] Appropriate caching strategy

## Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| 🔴 Critical | Security risk, data loss | Block merge |
| 🟡 Warning | Bug, bad pattern | Request changes |
| 🟢 Suggestion | Style, minor improvement | Optional |

## Output Template

```markdown
## Review: [PR Title]

**Verdict:** ✅ Approve / ⚠️ Changes Requested / ❌ Blocked

### Issues
[List with severity, file:line, description, fix]

### Positives
[What's done well]
```
