# Cursor Rules

This directory contains AI behavior rules for Cursor IDE. Rules define coding conventions, response quality standards, and project-specific context that guide AI-assisted development.

## Rule Numbering Scheme

| Range | Purpose | Examples |
|-------|---------|----------|
| `000-099` | Core system | Router, meta-generator, code quality, security |
| `100-199` | Language-specific | Python, JavaScript, Rust, Go |
| `200-299` | Project-specific | Your project context, domain terms |
| `300-399` | Capabilities | Testing, API patterns, architecture |

## Current Rules

| Rule | Description | Always Apply |
|------|-------------|--------------|
| `000-onboarding.mdc` | Adapts rules to existing codebases | No |
| `001-router.mdc` | Context detection and rule routing | Yes |
| `002-meta-generator.mdc` | Self-improving rule generation | No |
| `003-code-quality.mdc` | Universal code standards | Yes |
| `004-response-quality.mdc` | AI response guidelines | Yes |
| `005-security.mdc` | Security requirements (priority 100) | Yes |
| `100-python.mdc` | Python coding conventions | No |
| `102-git.mdc` | Git workflow conventions | No |
| `200-project.mdc` | Project-specific context | Yes |
| `301-testing.mdc` | Testing patterns and practices | No |
| `302-validation.mdc` | Data validation patterns | No |
| `303-api-patterns.mdc` | API design conventions | No |
| `304-architecture.mdc` | Architectural patterns | No |
| `305-dependency-graph.mdc` | Impact analysis awareness | No |
| `306-project-management.mdc` | Task breakdown standards | No |

## Rule Format

Each rule file uses MDC (Markdown with YAML frontmatter) format:

```markdown
---
description: Brief description of the rule
globs: "**/*.py"  # File patterns this rule applies to
alwaysApply: false  # Whether to always load this rule
priority: 50  # Optional: higher = more important (security = 100)
---

# Rule Title

## Section
- Directive 1
- Directive 2
```

## Creating New Rules

1. Choose the appropriate number range for your rule type
2. Create a file with format `NNN-name.mdc`
3. Add required frontmatter fields (`description`, `globs`, `alwaysApply`)
4. Keep directives concise (<40 tokens each)
5. Run validation: `python scripts/validate_rules.py`

## Validation

Check your rules for issues:

```bash
# Validate all rules
python scripts/validate_rules.py .cursor/rules/

# Check for conflicts
python scripts/detect_rule_conflicts.py .cursor/rules/
```

## Priority System

When rules conflict, priority determines which wins:

- **10-20**: Global rules (code quality, response quality)
- **30-60**: Domain rules (language-specific, project-specific)
- **80-100**: Override rules (security requirements)

Security rules (priority 100) always take precedence.

## Best Practices

1. **Be specific**: Target rules to relevant file patterns
2. **Stay concise**: Each directive should be <40 tokens
3. **Avoid redundancy**: Check existing rules before creating new ones
4. **Test changes**: Run validation after modifying rules
5. **Document decisions**: Explain *why* in comments, not just *what*

## Troubleshooting

**Rules not applying?**
- Reload Cursor window: `Cmd/Ctrl+Shift+P` → "Developer: Reload Window"
- Check glob pattern matches your files
- Verify `alwaysApply` setting

**Conflicting rules?**
- Run conflict detection: `python scripts/detect_rule_conflicts.py`
- Use priority to resolve: higher priority wins
- Use `200-project.mdc` for project-specific overrides
