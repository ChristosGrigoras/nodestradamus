# Troubleshooting

## OpenCode Issues

### OpenCode Not Responding to Comments

1. **Check workflow ran:** Go to Actions tab, look for failed runs
2. **Check permissions:** Settings → Actions → General → Workflow permissions
3. **Check trigger:** Comment must contain `/opencode` or `/oc`
4. **Check user permissions:** Only OWNER, MEMBER, or COLLABORATOR can trigger

### "User opencode-agent[bot] does not have write permissions"

1. Go to Settings → Actions → General
2. Set "Workflow permissions" to "Read and write permissions"
3. Check "Allow GitHub Actions to create and approve pull requests"
4. Click Save

### "fatal: empty ident name not allowed"

The workflow is missing git config. Ensure these lines exist in your workflow:

```yaml
- name: Configure Git
  run: |
    git config --global user.name "opencode-agent[bot]"
    git config --global user.email "opencode-agent[bot]@users.noreply.github.com"
```

### OpenCode Creates Empty PRs

- Check that DEEPSEEK_API_KEY or other model API keys are set in repository secrets
- Verify the model name is correct in the workflow file
- Check workflow logs for API errors

---

## Cursor Issues

### Cursor Not Following Rules

1. Rules must be in `.cursor/rules/` directory
2. File extension must be `.mdc`
3. Check the frontmatter is valid:
   ```yaml
   ---
   description: What this rule does
   globs: "**/*.py"
   alwaysApply: false
   ---
   ```
4. **Reload window after changes:** `Cmd/Ctrl+Shift+P` → "Developer: Reload Window"

### Rule Changes Not Taking Effect

After creating or modifying a rule:
1. Save the file (`Cmd/Ctrl+S`)
2. Reload window: `Cmd/Ctrl+Shift+P` → "Developer: Reload Window"
3. Start a new chat session (rules may not apply to existing chats)

### Cursor Ignores Specific Rules

- Check rule `globs` pattern matches your files
- Check `alwaysApply` setting
- Higher priority rules may be overriding
- Rule may conflict with another rule

### AI Modifies Rules Incorrectly

If Cursor's agent edits `.mdc` files, changes may not save correctly:
1. Manually save the file (`Cmd/Ctrl+S`)
2. Reload window
3. Verify the changes are correct

---

## Meta-Generator Issues

### Meta-Generator Not Suggesting Rules

- Need 3+ similar corrections
- Corrections must follow a detectable pattern
- Try asking: "Review recent changes and tell me what patterns you notice"

### Suggested Rule Is Wrong

- Reject the suggestion
- Manually create the rule with correct logic
- The meta-generator learns from your corrections

---

## Graph Issues

### Graph Generation Fails

```bash
# Check Python version
python --version  # Should be 3.8+

# Check script exists
ls scripts/analyze_python_deps.py

# Run with verbose output
python scripts/analyze_python_deps.py src/ --verbose
```

### AI Doesn't Use Graphs

1. Check `.cursor/graph/` contains graph files
2. Check `305-dependency-graph.mdc` rule exists
3. Reload Cursor window
4. Verify graph files are valid JSON

### Co-occurrence Graph Is Empty

- Need at least 50 commits for meaningful data
- Run with more history: `git fetch --unshallow` then regenerate
- Lower the threshold: `--threshold 0.2`

---

## GitHub Actions Issues

### Workflow Never Triggers

1. Check workflow file is in `.github/workflows/`
2. Check YAML syntax is valid
3. Check trigger conditions match your event
4. Check workflow isn't disabled in Actions settings

### Workflow Fails with Permission Error

Add explicit permissions to the job:

```yaml
jobs:
  my-job:
    permissions:
      contents: write
      pull-requests: write
      issues: write
```

### Secrets Not Available

1. Check secret name matches exactly (case-sensitive)
2. Secrets aren't available in forked repos by default
3. Check secret is set at correct level (repo vs org)

---

## Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| Rules not loading | Reload Window (`Ctrl/Cmd+Shift+P`) |
| OpenCode no permission | Settings → Actions → Workflow permissions |
| Workflow not triggering | Check trigger conditions and file path |
| Graph not updating | Run script manually, check output |
| Meta-generator silent | Need 3+ similar corrections |
