# Troubleshooting

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

## Graph and Cache Issues

### Graph Generation Fails

Nodestradamus builds the dependency graph via the `analyze_deps` tool (or `nodestradamus analyze`). If analysis fails:

```bash
# Check Python version
python --version  # Should be 3.12+

# Run analyzer from CLI (optional)
nodestradamus analyze /path/to/repo

# Or use MCP: call project_scout first, then analyze_deps with suggested_ignores
```

### Stale or Missing Graph / Embeddings

The graph and embeddings live under **`.nodestradamus/`** (in the repo or in the workspace cache).

1. **Inspect cache:** Use the `manage_cache` tool with `mode="info"` and `repo_path` set to your repo.
2. **Clear and rebuild:** Use `manage_cache` with `mode="clear"`, then run `analyze_deps` again (and `semantic_analysis` with `mode="embeddings"` if you use search).
3. If the AI doesn't use graphs, ensure the Nodestradamus MCP server is enabled and the AI is calling `analyze_deps` / `get_impact` / `analyze_graph`; the graph is used by those tools, not by a separate `.cursor/graph/` folder.

### Co-occurrence Results Empty

- Need at least 50 commits for meaningful data
- Run with more history: `git fetch --unshallow` then call `analyze_cooccurrence` again

### Embedding Generation Slow or Stalls on Large Repos

For very large codebases (100k+ files, like WebKit), embedding generation can take 1-2 hours and may encounter problematic chunks:

1. **Use streaming mode**: Set `streaming=true` when calling `semantic_analysis` with `mode="embeddings"`. This processes chunks in batches and shows progress.
2. **Check for skipped chunks**: The Mistral provider automatically skips chunks that fail (e.g., binary data, excessively long strings). Look for `WARNING: Skipping chunk` messages in logs.
3. **First query is slow**: Initial semantic search loads ~300k embeddings from disk (~45s). Subsequent queries are faster. In MCP server mode, embeddings stay in memory.
4. **Cost estimation**: Mistral `codestral-embed` costs ~$0.08 per 1M tokens. A 200k-chunk codebase uses ~50M tokens ≈ $4.

### Embedding API Errors (400 Bad Request)

If you see repeated `400 Bad Request` errors from the Mistral API:

1. **Automatic fallback**: Nodestradamus uses binary search to isolate problematic chunks and skips only the bad ones.
2. **Common causes**: Binary files, very long lines, unusual character encodings, or test fixtures with malformed data.
3. **Check skipped count**: The final embedding result reports how many chunks were skipped.
4. **Use local provider for testing**: Set `provider="local"` to use sentence-transformers (slower but no API errors).

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

## MCP Server Issues

### Nodestradamus MCP Server Not Starting

```bash
# Check installation
nodestradamus --version

# Try running directly
nodestradamus serve

# Check for port conflicts
lsof -i :8080
```

### MCP Tools Not Available in Cursor

1. Check `.cursor/mcp.json` configuration
2. Restart Cursor
3. Check MCP server logs for errors

---

## Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| Rules not loading | Reload Window (`Ctrl/Cmd+Shift+P`) |
| Workflow not triggering | Check trigger conditions and file path |
| Graph not updating | Run `analyze_deps`; clear cache with `manage_cache` if stale |
| Meta-generator silent | Need 3+ similar corrections |
| MCP server not found | Run `pip install nodestradamus` |
| Embeddings stalled | Check logs for 400 errors; bad chunks auto-skipped |
| First semantic search slow | Normal—embeddings loading from disk (~45s for 300k chunks) |