# Dependency Graphs

Dependency graphs help AI understand code relationships and perform **impact analysis** before making changes.

## What Graphs Provide

When you ask AI to modify a function, it can check the graph to see:
- **Downstream impact:** What calls this function?
- **Upstream dependencies:** What does this function call?
- **Co-occurrence:** What files tend to change together?

This prevents AI from making isolated changes that break dependent code.

---

## Graph Types

| Type | What It Tracks | Generator |
|------|----------------|-----------|
| **Static analysis** | Function calls, imports, inheritance | Language-specific tools |
| **Co-occurrence** | Files that change together in git | `scripts/analyze_git_cooccurrence.py` |

---

## Quick Start

### Python Dependencies

```bash
python scripts/analyze_python_deps.py src/ > .cursor/graph/python-deps.json
```

### TypeScript Dependencies

```bash
npm install -g madge
madge src/ --json > .cursor/graph/ts-deps.json
```

### Git Co-occurrence

```bash
python scripts/analyze_git_cooccurrence.py > .cursor/graph/co-occurrence.json
```

---

## How AI Uses Graphs

When `.cursor/graph/` contains graph files, the `305-dependency-graph.mdc` rule tells AI to:

1. **Check for downstream impacts** before modifying functions
2. **List affected files** in responses
3. **Update related code** proactively

### Example AI Response

```
I'll update the `calculate_total()` function. Based on the dependency graph:

**Downstream impact (3 files):**
- `orders/checkout.py` - calls `calculate_total()`
- `reports/summary.py` - calls `calculate_total()`
- `tests/test_pricing.py` - tests `calculate_total()`

I'll update all affected files to match the new signature.
```

---

## CI Integration

Keep graphs updated automatically with this workflow:

```yaml
# .github/workflows/update-graphs.yml
name: Update Dependency Graphs

on:
  push:
    branches: [main]
    paths: ['src/**', '*.py', '*.ts']

jobs:
  update-graphs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 100
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Generate graphs
        run: |
          python scripts/analyze_python_deps.py src/ > .cursor/graph/python-deps.json || true
          python scripts/analyze_git_cooccurrence.py > .cursor/graph/co-occurrence.json
      
      - name: Commit graphs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .cursor/graph/
          git commit -m "chore: update dependency graphs" || true
          git push
```

---

## Graph File Format

### Python Dependencies (`python-deps.json`)

```json
{
  "src/utils.py": {
    "imports": ["src/config.py", "src/database.py"],
    "functions": {
      "calculate_total": {
        "calls": ["get_tax_rate", "apply_discount"],
        "called_by": ["checkout.process_order", "reports.generate_summary"]
      }
    }
  }
}
```

### Co-occurrence (`co-occurrence.json`)

```json
{
  "src/auth.py": {
    "frequently_changed_with": [
      {"file": "src/users.py", "correlation": 0.85},
      {"file": "tests/test_auth.py", "correlation": 0.92}
    ]
  }
}
```

---

## Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **Config files** | JSON/YAML dependencies are implicit | AI searches for usage at change time |
| **Dynamic calls** | `getattr()`, reflection won't appear | Document in comments |
| **Cross-repo** | Dependencies across repositories not tracked | Document in `200-project.mdc` |

When graphs are missing or incomplete, AI falls back to searching the codebase.

---

## Scripts Reference

### `analyze_python_deps.py`

Analyzes Python source files for:
- Import statements
- Function definitions
- Function calls
- Class inheritance

```bash
python scripts/analyze_python_deps.py <directory> [--output json|dot]
```

### `analyze_git_cooccurrence.py`

Analyzes git history to find files that frequently change together:
- Looks at last 500 commits by default
- Calculates correlation scores
- Identifies change patterns

```bash
python scripts/analyze_git_cooccurrence.py [--commits 500] [--threshold 0.3]
```

---

For graph format details, see `.cursor/graph/README.md`.
