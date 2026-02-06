# GitHub Actions Workflows

This template includes workflows for automated development tasks.

## Workflow 1: `update-cursorrules.yml` - Auto-Update Rules

**Location:** `.github/workflows/update-cursorrules.yml`

**Triggers when:** Code is pushed to main (excluding changes to `.cursor/` itself).

**What it does:**
1. Analyzes recent commits
2. Posts a comment to the General Tasks issue
3. Triggers rule updates based on code changes

**Why this design:** Keeps Cursor rules in sync with codebase evolution.

---

## Workflow 2: `housekeeping.yml` - Monthly Cleanup

**Location:** `.github/workflows/housekeeping.yml`

**Triggers when:** First day of each month at midnight UTC.

**What it does:**
1. Creates a new housekeeping issue
2. Scans for cleanup tasks (dead code, unused imports, etc.)

**Full workflow file:**
```yaml
name: Monthly Housekeeping

on:
  schedule:
    - cron: '0 0 1 * *'   # First day of month, midnight UTC
  workflow_dispatch:       # Allow manual trigger

jobs:
  housekeeping:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      issues: write
    
    steps:
      - name: Create housekeeping issue
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          DATE=$(date +%Y-%m)
          curl -s -X POST \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            -d "{\"title\": \"Housekeeping: $DATE\", \"body\": \"Monthly codebase review:\n\n1. Dead code and unused imports\n2. Outdated dependencies\n3. Files that could be archived\n4. Potential refactoring opportunities\"}" \
            "https://api.github.com/repos/${{ github.repository }}/issues"
```

**To trigger manually:**
1. Go to Actions tab
2. Select "Monthly Housekeeping"
3. Click "Run workflow"

---

## Workflow 3: `issue-triage.yml` - Auto-Triage New Issues

**Location:** `.github/workflows/issue-triage.yml`

**Triggers when:** A new issue is opened.

**What it does:**
1. Filters spam by checking account age (>30 days)
2. Reviews the issue content
3. Adds appropriate labels

**Why filter by account age:** New GitHub accounts (<30 days) are more likely to be spam bots. This filter reduces noise while still handling legitimate users.

---

## Workflow 4: `update-graphs.yml` - Dependency Graph Updates

**Location:** `.github/workflows/update-graphs.yml`

**Triggers when:** Code is pushed to main that affects source files.

**What it does:**
1. Generates Python dependency graph
2. Generates git co-occurrence graph
3. Commits updated graphs

See [Dependency Graphs](dependency-graphs.md) for details.

---

## Workflow 5: `validate-rules.yml` - Rule Validation

**Location:** `.github/workflows/validate-rules.yml`

**Triggers when:** Changes to `.cursor/rules/` are pushed.

**What it does:**
1. Validates rule file syntax
2. Checks for conflicts between rules
3. Reports issues as workflow annotations
