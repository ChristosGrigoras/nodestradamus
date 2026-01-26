# GitHub Actions Workflows

This template includes five workflows for AI-assisted development.

## Workflow 1: `opencode.yml` - Main OpenCode Trigger

**Location:** `.github/workflows/opencode.yml`

**Triggers when:** Someone comments `/opencode` or `/oc` on an issue or PR.

**What it does:**
1. Checks out the repository
2. Configures Git identity for commits
3. Runs OpenCode with the comment as instructions
4. OpenCode creates a PR with changes

**Full workflow file:**
```yaml
name: opencode

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]

jobs:
  opencode:
    # Only allow trusted users (owner, org members, collaborators)
    # This prevents random users from triggering OpenCode on public repos
    if: |
      !startsWith(github.event.comment.user.login, 'opencode') &&
      (github.event.comment.author_association == 'OWNER' ||
       github.event.comment.author_association == 'MEMBER' ||
       github.event.comment.author_association == 'COLLABORATOR') &&
      (contains(github.event.comment.body, '/oc') ||
       contains(github.event.comment.body, '/opencode'))
    runs-on: ubuntu-latest
    
    # Required permissions
    permissions:
      id-token: write        # For OpenCode authentication
      contents: write        # To create commits
      pull-requests: write   # To create PRs
      issues: write          # To comment on issues
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v6
        with:
          fetch-depth: 0     # Full history for better context
      
      - name: Configure Git
        run: |
          git config --global user.name "opencode-agent[bot]"
          git config --global user.email "opencode-agent[bot]@users.noreply.github.com"
      
      - name: Run OpenCode
        uses: anomalyco/opencode/github@latest
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          model: opencode/big-pickle
          use_github_token: true
```

**Key settings explained:**

| Setting | Purpose |
|---------|---------|
| `author_association` check | **Security:** Only allows repo owner, org members, or collaborators to trigger. Prevents random users from running OpenCode on public repos. |
| `fetch-depth: 0` | Get full git history (helps AI understand context) |
| `git config` | Identity for commits (required or you get "empty ident" error) |
| `GITHUB_TOKEN` | Auto-provided secret for GitHub API access |
| `use_github_token: true` | Tell OpenCode to use the token |

**Security Note:** The `author_association` filter is critical for public repositories. Without it, any GitHub user could comment `/opencode` and trigger code changes. Valid values: `OWNER`, `MEMBER`, `COLLABORATOR`, `CONTRIBUTOR`, `NONE`.

---

## Workflow 2: `update-cursorrules.yml` - Auto-Update Rules

**Location:** `.github/workflows/update-cursorrules.yml`

**Triggers when:** Code is pushed to main (excluding changes to `.cursor/` itself).

**What it does:**
1. Analyzes recent commits
2. Posts a comment to the General Tasks issue
3. That comment triggers OpenCode to update rules

**Why this design:** GitHub Actions can't directly trigger other Actions, so we use an issue comment as a bridge.

**Full workflow file:**
```yaml
name: Update Cursor Rules

on:
  push:
    branches: [main]
    paths-ignore:
      - '.cursor/**'      # Don't trigger on rule changes
      - '.cursorrules'

jobs:
  update-rules:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v6
        with:
          fetch-depth: 0
      
      - name: Configure Git
        run: |
          git config --global user.name "opencode-agent[bot]"
          git config --global user.email "opencode-agent[bot]@users.noreply.github.com"
      
      - name: Create analysis issue comment
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GENERAL_TASKS_ISSUE: ${{ vars.GENERAL_TASKS_ISSUE || '1' }}
        run: |
          # Get last 5 commits
          CHANGES=$(git log -5 --oneline --no-merges)
          
          # Post comment to trigger OpenCode (uses GENERAL_TASKS_ISSUE variable)
          curl -s -X POST \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            -d "{\"body\": \"/opencode Analyze recent commits and update .cursor/rules/*.mdc files if needed. Recent changes:\n\n\`\`\`\n${CHANGES}\n\`\`\`\n\nUpdate rules to reflect any new patterns, conventions, or file structures observed.\"}" \
            "https://api.github.com/repos/${{ github.repository }}/issues/${GENERAL_TASKS_ISSUE}/comments"
```

---

## Workflow 3: `housekeeping.yml` - Monthly Cleanup

**Location:** `.github/workflows/housekeeping.yml`

**Triggers when:** First day of each month at midnight UTC.

**What it does:**
1. Creates a new housekeeping issue
2. Asks the `@housekeeper` agent to scan for cleanup tasks

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
            -d "{\"title\": \"Housekeeping: $DATE\", \"body\": \"/opencode @housekeeper Scan the codebase and report:\n\n1. Dead code and unused imports\n2. Outdated dependencies\n3. Files that could be archived\n4. Potential refactoring opportunities\n\nCreate a summary with recommendations.\"}" \
            "https://api.github.com/repos/${{ github.repository }}/issues"
```

**To trigger manually:**
1. Go to Actions tab
2. Select "Monthly Housekeeping"
3. Click "Run workflow"

---

## Workflow 4: `opencode-review.yml` - Auto-Review PRs

**Location:** `.github/workflows/opencode-review.yml`

**Triggers when:** A pull request is opened, updated, or marked ready for review.

**What it does:**
1. Automatically reviews the PR for code quality issues
2. Looks for potential bugs
3. Suggests improvements
4. Posts review comments

**Full workflow file:**
```yaml
name: opencode-review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
      pull-requests: read
      issues: read
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v6
        with:
          persist-credentials: false
      
      - name: Run OpenCode Review
        uses: anomalyco/opencode/github@latest
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          model: deepseek/deepseek-chat
          use_github_token: true
          prompt: |
            Review this pull request:
            - Check for code quality issues
            - Look for potential bugs
            - Suggest improvements
            - Verify error handling is present
            - Check for missing tests
```

**Note:** If no `prompt` is provided for `pull_request` events, OpenCode defaults to reviewing the PR.

---

## Workflow 5: `issue-triage.yml` - Auto-Triage New Issues

**Location:** `.github/workflows/issue-triage.yml`

**Triggers when:** A new issue is opened.

**What it does:**
1. Filters spam by checking account age (>30 days)
2. Reviews the issue content
3. Provides helpful responses or documentation links
4. Adds appropriate labels

**Full workflow file:**
```yaml
name: Issue Triage

on:
  issues:
    types: [opened]

jobs:
  triage:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: write
      pull-requests: write
      issues: write
    
    steps:
      - name: Check account age (spam filter)
        id: check
        uses: actions/github-script@v7
        with:
          script: |
            const user = await github.rest.users.getByUsername({
              username: context.payload.issue.user.login
            });
            const created = new Date(user.data.created_at);
            const days = (Date.now() - created) / (1000 * 60 * 60 * 24);
            return days >= 30;
          result-encoding: string
      
      - name: Checkout repository
        if: steps.check.outputs.result == 'true'
        uses: actions/checkout@v6
        with:
          persist-credentials: false
      
      - name: Run OpenCode Triage
        if: steps.check.outputs.result == 'true'
        uses: anomalyco/opencode/github@latest
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        with:
          model: deepseek/deepseek-chat
          prompt: |
            Review this issue. If there's a clear fix or relevant docs:
            - Provide documentation links
            - Add error handling guidance for code examples
            - Suggest a potential solution if straightforward
            Otherwise, acknowledge the issue and ask for more details if needed.
```

**Why filter by account age:** New GitHub accounts (<30 days) are more likely to be spam bots. This filter reduces noise while still handling legitimate users.

---

## Workflow 6: `update-graphs.yml` - Dependency Graph Updates

**Location:** `.github/workflows/update-graphs.yml`

**Triggers when:** Code is pushed to main that affects source files.

**What it does:**
1. Generates Python dependency graph
2. Generates git co-occurrence graph
3. Commits updated graphs

See [Dependency Graphs](dependency-graphs.md) for details.
