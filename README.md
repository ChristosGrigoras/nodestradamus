# Self-Evolving AI Workflow Template

A reusable template for setting up **Cursor AI** (local IDE assistant) + **OpenCode** (GitHub AI agent) with self-improving rules that learn from your corrections.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [GitHub Configuration](#github-configuration)
- [GitHub Actions Workflows Explained](#github-actions-workflows-explained)
- [Usage Examples](#usage-examples)
- [Cursor AI Rules Explained](#cursor-ai-rules-explained)
- [OpenCode Configuration Explained](#opencode-configuration-explained)
- [The Meta-Generator: Self-Improving Rules](#the-meta-generator-self-improving-rules)
- [Rule Governance](#rule-governance)
- [Customization Guide](#customization-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

This template creates a unified AI development workflow where:

| Tool | Where | Purpose |
|------|-------|---------|
| **Cursor AI** | Your local IDE | Real-time coding assistance, follows `.cursor/rules/` |
| **OpenCode** | GitHub Actions | Automated code changes via issue/PR comments |

Both tools share the same coding standards, so you get consistent behavior locally and on GitHub.

---

## How It Works

### The Feedback Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   You write code in Cursor                                      │
│         ↓                                                       │
│   Cursor AI helps (follows .cursor/rules/)                      │
│         ↓                                                       │
│   You correct the AI's output                                   │
│         ↓                                                       │
│   Meta-generator detects pattern (3+ similar corrections)       │
│         ↓                                                       │
│   AI suggests: "Should I create a rule for this?"               │
│         ↓                                                       │
│   You approve → Rule added to .cursor/rules/                    │
│         ↓                                                       │
│   Future code follows your preference automatically             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### GitHub Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   You comment on GitHub issue: "/opencode add login feature"    │
│         ↓                                                       │
│   GitHub Action triggers OpenCode                               │
│         ↓                                                       │
│   OpenCode reads AGENTS.md + opencode.json for instructions     │
│         ↓                                                       │
│   OpenCode creates PR with the changes                          │
│         ↓                                                       │
│   You review and merge                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
your-project/
│
├── .cursor/
│   └── rules/                      # Cursor AI reads these files
│       ├── 001-router.mdc          # Routes to correct rules based on context
│       ├── 002-meta-generator.mdc  # Detects patterns in your corrections
│       ├── 003-code-quality.mdc    # Universal coding standards
│       ├── 004-response-quality.mdc# How AI should communicate
│       ├── 100-python.mdc          # Python-specific rules
│       └── 200-project.mdc         # Your project's specific context
│
├── .opencode/
│   ├── agents/                     # Custom AI agents for OpenCode
│   │   ├── reviewer.md             # Code review agent
│   │   ├── documenter.md           # Documentation agent
│   │   └── housekeeper.md          # Cleanup/maintenance agent
│   │
│   └── skills/                     # Reusable behaviors for agents
│       ├── code-review/SKILL.md    # How to do code reviews
│       ├── git-commit/SKILL.md     # How to write commit messages
│       ├── housekeeping/SKILL.md   # How to find cleanup tasks
│       └── rules-sync/SKILL.md     # How to update AI rules
│
├── .github/
│   └── workflows/
│       ├── opencode.yml            # Main workflow: /opencode triggers
│       ├── update-cursorrules.yml  # Auto-updates rules on push
│       ├── housekeeping.yml        # Monthly cleanup scan
│       ├── opencode-review.yml     # Auto-review PRs when opened
│       └── issue-triage.yml        # Auto-triage new issues
│
├── AGENTS.md                       # OpenCode reads this for instructions
├── opencode.json                   # OpenCode configuration
└── README.md                       # This file
```

---

## Installation

### Step 1: Install OpenCode Locally

OpenCode can run both locally (terminal) and on GitHub (Actions). Install locally first:

```bash
# Download and install OpenCode
curl -fsSL https://opencode.ai/install | bash
```

This installs OpenCode to `~/.opencode/bin/`. The installer adds it to your PATH in `~/.bashrc`.

**Activate in current terminal:**
```bash
source ~/.bashrc
```

**Or manually add to PATH:**
```bash
export PATH="$HOME/.opencode/bin:$PATH"
```

**Verify installation:**
```bash
opencode --version
# Should output: opencode version X.X.X
```

**If `opencode: command not found`:**
```bash
# Check if binary exists
ls -la ~/.opencode/bin/

# Create symlink in a standard PATH location
mkdir -p ~/.local/bin
ln -sf ~/.opencode/bin/opencode ~/.local/bin/opencode

# Add ~/.local/bin to PATH if not already
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Configure OpenCode API Key (Optional)

For local terminal usage, you may need an API key:

```bash
# Set API key (get from https://opencode.ai/settings)
export OPENCODE_API_KEY="your-api-key-here"

# Or add to ~/.bashrc for persistence
echo 'export OPENCODE_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### Step 3: Clone This Template

```bash
# Clone the template
git clone https://github.com/ChristosGrigoras/opencode_plan.git my-project

# Enter the directory
cd my-project

# Remove template's git history and start fresh
rm -rf .git
git init
```

### Step 4: Customize for Your Project

Edit these files:

1. **`.cursor/rules/200-project.mdc`** - Add your project's specific context
2. **`AGENTS.md`** - Update project description for OpenCode
3. **`README.md`** - Replace with your project's documentation

### Step 5: Push to Your GitHub Repository

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add -A
git commit -m "Initial commit from AI workflow template"
git push -u origin main
```

---

## GitHub Configuration

### Step 1: Repository Workflow Permissions

OpenCode needs permission to create commits and PRs. Configure this:

1. **Go to your repository on GitHub**
2. **Click Settings** (gear icon, top right)
3. **Left sidebar → Actions → General**
4. **Scroll to "Workflow permissions"**
5. **Select these options:**

```
☑ Read and write permissions
  (Workflows have read and write permissions in the repository for all scopes)

☑ Allow GitHub Actions to create and approve pull requests
```

6. **Click "Save"**

**Without these settings, you'll see errors like:**
- `User opencode-agent[bot] does not have write permissions`
- `fatal: unable to access '...': The requested URL returned error: 403`

### Step 2: Create the General Tasks Issue

The `update-cursorrules.yml` workflow needs an issue to post comments to:

1. **Go to Issues tab → New Issue**
2. **Title:** `General tasks`
3. **Body:** 
   ```
   Ongoing tasks and automated updates.
   
   OpenCode will post analysis comments here when the codebase changes.
   ```
4. **Submit**

**Note the issue number.** Then set it as a repository variable:

1. Go to **Settings → Secrets and variables → Actions → Variables**
2. Click **"New repository variable"**
3. Name: `GENERAL_TASKS_ISSUE`, Value: your issue number (e.g., `1`)

If not set, the workflow defaults to issue `#1`.

### Step 3: Create a Personal Access Token (PAT)

For local Git operations and enhanced GitHub API access:

#### Option A: Fine-Grained Token (Recommended)

1. **Go to:** https://github.com/settings/tokens?type=beta
2. **Click "Generate new token"**
3. **Configure:**

| Setting | Value |
|---------|-------|
| Token name | `opencode-workflow` |
| Expiration | 90 days (or custom) |
| Repository access | "Only select repositories" → Select your repo |

4. **Permissions (expand each section):**

| Category | Permission | Access Level |
|----------|------------|--------------|
| **Repository permissions** | | |
| Contents | Read and write |
| Issues | Read and write |
| Pull requests | Read and write |
| Metadata | Read-only (auto-selected) |

5. **Click "Generate token"**
6. **Copy the token immediately** (starts with `github_pat_`)

#### Option B: Classic Token (Simpler)

1. **Go to:** https://github.com/settings/tokens
2. **Click "Generate new token (classic)"**
3. **Configure:**

| Setting | Value |
|---------|-------|
| Note | `opencode-workflow` |
| Expiration | 90 days |
| Scopes | ☑ `repo` (full control of private repositories) |

4. **Click "Generate token"**
5. **Copy the token** (starts with `ghp_`)

### Step 4: Configure Git Credentials Locally

Store your PAT so Git doesn't ask for it repeatedly:

```bash
# Enable credential storage
git config --global credential.helper store

# Set your GitHub username
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clone or push once - enter PAT as password when prompted
git push origin main
# Username: your-github-username
# Password: [paste your PAT here]

# Credentials are now stored in ~/.git-credentials
```

**Alternative: Include username in remote URL:**
```bash
git remote set-url origin https://YOUR_USERNAME@github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Step 5: Add PAT to GitHub Secrets (Optional)

If you want workflows to use your PAT instead of `GITHUB_TOKEN`:

1. **Go to:** Repository → Settings → Secrets and variables → Actions
2. **Click "New repository secret"**
3. **Add:**

| Name | Value |
|------|-------|
| `OPENCODE_PAT` | Your PAT (paste the full token) |

4. **Update workflow to use it:**
```yaml
env:
  GITHUB_TOKEN: ${{ secrets.OPENCODE_PAT }}
```

---

## GitHub Actions Workflows Explained

This template includes three workflows:

### Workflow 1: `opencode.yml` - Main OpenCode Trigger

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
    # Only run if comment contains trigger
    if: |
      contains(github.event.comment.body, '/oc') ||
      contains(github.event.comment.body, '/opencode')
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
| `fetch-depth: 0` | Get full git history (helps AI understand context) |
| `git config` | Identity for commits (required or you get "empty ident" error) |
| `GITHUB_TOKEN` | Auto-provided secret for GitHub API access |
| `use_github_token: true` | Tell OpenCode to use the token |

### Workflow 2: `update-cursorrules.yml` - Auto-Update Rules

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

### Workflow 3: `housekeeping.yml` - Monthly Cleanup

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

### Workflow 4: `opencode-review.yml` - Auto-Review PRs

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
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          model: anthropic/claude-sonnet-4-20250514
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

### Workflow 5: `issue-triage.yml` - Auto-Triage New Issues

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
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        with:
          model: anthropic/claude-sonnet-4-20250514
          prompt: |
            Review this issue. If there's a clear fix or relevant docs:
            - Provide documentation links
            - Add error handling guidance for code examples
            - Suggest a potential solution if straightforward
            Otherwise, acknowledge the issue and ask for more details if needed.
```

**Why filter by account age:** New GitHub accounts (<30 days) are more likely to be spam bots. This filter reduces noise while still handling legitimate users.

---

## Usage Examples

### Using OpenCode via GitHub Comments

On any issue or pull request, comment:

```
/opencode create a user authentication module with login and logout functions
```

```
/oc fix the TypeError in utils.py line 45
```

```
/opencode add comprehensive docstrings to all functions in api.py
```

```
/opencode refactor the database connection to use connection pooling
```

### Using Custom Agents

```
/opencode @reviewer check the code in this PR for security issues
```

```
/opencode @documenter generate API documentation for the routes module
```

```
/opencode @housekeeper scan the codebase for dead code and unused imports
```

### Using Cursor AI Locally

Just open your project in Cursor IDE. The rules in `.cursor/rules/` are automatically applied.

**Meta-commands you can use in Cursor chat:**

```
@rules              # Show active rules
@context python     # Force Python context
@clean              # Minimal context mode
```

---

## Cursor AI Rules Explained

### Rule Numbering Convention

| Range | Purpose | Example |
|-------|---------|---------|
| `001-099` | Core system rules | Router, meta-generator |
| `100-199` | Language-specific | Python, JavaScript, Go |
| `200-299` | Project-specific | Your project context |

### File: `001-router.mdc`

**Purpose:** Detects what you're working on and loads the right rules.

**How it works:**
- Sees `*.py` file → loads Python rules
- Sees `Dockerfile` → loads DevOps rules
- Sees `*.md` file → loads documentation rules

### File: `002-meta-generator.mdc`

**Purpose:** The self-improvement engine. Watches for patterns in your corrections.

**Trigger condition:** You correct AI-generated code 3+ times with the same pattern.

**Example:**
1. AI generates function without error handling
2. You add error handling
3. AI generates another function without error handling
4. You add error handling again
5. Third time...
6. AI says: "I noticed you keep adding error handling. Should I create a rule for this?"
7. You say "yes"
8. Rule is added to `100-python.mdc`

### File: `003-code-quality.mdc`

**Purpose:** Universal coding standards that apply to all languages.

**Contains:**
- Naming conventions (descriptive, no abbreviations)
- Function guidelines (single responsibility, <50 lines)
- Comment standards (explain why, not what)
- Error handling (explicit, meaningful messages)
- Security basics (no hardcoded secrets)

### File: `004-response-quality.mdc`

**Purpose:** How the AI should communicate with you.

**Key behaviors:**
- Be direct, don't ask "Do you want me to..."
- Provide complete, working solutions
- Don't over-explain obvious concepts
- Never create `final_v2.py` - fix the original

### File: `100-python.mdc`

**Purpose:** Python-specific conventions.

**Contains:**
- Type hints requirement
- Docstring format (Google style with Args, Returns, Raises)
- snake_case naming
- f-string preference
- Function length guidelines

### File: `200-project.mdc`

**Purpose:** Your project's specific context.

**You should add:**
- Project description
- Key directories and their purpose
- Domain-specific terminology
- External services/APIs used
- Team conventions

---

## OpenCode Configuration Explained

### File: `AGENTS.md`

This is what OpenCode reads first. It should contain:

```markdown
# Project Name

Brief description of what this project does.

## Project Structure
- `src/` - Main source code
- `tests/` - Test files
- `docs/` - Documentation

## Code Standards
- Your specific conventions
- Technologies used

## Important Notes
- Any context OpenCode needs
```

### File: `opencode.json`

Configuration file that controls OpenCode's behavior:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": [
    ".cursor/rules/003-code-quality.mdc",    // Share Cursor's rules!
    ".cursor/rules/004-response-quality.mdc"
  ],
  "agent": {
    "build": {
      "model": "opencode/big-pickle",
      "temperature": 0.3
    }
  },
  "permission": {
    "edit": "allow",
    "bash": {
      "git status": "allow",
      "git diff": "allow"
    }
  }
}
```

**Key insight:** The `instructions` array points to Cursor's rule files. This means both AIs follow the same standards!

### Custom Agents (`.opencode/agents/`)

Create specialized AI personas:

**`reviewer.md`:**
```markdown
# Code Reviewer Agent

You are a senior code reviewer. Focus on:
- Code quality and readability
- Security vulnerabilities
- Performance issues
- Best practices

Be constructive but thorough. Always explain why something should change.
```

### Skills (`.opencode/skills/*/SKILL.md`)

Reusable behaviors that agents can use:

**`code-review/SKILL.md`:**
```markdown
# Code Review Skill

## Checklist
1. Check for security issues (SQL injection, XSS, etc.)
2. Verify error handling
3. Look for performance bottlenecks
4. Ensure tests exist
5. Check documentation

## Output Format
- 🔴 Critical: Must fix
- 🟡 Warning: Should fix
- 🟢 Suggestion: Nice to have
```

---

## The Meta-Generator: Self-Improving Rules

### How Pattern Detection Works

The meta-generator in `002-meta-generator.mdc` observes:

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

---

## Rule Governance

Long-term success with AI rules relies on governance, not just writing rules.

### Rule Priority System

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

### Rule Ownership

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

### Quarterly Rule Review

Set calendar reminders to revisit rules:

**Review checklist:**
- [ ] Code style changes since last review?
- [ ] Architectural updates to document?
- [ ] Outdated instructions to remove?
- [ ] New patterns to add?
- [ ] Rules that are ignored frequently?

### Versioning for Major Rule Updates

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

### Deprecation Strategy

Mark outdated rules before removing:

```markdown
# ⚠️ DEPRECATED: Will be removed in Q2 2026
# Replaced by: 101-typescript.mdc
```

This prevents confusion during transitions.

### Treat Rule Changes as Code Changes

All rule edits should pass through:
1. Pull request with clear description
2. Review by rule owner
3. Visual test of generation quality
4. Merge only after approval

A minor rule error can influence hundreds of files — treat rules seriously.

---

## Customization Guide

### Adding a New Language

Create `.cursor/rules/101-javascript.mdc`:

```markdown
---
description: JavaScript coding conventions
globs: "**/*.{js,jsx,ts,tsx}"
alwaysApply: false
---

# JavaScript

## Conventions
- Use const/let, never var
- Prefer arrow functions
- Use async/await over .then()
- Destructure when possible

## Naming
- camelCase for variables and functions
- PascalCase for components and classes
- UPPER_SNAKE for constants
```

### Adding a New OpenCode Agent

Create `.opencode/agents/security-auditor.md`:

```markdown
# Security Auditor Agent

You are a security-focused code auditor. Your job is to find vulnerabilities.

## Focus Areas
- SQL injection
- XSS vulnerabilities
- Authentication weaknesses
- Secrets in code
- Insecure dependencies

## Response Format
Always provide:
1. Severity (Critical/High/Medium/Low)
2. Location (file and line)
3. Description of the issue
4. Recommended fix with code example
```

Use it: `/opencode @security-auditor audit the authentication module`

### Modifying the Auto-Update Workflow

The `update-cursorrules.yml` workflow triggers on every push. To change frequency:

```yaml
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday midnight
```

---

## Troubleshooting

### OpenCode Not Responding to Comments

1. **Check workflow ran:** Go to Actions tab, look for failed runs
2. **Check permissions:** Settings → Actions → General → Workflow permissions
3. **Check trigger:** Comment must start with `/opencode` or `/oc`

### "User opencode-agent[bot] does not have write permissions"

1. Go to Settings → Actions → General
2. Set "Workflow permissions" to "Read and write permissions"
3. Check "Allow GitHub Actions to create and approve pull requests"

### "fatal: empty ident name not allowed"

The workflow is missing git config. Ensure these lines exist in your workflow:

```yaml
- name: Configure Git
  run: |
    git config --global user.name "opencode-agent[bot]"
    git config --global user.email "opencode-agent[bot]@users.noreply.github.com"
```

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

### Meta-Generator Not Suggesting Rules

- Need 3+ similar corrections
- Corrections must follow a pattern
- Try asking: "Review recent changes and tell me what patterns you notice"

---

## License

MIT - Use this template freely for any project.

---

## Credits

- [OpenCode](https://opencode.ai) - AI coding agent
- [Cursor](https://cursor.sh) - AI-powered IDE

---

*Template maintained by [@ChristosGrigoras](https://github.com/ChristosGrigoras)*
