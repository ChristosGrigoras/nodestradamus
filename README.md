# Self-Evolving AI Workflow Template

A reusable template for setting up **Cursor AI** (local IDE assistant) + **OpenCode** (GitHub AI agent) with self-improving rules that learn from your corrections.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [GitHub Configuration](#github-configuration)
- [Usage Examples](#usage-examples)
- [Cursor AI Rules Explained](#cursor-ai-rules-explained)
- [OpenCode Configuration Explained](#opencode-configuration-explained)
- [The Meta-Generator: Self-Improving Rules](#the-meta-generator-self-improving-rules)
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
│       └── housekeeping.yml        # Monthly cleanup scan
│
├── AGENTS.md                       # OpenCode reads this for instructions
├── opencode.json                   # OpenCode configuration
└── README.md                       # This file
```

---

## Installation

### Step 1: Install OpenCode Locally (Optional)

If you want to use OpenCode from your terminal:

```bash
curl -fsSL https://opencode.ai/install | bash
source ~/.bashrc  # or restart your terminal
```

Verify installation:
```bash
opencode --version
```

### Step 2: Clone This Template

```bash
# Clone the template
git clone https://github.com/ChristosGrigoras/opencode_plan.git my-project

# Enter the directory
cd my-project

# Remove template's git history and start fresh
rm -rf .git
git init
```

### Step 3: Customize for Your Project

Edit these files:

1. **`.cursor/rules/200-project.mdc`** - Add your project's specific context
2. **`AGENTS.md`** - Update project description for OpenCode
3. **`README.md`** - Replace with your project's documentation

### Step 4: Push to Your GitHub Repository

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add -A
git commit -m "Initial commit from AI workflow template"
git push -u origin main
```

---

## GitHub Configuration

### Required: Repository Permissions

1. Go to your repo on GitHub
2. Navigate to **Settings → Actions → General**
3. Under "Workflow permissions":
   - Select **"Read and write permissions"**
   - Check **"Allow GitHub Actions to create and approve pull requests"**
4. Click **Save**

### Required: Create a General Tasks Issue

The `update-cursorrules.yml` workflow posts to issue #4. Create it:

1. Go to **Issues → New Issue**
2. Title: `General tasks`
3. Body: `Ongoing tasks and automated updates`
4. Submit

**Important:** This should be issue #4. If it's a different number, update the workflow:

```yaml
# In .github/workflows/update-cursorrules.yml, line 41
"https://api.github.com/repos/${{ github.repository }}/issues/YOUR_ISSUE_NUMBER/comments"
```

### Optional: Fine-Grained Personal Access Token

For enhanced permissions, create a PAT:

1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Create token with:
   - **Repository access:** Your specific repo
   - **Permissions:** Contents (Read and write), Issues (Read and write), Pull requests (Read and write)
3. Add to repo: Settings → Secrets and variables → Actions → New repository secret
   - Name: `OPENCODE_PAT`
   - Value: Your token

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
