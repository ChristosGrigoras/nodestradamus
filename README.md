# Self-Evolving AI Workflow Template

A reusable template for setting up **Cursor AI** + **OpenCode** with self-improving rules.

## What's Included

```
├── .cursor/rules/           # Cursor AI configuration
│   ├── 001-router.mdc       # Context detection & routing
│   ├── 002-meta-generator.mdc  # Self-improving rules (detects patterns)
│   ├── 003-code-quality.mdc    # Universal coding standards
│   ├── 004-response-quality.mdc # AI communication style
│   ├── 100-python.mdc       # Python-specific conventions
│   └── 200-project.mdc      # Project-specific context
├── .opencode/               # OpenCode configuration
│   ├── agents/              # Custom agents (reviewer, documenter, housekeeper)
│   └── skills/              # Reusable skills (code-review, git-commit, etc.)
├── .github/workflows/
│   ├── opencode.yml         # Trigger via /opencode comments
│   ├── update-cursorrules.yml  # Auto-update rules on push
│   └── housekeeping.yml     # Monthly cleanup tasks
├── AGENTS.md                # OpenCode main instructions
├── opencode.json            # OpenCode config (shares Cursor rules)
└── README.md                # This file
```

## Key Features

### 🔄 Self-Improving Rules (Meta-Generator)
When you correct AI-generated code 3+ times with the same pattern:
1. AI detects the pattern
2. Suggests creating a rule
3. You approve → rule is added
4. Future code follows your preference

### 🤝 Shared Standards
Both Cursor and OpenCode use the same quality rules:
- `opencode.json` references `.cursor/rules/003-code-quality.mdc`
- Consistent behavior across local and GitHub AI

### 🤖 GitHub Automation
- `/opencode <task>` or `/oc <task>` on issues/PRs
- Auto-updates rules on codebase changes
- Monthly housekeeping scans

## Quick Start

### 1. Install OpenCode
```bash
curl -fsSL https://opencode.ai/install | bash
```

### 2. Clone This Template
```bash
git clone https://github.com/ChristosGrigoras/opencode_plan.git my-project
cd my-project
rm -rf .git && git init
```

### 3. Push to Your Repo
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add -A && git commit -m "Initial commit from template"
git push -u origin main
```

### 4. Configure GitHub
1. **Settings → Actions → General**
   - Workflow permissions: "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

2. **Create a "General tasks" issue** (issue #4 is used by auto-update workflow)

## Using OpenCode via GitHub

Comment on any issue or PR:
```
/opencode add user authentication to the API
/oc fix the bug in utils.py
/opencode refactor the database module
```

OpenCode will create a PR with the changes.

## Cursor AI Rules

| File | Purpose |
|------|---------|
| `001-router.mdc` | Detects context (Python, DevOps, etc.) and loads relevant rules |
| `002-meta-generator.mdc` | Observes corrections, suggests new rules |
| `003-code-quality.mdc` | Universal standards (naming, functions, errors) |
| `004-response-quality.mdc` | How AI should communicate |
| `100-python.mdc` | Python-specific (type hints, docstrings) |
| `200-project.mdc` | Your project's specific context |

## OpenCode Agents

| Agent | Trigger | Purpose |
|-------|---------|---------|
| `@reviewer` | Code review requests | Quality, security, performance checks |
| `@documenter` | Documentation tasks | Generate/update docs |
| `@housekeeper` | Cleanup tasks | Find dead code, outdated deps |

## Customization

1. **Add language rules**: Create `101-javascript.mdc`, `102-go.mdc`, etc.
2. **Add project context**: Update `200-project.mdc` with your specifics
3. **Create new agents**: Add files to `.opencode/agents/`
4. **Define skills**: Add to `.opencode/skills/`

## License

MIT - Use this template for any project.
