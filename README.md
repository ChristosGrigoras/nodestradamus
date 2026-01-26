# AImanac

**Your AI's almanac.** A template that unifies Cursor AI and OpenCode under shared rules, self-improving conventions, and dependency-aware impact analysis — so your codebase stays coherent across hundreds of AI interactions.

---

## TL;DR

| Problem | Solution |
|---------|----------|
| AI outputs vary wildly between sessions | **Shared rules** constrain both Cursor and OpenCode |
| AI breaks code it doesn't "see" | **Dependency graphs** enable impact analysis |
| You keep correcting the same mistakes | **Meta-generator** learns from your corrections |
| Rules drift across projects | **Portable template** travels with you |

---

## Features

### 🎯 Shared Rules
Both Cursor AI and OpenCode read from `.cursor/rules/`. Same conventions, same output quality, regardless of which tool or session.

### 📊 Dependency Graphs  
AI checks what calls a function before modifying it. Changes propagate correctly. No more isolated fixes that break dependent code.

### 🔄 Self-Improving Rules
Correct the AI 3+ times with the same pattern → it offers to create a rule. Your preferences become permanent.

### 🔒 Security by Default
Highest-priority rules prevent hardcoded secrets, SQL injection, and other vulnerabilities.

### 🤖 GitHub Automation
Comment `/opencode add login feature` on any issue → AI creates a PR. Auto-review PRs, auto-triage issues, monthly housekeeping.

---

## Quick Start

```bash
# Clone the template
git clone https://github.com/ChristosGrigoras/aimanac.git my-project
cd my-project

# Start fresh
rm -rf .git && git init

# Customize
# 1. Edit .cursor/rules/200-project.mdc (your project context)
# 2. Edit AGENTS.md (OpenCode instructions)
# 3. Replace this README

# Push to GitHub
git remote add origin https://github.com/YOU/YOUR_REPO.git
git add -A && git commit -m "Initial commit" && git push -u origin main
```

For detailed installation: [docs/installation.md](docs/installation.md)

---

## Project Structure

```
your-project/
├── .cursor/
│   ├── rules/                 # AI behavior rules
│   │   ├── 001-router.mdc     # Context detection
│   │   ├── 002-meta-generator.mdc  # Self-improvement
│   │   ├── 003-code-quality.mdc    # Universal standards
│   │   ├── 004-response-quality.mdc
│   │   ├── 005-security.mdc   # Security (priority 100)
│   │   ├── 100-python.mdc     # Language rules
│   │   ├── 200-project.mdc    # Your project context
│   │   ├── 301-testing.mdc    # Testing patterns
│   │   └── 305-dependency-graph.mdc  # Impact analysis
│   └── graph/                 # Dependency graphs
│
├── .opencode/
│   ├── agents/                # Custom AI personas
│   └── skills/                # Reusable behaviors
│
├── .github/workflows/
│   ├── opencode.yml           # /opencode comment trigger
│   ├── opencode-review.yml    # Auto-review PRs
│   ├── update-cursorrules.yml # Auto-update rules
│   ├── update-graphs.yml      # Auto-update graphs
│   └── housekeeping.yml       # Monthly cleanup
│
├── scripts/
│   ├── analyze_python_deps.py
│   └── analyze_git_cooccurrence.py
│
├── AGENTS.md                  # OpenCode reads this
└── opencode.json              # OpenCode config
```

---

## How It Works

### The Feedback Loop

```
You code in Cursor
       ↓
Cursor AI helps (follows .cursor/rules/)
       ↓
You correct the AI's output
       ↓
Meta-generator detects pattern (3+ corrections)
       ↓
AI offers to create a rule
       ↓
Future code follows your preference automatically
```

### GitHub Integration

```
Comment on issue: "/opencode add login feature"
       ↓
GitHub Action triggers OpenCode
       ↓
OpenCode reads AGENTS.md + rules
       ↓
Creates PR with changes
       ↓
You review and merge
```

### Impact Analysis

```
You: "Rename calculate_total to compute_total"
       ↓
AI checks dependency graph
       ↓
Finds 3 files that call this function
       ↓
Updates all 3 files in the same PR
```

---

## Rule Numbering

| Range | Purpose | Examples |
|-------|---------|----------|
| `001-099` | Core system | Router, meta-generator, security |
| `100-199` | Language | Python, JavaScript, Rust |
| `200-299` | Project | Your context, domain terms |
| `300-399` | Capabilities | Testing, API patterns, architecture |

---

## Multi-Project Usage

Core rules stay constant, language/project rules swap:

```
         Your Core Rules
         (code-quality, security, architecture)
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
Rust Chatbot    Python RAG    Next.js App
+ rust rules    + python      + typescript
+ project ctx   + project ctx + project ctx
```

Monday you build a Rust chatbot. Tuesday a Python RAG system. Both feel like the same disciplined developer wrote them.

---

## Generate Dependency Graphs

```bash
# Python
python scripts/analyze_python_deps.py src/ > .cursor/graph/python-deps.json

# TypeScript
npx madge src/ --json > .cursor/graph/ts-deps.json

# Git co-occurrence (files that change together)
python scripts/analyze_git_cooccurrence.py > .cursor/graph/co-occurrence.json
```

Graphs are auto-updated by the `update-graphs.yml` workflow.

---

## Usage Examples

### GitHub Comments

```
/opencode create user authentication with login and logout
/oc fix the TypeError in utils.py line 45
/opencode @reviewer check this PR for security issues
/opencode @housekeeper scan for dead code
```

### Cursor AI

Just open your project — rules apply automatically.

Trigger meta-generator:
```
Review recent changes and tell me what patterns you notice
```

---

## GitHub Setup

1. **Workflow permissions:** Settings → Actions → General → "Read and write permissions" ✓
2. **Create "General tasks" issue:** For auto-update comments
3. **Add secrets:** `DEEPSEEK_API_KEY` (or your model's key)

Full guide: [docs/github-setup.md](docs/github-setup.md)

---

## Security

The `/opencode` trigger only works for:
- Repository owner
- Organization members  
- Collaborators

Random users cannot trigger OpenCode on public repos.

---

## Documentation

| Topic | Link |
|-------|------|
| Installation | [docs/installation.md](docs/installation.md) |
| GitHub Setup | [docs/github-setup.md](docs/github-setup.md) |
| Workflows | [docs/workflows.md](docs/workflows.md) |
| Cursor Rules | [docs/cursor-rules.md](docs/cursor-rules.md) |
| OpenCode Config | [docs/opencode-config.md](docs/opencode-config.md) |
| Dependency Graphs | [docs/dependency-graphs.md](docs/dependency-graphs.md) |
| Customization | [docs/customization.md](docs/customization.md) |
| Rule Governance | [docs/governance.md](docs/governance.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |

---

## License

MIT — Use freely for any project.

---

## Credits

- [OpenCode](https://opencode.ai) — AI coding agent
- [Cursor](https://cursor.sh) — AI-powered IDE

---

*Template maintained by [@ChristosGrigoras](https://github.com/ChristosGrigoras)*
