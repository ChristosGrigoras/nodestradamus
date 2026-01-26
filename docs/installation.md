# Installation

## Step 1: Install OpenCode Locally

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

## Step 2: Configure OpenCode API Key (Optional)

For local terminal usage, you may need an API key:

```bash
# Set API key (get from https://opencode.ai/settings)
export OPENCODE_API_KEY="your-api-key-here"

# Or add to ~/.bashrc for persistence
echo 'export OPENCODE_API_KEY="your-api-key-here"' >> ~/.bashrc
```

## Step 3: Clone This Template

```bash
# Clone the template
git clone https://github.com/ChristosGrigoras/aimanac.git my-project

# Enter the directory
cd my-project

# Remove template's git history and start fresh
rm -rf .git
git init
```

## Step 4: Customize for Your Project

Edit these files:

1. **`.cursor/rules/200-project.mdc`** - Add your project's specific context
2. **`AGENTS.md`** - Update project description for OpenCode
3. **`README.md`** - Replace with your project's documentation

## Step 5: Push to Your GitHub Repository

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add -A
git commit -m "Initial commit from AI workflow template"
git push -u origin main
```

---

## Adding to Existing Projects

If you're adding this template to an **existing codebase** (not starting fresh), follow these steps.

### Step 1: Copy the Template Files

From the template repo, copy these to your existing project:

```bash
# Clone template temporarily
git clone https://github.com/ChristosGrigoras/aimanac.git /tmp/ai-template

# Copy Cursor rules (required)
cp -r /tmp/ai-template/.cursor/rules/ your-project/.cursor/rules/

# Copy OpenCode config (optional - for GitHub integration)
cp /tmp/ai-template/opencode.json your-project/
cp /tmp/ai-template/AGENTS.md your-project/
cp -r /tmp/ai-template/.github/workflows/ your-project/.github/workflows/
cp -r /tmp/ai-template/.opencode/ your-project/.opencode/

# Cleanup
rm -rf /tmp/ai-template
```

**If you already have `.cursor/rules/`:**

```bash
# Option A: Backup existing rules first
mv your-project/.cursor/rules/ your-project/.cursor/rules-backup/
cp -r /tmp/ai-template/.cursor/rules/ your-project/.cursor/rules/

# Option B: Copy only new files (keep your existing rules)
cp -rn /tmp/ai-template/.cursor/rules/* your-project/.cursor/rules/

# Option C: Copy template rules to a separate folder for manual merge
cp -r /tmp/ai-template/.cursor/rules/ your-project/.cursor/rules-template/
# Then ask Cursor: "Merge rules-template/ into my existing rules/"
```

**Minimal install (Cursor only, no existing rules):**
```bash
# Just the rules folder
cp -r /tmp/ai-template/.cursor/rules/ your-project/.cursor/rules/
```

### Step 2: Adapt Rules to Your Conventions

The default rules are prescriptive — they define conventions like:
- `snake_case` for Python functions
- Google-style docstrings
- pytest patterns

Your existing codebase may use different conventions. Without adaptation, AI suggestions will clash with your code style.

Open Cursor in your project and ask it to adapt:

```
Adapt rules to my codebase
```

Or any of these trigger phrases:
- "Compare rules to my code"
- "Onboard this project"
- "Check rule conflicts"
- "Analyze my conventions"

### What Happens

Cursor will:

1. **Sample your codebase** — Read 5-10 representative files
2. **Detect conventions** — Naming, docstrings, testing patterns, etc.
3. **Compare against rules** — Identify conflicts
4. **Report findings** — Show a clear table of conflicts
5. **Propose changes** — Suggest specific rule edits
6. **Apply on approval** — Update `200-project.mdc` with your conventions

### Example Output

```markdown
## Codebase Analysis

| Area | Your Codebase | Current Rules | Conflict? |
|------|---------------|---------------|-----------|
| Naming | camelCase | snake_case | ⚠️ Yes |
| Docstrings | None | Google style | ⚠️ Yes |
| Testing | unittest | pytest | ⚠️ Yes |
| Imports | Relative | No preference | ✅ OK |

### Recommended: Update 200-project.mdc

- Use camelCase for functions (override 100-python.mdc)
- Docstrings optional for internal functions
- Use unittest.TestCase patterns
```

### After Onboarding

Reload Cursor to apply the adapted rules:

**Windows/Linux:** `Ctrl+Shift+P` → "Developer: Reload Window"  
**macOS:** `Cmd+Shift+P` → "Developer: Reload Window"

Your AI assistant will now follow your project's conventions, not the template defaults.
