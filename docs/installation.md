# Installation

## Step 1: Install Nodestradamus

```bash
# Install from PyPI
pip install nodestradamus

# Or install from source
git clone https://github.com/ChristosGrigoras/nodestradamus.git
cd nodestradamus && pip install -e .

# With FAISS for faster similarity search (optional, recommended for large codebases)
pip install nodestradamus[faiss]

# With Rust acceleration (optional, requires Rust toolchain)
pip install maturin
maturin develop --release
```

**Verify installation:**
```bash
nodestradamus --version
```

**Optional dependencies:**

| Extra | Install Command | Purpose |
|-------|-----------------|---------|
| `faiss` | `pip install nodestradamus[faiss]` | Faster similarity search for large codebases (>10K chunks) |
| `mistral` | `pip install nodestradamus[mistral]` | Mistral API embeddings (Codestral Embed) |
| `all` | `pip install nodestradamus[all]` | All optional dependencies |

**Environment (optional):** Nodestradamus loads a `.env` file when the package is imported. Copy `.env.example` to `.env` and set:

- `NODESTRADAMUS_EMBEDDING_PROVIDER` — `local` (default) or `mistral`
- `MISTRAL_API_KEY` — required for Mistral embeddings (Codestral Embed API)
- `NODESTRADAMUS_EMBEDDING_MODEL` — optional override for local provider (default: jinaai/jina-embeddings-v2-base-code)
- `NODESTRADAMUS_EMBEDDING_WORKERS` — parallel API workers for Mistral provider (default: 4, ~3.5x speedup)
- `NODESTRADAMUS_FAISS_THRESHOLD` — chunk count above which FAISS is used (default: 10000)

Without `.env`, the local embedding provider is used. With `NODESTRADAMUS_EMBEDDING_PROVIDER=mistral` and `MISTRAL_API_KEY` set, semantic search uses the Mistral API.

**Cache:** Nodestradamus stores analysis cache under `.nodestradamus/`:
- `graph.msgpack` — dependency graph
- `parse_cache.json` — parse cache
- `fingerprints.msgpack` — structural fingerprint index
- `nodestradamus.db` — SQLite database with chunk metadata and embeddings
- `embeddings.faiss` — FAISS vector index (rebuilt on-demand from SQLite)
- `embeddings.npz` — legacy format (still supported for backward compatibility)

Embeddings are updated **incrementally** — only changed chunks are re-embedded. Use the `manage_cache` tool in MCP to inspect or clear it.

**Lazy embedding with scopes:** For large monorepos, you can embed specific directories:

```python
# Embed only the auth module
compute_embeddings('/path/to/repo', scope='libs/auth')

# Later, embed another module - accumulates with previous
compute_embeddings('/path/to/repo', scope='libs/users')

# Search finds results from both scopes
semantic_search('/path/to/repo', query='authentication')
```

Each scoped embedding appends to SQLite. FAISS is rebuilt on-demand when you search, ensuring all chunks are searchable with consistent IDs.

## Step 2: Configure MCP Server

Add Nodestradamus to your Cursor MCP configuration (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "nodestradamus": {
      "command": "nodestradamus",
      "args": ["serve"]
    }
  }
}
```

## Step 3: Clone This Template

```bash
# Clone the template
git clone https://github.com/ChristosGrigoras/nodestradamus.git my-project

# Enter the directory
cd my-project

# Remove template's git history and start fresh
rm -rf .git
git init
```

## Step 4: Customize for Your Project

Edit these files:

1. **`.cursor/rules/200-project.mdc`** - Add your project's specific context
2. **`AGENTS.md`** - Update project description
3. **`README.md`** - Replace with your project's documentation

## Step 5: Push to Your GitHub Repository

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add -A
git commit -m "Initial commit from Nodestradamus template"
git push -u origin main
```

---

## Adding to Existing Projects

If you're adding this template to an **existing codebase** (not starting fresh), follow these steps.

### Step 1: Copy the Template Files

From the template repo, copy these to your existing project:

```bash
# Clone template temporarily
git clone https://github.com/ChristosGrigoras/nodestradamus.git /tmp/nodestradamus-template

# Copy Cursor rules (required)
cp -r /tmp/nodestradamus-template/.cursor/rules/ your-project/.cursor/rules/

# Copy GitHub workflows (optional)
cp -r /tmp/nodestradamus-template/.github/workflows/ your-project/.github/workflows/

# Cleanup
rm -rf /tmp/nodestradamus-template
```

**If you already have `.cursor/rules/`:**

```bash
# Option A: Backup existing rules first
mv your-project/.cursor/rules/ your-project/.cursor/rules-backup/
cp -r /tmp/nodestradamus-template/.cursor/rules/ your-project/.cursor/rules/

# Option B: Copy only new files (keep your existing rules)
cp -rn /tmp/nodestradamus-template/.cursor/rules/* your-project/.cursor/rules/

# Option C: Copy template rules to a separate folder for manual merge
cp -r /tmp/nodestradamus-template/.cursor/rules/ your-project/.cursor/rules-template/
# Then ask Cursor: "Merge rules-template/ into my existing rules/"
```

**Minimal install (Cursor only, no existing rules):**
```bash
# Just the rules folder
cp -r /tmp/nodestradamus-template/.cursor/rules/ your-project/.cursor/rules/
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
