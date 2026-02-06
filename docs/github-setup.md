# GitHub Configuration

## Step 1: Repository Workflow Permissions

GitHub Actions workflows need permission to create commits and PRs. Configure this:

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
- `fatal: unable to access '...': The requested URL returned error: 403`
- Permission denied when pushing from workflows

## Step 2: Create a Personal Access Token (PAT)

For local Git operations and enhanced GitHub API access:

### Option A: Fine-Grained Token (Recommended)

1. **Go to:** https://github.com/settings/tokens?type=beta
2. **Click "Generate new token"**
3. **Configure:**

| Setting | Value |
|---------|-------|
| Token name | `nodestradamus-workflow` |
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

### Option B: Classic Token (Simpler)

1. **Go to:** https://github.com/settings/tokens
2. **Click "Generate new token (classic)"**
3. **Configure:**

| Setting | Value |
|---------|-------|
| Note | `nodestradamus-workflow` |
| Expiration | 90 days |
| Scopes | ☑ `repo` (full control of private repositories) |

4. **Click "Generate token"**
5. **Copy the token** (starts with `ghp_`)

## Step 3: Configure Git Credentials Locally

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

## Step 4: Add Secrets to GitHub (For Workflows)

If your workflows need API keys or tokens:

1. **Go to:** Repository → Settings → Secrets and variables → Actions
2. **Click "New repository secret"**
3. **Add your secrets:**

| Name | Description |
|------|-------------|
| `DEEPSEEK_API_KEY` | API key for DeepSeek models (if using) |

4. **Reference in workflows:**
```yaml
env:
  DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
```

## Step 5: GitHub MCP Server (Optional)

The [GitHub MCP Server](https://github.com/github/github-mcp-server) lets Cursor (or other MCP hosts) talk to GitHub from chat/composer: list repos, read issues/PRs, search code, create branches, etc.

### Prerequisites

- Cursor IDE v0.48.0+ (for remote server)
- A [GitHub Personal Access Token](https://github.com/settings/personal-access-tokens/new) with at least `repo` scope (and optionally `read:org`, `notifications`, etc., depending on what you want the AI to do)
- For **local** setup only: [Docker](https://www.docker.com/) installed and running

### Option A: Remote server (recommended)

Uses GitHub’s hosted server; no Docker. Cursor must support Streamable HTTP (v0.48.0+).

1. Open your **global** MCP config: `~/.cursor/mcp.json` (create it if it doesn’t exist).
2. Add or merge this (use your real PAT; never commit this file):

```json
{
  "mcpServers": {
    "github": {
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer YOUR_GITHUB_PAT"
      }
    }
  }
}
```

3. Replace `YOUR_GITHUB_PAT` with your [GitHub PAT](https://github.com/settings/tokens).
4. Save, then **restart Cursor**.
5. In **Settings → Tools & Integrations → MCP**, confirm `github` shows a green dot.

### Option B: Local server (Docker)

Runs the server in a container; useful if you prefer not to use the remote endpoint or need custom toolsets.

1. Ensure Docker is running.
2. Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_PAT"
      }
    }
  }
}
```

3. Replace `YOUR_GITHUB_PAT` with your token.
4. Save and restart Cursor.

### Configuration scope

- **Global (all projects):** `~/.cursor/mcp.json`
- **This project only:** `.cursor/mcp.json` in the repo root (same JSON shape under `mcpServers`)

Use project-specific config if you want GitHub MCP only for this repo; otherwise global is simpler.

### How to use it

1. In Cursor, open **Chat** or **Composer** (Agent mode).
2. The model can call GitHub MCP tools when your question implies GitHub (repos, issues, PRs, code search, etc.).
3. Example prompts:
   - “List my GitHub repositories”
   - “List open pull requests in owner/repo”
   - “Search code in repo X for function Y”
   - “Create a new branch called feature/foo in owner/repo”
   - “Get the diff for PR #42 in owner/repo”

The server exposes many tools (repos, issues, pull requests, actions, code search, etc.). You don’t need to name the tool; describe the task and the model will choose the right one (and may ask for owner/repo or PR number if needed).

### Troubleshooting

| Issue | What to try |
|-------|--------------|
| Remote: “Streamable HTTP” or connection errors | Use Cursor v0.48.0+ and check firewall/proxy. |
| Auth errors | Confirm PAT has `repo` (and other scopes you need); token not expired. |
| Local: “Docker not found” or image errors | Start Docker Desktop; run `docker pull ghcr.io/github/github-mcp-server`. |
| Tools never appear | Restart Cursor; ensure `mcp.json` is valid JSON and `github` has a green dot in MCP settings. |
| PAT in config | Never commit `mcp.json` with a real token; use global config and restrict file permissions, e.g. `chmod 600 ~/.cursor/mcp.json`. |
