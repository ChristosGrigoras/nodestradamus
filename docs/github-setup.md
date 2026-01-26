# GitHub Configuration

## Step 1: Repository Workflow Permissions

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

## Step 2: Create the General Tasks Issue

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

## Step 3: Create a Personal Access Token (PAT)

For local Git operations and enhanced GitHub API access:

### Option A: Fine-Grained Token (Recommended)

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

### Option B: Classic Token (Simpler)

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

## Step 4: Configure Git Credentials Locally

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

## Step 5: Add PAT to GitHub Secrets (Optional)

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
