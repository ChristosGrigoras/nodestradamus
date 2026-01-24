# OpenCode Plan Project

A demonstration project showcasing OpenCode GitHub integration for automated code generation and task management.

## About OpenCode

OpenCode is an AI-powered development assistant that integrates with GitHub through slash commands. It can automatically create, modify, and refactor code based on natural language instructions provided in GitHub comments.

## How to Use OpenCode

### Triggering OpenCode via GitHub Comments

To trigger OpenCode, use the `/opencode` command in GitHub issue comments or pull request review comments:

```
/opencode [instruction]
```

**Examples:**
- `/opencode create a hello.py file that prints Hello World`
- `/opencode add a function to calculate the sum of two numbers`
- `/opencode refactor the authentication logic`
- `/opencode write tests for the user service`

### Supported Commands

OpenCode understands various types of instructions:
- **File creation**: Create new files with specific functionality
- **Code modification**: Update existing code with new features
- **Refactoring**: Improve code structure and organization
- **Testing**: Generate unit tests and integration tests
- **Documentation**: Add comments and documentation
- **Debugging**: Fix bugs and resolve issues

## Project Structure

This project currently contains:

- **README.md** - Project documentation (this file)
- **.github/workflows/opencode.yml** - GitHub Action workflow for OpenCode integration

### Python Files

Currently, there are no Python files in the project. The project is set up as a demonstration of OpenCode capabilities and can be populated with Python files using OpenCode commands.

## Setup Instructions

### Prerequisites

1. A GitHub repository
2. GitHub Actions enabled
3. Appropriate permissions for the repository

### Installation

1. **Create the GitHub Action workflow** in `.github/workflows/opencode.yml`:

```yaml
name: opencode

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]

jobs:
  opencode:
    if: |
      contains(github.event.comment.body, '/oc') ||
      contains(github.event.comment.body, '/opencode')
    runs-on: ubuntu-latest
    permissions:
      id-token: write
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

      - name: Run OpenCode
        uses: anomalyco/opencode/github@latest
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          model: opencode/big-pickle
          use_github_token: true
```

2. **Commit and push** the workflow file to your repository

3. **Enable GitHub Actions** in your repository settings if not already enabled

### Usage

Once set up, you can use OpenCode by:

1. Creating an issue or pull request
2. Adding a comment with `/opencode` followed by your instruction
3. OpenCode will automatically process your request and create a pull request with the changes

## Example Workflow

1. **Create an issue**: "Add a greeting function"
2. **Comment**: `/opencode create hello.py with a greet function that takes a name parameter`
3. **OpenCode processes**: Creates hello.py with the requested function
4. **Pull request created**: OpenCode automatically creates a PR with the changes

## Permissions

The GitHub Action requires the following permissions:
- `id-token: write` - For authentication
- `contents: write` - To modify repository files
- `pull-requests: write` - To create and manage pull requests
- `issues: write` - To interact with issues

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure the GitHub Action has the required permissions
2. **Command not recognized**: Make sure to use `/opencode` or `/oc` at the beginning of your comment
3. **No response**: Check the Actions tab in GitHub for any error messages

### Getting Help

- Check the GitHub Action logs for detailed error messages
- Ensure your comment starts with `/opencode` or `/oc`
- Verify that the workflow file is correctly placed in `.github/workflows/`

## Contributing

This project serves as a demonstration of OpenCode capabilities. Feel free to experiment with different `/opencode` commands to explore its features and capabilities.