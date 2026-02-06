# Publishing to PyPI

## Prerequisites

1. **PyPI account** — Create at https://pypi.org/account/register/
2. **API token** — Create at https://pypi.org/manage/account/token/

## One-Time Setup

```bash
# Install build and upload tools
pip install build twine maturin

# Store your PyPI token (do this once)
# Create ~/.pypirc with:
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE
EOF

# Secure the file
chmod 600 ~/.pypirc
```

**Never commit `~/.pypirc` or real tokens.** Use environment variables or CI secrets for automation.

## Build and Upload

This project uses **maturin** (Rust extension), not `python -m build`.

```bash
# Navigate to the repo
cd /path/to/nodestradamus

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package (wheel + sdist into dist/)
maturin build --release --out dist --sdist

# This creates under dist/, for example:
#   nodestradamus-0.1.0.tar.gz
#   nodestradamus-0.1.0-*.whl

# Upload to PyPI
twine upload dist/*
```

## Verify

```bash
# Wait a minute, then:
pip install nodestradamus

# Test it works
nodestradamus --version
```

## Test First (TestPyPI)

```bash
# Upload to test server
twine upload --repository testpypi dist/*

# Install from test server
pip install --index-url https://test.pypi.org/simple/ nodestradamus
```

## Checklist Before Publishing

| Item | Check |
|------|-------|
| `pyproject.toml` has correct `name = "nodestradamus"` | ☐ |
| `version` is set (e.g. `"0.1.0"`) | ☐ |
| `description` is filled in | ☐ |
| `authors` is set | ☐ |
| `license` is set | ☐ |
| `readme = "README.md"` (or equivalent) is set | ☐ |
| `keywords` for discoverability | ☐ |
| `classifiers` for PyPI categories | ☐ |
| All dependencies in `[project.dependencies]` | ☐ |
| Entry point for CLI: `[project.scripts]` | ☐ |
| Rust extension builds: `maturin build --release` | ☐ |

## After Publishing

- Package page: https://pypi.org/project/nodestradamus/
- Users can install with:
  - `pip install nodestradamus`
  - `pip install nodestradamus[faiss]`
  - `pip install nodestradamus[all]`
