# OpenCode Plan Project

Python project with OpenCode GitHub integration and Cursor AI for local development.

## Project Structure

- `.github/workflows/` - GitHub Actions (OpenCode integration)
- `.cursor/rules/` - Cursor AI rules
- `.opencode/` - OpenCode agents and skills

## Code Standards

- Use Python 3.x with type hints
- Include docstrings for all functions
- Use snake_case for functions and variables
- Prefer f-strings over .format()
- Keep functions small and focused (< 20 lines ideal)

## Response Quality

- Be direct - get straight to the point
- Provide complete solutions - working code that can be copy-pasted
- Don't ask "Do you want me to..." - just do it
- Don't create `final_v2.py`, `fixed_final.py` - fix the original

## Integrations

- **OpenCode GitHub Agent**: Comment `/opencode <task>` on issues/PRs
- **Cursor AI**: Local IDE assistant with `.cursor/rules/`

## External References

For detailed coding standards: @.cursor/rules/003-code-quality.mdc
For response guidelines: @.cursor/rules/004-response-quality.mdc
