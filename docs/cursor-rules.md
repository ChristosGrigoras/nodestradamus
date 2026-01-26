# Cursor AI Rules

Rules are `.mdc` files in `.cursor/rules/` that tell Cursor AI how to behave.

## Rule Numbering Convention

| Range | Purpose | Example |
|-------|---------|---------|
| `001-099` | Core system rules | Router, meta-generator, security |
| `100-199` | Language-specific | Python, JavaScript, Go |
| `200-299` | Project-specific | Your project context |
| `300-399` | Capability-specific | API patterns, auth, database, testing |

**Capability Rules (300-399):** For larger projects, create rules for specific capabilities like authentication, database access, or API design.

---

## Core Rules (001-099)

### `001-router.mdc`

**Purpose:** Detects what you're working on and loads the right rules.

**How it works:**
- Sees `*.py` file → loads Python rules
- Sees `Dockerfile` → loads DevOps rules
- Sees `*.md` file → loads documentation rules

### `002-meta-generator.mdc`

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

### `003-code-quality.mdc`

**Purpose:** Universal coding standards that apply to all languages.

**Contains:**
- **Critical Partner Mindset:** Question assumptions, prioritize truth over agreement
- **Execution Sequence:** Search first, reuse first, no assumptions, challenge ideas
- Naming conventions (descriptive, no abbreviations)
- Function guidelines (single responsibility, <50 lines)
- Comment standards (explain why, not what)
- Error handling (explicit, meaningful messages)
- Security basics (no hardcoded secrets)

### `004-response-quality.mdc`

**Purpose:** How the AI should communicate with you.

**Key behaviors:**
- Be direct, don't ask "Do you want me to..."
- Provide complete, working solutions
- Don't over-explain obvious concepts
- Never create `final_v2.py` - fix the original
- Never use `// ...existing code...` placeholders

### `005-security.mdc`

**Purpose:** Security guardrails that prevent common vulnerabilities.

**Protections:**
- Secrets: Never expose in logs, frontend, or hardcode
- SQL injection: Always use parameterized queries
- Command injection: Block dangerous shell commands
- Path traversal: Validate all file paths
- SSRF/XXE: Validate URLs, disable external entities

This rule has `priority: 100` (highest) to ensure security overrides other rules.

---

## Language Rules (100-199)

### `100-python.mdc`

**Purpose:** Python-specific conventions.

**Contains:**
- Type hints requirement
- Docstring format (Google style with Args, Returns, Raises)
- snake_case naming
- f-string preference
- Function length guidelines

---

## Project Rules (200-299)

### `200-project.mdc`

**Purpose:** Your project's specific context.

**You should add:**
- Project description
- Key directories and their purpose
- Domain-specific terminology
- External services/APIs used
- Team conventions

---

## Capability Rules (300-399)

### `301-testing.mdc`

**Purpose:** Language-agnostic testing patterns.

**Contains:**
- AAA pattern (Arrange → Act → Assert)
- Test isolation and independence
- Mocking guidelines (mock boundaries, not internals)
- Coverage philosophy (meaningful, not 100%)
- What to test vs what NOT to test

### `302-validation.mdc`

**Purpose:** Data validation patterns and boundaries.

**Contains:**
- Fail-fast validation principles
- Validation boundaries (user input, APIs, files)
- Error message guidelines
- Type coercion rules
- Schema-first approach

### `303-api-patterns.mdc`

**Purpose:** REST API design conventions.

**Contains:**
- HTTP method semantics (GET, POST, PUT, PATCH, DELETE)
- URL design (nouns, plurals, nesting)
- Consistent response/error formats
- Status code usage
- Authentication patterns

### `304-architecture.mdc`

**Purpose:** Architectural patterns for AI-assisted development.

**Key insight:** AI agents make implicit architectural decisions if not guided explicitly.

**Contains:**
- Layered architecture (controllers → services → repositories)
- Separation of concerns
- Dependency injection patterns
- Non-functional requirements (observability, resilience, scalability)
- Statelessness for horizontal scaling
- Domain boundary guidelines

### `305-dependency-graph.mdc`

**Purpose:** Instructs AI to use dependency graphs for impact analysis.

**When graphs exist in `.cursor/graph/`, AI will:**
1. Check for downstream impacts before modifying functions
2. List affected files in responses
3. Update related code proactively

---

## Applying Rule Changes

**Important:** When you create or modify a rule file, Cursor may not pick up changes immediately.

**To apply rule changes:**

1. **Reload Window** (recommended):
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Developer: Reload Window"
   - Press Enter

2. **Or start a new chat session:**
   - New rules only apply to new chat/agent sessions
   - Close current chat and start fresh

**When AI modifies rules:**
If Cursor's agent edits `.mdc` files, changes may not save correctly. Workaround:
1. Manually save the file (`Cmd/Ctrl+S`)
2. Reload window as above

**Quick reference:**
| Action | Required Step |
|--------|---------------|
| Create new rule | Reload Window |
| Edit existing rule | Reload Window |
| AI modifies rule | Save manually → Reload Window |
