# Task Planning Skill

## Purpose
Create structured, design-first task breakdowns for features and projects.

## Task File Template

When creating task files, use this structure:

```markdown
# [Feature Name]

## Brief Description
- What the feature does and why
- User value proposition
- Key constraints

## Design Phase (Complete First)
- [ ] Document requirements and acceptance criteria
- [ ] Research existing solutions and patterns
- [ ] Create user flows and wireframes
- [ ] Define data models and API contracts
- [ ] Validate design with stakeholders
- [ ] Get approval before implementation

## Setup Phase
- [ ] Create feature branch: `feat/[name]`
- [ ] Verify environment and dependencies
- [ ] Confirm access to required services

## Implementation Phase
- [ ] [Task 1] - with test and commit
- [ ] [Task 2] - with test and commit
- [ ] [Task 3] - with test and commit

## Relevant Files
- `path/to/file1`
- `path/to/file2`

## Success Criteria
- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated
```

## Guidelines

### Task Granularity
- Break into 5-10 specific tasks
- Each task completable in 2-4 hours
- One task = one commit

### Design-First
- Complete design phase before any implementation
- Validate assumptions before coding
- Document decisions and rationale

### Implementation Workflow
For each task:
1. Check for blockers
2. Implement the change
3. Test immediately
4. Fix any issues
5. Commit with descriptive message
6. Mark task complete
7. Proceed to next

### Environment Safety
- Never auto-create `.env` files
- Reference `.env.example` for patterns
- Ask user for configuration guidance

## Output
Create a `[feature-name]-tasks.md` file with the above structure.
