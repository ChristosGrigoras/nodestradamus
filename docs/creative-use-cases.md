# Creative Use Cases for Nodestradamus

Beyond standard dependency analysis, Nodestradamus's building blocks can be combined for powerful, unconventional workflows.

## Building Blocks

| Capability | What It Knows |
|------------|---------------|
| Dependency Graph | What code calls what, imports what |
| Semantic Embeddings | Code meaning, finds similar code |
| Git Co-occurrence | Files that change together historically |
| PageRank/Betweenness | Critical code, bottlenecks |
| Community Detection | Clusters of related code |
| Impact Analysis | Blast radius of changes |
| String Analysis | Traces string literals across codebase |
| Cycle Detection | Circular dependencies |

---

## Developer Workflows

### Onboarding Pathfinder

Generate an optimal "reading order" for new team members using dependency topology:

```
1. Start from leaf nodes (files with few dependencies)
2. Work toward high-PageRank hubs
3. Creates a curriculum: "Read these 5 files first, then these 10, then the core"
```

**Implementation:**
```python
# Get PageRank scores
analyze_graph(repo_path, algorithm="pagerank")

# Sort files: lowest PageRank first (dependencies before dependents)
# Generate reading order from leaves → hubs
```

**Output:** Auto-generate `LEARNING_PATH.md` for any codebase.

---

### Merge Conflict Predictor

Combine git co-occurrence with in-flight PR data:

```
"You're editing auth.py. Heads up: session.py and middleware.py 
have an 85% co-occurrence rate and are in 2 open PRs"
```

**Implementation:**
```python
# Get files that change together
analyze_cooccurrence(repo_path, commits=500)

# Cross-reference with open PRs
# Alert when editing high-cooccurrence files
```

---

### PR Size/Risk Estimator

Before submitting a PR, compute blast radius:

```python
total_impact = sum([get_impact(file) for file in changed_files])
# → "This PR has a blast radius of 127 files"
# → Suggest splitting if risk is high
```

**Use as:** Pre-commit hook or CI check.

---

### Refactoring Safety Score

Before a big refactor, compute a safety metric:

```python
safety_score = (
    1.0 / impact_radius          # Smaller blast = safer
    * (1 - cycle_participation)   # Fewer cycles = safer
    * test_cooccurrence_rate      # More test coverage = safer
)
```

**Output:** "This refactor is safe" vs "This touches 40% of the codebase — consider splitting"

---

## Feature & Configuration Management

### Feature Flag Blast Radius

Trace feature toggles through the codebase:

```python
# Find all code controlled by a feature flag
analyze_strings(repo_path, mode="usages", target_string="ENABLE_NEW_CHECKOUT")

# Combine with impact analysis for transitive effects
get_impact(repo_path, file_path="features/checkout.py")
```

**Answers:** "What breaks if I remove this feature flag?"

---

### Dead Feature Detector

Cross-reference to find abandoned features:

1. **Dead code analysis** → unused functions
2. **String analysis** → UI strings like `"Add to Cart"`
3. **Intersection** → features that exist in code but are never reached

```python
# Find dead code
codebase_health(repo_path, checks=["dead_code"])

# Find UI strings with no references
analyze_strings(repo_path, mode="refs", min_files=1)

# Cross-reference: strings in dead code = dead features
```

---

## Architecture & Team Organization

### Team Topology Suggester

Use community detection to suggest team boundaries:

```python
# Detect natural code clusters
analyze_graph(repo_path, algorithm="communities")

# Output:
# "These 47 files form a cohesive unit — consider assigning to one team"
# Metagraph shows inter-team dependencies
```

**Quantifies:** Conway's Law violations.

---

### Architecture Drift Monitor

Compare detected communities vs declared architecture layers:

```python
# Define intended layers
layers = [["api/", "routes/"], ["services/"], ["models/", "db/"]]

# Validate
analyze_graph(repo_path, algorithm="layers", layers=layers)

# CI check: "New code in models/ imports from api/ — layer violation!"
```

---

### Microservice Extraction Planner

Find clean extraction boundaries:

```python
# Find natural clusters
communities = analyze_graph(repo_path, algorithm="communities")

# Find cycles (blockers to extraction)
cycles = analyze_graph(repo_path, algorithm="cycles")

# Communities with minimal inter-edges = good extraction candidates
# Cycles across would-be boundaries = blockers to resolve first
```

---

## Testing & Quality

### Test Coverage Risk Score

Combine criticality with test proximity:

```python
# How critical is this code?
pagerank_scores = analyze_graph(repo_path, algorithm="pagerank")

# Does changing it trigger test changes?
cooccurrence = analyze_cooccurrence(repo_path)

# Risk = High PageRank + Low test co-occurrence
for file, score in pagerank_scores:
    test_cooccurrence = get_test_cooccurrence(file, cooccurrence)
    if score > 0.01 and test_cooccurrence < 0.3:
        print(f"DANGER ZONE: {file} — critical but undertested")
```

---

### Documentation Staleness Detector

Find docs that drift from code:

```python
# Git co-occurrence of code files with doc files
cooccurrence = analyze_cooccurrence(repo_path)

# If code changes frequently but docs rarely change together → stale
for code_file in get_code_files():
    doc_cooccurrence = get_cooccurrence_with_docs(code_file)
    if doc_cooccurrence < 0.1:
        print(f"Stale docs: {code_file} changes often but docs don't follow")
```

---

## Security & Compliance

### Security Blast Radius Prioritization

Given a CVE in a dependency:

```python
# Trace what code touches the vulnerable package
get_impact(repo_path, file_path="venv/lib/.../vulnerable_package.py")

# PageRank-sort the results
# Triage by "which vulnerable code paths are most critical?"
```

---

### Dependency License Auditor

Trace external imports to identify license exposure:

```python
# Get all external imports
deps = analyze_deps(repo_path)
external = deps["external_imports"]

# For each GPL/AGPL import, trace what internal code touches it
for lib in external:
    if is_copyleft(lib):
        impact = get_impact(repo_path, file_path=lib)
        print(f"License exposure: {lib} affects {len(impact)} files")
```

---

### Compliance Evidence Generator

For security audits, show all paths from input to persistence:

```python
# Find all paths from user input to database writes
analyze_graph(
    repo_path,
    algorithm="path",
    source="api/routes/user_input.py",
    target="db/write.py"
)

# Evidence that validation exists on every path
```

---

## Code Review & Collaboration

### Code Review Auto-Assignment

Map communities to expert areas:

```python
# Detect communities
communities = analyze_graph(repo_path, algorithm="communities")

# Map to teams
community_owners = {
    "community_3": "@security-team",  # auth/*, session/*, middleware/*
    "community_7": "@data-team",       # models/*, db/*, migrations/*
}

# On PR: "Changes touch auth community — assign to @security-team"
```

---

### Branch Strategy Advisor

Use co-occurrence to suggest feature branch scope:

```python
cooccurrence = analyze_cooccurrence(repo_path)

# "These 12 files have 90%+ co-occurrence — work on them together"
# "These 3 files never change together — safe to parallelize"
```

---

## Performance & Debugging

### Performance Bottleneck Finder

High betweenness centrality = on many dependency paths = likely hot path:

```python
# Find code that sits on many paths
analyze_graph(repo_path, algorithm="betweenness", top_n=10)

# "These functions are called from everywhere — optimize first"
```

---

### Incident Investigation Accelerator

Given a stack trace location:

```python
# 1. What called this?
get_impact(repo_path, file_path="error_location.py", symbol="failing_function")

# 2. What else might have the same bug?
semantic_analysis(repo_path, mode="similar", symbol="failing_function")

# Surface "related bugs" across the codebase
```

---

### Copy-Paste Genealogy

Track code duplication over time:

```python
# Find semantic duplicates
semantic_analysis(repo_path, mode="duplicates", threshold=0.85)

# "This function was copy-pasted 7 times across 3 months"
# Track the drift between copies
```

---

## Miscellaneous

### Interview Question Generator

Use PageRank to find critical code, then generate questions:

```python
# Find most important code
critical = analyze_graph(repo_path, algorithm="pagerank", top_n=5)

# Search for explanatory content
for file in critical:
    semantic_analysis(repo_path, mode="search", query=f"explain {file}")

# Generate: "Walk me through how RunnableSequence.invoke() works"
```

---

## Implementation Priority

| Idea | Effort | Impact | Notes |
|------|--------|--------|-------|
| Feature Flag Blast Radius | Low | High | Combines existing tools |
| Test Coverage Risk Score | Low | High | PageRank + co-occurrence fusion |
| Team Topology Suggester | Medium | High | Novel value proposition |
| Onboarding Pathfinder | Medium | High | Generates artifacts |
| PR Risk Estimator | Low | Medium | Actionable pre-commit hook |
| Architecture Drift Monitor | Low | Medium | Already have `layers` algorithm |
| Incident Investigation | Low | Medium | Combines impact + semantic |

---

## Contributing New Use Cases

Found a creative use? Open a PR adding your workflow to this document with:

1. **Problem statement** — What question does it answer?
2. **Implementation** — Which Nodestradamus tools to combine
3. **Example output** — What the user sees
