# Quick Health Check Guide

**Time to complete:** 10-15 minutes  
**Coverage:** Essential analysis for any codebase

---

## What is Nodestradamus?

Nodestradamus is a **Model Context Protocol (MCP) server** — a codebase intelligence layer designed for AI assistants. Users don't run Nodestradamus commands directly; they ask questions to an AI assistant, which uses Nodestradamus tools behind the scenes.

**How it works:**

```
User: "What is this codebase?"
  ↓
AI (internally): Calls project_scout tool
  ↓
AI (to user): "This is a Python web framework built on Starlette 
              and Pydantic, with 1,252 files and excellent test 
              coverage..."
```

This guide shows what happens behind the scenes when users ask common questions.

---

## Quick Reference

| User Asks | AI Uses | Time |
|-----------|---------|------|
| "What is this codebase?" | `project_scout` | 3s |
| "How is the code organized?" | `analyze_deps` | 5s |
| "Is the codebase healthy?" | `codebase_health` | 5s |
| "Give me an overview" | `quick_start` | 15s |

---

## Step 1: Initial Reconnaissance (3 seconds)

**User question:** "What is this codebase?"

**AI calls:** `project_scout`

**What you get:**
- Primary language
- Detected frameworks
- Key directories
- Entry points
- Contributor count
- **lazy_options** — When to use LazyGraph, LazyEmbeddingGraph, or lazy embedding (for monorepos or large codebases); for those repos, `next_steps` includes LazyEmbeddingGraph

**Example:**
- Language: Python (1,252 files)
- Frameworks: FastAPI, Starlette, Pydantic, Uvicorn
- Structure: Main package, docs, examples, tests
- Entry point: `app/__main__.py`

**Decision point:**
- ✅ Clear structure → Continue to Step 2
- ⚠️ Unfamiliar tech → Research frameworks first
- 🚫 Multiple languages → May need custom analysis

---

## Step 2: Dependency Graph (5 seconds)

**User question:** "What are the main modules and connections?"

**AI calls:** `analyze_deps`

**What you get:**
- Total files, nodes, edges
- Most central files (top callers)
- Most depended upon (top called)
- External dependencies

**Example:**
- Scale: 6,748 nodes, 14,919 edges
- Core: `routing.py` (62 calls), `dependencies/utils.py` (57 calls)
- Most used: `__init__.py` (712 dependents)
- Key insight: Tests 3x larger than core library

**Key insight:** Most central file = likely most important to understand

---

## Step 3: Health Check (5 seconds)

**User question:** "How healthy is this codebase?"

**AI calls:** `codebase_health`

**What you get:**
- Health grade (A to F)
- Dead code count
- Duplicate pairs
- Circular dependencies
- Top bottlenecks
- Documentation coverage

**Example:**
- **Grade: B-**
- Dead code: 0 ✅
- Duplicates: 30 pairs ⚠️
- Cycles: 3,828 🔴
- Bottleneck: `__init__.py` (0.106) 🚨
- Doc coverage: 21.5% 📚

**Immediate actions:**
- Grade C or lower → Deep dive required
- High cycles + bottlenecks → Architecture review needed
- Low doc coverage → Documentation sprint

---

## Step 4: Comprehensive Overview (15 seconds)

**User question:** "Give me a complete overview of this codebase"

**AI calls:** `quick_start`

This combines Steps 1-3 automatically plus pre-computes embeddings for semantic search.

**What you get:**
Combines Steps 1-3 plus:
- Pre-computed embeddings for semantic search
- Graph algorithms (PageRank, Betweenness)
- Combined JSON report

**When to use:**
- ✅ First-time codebase exploration
- ✅ Onboarding new engineers
- ✅ CI/CD health checks
- ✅ Monthly maintenance reviews

**When NOT to use:**
- ❌ Need specific targeted analysis
- ❌ Want to learn tool-by-tool
- ❌ Have strict time limits in CI

---

## Interpreting Results

### Health Grades

| Grade | Meaning | Action |
|-------|---------|--------|
| **A** | Excellent | Maintain standards |
| **B** | Good | Minor improvements |
| **C** | Concerning | Review technical debt |
| **D** | Poor | Refactoring required |
| **F** | Critical | Immediate intervention |

### Bottleneck Scores (Betweenness)

| Score | Severity | Meaning |
|-------|----------|---------|
| **>0.10** | 🚨 Extreme | Sits on >10% of all paths |
| **0.05-0.10** | 🔴 Critical | Major choke point |
| **0.02-0.05** | 🟡 High | Important connector |
| **<0.02** | 🟢 Moderate | Normal coupling |

### Cycle Counts

| Count | Assessment | Action |
|-------|-----------|--------|
| **0-10** | ✅ Excellent | Well-designed |
| **10-100** | ⚠️ Acceptable | Monitor growth |
| **100-1000** | 🔴 High | Refactoring needed |
| **>1000** | 🚨 Critical | Architecture review |

---

## Common Patterns

### Pattern 1: Export Hub Bottleneck

**Symptom:** `__init__.py` has very high betweenness (>0.10)

**Meaning:** Single point of failure - all imports go through one file

**Action:** 
- Document as critical file
- Require extra PR review for changes
- Consider breaking into smaller modules

---

### Pattern 2: Test-Heavy Codebase

**Symptom:** Test files outnumber core files 3:1 or more

**Meaning:** Excellent test coverage (good!)

**Action:**
- Maintain test-to-code ratio
- Use tests as documentation

---

### Pattern 3: Script Duplication

**Symptom:** High duplicates in `scripts/` or `tools/` folders

**Meaning:** Utility functions copy-pasted across scripts

**Action:**
- Extract to `scripts/utils.py` or `scripts/shared.py`
- Low risk, high value refactoring

---

### Pattern 4: Documentation Drift

**Symptom:** >50% stale documentation references

**Meaning:** Code evolved faster than docs

**Action:**
- Focus on high-confidence broken links first
- Update or remove outdated references
- Consider doc generation from code

---

## Next Steps

### If Health Grade is A-B:
1. ✅ Document critical files identified by bottleneck analysis
2. ✅ Set up `get_changes_since_last` for CI monitoring
3. ✅ Address minor duplicates if easy

### If Health Grade is C-D:
1. 🔴 Run detailed analysis with `analyze_graph` algorithms
2. 🔴 Prioritize breaking high-betweenness cycles
3. 🔴 Plan refactoring sprint for duplicates

### If Health Grade is F:
1. 🚨 Immediate architecture review meeting
2. 🚨 Consider rewrite vs refactor
3. 🚨 Block new features until health improves

---

## Tools Summary

### Included in Quick Start
- ✅ `project_scout` - Repository overview
- ✅ `analyze_deps` - Dependency graph
- ✅ `codebase_health` - Health metrics
- ✅ `semantic_analysis` (embeddings) - Pre-compute for search

### Not Included (Run Separately)
- `analyze_graph` - Detailed graph algorithms
- `get_impact` - Blast radius analysis
- `analyze_cooccurrence` - Git history analysis
- `semantic_analysis` (search/duplicates) - Specific queries
- `find_similar` - Structural fingerprinting
- `analyze_docs` - Documentation deep-dive
- `get_changes_since_last` - Change tracking

See [Getting Started Workflow](getting-started-workflow.md) for the full tool sequence.

---

## Time Investment vs Value

| Investment | What You Get | When to Use |
|-----------|--------------|-------------|
| **5 min** (Steps 1-3) | Basic understanding | Triage, first look |
| **15 min** (Step 4) | Comprehensive overview | Onboarding, reviews |
| **30-60 min** (Full analysis) | Deep insights | Major refactoring |
| **Ongoing** (CI integration) | Health tracking | Continuous monitoring |

---

## Common Questions

**Q: Do I need to learn Nodestradamus commands?**  
A: No. Nodestradamus is an MCP — you just ask questions to your AI assistant. The AI selects and calls the right tools automatically.

**Q: What questions can I ask?**  
A: Any question about code architecture, dependencies, impact, quality, or documentation. Examples:
- "What is this codebase?"
- "What happens if I change X?"
- "Is there duplicate code?"
- "How do I implement authentication?"

**Q: How does the AI know which tool to use?**  
A: The AI maps your question to the appropriate tool. "What's critical?" → PageRank. "What breaks?" → Impact analysis. "How do I..." → Semantic search.

**Q: Can Nodestradamus be used in CI/CD?**  
A: Yes. The AI can set up automated health checks using `quick_start` and track changes over time with `get_changes_since_last`.

**Q: What languages does Nodestradamus support?**  
A: Python, TypeScript, Rust. For multi-language repos, the AI analyzes each and combines insights.

**Q: What size codebase works best?**  
A: 100+ files. Smaller projects don't need automated analysis. Larger projects (10,000+ files) work well but may need targeted queries rather than full analysis.

---

## The MCP Advantage

Unlike traditional analysis tools:

| Traditional Tool | Nodestradamus (MCP) |
|------------------|---------------|
| You learn commands | You ask questions |
| You interpret output | AI interprets for you |
| One perspective at a time | AI combines multiple tools |
| Static reports | Dynamic, context-aware answers |

**Result:** You get codebase intelligence without learning a new tool.
