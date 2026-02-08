# Per-Sub-Agent Skills: Evolving Skills at Every Level

## The Problem with Monolithic Skills

Currently, **one skill** is injected into the **orchestrator** agent only:

```
EvoTest Evolver
    │
    ▼
┌─────────────────────────────┐
│  Orchestrator Skill         │  ← ONE skill does everything
│  (workflow + labs + diff dx) │
└─────────────┬───────────────┘
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
  PE        Labs      Imaging    ← simple tools, no skills
              │
              ▼
        Lab Interpreter          ← FIXED instructions, not evolved
              │
              ▼
          Challenger             ← FIXED instructions, not evolved
```

This creates three problems:

1. **Misattributed failures**: When the evaluator reports low "Laboratory Tests"
   scores, the Evolver modifies the orchestrator skill — but the orchestrator
   doesn't interpret labs. The Lab Interpreter does. The fix never reaches the
   actual failure point.

2. **Skill bloat**: One skill must cover workflow strategy, lab interpretation,
   differential reasoning, treatment selection, and severity grading. It grows
   toward the 500-token budget trying to do everything, diluting each instruction.

3. **Coupled evolution**: The Lab Interpreter converges quickly (lab patterns are
   stable) but the Challenger needs more iterations (differential reasoning is
   harder). With one monolithic skill, they can't evolve independently.

## Proposed: Per-Sub-Agent Skills

```
EvoTest Evolver
    │
    ├── generates ──► Orchestrator Skill  (workflow, tool selection, when to stop)
    ├── generates ──► Lab Interpreter Skill  (pattern recognition, clinical significance)
    └── generates ──► Challenger Skill  (differential reasoning, bias detection)
    │
    ▼
┌─────────────────────────────┐
│  Orchestrator               │  ← Skill: workflow strategy
│  instructions = base +      │
│    orchestrator_skill       │
└─────────────┬───────────────┘
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
  PE        Labs      Imaging
              │
              ▼
┌─────────────────────────────┐
│  Lab Interpreter            │  ← Skill: lab pattern recognition
│  instructions = base +      │
│    lab_interpreter_skill    │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Challenger                 │  ← Skill: differential reasoning
│  instructions = base +      │
│    challenger_skill         │
└─────────────────────────────┘
```

Each sub-agent gets its own evolving SKILL.md. The Evolver generates a **skill
bundle** (3 files) per episode instead of one monolithic skill.

## Why This is Better for EvoTest

### 1. Targeted Evolution

The Evolver can see trajectory data showing *which sub-agent* produced *which
output*. When labs are misinterpreted:

**Before (monolithic):** Evolver writes "pay attention to lipase values" in the
orchestrator skill → orchestrator passes this info to Lab Interpreter → but Lab
Interpreter has FIXED instructions and ignores it.

**After (per-agent):** Evolver writes "lipase >3x ULN with abdominal pain is
diagnostic for acute pancreatic injury" directly in the Lab Interpreter skill →
Lab Interpreter uses it immediately.

### 2. Better Signal Decomposition

PathologyEvaluator already scores different aspects separately:

| Metric | Maps to Sub-Agent |
|--------|-------------------|
| Diagnosis accuracy | Orchestrator (final decision) |
| Physical Examination | Orchestrator (tool selection) |
| Laboratory Tests | Lab Interpreter (interpretation quality) |
| Treatment correctness | Orchestrator + Challenger (severity assessment) |
| Invalid Tools | Orchestrator (tool selection) |

The Evolver can use this mapping to decide *which skill to modify*.

### 3. Independent Convergence

- Lab Interpreter skill may converge in 3 episodes (patterns are finite and stable)
- Challenger skill may need 7 episodes (differential reasoning is harder)
- Orchestrator skill evolves throughout (depends on the other two)

With per-agent skills, the UCB tree can track convergence per-agent and focus
evolution effort where it's needed most.

### 4. Smaller, Focused Skills

Instead of one 500-token skill:

| Skill | Budget | Focus |
|-------|--------|-------|
| Orchestrator | ~200 tokens | Tool selection, workflow ordering, when to diagnose |
| Lab Interpreter | ~150 tokens | Lab pattern recognition, reference ranges, significance |
| Challenger | ~150 tokens | Differential reasoning, bias detection, evidence gaps |

Each skill stays concise and focused. Matches Anthropic's "concise is key"
principle from the [Skills spec](https://agentskills.io/specification).

### 5. Progressive Disclosure (Zero Cost When Not Used)

Sub-agent skills are only loaded when that sub-agent is called. If the agent
diagnoses from PE alone (no labs ordered), the Lab Interpreter skill costs zero
tokens. This naturally optimizes token usage.

## Implementation

### Skill Bundle Structure

```
skills/evo_sdk/
    episode_5/
        orchestrator.md        # Orchestrator skill
        lab_interpreter.md     # Lab Interpreter skill
        challenger.md          # Challenger skill
```

### Injection Points

**Orchestrator** — already implemented (via `clinical_instructions()`):
```python
def clinical_instructions(ctx, agent):
    base = "..."
    skill = ctx.context.skill_content  # ← orchestrator skill
    if skill:
        base += f"\n\n## Clinical Reasoning Skill\n\n{skill}"
    return base
```

**Lab Interpreter** — new: inject skill into instructions:
```python
def create_lab_interpreter_tool(model_name, skill_content=None):
    instructions = LAB_INTERPRETER_INSTRUCTIONS
    if skill_content:
        instructions += f"\n\n## Lab Interpretation Skill\n\n{skill_content}"
    lab_interpreter = Agent(
        name="Lab Interpreter",
        instructions=instructions,
        ...
    )
```

**Challenger** — same pattern:
```python
def create_challenger_tool(model_name, skill_content=None):
    instructions = CHALLENGER_INSTRUCTIONS
    if skill_content:
        instructions += f"\n\n## Diagnostic Challenge Skill\n\n{skill_content}"
    ...
```

### Evolver Prompt Changes

Currently the Evolver generates one skill. With per-agent skills, the Evolver
generates a structured output with three sections:

```
Given the trajectory failures, generate THREE focused skills:

## ORCHESTRATOR SKILL
[Workflow strategy, tool selection, when to stop investigating]

## LAB INTERPRETER SKILL
[Lab pattern recognition, reference ranges, clinical significance]

## CHALLENGER SKILL
[Differential reasoning, what to challenge, bias detection patterns]
```

The Evolver sees the full trajectory including sub-agent outputs, so it can
attribute failures to specific agents and target improvements accordingly.

### EvoTest State Changes

The UCB tree node stores a skill bundle instead of a single skill:

```python
node = {
    "skill_bundle": {
        "orchestrator": "...",
        "lab_interpreter": "...",
        "challenger": "..."
    },
    "composite_score": 4.2,
    ...
}
```

## Future: Tools as Sub-Agents

Beyond the existing sub-agents, tools themselves could become skill-equipped
sub-agents:

| Current Tool | Potential Sub-Agent | Skill Would Teach |
|---|---|---|
| `physical_examination` | PE Interpreter | Highlight abnormals, suggest differentials from exam |
| `laboratory_tests` | Lab Ordering Advisor | Which labs to order given clinical picture (prevent shotgun) |
| `imaging` | Imaging Interpreter | Extract key findings from radiology report, grade severity |

However, this adds an LLM call per tool invocation (cost + latency). The current
architecture already addresses the main failure modes:
- Lab interpretation → Lab Interpreter sub-agent
- Diagnostic rigor → Challenger sub-agent
- Tool hallucination → SDK function calling (eliminated)
- Lab annotation → Approach 3 tool output augmentation (zero-cost)

**Recommendation**: Implement per-sub-agent skills first (low cost, high impact).
Consider tools-as-sub-agents later only if the per-agent approach plateaus.

## Comparison with Anthropic Agent Skills Standard

Our per-sub-agent skills map naturally to the [Anthropic Agent Skills
standard](https://github.com/anthropics/skills):

| Anthropic Concept | Our Implementation |
|---|---|
| SKILL.md with YAML frontmatter | `orchestrator.md`, `lab_interpreter.md`, `challenger.md` |
| Progressive disclosure (metadata → body → resources) | Sub-agent skill loaded only when sub-agent invoked |
| "Concise is key" | Three focused ~150-token skills vs one bloated 500-token skill |
| Scripts as bundled resources | Clinical annotation logic (`nlp.py`) as bundled tool augmentation |
| Iteration (Step 6) | EvoTest evolutionary loop |

The key difference: Anthropic's skills are **manually authored**. Ours are
**automatically evolved** by EvoTest from trajectory failures and clinical
guidelines, then sanitized to remove disease name leakage.
