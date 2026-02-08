# Agent Architecture v2: Evidence-Based Redesign

Based on analysis of 8 repos on this PC + 10 SOTA papers (2025-2026).

## What the Research Says

| System | Accuracy | Key Innovation |
|--------|----------|----------------|
| MAI-DxO (Microsoft, Jul 2025) | **85.5%** | 5 virtual physicians: Hypothesis + Test-Chooser + Challenger + Stewardship + Checklist |
| EvoClinician (Jan 2026) | **59.8%** | Action-level credit assignment (not trajectory-level), prompt + memory evolution |
| MedAgentSim (MICCAI 2025) | **79.5% on MIMIC-IV** | Experience replay memory, correctly diagnosed cases as retrievable examples |
| AMIE (Google, Nature 2025) | Beat physicians 28/32 axes | Structured differential is the single most important output |
| AgentClinic (ICLR 2024) | **+92%** from notebook tool | Persistent structured state across turns |
| MDAgents (NeurIPS 2024) | +11.8% adaptive | 4-5 agents optimal; beyond that, performance declines |
| MedKGI (Jan 2026) | 69.8% | OSCE-format state tracking + knowledge graph constraint |
| Systematic Review (PMC, Aug 2025) | 0% → 76% on biased cases | Challenger/devil's advocate is highest-value addition |

**Three interventions validated across ALL top systems:**

1. **Explicit differential tracking** — MAI-DxO, AMIE, AgentClinic, MedKGI
2. **Adversarial challenge** — MAI-DxO, Systematic Review
3. **Structured output** — All systems use it

## What Our Repos Already Have

| Repo | Architecture | Agents | Differential? | Challenger? | Structured Output? |
|------|-------------|--------|---------------|-------------|-------------------|
| Hager (LangChain) | ReAct | 1 | No | No | No (regex) |
| MIMIC-Plain | ReAct/ReWOO | 1 | No | No | JSON (final only) |
| MIMIC-ReAct | ReAct + RAG | 1 | No | No | No (regex) |
| MIMIC_rewoo | ReWOO 3-phase | 1 (planner+solver) | No | No | Text |
| **codes_openai_agent** | **Multi-agent** | **3** | **No** | **Yes (once)** | **Yes (Pydantic)** |

Our SDK agent is already the most advanced, but it's missing the **#1 intervention**:
explicit differential tracking. And the Challenger is only called once at the end,
not throughout reasoning.

## The Gap: No Explicit Differential Tracking

**Current flow** — The orchestrator reasons in free text. There is no persistent
structured state for hypotheses:

```
Patient History → think → PE → think → Labs → Lab Interpreter → think → Imaging → think → Challenger → Diagnose
                  ^^^^         ^^^^                                ^^^^
                  All "thinking" is internal to the LLM — no structured differential exists
```

**What MAI-DxO does** — Dr. Hypothesis maintains a probability-ranked differential
that updates after EVERY finding:

```
Patient History → Initialize differential [A: 40%, B: 30%, C: 20%, D: 10%]
  → PE findings → Update [A: 55%, B: 25%, C: 15%, D: 5%]
  → Lab results → Update [A: 70%, B: 20%, C: 8%, D: 2%]
  → Imaging     → Update [A: 90%, B: 8%, C: 2%]
  → Challenge   → Confirm or revise
  → Diagnose
```

AgentClinic showed that just giving the agent a **notebook tool** for persistent
state gave **92% relative improvement**. The state doesn't need to be complex —
it just needs to be explicit and persistent.

## Proposed Architecture

**4 agents** (within MDAgents' optimal range of 4-5):

```
┌────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (strong model)                           │
│                                                        │
│  Role: Flow control, tool selection, final diagnosis   │
│  State: DiagnosticState (structured, persistent)       │
│  Skill: Workflow strategy, when to stop investigating  │
│                                                        │
│  Tools: PE, Labs, Imaging, DiagCrit                    │
│  Sub-agents: ↓ ↓ ↓                                    │
└───────┬──────────┬──────────┬─────────────────────────┘
        │          │          │
        ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │   Lab    │ │Differen- │ │Challenger│
  │Interpret.│ │  tial    │ │          │
  │          │ │ Tracker  │ │          │
  │(cheap)   │ │(cheap)   │ │(cheap)   │
  └──────────┘ └──────────┘ └──────────┘
  Called after   Called after   Called when
  lab results   EVERY finding  differential
                               narrows to 1
                               AND before
                               final diagnosis
```

### Agent 1: Orchestrator (existing, enhanced)

**What changes**: Maintains a `DiagnosticState` in PatientContext.

```python
@dataclass
class DiagnosticState:
    """Persistent structured state across reasoning turns."""
    differential: list[dict]   # [{diagnosis, probability, evidence_for, evidence_against}]
    evidence_log: list[str]    # All findings accumulated so far
    tests_done: list[str]      # What's been ordered (prevents repetition)
    confidence: float          # Highest hypothesis probability (0-1)
```

After receiving patient history, the orchestrator initializes the differential.
After each tool result, it calls the Differential Tracker to update it.
When confidence exceeds a threshold, it calls the Challenger before finalizing.

**Skill focus**: Workflow strategy, tool selection, when to stop.

### Agent 2: Lab Interpreter (existing, unchanged)

**What it does**: Analyzes raw lab values, identifies abnormals, finds patterns.

**Why it's a separate agent**: Lab interpretation requires comparing dozens of
numerical values against reference ranges and recognizing combinatorial patterns
(e.g., elevated WBC + CRP + left shift = acute inflammation). This is the
paper's worst failure mode (26-77% correct).

**Skill focus**: Pattern recognition, reference ranges, clinical significance.

### Agent 3: Differential Tracker (NEW)

**What it does**: Called after EVERY new finding (PE, labs, imaging). Takes the
current differential + new finding → returns updated differential with reasoning.

```
Input:  Current differential + new finding
Output: DifferentialUpdate {
    updated_differential: [{diagnosis, probability, evidence_for, evidence_against}],
    reasoning: "WBC 14.5 with left shift supports inflammatory etiology,
                increases appendicitis from 40% to 55%, decreases pancreatitis
                from 30% to 20% (normal lipase expected in pancreatitis)",
    most_discriminating_next_test: "CT Abdomen (would differentiate appendicitis vs diverticulitis)"
}
```

**Why separate from orchestrator**: Offloads the most cognitively demanding task
(Bayesian updating of multiple hypotheses) to a focused agent. Keeps the
orchestrator's context clean for tool selection. The Differential Tracker's output
is structured (Pydantic), not free text — this is the "notebook" that AgentClinic
showed gives 92% improvement.

**Why not just put this in the orchestrator's instructions**: Because the
orchestrator also has to manage tool calling, format compliance, turn counting,
and final output. Asking it to also maintain a rigorous differential in its
"thinking" leads to the free-text reasoning that current agents do poorly.
Separating it forces the differential to be explicit and structured.

**Skill focus**: Hypothesis management, Bayesian reasoning, discriminating tests.

### Agent 4: Challenger (existing, enhanced)

**What changes**: Called at TWO points instead of one:
1. When the differential narrows to a single dominant hypothesis (confidence > 0.7)
2. Before finalizing the diagnosis

Currently the Challenger is called once at the very end. MAI-DxO's Dr. Challenger
participates in every round. Our compromise: call it when the agent is about to
"lock in" on a diagnosis, giving it a chance to force reconsideration.

**Skill focus**: Cognitive bias detection, overlooked evidence, alternative diagnoses.

## Why Not More Agents?

MAI-DxO uses 5 agents (Hypothesis, Test-Chooser, Challenger, Stewardship,
Checklist). MDAgents found 4-5 is optimal. Why only 4 for us?

**We don't need Dr. Test-Chooser**: Our tool space is small (PE, Labs, Imaging,
DiagCrit). Test selection is not the bottleneck — the agent doesn't have dozens
of test options to choose from. The Differential Tracker already suggests the
most discriminating next test as part of its output.

**We don't need Dr. Stewardship**: Cost-aware testing matters in real clinical
practice but is not scored by PathologyEvaluator. Adding it would add latency
without improving measured metrics.

**We don't need Dr. Checklist**: SDK function calling already ensures valid tool
names and structured output. The checklist role is handled by the framework.

## Per-Agent Skills + EvoTest

Each agent gets its own evolving SKILL.md (from our per_agent_skills.md design):

```
EvoTest Evolver
  │
  ├── Orchestrator Skill    → workflow, tool selection, confidence thresholds
  ├── Lab Interpreter Skill → pattern recognition, reference ranges
  ├── Diff Tracker Skill    → hypothesis management, Bayesian reasoning
  └── Challenger Skill      → bias detection, what to challenge
```

The Evolver generates a **skill bundle** per episode. It sees the full trajectory
including sub-agent outputs, maps failures to specific agents, and targets
improvements where they're needed.

**Action-level grading** (from EvoClinician): Instead of just scoring the whole
episode, grade individual actions:
- HIGH_YIELD: Lab Interpreter correctly identified pancreatitis from lipase pattern
- LOW_YIELD: Ordered unnecessary CBC when WBC was already checked
- CRITICAL_ERROR: Challenger accepted diagnosis despite contradicting evidence

This gives the Evolver +5.2 pts better signal than trajectory-level scoring.

## Comparison: Current vs. Proposed

| Dimension | Current SDK Agent | Proposed v2 |
|-----------|------------------|-------------|
| Agents | 3 (Orch + Lab Interp + Challenger) | 4 (+Differential Tracker) |
| Differential tracking | None (free-text reasoning) | Structured DiagnosticState, updated after every finding |
| Challenger timing | Once (before final diagnosis) | Twice (when confidence > 0.7 + before final) |
| Skills | 1 monolithic → orchestrator only | 4 per-agent skills, independently evolved |
| Evolver signal | Trajectory-level composite score | Action-level grading (HIGH_YIELD/LOW_YIELD/CRITICAL_ERROR) |
| State persistence | PatientContext (tool flags only) | DiagnosticState (differential + evidence log + confidence) |
| Token overhead | ~200 tokens (1 skill) | ~600 tokens (4 skills) + sub-agent calls |

## What This Doesn't Change

- Same PatientContext with tool flags (pe_done, labs_done, imaging_done)
- Same PathologyEvaluator for scoring
- Same evaluator_adapter.py bridge (sub-agents filtered out)
- Same vLLM backend
- Same EvoTest UCB tree structure
- Same clinical guidelines integration
- Same annotate_clinical lab augmentation

## Implementation Priority

| Step | Change | Effort | Impact (evidence) |
|------|--------|--------|-------------------|
| 1 | Add DiagnosticState to PatientContext | Small | Highest (AgentClinic: +92%) |
| 2 | Create Differential Tracker sub-agent | Medium | Highest (MAI-DxO: +32 pts) |
| 3 | Per-agent skills in EvoTest | Medium | High (targeted evolution) |
| 4 | Call Challenger at confidence > 0.7 | Small | High (0% → 76% on biased) |
| 5 | Action-level grading in Evolver | Medium | High (EvoClinician: +5.2 pts) |

Steps 1-2 are the architectural changes. Steps 3-5 are EvoTest improvements.
All can be done incrementally — each step delivers value independently.

## References

- MAI-DxO: https://arxiv.org/html/2506.22405v1 (Microsoft, July 2025)
- EvoClinician: https://arxiv.org/html/2601.22964v1 (January 2026)
- AMIE: https://www.nature.com/articles/s41586-025-08869-4 (Google, Nature 2025)
- MDAgents: https://arxiv.org/abs/2404.15155 (MIT, NeurIPS 2024)
- MedAgentSim: https://arxiv.org/abs/2503.22678 (MICCAI 2025)
- AgentClinic: https://arxiv.org/abs/2405.07960 (ICLR 2024)
- MedKGI: https://arxiv.org/abs/2512.24181 (January 2026)
- AI Agents in Clinical Medicine: https://pmc.ncbi.nlm.nih.gov/articles/PMC12407621/
