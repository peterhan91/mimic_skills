# How to Best Improve Our Diagnostic Agent: Strategic Analysis

## The Core Insight

**The gap between LLMs (45-55%) and doctors (89%) is not a knowledge gap — it is
a reasoning process gap.**

GPT-4 scores 90% on static medical QA (multiple choice), but drops to 52% on the
SAME questions when it must gather information interactively (AgentClinic). Doctors
and LLMs have similar medical knowledge. Doctors have structured cognitive procedures
for wielding that knowledge. Our agent has none.

The current prompt (line 23 of `prompts.py`) says only:

> "reflect on your progress and decide what to do next"

No differential tracking. No discriminative test selection. No active disconfirmation.
The agent just... reacts to each observation independently.

---

## Root Cause Analysis: Why 45-55% Instead of 89%

Ranked by contribution to the accuracy gap:

### 1. No Hypothesis-Driven Reasoning (~20-30% of the gap)

Doctors form 3-5 differential diagnoses from the chief complaint and continuously
update them as evidence arrives. They select tests that MAXIMALLY DISCRIMINATE
between remaining hypotheses, not tests that seem generically relevant.

**Our agent**: Free-text "Thought:" with no structured differential. Anchors on
the first plausible diagnosis and selectively highlights confirming evidence
(confirmation bias). Never asks "what evidence would DISPROVE my leading hypothesis?"

**Evidence**: MAI-DxO (Microsoft, 2025) achieved 81.9% accuracy on hard NEJM cases
vs 49.3% for vanilla GPT-4o — a +32 point gain — primarily by forcing structured
differential tracking with Bayesian updating.

### 2. Lab Interpretation Void (~15-20% of the gap)

Default lab output: `(Blood) WBC: 14.5 K/uL` — no reference range, no clinical
context, no severity. The agent must infer from training data that 14.5 > 10.0
(the upper normal) and that elevated WBC suggests infection.

Lab interpretation accuracy: **26-77%** across pathologies. This is the paper's #1
documented failure mode.

**Approach 3** (already implemented, disabled by default) partially addresses this
with clinical annotations. But even with annotations, the agent doesn't reason about
lab COMBINATIONS (e.g., "elevated WBC + elevated CRP + left shift = strong acute
inflammatory signal").

### 3. Wasted Turns on Tool Hallucination (~5-10% of the gap)

Agent invents tool names every 2-5 patients (e.g., "CT Scan" instead of "Imaging").
Each hallucination wastes 1 of 10 allowed iterations. With only 10 turns to do
PE + Labs + Imaging + Diagnosis, losing 1-2 turns to format errors means incomplete
workups.

Regex parsing is fragile: 19+ regex substitutions to strip malformed output. Custom
parsing penalties tracked but never corrected — agent repeats same mistakes.

### 4. No Severity-Stratified Treatment (~5-10% of the gap)

Treatment planning accuracy is only **30.5%** even when diagnosis is correct
(MedR-Bench 2025). The agent says "appendectomy" without distinguishing uncomplicated
(laparoscopic) from perforated (open exploration with washout).

Pancreatitis evaluator expects Support + Drainage + ERCP + Cholecystectomy decisions
based on necrosis/biliary status — the agent has no framework for severity grading.

### 5. PE Skipped (~3-5% of the gap)

47% of trajectories skip PE or do it late. The prompt lists tools equally with no
hierarchy. Without PE findings, the agent must diagnose from labs + imaging alone,
which is clinically harder (and misses discriminating signs like Murphy's sign vs.
McBurney's point tenderness).

---

## Interventions Ranked by Expected Impact

### Tier 1: Transformative (expected +20-35% accuracy combined)

#### A. Explicit Differential Tracking Protocol

**What**: Inject into skill/instructions a mandatory cognitive protocol:

```
BEFORE each action:
1. State your current differential (top 3 diagnoses + approximate likelihood %)
2. For your top 2 hypotheses, identify the finding that would MOST distinguish them
3. Order THAT specific test

AFTER each observation:
4. Update differential: what moved up, what moved down, and WHY
5. Check: is any hypothesis >80% likely with no red flags? If yes → Final Diagnosis
6. Check: have you actively looked for evidence AGAINST your leading hypothesis?
```

**Why this works**: Encodes the core cognitive pattern doctors use. Forces the agent
to maintain structured state across turns rather than reacting independently to each
observation. Prevents premature closure (the #1 cause of diagnostic error in medicine
and in LLMs).

**Evidence**: MAI-DxO gained +32 points from structured differential alone.
Multi-agent devil's advocate studies went from 0% to 76% on biased scenarios.
MedKGI gained 25.5 points from knowledge-grounded hypothesis tracking.

**Implementation options** (in order of increasing effort):
1. Skill text in `{examples}` slot (zero code change, test immediately)
2. Approach 1 system prompt injection (minimal code change)
3. SDK prototype with `output_type=DifferentialUpdate` (structural enforcement)

#### B. Multi-Agent Cognitive Roles (at minimum: Challenger)

**What**: Before committing to a final diagnosis, the orchestrator calls a
Challenger agent that specifically argues AGAINST the leading diagnosis:

```
Challenger prompt: "The diagnostic agent believes this patient has [DIAGNOSIS].
Here is the evidence gathered: [EVIDENCE]. Your job is to:
1. Identify the strongest alternative diagnosis
2. Point out any evidence that contradicts [DIAGNOSIS]
3. Suggest one more test that would definitively confirm or rule out [DIAGNOSIS]
If the evidence overwhelmingly supports [DIAGNOSIS], say 'CONFIRMED'."
```

**Why this works**: LLMs exhibit strong confirmation bias. Once anchored on a
diagnosis, they selectively interpret subsequent evidence as confirming. A dedicated
adversary breaks this pattern. Research shows multi-agent discussion with a challenger
role accounts for the majority of accuracy gains in debiased clinical reasoning.

**Evidence**: 0% → 76% on biased scenarios with devil's advocate (JMIR 2024).
MAI-DxO uses "Dr. Challenger" as one of 5 roles in its 81.9% system.

**Implementation**: SDK agents-as-tools pattern. The Challenger is a separate Agent
called as a tool before committing to Final Diagnosis. Can use a cheap model (mini)
since it only needs to critique, not generate.

### Tier 2: High Impact (expected +5-15% accuracy combined)

#### C. Eliminate Tool Hallucination via Function Calling

**What**: Replace regex-based Thought/Action/Observation parsing with native
function calling (JSON schema-validated tool calls).

**Why**: Immediately recovers 1-2 wasted turns per trajectory. Zero hallucinated
tools. Zero parsing penalties. The agent spends 100% of its 10 turns on actual
clinical reasoning instead of fighting format compliance.

**Evidence**: Every 2-5 patients currently waste a turn (Hager paper). With 10 max
turns, that's 10-20% of reasoning budget lost to format errors.

**Implementation**: SDK `@function_tool` decorator. Or: Anthropic/OpenAI native
function calling API within current framework (less migration effort).

#### D. Structured Final Output

**What**: Force the agent to produce a complete structured diagnosis:

```python
class DiagnosticResult(BaseModel):
    diagnosis: str
    severity: Literal["uncomplicated", "moderate", "severe", "critical"]
    evidence: list[str]  # Key findings supporting diagnosis
    differential_ruled_out: list[str]  # What was considered and why eliminated
    treatment: str
    treatment_rationale: str  # Links severity to treatment choice
```

**Why**: (1) Eliminates parsing errors entirely (output_type enforcement).
(2) Forces severity grading, directly addressing the 30.5% treatment accuracy gap.
(3) Requires the agent to justify its diagnosis with evidence and explain ruled-out
alternatives — this alone improves reasoning quality.

**Implementation**: SDK `output_type` parameter. Or: structured output mode in
OpenAI/Anthropic API within current framework.

#### E. Lab Interpreter Sub-Agent

**What**: Dedicated agent (cheap model) that receives raw lab values + reference
ranges and returns structured clinical interpretation:

```
Input:  "WBC: 14.5 K/uL [4.0-10.0], CRP: 85 mg/L [0-10], Lipase: 42 U/L [13-60]"
Output: "INFLAMMATORY MARKERS: WBC significantly elevated (45% above normal) and
         CRP markedly elevated (8.5x normal) — strong acute inflammatory signal.
         PANCREATIC MARKERS: Lipase is normal — ARGUES AGAINST pancreatitis.
         CLINICAL SIGNIFICANCE: Pattern most consistent with acute inflammatory
         process (appendicitis, cholecystitis, diverticulitis). Normal lipase
         effectively rules out pancreatitis."
```

**Why**: Goes beyond Approach 3 static annotations. The sub-agent can reason about
COMBINATIONS and explicitly state what findings ARGUE AGAINST certain diagnoses
(discriminative interpretation, not just abnormality flagging).

**Implementation**: SDK agents-as-tools. Lab Interpreter called automatically after
each RunLaboratoryTests tool call. Uses a cheap model (gpt-4o-mini) since the task
is interpretive, not creative.

### Tier 3: Moderate Impact (expected +2-5% accuracy)

#### F. Tool Guardrails for Workflow Enforcement

Input guardrail on Labs/Imaging: "If PE hasn't been done yet, prepend a reminder:
'Consider performing Physical Examination first to guide targeted test selection.'"
Soft enforcement — doesn't block, just nudges.

#### G. Dynamic Per-Patient Skill Injection

`instructions=function(ctx)` generates context-dependent instructions after the
first round narrows the differential. If PE suggests RLQ tenderness, dynamically
inject appendicitis-relevant lab guidance.

#### H. Enable Approach 3 by Default

`annotate_clinical=True` should be the default, not opt-in. Zero prompt cost,
significant lab interpretation improvement. Already implemented and tested.

---

## Optimal Architecture (SDK-Based)

```
┌─────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                     │
│  instructions = differential_reasoning_protocol          │
│  output_type = DiagnosticResult (Pydantic)               │
│  model = "gpt-4o" or "llama-3.3-70b" (via LiteLLM)      │
│                                                          │
│  Tools:                                                  │
│  ├── @function_tool: perform_physical_exam()              │
│  ├── @function_tool: run_lab_tests()                      │
│  │     └── output_guardrail: auto-call lab_interpreter   │
│  ├── @function_tool: run_imaging()                        │
│  ├── @function_tool: read_diagnostic_criteria()           │
│  ├── lab_interpreter.as_tool()  [gpt-4o-mini]            │
│  └── challenger.as_tool()       [gpt-4o-mini]            │
│                                                          │
│  input_guardrails:                                       │
│  └── pe_first_reminder (nudge if PE not done)            │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────────┐
         ▼             ▼                 ▼
   ┌───────────┐ ┌───────────┐    ┌──────────────┐
   │    Lab     │ │Challenger │    │   Clinical   │
   │Interpreter│ │   Agent   │    │    Tools     │
   │           │ │           │    │  (PE, Labs,  │
   │ Reasons   │ │ Argues    │    │   Imaging,   │
   │ about lab │ │ AGAINST   │    │  DiagCrit)   │
   │ combina-  │ │ leading   │    │              │
   │ tions +   │ │ diagnosis │    │ @function_   │
   │ clinical  │ │ before    │    │ tool with    │
   │ patterns  │ │ commit    │    │ JSON schema  │
   │           │ │           │    │ validation   │
   │ [mini]    │ │ [mini]    │    │              │
   └───────────┘ └───────────┘    └──────────────┘
```

**Agent loop flow:**

```
1. Read patient history → Form initial differential (top 3)
2. PE → Update differential (which diagnoses gained/lost evidence)
3. Labs → Lab Interpreter analyzes → Update differential
4. If differential is clear → call Challenger before committing
5. If Challenger says "CONFIRMED" → Final Diagnosis with severity + treatment
6. If Challenger raises doubts → Order one more discriminative test → repeat
7. If Challenger says alternative is stronger → Switch leading diagnosis
```

---

## Implementation Roadmap

### Phase 0: Quick Win (1 day, no code changes)

Write a "Differential Reasoning Skill" as a text document. Inject via existing
Approach 2 (`{examples}` slot). Enable `annotate_clinical=True`. Test on existing
LangChain agent with Llama 3.3 on the train split.

Expected: +5-15% accuracy from skill text alone + lab annotations.

### Phase 1: SDK Prototype (1-2 weeks)

Build parallel SDK-based agent in `codes_openai_agent/`:
- 4 tools as `@function_tool`
- `output_type=DiagnosticResult`
- Differential reasoning in instructions
- Same patient data + PathologyEvaluator (adapter needed)

Expected: +10-20% accuracy from structural enforcement + no hallucinated tools.

### Phase 2: Multi-Agent (1-2 weeks)

Add Lab Interpreter and Challenger sub-agents:
- Lab Interpreter called automatically after each lab result
- Challenger called before committing to Final Diagnosis

Expected: +5-10% additional accuracy from debiased reasoning.

### Phase 3: EvoTest Integration (2-3 weeks)

Use SDK's built-in tracing to feed trajectories to EvoTest's Evolver:
- Evolver analyzes differential tracking quality, not just final accuracy
- Evolves the reasoning protocol (not just prompt text)
- UCB selection prevents regression

Expected: +2-5% additional accuracy from automated skill refinement.

### Total Expected Trajectory

```
Baseline (Hager, Llama 3.3):        ~50%
+ Differential skill + annotations:  ~60-65%  (Phase 0)
+ SDK structural enforcement:        ~65-70%  (Phase 1)
+ Multi-agent reasoning:             ~70-80%  (Phase 2)
+ EvoTest refinement:                ~75-85%  (Phase 3)
Doctor benchmark:                     89%
```

These are estimates based on published results from comparable interventions.
Actual numbers will depend on model quality, patient case difficulty, and
implementation details. But the direction is clear: encoding cognitive procedures
matters more than adding medical facts.

---

## Key Research References

| Paper | Year | Key Finding |
|---|---|---|
| Hager et al. (Nature Medicine) | 2024 | LLMs 45-55% vs doctors 89% on MIMIC-CDM |
| MAI-DxO (Microsoft) | 2025 | 81.9% with 5-role structured differential (+32 over vanilla) |
| AgentClinic | 2024 | GPT-4 drops from 90% → 52% when interactive (knowledge ≠ reasoning) |
| MedR-Bench (Nature Comms) | 2025 | Treatment planning 30.5% even with correct diagnosis |
| MedKGI | 2025 | +25.5 points from knowledge-grounded hypothesis tracking |
| Multi-agent debiasing (JMIR) | 2024 | 0% → 76% with devil's advocate on biased scenarios |
| mARC-QA (Scientific Reports) | 2025 | LLMs <50% on adversarial cases, doctors 66% (Einstellung effect) |
| LLMs lack metacognition (Nature Comms) | 2024 | Confident even when correct options absent |
| Dual Process Theory + LLMs (Nature Rev) | 2025 | LLMs default to System 1; need System 2 scaffolding |
