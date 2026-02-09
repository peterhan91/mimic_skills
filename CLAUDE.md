# MIMIC-Skills: Improving Clinical Decision-Making with Agent Skills

## Project Goal

Improve LLM performance on the autonomous clinical decision-making benchmark
established by Hager et al. (Nature Medicine, 2024) by injecting **agent skills**
generated via the [upskill](https://github.com/huggingface/upskill) framework.

**Core hypothesis**: Real clinical reasoning patterns, extracted from MIMIC-IV
discharge summaries and distilled into agent skills, can close the diagnostic
accuracy gap between LLMs (45-55%) and physicians (89%).

---

## Background

### Hager's Agent (what we are improving)

A LangChain ZeroShotAgent that iteratively diagnoses patients:

```
Input:  Patient History (HPI + PMH + Social/Family)
Loop:   Thought → Action → Observation  (max 10 iterations)
Tools:  Physical Examination, Laboratory Tests, Imaging, [Diagnostic Criteria]
Output: Final Diagnosis + Treatment
Eval:   PathologyEvaluator (per-pathology, multi-metric)
```

**Key files:**
- `codes_Hager/.../agents/prompts.py` — CHAT_TEMPLATE (system prompt)
- `codes_Hager/.../agents/agent.py` — CustomZeroShotAgent + build_agent_executor_ZeroShot
- `codes_Hager/.../tools/Actions.py` — Tool implementations (PE, Labs, Imaging, DiagCrit)
- `codes_Hager/.../evaluators/pathology_evaluator.py` — Base evaluator with trajectory scoring
- `codes_Hager/.../evaluators/{pathology}_evaluator.py` — Per-pathology evaluators
- `codes_Hager/.../run.py` — Main evaluation entry point (Hydra config)
- `codes_Hager/.../configs/config.yaml` — Default configuration

### CHAT_TEMPLATE Structure (critical — all integration approaches target this)

```
{system_tag_start}                              ← SYSTEM SECTION
  "You are a medical AI assistant..."           ← Role + goal
  Format instructions (Thought/Action/Obs)      ← Output format rules
  Tool descriptions (PE, Labs, Imaging)         ← Available tools
  {add_tool_descr}                              ← Optional extra tools (e.g., DiagCrit)
{system_tag_end}
{user_tag_start}                                ← USER SECTION
  {examples}                                    ← Tool-use examples, few-shot examples
  "Consider the following case..."
  "Patient History: {input}"
{user_tag_end}
{ai_tag_start}                                  ← AGENT SECTION
  "Thought:{agent_scratchpad}"                  ← Multi-turn reasoning chain
```

### Paper's Identified Failure Modes

| # | Failure Mode | Paper Evidence |
|---|---|---|
| 1 | Poor diagnostic accuracy | 45-55% (autonomous) vs 89% (doctors) |
| 2 | Cannot interpret lab results | Low: 26-70%, High: 24-77% correct |
| 3 | Skip physical examination | WizardLM only 53.1% PE first |
| 4 | Insufficient treatment recs | Miss surgery for severe cases |
| 5 | Hallucinate nonexistent tools | Every 2-5 patients |
| 6 | Sensitive to info quantity | More info = worse (up to -18%) |
| 7 | Sensitive to info order | Up to 18% accuracy swing |
| 8 | Sensitive to phrasing | "final" vs "main" diagnosis matters |

### Paper's Existing Mitigations (manual, tested individually)

- Tool use examples in prompt (`{examples}`) → helped format compliance
- Few-shot patient examples (`{examples}`) → small accuracy gain
- Reference ranges on lab output (tool output) → mixed results
- Binned lab results Low/Normal/High (tool output) → mixed results
- Diagnostic criteria tool (new tool) → helped some pathologies
- Automatic summarization (agent scratchpad) → small consistent gain

### What upskill adds

Instead of hand-crafting prompt modifications one at a time, upskill:
1. Generates a coherent skill document from clinical reasoning traces
2. Iteratively refines the skill based on evaluation failures
3. Produces a portable SKILL.md that can be version-controlled and shared

**How upskill injects skills** (from `fastagent_integration.py`):
```python
def compose_instruction(instruction, skill):
    return f"{instruction}\n\n## Skill: {skill.name}\n\n{skill.body}"
```
Simple string concatenation onto the system instruction. upskill is agnostic to
the downstream agent — it just appends skill content to whatever system prompt
exists. This means we must decide WHERE and HOW to integrate with Hager's
specific architecture.

**Role boundaries:**
| Tool | Does | Does NOT do |
|---|---|---|
| Opus | Read discharge summaries, synthesize reasoning traces | Diagnose patients |
| upskill | Generate SKILL.md, refine from failures | Evaluate Hager's agent |
| Hager's framework | Run agent loop, evaluate trajectories, produce metrics | Generate skills |

---

## Skill Integration Approaches

Hager's agent has multiple surfaces where skill-derived content can be injected.
Each addresses different failure modes and carries different trade-offs.

### Approach 1: System Prompt Injection

**Where**: Between tool descriptions and `{system_tag_end}` in CHAT_TEMPLATE.

**What it does**: Adds authoritative instructions the model must follow. Acts as
rules (e.g., "Always perform Physical Examination first").

**Addresses**: Failure modes 3 (PE ordering), 5 (tool hallucination), 8 (phrasing).

**Trade-off**: Competes with existing format instructions for model attention.
The paper showed models are sensitive to system prompt changes — adding content
here could help OR hurt depending on phrasing.

```python
# In prompts.py, CHAT_TEMPLATE — inject before {system_tag_end}:
# ...tool descriptions...
# {skill_system_content}     ← workflow rules, tool usage rules
# {system_tag_end}
```

### Approach 2: User-Section Examples (`{examples}` slot)

**Where**: The `{examples}` placeholder in CHAT_TEMPLATE, in the user section.

**What it does**: Provides worked examples of correct clinical reasoning. Acts as
demonstrations (show, don't tell).

**Addresses**: Failure modes 1 (accuracy), 3 (PE ordering), 4 (treatment).

**Trade-off**: This is where the paper's existing mitigations already go (tool-use
examples, few-shot examples). Natural location for skill traces. Easy to compare
against the paper's baselines since they used the same slot.

```python
# In agent.py, build_agent_executor_ZeroShot:
# skill_content goes into tool_use_examples, which fills {examples}
```

### Approach 3: Tool Output Augmentation

**Where**: Modify what tools return in `Actions.py`, not the prompt.

**What it does**: Annotates tool outputs with skill-derived interpretations.
For example, lab results return with explicit Low/Normal/High annotations and
clinical significance.

**Addresses**: Failure mode 2 (lab interpretation) — the paper's #1 gap.

**Trade-off**: Zero prompt tokens consumed. Information arrives at the exact moment
the agent needs it. The paper already tested variants of this (`bin_lab_results`,
`include_ref_range`) with mixed results, but those were crude (just prepending
"HIGH" vs providing clinical context like "elevated WBC suggests acute inflammation").

```python
# In Actions.py, modify retrieve_lab_tests:
# Instead of: "WBC: 14.5 K/uL"
# Return:     "WBC: 14.5 K/uL | RR: [4.0-10.0] | HIGH — suggests acute inflammation"
```

A skill-derived version goes beyond the paper's approach: instead of just labeling
High/Normal/Low, it adds **clinical interpretation** (what the abnormality means
in context). This teaches the model not just WHAT is abnormal but WHY it matters.

### Approach 4: Enhanced Diagnostic Criteria Tool

**Where**: Augment the existing `ReadDiagnosticCriteria` tool in `Actions.py`.

**What it does**: When the agent queries diagnostic criteria, return richer
skill-derived content: decision trees, severity grading, differential reasoning,
treatment guidelines per severity.

**Addresses**: Failure modes 1 (accuracy), 4 (treatment).

**Trade-off**: Only helps if the agent actually calls the tool. The paper showed
agents often skip optional tools. Could mitigate by making the tool mandatory
(auto-invoke after imaging) or by putting key content elsewhere.

```python
# In prompts.py, augment DIAGNOSTIC_CRITERIA_APPENDICITIS with:
# - Alvarado score calculation
# - Severity grading (uncomplicated vs perforated vs gangrenous)
# - Treatment decision tree per severity
# - Key differentials to rule out (cholecystitis, ovarian torsion, ectopic)
```

### Approach 5: Hybrid (recommended)

**Split skill content by purpose and inject at the optimal location.**

| Skill Content | Injection Point | Approach | Why Here |
|---|---|---|---|
| Workflow rules ("PE first, then targeted labs, then imaging") | System prompt | 1 | Authoritative, always visible |
| Worked clinical reasoning example | `{examples}` slot | 2 | Demonstration, natural position |
| Lab interpretation with clinical context | Tool output augmentation | 3 | Right-time, zero prompt cost |
| Diagnostic criteria + treatment guidelines | DiagCrit tool (enhanced) | 4 | On-demand reference |

This decomposes the skill into components and delivers each at the moment and
location where it has maximum impact, rather than dumping everything into one
prompt section.

**Implementation**: Generate one SKILL.md via upskill, then a post-processing
script splits it into sections for each injection point. Or generate separate
focused skills per approach.

### Approach Comparison

| Approach | Prompt tokens | Addresses failure # | Engineering effort | Paper precedent |
|---|---|---|---|---|
| 1. System prompt | ~200 | 3, 5, 8 | Minimal | No (new) |
| 2. Examples slot | ~300 | 1, 3, 4 | Minimal | Yes (few-shot, tool examples) |
| 3. Tool augmentation | 0 | 2 | Moderate | Yes (bin_lab, ref_range) |
| 4. Enhanced DiagCrit | 0 (on-demand) | 1, 4 | Moderate | Yes (diag criteria tool) |
| 5. Hybrid | ~200 prompt + 0 tool | All | Most | Combines all |

### Experimental Design for Approaches

To determine which approach works best, test them independently first,
then combine:

```
Experiment matrix:
  Baseline                              (no skill)
  Approach 1 only                       (system rules)
  Approach 2 only                       (examples)
  Approach 3 only                       (tool augmentation)
  Approach 1 + 2                        (prompt-only skill)
  Approach 1 + 2 + 3                    (prompt + tools)
  Approach 5 full hybrid                (all approaches)
  Paper mitigations                     (tool examples + summarization)
  Paper mitigations + Approach 5        (everything combined)
```

This ablation study reveals which integration surface contributes most to
each metric, and whether approaches are complementary or redundant.

---

## Approach 6: OpenAI Agents SDK Rewrite

### Overview

Replace Hager's LangChain ZeroShotAgent with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
(`pip install openai-agents`). This is a lightweight, production-ready framework
for multi-agent workflows (successor to OpenAI's "Swarm"). Full analysis in
[docs/openai_agents_sdk.md](docs/openai_agents_sdk.md).

### Why This Matters

The SDK directly eliminates two of the paper's top failure modes:

| Failure Mode | Current Mitigation | SDK Solution |
|---|---|---|
| **#5 Tool hallucination** (every 2-5 patients) | Fuzzy matching + penalty | **Eliminated**: `@function_tool` + JSON schema constrains tool calls |
| **#8 Phrasing sensitivity** (up to 18% swing) | Format instructions in prompt | **Eliminated**: `output_type=DiagnosticResult` (Pydantic) enforces schema |
| **#2 Lab interpretation** (26-77% correct) | Approach 3 annotations | Agent-as-tool: dedicated lab interpreter sub-agent |
| **#3 Skip PE** (53% PE first) | Prompt rules | Tool guardrails enforce ordering |
| **#4 Treatment gaps** | Enhanced DiagCrit | Structured output requires treatment field |

### Key Design Patterns to Adopt

**1. Structured Diagnostic Output** — Agent loop won't terminate until the LLM
produces a valid `DiagnosticResult` with diagnosis, confidence, evidence,
differential, treatment, and severity fields. Eliminates parsing errors entirely.

**2. Multi-Agent via Agents-as-Tools** — Dedicated sub-agents for lab interpretation
and imaging interpretation. The orchestrator calls them as tools, gets structured
results back, and continues reasoning. Addresses failure mode #2 without consuming
prompt tokens.

**3. Dynamic Skill Injection** — `Agent.instructions` accepts a function
`(RunContextWrapper) -> str` that generates per-patient, per-pathology instructions
at runtime. Skills, clinical guidelines, and patient context injected dynamically.

**4. Tool Guardrails** — Input/output validation on each tool call. Can enforce
PE-first ordering, reject nonsensical lab combinations, and annotate results
before they reach the agent.

**5. Built-in Tracing** — Every tool call, reasoning step, and handoff automatically
captured as spans. Feeds directly into EvoTest's Evolver for trajectory analysis.

**6. Model Flexibility** — Via LiteLLM integration, any model works: Llama 3.3 70B
(local vLLM), Claude, GPT-4o, Gemini. Different agents can use different models
(cheap specialist + strong orchestrator).

### Implementation Plan

**Phase 1: Parallel Prototype** — Build SDK-based agent alongside Hager's code.
Same patient data, same PathologyEvaluator, apples-to-apples comparison.

```
codes_openai_agent/
  agent.py              ← Agent + tools definition
  tools.py              ← @function_tool implementations
  context.py            ← PatientContext dataclass
  evaluator_adapter.py  ← Adapt PathologyEvaluator for SDK trajectories
  run.py                ← Runner.run() with skill injection
```

**Phase 2: Feature Adoption** — If full rewrite is too costly, selectively adopt
high-value patterns into existing framework: structured output parsing (replace
regex with Pydantic), tool validation, dynamic instruction generation.

**Phase 3: Full Migration** — Replace LangChain agent if Phase 1 shows improvement.

### Limitations

- **No workflow enforcement**: Cannot enforce PE → Labs → Imaging at framework level
  (relies on instructions; LangGraph better for strict ordering)
- **Migration effort**: Rewriting tools + evaluation adapter is non-trivial
- **OpenAI dependency**: Core framework designed for OpenAI API; other providers via
  LiteLLM adapter

### Updated Experiment Matrix

```
Experiment matrix (extended):
  Baseline (LangChain)                  (no skill, original framework)
  Approach 1 only                       (system rules)
  Approach 2 only                       (examples)
  Approach 3 only                       (tool augmentation)
  Approach 5 full hybrid                (all approaches, LangChain)
  Approach 6 baseline                   (SDK agent, no skill)          ← NEW
  Approach 6 + skill                    (SDK agent + skill injection)  ← NEW
  Approach 6 + multi-agent              (SDK + lab/imaging sub-agents) ← NEW
  Paper mitigations + Approach 5        (everything combined, LangChain)
```

---

## Experiment Plan: Option A (Iterative Refinement)

### Concept

Start from a single discharge summary, generate a skill, evaluate, then iteratively
improve the skill using additional summaries that target specific failures.

### Protocol

**Step 1: Select seed cases (1 per pathology)**

Pick one representative discharge summary per pathology from MIMIC-IV. Choose cases
that are clinically typical (not edge cases) so the initial skill captures the
standard workflow.

- 1 appendicitis case (straightforward: RLQ pain → labs → CT → appendectomy)
- 1 cholecystitis case (RUQ pain → labs with liver markers → US → cholecystectomy)
- 1 diverticulitis case (LLQ pain → labs → CT → antibiotics or surgery)
- 1 pancreatitis case (epigastric pain → lipase/amylase → CT → supportive care)

**Step 2: Generate initial traces**

Feed each discharge summary to Opus:

```
Prompt: "Read this discharge summary. Reconstruct the clinical reasoning
as a step-by-step trace in this exact format:

Thought: [what the clinician was thinking based on the presentation]
Action: Physical Examination
Observation: [the exam findings from the summary]

Thought: [interpretation of exam, what to order next and why]
Action: Laboratory Tests
Action Input: [the specific tests that were actually ordered]
Observation: [the lab values with reference ranges]

Thought: [interpretation of labs — explicitly state if each is Low/Normal/High]
Action: Imaging
Action Input: [modality and region that was actually done]
Observation: [the radiology report from the summary]

Thought: [synthesis of all findings, reasoning toward diagnosis]
Final Diagnosis: [the discharge diagnosis]
Treatment: [the actual treatment given]

Be explicit about WHY each step was taken and HOW each result was interpreted."
```

Output: `traces/option_a/seed_appendicitis.md`, etc.

**Step 3: Generate initial skills**

```bash
upskill generate "diagnose appendicitis in acute abdominal pain patients \
  using physical exam, labs, and imaging in the correct clinical workflow" \
  --from traces/option_a/seed_appendicitis.md --no-eval \
  -o ./skills/option_a/v1/appendicitis/

# Repeat for each pathology
```

**Step 4: Inject and evaluate (baseline comparison)**

Start with Approach 2 (examples slot) as the simplest integration. If lift is
observed, test Approach 5 (hybrid) for maximum impact.

```bash
# Baseline run (no skill)
python run.py pathology=appendicitis model=Llama3.3Instruct70B

# With skill v1 (Approach 2: examples slot)
python run.py pathology=appendicitis model=Llama3.3Instruct70B \
  skill_path=./skills/option_a/v1/appendicitis/SKILL.md
```

Record all PathologyEvaluator metrics (see Evaluation Metrics table below).

**Step 5: Analyze failures and select next case**

From the evaluation results, identify specific failure patterns:
- Which patients were misdiagnosed?
- What labs were missing?
- Was PE skipped?
- Was treatment incomplete?

Select a NEW discharge summary from MIMIC-IV that exemplifies the failure pattern.
For example, if the model fails on cholecystitis when liver enzymes are borderline,
pick a cholecystitis case with borderline liver enzymes.

**Step 6: Generate refined trace and improve skill**

Convert the new discharge summary to a trace (same as Step 2).

```bash
upskill generate "improve: model fails when [specific failure pattern]. \
  Add reasoning for [specific clinical scenario]" \
  --from ./skills/option_a/v1/appendicitis/ -o ./skills/option_a/v2/appendicitis/
```

**Step 7: Re-evaluate and repeat**

```bash
python run.py pathology=appendicitis model=Llama3.3Instruct70B \
  skill_path=./skills/option_a/v2/appendicitis/SKILL.md
```

Repeat Steps 5-7 until skill performance plateaus or reaches target. Track
version history: v1, v2, v3, ... with scores at each version.

### Expected Iteration Cycle

```
v1: Seed case → skill → eval → identify failures
v2: Failure-targeted case → refine skill → eval → identify failures
v3: Another failure-targeted case → refine skill → eval → ...
vN: Plateau (typically 3-5 iterations based on upskill blog results)
```

### Strengths
- Surgical: each iteration targets a specific weakness
- Traceable: you know exactly which case fixed which failure
- Lightweight: only processes 1 new summary per iteration

### Risks
- May overfit to specific failure cases
- Slow to converge if failures are diverse
- Skill may grow too long if each iteration adds content

---

## Experiment Plan: Option C (Synthesis from Many Summaries)

### Concept

Feed Opus many discharge summaries at once and have it synthesize common clinical
reasoning patterns into one comprehensive trace per pathology. The skill captures
the collective wisdom of many real cases, not just one.

### Protocol

**Step 1: Sample discharge summaries**

From MIMIC-IV, stratified sampling per pathology:
- 20 appendicitis summaries
- 20 cholecystitis summaries
- 20 diverticulitis summaries
- 20 pancreatitis summaries

Sample should be diverse: vary in severity (uncomplicated vs complicated),
patient demographics, and clinical presentation to capture a broad range of patterns.

**Step 2: Synthesize patterns with Opus**

Feed all 20 summaries per pathology to Opus in one (or batched) prompt:

```
Prompt: "Here are 20 discharge summaries of patients diagnosed with
[PATHOLOGY] presenting to the ED with acute abdominal pain.

Analyze the diagnostic workups these doctors performed and identify:
1. Common presenting symptoms and their frequency
2. Key physical exam findings that distinguished this from other causes
3. Which laboratory tests were consistently ordered and which abnormal
   values were decisive (include reference ranges)
4. Which imaging modality and region confirmed the diagnosis
5. How treatment was determined based on severity (uncomplicated vs
   complicated/perforated/gangrenous)
6. Common differential diagnoses that were ruled out and how

Then produce ONE comprehensive clinical reasoning trace that captures
these patterns, in this exact format:

Thought: [typical clinical reasoning at this stage]
Action: Physical Examination
Observation: [representative findings with variants noted]
...
Final Diagnosis: [PATHOLOGY]
Treatment: [standard treatment with severity-based escalation]

The trace should represent the TYPICAL diagnostic workflow, noting where
variations occur (e.g., 'US preferred but CT used in 60% of cases')."
```

Output: `traces/option_c/synthesized_appendicitis.md`, etc.

**Step 3: Generate skills**

```bash
upskill generate "diagnose appendicitis in acute abdominal pain patients \
  following evidence-based clinical workflow patterns" \
  --from traces/option_c/synthesized_appendicitis.md --no-eval \
  -o ./skills/option_c/synth/appendicitis/

# Repeat for each pathology
```

**Step 4: Inject and evaluate**

Same approach selection as Option A Step 4. Start with Approach 2, then test
Approach 5 hybrid.

**Step 5: Iterate with failure-targeted refinement**

After initial evaluation, combine Option C with Option A: use failures to
guide selection of additional summaries, refine the synthesized skill.

```bash
upskill generate "improve: model struggles with severity grading in pancreatitis. \
  Add reasoning for Ranson criteria and when to escalate to necrosectomy" \
  --from ./skills/option_c/synth/pancreatitis/ \
  -o ./skills/option_c/synth-v2/pancreatitis/
```

### Strengths
- Broad: captures patterns from many cases, less likely to overfit
- Efficient: one skill generation step covers diverse presentations
- Robust: rare patterns and edge cases included from the start

### Risks
- May average out distinctive reasoning patterns
- Opus may miss important details when processing many summaries
- Synthesis quality depends on Opus's ability to identify patterns

---

## Option A vs Option C: When to Use Which

| Dimension | Option A (Iterative) | Option C (Synthesis) |
|---|---|---|
| Starting point | 1 case per pathology | 20 cases per pathology |
| Iteration speed | Fast per iteration | Slower initial setup |
| Failure targeting | Precise | Broad then targeted |
| Overfitting risk | Higher | Lower |
| Skill interpretability | High (trace from specific case) | Medium (synthesized patterns) |
| Recommended for | Quick prototyping, specific known failures | Production skills, comprehensive coverage |

**Recommended approach**: Start with Option A to validate the pipeline end-to-end
(injection works, evaluation runs, metrics move). Then switch to Option C
for the final skill versions used in reported results. Test integration approaches
(1-5) independently to find the best combination.

---

## Technical Implementation Details

### Skill Injection Code

**For Approach 1 (system prompt) and Approach 2 (examples slot):**

File: `codes_Hager/.../agents/agent.py`, function `build_agent_executor_ZeroShot`

```python
def build_agent_executor_ZeroShot(
    patient, llm, ...,
    skill_path=None,            # NEW PARAMETER
    skill_inject="examples",    # NEW: "examples" | "system" | "both"
):
    # ... existing code ...

    # Load skill if provided
    skill_system = ""
    skill_examples = ""
    if skill_path:
        with open(skill_path, "r") as f:
            raw = f.read()
        # Strip YAML frontmatter
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                raw = parts[2].strip()

        if skill_inject in ("examples", "both"):
            skill_examples = f"\n{raw}\n\n"
        if skill_inject in ("system", "both"):
            skill_system = f"\n{raw}"

    tool_use_examples = skill_examples + tool_use_examples
    add_tool_descr = add_tool_descr + skill_system
    prompt = create_prompt(tags, tool_names, add_tool_descr, tool_use_examples)
    # ... rest unchanged ...
```

**For Approach 3 (tool output augmentation):**

File: `codes_Hager/.../tools/Actions.py`, function `retrieve_lab_tests`

```python
def retrieve_lab_tests(action_input, action_results, lab_test_mapping_df,
                       include_ref_range, bin_lab_results,
                       annotate_clinical=False):    # NEW PARAMETER
    result_string = ""
    for test in action_input:
        if test in action_results["Laboratory Tests"]:
            result_string += create_lab_test_string(...)
            if annotate_clinical:
                # Add skill-derived clinical interpretation
                result_string += get_clinical_annotation(test, value, ref_range)
    return result_string
```

**For Approach 4 (enhanced DiagCrit tool):**

File: `codes_Hager/.../agents/prompts.py`, augment existing diagnostic criteria
constants with skill-derived content (severity grading, treatment decision trees).

Add to `configs/config.yaml`:
```yaml
skill_path:                     # Optional path to SKILL.md
skill_inject: examples          # "examples" | "system" | "both"
annotate_clinical: False        # Approach 3: annotate lab results
enhanced_diagnostic_criteria: False  # Approach 4: richer DiagCrit
```

### Skill Sanitization (Diagnosis Leakage Prevention)

Hager's framework sanitizes patient data — disease names like "appendicitis"
are replaced with "____" in Patient History, PE, and Radiology. If our skill
contains disease names, we re-introduce the exact leakage the framework removes.

**Script**: `scripts/sanitize_skill.py`

```python
# As a library
from scripts.sanitize_skill import sanitize_skill_text
clean = sanitize_skill_text(raw_skill_text)

# CLI
python scripts/sanitize_skill.py skills/v1/acute_abdominal_pain.md --inplace --report
```

Replaces disease names and 1:1 procedure names with `____` (same mask
Hager uses in `dataset.py` line 734). Does NOT mask anatomical terms
(appendiceal, gallbladder, pancreas) needed for clinical reasoning.

**This must be the last step before injecting a skill into the agent.**

### Evaluation Metrics (from PathologyEvaluator)

All experiments must report these metrics per pathology:

| Metric | Source | Addresses Failure # |
|---|---|---|
| Diagnosis accuracy | `scores["Diagnosis"]` | 1 |
| Gracious diagnosis accuracy | `scores["Gracious Diagnosis"]` | 1 |
| PE performed first | `scores["Physical Examination"]` | 3 |
| PE performed at all | `scores["Late Physical Examination"]` | 3 |
| Correct lab categories | `scores["Laboratory Tests"]` | 2 |
| Imaging score | `scores["Imaging"]` | 1 |
| Treatment correctness | `answers["Treatment Requested"]` vs `["Treatment Required"]` | 4 |
| Action parsing errors | `scores["Action Parsing"]` | 5, 8 |
| Invalid tool requests | `scores["Invalid Tools"]` | 5 |
| Number of rounds | `scores["Rounds"]` | 6 |
| Unnecessary lab tests | `answers["Unnecessary Laboratory Tests"]` | 6 |
| Unnecessary imaging | `answers["Unnecessary Imaging"]` | 6 |

### Models to Test

The skill's value is most apparent on weaker/cheaper models (upskill's
teacher-student premise). Test on a range:

| Model | Role | Context |
|---|---|---|
| Llama 3.3 70B | Primary student model (open, local) | 128K tokens |
| Llama 3.1 70B | Secondary student model | 128K tokens |
| GPT-3.5 Turbo | Cheap cloud student | 16K tokens |
| GPT-4 | Strong cloud model (expect smallest lift) | 128K tokens |

### Skill Token Budget

Keep each SKILL.md under 500 tokens. With the hybrid approach (Approach 5),
only ~200 tokens go into the prompt (rules + brief example); the rest is
delivered through tool augmentation at zero prompt cost.

For models with 128K context (Llama 3.3), this constraint is relaxed.
For GPT-3.5 at 16K, skill brevity matters more.

### Reproducing Baselines

Before testing skills, reproduce the paper's baseline results to validate the
pipeline works correctly:

```bash
python run.py pathology=appendicitis model=Llama3.3Instruct70B summarize=True
python run.py pathology=cholecystitis model=Llama3.3Instruct70B summarize=True
python run.py pathology=diverticulitis model=Llama3.3Instruct70B summarize=True
python run.py pathology=pancreatitis model=Llama3.3Instruct70B summarize=True
```

Llama 3.3 results won't exactly match the paper's Llama 2 results, but the
pattern (appendicitis > pancreatitis > cholecystitis/diverticulitis) should hold.

---

## Experimental Controls

### Required comparisons

1. **Baseline**: Hager's agent, original CHAT_TEMPLATE, no skill
2. **Paper mitigations**: tool_use_examples + summarization (paper's best config)
3. **Approach 2 only**: Skill in examples slot
4. **Approach 3 only**: Tool output augmentation (lab annotations)
5. **Approach 5 hybrid**: All approaches combined
6. **Paper mitigations + Approach 5**: Everything combined

This reveals whether the skill is complementary to or redundant with existing
mitigations.

### Statistical requirements

Following the paper's protocol:
- Run each configuration with multiple random seeds (paper used 20 seeds over 80 patients)
- Two-sided Student's t-test with unequal variances
- Bonferroni correction for multiple comparisons

---

## Directory Structure

```
mimic_skills/
├── CLAUDE.md                          # This file
├── README.md                          # Quick-start overview
├── scripts/
│   ├── split_data.py                  # Split MIMIC-CDM pkl → train/test/remaining
│   ├── prepare_split_for_hager.py     # Copy split as {path}_hadm_info_first_diag.pkl
│   ├── sanitize_skill.py              # Remove disease name leakage from skills
│   ├── extract_trajectories.py        # Parse results pkl → JSON for Evolver
│   ├── evolve_skill.py               # Call Opus to evolve skills from trajectories
│   ├── compare_runs.py               # Side-by-side comparison of two runs
│   ├── evaluate_run.py               # Run PathologyEvaluator on results
│   ├── parse_guidelines.py           # Extract clinical guidelines to markdown
│   ├── evotest_train.sh              # EvoTest evolutionary optimization loop
│   ├── evotest_test.sh              # Test best skill on 100-patient test set
│   ├── evotest_full.sh              # Full pipeline: train → test best skill
│   ├── container.sh                 # Apptainer container launcher (GPU server)
│   └── start_vllm.sh               # Start vLLM server
├── data_splits/                       # Created by split_data.py
│   ├── appendicitis/
│   │   ├── train.pkl (10)
│   │   ├── test.pkl (100)
│   │   └── remaining.pkl (809)
│   ├── cholecystitis/ (10/100/514)
│   ├── diverticulitis/ (10/100/134)
│   └── pancreatitis/ (10/100/391)
├── traces/                            # Generated reasoning traces
├── skills/                            # Generated SKILL.md files (sanitized)
│   ├── v1/acute_abdominal_pain.md     # General skill (all pathologies)
│   ├── v2/acute_abdominal_pain.md     # Refined after v1 eval
│   ├── evo/                           # EvoTest-generated episodes
│   └── ...
├── results/                           # Evaluation results from GPU server
├── codes_Hager/                       # Hager's framework (modified)
├── codes_openai_agent/                # OpenAI Agents SDK variant (Approach 6)
├── MIMIC-CDM-IV/                      # Patient data (2,400 curated cases)
├── guidelines/                        # Clinical guideline summaries per pathology
├── docs/                              # Reference materials
│   ├── WORKFLOW.md                    # Step-by-step workflow (local + GPU server)
│   ├── EXAMPLE_WALKTHROUGH.md         # Complete example of one evolution cycle
│   ├── EVOTEST_ADAPTATION.md          # EvoTest integration plan
│   ├── Hager_Rueckert.md              # Paper notes
│   ├── upskill.md                     # upskill blog post reference
│   ├── openai_agents_sdk.md           # OpenAI Agents SDK analysis (Approach 6)
│   ├── improvement_strategy.md        # Strategic analysis: how to best improve the agent
│   ├── qwen3-a3b.md                   # Qwen3-30B-A3B model card
│   ├── medgemma.md                    # MedGemma model card (historical)
│   ├── discharge_summary.md           # Sample discharge summary
│   └── 7583_EvoTest_Evolutionary_Test.pdf  # EvoTest paper
└── .gitignore
```

---

## External Code and Data Paths (this PC)

### MIMIC-IV Data

| Path | Contents |
|---|---|
| `/Users/tianyuhan/Documents/data/mimiciv/3.1/hosp/` | MIMIC-IV 3.1 structured tables (admissions.csv, labevents.csv, patients.csv, d_labitems.csv, etc.) |
| `/Users/tianyuhan/Documents/data/mimiciv/3.1/note/` | MIMIC-IV 3.1 clinical notes (discharge.csv, radiology.csv — uncompressed) |
| `/Users/tianyuhan/Documents/data/mimic-iv-note/2.2/note/` | MIMIC-IV 2.2 notes (discharge.csv.gz, radiology.csv.gz — compressed) |
| `/Users/tianyuhan/Documents/data/mimiciv/3.1/hosp/lab_test_mapping.pkl` | Lab test ID → name mapping (needed by Hager's framework) |

### MIMIC-CDM Curated Datasets

| Path | Contents |
|---|---|
| `/Users/tianyuhan/Documents/GitHub/MIMIC-CDM/data/` | Official MIMIC-CDM release: 4 pathology pkls + lab_test_mapping.pkl + CSVs (discharge_diagnosis, physical_examination, laboratory_tests, radiology_reports, etc.) |
| `/Users/tianyuhan/Documents/GitHub/MIMIC-Clinical-Decision-Making-Dataset/` | Original dataset creation code (CreateDataset.py) + all 4 pathology pkls (clean, first_diag variants) |
| `./MIMIC-CDM-IV/` | In-repo copy: 11 pathology pkls + train/test splits (appendicitis through subarachnoid_hemorrhage) |
| `./data_splits/` | Our train(10)/test(100)/remaining splits per pathology (created by `scripts/split_data.py`) |

### Related Repos (cloned locally)

| Path | Repo | Purpose |
|---|---|---|
| `/Users/tianyuhan/Documents/GitHub/upskill/` | [huggingface/upskill](https://github.com/huggingface/upskill) | Skill generation framework (fastagent_integration.py, propose-skills.sh) |
| `/Users/tianyuhan/Documents/GitHub/reflexion/` | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | Baseline: verbal self-reflection (alfworld, hotpotqa, programming runs) |
| `/Users/tianyuhan/Documents/GitHub/MIMIC-Clinical-Decision-Making-Framework/` | Hager et al. original | Standalone copy of the clinical agent framework |
| `/Users/tianyuhan/Documents/GitHub/MIMIC-CDM/` | Official MIMIC-CDM repo | Contains data/, analysis code, agents |
| `/Users/tianyuhan/Documents/GitHub/MIMIC-CDM-Bench/` | MIMIC-CDM-Bench | Benchmarking variant of Hager's framework |
| `/Users/tianyuhan/Documents/GitHub/MIMIC-ReAct/` | MIMIC-ReAct variant | Has clinical guidelines PDFs in `guidelines/` (WSES appendicitis 2020, Tokyo cholecystitis 2018, AGA/ACG pancreatitis 2024, AGA/WSES diverticulitis 2020) |
| `/Users/tianyuhan/Documents/GitHub/MIMIC-Plain/` | MIMIC-Plain | GRPO/SFT training variant |
| `/Users/tianyuhan/Documents/GitHub/MIMIC_rewoo/` | MIMIC ReWOO agent | Has lab_test_mapping_III/IV.pkl and itemid_ref_ranges |

### Hager's Framework Path Config (for GPU server)

The framework expects a `paths` YAML config with these fields:

```yaml
# configs/paths/server.yaml
base_mimic: ./data_splits/appendicitis          # Directory containing {pathology}_hadm_info_first_diag.pkl
base_models: /path/to/huggingface/cache          # HuggingFace model weights
lab_test_mapping_path: /path/to/lab_test_mapping.pkl  # Lab ID → name mapping
local_logging_dir: ./results                     # Output directory for trajectories + evals
```

Known locations for `lab_test_mapping.pkl`:
- `/Users/tianyuhan/Documents/GitHub/MIMIC-CDM/data/lab_test_mapping.pkl`
- `/Users/tianyuhan/Documents/data/mimiciv/3.1/hosp/lab_test_mapping.pkl`
- `/Users/tianyuhan/Documents/GitHub/MIMIC_rewoo/lab_test_mapping_IV.pkl`

### Key External Resources

| Resource | Path / Location |
|---|---|
| EvoTest paper | `./docs/7583_EvoTest_Evolutionary_Test.pdf` (in-repo) |
| EvoTest repo (to fork) | https://github.com/yf-he/EvoTest (not yet cloned) |
| OpenAI Agents SDK | https://github.com/openai/openai-agents-python — Approach 6 framework |
| OpenAI Agents SDK docs | https://openai.github.io/openai-agents-python/ |
| Clinical guidelines (PDFs) | `/Users/tianyuhan/Documents/GitHub/MIMIC-ReAct/guidelines/` |
| Discharge summaries (for skill grounding) | `Discharge` field in each pkl, or `/Users/tianyuhan/Documents/data/mimiciv/3.1/note/discharge.csv` |

---

## Next Steps (in order)

1. Reproduce baselines with Llama 3.3 70B on all 4 pathologies
2. Implement skill injection in `agent.py` (support Approaches 1, 2, and 5)
3. Implement tool output augmentation in `Actions.py` (Approach 3)
4. Run Option A end-to-end on appendicitis (simplest pathology, Approach 2 first)
5. Test integration approaches independently (ablation study)
6. If lift observed, extend to all 4 pathologies
7. Run Option C for comprehensive skills
8. Compare: Option A vs Option C vs paper mitigations vs hybrid
9. Test across multiple models (Llama 3.3, GPT-3.5, GPT-4)
10. **Approach 6**: Build parallel SDK-based agent prototype (see `docs/openai_agents_sdk.md`)
    - Implement `@function_tool` versions of PE, Labs, Imaging, DiagCrit
    - Add `output_type=DiagnosticResult` for structured output
    - Test agents-as-tools pattern for lab interpretation sub-agent
    - Compare SDK agent vs LangChain agent on same patient set
11. Statistical analysis and write-up
