# EvoTest Adaptation for Clinical Decision-Making

## TL;DR

We plan to adapt [EvoTest](https://github.com/yf-he/EvoTest) (ICLR 2026 submission) — an
evolutionary test-time learning framework — to **automate Option A's iterative
skill refinement loop**. Instead of a human analyzing agent failures and manually
selecting cases to refine skills, an **Evolver Agent (Opus/o3)** will analyze the
full diagnostic trajectory transcripts and PathologyEvaluator scores, then
automatically evolve the entire agent configuration (prompt, memory,
hyperparameters, tool-use logic) across episodes.

---

## 1. EvoTest Paper Summary

**Paper**: "EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic
Systems" ([arXiv:2510.13220](https://arxiv.org/abs/2510.13220))

**Setting**: An agent plays the same text-adventure game (Jericho) for K=50
consecutive episodes, improving from one episode to the next.

**Architecture**:
- **Actor Agent**: Plays one full episode with a fixed configuration
- **Evolver Agent**: After each episode, analyzes the trajectory transcript and
  produces an improved configuration for the next episode

**Agentic Configuration** `chi = (p, M, h, u)`:

| Component | What it is | What the Evolver does to it |
|---|---|---|
| Policy prompt (p) | High-level strategy instructions | Rewrites based on what worked/failed |
| Memory (M) | Success/failure state-action database | Logs effective patterns, prunes failures |
| Hyperparameters (h) | Temperature, exploration strength | Adjusts based on agent behavior |
| Tool-use routines (u) | State extraction code + memory access rules | Refines how agent uses its tools |

**Selection**: UCB (Upper Confidence Bound) algorithm picks the next configuration
from {parent, children}, balancing exploration of untested mutations vs
exploitation of proven configs. This prevents catastrophic regression.

**Key results** (Table 1): EvoTest achieves 0.47/0.50 avg AUC (Gemini/Claude)
across 6 games, beating all baselines including Reflexion (0.32/0.34),
EvoPrompt (0.34/0.36), and even online RL methods (GRPO: 0.30).

**Why it works**: Whole-system evolution (not just prompt editing) + narrative-based
credit assignment from full transcripts (not just scalar rewards) + UCB for
stable learning.

---

## 2. Why This Maps to Our Clinical Setting

### Direct Mapping

| EvoTest (Jericho Games) | Our Setting (Hager's Clinical Agent) |
|---|---|
| Actor plays text-adventure game | ZeroShotAgent diagnoses patient (ReAct loop) |
| Episode = 1 game playthrough (110 steps) | Episode = 1 patient batch (N patients) |
| Trajectory = state, action, reward per step | Trajectory = Thought, Action, Observation per turn |
| Scalar reward (game points) | **Rich multi-metric** (PathologyEvaluator: 12+ metrics) |
| Policy prompt = game strategy | CHAT_TEMPLATE + skill content |
| Memory = state-action lookup table | Successful/failed clinical reasoning patterns |
| Tool-use = Python state extractor | How agent uses PE, Labs, Imaging, DiagCrit tools |
| Evolver = o3 analyzing game transcript | Evolver = Opus analyzing clinical trajectories |

### Why Our Setting Is Actually Better for EvoTest

1. **Richer reward signal**: PathologyEvaluator gives 12+ separate metrics vs
   Jericho's single scalar score. The Evolver gets precise failure info like
   "missed inflammation labs" or "skipped physical exam", not just "+10 points".

2. **Structured transcripts**: Thought/Action/Observation format is cleaner than
   raw game text. Credit assignment is easier.

3. **Clinical domain knowledge**: Opus/o3 have strong medical knowledge. The
   Evolver can reason about *why* a clinical decision was wrong (e.g., "ordered
   CT before US for suspected cholecystitis — guidelines recommend US first").

### Key Difference: Episode Structure

- **EvoTest**: Replays the SAME game 50 times. Progress is cumulative because the
  agent learns the specific game walkthrough.
- **Our setting**: Each patient is different. Cannot "replay" the same patient.

**Solution**: Define an "episode" as running the agent on a **fixed validation set
of N patients**. Score = aggregate PathologyEvaluator metrics across all N.
The Evolver sees all N transcripts. The evolved config must generalize across
patients (no patient-specific memorization). This is strictly harder but
produces more robust, generalizable skills.

---

## 3. What Self-Evolution Replaces

### Before (Option A — Manual)

```
v1: Human picks seed case
    → Opus generates trace → upskill generates skill
    → Run Hager's agent → PathologyEvaluator scores
    → Human analyzes failures
    → Human selects failure-targeted case

v2: Human picks failure-targeted case
    → Opus generates trace → upskill refines skill
    → Run Hager's agent → PathologyEvaluator scores
    → Human analyzes failures
    ...

v3-v5: Repeat until plateau (limited by human time/effort)
```

### After (EvoTest-Style — Automated)

```
Episode 1: Run agent on all N patients with initial config chi^(1)
           → Collect all trajectory transcripts + PathologyEvaluator scores
    Evolver (Opus/o3):
      - Analyze transcripts: "Agent missed PE first in 40% of cases"
      - Prompt mutation: Add rule "ALWAYS do Physical Examination first"
      - Memory update: Log {elevated_WBC → order_inflammation_labs, +1}
      - Hyperparameter: Lower temperature (agent was hallucinating tools)
      - Tool-use: "Check labs before ordering imaging"
    → Produce child config chi^(2)
    UCB: Select from {chi^(1), chi^(2)}

Episode 2: Run agent with chi^(selected) on same N patients
    Evolver: Produce chi^(3)
    UCB: Select from {chi^(1), chi^(2), chi^(3)}

...

Episode K: Converged config with best aggregate score
```

### Advantages of Automated Self-Evolution

| Dimension | Option A (manual) | EvoTest-style |
|---|---|---|
| Human effort per iteration | High | **Zero** |
| Iterations possible | ~3-5 | **50+** |
| What evolves | Skill content only | **Entire system** (prompt + memory + hyperparams + tools) |
| Failure targeting | Human intuition | **Automated credit assignment** via transcript analysis |
| Regression protection | None | **UCB selection** (proven configs as fallback) |
| Cost per iteration | 1 upskill call + 1 eval run | 1 Evolver LLM call + 1 eval run |

---

## 4. Implementation Plan

### Repository to Fork

**[github.com/yf-he/EvoTest](https://github.com/yf-he/EvoTest)**

```
EvoTest/
  src/
    our_agent.py      # Core: Actor + Evolver + UCB + memory (~500 LOC)
    evaluation.py      # Episode loop + logging (~200 LOC)
    env.py             # Jericho game wrapper → REPLACE with clinical env
    naive_agent.py     # Static baseline
    summary_agent.py   # Summary baseline
    memory_agent.py    # Memory baseline
    rag_agent.py       # RAG baseline
    openai_helpers.py  # LLM API wrapper (keep, works with OpenRouter)
    utils.py           # Helpers
  main.py              # CLI entry point
```

Requirements: Python 3.10+, OpenRouter API access (supports Opus, o3, Gemini, etc.)

### Files to Modify

| File | Change | Effort |
|---|---|---|
| `src/env.py` | Replace Jericho wrapper with clinical environment: load patient cases from MIMIC-CDM JSON, run Hager's `build_agent_executor_ZeroShot`, call `PathologyEvaluator` | Medium |
| `src/our_agent.py` | Replace game actions with clinical tool vocabulary. Adapt Evolver master prompt for medical domain (see Section 5). Replace state extractor with clinical state summarizer. | Medium |
| `src/evaluation.py` | Adapt episode loop: each "episode" runs N patients. Scoring = aggregate PathologyEvaluator metrics. Log per-patient details. | Low |
| `main.py` | Add clinical args: `--pathology`, `--patient_batch`, `--base_mimic`, `--skill_inject`, `--lab_test_mapping_path` | Low |
| New: `src/clinical_env.py` | Wrapper around Hager's framework: loads patient data, builds agent executor, runs agent, evaluates trajectory, returns structured score dict | Medium |

### Phase-by-Phase

**Phase 1: Environment wrapper** (Day 1 morning)
- Create `clinical_env.py` that wraps Hager's agent pipeline
- Input: patient hadm_info dict + agent configuration chi
- Output: trajectory transcript (list of Thought/Action/Observation) + PathologyEvaluator score dict
- Test on 1 patient to verify the full loop works

**Phase 2: Adapt Evolver** (Day 1 afternoon)
- Rewrite the Evolver master prompt for clinical domain (Section 5 below)
- Adapt memory structure: success = (clinical_state, action, score_delta), failure = (clinical_state, action, reason)
- Replace state extractor with clinical progress summarizer (what findings gathered so far, differential narrowing)

**Phase 3: Adapt evaluation loop** (Day 2 morning)
- Define "episode" = run agent on fixed batch of N patients
- Aggregate scoring: weighted combination of PathologyEvaluator metrics
- UCB selection across configurations
- Logging: per-patient transcripts + scores + evolved configs

**Phase 4: Run experiments** (Day 2 afternoon onward)
- Start with appendicitis (simplest pathology, ~100 patients in MIMIC-CDM)
- 50 episodes, Opus as Evolver, Llama 3.3 70B as Actor
- Compare: Static → Reflexion → EvoTest-Clinical
- Extend to cholecystitis, diverticulitis, pancreatitis

---

## 5. Clinical Evolver Master Prompt (Draft)

This adapts EvoTest's Appendix H for the medical domain:

```
You are a clinical AI system optimizer. Your task is to analyze diagnostic
trajectory transcripts and evaluation scores from a batch of patient cases,
then generate a new, improved configuration for the diagnostic agent.

The agent uses a ReAct loop (Thought → Action → Observation) with these tools:
- Physical Examination: returns exam findings
- Laboratory Tests: order specific lab tests, get results
- Imaging: order scan (modality + region), get radiology report
- Diagnostic Criteria: look up diagnostic guidelines for a pathology

The agent must output: Final Diagnosis + Treatment

The agent's configuration has four components:
1. Clinical skill prompt (diagnostic strategy and workflow rules)
2. Memory (successful/failed clinical reasoning patterns)
3. Hyperparameters (temperature for LLM inference)
4. Tool-use guidance (when/how to use PE, Labs, Imaging)

Current clinical skill prompt:
"{cur_prompt}"

PathologyEvaluator results across {N} patients:
{scores_summary}
(Metrics: Diagnosis accuracy, PE ordering, Lab appropriateness, Imaging score,
Treatment correctness, Action parsing errors, Invalid tool requests, Rounds used)

Patient trajectory transcripts:
--- TRAJECTORIES START ---
{formatted_trajectories}
--- TRAJECTORIES END ---

{negative_patterns_section}

Generate a new, improved configuration:

PART 1: Generate improved clinical skill prompt.
- Create a "Diagnostic Workflow" section: step-by-step procedure based on
  patterns that led to correct diagnoses.
- Create a "Common Mistakes to Avoid" section: patterns that led to failures.
- Create a "Lab Interpretation Guide" section if lab-related errors were common.
- Be specific about tool ordering (PE first, then targeted labs, then imaging).

PART 2: Generate memory updates.
Extract all state-action pairs that led to correct diagnoses or score improvements.
Format as JSON: [{"clinical_state": "...", "action": "...", "score_delta": N}]
Also extract failure patterns: [{"clinical_state": "...", "action": "...", "reason": "..."}]

PART 3: Suggest hyperparameter adjustments.
- If the agent hallucinated tools or gave invalid actions, lower temperature.
- If the agent was repetitive or stuck in loops, raise temperature.
- Output: {"temperature": X}

PART 4: Generate tool-use guidance.
Propose a single rule for when/how the agent should consult its memory or
prioritize certain tools based on observed failure patterns.
```

---

## 6. Scoring Function Design

EvoTest uses scalar game score. We need to map PathologyEvaluator's multi-metric
output to a single scalar for UCB selection.

**Proposed weighted score** (per patient):

```python
score = (
    3.0 * scores["Diagnosis"]                    # Primary goal
  + 1.0 * scores["Physical Examination"]          # PE done first
  + 0.5 * scores["Late Physical Examination"]     # PE done at all
  + 1.0 * (scores["Laboratory Tests"] / max_lab)  # Normalized lab score
  + 1.0 * (scores["Imaging"] / 2.0)               # Normalized imaging (max 2)
  + 1.0 * treatment_score                          # Computed from answers
  - 0.5 * scores["Invalid Tools"]                  # Penalize hallucination
  - 0.5 * scores["Action Parsing"]                 # Penalize bad formatting
)
```

**Episode score** = mean(patient_scores) across all N patients in the batch.

This gives ~8.0 max per patient. UCB operates on the episode-level aggregate.

---

## 7. Experiment Matrix

Once the adaptation is working, run this comparison:

| Experiment | Description | Episodes |
|---|---|---|
| Static | Hager's original agent, no learning | 1 (baseline) |
| Reflexion | Append self-reflection after each batch | 50 |
| EvoTest-Prompt | Evolve only the prompt (ablation) | 50 |
| EvoTest-Full | Full system evolution (prompt + memory + hyperparams + tools) | 50 |
| EvoTest-Full + Option C seed | Start from Option C synthesized skill, then evolve | 50 |

Run per pathology: appendicitis, cholecystitis, diverticulitis, pancreatitis.
Models: Llama 3.3 70B (Actor), Opus (Evolver).

---

## 8. Key References

| Paper | Relevance | Code |
|---|---|---|
| EvoTest (He et al., 2025) | Primary method we're adapting | [github.com/yf-he/EvoTest](https://github.com/yf-he/EvoTest) |
| Reflexion (Shinn et al., 2023) | Baseline: verbal self-reflection | [github.com/noahshinn/reflexion](https://github.com/noahshinn/reflexion) |
| EvoAgent (Yuan et al., 2024) | Evolutionary multi-agent generation | [github.com/siyuyuan/evoagent](https://github.com/siyuyuan/evoagent) |
| EvoAgentX | General self-evolving agent framework | [github.com/EvoAgentX/EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) |
| Hager et al. (2024) | Original clinical agent we're improving | codes_Hager/ in this repo |
| upskill (HuggingFace) | Skill generation (used for initial seed) | [github.com/huggingface/upskill](https://github.com/huggingface/upskill) |

---

## 9. Open Questions for Tomorrow

1. **Episode batch size**: How many patients per episode? All ~100 per pathology
   (thorough but slow) or a subsample of 20 (faster iteration)?

2. **Evolver model**: Opus vs o3? EvoTest Table 4 shows o3 is best. We could use
   Opus (our primary model) or try o3 for comparison.

3. **Fixed vs rotating patient batch**: Should every episode use the exact same
   patients (like replaying Jericho), or rotate patients to test generalization?
   Fixed is more comparable to EvoTest; rotating is more realistic.

4. **Integration with upskill**: Use Option C's synthesized skill as chi^(1)
   initial config? Or start from a generic clinical prompt and let EvoTest
   discover the skill from scratch (more comparable to paper's setup)?

5. **Approach integration**: EvoTest naturally handles Approach 1 (system prompt)
   and Approach 2 (examples). Can we also evolve Approach 3 (tool output
   augmentation) and Approach 4 (enhanced DiagCrit)?

---

## 10. File Locations in This Repo

```
mimic_skills/
  CLAUDE.md                          # Main project plan (Option A, C, integration approaches)
  docs/
    EVOTEST_ADAPTATION.md            # This file
    7583_EvoTest_Evolutionary_Test.pdf  # The EvoTest paper
  codes_Hager/...                    # Hager's clinical agent framework
  MIMIC-CDM-IV/                      # Patient data (curated test cases)
```

Tomorrow's first step: Fork EvoTest repo, create `src/clinical_env.py` wrapper.
