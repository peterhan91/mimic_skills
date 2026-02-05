# Workflow: Iterating with Claude Code + GPU Server

## Overview

Two machines, two roles:

| Machine | Role | What Runs |
|---|---|---|
| **Local (Mac + Claude Code)** | Brain: plan, code, analyze, evolve | Code editing, Evolver Agent (Opus API calls), result analysis |
| **GPU Server** | Muscle: run the agent | Hager's agent with Llama 3.3 70B (4-bit), PathologyEvaluator |

The loop: write code locally → push to server → run agent → pull results back → analyze → repeat.

---

## Data Splits (already created)

```
data_splits/
  appendicitis/
    train.pkl           10 admissions — Evolver iterates on these
    test.pkl           100 admissions — held-out final evaluation
    remaining.pkl      809 admissions — Option C synthesis pool
    *_hadm_ids.txt     ID lists for inspection
  cholecystitis/       10 / 100 / 514
  diverticulitis/      10 / 100 / 134
  pancreatitis/        10 / 100 / 391
```

**Note**: IDs are hospital admission IDs (hadm_id), not patient IDs (subject_id).
A few patients have multiple admissions, but it's nearly 1:1 for our 4 pathologies.

Recreate with: `python scripts/split_data.py --seed 42 --n_train 10 --n_test 100`

---

## Step-by-Step: Concrete Example (Appendicitis)

### Step 0: Setup GPU Server

```bash
# On GPU server
git clone <this-repo-url> mimic_skills
cd mimic_skills

# Install Hager's framework dependencies
cd codes_Hager/MIMIC-Clinical-Decision-Making-Framework
pip install -r requirements.txt  # or equivalent

# Verify Llama 3.3 70B weights are available
# The model loads from HuggingFace cache or base_models path
# Needs ~40GB VRAM at 4-bit (one A100-80GB or two A6000-48GB)

# Create a paths config pointing to your server paths
cat > configs/paths/server.yaml << 'EOF'
# @package _global_
defaults:
  - _self_

base_mimic: /path/to/mimic_skills/data_splits/appendicitis
base_models: /path/to/huggingface/cache
lab_test_mapping_path: /path/to/mimic_skills/codes_Hager/MIMIC-Clinical-Decision-Making-Framework/dataset/lab_test_mapping.pkl
local_logging_dir: /path/to/mimic_skills/results
EOF
```

### Step 1: Baseline Run — Static Agent on Train Split (GPU Server)

Run Hager's original agent (no skill) on the 10 training admissions.

**Purpose**: Establish what the agent does without any help. Get baseline
trajectories and scores to give the Evolver a starting point.

```bash
# On GPU server
cd codes_Hager/MIMIC-Clinical-Decision-Making-Framework

# Run on the 10 train admissions only
python run.py \
  pathology=appendicitis \
  model=Llama3.3Instruct70B \
  paths=server \
  base_mimic=/path/to/mimic_skills/data_splits/appendicitis \
  summarize=True \
  run_descr=_baseline_train10
```

**What changes for our split**: Hager's `run.py` loads the full pkl and iterates
all keys. We need ONE small modification — either:

(a) Modify `run.py` to accept a `patient_list_path` arg (it already has this!
line 25 of config.yaml) and filter by our train_hadm_ids.txt, OR

(b) Just point `base_mimic` to our split pkl (train.pkl already contains
only 10 admissions, but it's named differently from the convention
`{pathology}_hadm_info_first_diag.pkl`).

**Simplest approach**: Rename or symlink so the framework finds it:

```bash
# On GPU server — make train split loadable by Hager's framework
cd /path/to/mimic_skills/data_splits/appendicitis
cp train.pkl appendicitis_hadm_info_first_diag.pkl
```

Now `run.py` with `base_mimic=data_splits/appendicitis` will load only the
10 train admissions.

**Output**: Results pkl in `results/` directory with per-patient trajectories and scores.

### Step 2: Extract Trajectories + Scores (Local or Server)

```bash
# Parse the results pkl to get human-readable transcripts
python scripts/extract_trajectories.py \
  --results_pkl results/appendicitis_*_baseline_train10/appendicitis_*_results.pkl \
  --eval_pkl results/appendicitis_*_baseline_train10/appendicitis_*_eval.pkl \
  --output trajectories/baseline_appendicitis_train10.json
```

**We need to write this script** (`scripts/extract_trajectories.py`). It should:
1. Load the results pkl (contains agent trajectories)
2. Load the eval pkl (contains PathologyEvaluator scores)
3. Output a JSON with per-admission: trajectory text + scores dict

### Step 3: Evolver Analyzes Trajectories (Local — Claude Code)

This is where Claude Code / Opus acts as the Evolver Agent.

**Input to Evolver** (via prompt or script):
- 10 trajectory transcripts (Thought/Action/Observation chains)
- 10 PathologyEvaluator score dicts
- Current config (initially: Hager's default CHAT_TEMPLATE, no skill)

**Evolver produces**:
1. An evolved skill prompt (to inject into CHAT_TEMPLATE)
2. Clinical memory entries (success/failure patterns)
3. Hyperparameter suggestions (temperature)
4. Tool-use rules

**For the first iteration, this can be done manually with Claude Code**:

```
Read trajectories/baseline_appendicitis_train10.json

Analyze the 10 diagnostic trajectories. For each:
- Did the agent get the diagnosis right?
- Did it perform PE first?
- Were the right labs ordered?
- Was imaging appropriate?
- Was treatment adequate?

Then generate an improved clinical skill prompt that addresses
the most common failure patterns. Output as skills/v1/appendicitis.md
```

Later, we automate this with the EvoTest Evolver loop.

### Step 4: Inject Skill and Re-run on Train (GPU Server)

```bash
# On GPU server
# Copy the evolved skill to server
scp skills/v1/appendicitis.md server:/path/to/mimic_skills/skills/

# Run with skill injected (requires the agent.py modification from CLAUDE.md)
python run.py \
  pathology=appendicitis \
  model=Llama3.3Instruct70B \
  paths=server \
  base_mimic=/path/to/mimic_skills/data_splits/appendicitis \
  summarize=True \
  skill_path=/path/to/skills/v1/appendicitis.md \
  run_descr=_v1_train10
```

### Step 5: Compare and Iterate (Local)

```bash
# Pull results back
scp -r server:/path/to/results/appendicitis_*_v1_train10 results/

# Extract and compare
python scripts/compare_runs.py \
  --baseline results/appendicitis_*_baseline_train10/ \
  --evolved results/appendicitis_*_v1_train10/ \
  --output comparisons/v1_vs_baseline.md
```

Did scores improve? If yes → continue evolving. If no → Evolver re-analyzes.

**Repeat Steps 3-5** for v2, v3, ... until convergence.

### Step 6: Final Evaluation on Held-Out Test (GPU Server)

Once the skill has converged (typically 3-10 iterations):

```bash
# On GPU server — prepare test split
cd /path/to/mimic_skills/data_splits/appendicitis
cp test.pkl appendicitis_hadm_info_first_diag.pkl  # overwrite train symlink

# Run baseline on test
python run.py pathology=appendicitis model=Llama3.3Instruct70B \
  paths=server base_mimic=data_splits/appendicitis \
  run_descr=_baseline_test100

# Run best evolved skill on test
python run.py pathology=appendicitis model=Llama3.3Instruct70B \
  paths=server base_mimic=data_splits/appendicitis \
  skill_path=skills/v_best/appendicitis.md \
  run_descr=_vbest_test100
```

Compare on 100 held-out admissions. This is the number we report.

---

## What Claude Code Does vs What GPU Server Does

### Claude Code (Local Mac)

| Task | How |
|---|---|
| Edit code (agent.py, Actions.py, scripts) | Claude Code Edit/Write tools |
| Write Evolver prompts | Direct in conversation |
| Analyze trajectories | Read JSON + reason about failures |
| Generate evolved skills | Write to skills/v{N}/{pathology}.md |
| Compare evaluation results | Read result pkls + compute diffs |
| Plan next iteration | Based on score changes |

### GPU Server

| Task | How |
|---|---|
| Run Hager's agent (Llama 3.3 70B) | `python run.py ...` |
| PathologyEvaluator scoring | Automatic (part of run.py pipeline) |
| Store trajectories + results | Output pkl files |

### Sync Between Machines

```bash
# Push code changes to server
rsync -avz --exclude='*.pkl' --exclude='results/' \
  /Users/tianyuhan/Documents/GitHub/mimic_skills/ \
  server:/path/to/mimic_skills/

# Pull results from server
rsync -avz server:/path/to/mimic_skills/results/ \
  /Users/tianyuhan/Documents/GitHub/mimic_skills/results/
```

Or use git:
```bash
# Local: commit code changes
git add -A && git commit -m "v2 skill + agent.py modifications"
git push

# Server: pull and run
git pull && python run.py ...

# Server: commit results
git add results/ && git commit -m "v2 results" && git push

# Local: pull results
git pull
```

---

## Scripts to Write (before starting experiments)

### Priority 1: Required for Step 1

**`scripts/prepare_split_for_hager.py`** — Makes our split pkl loadable by
Hager's `run.py` without modifying run.py:

```python
"""
Copies train.pkl or test.pkl as {pathology}_hadm_info_first_diag.pkl
so Hager's framework can load it directly.

Usage:
  python scripts/prepare_split_for_hager.py --pathology appendicitis --split train
  python scripts/prepare_split_for_hager.py --pathology appendicitis --split test
"""
```

### Priority 2: Required for Step 2

**`scripts/extract_trajectories.py`** — Parses Hager's output pkls into
readable JSON for the Evolver to analyze:

```python
"""
Extracts trajectory transcripts + PathologyEvaluator scores from results.

Usage:
  python scripts/extract_trajectories.py \
    --results_dir results/appendicitis_ZeroShot_*/ \
    --output trajectories/episode_1.json

Output format:
{
  "admissions": [
    {
      "hadm_id": 29668508,
      "trajectory": [
        {"thought": "...", "action": "Physical Examination", "observation": "..."},
        {"thought": "...", "action": "Laboratory Tests", "action_input": [...], "observation": "..."},
        ...
      ],
      "prediction": "Final Diagnosis: Acute Appendicitis\nTreatment: ...",
      "scores": {"Diagnosis": 1, "Physical Examination": 1, "Laboratory Tests": 1, ...},
      "answers": {"Diagnosis": "Acute Appendicitis", "Treatment": "...", ...}
    },
    ...
  ],
  "aggregate": {
    "diagnosis_accuracy": 0.7,
    "pe_first_rate": 0.5,
    ...
  }
}
"""
```

### Priority 3: Required for Step 5

**`scripts/compare_runs.py`** — Side-by-side comparison of two runs:

```python
"""
Compares two evaluation runs and outputs a markdown summary.

Usage:
  python scripts/compare_runs.py \
    --baseline results/appendicitis_*_baseline/ \
    --evolved results/appendicitis_*_v1/ \
    --output comparisons/v1_vs_baseline.md
"""
```

### Priority 4: For automated EvoTest loop (later)

**`scripts/evotest_loop.py`** — Orchestrates the full Act-Evolve cycle:

```python
"""
Automated EvoTest loop for clinical decision-making.

For each episode:
1. Run Hager's agent on train split (calls run.py on GPU server via SSH)
2. Extract trajectories and scores
3. Call Evolver (Opus API) with transcripts + scores
4. Parse Evolver output into new config (skill, memory, hyperparams)
5. UCB selection of best config
6. Repeat

Usage:
  python scripts/evotest_loop.py \
    --pathology appendicitis \
    --episodes 20 \
    --evolver_model claude-opus-4-5-20251101 \
    --gpu_server user@server
"""
```

---

## Tomorrow's Concrete Checklist

### Morning (Local — Claude Code)

- [ ] Implement skill injection in `agent.py` (`skill_path` + `skill_inject` params)
- [ ] Write `scripts/prepare_split_for_hager.py`
- [ ] Write `scripts/extract_trajectories.py`
- [ ] Push code to GPU server

### Afternoon (GPU Server)

- [ ] Verify Hager's framework runs: `python run.py` on 1 patient
- [ ] Run baseline on 10 train admissions (appendicitis)
- [ ] Pull results back to local

### Evening (Local — Claude Code)

- [ ] Extract trajectories from baseline results
- [ ] Analyze with Claude: identify failure patterns
- [ ] Generate v1 skill for appendicitis
- [ ] Push skill to server, run v1 on train
- [ ] Compare v1 vs baseline

---

## File Structure After Setup

```
mimic_skills/
  CLAUDE.md                        # Master plan
  EVOTEST_ADAPTATION.md            # EvoTest integration plan
  WORKFLOW.md                      # This file
  scripts/
    split_data.py                  # Done ✓
    prepare_split_for_hager.py     # TODO
    extract_trajectories.py        # TODO
    compare_runs.py                # TODO
    evotest_loop.py                # TODO (later)
  data_splits/                     # Done ✓
    appendicitis/
      train.pkl (10)
      test.pkl (100)
      remaining.pkl (809)
    cholecystitis/ ...
    diverticulitis/ ...
    pancreatitis/ ...
  skills/
    v1/appendicitis.md             # First evolved skill
    v2/appendicitis.md             # Second iteration
    ...
  trajectories/
    baseline_appendicitis.json     # Parsed trajectories for Evolver
    v1_appendicitis.json
    ...
  results/                         # Raw output from GPU server
    appendicitis_ZeroShot_*_baseline_train10/
    appendicitis_ZeroShot_*_v1_train10/
    ...
  comparisons/
    v1_vs_baseline.md
    ...
  codes_Hager/...                  # Framework (modified agent.py)
  MIMIC-CDM-IV/...                 # Original full data
```
