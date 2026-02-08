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

## Automated EvoTest Loop (`scripts/evotest_clinical.py`)

The manual Steps 1-6 above work but require human effort per iteration.
`evotest_clinical.py` automates the entire loop using EvoTest-style evolutionary
optimization with UCB tree-based exploration and regression protection.

**One episode** = run agent on all 4 pathologies x 10 train patients (40 total)
with a given skill → evaluate → extract trajectories → compute composite score.
The Evolver (Opus) analyzes failures and generates an improved skill. UCB selects
the best parent for the next evolution — if episode 4 regresses, episode 5 can
branch from episode 3 instead of building on a bad skill.

### Prerequisites

1. **Data splits** in place (already done):
   ```
   data_splits/{pathology}/train.pkl                         # 10 patients each
   data_splits/{pathology}/{pathology}_hadm_info_first_diag.pkl  # symlink for Hager
   MIMIC-CDM-IV/lab_test_mapping.pkl
   ```

2. **ANTHROPIC_API_KEY** — set in `.env` at project root or export it:
   ```bash
   echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
   ```

3. **GPU server** — the agent model (Qwen3-30B-A3B, Llama 3.3 70B, etc.) runs
   locally via HuggingFace and needs GPU. The script calls `run.py` as a
   subprocess, so run `evotest_clinical.py` on the GPU server itself.

### Clinical Guidelines

The Evolver automatically loads evidence-based clinical practice guidelines
(PubMed, NICE) to ground evolved skills in peer-reviewed diagnostic and treatment
protocols. Guidelines are loaded from `guidelines/{pathology}/evolver_context.md`
(~3-4KB per pathology, ~15KB total) and injected as a dedicated section in the
Evolver prompt — alongside the discharge summaries and failed trajectories.

**Guidelines are ON by default.** You'll see this in the output:
```
Loaded clinical guidelines (14297 chars) from /path/to/guidelines
```

To regenerate guidelines from the source JSONL:
```bash
python scripts/parse_guidelines.py --input open_guidelines.jsonl --output-dir guidelines
```

To disable guidelines (for ablation comparison):
```bash
python scripts/evotest_clinical.py --episodes 10 --no-guidelines
```

### Quick Validation (Dry Run)

Prints all subprocess commands and the Evolver prompt without executing anything.
Use this to verify paths are correct before committing GPU time:

```bash
python scripts/evotest_clinical.py --dry-run --episodes 2
```

### Running the Full Loop

```bash
# Standard run: 10 episodes, Qwen3-30B-A3B, Opus as Evolver, guidelines ON
python scripts/evotest_clinical.py \
    --episodes 10 \
    --model Qwen3_30B_A3B \
    --evolver-model claude-opus-4-6 \
    --annotate-clinical True
```

Each episode takes ~20-40 min on GPU (40 patients), so 10 episodes = ~3-7 hours.

### Warm-Starting from an Existing Skill

If you already have a skill from `run_experiment_multi.sh` (e.g., v2), use it as
the seed for episode 0 instead of running a cold baseline. Saves ~2 episodes of
convergence:

```bash
python scripts/evotest_clinical.py \
    --episodes 10 \
    --model Qwen3_30B_A3B \
    --initial-skill skills/v2/acute_abdominal_pain.md
```

### Resuming After Interruption

State is saved to `evotest_state/state.json` after every episode. Resume from
where you left off:

```bash
python scripts/evotest_clinical.py --resume --episodes 15
```

### Recommended Settings

| Setting | Value | Why |
|---|---|---|
| `--episodes` | **10-15** | Diminishing returns after ~10 |
| `--model` | **Qwen3_30B_A3B** | Best configured local model; only 3B active params so fast |
| `--evolver-model` | **claude-opus-4-6** | Strongest Evolver; best clinical reasoning |
| `--annotate-clinical` | **True** | Lab annotations (Approach 3) — proven to help |
| `--initial-skill` | a v1/v2 skill if available | Warm-start beats cold-start |
| `--exploration-constant` | **1.0** (default) | Standard UCB; lower to 0.5 for more exploitation |
| `--depth-constant` | **0.8** (default) | Depth decay for exploration |
| `--drop-threshold` | **1.0** (default) | ~15% of max score (6.5); prevents catastrophic regression |

### All CLI Options

```
python scripts/evotest_clinical.py --help

--episodes N           Total episodes to run (default: 10)
--model NAME           Agent model (default: Qwen3_30B_A3B)
--evolver-model NAME   Anthropic model for Evolver (default: claude-opus-4-6)
--annotate-clinical    Enable clinical lab annotations (default: True)
--exploration-constant UCB exploration constant c (default: 1.0)
--depth-constant       UCB depth decay alpha (default: 0.8)
--drop-threshold       Force-best-after-drop threshold (default: 1.0)
--force-best-after-drop / --no-force-best-after-drop
--initial-skill PATH   Seed skill for episode 0 (optional)
--guidelines-dir DIR   Path to guidelines/ directory (default: auto-detect)
--no-guidelines        Disable clinical guidelines in Evolver prompt
--resume               Resume from evotest_state/state.json
--dry-run              Print commands without executing
```

### What to Expect

| Phase | Diagnosis | PE First | Notes |
|---|---|---|---|
| Episode 0 (baseline) | ~40-50% | ~50% | No skill, just the raw agent |
| Episodes 1-3 | ~55-65% | ~80-90% | Biggest gains from "PE first" + lab ordering rules |
| Episodes 4-7 | ~65-75% | ~85-90% | Diminishing returns; lab/imaging scores improve |
| Episodes 8-10+ | ~70-80% | ~85-90% | Plateau; UCB explores alternative branches |

### Output Files

```
skills/evo/
  episode_0.md              # Sanitized skill (episode 0 = baseline or seed)
  episode_0_raw.md          # Raw skill before sanitization
  episode_1.md              # Evolved skill from episode 1
  ...
trajectories/
  evo_ep0_appendicitis.json # Per-pathology trajectory JSON
  evo_ep0_cholecystitis.json
  ...
evotest_state/
  state.json                # Full UCB tree + scores (for --resume)
```

### Final Evaluation on Test Set

After the loop completes, the best skill path is printed (e.g.,
`Best skill: skills/evo/episode_7.md`). Evaluate it on the held-out 100 patients
per pathology:

```bash
BEST_SKILL=skills/evo/episode_7.md  # from the script's final output

for PATHOLOGY in appendicitis cholecystitis diverticulitis pancreatitis; do
    # Point to test split
    cp data_splits/$PATHOLOGY/test.pkl \
       data_splits/$PATHOLOGY/${PATHOLOGY}_hadm_info_first_diag.pkl

    # Run with best skill
    cd codes_Hager/MIMIC-Clinical-Decision-Making-Framework
    python run.py \
        pathology=$PATHOLOGY \
        model=Qwen3_30B_A3B \
        base_mimic=../../data_splits/$PATHOLOGY \
        base_models=$HF_HOME \
        lab_test_mapping_path=../../MIMIC-CDM-IV/lab_test_mapping.pkl \
        local_logging_dir=../../results \
        summarize=True \
        annotate_clinical=True \
        skill_path=../../$BEST_SKILL \
        run_descr=_evotest_best_test100
    cd ../..

    # Evaluate
    python scripts/evaluate_run.py \
        --results_dir $(ls -td results/*${PATHOLOGY}*_evotest_best_test100* | head -1) \
        --pathology $PATHOLOGY \
        --patient_data data_splits/$PATHOLOGY/test.pkl
done
```

### How EvoTest Differs from Linear `run_iterations.sh`

| Aspect | `run_iterations.sh` (v1→v2→v3) | `evotest_clinical.py` |
|---|---|---|
| Exploration | Linear chain only | UCB tree — can branch and backtrack |
| Regression protection | None — v3 builds on v2 even if v2 was bad | Force-best-after-drop reverts to best node |
| Evolver context | Sees only previous version's trajectories | Sees evolution history, failed skills, per-metric targets |
| Skill selection | Always uses the latest | UCB selects highest-potential parent |
| Resumability | Must restart from scratch | `--resume` continues from checkpoint |

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

## Scripts

| Script | Status | Purpose |
|---|---|---|
| `scripts/split_data.py` | Done | Split MIMIC-CDM pkl into train/test/remaining |
| `scripts/prepare_split_for_hager.py` | Done | Copy split as `{pathology}_hadm_info_first_diag.pkl` |
| `scripts/extract_trajectories.py` | Done | Parse results pkl → JSON with trajectories + scores |
| `scripts/evaluate_run.py` | Done | Run PathologyEvaluator on results pkl |
| `scripts/evolve_skill.py` | Done | Evolver: analyze trajectories → generate improved skill |
| `scripts/sanitize_skill.py` | Done | Remove disease name leakage from skills |
| `scripts/compare_runs.py` | Done | Side-by-side comparison of two runs |
| `scripts/parse_guidelines.py` | Done | Extract disease-specific guidelines from `open_guidelines.jsonl` |
| `scripts/run_experiment.sh` | Done | Single-pathology evolution cycle (bash) |
| `scripts/run_experiment_multi.sh` | Done | Multi-pathology evolution cycle (bash) |
| `scripts/run_iterations.sh` | Done | Multi-version linear orchestrator (bash) |
| `scripts/evotest_clinical.py` | Done | **Automated EvoTest loop** with UCB tree (see section above) |

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

## File Structure

```
mimic_skills/
  CLAUDE.md                          # Master plan
  docs/
    EVOTEST_ADAPTATION.md            # EvoTest integration plan
    WORKFLOW.md                      # This file
    EXAMPLE_WALKTHROUGH.md           # Concrete walkthrough of one cycle
  open_guidelines.jsonl              # 37,970 clinical guidelines (gitignored)
  scripts/
    split_data.py                    # Split MIMIC-CDM pkl → train/test/remaining
    prepare_split_for_hager.py       # Copy split as *_hadm_info_first_diag.pkl
    extract_trajectories.py          # Parse results pkl → JSON
    evaluate_run.py                  # Run PathologyEvaluator
    evolve_skill.py                  # Evolver: trajectories → improved skill
    sanitize_skill.py                # Remove disease name leakage
    compare_runs.py                  # Side-by-side comparison
    parse_guidelines.py              # Extract disease-specific guidelines from JSONL
    run_experiment.sh                # Single-pathology bash pipeline
    run_experiment_multi.sh          # Multi-pathology bash pipeline
    run_iterations.sh                # Multi-version linear orchestrator
    evotest_clinical.py              # Automated EvoTest loop with UCB tree
  guidelines/
    all_pathologies_context.md       # Combined guidelines (~15KB)
    appendicitis/
      evolver_context.md             # Condensed for Evolver (~3KB)
      summary.md                     # Fuller summary (~10KB)
      full/                          # Complete guideline text (gitignored)
    cholecystitis/                   # Same structure
    diverticulitis/                  # Same structure
    pancreatitis/                    # Same structure
  data_splits/
    appendicitis/
      train.pkl (10)
      test.pkl (100)
      remaining.pkl (809)
      appendicitis_hadm_info_first_diag.pkl  # symlink for Hager
    cholecystitis/ ...
    diverticulitis/ ...
    pancreatitis/ ...
  skills/
    v1/acute_abdominal_pain.md       # Linear evolution (run_experiment_multi.sh)
    v2/acute_abdominal_pain.md
    evo/                             # EvoTest evolution (evotest_clinical.py)
      episode_0.md                   # Sanitized skills per episode
      episode_0_raw.md              # Raw skills before sanitization
      episode_1.md
      ...
  trajectories/
    baseline_appendicitis_train10.json     # Linear pipeline trajectories
    evo_ep0_appendicitis.json              # EvoTest trajectories
    evo_ep0_cholecystitis.json
    ...
  evotest_state/
    state.json                       # UCB tree checkpoint (for --resume)
  results/                           # Raw output from GPU server
  comparisons/                       # Comparison reports
  codes_Hager/...                    # Framework (modified agent.py)
  MIMIC-CDM-IV/...                   # Original full data
  EvoTest/...                        # Reference EvoTest repo (not used at runtime)
```
