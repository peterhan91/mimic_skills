# OpenAI Agents SDK Clinical Diagnostic Agent

A multi-agent clinical diagnostic system built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Replaces Hager et al.'s LangChain ZeroShotAgent with structured output, function-call-validated tools, and specialist sub-agents — while using the same patient data and PathologyEvaluator for apples-to-apples comparison.

## Architecture

```
                  +-----------------------+
                  |  Clinical Diagnostician|  (Orchestrator)
                  |  model: gpt-4o        |
                  |  output: DiagnosticResult (Pydantic)
                  +----------+------------+
                             |
          +------------------+------------------+
          |          |           |         |          |            |
   physical_   laboratory_   imaging  diagnostic  interpret_    challenge_
   examination   tests                 criteria   lab_results   diagnosis
   (tool)       (tool)       (tool)    (tool)     (sub-agent)   (sub-agent)
                                                   gpt-4o-mini   gpt-4o-mini
                                                   -> LabInterp  -> ChallengerFeedback
```

**What the SDK fixes vs LangChain:**

| Failure Mode (Hager et al.) | LangChain | SDK |
|---|---|---|
| Tool hallucination (every 2-5 pts) | Fuzzy match + penalty | Eliminated: `@function_tool` validates JSON schema |
| Phrasing sensitivity (18% swing) | Format instructions | Eliminated: `output_type=DiagnosticResult` enforces schema |
| Lab interpretation (26-77%) | Approach 3 annotations | Lab Interpreter sub-agent + annotations |
| Skip PE (47% skip or late) | Prompt rules | `is_enabled` + `tool_input_guardrail` blocks labs/imaging before PE |
| Treatment gaps | Free-text parsing | `treatment` + `severity` required fields |

## Prerequisites

```bash
pip install openai-agents pandas anthropic
```

For non-OpenAI models (Claude, Llama, Qwen):
```bash
pip install "openai-agents[litellm]"
```

**Environment variables:**

| Variable | Required for | Example |
|---|---|---|
| `OPENAI_API_KEY` | Running the agent with OpenAI models | `sk-...` |
| `ANTHROPIC_API_KEY` | EvoTest Evolver (skill evolution) | `sk-ant-...` |

Or place them in a `.env` file at the project root.

**Data requirements:**

- `data_splits/` directory with train/test splits (created by `scripts/split_data.py`)
- `MIMIC-CDM-IV/lab_test_mapping.pkl` (lab ID-to-name mapping)

## Quick Start

### Run on a single patient (smoke test)

```bash
cd codes_openai_agent
python run.py --pathology appendicitis --max-patients 1
```

### Run on the train split (10 patients)

```bash
python run.py --pathology appendicitis --split train
```

### Run on the test split (100 patients)

```bash
python run.py --pathology appendicitis --split test
```

### Inject a skill

```bash
python run.py --pathology appendicitis \
    --skill-path ../skills/v2/acute_abdominal_pain.md
```

### Use a non-OpenAI model via LiteLLM

```bash
# Claude
python run.py --pathology appendicitis \
    --litellm-model anthropic/claude-sonnet-4-5-20250929

# Local vLLM (Qwen, Llama, etc.)
python run.py --pathology appendicitis \
    --litellm-model openai/Qwen3-30B-A3B \
    --litellm-base-url http://gpu-server:8000/v1
```

### Full `run.py` options

```
--pathology       appendicitis | cholecystitis | diverticulitis | pancreatitis
--model           OpenAI model name (default: gpt-4o)
--litellm-model   LiteLLM model string for non-OpenAI models
--litellm-base-url  Base URL for self-hosted models
--data-dir        Override data directory (default: ../data_splits/{pathology})
--split           train | test | full (default: train)
--lab-mapping     Override lab_test_mapping.pkl path
--output-dir      Output directory (default: ../results)
--annotate-clinical / --no-annotate-clinical  Lab annotations (default: on)
--skill-path      Path to SKILL.md for injection
--max-patients    Limit number of patients
--max-turns       Agent turn limit per patient (default: 20)
--seed            Random seed (default: 2023)
```

## Output

Each run creates a directory under `results/` containing:

```
results/{pathology}_sdk_{model}_{timestamp}/
    *_results.pkl      # Per-patient diagnosis, treatment, confidence, trajectory length
    *_eval.pkl         # Per-patient PathologyEvaluator scores and answers
    *_summary.json     # Human-readable JSON with config + all results + all evals
```

Aggregate metrics are printed to stdout after the run completes.

## EvoTest: Automated Evolutionary Skill Optimization

`evotest_loop.py` runs the full EvoTest loop **in-process** (no subprocess overhead). It iteratively evolves a clinical reasoning skill using a UCB tree to balance exploration vs exploitation.

### How it works

```
Episode 0:  Run agent with no skill (or --initial-skill) on all pathologies
            Compute composite score, save trajectories

Episode N:  UCB selects best parent node
            Evolver (Claude Opus) analyzes parent's failed trajectories
            Evolver generates improved skill
            Run agent with new skill on all pathologies
            Score, create child node, update tree
            Save checkpoint (resumable)
```

The composite score weights the PathologyEvaluator metrics:

```
3.0 * Diagnosis
+ 1.0 * PE First
+ 0.5 * PE at all
+ 1.0 * Labs (normalized)
+ 1.0 * Imaging (normalized)
- 0.5 * Invalid Tools
- 0.3 * Parsing Errors
```

### Run EvoTest

```bash
# From project root (not codes_openai_agent/)
python codes_openai_agent/evotest_loop.py \
    --episodes 10 \
    --model gpt-4o \
    --evolver-model claude-opus-4-6
```

### Fast iteration on one pathology

```bash
python codes_openai_agent/evotest_loop.py \
    --episodes 5 \
    --pathologies appendicitis \
    --model gpt-4o
```

### Use a local model

```bash
python codes_openai_agent/evotest_loop.py \
    --episodes 10 \
    --litellm-model openai/Qwen3-30B-A3B \
    --litellm-base-url http://gpu-server:8000/v1 \
    --evolver-model claude-opus-4-6
```

### Warm-start from an existing skill

```bash
python codes_openai_agent/evotest_loop.py \
    --episodes 10 \
    --initial-skill skills/v2/acute_abdominal_pain.md
```

### Resume from checkpoint

```bash
python codes_openai_agent/evotest_loop.py --resume --episodes 15
```

### Dry run (no API calls, prints Evolver prompt)

```bash
python codes_openai_agent/evotest_loop.py --dry-run --episodes 2
```

### Full `evotest_loop.py` options

```
Agent:
  --model             OpenAI model (default: gpt-4o)
  --litellm-model     LiteLLM model string
  --litellm-base-url  Base URL for self-hosted models
  --annotate-clinical / --no-annotate-clinical  Lab annotations (default: on)
  --max-turns         Agent turn limit per patient (default: 20)
  --split             train | test | full (default: train)
  --pathologies       Subset of pathologies (default: all 4)

Evolution:
  --episodes              Total episodes (default: 10)
  --evolver-model         Anthropic model for Evolver (default: claude-opus-4-6)
  --exploration-constant  UCB exploration c (default: 1.0)
  --depth-constant        UCB depth decay alpha (default: 0.8)
  --drop-threshold        Force-best-after-drop threshold (default: 1.0)
  --force-best-after-drop / --no-force-best-after-drop  (default: on)
  --initial-skill         Seed skill for episode 0

Control:
  --resume    Resume from evotest_state_sdk/state.json
  --dry-run   Print Evolver prompt without running anything
```

### EvoTest output files

```
skills/evo_sdk/
    episode_0.md          # Sanitized skill (injected into agent)
    episode_0_raw.md      # Raw skill before sanitization
    episode_1.md
    ...

trajectories_sdk/
    evo_ep0_appendicitis.json    # Per-pathology trajectory + scores
    evo_ep0_cholecystitis.json
    ...

evotest_state_sdk/
    state.json            # Full UCB tree checkpoint (for --resume)
```

## File Reference

```
codes_openai_agent/
    run.py                 # CLI entry point: run agent on patients, evaluate, save results
    manager.py             # ClinicalDiagnosisManager: creates context, builds agent, runs diagnosis
    context.py             # PatientContext: mutable state shared across tools and agents
    models.py              # Pydantic output models (DiagnosticResult, LabInterpretation, ChallengerFeedback)
    tools.py               # @function_tool wrappers around Hager's retrieve_* functions
    guardrails.py          # PE-first tool input guardrail + diagnosis output guardrail
    hooks.py               # ClinicalRunHooks: tool call logging, timing, observability
    evaluator_adapter.py   # Converts SDK RunResult -> PathologyEvaluator (AgentAction, observation) format
    hager_imports.py       # Handles `agents` package naming collision between SDK and Hager's code
    evotest_loop.py        # EvoTest evolutionary skill optimization loop (async, in-process)
    sub_agents/
        orchestrator.py    # Main diagnostic agent with dynamic skill injection
        lab_interpreter.py # Lab Interpreter sub-agent (gpt-4o-mini)
        challenger.py      # Challenger/devil's advocate sub-agent (gpt-4o-mini)
```

### Data flow

```
run.py
 -> load_patients()                       # Hager's pickle files
 -> ClinicalDiagnosisManager(config)
     -> _load_skill()                     # Strip YAML frontmatter from SKILL.md
 -> for each patient:
     -> manager.run(id, data)
         -> PatientContext(...)            # Mutable state for this patient
         -> create_orchestrator(model)    # Agent + tools + sub-agents + guardrails
         -> ClinicalRunHooks()            # Tool call logging + timing
         -> Runner.run(orchestrator, hooks=hooks, error_handlers=...)
             -> PE (is_enabled unlocks labs/imaging)
             -> Labs (guardrail enforces PE-first) -> interpret_lab_results
             -> Imaging (guardrail enforces PE-first) -> challenge_diagnosis
             -> DiagnosticResult (output_guardrail validates content)
         -> log token usage               # input/output/total tokens per patient
     -> convert_sdk_result()              # SDK RunResult -> [(AgentAction, obs)] trajectory
     -> evaluator._evaluate_agent_trajectory()  # Hager's PathologyEvaluator
     -> save results + evals
```

### Evaluator bridge

The SDK produces `RunResult.new_items` (tool calls + outputs). `evaluator_adapter.py` converts these into the exact `(AgentAction, observation)` tuple format that Hager's PathologyEvaluator expects:

- SDK tool names (`physical_examination`) are mapped to Hager names (`Physical Examination`)
- Sub-agent tools (`interpret_lab_results`, `challenge_diagnosis`) are filtered out (not scored)
- `custom_parsings=0` always (SDK JSON validation = no parsing errors)
- Lab itemids from `PatientContext.lab_itemid_log` are substituted for accurate lab scoring

## SDK Features Adopted

| Feature | Where | What it does |
|---|---|---|
| `tool_input_guardrail` | `guardrails.py` → `tools.py` | Blocks lab/imaging calls if PE hasn't been done (failure mode #3) |
| `output_guardrail` | `guardrails.py` → `orchestrator.py` | Validates DiagnosticResult has meaningful diagnosis, treatment, evidence |
| `is_enabled` | `tools.py` | Hides lab/imaging tools from model until PE is done (belt-and-suspenders with guardrail) |
| `ModelSettings` | `orchestrator.py` | `temperature=0.0, parallel_tool_calls=False` for deterministic, sequential tool use |
| `RunHooks` | `hooks.py` → `manager.py` | Logs tool calls with timing; tracks tool ordering in `PatientContext.tool_call_log` |
| `RunErrorHandlers` | `manager.py` | Graceful degradation: returns best-effort `DiagnosticResult` on max turns exceeded |
| `max_turns` on `as_tool()` | `lab_interpreter.py`, `challenger.py` | Caps sub-agent loops at 3 turns to prevent runaway costs |
| Token usage tracking | `manager.py` | Logs input/output/total tokens and request count per patient |

## Reference: Original OpenAI Agents SDK

This agent is built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (`pip install openai-agents`). Key reference materials:

- **Repository**: https://github.com/openai/openai-agents-python
- **Documentation**: https://openai.github.io/openai-agents-python/
- **Examples** (patterns we adopt): https://github.com/openai/openai-agents-python/tree/main/examples
  - `financial_research_agent/` — multi-agent orchestrator pattern (our architecture model)
  - `agent_patterns/` — guardrails, hooks, input/output validation

## Comparison with LangChain Agent

The LangChain agent lives in `codes_Hager/` and uses `scripts/evotest_clinical.py` for its EvoTest loop. The two are completely independent:

| | LangChain (`codes_Hager/`) | SDK (`codes_openai_agent/`) |
|---|---|---|
| Agent framework | LangChain ZeroShotAgent | OpenAI Agents SDK |
| EvoTest script | `scripts/evotest_clinical.py` | `codes_openai_agent/evotest_loop.py` |
| Episode execution | 3 subprocesses (run, evaluate, extract) | In-process async |
| State directory | `evotest_state/` | `evotest_state_sdk/` |
| Trajectories | `trajectories/` | `trajectories_sdk/` |
| Skills | `skills/evo/` | `skills/evo_sdk/` |
| Evaluator | Same PathologyEvaluator | Same PathologyEvaluator |
| Patient data | Same `data_splits/` | Same `data_splits/` |

Both produce scores from the same evaluator on the same patients, enabling direct comparison.
