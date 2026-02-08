# MIMIC-Skills

Improving LLM performance on the autonomous clinical decision-making benchmark
(Hager et al., Nature Medicine 2024) by injecting **agent skills** — structured
clinical reasoning patterns extracted from MIMIC-IV discharge summaries.

## The Problem

LLMs diagnose acute abdominal pain patients at 45-55% accuracy vs. 89% for
physicians. Key failure modes: skipping physical exams, misinterpreting labs,
hallucinating tools, and missing treatments.

## Our Approach

1. Extract clinical reasoning traces from real discharge summaries
2. Distill them into portable SKILL.md files (via evolve_skill.py)
3. Inject skills into Hager's ReAct agent at multiple surfaces (system prompt,
   examples slot, tool output annotations)
4. Iteratively refine skills using trajectory failures (EvoTest-style evolution)

## Quick Start

```bash
# Run baseline (no skill) on train split
python codes_Hager/MIMIC-Clinical-Decision-Making-Framework/run.py \
  pathology=appendicitis model=Qwen3_30B_A3B

# Run with skill injection
python codes_Hager/MIMIC-Clinical-Decision-Making-Framework/run.py \
  pathology=appendicitis model=Qwen3_30B_A3B \
  skill_path=skills/v3/acute_abdominal_pain.md skill_inject=both

# Multi-pathology experiment loop
bash scripts/run_experiment_multi.sh
```

## Pathologies

Appendicitis, cholecystitis, diverticulitis, pancreatitis — 10 train / 100 test
patients each, split from MIMIC-CDM-IV.

## Documentation

- [CLAUDE.md](CLAUDE.md) — Full project specification (approaches, experiment plans, architecture)
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — Step-by-step workflow (local + GPU server)
- [docs/EXAMPLE_WALKTHROUGH.md](docs/EXAMPLE_WALKTHROUGH.md) — Complete example of one evolution cycle
- [docs/EVOTEST_ADAPTATION.md](docs/EVOTEST_ADAPTATION.md) — EvoTest integration plan

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/evolve_skill.py` | Generate/refine skills from trajectory failures |
| `scripts/extract_trajectories.py` | Convert result pkl → JSON for the Evolver |
| `scripts/evaluate_run.py` | Run PathologyEvaluator on experiment results |
| `scripts/sanitize_skill.py` | Remove disease name leakage from skills |
| `scripts/compare_runs.py` | Side-by-side metric comparison of two runs |

## References

- Hager et al., "Evaluation and Mitigation of the Limitations of Large Language
  Models in Clinical Decision-Making", Nature Medicine, 2024
- [MIMIC-CDM Framework](https://github.com/MIMIC-Clinical-Decision-Making/MIMIC-Clinical-Decision-Making-Framework)
