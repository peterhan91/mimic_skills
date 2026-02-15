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

# Full experiment: EvoTest training (4×10) + test evaluation (7×100)
bash scripts/evotest_full.sh 10 Qwen3_30B_A3B

# With patient simulator (agent gets only chief complaint, must Ask Patient)
bash scripts/evotest_full.sh --patient-sim 10 Qwen3_30B_A3B

# With Tree of Thoughts agent
bash scripts/evotest_full.sh --agent ToT 10 Qwen3_30B_A3B

# All flags combine (2×2 matrix: agent × patient-sim)
bash scripts/evotest_full.sh --agent ToT --patient-sim 10 Qwen3_30B_A3B
```

## Pathologies

**Train (4):** Appendicitis, cholecystitis, diverticulitis, pancreatitis — 10 train / 100 test patients each.
**Test-only (3):** Cholangitis, bowel obstruction, pyelonephritis — test generalization to unseen diseases.

## Documentation

- [CLAUDE.md](CLAUDE.md) — Full project specification (approaches, experiment plans, architecture)
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — Step-by-step workflow (local + GPU server)
- [docs/EXAMPLE_WALKTHROUGH.md](docs/EXAMPLE_WALKTHROUGH.md) — Complete example of one evolution cycle
- [docs/EVOTEST_ADAPTATION.md](docs/EVOTEST_ADAPTATION.md) — EvoTest integration plan
- [docs/per_agent_skills.md](docs/per_agent_skills.md) — Per-sub-agent skill evolution architecture

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/evotest_full.sh` | Full pipeline: train → select best skill → test (7×100) |
| `scripts/evotest_train.sh` | EvoTest training loop (4 pathologies × 10 patients) |
| `scripts/evotest_test.sh` | Test a skill on held-out test set (7 pathologies × 100) |
| `scripts/evotest_clinical.py` | Core EvoTest engine: UCB tree, Evolver, regression protection |
| `scripts/evolve_skill.py` | Generate/refine skills from trajectory failures |
| `scripts/extract_trajectories.py` | Convert result pkl → JSON for the Evolver |
| `scripts/evaluate_run.py` | Run PathologyEvaluator on experiment results |
| `scripts/sanitize_skill.py` | Remove disease name leakage from skills |
| `scripts/compare_runs.py` | Side-by-side metric comparison of two runs |

## References

- Hager et al., "Evaluation and Mitigation of the Limitations of Large Language
  Models in Clinical Decision-Making", Nature Medicine, 2024
- [MIMIC-CDM Framework](https://github.com/MIMIC-Clinical-Decision-Making/MIMIC-Clinical-Decision-Making-Framework)
- [Anthropic Agent Skills](https://github.com/anthropics/skills) — Official skills
  standard and examples ([spec](https://agentskills.io/specification),
  [skill-creator reference](https://github.com/anthropics/skills/tree/main/skills/skill-creator))
- [Agent Skills blog post](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
