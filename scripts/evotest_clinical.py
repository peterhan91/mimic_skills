"""
EvoTest-style evolutionary skill optimization for Hager's clinical agent.

Replaces the linear run_iterations.sh pipeline with UCB tree-based exploration
+ regression protection. Each episode runs the agent on all 4 pathologies x
train patients, collects trajectories, computes a composite score, and the
Evolver (Opus) generates an improved skill for the next episode.

Key EvoTest features ported:
  - UCB tree (calculate_ucb + node management + depth decay)
  - Force-best-after-drop (regression protection)
  - Negative memory for Evolver (show failed skills + scores)
  - Resumable state persistence (JSON checkpoint)

Usage:
    python scripts/evotest_clinical.py \
        --episodes 10 \
        --model Qwen3_30B_A3B \
        --evolver-model claude-opus-4-6 \
        --annotate-clinical True

    # Resume from checkpoint:
    python scripts/evotest_clinical.py --resume --episodes 15

    # Dry run (prints Evolver prompt, no API calls or agent runs):
    python scripts/evotest_clinical.py --dry-run --episodes 1

Requires: ANTHROPIC_API_KEY environment variable (for Evolver).
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Load .env if present (for ANTHROPIC_API_KEY, HF_HOME, etc.)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip().strip("'\"")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAMEWORK_DIR = PROJECT_DIR / "codes_Hager" / "MIMIC-Clinical-Decision-Making-Framework"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

# Add scripts to path for imports
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from scripts.sanitize_skill import sanitize_skill_text
from scripts.evolve_skill import (
    load_trajectories,
    identify_failures,
    format_trajectory_summary,
    format_discharge_summary,
    build_aggregate_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATHOLOGIES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

# Max "Laboratory Tests" score per pathology (from evaluators: 1 point per required category)
MAX_LAB_SCORE = {
    "appendicitis": 1,    # Inflammation
    "cholecystitis": 3,   # Inflammation, Liver, Gallbladder
    "diverticulitis": 1,  # Inflammation
    "pancreatitis": 3,    # Inflammation, Pancreas, Seriousness
}
STATE_DIR = PROJECT_DIR / "evotest_state"
STATE_FILE = STATE_DIR / "state.json"
DEFAULT_SPLIT = "train"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest_results_dir(results_dir, pathology, descr):
    """Find the most recently modified results directory matching pathology + descr.

    Mirrors the shell logic: ls -td "$RESULTS_DIR"/*"$PATHOLOGY"*"$DESCR"* | head -1
    """
    pattern = os.path.join(str(results_dir), f"*{pathology}*{descr}*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Sort by modification time, newest first
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def run_subprocess(cmd, description, cwd=None, dry_run=False):
    """Run a subprocess command, printing the command and checking for errors."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n  [{description}] {cmd_str}")
    if dry_run:
        print("  (dry run — skipping)")
        return True
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: {description} failed (exit code {result.returncode})")
        return False
    return True


# ---------------------------------------------------------------------------
# ClinicalEvoTest
# ---------------------------------------------------------------------------
class ClinicalEvoTest:
    def __init__(self, args):
        self.args = args
        self.nodes = []
        self.best_node_idx = None
        self.best_score = float("-inf")
        self.last_episode_score = None
        self.completed_episodes = 0

        # Paths
        self.data_dir = PROJECT_DIR / "data_splits"
        self.results_dir = PROJECT_DIR / "results"
        self.traj_dir = PROJECT_DIR / "trajectories"
        self.skills_dir = PROJECT_DIR / "skills" / "evo"
        self.log_dir = PROJECT_DIR / "logs"
        self.lab_test_mapping = PROJECT_DIR / "MIMIC-CDM-IV" / "lab_test_mapping.pkl"
        self.base_models = os.environ.get(
            "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        )

    # ------------------------------------------------------------------
    # UCB Tree
    # ------------------------------------------------------------------
    def calculate_ucb(self, node_idx):
        """UCB = score + c * alpha^depth * sqrt(log(N_total) / (1 + N_children))"""
        node = self.nodes[node_idx]
        num_children = len(node["children_idxs"])
        total_nodes = max(2, len(self.nodes))
        c = self.args.exploration_constant
        alpha = self.args.depth_constant
        exploration = c * (alpha ** node["depth"]) * math.sqrt(
            math.log(total_nodes) / (1 + num_children)
        )
        return node["score"] + exploration

    def select_parent(self):
        """Select parent node via UCB, with force-best-after-drop safety."""
        if not self.nodes:
            return -1

        # Safety: if last episode dropped far below best, force-select best
        if (
            self.args.force_best_after_drop
            and self.last_episode_score is not None
            and self.best_node_idx is not None
            and (self.best_score - self.last_episode_score) >= self.args.drop_threshold
        ):
            print(
                f"  [UCB] Force-selecting best node {self.best_node_idx} "
                f"(score={self.best_score:.3f}) after drop "
                f"(last={self.last_episode_score:.3f}, "
                f"threshold={self.args.drop_threshold})"
            )
            return self.best_node_idx

        # Normal UCB selection
        best_ucb_idx = max(range(len(self.nodes)), key=self.calculate_ucb)
        ucb_val = self.calculate_ucb(best_ucb_idx)
        print(
            f"  [UCB] Selected node {best_ucb_idx} "
            f"(score={self.nodes[best_ucb_idx]['score']:.3f}, "
            f"ucb={ucb_val:.3f}, depth={self.nodes[best_ucb_idx]['depth']})"
        )
        return best_ucb_idx

    # ------------------------------------------------------------------
    # Composite Scoring
    # ------------------------------------------------------------------
    def compute_composite_score(self, all_trajectory_data):
        """Compute weighted composite score across all pathologies.

        Per-patient score (max ~6.5):
            3.0 * Diagnosis
          + 1.0 * Physical Examination (PE first)
          + 0.5 * Late Physical Examination (PE at all)
          + 1.0 * (Laboratory Tests / max_lab_score)
          + 1.0 * (Imaging / 2.0)
          - 0.5 * Invalid Tools
          - 0.3 * (1 - Action Parsing)

        Returns (composite, per_metric_aggregate, per_pathology).
        """
        all_patient_scores = []
        per_pathology = {}

        for data in all_trajectory_data:
            pathology = data["pathology"]
            max_lab = MAX_LAB_SCORE.get(pathology, 1)
            patient_scores = []

            for admission in data["admissions"]:
                s = admission["scores"]
                ps = (
                    3.0 * s.get("Diagnosis", 0)
                    + 1.0 * s.get("Physical Examination", 0)
                    + 0.5 * s.get("Late Physical Examination", 0)
                    + 1.0 * min(s.get("Laboratory Tests", 0) / max_lab, 1.0)
                    + 1.0 * min(s.get("Imaging", 0) / 2.0, 1.0)
                    - 0.5 * min(s.get("Invalid Tools", 0), 2)
                    - 0.3 * (1.0 - s.get("Action Parsing", 1))
                )
                patient_scores.append(ps)
                all_patient_scores.append(ps)

            per_pathology[pathology] = (
                sum(patient_scores) / len(patient_scores) if patient_scores else 0.0
            )

        composite = sum(all_patient_scores) / len(all_patient_scores) if all_patient_scores else 0.0

        # Aggregate raw metrics
        per_metric = {}
        for data in all_trajectory_data:
            for key in data.get("aggregate", {}):
                if key not in per_metric:
                    per_metric[key] = []
                per_metric[key].append(data["aggregate"][key])
        per_metric_avg = {k: sum(v) / len(v) for k, v in per_metric.items() if v}

        return composite, per_metric_avg, per_pathology

    # ------------------------------------------------------------------
    # Episode Orchestration
    # ------------------------------------------------------------------
    def run_episode(self, skill_text, episode_num):
        """Run agent on all pathologies with given skill, evaluate, extract, score.

        Returns (composite_score, per_metric, per_pathology, trajectory_paths) or None on failure.
        """
        descr = f"_evo_ep{episode_num}"
        skill_file = None

        # Save and sanitize skill if provided
        if skill_text:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            raw_path = self.skills_dir / f"episode_{episode_num}_raw.md"
            raw_path.write_text(skill_text)

            sanitized = sanitize_skill_text(skill_text)
            skill_file = self.skills_dir / f"episode_{episode_num}.md"
            skill_file.write_text(sanitized)
            print(f"  Saved skill: {skill_file} ({len(sanitized)} chars)")

        trajectory_paths = []
        all_trajectory_data = []

        for pathology in PATHOLOGIES:
            patient_data = self.data_dir / pathology / f"{DEFAULT_SPLIT}.pkl"
            if not patient_data.exists():
                print(f"  WARNING: {patient_data} not found, skipping {pathology}")
                continue

            # --- 1. Run agent ---
            run_cmd = [
                "python", "run.py",
                f"pathology={pathology}",
                f"model={self.args.model}",
                f"base_mimic={self.data_dir / pathology}",
                f"base_models={self.base_models}",
                f"lab_test_mapping_path={self.lab_test_mapping}",
                f"local_logging_dir={self.results_dir}",
                "summarize=True",
                f"annotate_clinical={self.args.annotate_clinical}",
                f"run_descr={descr}",
            ]
            if skill_file:
                run_cmd.append(f"skill_path={skill_file}")

            ok = run_subprocess(
                run_cmd,
                f"run.py {pathology} ep{episode_num}",
                cwd=str(FRAMEWORK_DIR),
                dry_run=self.args.dry_run,
            )
            if not ok:
                print(f"  ERROR: Agent run failed for {pathology}, aborting episode")
                return None

            # --- 2. Find results dir ---
            results_subdir = find_latest_results_dir(self.results_dir, pathology, descr)
            if not results_subdir and not self.args.dry_run:
                print(f"  ERROR: No results dir found for {pathology}{descr}")
                return None

            if self.args.dry_run:
                trajectory_paths.append(f"(dry-run) trajectories/evo_ep{episode_num}_{pathology}.json")
                continue

            # --- 3. Evaluate ---
            ok = run_subprocess(
                [
                    "python", str(SCRIPTS_DIR / "evaluate_run.py"),
                    "--results_dir", results_subdir,
                    "--pathology", pathology,
                    "--patient_data", str(patient_data),
                ],
                f"evaluate {pathology}",
            )
            if not ok:
                print(f"  ERROR: Evaluation failed for {pathology}, aborting episode")
                return None

            # --- 4. Extract trajectories ---
            self.traj_dir.mkdir(parents=True, exist_ok=True)
            traj_output = self.traj_dir / f"evo_ep{episode_num}_{pathology}.json"
            ok = run_subprocess(
                [
                    "python", str(SCRIPTS_DIR / "extract_trajectories.py"),
                    "--results_dir", results_subdir,
                    "--pathology", pathology,
                    "--patient_data", str(patient_data),
                    "--output", str(traj_output),
                ],
                f"extract {pathology}",
            )
            if not ok:
                print(f"  ERROR: Trajectory extraction failed for {pathology}")
                return None

            trajectory_paths.append(str(traj_output))
            data = load_trajectories(str(traj_output))
            all_trajectory_data.append(data)

        if self.args.dry_run:
            return 0.0, {}, {}, trajectory_paths

        if not all_trajectory_data:
            print("  ERROR: No trajectory data collected")
            return None

        composite, per_metric, per_pathology = self.compute_composite_score(all_trajectory_data)
        return composite, per_metric, per_pathology, trajectory_paths

    # ------------------------------------------------------------------
    # Evolver
    # ------------------------------------------------------------------
    def evolve_skill(self, parent_node, trajectory_data_list):
        """Call the Evolver (Opus) to generate an improved skill."""
        prompt = self._build_evolver_prompt(parent_node, trajectory_data_list)

        if self.args.dry_run:
            print(f"\n{'='*60}")
            print("DRY RUN — Evolver prompt:")
            print(f"{'='*60}")
            print(prompt[:3000])
            if len(prompt) > 3000:
                print(f"... [{len(prompt) - 3000} chars truncated]")
            print(f"\nPrompt length: {len(prompt)} chars")
            return "(dry-run skill placeholder)"

        print(f"\n  Calling Evolver ({self.args.evolver_model})...")
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.args.evolver_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        skill_text = message.content[0].text
        print(f"  Evolver produced skill ({len(skill_text)} chars)")
        return skill_text

    def _build_evolver_prompt(self, parent_node, trajectory_data_list):
        """Build the Evolver prompt with evolution history, metrics, and trajectories."""
        # --- Section 1: Evolution history ---
        history_lines = []
        if self.best_node_idx is not None:
            best = self.nodes[self.best_node_idx]
            history_lines.append(
                f"- **Best skill so far** (node {self.best_node_idx}, "
                f"episode {best['episode_num']}): composite score = {best['score']:.3f}"
            )
        if parent_node:
            history_lines.append(
                f"- **Parent skill** (node {parent_node['idx']}, "
                f"episode {parent_node['episode_num']}): composite score = {parent_node['score']:.3f}"
            )

        # Recent failed attempts (nodes worse than parent)
        failed_nodes = [
            n for n in self.nodes
            if n["score"] < (parent_node["score"] if parent_node else 0)
            and n["skill_text"]
        ]
        failed_nodes.sort(key=lambda n: n["score"])
        for fn in failed_nodes[:3]:
            history_lines.append(
                f"- **Failed attempt** (node {fn['idx']}, score={fn['score']:.3f}): "
                f"skill preview: {fn['skill_text'][:200]}..."
            )

        history_section = "\n".join(history_lines) if history_lines else "(first episode)"

        # --- Section 2: Current performance ---
        if trajectory_data_list:
            aggregate_table = build_aggregate_table(trajectory_data_list)
        else:
            aggregate_table = "(no trajectory data)"

        # Per-metric analysis
        metric_lines = []
        if parent_node and parent_node.get("per_metric"):
            pm = parent_node["per_metric"]
            for key, val in pm.items():
                if key == "Diagnosis":
                    metric_lines.append(
                        f"- **Diagnosis accuracy**: {val:.0%} — "
                        f"{'needs the most improvement' if val < 0.5 else 'reasonable, maintain or improve'}"
                    )
                elif key == "Physical Examination":
                    metric_lines.append(f"- **PE first**: {val:.0%}")
                elif key == "Laboratory Tests":
                    metric_lines.append(f"- **Lab score avg**: {val:.2f}")
                elif key == "Imaging":
                    metric_lines.append(f"- **Imaging score avg**: {val:.2f}")
                elif key == "Invalid Tools":
                    metric_lines.append(f"- **Invalid tools avg**: {val:.2f}")
        metric_section = "\n".join(metric_lines) if metric_lines else "(no metric data)"

        # --- Section 3: Failed trajectories ---
        gap_analyses = []
        if trajectory_data_list:
            all_failures = []
            for data in trajectory_data_list:
                failures = identify_failures(data)
                all_failures.extend(failures)

            # Pick up to 3 per pathology, max 12 total
            for pathology in PATHOLOGIES:
                path_failures = [f for f in all_failures if f["pathology"] == pathology]
                for fail in path_failures[:3]:
                    admission = fail["admission"]
                    reasons = ", ".join(fail["reasons"])
                    analysis = (
                        f"---\n"
                        f"{format_trajectory_summary(admission, pathology=fail['pathology'])}\n\n"
                        f"**Failure reasons**: {reasons}\n\n"
                        f"**Real Doctor's Discharge Summary**:\n```\n"
                        f"{format_discharge_summary(admission)}\n```\n"
                    )
                    gap_analyses.append(analysis)
            gap_analyses = gap_analyses[:12]

        gap_section = "\n".join(gap_analyses) if gap_analyses else "(no failures to analyze)"

        # --- Section 4: Parent skill ---
        parent_skill_section = ""
        if parent_node and parent_node.get("skill_text"):
            parent_skill_section = (
                f"## Parent Skill (score={parent_node['score']:.3f})\n\n"
                f"Analyze where it helped and where it failed, then IMPROVE it:\n\n"
                f"```markdown\n{parent_node['skill_text']}\n```\n\n"
            )

        prompt = f"""You are a clinical AI system optimizer. Your task is to analyze diagnostic agent trajectories and generate an improved clinical reasoning skill.

## Evolution History

{history_section}

## Current Agent Performance

{aggregate_table}

### Per-Metric Analysis

{metric_section}

{parent_skill_section}## Failed Trajectories with Gap Analysis

{gap_section}

## Your Task

Generate an improved GENERAL clinical reasoning workflow skill for diagnosing patients presenting with acute abdominal pain. This skill must:

1. **Teach hypothesis-driven diagnostic reasoning** — maintain a running differential diagnosis, and choose each test to maximally reduce uncertainty between remaining hypotheses
2. **Address the specific failure patterns above** — focus on what went wrong and teach the correct approach
3. **Be grounded in what real doctors did** — use the discharge summary evidence
4. **Work across ALL pathologies** — must handle appendicitis, cholecystitis, diverticulitis, pancreatitis and any other acute abdominal pain cause
5. **Stay under 500 tokens** — concise, actionable instructions
6. **NOT use disease names** — use ____ as a mask for any disease or procedure name that would reveal the diagnosis

The skill should be written as markdown with clear step-by-step instructions:
- When to do Physical Examination (should always be FIRST)
- How to select labs based on exam findings (not shotgun ordering)
- How to choose imaging modality based on suspected pathology location
- How to interpret lab values in context
- When to recommend surgical vs conservative treatment
- How to maintain and update a differential diagnosis after each observation

Output ONLY the skill content in markdown format. No preamble or explanation."""

        return prompt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self):
        """Save full tree to evotest_state/state.json."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        # Convert -inf to None for JSON serialization (json can't encode infinity)
        best_score_safe = self.best_score if math.isfinite(self.best_score) else None
        state = {
            "nodes": self.nodes,
            "best_node_idx": self.best_node_idx,
            "best_score": best_score_safe,
            "last_episode_score": self.last_episode_score,
            "completed_episodes": self.completed_episodes,
            "args": {
                "model": self.args.model,
                "evolver_model": self.args.evolver_model,
                "annotate_clinical": self.args.annotate_clinical,
                "exploration_constant": self.args.exploration_constant,
                "depth_constant": self.args.depth_constant,
                "drop_threshold": self.args.drop_threshold,
                "force_best_after_drop": self.args.force_best_after_drop,
            },
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"  State saved to {STATE_FILE} ({len(self.nodes)} nodes)")

    def load_state(self):
        """Load state from checkpoint. Returns True if successful."""
        if not STATE_FILE.exists():
            print(f"  No state file found at {STATE_FILE}")
            return False
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        self.nodes = state["nodes"]
        self.best_node_idx = state["best_node_idx"]
        self.best_score = state["best_score"] if state["best_score"] is not None else float("-inf")
        self.last_episode_score = state["last_episode_score"]
        self.completed_episodes = state["completed_episodes"]
        print(
            f"  Resumed from {STATE_FILE}: "
            f"{self.completed_episodes} episodes, "
            f"{len(self.nodes)} nodes, "
            f"best_score={self.best_score:.3f}"
        )
        return True

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def run(self):
        """Main EvoTest loop."""
        start_time = time.time()

        # Resume or fresh start
        if self.args.resume:
            if not self.load_state():
                print("  Cannot resume — starting fresh")

        start_episode = self.completed_episodes
        total_episodes = self.args.episodes

        if start_episode >= total_episodes:
            print(f"Already completed {start_episode} episodes (target={total_episodes}). "
                  f"Increase --episodes to continue.")
            return

        print(f"\n{'='*70}")
        print(f"EvoTest Clinical — Episodes {start_episode}..{total_episodes - 1}")
        print(f"  Model: {self.args.model}")
        print(f"  Evolver: {self.args.evolver_model}")
        print(f"  Annotate clinical: {self.args.annotate_clinical}")
        print(f"  UCB c={self.args.exploration_constant}, alpha={self.args.depth_constant}")
        print(f"  Drop threshold: {self.args.drop_threshold}")
        print(f"{'='*70}\n")

        for episode_num in range(start_episode, total_episodes):
            ep_start = time.time()
            print(f"\n{'='*70}")
            print(f"EPISODE {episode_num}")
            print(f"{'='*70}")

            if episode_num == 0 and not self.nodes:
                # --- Episode 0: Baseline or seed skill ---
                skill_text = ""
                if self.args.initial_skill:
                    skill_path = Path(self.args.initial_skill)
                    if skill_path.exists():
                        skill_text = skill_path.read_text()
                        print(f"  Using initial skill from {skill_path}")
                    else:
                        print(f"  WARNING: {skill_path} not found, running without skill")

                print("  Running baseline episode...")
                result = self.run_episode(skill_text if skill_text else None, episode_num)
                if result is None:
                    print("  Episode 0 failed — aborting")
                    return

                composite, per_metric, per_pathology, traj_paths = result
                node = {
                    "idx": 0,
                    "skill_text": skill_text,
                    "sanitized_skill": sanitize_skill_text(skill_text) if skill_text else "",
                    "score": composite,
                    "per_metric": per_metric,
                    "per_pathology": per_pathology,
                    "parent_idx": -1,
                    "children_idxs": [],
                    "depth": 0,
                    "trajectory_paths": traj_paths,
                    "episode_num": episode_num,
                }
                self.nodes.append(node)
                self.best_node_idx = 0
                self.best_score = composite
                self.last_episode_score = composite

            else:
                # --- Episodes 1..N: UCB select → evolve → run → score ---
                parent_idx = self.select_parent()
                parent_node = self.nodes[parent_idx]

                # Load parent's trajectory data for Evolver context
                trajectory_data_list = []
                for tpath in parent_node.get("trajectory_paths", []):
                    if os.path.exists(tpath):
                        trajectory_data_list.append(load_trajectories(tpath))

                # Evolve skill
                print(f"  Evolving from node {parent_idx} (score={parent_node['score']:.3f})...")
                new_skill = self.evolve_skill(parent_node, trajectory_data_list)

                # Run episode with new skill
                print(f"  Running episode with evolved skill...")
                result = self.run_episode(new_skill, episode_num)
                if result is None:
                    print(f"  Episode {episode_num} failed — creating failed node and continuing")
                    node = {
                        "idx": len(self.nodes),
                        "skill_text": new_skill,
                        "sanitized_skill": sanitize_skill_text(new_skill),
                        "score": 0.0,
                        "per_metric": {},
                        "per_pathology": {},
                        "parent_idx": parent_idx,
                        "children_idxs": [],
                        "depth": parent_node["depth"] + 1,
                        "trajectory_paths": [],
                        "episode_num": episode_num,
                    }
                    self.nodes.append(node)
                    parent_node["children_idxs"].append(node["idx"])
                    self.last_episode_score = 0.0
                    self.completed_episodes = episode_num + 1
                    self.save_state()
                    continue

                composite, per_metric, per_pathology, traj_paths = result

                # Create child node
                node = {
                    "idx": len(self.nodes),
                    "skill_text": new_skill,
                    "sanitized_skill": sanitize_skill_text(new_skill),
                    "score": composite,
                    "per_metric": per_metric,
                    "per_pathology": per_pathology,
                    "parent_idx": parent_idx,
                    "children_idxs": [],
                    "depth": parent_node["depth"] + 1,
                    "trajectory_paths": traj_paths,
                    "episode_num": episode_num,
                }
                self.nodes.append(node)
                parent_node["children_idxs"].append(node["idx"])

                # Update best
                self.last_episode_score = composite
                if composite > self.best_score:
                    self.best_score = composite
                    self.best_node_idx = node["idx"]
                    print(f"  *** NEW BEST: score={composite:.3f} (node {node['idx']}) ***")

            # Print episode summary
            node = self.nodes[-1]
            ep_elapsed = time.time() - ep_start
            print(f"\n  Episode {episode_num} summary:")
            print(f"    Composite score: {node['score']:.3f}")
            print(f"    Best so far:     {self.best_score:.3f} (node {self.best_node_idx})")
            if node.get("per_pathology"):
                for p, s in node["per_pathology"].items():
                    print(f"    {p:20s}: {s:.3f}")
            print(f"    Tree size:       {len(self.nodes)} nodes")
            print(f"    Duration:        {ep_elapsed:.0f}s")

            self.completed_episodes = episode_num + 1
            self.save_state()

        # Final summary
        total_elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"EVOTEST COMPLETE")
        print(f"{'='*70}")
        print(f"  Episodes:     {self.completed_episodes}")
        print(f"  Nodes:        {len(self.nodes)}")
        print(f"  Best score:   {self.best_score:.3f} (node {self.best_node_idx})")
        print(f"  Duration:     {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")

        if self.best_node_idx is not None:
            best = self.nodes[self.best_node_idx]
            best_skill_path = self.skills_dir / f"episode_{best['episode_num']}.md"
            print(f"  Best skill:   {best_skill_path}")
            if best.get("per_pathology"):
                print(f"  Per-pathology:")
                for p, s in best["per_pathology"].items():
                    print(f"    {p:20s}: {s:.3f}")

        # Print tree structure
        print(f"\n  Tree structure:")
        for n in self.nodes:
            indent = "  " * n["depth"]
            marker = " ***BEST***" if n["idx"] == self.best_node_idx else ""
            print(
                f"    {indent}node {n['idx']} "
                f"(ep={n['episode_num']}, score={n['score']:.3f}, "
                f"depth={n['depth']}, children={len(n['children_idxs'])})"
                f"{marker}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EvoTest-style evolutionary skill optimization for clinical agent"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Total number of episodes to run (default: 10)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen3_30B_A3B",
        help="Agent model name (as configured in Hager's framework)"
    )
    parser.add_argument(
        "--evolver-model", type=str, default="claude-opus-4-6",
        help="Anthropic model for the Evolver (default: claude-opus-4-6)"
    )
    parser.add_argument(
        "--annotate-clinical", type=str, default="True",
        help="Enable clinical lab annotations (default: True)"
    )
    parser.add_argument(
        "--exploration-constant", type=float, default=1.0,
        help="UCB exploration constant c (default: 1.0)"
    )
    parser.add_argument(
        "--depth-constant", type=float, default=0.8,
        help="UCB depth decay alpha (default: 0.8)"
    )
    parser.add_argument(
        "--drop-threshold", type=float, default=1.0,
        help="Force-best-after-drop threshold (default: 1.0)"
    )
    parser.add_argument(
        "--force-best-after-drop", action="store_true", default=True,
        help="Force-select best node after large score drop (default: True)"
    )
    parser.add_argument(
        "--no-force-best-after-drop", action="store_false", dest="force_best_after_drop",
        help="Disable force-best-after-drop"
    )
    parser.add_argument(
        "--initial-skill", type=str, default=None,
        help="Path to initial seed skill for episode 0 (optional)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands and Evolver prompt without running anything"
    )
    args = parser.parse_args()

    # Validate
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    runner = ClinicalEvoTest(args)
    runner.run()


if __name__ == "__main__":
    main()
