"""
EvoTest-style evolutionary skill optimization for the OpenAI Agents SDK agent.

Parallel to scripts/evotest_clinical.py but runs the SDK agent IN-PROCESS via
async Python — no subprocess overhead, direct access to RunResult/trajectories/
scores.

Key EvoTest features:
  - UCB tree (calculate_ucb + node management + depth decay)
  - Force-best-after-drop (regression protection)
  - Negative memory for Evolver (show failed skills + scores)
  - Resumable state persistence (JSON checkpoint)

Usage:
    python codes_openai_agent/evotest_loop.py \
        --episodes 10 \
        --model gpt-4o \
        --evolver-model claude-opus-4-6

    # With LiteLLM model:
    python codes_openai_agent/evotest_loop.py \
        --episodes 10 \
        --litellm-model openai/Qwen3-30B-A3B \
        --litellm-base-url http://gpu:8000/v1

    # Resume from checkpoint:
    python codes_openai_agent/evotest_loop.py --resume --episodes 15

    # Dry run (prints Evolver prompt, no API calls or agent runs):
    python codes_openai_agent/evotest_loop.py --dry-run --episodes 1

    # Fast iteration on one pathology:
    python codes_openai_agent/evotest_loop.py --pathologies appendicitis --episodes 5

Requires: ANTHROPIC_API_KEY environment variable (for Evolver).
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths & .env loading
# ---------------------------------------------------------------------------
_OUR_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = Path(_OUR_ROOT).parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

# Ensure our own package and scripts are importable
if _OUR_ROOT not in sys.path:
    sys.path.insert(0, _OUR_ROOT)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Disable OpenAI Agents SDK tracing (avoids 401 errors when OPENAI_API_KEY is dummy)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Load .env if present (for ANTHROPIC_API_KEY, etc.)
_env_file = PROJECT_DIR / ".env"
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
# Imports from scripts/ (reuse)
# ---------------------------------------------------------------------------
from sanitize_skill import sanitize_skill_text
from evolve_skill import (
    identify_failures,
    format_trajectory_summary,
    format_discharge_summary,
    build_aggregate_table,
    load_guidelines_context,
    DEFAULT_GUIDELINES_DIR,
)

# ---------------------------------------------------------------------------
# Imports from codes_openai_agent/ (reuse)
# ---------------------------------------------------------------------------
from manager import ClinicalDiagnosisManager, ManagerConfig
from evaluator_adapter import convert_sdk_result
from hager_imports import load_hadm_from_file, load_evaluator
from run import build_reference_tuple, append_to_pickle_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_PATHOLOGIES = [
    "appendicitis", "cholecystitis", "diverticulitis", "pancreatitis",
    "cholangitis", "bowel_obstruction", "pyelonephritis",
]

# Max "Laboratory Tests" score per pathology (from evaluators: 1 point per required category)
MAX_LAB_SCORE = {
    "appendicitis": 1,        # Inflammation
    "cholecystitis": 3,       # Inflammation, Liver, Gallbladder
    "diverticulitis": 1,      # Inflammation
    "pancreatitis": 3,        # Inflammation, Pancreas, Seriousness
    "cholangitis": 3,         # Inflammation, Liver, Biliary
    "bowel_obstruction": 2,   # Inflammation, Electrolytes
    "pyelonephritis": 3,      # Inflammation, Renal, Urinalysis
}

STATE_DIR = PROJECT_DIR / "evotest_state_sdk"
STATE_FILE = STATE_DIR / "state.json"
EPISODE_LOG = STATE_DIR / "episode_log.jsonl"

logger = logging.getLogger("evotest_sdk")


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def print_metrics_table(per_pathology, per_metric):
    """Log a formatted metrics table after each episode."""
    header = f"    {'Pathology':<18s} {'Composite':>9s} {'Dx':>5s} {'G.Dx':>5s} {'PE':>5s} {'Labs':>5s} {'Img':>5s} {'InvT':>5s}"
    sep = "    " + "-" * len(header.strip())
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for p, composite in per_pathology.items():
        logger.info(f"    {p:<18s} {composite:>9.3f}")
    logger.info(sep)
    if per_metric:
        dx = per_metric.get("Diagnosis", 0)
        gdx = per_metric.get("Gracious Diagnosis", 0)
        pe = per_metric.get("Physical Examination", 0)
        labs = per_metric.get("Laboratory Tests", 0)
        img = per_metric.get("Imaging", 0)
        inv = per_metric.get("Invalid Tools", 0)
        logger.info(
            f"    {'AVERAGE':<18s} {'':>9s} {dx:>5.2f} {gdx:>5.2f} {pe:>5.2f} {labs:>5.2f} {img:>5.2f} {inv:>5.2f}"
        )
        logger.info(sep)


def append_episode_jsonl(episode_num, composite, best_score, per_pathology, per_metric, duration_s, parent_idx, node_idx):
    """Append one JSON line per episode to episode_log.jsonl."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "episode": episode_num,
        "node": node_idx,
        "parent": parent_idx,
        "composite": round(composite, 4),
        "best": round(best_score, 4),
        "duration_s": round(duration_s, 1),
        "per_pathology": {k: round(v, 4) for k, v in per_pathology.items()} if per_pathology else {},
        "per_metric": {k: round(v, 4) for k, v in per_metric.items()} if per_metric else {},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(EPISODE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def setup_evotest_logging(log_dir):
    """Configure logging to both file and console.

    Returns the log file path. Creates {log_dir}/evotest_sdk.log.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "evotest_sdk.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return str(log_path)


# ---------------------------------------------------------------------------
# Trajectory Format Bridge
# ---------------------------------------------------------------------------
def _json_safe(val):
    """Convert a value to something JSON-serializable."""
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(item) for item in val]
    return str(val)


def _safe_tool_input(tool_input):
    """Make tool_input JSON-serializable."""
    if isinstance(tool_input, dict):
        result = {}
        for k, v in tool_input.items():
            if isinstance(v, dict):
                result[k] = {str(kk): str(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                result[k] = [str(item) for item in v]
            else:
                result[k] = str(v)
        return result
    return str(tool_input)


def _safe_answers(answers):
    """Make answers dict JSON-serializable."""
    result = {}
    for k, v in answers.items():
        if isinstance(v, dict):
            result[k] = {str(kk): _json_safe(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            result[k] = [_json_safe(item) for item in v]
        else:
            result[k] = _json_safe(v)
    return result


def _trajectory_to_json_steps(trajectory):
    """Convert [(AgentAction, observation)] to JSON-safe step list.

    Matches the format produced by extract_trajectories.py:parse_trajectory_steps().
    """
    steps = []
    for action, observation in trajectory:
        step = {
            "tool": action.tool,
            "tool_input": _safe_tool_input(action.tool_input),
            "log": action.log,
            "custom_parsings": getattr(action, "custom_parsings", 0),
            "observation": str(observation)[:2000],
        }
        steps.append(step)
    return steps


def build_admission_record(
    patient_id, patient_data, trajectory, prediction, eval_result, final_output
):
    """Build a single admission dict in the format expected by evolve_skill.py.

    Args:
        patient_id: HADM ID.
        patient_data: hadm_info_clean[id] dict.
        trajectory: [(AgentAction, observation)] from convert_sdk_result.
        prediction: "Final Diagnosis: ...\\nTreatment: ..." string.
        eval_result: Dict with 'scores' and 'answers' from PathologyEvaluator.
        final_output: DiagnosticResult from SDK agent.

    Returns:
        Dict matching extract_trajectories.py output format.
    """
    return {
        "hadm_id": patient_id,
        "input": patient_data.get("Patient History", ""),
        "output": prediction,
        "trajectory": _trajectory_to_json_steps(trajectory),
        "scores": eval_result.get("scores", {}),
        "answers": _safe_answers(eval_result.get("answers", {})),
        "discharge_summary": patient_data.get("Discharge", ""),
        "discharge_diagnosis": patient_data.get("Discharge Diagnosis", ""),
    }


def build_trajectory_data(pathology, admissions):
    """Wrap admissions list into the trajectory data format expected by evolve_skill.py.

    Returns dict with keys: pathology, n_patients, aggregate, admissions.
    """
    aggregate = {}
    if admissions:
        score_keys = set()
        for a in admissions:
            score_keys.update(a["scores"].keys())
        for key in score_keys:
            values = [a["scores"].get(key, 0) for a in admissions]
            aggregate[key] = sum(values) / len(values) if values else 0

    return {
        "pathology": pathology,
        "n_patients": len(admissions),
        "aggregate": aggregate,
        "admissions": admissions,
    }


# ---------------------------------------------------------------------------
# SDKEvoTest
# ---------------------------------------------------------------------------
class SDKEvoTest:
    """EvoTest evolutionary skill optimization for the OpenAI Agents SDK agent."""

    def __init__(self, args):
        self.args = args
        self.nodes = []
        self.best_node_idx = None
        self.best_score = float("-inf")
        self.last_episode_score = None
        self.completed_episodes = 0

        # Paths (separate from LangChain version)
        self.data_dir = PROJECT_DIR / "data_splits"
        self.traj_dir = PROJECT_DIR / "trajectories_sdk"
        self.skills_dir = PROJECT_DIR / "skills" / "evo_sdk"
        self.lab_test_mapping = PROJECT_DIR / "MIMIC-CDM-IV" / "lab_test_mapping.pkl"

        # Incremental results pkl (crash-resilient, matching Hager pattern)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.results_pkl = STATE_DIR / "results.pkl"
        self.eval_pkl = STATE_DIR / "eval.pkl"

        # Resolved model (str or LitellmModel)
        self._resolved_model = None

        # Load clinical guidelines for Evolver context
        self.guidelines_context = ""
        if not getattr(args, "no_guidelines", False):
            gdir = getattr(args, "guidelines_dir", None) or DEFAULT_GUIDELINES_DIR
            self.guidelines_context = load_guidelines_context(gdir, pathologies=args.pathologies)
            if self.guidelines_context:
                logger.info(f"  Loaded clinical guidelines ({len(self.guidelines_context)} chars) from {gdir}")

    def _resolve_model(self):
        """Resolve --model / --litellm-model into the value ManagerConfig expects."""
        if self._resolved_model is not None:
            return self._resolved_model

        if self.args.litellm_model:
            from agents.extensions.models.litellm_model import LitellmModel
            self._resolved_model = LitellmModel(
                model=self.args.litellm_model,
                **({"base_url": self.args.litellm_base_url} if self.args.litellm_base_url else {}),
            )
        else:
            self._resolved_model = self.args.model
        return self._resolved_model

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
            logger.info(
                f"  [UCB] Force-selecting best node {self.best_node_idx} "
                f"(score={self.best_score:.3f}) after drop "
                f"(last={self.last_episode_score:.3f}, "
                f"threshold={self.args.drop_threshold})"
            )
            return self.best_node_idx

        # Normal UCB selection
        best_ucb_idx = max(range(len(self.nodes)), key=self.calculate_ucb)
        ucb_val = self.calculate_ucb(best_ucb_idx)
        logger.info(
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
                    - 0.3 * s.get("Action Parsing", 0)
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
    # Episode Runner (async, in-process)
    # ------------------------------------------------------------------
    async def run_episode(self, skill_text, episode_num, sub_agent_skill_text=None):
        """Run SDK agent on all pathologies with given skill, evaluate, score.

        All in-process via async Python — no subprocess overhead.

        Args:
            skill_text: Orchestrator skill text (or None).
            episode_num: Episode number for file naming.
            sub_agent_skill_text: Raw sub-agent skill text with section delimiters (or None).

        Returns (composite_score, per_metric, per_pathology, trajectory_paths) or None on failure.
        """
        skill_file = None
        sub_agent_skill_file = None

        # Save and sanitize sub-agent skill if provided
        if sub_agent_skill_text:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            raw_sub_path = self.skills_dir / f"episode_{episode_num}_subagents_raw.md"
            raw_sub_path.write_text(sub_agent_skill_text)

            sanitized_sub = sanitize_skill_text(sub_agent_skill_text)
            sub_agent_skill_file = self.skills_dir / f"episode_{episode_num}_subagents.md"
            sub_agent_skill_file.write_text(sanitized_sub)
            logger.info(f"  Saved sub-agent skill: {sub_agent_skill_file} ({len(sanitized_sub)} chars)")

        # Save and sanitize skill if provided
        if skill_text:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            raw_path = self.skills_dir / f"episode_{episode_num}_raw.md"
            raw_path.write_text(skill_text)

            sanitized = sanitize_skill_text(skill_text)
            skill_file = self.skills_dir / f"episode_{episode_num}.md"
            skill_file.write_text(sanitized)
            logger.info(f"  Saved skill: {skill_file} ({len(sanitized)} chars)")

        trajectory_paths = []
        all_trajectory_data = []
        pathologies = self.args.pathologies

        for pathology in pathologies:
            patient_data_path = self.data_dir / pathology / f"{self.args.split}.pkl"
            if not patient_data_path.exists():
                logger.warning(f"  {patient_data_path} not found, skipping {pathology}")
                continue

            if self.args.dry_run:
                trajectory_paths.append(
                    f"(dry-run) trajectories_sdk/evo_ep{episode_num}_{pathology}.json"
                )
                continue

            # Load patients
            patients = load_hadm_from_file(self.args.split, base_mimic=str(self.data_dir / pathology))
            patient_ids = list(patients.keys())
            logger.info(f"  [{pathology}] Running {len(patient_ids)} patients...")

            # Create manager (sub-agents use same model as orchestrator)
            config = ManagerConfig(
                model=self._resolve_model(),
                lab_test_mapping_path=str(self.lab_test_mapping),
                annotate_clinical=self.args.annotate_clinical,
                skill_path=str(skill_file) if skill_file else None,
                sub_agent_skill_path=str(sub_agent_skill_file) if sub_agent_skill_file else None,
                max_turns=self.args.max_turns,
            )
            # sub_agent_model defaults to None → same as main model
            manager = ClinicalDiagnosisManager(config)

            admissions = []
            try:
                from tqdm import tqdm
                patient_iter = tqdm(
                    enumerate(patient_ids),
                    total=len(patient_ids),
                    desc=f"    {pathology:.<18s}",
                    unit="pat",
                    leave=True,
                    bar_format="{desc} {bar:20} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
            except ImportError:
                patient_iter = enumerate(patient_ids)
            for pid_idx, patient_id in patient_iter:
                logger.debug(f"    [{pid_idx + 1}/{len(patient_ids)}] Processing patient: {patient_id}")

                try:
                    final_output, run_result, ctx = await manager.run(
                        patient_id, patients[patient_id]
                    )

                    # Convert to evaluator format
                    trajectory, prediction = convert_sdk_result(
                        run_result,
                        final_output,
                        itemid_log=ctx.lab_itemid_log,
                    )

                    # Evaluate
                    evaluator = load_evaluator(pathology)
                    reference = build_reference_tuple(patients[patient_id])
                    eval_result = evaluator._evaluate_agent_trajectory(
                        prediction=prediction,
                        input=patients[patient_id]["Patient History"],
                        agent_trajectory=trajectory,
                        reference=reference,
                    )

                    admission = build_admission_record(
                        patient_id, patients[patient_id],
                        trajectory, prediction, eval_result, final_output,
                    )
                    admissions.append(admission)

                    dx_score = eval_result["scores"].get("Diagnosis", 0)
                    logger.info(f"      -> Dx={dx_score} ({final_output.diagnosis[:40]})")
                    logger.info(f"      Eval: {eval_result['scores']}")

                    # Incremental save (crash-resilient)
                    append_to_pickle_file(self.results_pkl, {
                        "episode": episode_num, "pathology": pathology,
                        "patient_id": patient_id, "diagnosis": final_output.diagnosis,
                        "scores": eval_result["scores"],
                    })
                    append_to_pickle_file(self.eval_pkl, {
                        "episode": episode_num, "pathology": pathology,
                        "patient_id": patient_id, "eval": eval_result,
                    })

                except Exception as e:
                    logger.error(f"      ERROR on patient {patient_id}: {e}")
                    logger.debug(traceback.format_exc())
                    # Create dummy admission with zero scores so the episode can continue
                    dummy_scores = {
                        "Diagnosis": 0, "Gracious Diagnosis": 0,
                        "Physical Examination": 0, "Late Physical Examination": 0,
                        "Laboratory Tests": 0, "Imaging": 0,
                        "Action Parsing": 0, "Invalid Tools": 0, "Rounds": 0,
                    }
                    admission = {
                        "hadm_id": patient_id,
                        "input": patients[patient_id].get("Patient History", ""),
                        "output": f"ERROR: {e}",
                        "trajectory": [],
                        "scores": dummy_scores,
                        "answers": {},
                        "discharge_summary": patients[patient_id].get("Discharge", ""),
                        "discharge_diagnosis": patients[patient_id].get("Discharge Diagnosis", ""),
                    }
                    admissions.append(admission)

            # Build trajectory data and save JSON
            traj_data = build_trajectory_data(pathology, admissions)
            all_trajectory_data.append(traj_data)

            self.traj_dir.mkdir(parents=True, exist_ok=True)
            traj_output = self.traj_dir / f"evo_ep{episode_num}_{pathology}.json"
            with open(traj_output, "w") as f:
                json.dump(traj_data, f, indent=2, default=str)
            trajectory_paths.append(str(traj_output))
            logger.info(f"  [{pathology}] Saved trajectory: {traj_output}")

        if self.args.dry_run:
            return 0.0, {}, {}, trajectory_paths

        if not all_trajectory_data:
            logger.error("  No trajectory data collected")
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
            logger.info(f"\n{'='*60}")
            logger.info("DRY RUN — Evolver prompt:")
            logger.info(f"{'='*60}")
            logger.info(prompt[:3000])
            if len(prompt) > 3000:
                logger.info(f"... [{len(prompt) - 3000} chars truncated]")
            logger.info(f"Prompt length: {len(prompt)} chars")
            return "(dry-run skill placeholder)"

        logger.info(f"  Calling Evolver ({self.args.evolver_model})...")
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.args.evolver_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        skill_text = message.content[0].text
        logger.info(f"  Evolver produced skill ({len(skill_text)} chars)")
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

        # --- Section 2: Current performance (blinded: no disease names) ---
        if trajectory_data_list:
            aggregate_table = build_aggregate_table(trajectory_data_list, blind=True)
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

            pathologies = self.args.pathologies
            for pathology in pathologies:
                path_failures = [f for f in all_failures if f["pathology"] == pathology]
                for fail in path_failures[:2]:
                    admission = fail["admission"]
                    reasons = ", ".join(fail["reasons"])
                    analysis = (
                        f"---\n"
                        f"{format_trajectory_summary(admission, pathology=fail['pathology'], blind=True)}\n\n"
                        f"**Failure reasons**: {reasons}\n\n"
                        f"**Real Doctor's Discharge Summary**:\n```\n"
                        f"{format_discharge_summary(admission, blind=True)}\n```\n"
                    )
                    gap_analyses.append(analysis)
            gap_analyses = gap_analyses[:14]

        gap_section = "\n".join(gap_analyses) if gap_analyses else "(no failures to analyze)"

        # --- Section 4: Parent skill ---
        parent_skill_section = ""
        if parent_node and parent_node.get("skill_text"):
            parent_skill_section = (
                f"## Parent Skill (score={parent_node['score']:.3f})\n\n"
                f"Analyze where it helped and where it failed, then IMPROVE it:\n\n"
                f"```markdown\n{parent_node['skill_text']}\n```\n\n"
            )

        # --- Section 5: Clinical guidelines ---
        guidelines_section = ""
        if self.guidelines_context:
            guidelines_section = (
                f"## Evidence-Based Clinical Guidelines\n\n"
                f"The following are condensed extracts from peer-reviewed clinical practice guidelines "
                f"(PubMed, NICE). Use these to ground your skill in evidence-based diagnostic and "
                f"treatment protocols.\n\n"
                f"{self.guidelines_context}\n\n"
            )

        prompt = f"""You are a clinical AI system optimizer. Your task is to analyze diagnostic agent trajectories and generate an improved clinical reasoning skill.

## Agent Architecture

The agent uses the OpenAI Agents SDK with these capabilities:
- **Structured output** (DiagnosticResult): diagnosis, confidence, key_evidence, differential, treatment, severity — all required fields, enforced by Pydantic schema
- **JSON-validated tool calls**: tool hallucination and parsing errors are impossible (Action Parsing = 1.0, Invalid Tools = 0 always)
- **Lab Interpreter sub-agent**: dedicated specialist for interpreting lab panels
- **Challenger sub-agent**: devil's advocate that challenges proposed diagnoses

Focus your skill improvements on REASONING QUALITY — the agent already handles formatting perfectly.

## Evolution History

{history_section}

## Current Agent Performance

{aggregate_table}

### Per-Metric Analysis

{metric_section}

{parent_skill_section}{guidelines_section}## Failed Trajectories with Gap Analysis

{gap_section}

## Your Task

Generate an improved GENERAL clinical reasoning workflow skill for diagnosing patients presenting with acute abdominal pain. The skill must work for ANY abdominal condition — not just a fixed set of diseases. This skill must:

1. **Teach hypothesis-driven diagnostic reasoning** — maintain a running differential, choose each test to maximally discriminate between remaining hypotheses. Do NOT write disease-specific decision trees or "if symptom X then disease Y" rules.
2. **Teach organ-system-based localization** — map pain location to anatomical structures (e.g., RUQ → hepatobiliary, gallbladder, right kidney, hepatic flexure; epigastric → stomach, pancreas, aorta), then reason about which organ is affected based on additional findings.
3. **Address the specific failure patterns above** — focus on what went wrong and teach the correct REASONING APPROACH (not a disease-specific fix).
4. **Work for ANY acute abdominal condition** — the skill must generalize to diseases the agent has never seen, including obstruction, ischemia, perforation, renal, gynecological, and vascular causes.
5. **Stay under 500 tokens** — concise, actionable instructions.
6. **NOT use disease names** — use ____ as a mask. Do NOT use thinly-disguised patterns like "____itis (appendiceal)" that effectively name the disease.

Focus on PROCESS, not CONTENT:
- ALWAYS do Physical Examination first — it localizes the problem and generates the initial differential
- Select labs that discriminate between the top 2-3 hypotheses (not shotgun ordering)
- Choose imaging modality by suspected organ system, not by suspected disease
- Interpret results by updating the differential (which hypotheses are supported/eliminated?)
- Decide treatment by severity indicators (peritonitis, sepsis, obstruction, perforation) not by diagnosis name

Output ONLY the skill content in markdown format. No preamble or explanation."""

        return prompt

    # ------------------------------------------------------------------
    # Sub-Agent Evolver
    # ------------------------------------------------------------------
    def evolve_sub_agent_skills(self, parent_node, trajectory_data_list, orchestrator_skill):
        """Call the Evolver (Opus) to generate improved sub-agent skills.

        Returns the raw text with ``<!-- SECTION: lab_interpreter -->`` and
        ``<!-- SECTION: challenger -->`` delimiters.
        """
        prompt = self._build_sub_agent_evolver_prompt(
            parent_node, trajectory_data_list, orchestrator_skill
        )

        if self.args.dry_run:
            logger.info(f"\n{'='*60}")
            logger.info("DRY RUN — Sub-Agent Evolver prompt:")
            logger.info(f"{'='*60}")
            logger.info(prompt[:3000])
            if len(prompt) > 3000:
                logger.info(f"... [{len(prompt) - 3000} chars truncated]")
            logger.info(f"Sub-agent prompt length: {len(prompt)} chars")
            return (
                "<!-- SECTION: lab_interpreter -->\n(dry-run lab interpreter skill)\n\n"
                "<!-- SECTION: challenger -->\n(dry-run challenger skill)"
            )

        logger.info(f"  Calling Sub-Agent Evolver ({self.args.evolver_model})...")
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.args.evolver_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        skill_text = message.content[0].text
        logger.info(f"  Sub-Agent Evolver produced skill ({len(skill_text)} chars)")
        return skill_text

    def _build_sub_agent_evolver_prompt(self, parent_node, trajectory_data_list, orchestrator_skill):
        """Build Evolver prompt for Lab Interpreter and Challenger sub-agents."""
        # --- Performance data (same as orchestrator) ---
        if trajectory_data_list:
            aggregate_table = build_aggregate_table(trajectory_data_list)
        else:
            aggregate_table = "(no trajectory data)"

        # --- Failed trajectories (focus on lab/reasoning failures) ---
        gap_analyses = []
        if trajectory_data_list:
            all_failures = []
            for data in trajectory_data_list:
                failures = identify_failures(data)
                all_failures.extend(failures)

            for fail in all_failures[:8]:
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

        gap_section = "\n".join(gap_analyses) if gap_analyses else "(no failures to analyze)"

        # --- Parent sub-agent skills ---
        parent_sub_section = ""
        if parent_node and parent_node.get("sub_agent_skill_text"):
            parent_sub_section = (
                f"## Parent Sub-Agent Skills (for reference — improve these)\n\n"
                f"```markdown\n{parent_node['sub_agent_skill_text']}\n```\n\n"
            )

        # --- Orchestrator skill context ---
        orch_section = ""
        if orchestrator_skill:
            orch_section = (
                f"## Current Orchestrator Skill\n\n"
                f"The main diagnostic agent uses this skill. Your sub-agent skills should "
                f"COMPLEMENT it (not duplicate). Focus on what the orchestrator does NOT cover:\n\n"
                f"```markdown\n{orchestrator_skill[:500]}\n```\n\n"
            )

        # --- Clinical guidelines ---
        guidelines_section = ""
        if self.guidelines_context:
            guidelines_section = (
                f"## Evidence-Based Clinical Guidelines\n\n"
                f"{self.guidelines_context}\n\n"
            )

        prompt = f"""You are a clinical AI system optimizer. Your task is to generate improved instructions for two specialist sub-agents that assist the main diagnostic agent.

## Agent Architecture

The diagnostic system has 3 agents:
1. **Orchestrator** — main agent that examines, orders tests, reasons, diagnoses
2. **Lab Interpreter** — specialist called after lab results to identify patterns and abnormalities
3. **Challenger** — devil's advocate called before finalizing diagnosis to catch errors

You are generating evolved skills for agents #2 and #3 ONLY.

## Current Agent Performance

{aggregate_table}

{orch_section}{parent_sub_section}{guidelines_section}## Failed Trajectories

{gap_section}

## Your Task

Generate improved skills for BOTH sub-agents. Output in this EXACT format with HTML comment delimiters:

<!-- SECTION: lab_interpreter -->
(Your improved Lab Interpreter instructions here — max 300 tokens)

<!-- SECTION: challenger -->
(Your improved Challenger instructions here — max 300 tokens)

### Lab Interpreter Guidelines:
- Focus on PATTERN recognition across multiple lab values
- Teach which combinations of abnormalities point to specific pathological processes
- Include reference ranges and what deviations mean clinically
- Emphasize acute vs chronic patterns
- Do NOT name specific diseases — describe pathological processes

### Challenger Guidelines:
- Focus on common ANCHORING BIASES in acute abdominal pain diagnosis
- Teach which alternative diagnoses are commonly missed for each presentation
- Include specific evidence patterns that should trigger reconsideration
- Emphasize SEVERITY assessment — when conservative management is insufficient
- Do NOT name specific diseases — describe pathological processes

Output ONLY the two sections in the format above. No preamble or explanation."""

        return prompt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self):
        """Save full tree to evotest_state_sdk/state.json."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        best_score_safe = self.best_score if math.isfinite(self.best_score) else None
        state = {
            "nodes": self.nodes,
            "best_node_idx": self.best_node_idx,
            "best_score": best_score_safe,
            "last_episode_score": self.last_episode_score,
            "completed_episodes": self.completed_episodes,
            "args": {
                "model": self.args.model,
                "litellm_model": self.args.litellm_model,
                "litellm_base_url": self.args.litellm_base_url,
                "evolver_model": self.args.evolver_model,
                "annotate_clinical": self.args.annotate_clinical,
                "max_turns": self.args.max_turns,
                "split": self.args.split,
                "pathologies": self.args.pathologies,
                "exploration_constant": self.args.exploration_constant,
                "depth_constant": self.args.depth_constant,
                "drop_threshold": self.args.drop_threshold,
                "force_best_after_drop": self.args.force_best_after_drop,
            },
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"  State saved to {STATE_FILE} ({len(self.nodes)} nodes)")

    def load_state(self):
        """Load state from checkpoint. Returns True if successful."""
        if not STATE_FILE.exists():
            logger.warning(f"  No state file found at {STATE_FILE}")
            return False
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        self.nodes = state["nodes"]
        self.best_node_idx = state["best_node_idx"]
        self.best_score = state["best_score"] if state["best_score"] is not None else float("-inf")
        self.last_episode_score = state["last_episode_score"]
        self.completed_episodes = state["completed_episodes"]
        logger.info(
            f"  Resumed from {STATE_FILE}: "
            f"{self.completed_episodes} episodes, "
            f"{len(self.nodes)} nodes, "
            f"best_score={self.best_score:.3f}"
        )
        return True

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    async def run(self):
        """Main EvoTest loop (async)."""
        start_time = time.time()
        episode_durations = []  # Track for ETA calculation

        # Setup logging — writes to evotest_state_sdk/evotest_sdk.log
        log_path = setup_evotest_logging(STATE_DIR)

        # Resume or fresh start
        if self.args.resume:
            if not self.load_state():
                logger.info("  Cannot resume — starting fresh")

        start_episode = self.completed_episodes
        total_episodes = self.args.episodes

        if start_episode >= total_episodes:
            logger.info(
                f"Already completed {start_episode} episodes (target={total_episodes}). "
                f"Increase --episodes to continue."
            )
            return

        model_label = self.args.litellm_model or self.args.model
        logger.info(f"{'='*70}")
        logger.info(f"EvoTest SDK | {total_episodes} episodes | {len(self.args.pathologies)} pathologies")
        logger.info(f"{'='*70}")
        logger.info(f"  Model:            {model_label}")
        logger.info(f"  Evolver:          {self.args.evolver_model}")
        logger.info(f"  Pathologies:      {', '.join(self.args.pathologies)}")
        logger.info(f"  Split:            {self.args.split}")
        logger.info(f"  Annotate clinical:{self.args.annotate_clinical}")
        logger.info(f"  Max turns:        {self.args.max_turns}")
        logger.info(f"  UCB c={self.args.exploration_constant}, alpha={self.args.depth_constant}")
        logger.info(f"  Drop threshold:   {self.args.drop_threshold}")
        logger.info(f"  Log file:         {log_path}")
        logger.info(f"  Episode log:      {EPISODE_LOG}")
        logger.info(f"{'='*70}")

        for episode_num in range(start_episode, total_episodes):
            ep_start = time.time()
            episodes_done = episode_num - start_episode
            eta_str = ""
            if episode_durations:
                avg_dur = sum(episode_durations) / len(episode_durations)
                remaining = (total_episodes - episode_num) * avg_dur
                eta_str = f" | ETA: {format_duration(remaining)}"

            logger.info(f"{'='*70}")
            logger.info(f"EPISODE {episode_num}/{total_episodes - 1} [{episodes_done}/{total_episodes - start_episode} done]{eta_str}")
            logger.info(f"{'='*70}")

            if episode_num == 0 and not self.nodes:
                # --- Episode 0: Baseline or seed skill ---
                skill_text = ""
                if self.args.initial_skill:
                    skill_path = Path(self.args.initial_skill)
                    if skill_path.exists():
                        skill_text = skill_path.read_text()
                        logger.info(f"  Using initial skill from {skill_path}")
                    else:
                        logger.warning(f"  {skill_path} not found, running without skill")

                logger.info("  Running baseline episode...")
                result = await self.run_episode(skill_text if skill_text else None, episode_num)
                if result is None:
                    logger.error("  Episode 0 failed — aborting")
                    return

                composite, per_metric, per_pathology, traj_paths = result
                node = {
                    "idx": 0,
                    "skill_text": skill_text,
                    "sanitized_skill": sanitize_skill_text(skill_text) if skill_text else "",
                    "sub_agent_skill_text": "",
                    "sanitized_sub_agent_skill": "",
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
                # --- Episodes 1..N: UCB select -> evolve -> run -> score ---
                parent_idx = self.select_parent()
                parent_node = self.nodes[parent_idx]

                # Load parent's trajectory data for Evolver context
                trajectory_data_list = []
                for tpath in parent_node.get("trajectory_paths", []):
                    if os.path.exists(tpath):
                        with open(tpath, "r") as f:
                            trajectory_data_list.append(json.load(f))

                # Evolve orchestrator skill
                logger.info(f"  Evolving from node {parent_idx} (score={parent_node['score']:.3f})...")
                new_skill = self.evolve_skill(parent_node, trajectory_data_list)

                # Evolve sub-agent skills (second Opus call)
                logger.info(f"  Evolving sub-agent skills...")
                new_sub_agent_skill = self.evolve_sub_agent_skills(
                    parent_node, trajectory_data_list, new_skill
                )

                # Run episode with both skills
                logger.info(f"  Running episode with evolved skills...")
                result = await self.run_episode(new_skill, episode_num, sub_agent_skill_text=new_sub_agent_skill)
                if result is None:
                    logger.error(f"  Episode {episode_num} failed — creating failed node and continuing")
                    node = {
                        "idx": len(self.nodes),
                        "skill_text": new_skill,
                        "sanitized_skill": sanitize_skill_text(new_skill),
                        "sub_agent_skill_text": new_sub_agent_skill or "",
                        "sanitized_sub_agent_skill": sanitize_skill_text(new_sub_agent_skill) if new_sub_agent_skill else "",
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
                    ep_elapsed = time.time() - ep_start
                    episode_durations.append(ep_elapsed)
                    continue

                composite, per_metric, per_pathology, traj_paths = result

                # Create child node
                node = {
                    "idx": len(self.nodes),
                    "skill_text": new_skill,
                    "sanitized_skill": sanitize_skill_text(new_skill),
                    "sub_agent_skill_text": new_sub_agent_skill or "",
                    "sanitized_sub_agent_skill": sanitize_skill_text(new_sub_agent_skill) if new_sub_agent_skill else "",
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
                    logger.info(f"  *** NEW BEST: score={composite:.3f} (node {node['idx']}) ***")

            # Episode summary with metrics table
            node = self.nodes[-1]
            ep_elapsed = time.time() - ep_start
            episode_durations.append(ep_elapsed)
            avg_dur = sum(episode_durations) / len(episode_durations)
            remaining_eps = total_episodes - (episode_num + 1)
            eta = remaining_eps * avg_dur

            logger.info(f"")
            logger.info(f"  Episode {episode_num} completed in {format_duration(ep_elapsed)}")
            logger.info(f"    Composite: {node['score']:.3f} | Best: {self.best_score:.3f} (node {self.best_node_idx})")
            if node.get("per_pathology"):
                print_metrics_table(node["per_pathology"], node.get("per_metric", {}))
            logger.info(f"    Tree: {len(self.nodes)} nodes | Avg: {format_duration(avg_dur)}/ep | ETA: {format_duration(eta)}")

            # Append to structured episode log
            append_episode_jsonl(
                episode_num=episode_num,
                composite=node["score"],
                best_score=self.best_score,
                per_pathology=node.get("per_pathology", {}),
                per_metric=node.get("per_metric", {}),
                duration_s=ep_elapsed,
                parent_idx=node.get("parent_idx", -1),
                node_idx=node["idx"],
            )

            self.completed_episodes = episode_num + 1
            self.save_state()

        # Final summary
        total_elapsed = time.time() - start_time
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"EVOTEST SDK COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"  Episodes:     {self.completed_episodes}")
        logger.info(f"  Nodes:        {len(self.nodes)}")
        logger.info(f"  Best score:   {self.best_score:.3f} (node {self.best_node_idx})")
        logger.info(f"  Total time:   {format_duration(total_elapsed)}")
        if episode_durations:
            logger.info(f"  Avg/episode:  {format_duration(sum(episode_durations) / len(episode_durations))}")
        logger.info(f"  Log file:     {log_path}")
        logger.info(f"  Episode log:  {EPISODE_LOG}")

        if self.best_node_idx is not None:
            best = self.nodes[self.best_node_idx]
            best_skill_path = self.skills_dir / f"episode_{best['episode_num']}.md"
            logger.info(f"  Best skill:   {best_skill_path}")
            if best.get("per_pathology"):
                logger.info(f"  Per-pathology:")
                for p, s in best["per_pathology"].items():
                    logger.info(f"    {p:20s}: {s:.3f}")

        # Score progression
        if len(self.nodes) > 1:
            logger.info(f"")
            logger.info(f"  Score progression:")
            max_score = max(n["score"] for n in self.nodes) if self.nodes else 1
            for n in self.nodes:
                bar_len = int(30 * n["score"] / max_score) if max_score > 0 else 0
                bar = "#" * bar_len + "." * (30 - bar_len)
                best_marker = " * best" if n["idx"] == self.best_node_idx else ""
                logger.info(f"    Ep {n['episode_num']:>2d}: {bar} {n['score']:.3f}{best_marker}")

        # Tree structure
        logger.info(f"")
        logger.info(f"  Tree structure:")
        for n in self.nodes:
            indent = "  " * n["depth"]
            marker = " ***BEST***" if n["idx"] == self.best_node_idx else ""
            logger.info(
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
        description="EvoTest-style evolutionary skill optimization for the OpenAI Agents SDK agent"
    )
    # --- Same as evotest_clinical.py ---
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Total number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--evolver-model", type=str, default="claude-opus-4-6",
        help="Anthropic model for the Evolver (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--exploration-constant", type=float, default=1.0,
        help="UCB exploration constant c (default: 1.0)",
    )
    parser.add_argument(
        "--depth-constant", type=float, default=0.8,
        help="UCB depth decay alpha (default: 0.8)",
    )
    parser.add_argument(
        "--drop-threshold", type=float, default=1.0,
        help="Force-best-after-drop threshold (default: 1.0)",
    )
    parser.add_argument(
        "--force-best-after-drop", action="store_true", default=True,
        help="Force-select best node after large score drop (default: True)",
    )
    parser.add_argument(
        "--no-force-best-after-drop", action="store_false", dest="force_best_after_drop",
        help="Disable force-best-after-drop",
    )
    parser.add_argument(
        "--initial-skill", type=str, default=None,
        help="Path to initial seed skill for episode 0 (optional)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands and Evolver prompt without running anything",
    )

    # --- SDK-specific ---
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--litellm-model", type=str, default=None,
        help="LiteLLM model string, e.g. 'openai/Qwen3-30B-A3B'",
    )
    parser.add_argument(
        "--litellm-base-url", type=str, default=None,
        help="Base URL for LiteLLM (e.g. http://gpu:8000/v1)",
    )
    parser.add_argument(
        "--annotate-clinical", action="store_true", default=True,
        help="Enable clinical lab annotations (default: True)",
    )
    parser.add_argument(
        "--no-annotate-clinical", action="store_false", dest="annotate_clinical",
        help="Disable clinical lab annotations",
    )
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum agent turns per patient (default: 20)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "test", "full"],
        help="Which data split to use (default: train = 10 patients)",
    )
    parser.add_argument(
        "--pathologies", type=str, nargs="+", default=ALL_PATHOLOGIES,
        choices=ALL_PATHOLOGIES,
        help="Pathologies to include (default: all 7)",
    )
    parser.add_argument(
        "--guidelines-dir", type=str, default=None,
        help="Path to guidelines/ directory (default: auto-detect from project root)",
    )
    parser.add_argument(
        "--no-guidelines", action="store_true",
        help="Disable clinical guidelines injection into Evolver prompt",
    )

    args = parser.parse_args()

    # Validate
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    runner = SDKEvoTest(args)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
