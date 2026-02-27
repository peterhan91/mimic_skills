"""Tree of Thoughts (ToT) agent for clinical diagnosis.

BFS over diagnostic paths: at each depth, sample k next-actions with
temperature, evaluate each path, keep top-b, repeat until final diagnosis
or max depth.
"""

import copy
import pickle
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import AgentFinish
from loguru import logger

from agents.AgentAction import AgentAction
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser, InvalidActionError
from agents.prompts import (
    CHAT_TEMPLATE,
    DIAG_CRIT_TOOL_DESCR,
    DIAG_CRIT_TOOL_USE_EXAMPLE,
    TOOL_USE_EXAMPLES,
    ASK_PATIENT_TOOL_DESCR,
    ASK_PATIENT_TOOL_USE_EXAMPLE,
)
from agents.tot_prompts import TOT_EVALUATION_PROMPT
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from evaluators.cholangitis_evaluator import CholangitisEvaluator
from evaluators.bowel_obstruction_evaluator import BowelObstructionEvaluator
from evaluators.pyelonephritis_evaluator import PyelonephritisEvaluator
from tools.Tools import (
    DoPhysicalExamination,
    ReadDiagnosticCriteria,
    RunImaging,
    RunLaboratoryTests,
)
from tools.utils import action_input_pretty_printer
from utils.nlp import calculate_num_tokens

# Maps pathology name → evaluator class (for GRPO rubric eval)
EVALUATOR_MAP = {
    "appendicitis": AppendicitisEvaluator,
    "cholecystitis": CholecystitisEvaluator,
    "diverticulitis": DiverticulitisEvaluator,
    "pancreatitis": PancreatitisEvaluator,
    "cholangitis": CholangitisEvaluator,
    "bowel_obstruction": BowelObstructionEvaluator,
    "pyelonephritis": PyelonephritisEvaluator,
}

# Max "Laboratory Tests" score per pathology (1 point per required category)
MAX_LAB_SCORE = {
    "appendicitis": 1,
    "cholecystitis": 3,
    "diverticulitis": 1,
    "pancreatitis": 3,
    "cholangitis": 3,
    "bowel_obstruction": 2,
    "pyelonephritis": 3,
}


@dataclass
class ToTState:
    """A single node in the search tree."""

    intermediate_steps: List[Tuple[AgentAction, str]] = field(default_factory=list)
    imaging_state: Dict = field(default_factory=dict)
    patient_sim_history: List[Tuple[str, str]] = field(default_factory=list)
    depth: int = 0
    value: float = 0.0
    finished: bool = False
    prediction: str = ""


class ToolResultCache:
    """Memoizes deterministic tool results (PE, Labs, DiagCrit).

    Imaging is excluded because it is stateful (already_requested_scans).
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def _key(self, tool_name: str, tool_input: Any) -> str:
        return sha256(f"{tool_name}::{tool_input}".encode()).hexdigest()

    def get(self, tool_name: str, tool_input: Any) -> Optional[str]:
        return self._cache.get(self._key(tool_name, tool_input))

    def put(self, tool_name: str, tool_input: Any, result: str) -> None:
        self._cache[self._key(tool_name, tool_input)] = result


class TreeOfThoughtsRunner:
    """BFS over diagnostic reasoning paths.

    Produces the same output format as LangChain AgentExecutor:
        {"output": str, "intermediate_steps": List[Tuple[AgentAction, str]]}
    """

    def __init__(
        self,
        llm,
        prompt: PromptTemplate,
        parser: DiagnosisWorkflowParser,
        tools: Dict[str, Any],
        patient: Dict,
        tags: Dict[str, str],
        stop_words: List[str],
        max_context_length: int,
        n_generate: int = 3,
        breadth: int = 2,
        max_depth: int = 20,
        temperature: float = 0.7,
        eval_temperature: float = 0.0,
        patient_simulator=None,
        eval_mode: str = "combined",
        pathology: Optional[str] = None,
    ):
        self.llm = llm
        self.prompt = prompt
        self.parser = parser
        self.tools = tools  # name -> tool instance
        self.patient = patient
        self.tags = tags
        self.stop_words = stop_words
        self.max_context_length = max_context_length
        self.n_generate = n_generate
        self.breadth = breadth
        self.max_depth = max_depth
        self.temperature = temperature
        self.eval_temperature = eval_temperature
        self.patient_simulator = patient_simulator
        self.eval_mode = eval_mode
        self.pathology = pathology
        self.cache = ToolResultCache()

    # ── public interface (matches AgentExecutor.__call__) ────────────

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        patient_input = inputs["input"]

        # Seed the search with an empty state
        frontier: List[ToTState] = [ToTState()]
        finished_states: List[ToTState] = []

        for depth in range(1, self.max_depth + 1):
            logger.info(
                f"[ToT] depth={depth}/{self.max_depth}  frontier={len(frontier)}  finished={len(finished_states)}"
            )

            candidates: List[ToTState] = []

            for state in frontier:
                new_states = self._generate_thoughts(state, patient_input)
                for ns in new_states:
                    if ns.finished:
                        finished_states.append(ns)
                    else:
                        candidates.append(ns)

            if not candidates:
                break

            # EVALUATE each candidate
            if self.eval_mode == "grpo" and self._has_ground_truth():
                self._evaluate_grpo(candidates, patient_input, depth)
            elif self.eval_mode == "grpo":
                # Fallback: no ground truth available (test time)
                self._evaluate_combined(candidates, patient_input, depth)
            elif self.eval_mode == "combined":
                self._evaluate_combined(candidates, patient_input, depth)
            else:
                for c in candidates:
                    c.value = self._evaluate_state(c, patient_input, depth)

            # SELECT top-b
            candidates.sort(key=lambda s: s.value, reverse=True)
            frontier = candidates[: self.breadth]

            logger.info(
                f"[ToT] kept {len(frontier)} states, scores={[round(s.value, 1) for s in frontier]}"
            )

        # Pick best finished state, or force-finish best frontier state
        if finished_states:
            best = max(finished_states, key=lambda s: s.value)
        elif frontier:
            best = max(frontier, key=lambda s: s.value)
            best = self._force_finish(best, patient_input)
        else:
            # Degenerate case: nothing generated at all
            best = ToTState(prediction="Unable to reach a diagnosis.", finished=True)

        return {
            "input": patient_input,
            "output": best.prediction,
            "intermediate_steps": best.intermediate_steps,
        }

    # ── GENERATE ────────────────────────────────────────────────────

    def _generate_thoughts(
        self, state: ToTState, patient_input: str
    ) -> List[ToTState]:
        """Sample k next-actions from the LLM with temperature, deduplicate."""
        scratchpad = self._build_scratchpad(state.intermediate_steps)
        prompt_text = self.prompt.format(
            input=patient_input, agent_scratchpad=scratchpad
        )

        # Check if we are near context limit
        n_tokens = calculate_num_tokens(self.llm.tokenizer, [prompt_text])
        if n_tokens >= self.max_context_length - 200:
            # Force a diagnosis instead of generating more actions
            forced = self._force_finish(state, patient_input)
            return [forced]

        seen_actions: set = set()
        new_states: List[ToTState] = []

        for _ in range(self.n_generate):
            raw_output = self.llm.generate_with_temperature(
                prompt_text,
                stop=self.stop_words,
                temperature=self.temperature,
            )
            parsed = self.parser.parse(raw_output)

            if isinstance(parsed, AgentFinish):
                ns = ToTState(
                    intermediate_steps=list(state.intermediate_steps),
                    imaging_state=copy.deepcopy(state.imaging_state),
                    patient_sim_history=list(state.patient_sim_history),
                    depth=state.depth + 1,
                    finished=True,
                    prediction=parsed.return_values.get("output", raw_output),
                    value=state.value,  # inherit parent's evaluated score
                )
                new_states.append(ns)
                continue

            # parsed is a list of AgentAction (usually length 1)
            for action in parsed:
                dedup_key = (action.tool, str(action.tool_input))
                if dedup_key in seen_actions:
                    continue
                seen_actions.add(dedup_key)

                observation, new_imaging_state, new_sim_history = self._execute_tool(
                    action, state.imaging_state, state.patient_sim_history
                )

                ns = ToTState(
                    intermediate_steps=list(state.intermediate_steps)
                    + [(action, observation)],
                    imaging_state=new_imaging_state,
                    patient_sim_history=new_sim_history or list(state.patient_sim_history),
                    depth=state.depth + 1,
                )
                new_states.append(ns)

        return new_states

    # ── EVALUATE ────────────────────────────────────────────────────

    def _evaluate_state(
        self, state: ToTState, patient_input: str, depth: int
    ) -> float:
        """Ask the LLM to rate this path 1-10."""
        scratchpad_text = self._build_scratchpad_plain(state.intermediate_steps)

        eval_prompt = TOT_EVALUATION_PROMPT.format(
            system_tag_start=self.tags["system_tag_start"],
            system_tag_end=self.tags["system_tag_end"],
            user_tag_start=self.tags["user_tag_start"],
            user_tag_end=self.tags["user_tag_end"],
            ai_tag_start=self.tags["ai_tag_start"],
            input=patient_input,
            scratchpad=scratchpad_text,
            depth=depth,
            max_depth=self.max_depth,
        )

        raw = self.llm.generate_with_temperature(
            eval_prompt,
            stop=["\n"],
            temperature=self.eval_temperature,
        )

        # Parse integer from response
        match = re.search(r"\d+", raw)
        if match:
            score = int(match.group())
            return float(min(max(score, 1), 10))
        return 5.0  # default if parsing fails

    # ── COMBINED EVALUATION (structural + LLM) ─────────────────────

    def _compute_structural_reward(self, state: ToTState) -> float:
        """Lean structural checks — PE ordering, invalid tools, duplicates.

        Deliberately omits lab and imaging scoring: the old GRPO reward
        gave +0.5 per union-lab hit, incentivising over-ordering (failure
        mode #6 from the paper).  Semantic assessment of lab relevance and
        imaging appropriateness is left to the LLM evaluator.
        """
        reward = 0.0
        seen_actions: set = set()

        for idx, (action, _observation) in enumerate(state.intermediate_steps):
            tool = action.tool

            # Duplicate detection
            dedup_key = (tool, str(action.tool_input))
            if dedup_key in seen_actions:
                reward -= 0.5
                continue
            seen_actions.add(dedup_key)

            # Invalid / hallucinated tool
            if tool == InvalidActionError.invalid_tool_str:
                reward -= 1.0
                continue

            # PE: full credit if first action, partial if late
            if tool == "Physical Examination":
                reward += 2.0 if idx == 0 else 0.5

        # Mild efficiency pressure
        reward -= len(state.intermediate_steps) * 0.1
        return reward

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        """Zero-mean, unit-variance normalisation with skip trick."""
        n = len(values)
        if n == 0:
            return []
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = var ** 0.5
        if std < 1e-4:
            return [0.0] * n
        return [(v - mean) / (std + 1e-4) for v in values]

    def _evaluate_combined(
        self, candidates: List[ToTState], patient_input: str, depth: int
    ) -> None:
        """Structural guardrails (0.3) + LLM semantic judgement (0.7).

        Both signals are independently normalised to zero-mean unit-variance
        before combining, so neither dominates due to scale differences.
        """
        if not candidates:
            return

        # Structural signal — instant, deterministic
        struct_rewards = [self._compute_structural_reward(c) for c in candidates]
        struct_adv = self._normalize(struct_rewards)

        # Semantic signal — LLM-as-judge
        llm_scores = [
            self._evaluate_state(c, patient_input, depth) for c in candidates
        ]
        llm_adv = self._normalize(llm_scores)

        # Weighted combination: LLM dominates, structure guards
        for c, sa, la in zip(candidates, struct_adv, llm_adv):
            c.value = 0.3 * sa + 0.7 * la

        logger.info(
            f"[ToT-combined] structural={[round(r, 2) for r in struct_rewards]} "
            f"llm={[round(s, 1) for s in llm_scores]} "
            f"final={[round(c.value, 2) for c in candidates]}"
        )

    # ── GRPO RUBRIC EVALUATION ──────────────────────────────────────

    def _has_ground_truth(self) -> bool:
        """Check if patient dict has ground-truth data for rubric scoring."""
        return bool(self.patient.get("Discharge Diagnosis"))

    def _build_reference_tuple(self) -> tuple:
        """Build the 5-element reference tuple for PathologyEvaluator."""
        return (
            self.patient.get("Discharge Diagnosis", ""),
            self.patient.get("ICD Diagnosis", []),
            self.patient.get("Procedures ICD9", []),
            self.patient.get("Procedures ICD10", []),
            self.patient.get("Procedures Discharge", []),
        )

    def _compute_rubric_reward(self, state: ToTState, patient_input: str) -> float:
        """Force-finish a candidate and score it with the real PathologyEvaluator.

        Returns the composite reward (same formula as evotest_clinical.py).
        """
        # Force-finish to get diagnosis + treatment text
        finished = self._force_finish(state, patient_input)

        # Fresh evaluator (has mutable state)
        evaluator_cls = EVALUATOR_MAP.get(self.pathology)
        if evaluator_cls is None:
            logger.warning(f"[ToT-grpo] No evaluator for pathology={self.pathology}, returning 0")
            return 0.0
        evaluator = evaluator_cls()

        reference = self._build_reference_tuple()

        eval_result = evaluator._evaluate_agent_trajectory(
            prediction=finished.prediction,
            input=patient_input,
            agent_trajectory=finished.intermediate_steps,
            reference=reference,
        )

        s = eval_result["scores"]
        max_lab = MAX_LAB_SCORE.get(self.pathology, 1)

        reward = (
            3.0 * s.get("Diagnosis", 0)
            + 1.0 * s.get("Physical Examination", 0)
            + 0.5 * s.get("Late Physical Examination", 0)
            + 1.0 * min(s.get("Laboratory Tests", 0) / max_lab, 1.0)
            + 1.0 * min(s.get("Imaging", 0) / 2.0, 1.0)
            - 0.5 * min(s.get("Invalid Tools", 0), 2)
            - 0.3 * (1 - s.get("Action Parsing", 0))
        )
        return reward

    def _evaluate_grpo(
        self, candidates: List[ToTState], patient_input: str, depth: int
    ) -> None:
        """GRPO-style evaluation: rubric rewards → normalize → set c.value."""
        if not candidates:
            return

        raw_rewards = [
            self._compute_rubric_reward(c, patient_input) for c in candidates
        ]
        advantages = self._normalize(raw_rewards)

        for c, adv in zip(candidates, advantages):
            c.value = adv

        logger.info(
            f"[ToT-grpo] depth={depth} "
            f"raw_rewards={[round(r, 3) for r in raw_rewards]} "
            f"advantages={[round(a, 3) for a in advantages]}"
        )

    # ── TOOL EXECUTION ──────────────────────────────────────────────

    def _execute_tool(
        self,
        action: AgentAction,
        imaging_state: Dict,
        patient_sim_history: Optional[List] = None,
    ) -> Tuple[str, Dict, Optional[List]]:
        """Execute a tool, using cache for deterministic tools.

        Returns:
            (observation, new_imaging_state, new_patient_sim_history)
        """
        tool_name = action.tool
        tool_input = action.tool_input
        sim_history = patient_sim_history

        # Check cache for deterministic tools (Ask Patient is keyed by
        # question + history, so identical contexts hit cache)
        if tool_name not in ("Imaging", "Ask Patient"):
            cached = self.cache.get(tool_name, tool_input)
            if cached is not None:
                return cached, imaging_state, sim_history

        # Cache Ask Patient by (question, history_tuple) for determinism
        if tool_name == "Ask Patient" and self.patient_simulator:
            history = list(sim_history or [])
            cache_key = (str(tool_input), tuple(history))
            cached = self.cache.get(tool_name, cache_key)
            if cached is not None:
                new_history = history + [(tool_input if isinstance(tool_input, str) else tool_input.get("action_input", str(tool_input)), cached)]
                return cached, imaging_state, new_history

        if tool_name not in self.tools:
            return (
                f"Invalid tool: {tool_name}. Use one of: {list(self.tools.keys())}",
                imaging_state,
                sim_history,
            )

        tool = self.tools[tool_name]

        try:
            if tool_name == "Imaging":
                branch_state = copy.deepcopy(imaging_state)
                branch_tool = RunImaging(
                    action_results=tool.action_results,
                    already_requested_scans=branch_state,
                )
                result = branch_tool._run(**tool_input)
                return result, branch_tool.already_requested_scans, sim_history
            elif tool_name == "Ask Patient" and self.patient_simulator:
                # Branch-aware: use explicit history, not tool's internal state
                history = list(sim_history or [])
                question = tool_input if isinstance(tool_input, str) else tool_input.get("action_input", str(tool_input))
                result = self.patient_simulator.respond(
                    question=question, history=history
                )
                cache_key = (str(tool_input), tuple(history))
                self.cache.put(tool_name, cache_key, result)
                new_history = history + [(question, result)]
                return result, imaging_state, new_history
            else:
                result = tool._run(**tool_input)
                self.cache.put(tool_name, tool_input, result)
                return result, imaging_state, sim_history
        except Exception as e:
            logger.warning(f"[ToT] Tool {tool_name} raised {type(e).__name__}: {e}")
            return f"Tool error: {e}", imaging_state, sim_history

    # ── SCRATCHPAD FORMATTING ───────────────────────────────────────

    def _build_scratchpad(
        self, steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Build scratchpad in exact same format as CustomZeroShotAgent._construct_scratchpad."""
        thoughts = ""
        for action, observation in steps:
            thoughts += action.log
            thoughts += (
                f"{self.tags['ai_tag_end']}"
                f"{self.tags['user_tag_start']}"
                f"Observation: {observation.strip()}"
                f"{self.tags['user_tag_end']}"
                f"{self.tags['ai_tag_start']}"
                f"Thought:"
            )
        return " " + thoughts.strip()

    def _build_scratchpad_plain(
        self, steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Build a plain-text scratchpad for the evaluation prompt."""
        parts = []
        for action, observation in steps:
            # Extract the thought from the log
            log = action.log.strip()
            parts.append(log)
            parts.append(f"Observation: {observation.strip()}")
        return "\n".join(parts)

    # ── FORCE FINISH ────────────────────────────────────────────────

    def _force_finish(self, state: ToTState, patient_input: str) -> ToTState:
        """Force the LLM to produce a final diagnosis from the current state."""
        scratchpad = self._build_scratchpad(state.intermediate_steps)
        # Append instruction to diagnose
        scratchpad += (
            f"{self.tags['ai_tag_end']}"
            f"{self.tags['user_tag_start']}"
            f"Provide a Final Diagnosis and Treatment."
            f"{self.tags['user_tag_end']}"
            f"{self.tags['ai_tag_start']}"
            f"Final"
        )

        prompt_text = self.prompt.format(
            input=patient_input, agent_scratchpad=scratchpad
        )

        raw = self.llm.generate_with_temperature(
            prompt_text,
            stop=self.stop_words,
            temperature=0.0,
        )

        # Prepend "Final" since we put it in the scratchpad
        prediction = "Final" + raw

        return ToTState(
            intermediate_steps=list(state.intermediate_steps),
            imaging_state=copy.deepcopy(state.imaging_state),
            patient_sim_history=list(state.patient_sim_history),
            depth=state.depth,
            finished=True,
            prediction=prediction,
            value=state.value,
        )


# ── FACTORY FUNCTION ────────────────────────────────────────────────


def build_tot_runner(
    patient,
    llm,
    lab_test_mapping_path,
    logfile,
    max_context_length,
    tags,
    include_ref_range,
    bin_lab_results,
    include_tool_use_examples,
    provide_diagnostic_criteria,
    summarize,
    model_stop_words,
    skill_path=None,
    skill_inject="examples",
    annotate_clinical=False,
    patient_simulator=None,
    # ToT-specific params
    tot_n_generate=10,
    tot_breadth=3,
    tot_max_depth=20,
    tot_temperature=1.0,
    tot_eval_temperature=0.0,
    tot_eval_mode="combined",
    pathology=None,
):
    """Build a TreeOfThoughtsRunner with the same interface as build_agent_executor_ZeroShot."""
    with open(lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    # Define tools (same as ZeroShot)
    tools_list = [
        DoPhysicalExamination(action_results=patient),
        RunLaboratoryTests(
            action_results=patient,
            lab_test_mapping_df=lab_test_mapping_df,
            include_ref_range=include_ref_range,
            bin_lab_results=bin_lab_results,
            annotate_clinical=annotate_clinical,
        ),
        RunImaging(action_results=patient),
    ]

    add_tool_use_examples = ""
    add_tool_descr = ""
    if provide_diagnostic_criteria:
        tools_list.append(ReadDiagnosticCriteria())
        add_tool_descr += DIAG_CRIT_TOOL_DESCR
        add_tool_use_examples += DIAG_CRIT_TOOL_USE_EXAMPLE

    if patient_simulator:
        from tools.patient_simulator import AskPatient
        tools_list.append(AskPatient(simulator=patient_simulator))
        add_tool_descr += ASK_PATIENT_TOOL_DESCR
        add_tool_use_examples += ASK_PATIENT_TOOL_USE_EXAMPLE

    tool_names = [tool.name for tool in tools_list]
    tools_dict = {tool.name: tool for tool in tools_list}

    # Build prompt (same as ZeroShot)
    tool_use_examples = ""
    if include_tool_use_examples:
        tool_use_examples = TOOL_USE_EXAMPLES.format(
            add_tool_use_examples=add_tool_use_examples
        )

    # Load and inject skill (same logic as ZeroShot)
    if skill_path:
        import os
        import sys

        if os.path.exists(skill_path):
            with open(skill_path, "r") as sf:
                raw = sf.read()
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    raw = parts[2].strip()

            try:
                _proj_root = os.path.normpath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..")
                )
                _scripts_dir = os.path.join(_proj_root, "scripts")
                if _scripts_dir not in sys.path:
                    sys.path.insert(0, _scripts_dir)
                from sanitize_skill import sanitize_skill_text

                raw = sanitize_skill_text(raw)
                logger.info("Sanitized skill text (disease names masked with ____)")
            except ImportError:
                logger.warning(
                    "Could not import sanitize_skill; skill injected without sanitization"
                )

            logger.info(f"Loaded skill from {skill_path} ({len(raw)} chars)")
            if skill_inject in ("examples", "both"):
                tool_use_examples = f"\n{raw}\n\n" + tool_use_examples
            if skill_inject in ("system", "both"):
                add_tool_descr = add_tool_descr + f"\n{raw}"
        else:
            logger.warning(f"Skill file not found: {skill_path}")

    prompt = PromptTemplate(
        template=CHAT_TEMPLATE,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": action_input_pretty_printer(tool_names, None),
            "add_tool_descr": add_tool_descr,
            "examples": tool_use_examples,
            "system_tag_start": tags["system_tag_start"],
            "user_tag_start": tags["user_tag_start"],
            "ai_tag_start": tags["ai_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_end": tags["user_tag_end"],
        },
    )

    parser = DiagnosisWorkflowParser(lab_test_mapping_df=lab_test_mapping_df)

    stop = list(
        ["Observation:", "Observations:", "observation:", "observations:"]
        + model_stop_words
    )

    return TreeOfThoughtsRunner(
        llm=llm,
        prompt=prompt,
        parser=parser,
        tools=tools_dict,
        patient=patient,
        tags=tags,
        stop_words=stop,
        max_context_length=max_context_length,
        n_generate=tot_n_generate,
        breadth=tot_breadth,
        max_depth=tot_max_depth,
        temperature=tot_temperature,
        eval_temperature=tot_eval_temperature,
        patient_simulator=patient_simulator,
        eval_mode=tot_eval_mode,
        pathology=pathology,
    )
