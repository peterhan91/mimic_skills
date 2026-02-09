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
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from agents.prompts import (
    CHAT_TEMPLATE,
    DIAG_CRIT_TOOL_DESCR,
    DIAG_CRIT_TOOL_USE_EXAMPLE,
    TOOL_USE_EXAMPLES,
)
from agents.tot_prompts import TOT_EVALUATION_PROMPT
from tools.Tools import (
    DoPhysicalExamination,
    ReadDiagnosticCriteria,
    RunImaging,
    RunLaboratoryTests,
)
from tools.utils import action_input_pretty_printer
from utils.nlp import calculate_num_tokens


@dataclass
class ToTState:
    """A single node in the search tree."""

    intermediate_steps: List[Tuple[AgentAction, str]] = field(default_factory=list)
    imaging_state: Dict = field(default_factory=dict)
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
        max_depth: int = 10,
        temperature: float = 0.7,
        eval_temperature: float = 0.0,
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

                observation, new_imaging_state = self._execute_tool(
                    action, state.imaging_state
                )

                ns = ToTState(
                    intermediate_steps=list(state.intermediate_steps)
                    + [(action, observation)],
                    imaging_state=new_imaging_state,
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

    # ── TOOL EXECUTION ──────────────────────────────────────────────

    def _execute_tool(
        self, action: AgentAction, imaging_state: Dict
    ) -> Tuple[str, Dict]:
        """Execute a tool, using cache for deterministic tools.

        Returns:
            (observation, new_imaging_state): The new_imaging_state reflects
            the actual mutations made by retrieve_imaging (only incremented
            when a scan is successfully returned, not on repeats or misses).
        """
        tool_name = action.tool
        tool_input = action.tool_input

        # Check cache for deterministic tools
        if tool_name != "Imaging":
            cached = self.cache.get(tool_name, tool_input)
            if cached is not None:
                return cached, imaging_state

        if tool_name not in self.tools:
            return (
                f"Invalid tool: {tool_name}. Use one of: {list(self.tools.keys())}",
                imaging_state,
            )

        tool = self.tools[tool_name]

        # For Imaging, create a fresh instance with this branch's state.
        # retrieve_imaging mutates already_requested_scans only on success,
        # so we capture the post-execution state from the branch tool.
        try:
            if tool_name == "Imaging":
                branch_state = copy.deepcopy(imaging_state)
                branch_tool = RunImaging(
                    action_results=tool.action_results,
                    already_requested_scans=branch_state,
                )
                result = branch_tool._run(**tool_input)
                return result, branch_tool.already_requested_scans
            else:
                result = tool._run(**tool_input)
                self.cache.put(tool_name, tool_input, result)
                return result, imaging_state
        except Exception as e:
            logger.warning(f"[ToT] Tool {tool_name} raised {type(e).__name__}: {e}")
            return f"Tool error: {e}", imaging_state

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
    # ToT-specific params
    tot_n_generate=3,
    tot_breadth=2,
    tot_max_depth=10,
    tot_temperature=0.7,
    tot_eval_temperature=0.0,
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
    )
