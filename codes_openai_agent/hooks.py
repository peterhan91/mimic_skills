"""Lifecycle hooks for observability during clinical diagnosis runs.

ClinicalRunHooks logs tool calls with timing and tracks tool ordering
in PatientContext.tool_call_log for post-hoc analysis.
"""

import logging
import time

from agents import RunHooks, RunContextWrapper

from context import PatientContext

logger = logging.getLogger("sdk_agent")


class ClinicalRunHooks(RunHooks[PatientContext]):
    """Observability hooks: log tool calls, timing, turn count."""

    def __init__(self):
        self.turn_count = 0
        self.tool_timings: list[tuple[str, float]] = []
        self._tool_start: float = 0.0

    async def on_tool_start(self, context, agent, tool):
        self._tool_start = time.time()
        self.turn_count += 1
        tool_name = tool.name if hasattr(tool, "name") else str(tool)
        logger.debug(f"  Tool start: {tool_name} (turn {self.turn_count})")
        context.context.tool_call_log.append(tool_name)

    async def on_tool_end(self, context, agent, tool, result):
        elapsed = time.time() - self._tool_start
        tool_name = tool.name if hasattr(tool, "name") else str(tool)
        self.tool_timings.append((tool_name, elapsed))
        logger.debug(f"  Tool end: {tool_name} ({elapsed:.1f}s)")
