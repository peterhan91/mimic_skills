# OpenAI Agents Python SDK — Analysis for MIMIC-Skills

**Repo**: https://github.com/openai/openai-agents-python
**Docs**: https://openai.github.io/openai-agents-python/
**Install**: `pip install openai-agents`

## What It Is

The OpenAI Agents SDK is a lightweight, Python-first framework for building
multi-agent workflows. It is the production-ready successor to OpenAI's
experimental "Swarm" framework (March 2025). Design philosophy: minimal
abstractions, code-first orchestration, no graph DSLs.

---

## Six Core Primitives

### 1. Agent

An LLM configured with instructions, tools, handoffs, and an optional output type.

```python
from agents import Agent

agent = Agent(
    name="Clinical Agent",
    instructions="You are a medical AI assistant...",  # OR a function(ctx) -> str
    model="gpt-4o",                                    # OR litellm/anthropic/claude-3-5-sonnet
    tools=[physical_exam, lab_tests, imaging],
    handoffs=[specialist_agent],
    output_type=DiagnosticResult,                      # Pydantic model → structured output
    input_guardrails=[safety_check],
    output_guardrails=[treatment_validator],
    hooks=AgentLifecycleHooks(),
)
```

Key: `instructions` can be a **dynamic function** `(RunContextWrapper) -> str` that
generates context-dependent instructions at runtime — perfect for per-patient
skill injection.

### 2. Runner

The execution engine (agent loop). Three methods:
- `Runner.run()` — async, returns `RunResult`
- `Runner.run_sync()` — synchronous wrapper
- `Runner.run_streamed()` — async with real-time partial results

**The agent loop**:
1. Invoke agent with current input
2. If agent produces `final_output` (matches `output_type` or plain text with no tool calls) → terminate
3. If agent issues handoff → switch to new agent, re-enter loop
4. If agent makes tool calls → execute tools, append results, re-enter loop
5. If `max_turns` exceeded (default 10) → raise `MaxTurnsExceeded`

### 3. Tools

Three categories:
- **Function tools**: `@function_tool` decorator (auto-generates JSON schema from type annotations)
- **Hosted tools**: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ComputerTool`
- **Agents as tools**: `agent.as_tool()` — sub-agent as callable tool, returns output without transferring control

```python
from agents import function_tool, RunContextWrapper

@function_tool
async def run_lab_tests(ctx: RunContextWrapper[PatientContext], tests: list[str]) -> str:
    """Order and retrieve laboratory test results for the current patient."""
    patient = ctx.context.current_patient
    ctx.context.ordered_tests.extend(tests)
    return format_lab_results(patient, tests)
```

### 4. Handoffs

Transfer full conversation control from one agent to another:

```python
triage_agent = Agent(
    name="Triage",
    instructions="Route to specialist based on chief complaint.",
    handoffs=[cardiology_agent, gastro_agent, general_agent]
)
```

Represented as tools to the LLM (`transfer_to_cardiology_agent`). Original agent's
turn ends; new agent takes over entire conversation.

### 5. Guardrails

Three layers:
- **Input guardrails**: Validate user input before/alongside first agent call
- **Output guardrails**: Validate final agent output before returning to user
- **Tool guardrails**: Wrap individual function tools with before/after validation

Tool guardrails support:
- *Input tool guardrails*: Run before tool execution; can skip call, replace output, or raise tripwire
- *Output tool guardrails*: Run after tool execution; can replace output or raise tripwire

Tripwire mechanism halts execution when validation fails.

### 6. Tracing

Built-in observability:
- **Trace** = one end-to-end `Runner.run()` call
- **Spans** = individual operations (LLM call, tool execution, handoff, guardrail)
- Default: sent to OpenAI dashboard
- Exporters: Langfuse, AgentOps, Braintrust, Arize, or custom

---

## Multi-Agent Orchestration Patterns

### Pattern A: Handoffs (Delegation)
One agent hands off full control. Suitable for triage/routing.

### Pattern B: Agents as Tools (Subordination)
Orchestrator calls sub-agents as tools, retains control, gets results back.

```python
lab_interpreter = Agent(name="Lab Interpreter", instructions="Interpret lab values...")
diagnostic_agent = Agent(
    name="Diagnostic Agent",
    tools=[lab_interpreter.as_tool()]  # sub-agent as tool
)
```

### Pattern C: Programmatic Orchestration
Plain Python control flow:
```python
result1 = await Runner.run(agent_a, input=patient_history)
if result1.final_output.needs_specialist:
    result2 = await Runner.run(agent_b, input=result1.final_output.data)
```

---

## Context Management

### RunContextWrapper (runtime context)
Typed wrapper around user-defined context. NOT sent to LLM — purely for passing
dependencies and state to tool functions, hooks, and guardrails.

```python
@dataclass
class PatientContext:
    patient_id: str
    history: dict
    ordered_tests: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

result = await Runner.run(agent, input="...", context=PatientContext(...))
```

### Sessions (persistent memory)
Store conversation history across multiple runs. Backends: SQLite, SQLAlchemy, Dapr, OpenAI-hosted, encrypted.

### Context strategies
- **Trimming**: Drop older turns, keep last N
- **Summarization**: Compress prior messages into structured summaries

---

## Model Flexibility

Any model via LiteLLM:
```python
pip install "openai-agents[litellm]"

agent = Agent(model="litellm/anthropic/claude-3-5-sonnet-20240620")
# Or: litellm/google/gemini-pro, litellm/together_ai/meta-llama/Llama-3.3-70B, etc.
```

Custom OpenAI-compatible endpoints (vLLM, Ollama):
```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://gpu-server:8000/v1", api_key="local")
agent = Agent(model=OpenAIChatCompletionsModel(model="llama-3.3-70b", openai_client=client))
```

Per-agent model selection: different agents in same workflow can use different models/providers.

---

## Structured Output (Critical for Clinical Use)

```python
class DiagnosticResult(BaseModel):
    diagnosis: str
    confidence: float
    evidence: list[str]
    treatment: str
    severity: str

agent = Agent(
    name="Diagnostic Agent",
    output_type=DiagnosticResult,  # Loop won't end until schema is filled
    tools=[physical_exam, lab_tests, imaging]
)
```

The agent loop terminates ONLY when the LLM produces valid output matching the
Pydantic schema. This eliminates parsing errors and ensures complete outputs.

---

## Relevance to Our Clinical Agent

### Direct Mapping: Hager → OpenAI Agents SDK

| Hager's Framework | OpenAI Agents SDK Equivalent |
|---|---|
| `CHAT_TEMPLATE` (system prompt) | `Agent.instructions` (static or dynamic function) |
| `{examples}` slot | Part of instructions or `RunContextWrapper` |
| ZeroShotAgent ReAct loop | Built-in agent loop in `Runner.run()` |
| `Actions.py` tools (PE, Labs, Imaging) | `@function_tool` decorated functions |
| `ReadDiagnosticCriteria` tool | `@function_tool` or agent-as-tool |
| `PathologyEvaluator` | External (run after `Runner.run()` completes) |
| `agent_scratchpad` | Automatic — managed by agent loop internally |
| Tool hallucination (#5) | **Eliminated** — function calling validates tool names |
| Output format parsing (#8) | **Eliminated** — structured output via `output_type` |
| Lab annotation (Approach 3) | `@function_tool` adds annotations before returning |
| Skill injection (Approach 1/2) | Dynamic `instructions` function injects per-patient skills |

### What It Fixes (Failure Modes from Hager et al.)

| Failure Mode | Current Mitigation | SDK Solution |
|---|---|---|
| #5 Hallucinate tools | Fuzzy matching + penalty | **Eliminated**: function calling constrains to defined tools |
| #8 Phrasing sensitivity | Format instructions in prompt | **Eliminated**: `output_type` enforces schema |
| #2 Lab interpretation | Approach 3 annotations | `@function_tool` with annotation logic + agent-as-tool lab interpreter |
| #3 Skip PE | Prompt rules | Tool guardrails can enforce PE-first ordering |
| #1 Poor accuracy | Skills | Dynamic instructions + multi-agent reasoning |
| #4 Treatment gaps | Enhanced DiagCrit | Structured output requires treatment field |

### What It Does NOT Fix

- **Step ordering enforcement**: SDK does NOT enforce PE → Labs → Imaging at framework level (relies on instructions). LangGraph would be better for strict workflow control.
- **Self-reflection**: No built-in "reflect and retry" mechanism (must implement programmatically).
- **Evaluation integration**: PathologyEvaluator would need adaptation since it currently parses LangChain trajectory formats.

---

## High-Impact Design Patterns to Adopt

### 1. Multi-Agent Clinical Reasoning (Agents-as-Tools)

Replace single omniscient agent with specialized sub-agents:

```python
lab_interpreter = Agent(
    name="Lab Interpreter",
    instructions="""Given lab results with reference ranges, provide clinical interpretation.
    For each abnormal value, explain:
    1. Severity (mild/moderate/severe)
    2. Clinical significance (what condition it suggests)
    3. What additional tests to consider""",
    model="gpt-4o-mini"  # cheap, fast specialist
)

imaging_interpreter = Agent(
    name="Imaging Interpreter",
    instructions="Interpret radiology reports, identify key findings...",
    model="gpt-4o-mini"
)

diagnostic_agent = Agent(
    name="Diagnostic Agent",
    instructions="You are a clinical reasoning agent. Use your specialist tools...",
    tools=[physical_exam, run_lab_tests, run_imaging,
           lab_interpreter.as_tool(),      # sub-agent for interpretation
           imaging_interpreter.as_tool()],  # sub-agent for interpretation
    output_type=DiagnosticResult,
    model="gpt-4o"  # strong orchestrator
)
```

**Why this matters**: The paper showed agents fail at interpreting labs (#2). A
dedicated lab interpreter sub-agent can receive the raw values and return
structured interpretations — better than static annotations (Approach 3)
because it can reason about combinations of values and clinical context.

### 2. Tool Guardrails for Clinical Safety

```python
from agents import function_tool

@function_tool(
    input_guardrails=[validate_lab_order],    # check before running
    output_guardrails=[annotate_lab_results]  # augment after running
)
async def run_lab_tests(ctx: RunContextWrapper[PatientContext], tests: list[str]) -> str:
    """Order and retrieve laboratory test results."""
    ...

async def validate_lab_order(ctx, tool_input):
    """Reject nonsensical test combinations, suggest corrections."""
    if not ctx.context.pe_performed and not any(t in tool_input.tests for t in STAT_TESTS):
        return GuardrailResult(
            tripwire_triggered=False,
            message="Consider performing Physical Examination first before ordering labs."
        )
    return GuardrailResult(tripwire_triggered=False)
```

### 3. Dynamic Skill Injection via Instructions Function

```python
def build_instructions(ctx: RunContextWrapper[PatientContext]) -> str:
    base = "You are a medical AI assistant diagnosing acute abdominal pain..."

    # Inject skill based on context
    if ctx.context.skill_path:
        with open(ctx.context.skill_path) as f:
            skill = f.read()
        base += f"\n\n## Clinical Reasoning Skill\n\n{skill}"

    # Inject patient-specific context
    if ctx.context.previous_findings:
        base += f"\n\nFindings so far: {ctx.context.previous_findings}"

    return base

agent = Agent(name="Diagnostic Agent", instructions=build_instructions)
```

### 4. Structured Diagnostic Output

```python
class DiagnosticResult(BaseModel):
    diagnosis: str = Field(description="Primary diagnosis")
    confidence: float = Field(ge=0, le=1, description="Confidence 0-1")
    evidence: list[str] = Field(description="Key findings supporting diagnosis")
    differential: list[str] = Field(description="Alternative diagnoses considered")
    treatment: str = Field(description="Recommended treatment plan")
    severity: Literal["uncomplicated", "moderate", "severe", "critical"]

agent = Agent(output_type=DiagnosticResult, ...)
# Agent MUST produce all fields before loop terminates
```

### 5. Built-in Tracing for EvoTest Integration

Every tool call, reasoning step, and handoff is automatically traced. Export
trajectory data directly to EvoTest's Evolver for skill refinement:

```python
from agents.tracing import custom_span

with custom_span("clinical_episode", data={"pathology": "appendicitis", "patient_id": "123"}):
    result = await Runner.run(agent, input=patient_history, context=ctx)
    # Trajectory automatically captured in trace
    # Export to EvoTest for skill evolution
```

---

## Implementation Strategy

### Phase 1: Parallel Prototype (Low Risk)

Build a new agent implementation using the SDK alongside Hager's existing code.
Both implementations use the same patient data and PathologyEvaluator for
apples-to-apples comparison.

```
codes_Hager/          ← existing LangChain agent (unchanged)
codes_openai_agent/   ← new SDK-based agent
  agent.py            ← Agent + tools definition
  tools.py            ← @function_tool implementations
  context.py          ← PatientContext dataclass
  evaluator_adapter.py← Adapt PathologyEvaluator for SDK trajectories
  run.py              ← Runner.run() with skill injection
```

### Phase 2: Feature Adoption

Selectively adopt high-value patterns into the existing framework:
- Structured output parsing (replace regex with Pydantic)
- Tool validation (input/output guardrails concept)
- Dynamic instruction generation

### Phase 3: Full Migration (If Phase 1 Shows Improvement)

Replace LangChain ZeroShotAgent with SDK-based agent as the primary
evaluation target. Keep Hager's patient data loading and evaluation
infrastructure.

---

## Key Advantages Over Current Architecture

| Aspect | Current (LangChain) | SDK-Based | Impact |
|---|---|---|---|
| Tool calling | Regex parsing (brittle) | JSON schema (guaranteed) | Eliminates failure #5, #8 |
| Output format | Text parsing | Pydantic model | Complete, structured diagnoses |
| Error recovery | Penalty tracking | Error fed back to LLM | Self-correction |
| Skill injection | String concat into template | Dynamic function | Per-patient, context-aware |
| Multi-agent | Single agent loop | Agents-as-tools | Specialized reasoning |
| Token management | Post-hoc truncation | Context strategies | Less information loss |
| Tracing | Custom logging | Built-in spans | Automatic trajectory capture |
| Model flexibility | Custom LLM wrapper | LiteLLM (100+ providers) | Easy A/B testing |
| Guardrails | None | Input/output/tool | Clinical safety validation |

## Key Limitations

1. **No workflow enforcement**: Cannot enforce PE → Labs → Imaging ordering at
   framework level (LangGraph better for this)
2. **Migration effort**: Rewriting tools + evaluation adapter is non-trivial
3. **Prompt control**: Less fine-grained than Hager's CHAT_TEMPLATE tag system
4. **OpenAI dependency**: Core framework designed for OpenAI API; other providers
   via LiteLLM adapter (works but less tested)

---

## References

- [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)
- [GitHub: openai/openai-agents-python](https://github.com/openai/openai-agents-python)
- [Multi-Agent Orchestration](https://openai.github.io/openai-agents-python/multi_agent/)
- [Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- [Tools](https://openai.github.io/openai-agents-python/tools/)
- [Tracing](https://openai.github.io/openai-agents-python/tracing/)
- [Context & Sessions](https://openai.github.io/openai-agents-python/context/)
- [LiteLLM Integration](https://openai.github.io/openai-agents-python/models/litellm/)
- [AgentKit (Enterprise)](https://openai.com/index/introducing-agentkit/)
- [AI-Agents-for-Medical-Diagnostics](https://github.com/ahmadvh/AI-Agents-for-Medical-Diagnostics)
- [AgentClinic Benchmark](https://agentclinic.github.io/)
