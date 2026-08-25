"""Agent runner — simulates an agent executing a task with optional skill injection.

The agent uses ReAct-style execution via LLM. When skills are provided, they are
injected into the system prompt as reference strategies the agent can consult.
The agent produces a structured trace + final output that can be verified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


# ── Agent output data model ─────────────────────────────────────────

@dataclass
class AgentResult:
    """Structured result of an agent executing a task."""
    task_id: str
    mode: str                                       # "baseline" or "skill_augmented"
    trace: str = ""                                  # full ReAct execution trace
    steps_taken: list[str] = field(default_factory=list)    # list of actions/steps
    tools_called: list[str] = field(default_factory=list)   # tool names used
    final_output: str = ""                           # agent's final answer
    output_criteria: dict[str, bool] = field(default_factory=dict)  # self-assessed criteria
    num_steps: int = 0
    completed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── System prompts ───────────────────────────────────────────────────

AGENT_SYSTEM_BASE = """\
You are a data analysis agent with access to the following tools:
- file_read(path, preview_rows=5): Read a file and preview its contents
- code_execute(language, code): Execute code (Python/R) and return output
- web_search(query): Search the web for information
- api_call(service, endpoint, payload): Call an external API

You work in a ReAct loop: Thought → Action → Observation → ... → Final Answer.

For each task:
1. Inspect the data first before acting.
2. Validate data quality before processing.
3. Handle edge cases (missing values, format issues, outliers).
4. Produce a concrete, verifiable output.

After completing the task, provide a structured summary in this EXACT JSON format:
```json
{{"steps_taken": ["step 1 description", ...], "tools_called": ["tool_name", ...], "output_criteria": {{"criterion_name": true/false, ...}}, "completed": true/false}}
```

The output_criteria should assess whether you achieved each of these specific criteria:
{criteria_text}

Be thorough and realistic in your execution trace. Show actual code snippets and realistic observations."""


SKILL_INJECTION = """

### Reference Skills (from prior experience)
You have access to the following learned skills from previous similar tasks.
Consult these strategies when applicable — they encode proven approaches:

{skills_text}

Use these skills to guide your approach, but adapt them to the specific task at hand."""


AGENT_USER = """\
### Task
{task_description}

### Input Data
{test_input}

Execute this task step by step using your available tools. Show your full ReAct execution trace."""


# ── Core agent execution ─────────────────────────────────────────────

def run_agent(
    task_id: str,
    task_description: str,
    test_input: str,
    output_criteria: dict[str, bool],
    skills: list[Skill] | None = None,
    mode: str = "baseline",
) -> AgentResult:
    """Run the simulated agent on a task.

    Args:
        task_id: Test case identifier.
        task_description: What the agent should accomplish.
        test_input: Description of the input data available.
        output_criteria: Expected output criteria to self-assess (keys only used).
        skills: Optional list of evolved skills to inject into the agent's prompt.
        mode: "baseline" or "skill_augmented".

    Returns:
        AgentResult with full trace, steps, tools, and criteria assessment.
    """
    # Build criteria text for the system prompt
    criteria_text = "\n".join(f"- {k}" for k in output_criteria.keys())

    system = AGENT_SYSTEM_BASE.format(criteria_text=criteria_text)

    # Inject skills if provided
    if skills:
        skills_text = "\n\n".join(
            f"**{s.name}**\n"
            f"- When to use: {s.when_to_use}\n"
            f"- Procedure: {s.procedure}\n"
            f"- Examples: {'; '.join(s.examples[:2]) if s.examples else 'N/A'}"
            for s in skills
        )
        system += SKILL_INJECTION.format(skills_text=skills_text)

    prompt = AGENT_USER.format(
        task_description=task_description,
        test_input=test_input,
    )

    # Run the agent
    raw = llm_call(prompt, system_prompt=system)

    # Parse the structured summary from the end of the response
    trace = raw
    result = AgentResult(task_id=task_id, mode=mode, trace=trace)

    # Extract the JSON summary block
    json_block = _extract_json_block(raw)
    if json_block:
        try:
            summary = json.loads(json_block)
            result.steps_taken = summary.get("steps_taken", [])
            result.tools_called = summary.get("tools_called", [])
            result.output_criteria = summary.get("output_criteria", {})
            result.completed = summary.get("completed", False)
            result.num_steps = len(result.steps_taken)
        except json.JSONDecodeError:
            # Fallback: try to extract what we can from the trace
            result = _fallback_parse(raw, task_id, mode, output_criteria)
    else:
        result = _fallback_parse(raw, task_id, mode, output_criteria)

    return result


def _extract_json_block(text: str) -> str | None:
    """Extract the last JSON code block from the text."""
    # Try to find ```json ... ```
    parts = text.rsplit("```json", 1)
    if len(parts) == 2:
        block = parts[1].split("```", 1)[0].strip()
        return block

    # Try to find the last { ... } block that looks like our summary
    last_brace = text.rfind("{")
    if last_brace >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(last_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[last_brace:i + 1]
                    if "steps_taken" in candidate or "output_criteria" in candidate:
                        return candidate
                    break
    return None


def _fallback_parse(
    raw: str, task_id: str, mode: str, output_criteria: dict
) -> AgentResult:
    """Fallback: use LLM to extract structured info from a messy trace."""
    extract_prompt = f"""Extract structured information from this agent execution trace.

Return JSON:
{{"steps_taken": ["step descriptions"], "tools_called": ["tool names"], "output_criteria": {{{", ".join(f'"{k}": true/false' for k in output_criteria)}}}, "completed": true/false}}

Trace:
{raw[:3000]}"""

    fallback_raw = llm_call(extract_prompt)
    fallback_raw = fallback_raw.strip()
    if fallback_raw.startswith("```"):
        fallback_raw = fallback_raw.split("\n", 1)[1]
        fallback_raw = fallback_raw.rsplit("```", 1)[0]

    result = AgentResult(task_id=task_id, mode=mode, trace=raw)
    try:
        data = json.loads(fallback_raw)
        result.steps_taken = data.get("steps_taken", [])
        result.tools_called = data.get("tools_called", [])
        result.output_criteria = data.get("output_criteria", {})
        result.completed = data.get("completed", False)
        result.num_steps = len(result.steps_taken)
    except json.JSONDecodeError:
        pass
    return result
