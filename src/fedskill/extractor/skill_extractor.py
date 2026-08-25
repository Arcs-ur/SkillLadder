"""Skill extractor — turns task traces into structured Skills via LLM.

Supports dual-mode extraction inspired by Trace2Skill's error/success analyst:
- Success analyst: extracts effective strategies from successful traces
- Error analyst: extracts defensive patterns from failed traces
"""

from __future__ import annotations

import json
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


EXTRACT_SYSTEM = """You are a skill-extraction engine for AI agents.
Given a task description and one or more execution traces in ReAct format
(Thought → Action → Observation loops with tool calls), identify reusable *skills* —
transferable strategies, tool-use patterns, heuristics, or multi-step procedures
that could help an agent solve similar tasks.

Pay special attention to:
- Tool selection patterns (when to use code_execute vs file_read vs api_call vs web_search)
- Error handling and recovery strategies
- Data validation and quality-check heuristics
- Multi-step reasoning chains that lead to correct tool usage
- Executable code patterns that can be generalized into reusable scripts

For each skill you identify, return a JSON array of objects with these keys:
- name (str): short descriptive name
- description (str): what the skill does
- when_to_use (str): conditions / cues for when to apply this skill
- procedure (str): step-by-step instructions including which tools to call
- examples (list[str]): 1-2 brief examples showing tool-call patterns
- scripts (list[object]): 0-3 reusable executable scripts extracted from the traces.
  Each script object has keys:
    - name (str): filename, e.g. "validate_path.py"
    - language (str): "python" or "bash"
    - code (str): generalized, parameterized code (replace hardcoded paths/values with
      function parameters; include imports and a brief docstring)
    - description (str): one-line explanation of what it does

  If no executable pattern is present in the traces for a skill, set scripts to [].
  Scripts should be GENERALIZED — not copy-pasted from traces. Replace specific file
  paths, URLs, API keys with parameters. Make them self-contained and reusable.

Return ONLY a JSON array, no extra text."""


# --------------------------------------------------------------------------
# Dual-mode prompts: error analyst vs success analyst
# --------------------------------------------------------------------------

ERROR_ANALYST_SYSTEM = """You are an ERROR ANALYST for AI agents.
Given a task description and execution traces from a FAILED agent session,
your job is to identify what went wrong and extract DEFENSIVE skills —
patterns that would help an agent AVOID similar failures in the future.

Focus on:
- What the agent tried that didn't work, and WHY it failed
- Common pitfalls, wrong tool choices, missing validations
- Error recovery strategies that could have helped
- Pre-conditions that should be checked BEFORE attempting an action
- Fallback strategies when the primary approach fails

For each skill, return a JSON array of objects with keys:
- name (str): short descriptive name (e.g. "validate-before-write", "search-fallback")
- description (str): what failure this skill prevents
- when_to_use (str): warning signs / conditions that indicate this risk
- procedure (str): step-by-step defensive procedure
- examples (list[str]): 1-2 examples of the failure and how the skill prevents it
- scripts (list[object]): reusable scripts (same format as above), or []

Return ONLY a JSON array, no extra text."""

SUCCESS_ANALYST_SYSTEM = """You are a SUCCESS ANALYST for AI agents.
Given a task description and execution traces from a SUCCESSFUL agent session,
your job is to identify the KEY STRATEGIES that made the agent succeed and
extract them as reusable skills.

Focus on:
- Which tool-use patterns were effective and why
- Multi-step strategies that led to correct results
- Clever workarounds or non-obvious approaches
- Efficient information gathering and validation patterns
- Executable code patterns that can be generalized

For each skill, return a JSON array of objects with keys:
- name (str): short descriptive name (e.g. "incremental-search", "verify-then-commit")
- description (str): what effective strategy this skill captures
- when_to_use (str): conditions where this strategy is applicable
- procedure (str): step-by-step procedure that led to success
- examples (list[str]): 1-2 examples showing the strategy in action
- scripts (list[object]): reusable scripts (same format as above), or []

Return ONLY a JSON array, no extra text."""


EXTRACT_USER = """## Task: {task_name}
### Task Description
{task_description}

### Execution Traces
{traces}

Please extract reusable skills from the above traces."""


def extract_skills(
    task_name: str,
    task_description: str,
    traces: list[str],
    outcome: str = "unknown",
) -> list[Skill]:
    """Extract skills from execution traces of a single task/client.

    Uses dual-mode extraction when outcome is known:
    - outcome="success" or score > 0.5: success analyst (effective strategies)
    - outcome="failure" or score <= 0.5: error analyst (defensive patterns)
    - outcome="unknown": generic extraction (backward compatible)

    Args:
        task_name: Name of the task (e.g. "Task-A").
        task_description: Natural-language description of the task.
        traces: List of execution trace strings.
        outcome: "success", "failure", or "unknown".

    Returns:
        A list of Skill objects extracted from the traces.
    """
    traces_text = "\n---\n".join(traces)
    prompt = EXTRACT_USER.format(
        task_name=task_name,
        task_description=task_description,
        traces=traces_text,
    )

    # Select analyst mode based on outcome
    if outcome == "success":
        system_prompt = SUCCESS_ANALYST_SYSTEM
    elif outcome == "failure":
        system_prompt = ERROR_ANALYST_SYSTEM
    else:
        system_prompt = EXTRACT_SYSTEM

    raw = llm_call(prompt, system_prompt=system_prompt, temperature=0.4)

    # Parse the JSON array from LLM output
    # Handle potential markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse LLM output for {task_name}, returning empty skill list.")
        return []

    skills: list[Skill] = []
    for item in items:
        skill = Skill(
            name=item.get("name", "Unnamed"),
            description=item.get("description", ""),
            when_to_use=item.get("when_to_use", ""),
            procedure=item.get("procedure", ""),
            examples=item.get("examples", []),
            scripts=item.get("scripts", []),
            source_task=task_name,
            level=0,
        )
        skills.append(skill)

    return skills
