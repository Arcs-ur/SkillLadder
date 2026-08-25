"""OpenAI-compatible LLM access used by extraction and evolution."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from openai import OpenAI


def get_llm_client() -> OpenAI:
    """Build a client from environment variables without provider-specific auth."""
    api_key = os.getenv("SKILLLADDER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set SKILLLADDER_API_KEY or OPENAI_API_KEY before making LLM calls."
        )

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(os.getenv("SKILLLADDER_TIMEOUT", "120")),
        "max_retries": int(os.getenv("SKILLLADDER_MAX_RETRIES", "5")),
    }
    base_url = os.getenv("SKILLLADDER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def llm_call(
    prompt: str | Sequence[dict[str, Any]],
    system_prompt: str = "",
    temperature: float | None = None,
) -> str:
    """Make one chat-completion call and return its text response."""
    model = os.getenv("SKILLLADDER_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError(
            "Set SKILLLADDER_MODEL (or OPENAI_MODEL) to an available model name."
        )

    if isinstance(prompt, str):
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
    else:
        messages = list(prompt)
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = get_llm_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
