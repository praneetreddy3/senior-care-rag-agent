"""LLM wrapper supporting two provider families behind one interface:

- "openai"    : any OpenAI-compatible endpoint (Groq, OpenAI, Azure OpenAI, ...)
- "anthropic" : Claude via the native Anthropic SDK

Pick the provider with LLM_PROVIDER in .env. If no API key is configured,
`generate` returns None so callers fall back to an extractive answer — the demo
runs with zero credentials on any provider.
"""
from __future__ import annotations

from typing import Iterator, Optional

from .config import settings

_client = None


def _get_client():
    """Lazily create and cache the provider client."""
    global _client
    if _client is not None:
        return _client
    if not settings.llm_enabled:
        return None

    if settings.llm_provider == "anthropic":
        from anthropic import Anthropic

        _client = Anthropic(api_key=settings.llm_api_key)
    else:  # openai-compatible (Groq / OpenAI / Azure / ...)
        from openai import OpenAI

        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def generate(prompt: str, system: str = "", temperature: float = 0.1) -> Optional[str]:
    """Return the model's completion, or None if no LLM is configured.

    The signature is identical across providers, so callers (the agent, the eval)
    never need to know which provider is active.
    """
    client = _get_client()
    if client is None:
        return None

    if settings.llm_provider == "anthropic":
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=temperature,
            system=system or None,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    # openai-compatible
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def generate_stream(
    prompt: str, system: str = "", temperature: float = 0.1
) -> Iterator[str]:
    """Yield the model's completion token-by-token (for streaming to the UI).

    Yields nothing if no LLM is configured — callers handle that fallback.
    """
    client = _get_client()
    if client is None:
        return

    if settings.llm_provider == "anthropic":
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=temperature,
            system=system or None,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
        return

    # openai-compatible
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
