"""LLM wrapper supporting two provider families behind one interface:

- "openai"    : any OpenAI-compatible endpoint (Groq, OpenAI, Azure OpenAI, ...)
- "anthropic" : Claude via the native Anthropic SDK

Pick the provider with LLM_PROVIDER in .env. If no API key is configured,
`generate` returns None so callers fall back to an extractive answer — the demo
runs with zero credentials on any provider.
"""
from __future__ import annotations

import re
import time
from typing import Iterator, Optional

from .config import settings

_client = None

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(m?s)", re.IGNORECASE)
_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_yes_no(text: Optional[str], default: bool = False) -> bool:
    """Robustly read a yes/no verdict out of an LLM response.

    Hosted frontier models reliably follow "reply strictly 'yes' or 'no'".
    Smaller/local models (e.g. via Ollama) often don't — they add preamble
    like "Based on the passage, yes" or "I'd say no because...". A naive
    `text.startswith("y")` check misses any reply that doesn't open with the
    literal word "yes", which is exactly what happened running this app's
    grade/verify/faithfulness checks against a local model: retrieval stayed
    healthy but grounded_rate and faithfulness both collapsed to near-zero
    because most real answers didn't parse as "yes" even when the model's
    actual judgment was yes. Searching for a whole-word "yes"/"no" anywhere in
    the first ~15 words is far more robust to real-world model phrasing.
    """
    if not text:
        return default
    # Only look at the first ~15 words — a verdict phrased far into a long
    # explanation is a different failure mode we don't try to fix here.
    snippet = " ".join(text.strip().split()[:15])
    yes_match = _YES_RE.search(snippet)
    no_match = _NO_RE.search(snippet)
    if yes_match and not no_match:
        return True
    if no_match and not yes_match:
        return False
    if yes_match and no_match:
        # Both appear (e.g. "no, that's not supported, so the answer is no")
        # — trust whichever comes first.
        return yes_match.start() < no_match.start()
    return default


def _retryable_error_types() -> tuple:
    """Collect whichever provider SDKs are installed' *transient* error
    classes: 429 rate limits, 5xx server errors (e.g. Groq returning 503
    "over capacity" — a temporary provider-side issue, unrelated to our token
    budget), and connection/timeout errors. Deliberately narrower than "any
    API error" — a 400 (bad request) or 401 (bad key) will never succeed on
    retry, so those should fail fast and visibly instead of silently retrying
    a real bug 8 times."""
    errors = []
    try:
        import openai

        errors += [
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ]
    except ImportError:
        pass
    try:
        import anthropic

        errors += [
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ]
    except ImportError:
        pass
    return tuple(errors)


def _suggested_wait_seconds(exc: Exception) -> Optional[float]:
    """Parse "...try again in 16.56s" / "...in 550ms" out of a rate-limit error
    message, if present, so we can wait exactly as long as the API says to
    instead of guessing with blind exponential backoff."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "ms" else value


def _with_retry(call, max_retries: int = 8):
    """Run `call()`, retrying on rate-limit (429) and transient server (5xx,
    connection, timeout) errors.

    Free-tier LLM APIs (e.g. Groq) enforce both per-minute and per-day token
    caps, and can also return 503 "over capacity" independent of your own
    usage. Rather than guessing a backoff delay, this parses the exact wait
    time the API tells us in a rate-limit error message (falling back to
    exponential backoff if that isn't present, which is always the case for
    5xx/connection errors) so retries succeed on the first attempt after
    waiting instead of retrying too early.
    """
    error_types = _retryable_error_types()
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return call()
        except error_types as exc:  # noqa: BLE001 - narrowed to rate-limit types
            if attempt == max_retries - 1:
                raise
            wait = _suggested_wait_seconds(exc)
            if wait is None:
                wait = delay
                delay = min(delay * 2, 30.0)
            else:
                wait += 0.5  # small safety margin past the API's own estimate
            print(f"[llm] rate limited, retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries}): {exc}")
            time.sleep(wait)


def _get_client():
    """Lazily create and cache the provider client."""
    global _client
    if _client is not None:
        return _client
    if not settings.llm_enabled:
        return None

    if settings.llm_provider == "anthropic":
        from anthropic import Anthropic

        # max_retries=0: let our own _with_retry own all rate-limit backoff.
        # Otherwise the SDK silently retries internally first, then our
        # wrapper retries again on top — waits compound instead of adding.
        _client = Anthropic(api_key=settings.llm_api_key, max_retries=0)
    else:  # openai-compatible (Groq / OpenAI / Azure / ...)
        from openai import OpenAI

        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            max_retries=0,
        )
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
        def _call():
            resp = client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=temperature,
                system=system or None,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        return _with_retry(_call)

    # openai-compatible
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def _call():
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    return _with_retry(_call)


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
