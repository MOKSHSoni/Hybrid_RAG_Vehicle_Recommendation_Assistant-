"""Thin wrapper around the local Ollama server.

Any connection failure or timeout raises OllamaError -- callers must
treat that differently from an invalid-JSON response: a dead endpoint
should fall back immediately, never be retried (see llm_extraction.py).

Performance note (benchmarked during Phase 3-5 development on the target
CPU-only laptop): qwen3:4b is a "thinking" model that, left unconstrained,
spends ~15s generating an internal reasoning preamble even for a trivial
one-word reply -- unacceptable latency for an interactive assistant on
this hardware. Neither the `think=False` API flag alone, nor a
"/no_think" prompt suffix, reliably suppressed this. Combining
`think=False` WITH a `format` JSON schema DOES force immediate structural
compliance (grammar-constrained decoding can't emit an open-ended
preamble before the required JSON) -- but for open-ended/creative prompts
(e.g. HyDE's "write a description"), the model was observed to instead
cram its entire reasoning monologue INSIDE the schema's string field
itself (e.g. {"result": "Okay, the user wants a hypothetical..."}) --
structurally valid JSON, semantically useless content, ~470 tokens and
15s+ instead of the ~2s/50-token answer actually wanted. The fix that
worked reliably: an explicit "output the answer directly, do not reason
out loud" instruction in the prompt. chat_text() below defends in depth
-- callers should still phrase prompts this way, and chat_text() detects
rambling-shaped output and retries once with a firmer instruction before
giving up and returning whatever it got (never crashes, never blocks
forever).
"""

import json
from typing import Any, Dict, List, Optional, Union

import ollama

import config


class OllamaError(Exception):
    """The Ollama server call itself failed (unreachable, timed out, model
    not pulled, etc.) -- distinct from the model returning bad content."""


_TEXT_SCHEMA = {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}

_RAMBLING_MARKERS = (
    "the user wants",
    "let me think",
    "let me draft",
    "let me check",
    "let me try",
    "i need to",
    "i should",
    "wait, is",
    "wait, that",
)
_RAMBLING_MAX_LEN = 500
_NO_RAMBLING_INSTRUCTION = (
    "IMPORTANT: output ONLY the final answer text, nothing else. Do not reason, plan, or "
    "think out loud -- do not write phrases like 'let me think' or 'the user wants'. "
    "You already know the answer; state it directly."
)


def chat(
    messages: List[Dict[str, str]],
    model: str = config.OLLAMA_MODEL,
    format: Optional[Union[str, Dict[str, Any]]] = None,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
    think: bool = False,
) -> str:
    """For structured (JSON schema) output. For free-form text generation,
    prefer chat_text() -- an unconstrained `format=None` call leaves
    thinking mode effectively unsuppressed (see module docstring)."""
    try:
        client = ollama.Client(host=config.OLLAMA_HOST, timeout=timeout)
        response = client.chat(
            model=model,
            messages=messages,
            format=format,
            think=think,
            options={"temperature": temperature},
        )
        return response["message"]["content"]
    except Exception as e:
        raise OllamaError(f"Ollama call failed ({type(e).__name__}): {e}") from e


def chat_text(
    messages: List[Dict[str, str]],
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
) -> str:
    """Free-form text generation (query rewriting, HyDE descriptions,
    etc.), routed through a trivial {"result": "..."} schema for the
    thinking-suppression speedup described in the module docstring.
    Detects rambling-shaped output (see _looks_like_rambling) and retries
    once with a firmer instruction; falls back to the raw response if the
    model ever ignores the schema entirely."""
    result = _chat_text_once(messages, model, temperature, timeout)
    if _looks_like_rambling(result):
        firmer_messages = messages + [{"role": "user", "content": _NO_RAMBLING_INSTRUCTION}]
        result = _chat_text_once(firmer_messages, model, temperature, timeout)
    return result


def _chat_text_once(messages, model, temperature, timeout) -> str:
    raw = chat(messages=messages, model=model, format=_TEXT_SCHEMA, temperature=temperature, timeout=timeout)
    try:
        return json.loads(raw)["result"].strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw.strip()


def _looks_like_rambling(text: str) -> bool:
    if len(text) > _RAMBLING_MAX_LEN:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _RAMBLING_MARKERS)
