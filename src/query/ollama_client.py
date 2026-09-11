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
structurally valid JSON, semantically useless content. The fix that
worked reliably for short outputs: an explicit "output the answer
directly, do not reason out loud" instruction in the prompt, plus
chat_text()'s automatic rambling-detection-and-retry.

For LONGER, more involved generation (Phase 11's multi-vehicle
explanations), even that retry sometimes still rambles both times. The
one approach confirmed to reliably separate thinking from the actual
answer regardless of task complexity: OMIT the `think` parameter from the
Ollama call entirely (not think=False -- actually omitted) and read only
`message.content`, never `message.thinking`. Ollama then properly
classifies reasoning into the `thinking` field and keeps `content` clean
-- confirmed empirically -- but the model still spends real time
thinking, so this is slower (tens of seconds) and only worth it as a
fallback when the fast schema-based path fails. See chat_long_form().
"""

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Union

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
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def chat(
    messages: List[Dict[str, str]],
    model: str = config.OLLAMA_MODEL,
    format: Optional[Union[str, Dict[str, Any]]] = None,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
    think: Optional[bool] = False,
) -> str:
    """For structured (JSON schema) output. For free-form text generation,
    prefer chat_text() (fast) or chat_long_form() (slower, robust
    fallback) -- an unconstrained format=None call with think=False leaves
    thinking mode effectively unsuppressed (see module docstring).

    think=None omits the parameter entirely rather than sending False --
    see module docstring for why that distinction matters.
    """
    try:
        client = ollama.Client(host=config.OLLAMA_HOST, timeout=timeout)
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            format=format,
            options={"temperature": temperature},
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        if think is not None:
            kwargs["think"] = think
        response = client.chat(**kwargs)
        return response["message"]["content"]
    except Exception as e:
        raise OllamaError(f"Ollama call failed ({type(e).__name__}): {e}") from e


def is_reachable(timeout: float = 5.0) -> bool:
    """Cheap reachability check for UI status banners (Phase 13) -- lists
    locally available models rather than invoking generation at all, so
    it stays fast regardless of how loaded the model currently is."""
    try:
        ollama.Client(host=config.OLLAMA_HOST, timeout=timeout).list()
        return True
    except Exception:
        return False


def chat_text(
    messages: List[Dict[str, str]],
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
) -> str:
    """Free-form text generation (query rewriting, HyDE descriptions,
    etc.), routed through a trivial {"result": "..."} schema for the
    thinking-suppression speedup described in the module docstring.
    Detects rambling-shaped output (see looks_like_rambling) and retries
    once with a firmer instruction; falls back to the raw response if the
    model ever ignores the schema entirely."""
    result = _chat_text_once(messages, model, temperature, timeout)
    if looks_like_rambling(result):
        firmer_messages = messages + [{"role": "user", "content": _NO_RAMBLING_INSTRUCTION}]
        result = _chat_text_once(firmer_messages, model, temperature, timeout)
    return result


def chat_long_form(
    messages: List[Dict[str, str]],
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: float = config.GENERATION_FALLBACK_TIMEOUT_SECONDS,
) -> str:
    """Robust (but slow) fallback for longer, more involved generation
    where chat_text()'s schema trick still rambles: lets the model think
    freely (think param omitted) and returns only message.content, which
    Ollama keeps clean of the reasoning trace regardless of task
    complexity. Strips any stray <think> tags defensively in case a given
    model/template ever leaks them into content anyway."""
    raw = chat(messages=messages, model=model, format=None, temperature=temperature, timeout=timeout, think=None)
    return _THINK_TAG_RE.sub("", raw).strip()


def _chat_text_once(messages, model, temperature, timeout) -> str:
    raw = chat(messages=messages, model=model, format=_TEXT_SCHEMA, temperature=temperature, timeout=timeout)
    try:
        return json.loads(raw)["result"].strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw.strip()


def chat_text_stream(
    messages: List[Dict[str, str]],
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: float = config.GENERATION_FALLBACK_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Streaming counterpart to chat_text(), for UI surfaces that want
    first-token latency instead of waiting on a whole answer.

    Non-obvious detail: this still uses the {"result": "..."} schema (same
    thinking-suppression reason as chat_text -- an unconstrained stream
    would stream the model's reasoning trace at the user). That means the
    raw stream is JSON, not prose, so this incrementally decodes the
    `result` string value and yields only the prose delta. Callers get
    clean text; the schema trick stays invisible.
    """
    try:
        client = ollama.Client(host=config.OLLAMA_HOST, timeout=timeout)
        stream = client.chat(
            model=model,
            messages=messages,
            format=_TEXT_SCHEMA,
            options={"temperature": temperature},
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            think=False,
            stream=True,
        )
        buffer = ""
        emitted = 0
        for part in stream:
            buffer += part["message"]["content"]
            decoded = _partial_json_string(buffer, "result")
            if decoded is not None and len(decoded) > emitted:
                yield decoded[emitted:]
                emitted = len(decoded)
    except Exception as e:
        raise OllamaError(f"Ollama stream failed ({type(e).__name__}): {e}") from e


_JSON_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _partial_json_string(buffer: str, key: str) -> Optional[str]:
    """Decode as much of the `"<key>": "..."` value as has streamed in.

    Returns None until the opening quote arrives. Escape sequences are
    decoded, and a trailing half-streamed escape (a lone backslash at the
    buffer edge) stops cleanly rather than emitting a stray character --
    the next chunk re-decodes it whole.
    """
    marker = f'"{key}"'
    start = buffer.find(marker)
    if start == -1:
        return None
    quote = buffer.find('"', start + len(marker) + 1)
    if quote == -1:
        return None

    out = []
    i = quote + 1
    while i < len(buffer):
        ch = buffer[i]
        if ch == "\\":
            if i + 1 >= len(buffer):
                break  # escape sequence still streaming -- stop cleanly
            out.append(_JSON_UNESCAPE.get(buffer[i + 1], buffer[i + 1]))
            i += 2
            continue
        if ch == '"':
            break  # closing quote -- string complete
        out.append(ch)
        i += 1
    return "".join(out)


def looks_like_rambling(text: str) -> bool:
    if len(text) > _RAMBLING_MAX_LEN:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _RAMBLING_MARKERS)
