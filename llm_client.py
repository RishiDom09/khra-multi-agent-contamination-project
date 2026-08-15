"""The single LLM boundary for the whole experiment.

Everything that touches a model goes through here, so the fair-comparison
invariant ("one model, held constant across architectures") is enforced by
construction: :func:`make_llm` is the only factory and it reads the model from
exactly one place, the active ``config.MODELS`` registry entry
(``config.MODEL_ID`` / ``config.PROVIDER``).

Two public functions:

``make_llm()``
    Return the shared model handle (a cached ``ChatGroq``, or a ``_MockLLM`` when
    running offline). Architectures call this once per question.

``call_llm(system_prompt, user_prompt, llm, is_critic=False)``
    Send one system+user pair and return a parsed dict:
      * normal agent -> ``{"answer", "confidence", "rationale", ...}``
      * critic       -> ``{"critique_of_A", "critique_of_B", ...}``

Qwen3-specific handling lives here too — reasoning-trace stripping, tolerant
answer extraction, and exponential backoff on Groq rate limits.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import Any, Optional

import config


ANSWER_FORMAT = (
    "\n\nRespond with ONLY a JSON object, no other text:\n"
    '{"answer": "<the single option letter>", "confidence": <number between 0 and 1>, '
    '"rationale": "<one or two sentences>"}'
)

CRITIC_FORMAT = (
    "\n\nRespond with ONLY a JSON object, no other text:\n"
    '{"critique_of_A": "<your challenge to proposal A>", '
    '"critique_of_B": "<your challenge to proposal B>"}'
)

ABSTAIN = ""

_LLM: Any = None  # module-level cache so we build one client per process


API_MODEL_SEEN: Optional[str] = None


def _reply_model(reply: Any) -> Optional[str]:
    """Extract the provider-reported model string from a reply, if present."""
    meta = getattr(reply, "response_metadata", None)
    if isinstance(meta, dict):
        for key in ("model_name", "model", "model_id"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    return None



class _MockLLM:
    """Offline stand-in so the whole pipeline runs with no API key.

    Deterministic (hashes the prompt) rather than random, so mock runs are
    reproducible like real ones. It deliberately "reacts to contamination": when
    the system prompt carries a false-belief line, the mock skews toward the
    misinfo-aligned option. That keeps the propagation metric exercised end to
    end in smoke tests instead of always reading zero.
    """

    model_name = "mock"

    def invoke(self, messages: list[Any]) -> Any:  
        raise NotImplementedError  


def make_llm(model: Optional[str] = None) -> Any:
    """Return the shared model handle. The ONLY place a model is constructed."""
    global _LLM
    if _LLM is not None and model is None:
        return _LLM

    if config.MOCK_MODE:
        llm: Any = _MockLLM()
    else:
        llm = _build_llm(model or config.MODEL_ID)
    if model is None:
        _LLM = llm
    return llm


def _build_llm(model_id: str) -> Any:
    """Construct the provider-specific LangChain chat model for ``model_id``.

    Every branch returns an object with the same ``.invoke(messages)`` surface
    and LangChain's standardized ``usage_metadata``, so nothing downstream
    (agents, architectures, token accounting, parsing) is provider-aware.
    Credentials come from the gitignored ``.env`` — see .env.example.
    """
    spec = config.MODEL_SPEC
    provider = config.PROVIDER
    max_tokens = spec.get("max_tokens") or config.MAX_TOKENS
    send_temperature = spec.get("send_temperature", True)

    if provider == "groq":
        # LEGACY pilot path (Qwen3). Kept runnable; not part of the new sweep.
        from langchain_groq import ChatGroq

        kwargs: dict[str, Any] = dict(
            model=model_id,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens,
            api_key=config.require_api_key(),
        )
        if config.REASONING_EFFORT is not None:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        if config.REASONING_FORMAT is not None:
            kwargs["reasoning_format"] = config.REASONING_FORMAT
        return ChatGroq(**kwargs)

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            region_name=config.require_aws_region(),
        )
        if send_temperature:
            kwargs["temperature"] = config.LLM_TEMPERATURE
        return ChatBedrockConverse(**kwargs)

    if provider == "bedrock-openai":
        from langchain_openai import ChatOpenAI

        base_url, api_key = config.require_bedrock_openai()
        kwargs = dict(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        if send_temperature:
            kwargs["temperature"] = config.LLM_TEMPERATURE
        return ChatOpenAI(**kwargs)

    if provider == "openrouter":
        # OpenRouter's unified OpenAI-compatible endpoint (Claude Opus 5,
        # GPT-5.6 Sol). One key covers every model; no per-model access grants.
        from langchain_openai import ChatOpenAI

        base_url, api_key = config.require_openrouter()
        kwargs = dict(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        if send_temperature:
            kwargs["temperature"] = config.LLM_TEMPERATURE
        return ChatOpenAI(**kwargs)

    if provider == "azure":
        endpoint, key = config.require_azure(spec)
        if "/openai/" in endpoint:
            from langchain_openai import ChatOpenAI

            kwargs = dict(
                model=model_id,
                base_url=endpoint,
                api_key=key,
                max_tokens=max_tokens,
            )
            if send_temperature:
                kwargs["temperature"] = config.LLM_TEMPERATURE
            return ChatOpenAI(**kwargs)

        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

        kwargs = dict(
            endpoint=endpoint,
            credential=key,
            model=model_id,
            max_tokens=max_tokens,
        )
        if send_temperature:
            kwargs["temperature"] = config.LLM_TEMPERATURE
        return AzureAIChatCompletionsModel(**kwargs)

    raise SystemExit(f"Unknown provider {provider!r} for model {model_id!r}.")


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def strip_reasoning(text: str) -> str:
    text = _THINK_BLOCK.sub(" ", text)
    text = _UNCLOSED_THINK.sub(" ", text)
    text = _CODE_FENCE.sub("", text)
    return text.strip()


def has_unclosed_think(text: str) -> bool:
    without_closed = _THINK_BLOCK.sub(" ", text)
    return bool(_UNCLOSED_THINK.search(without_closed))



_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"?\s*([A-H])\b', re.IGNORECASE)


def recover_from_trace(text: str, valid_letters: list[str]) -> Optional[dict[str, Any]]:
    matches = _ANSWER_FIELD.findall(text)
    letters = {ch.upper() for ch in valid_letters}
    committed = [m.upper() for m in matches if m.upper() in letters]
    if not committed:
        return None

    answer = committed[-1]
    confidence, rationale = 0.5, ""
    for obj in _iter_json_objects(text):
        if "answer" in obj and _letter_from_fragment(str(obj["answer"]), valid_letters) == answer:
            confidence = _coerce_confidence(obj.get("confidence"))
            rationale = str(obj.get("rationale", ""))
            break
    else:
        m = re.search(r'"rationale"\s*:\s*"([^"]{0,400})', text)
        if m:
            rationale = m.group(1)
        c = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if c:
            confidence = _coerce_confidence(c.group(1))

    return {
        "answer": answer,
        "confidence": confidence,
        "rationale": rationale,
        "parse_status": "recovered_from_trace",
    }


def _iter_json_objects(text: str):
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
                start = -1
            elif depth < 0:
                depth = 0
    for span in reversed(spans):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def extract_answer(text: str, valid_letters: list[str]) -> str:
    """Pull the final selected option letter out of a (reasoning-stripped) reply.

    Tries, in order: a JSON ``answer`` field, an explicit "answer/choice is X"
    phrasing, a bare letter on its own line, and finally the last standalone
    letter mentioned. Returns :data:`ABSTAIN` if nothing parses — the caller
    records an abstention rather than crashing.
    """
    letters = "".join(valid_letters)
    if not letters:
        return ABSTAIN

    # 1. JSON object with an "answer" key (the format we asked for).
    for obj in _iter_json_objects(text):
        if "answer" in obj:
            cand = _letter_from_fragment(str(obj["answer"]), valid_letters)
            if cand:
                return cand

    # 2. Explicit natural-language declaration, taking the LAST one (Qwen3
    #    restates its choice at the end after any hedging).
    declared = re.findall(
        rf"(?:answer|choice|option|select(?:ion)?)\D{{0,20}}?\b([{letters}])\b",
        text,
        re.IGNORECASE,
    )
    if declared:
        return declared[-1].upper()

    # 3. A bare letter on its own line, e.g. "B" or "B." or "**B**".
    bare = re.findall(rf"^\W{{0,4}}([{letters}])\W{{0,4}}$", text, re.MULTILINE)
    if bare:
        return bare[-1].upper()

    # 4. Last resort: the last standalone letter token anywhere.
    loose = re.findall(rf"\b([{letters}])\b", text)
    if loose:
        return loose[-1].upper()

    return ABSTAIN


def _letter_from_fragment(fragment: str, valid_letters: list[str]) -> str:
    """Coerce an ``answer`` field ("B", "b) Nelson", "Option A") to a letter."""
    frag = fragment.strip()
    if not frag:
        return ABSTAIN
    letters = "".join(valid_letters)
    m = re.match(rf"\W*([{letters}])\b", frag, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{letters}])\b", frag, re.IGNORECASE)
    return m.group(1).upper() if m else ABSTAIN


def _coerce_confidence(value: Any) -> float:
    """Clamp a model-supplied confidence into [0, 1]; default 0.5 if unusable."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    if conf > 1.0: 
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def parse_response(raw: str, valid_letters: list[str], is_critic: bool) -> dict[str, Any]:
    """Turn a raw model reply into the architecture-facing payload dict."""
    clean = strip_reasoning(raw)

    if is_critic:
        for obj in _iter_json_objects(clean):
            if "critique_of_A" in obj or "critique_of_B" in obj:
                return {
                    "critique_of_A": str(obj.get("critique_of_A", "")),
                    "critique_of_B": str(obj.get("critique_of_B", "")),
                    "parse_status": "ok",
                }
        text = clean or raw.strip()
        return {
            "critique_of_A": text,
            "critique_of_B": text,
            "parse_status": "unparsed_text",
        }

    answer = extract_answer(clean, valid_letters)
    confidence, rationale, status = 0.5, "", "ok"
    for obj in _iter_json_objects(clean):
        if "answer" in obj or "rationale" in obj or "confidence" in obj:
            confidence = _coerce_confidence(obj.get("confidence"))
            rationale = str(obj.get("rationale", ""))
            break
    else:
        status = "unparsed_json"
        rationale = clean[:400]

    if not answer:

        if has_unclosed_think(raw):
            recovered = recover_from_trace(raw, valid_letters)
            if recovered is not None:
                recovered["raw_excerpt"] = raw[-400:]
                return recovered

     
        return {
            "answer": ABSTAIN,
            "confidence": 0.0,
            "rationale": rationale or clean[:400],
            "parse_status": "abstain",
            "raw_excerpt": raw[-400:],
        }

    return {
        "answer": answer,
        "confidence": confidence,
        "rationale": rationale,
        "parse_status": status,
    }



CONTAMINATION_MARKER = "You have this extra information"


def _mock_response(
    system_prompt: str, user_prompt: str, valid_letters: list[str], is_critic: bool
) -> str:
    """Deterministic offline reply, seeded by the prompt pair."""
    if is_critic:
        return json.dumps(
            {
                "critique_of_A": "mock critique of A",
                "critique_of_B": "mock critique of B",
            }
        )

    seed = hashlib.sha256((system_prompt + user_prompt).encode()).hexdigest()
    rng = random.Random(int(seed[:16], 16))
    contaminated = CONTAMINATION_MARKER in system_prompt
    if contaminated:
        letter = valid_letters[int(seed[16], 16) % len(valid_letters)]
        if rng.random() < 0.2:  # occasional resistance
            letter = valid_letters[(valid_letters.index(letter) + 1) % len(valid_letters)]
        conf = round(rng.uniform(0.7, 0.99), 2)
    else:
        letter = valid_letters[int(seed[17], 16) % len(valid_letters)]
        conf = round(rng.uniform(0.4, 0.9), 2)

    return (
        "<think>mock reasoning trace that must be stripped</think>\n"
        + json.dumps({"answer": letter, "confidence": conf, "rationale": "mock rationale"})
    )


def _is_retryable(exc: BaseException) -> bool:
    """True for HTTP 429 / 5xx and transient connection errors."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "too many requests",
            "500",
            "502",
            "503",
            "504",
            "overloaded",
            "timeout",
            "timed out",
            "connection",
        )
    )


def _retry_delay(attempt: int, exc: BaseException) -> float:
    """Exponential backoff with jitter, honouring a server-suggested wait."""
    m = re.search(r"try again in ([\d.]+)s", str(exc), re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + config.RETRY_JITTER, config.RETRY_MAX_DELAY)
    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
    return min(delay, config.RETRY_MAX_DELAY) + random.uniform(0, config.RETRY_JITTER)


def is_rate_limit_error(result: dict[str, Any]) -> bool:
    if result.get("parse_status") != "api_error":
        return False
    err = str(result.get("error", "")).lower()
    return "rate_limit" in err or "rate limit" in err or "429" in err or "tokens per day" in err


def _token_count(reply: Any) -> int:
    usage = getattr(reply, "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("total_tokens"):
        return int(usage["total_tokens"])
    meta = getattr(reply, "response_metadata", None)
    if isinstance(meta, dict):
        tu = meta.get("token_usage") or meta.get("usage") or {}
        if tu.get("total_tokens"):
            return int(tu["total_tokens"])
    return 0


def _estimate_tokens(*texts: str) -> int:
    return sum(len(t) for t in texts) // 4


def _reply_text(reply: Any) -> str:
    """Extract the text content from a model reply (handles content blocks)."""
    raw = reply.content if hasattr(reply, "content") else str(reply)
    if isinstance(raw, list):  # some providers return a list of content blocks
        raw = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    return raw


_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def _reasoning_text(reply: Any) -> str:
    for container in (
        getattr(reply, "additional_kwargs", None),
        getattr(reply, "response_metadata", None),
    ):
        if isinstance(container, dict):
            for key in _REASONING_KEYS:
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return ""


def _was_truncated(reply: Any) -> bool:
    """True if generation stopped because it hit the token cap, not a stop token.
    """
    meta = getattr(reply, "response_metadata", None)
    if isinstance(meta, dict):
        if meta.get("finish_reason") == "length":
            return True
    info = getattr(reply, "additional_kwargs", None)  # some paths stash it here
    if isinstance(info, dict) and info.get("finish_reason") == "length":
        return True
    return False


# Instruction for the forced follow-up
_FORCE_ANSWER_INSTRUCTION = (
    "/no_think\nStop reasoning now and commit to an answer. Do NOT think further. "
    "Reply with ONLY the JSON object {{\"answer\": \"<one of {opts}>\", "
    "\"confidence\": <0-1>, \"rationale\": \"<short>\"}} — choose the single best "
    "option even if you are not fully certain."
)


def _force_answer(
    llm: Any, system: str, user_prompt: str, letters: list[str]
) -> Optional[dict[str, Any]]:
    """Re-ask with thinking suppressed and a tiny budget, forcing a committed answer.

    Returns a parsed answer dict (with its own ``tokens``), or None if the call
    itself failed. A still-unparseable reply is returned as-is so the caller keeps
    the honest abstain rather than inventing a letter.
    """
    instruction = _FORCE_ANSWER_INSTRUCTION.format(opts="/".join(letters))
    messages = [
        ("system", system),
        ("human", f"{user_prompt}\n\n{instruction}"),
    ]
    try:
        force_llm = llm.bind(max_tokens=config.FORCE_ANSWER_MAX_TOKENS)
    except Exception:  
        force_llm = llm
    try:
        reply = force_llm.invoke(messages)
    except Exception:  
        return None
    raw = _reply_text(reply)
    parsed = parse_response(raw, letters, is_critic=False)
    parsed["tokens"] = _token_count(reply)
    parsed["raw"] = raw
    return parsed


def call_llm(
    system_prompt: str,
    user_prompt: str,
    llm: Any = None,
    is_critic: bool = False,
    valid_letters: Optional[list[str]] = None,
) -> dict[str, Any]:
    letters = valid_letters or ["A", "B"]
    fmt = CRITIC_FORMAT if is_critic else ANSWER_FORMAT
    system = system_prompt + fmt

    if isinstance(llm, str) or llm is None:
        llm = make_llm(llm if isinstance(llm, str) and llm != config.MODEL_ID else None)

    if isinstance(llm, _MockLLM):
        raw = _mock_response(system, user_prompt, letters, is_critic)
        parsed = parse_response(raw, letters, is_critic)
        # Estimated so the token-budget logic is exercisable offline.
        parsed["tokens"] = _estimate_tokens(system, user_prompt, raw)
        parsed["raw"] = raw  # full-conversation capture, same as the real path
        return parsed

    messages = [("system", system), ("human", user_prompt)]
    last_exc: Optional[BaseException] = None
    for attempt in range(config.API_RETRIES):
        try:
            reply = llm.invoke(messages)
            global API_MODEL_SEEN
            if API_MODEL_SEEN is None:
                API_MODEL_SEEN = _reply_model(reply)
            raw = _reply_text(reply)
            if config.REQUEST_PAUSE:
                time.sleep(config.REQUEST_PAUSE)
            parsed = parse_response(raw, letters, is_critic)
            parsed["tokens"] = _token_count(reply)
            parsed["raw"] = raw
            trace = _reasoning_text(reply)
            if trace:
                parsed["reasoning"] = trace
            if not is_critic and parsed.get("parse_status") == "abstain":
                if trace:
                    recovered = recover_from_trace(trace, letters)
                    if recovered is not None:
                        recovered["tokens"] = parsed["tokens"]
                        recovered["raw_excerpt"] = trace[-400:]
                        recovered["raw"] = raw
                        recovered["reasoning"] = trace
                        return recovered

            # Forced-answer fallback: the reply ran out of tokens mid-reasoning
            if (
                not is_critic
                and parsed.get("parse_status") == "abstain"
                and config.FORCE_ANSWER_ON_TRUNCATION
                and _was_truncated(reply)
            ):
                forced = _force_answer(llm, system, user_prompt, letters)
                if forced is not None and forced.get("answer"):
                    forced["tokens"] = parsed["tokens"] + forced.get("tokens", 0)
                    forced["parse_status"] = "forced_answer"
                    forced["truncated_raw"] = raw
                    if trace:
                        forced["truncated_reasoning"] = trace
                    return forced
            return parsed
        except Exception as exc:  
            last_exc = exc
            if _is_retryable(exc):
                try:
                    import token_ledger

                    token_ledger.calibrate(str(exc))
                except Exception:  
                    pass
            if not _is_retryable(exc) or attempt == config.API_RETRIES - 1:
                break
            time.sleep(_retry_delay(attempt, exc))

    # Exhausted retries: record an abstention so one bad cell cannot kill a run.
    base = {"parse_status": "api_error", "error": str(last_exc)[:300], "tokens": 0}
    if is_critic:
        return {"critique_of_A": "", "critique_of_B": "", **base}
    return {"answer": ABSTAIN, "confidence": 0.0, "rationale": "", **base}
