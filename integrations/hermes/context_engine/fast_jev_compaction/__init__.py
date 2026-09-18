"""OpenRouter-backed Jev context engine for Hermes Agent.

Install this directory as ``~/.hermes/plugins/fast-jev-compaction``, set
``context.engine: fast_jev_compaction``, and provide ``OPENROUTER_API_KEY``.
The engine owns only the compaction pass and returns the original messages
unchanged whenever Jev, OpenRouter, or transcript normalization is not
trustworthy.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from agent.context_engine import ContextEngine


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "~typesafe/jev-latest"
DEFAULT_PROTECTED_TOOLS = frozenset({"edit", "write", "notebookedit", "apply_patch"})
STATE_CHAR_BUDGET = 84_000
RESULT_PREVIEW_CHARS = 160
TRUNCATE_HEAD_CHARS = 300


def _openrouter_api_key() -> str:
    """Read the active Hermes profile's key without crossing profile boundaries."""
    try:
        from agent.secret_scope import get_secret_str
    except ImportError:
        return os.getenv("OPENROUTER_API_KEY", "").strip()
    return get_secret_str("OPENROUTER_API_KEY", "").strip()


def _as_text(value: Any) -> str:
    """Return OpenAI message content as plain text without assuming one provider shape."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return "" if value is None else str(value)


def _abridge(text: str, limit: int) -> str:
    if len(text) <= limit * 2 + 40:
        return text
    return f"{text[:limit]}\n[… {len(text) - (2 * limit)} chars omitted …]\n{text[-limit:]}"


def _truncate_result(text: str) -> str:
    if len(text) <= TRUNCATE_HEAD_CHARS + 120:
        return text
    return (
        f"{text[:TRUNCATE_HEAD_CHARS]}\n"
        f"[fast-jev-compaction truncated {len(text) - TRUNCATE_HEAD_CHARS} chars of this tool result; "
        "re-run the tool if needed]"
    )


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "[unserializable arguments]"


@dataclass(frozen=True)
class _Pair:
    call_id: str
    name: str
    call_index: int
    result_index: int
    arguments: str
    result_text: str
    protected: bool


class FastJevCompactionEngine(ContextEngine):
    """A fail-open Hermes ``ContextEngine`` driven by ``~typesafe/jev-latest``."""

    threshold_percent = 0.75
    protect_first_n = 3
    protect_last_n = 6
    emit_automatic_compaction_status = False

    def __init__(self, context_length: int = 128_000) -> None:
        self.context_length = int(os.getenv("FAST_JEV_CONTEXT_LENGTH", context_length))
        self.threshold_percent = float(os.getenv("FAST_JEV_CONTEXT_THRESHOLD", self.threshold_percent))
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self._lock = threading.Lock()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "FastJevCompactionEngine":
        """Create an agent-local engine without copying its non-copyable lock.

        Hermes clones plugin engines when it creates child agents. Each clone
        must retain the observed usage and limits, but needs an independent
        lock so concurrent agent contexts cannot block one another.
        """
        clone = type(self)(self.context_length)
        memo[id(self)] = clone
        clone.threshold_percent = self.threshold_percent
        clone.threshold_tokens = self.threshold_tokens
        clone.last_prompt_tokens = self.last_prompt_tokens
        clone.last_completion_tokens = self.last_completion_tokens
        clone.last_total_tokens = self.last_total_tokens
        clone.compression_count = self.compression_count
        clone.protect_first_n = self.protect_first_n
        clone.protect_last_n = self.protect_last_n
        return clone

    @property
    def name(self) -> str:
        return "fast_jev_compaction"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.last_completion_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        self.last_total_tokens = int(
            usage.get("total_tokens") or self.last_prompt_tokens + self.last_completion_tokens
        )

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = self.last_prompt_tokens if prompt_tokens is None else prompt_tokens
        return bool(tokens and tokens >= self.threshold_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        return sum(len(_as_text(message.get("content"))) for message in messages) // 4 >= self.threshold_tokens

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Return a compacted valid OpenAI message list, or the exact input on any uncertainty."""
        original = copy.deepcopy(messages)
        if not _openrouter_api_key():
            return original
        if not force and not self.should_compress(current_tokens):
            return original
        if not self._lock.acquire(blocking=False):
            return original
        try:
            pairs = self._pairs(original)
            candidates = [pair for pair in pairs if self._eligible(pair, len(original))]
            if not candidates:
                return original
            state, visible = self._state(original, pairs, focus_topic, memory_context)
            candidates = [pair for pair in candidates if pair.call_id in visible]
            if not candidates:
                return original
            answers: Dict[str, float] = {}
            for batch in self._batches(candidates):
                answers.update(self._ask(state, batch))
            decisions = self._decisions(candidates, answers)
            if decisions and all(action == "drop_call" for action in decisions.values()):
                return original
            compacted = self._apply(original, pairs, decisions)
            if not self._valid(compacted):
                return original
            self.compression_count += 1
            # Hermes uses -1 as the "awaiting fresh usage" sentinel after a compaction.
            self.last_prompt_tokens = -1
            return compacted
        # A malformed provider response must never make Hermes lose context.
        # Compaction is optional, so ordinary runtime failures fail open.
        except Exception:
            return original
        finally:
            self._lock.release()

    def _pairs(self, messages: List[Dict[str, Any]]) -> List[_Pair]:
        calls: Dict[str, tuple[int, str, str]] = {}
        results: Dict[str, tuple[int, str, bool]] = {}
        for index, message in enumerate(messages):
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                call_id = call["id"]
                if call_id in calls or call_id in results:
                    raise ValueError("ambiguous tool call id")
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(function.get("name") or call.get("name") or "unknown")
                arguments = function.get("arguments", call.get("arguments", {}))
                calls[call_id] = (index, name, _json(arguments))
            if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
                call_id = message["tool_call_id"]
                if call_id in results:
                    raise ValueError("duplicate tool result id")
                results[call_id] = (
                    index,
                    _as_text(message.get("content")),
                    bool(message.get("is_error") or message.get("error")),
                )
        pairs: List[_Pair] = []
        for call_id, (call_index, name, arguments) in calls.items():
            result = results.get(call_id)
            if result is None:
                continue
            result_index, result_text, is_error = result
            if result_index < call_index:
                raise ValueError("result precedes tool call")
            pairs.append(
                _Pair(
                    call_id=call_id,
                    name=name,
                    call_index=call_index,
                    result_index=result_index,
                    arguments=arguments,
                    result_text=result_text,
                    protected=is_error or name.lower() in DEFAULT_PROTECTED_TOOLS,
                )
            )
        return pairs

    def _eligible(self, pair: _Pair, total: int) -> bool:
        return (
            not pair.protected
            and pair.call_index >= self.protect_first_n
            and pair.result_index < total - self.protect_last_n
        )

    def _state(
        self,
        messages: List[Dict[str, Any]],
        pairs: Iterable[_Pair],
        focus_topic: Optional[str],
        memory_context: str,
    ) -> tuple[Dict[str, Any], set[str]]:
        by_call_index: Dict[int, List[_Pair]] = {}
        for pair in pairs:
            by_call_index.setdefault(pair.call_index, []).append(pair)
        history: List[Dict[str, Any]] = []
        visible: set[str] = set()
        for index, message in enumerate(messages):
            entry: Dict[str, Any] = {
                "i": index,
                "role": str(message.get("role") or "user"),
                "text": _abridge(_as_text(message.get("content")), 500),
            }
            calls = []
            for pair in by_call_index.get(index, []):
                calls.append(
                    {
                        "id": pair.call_id,
                        "tool": pair.name,
                        "input": _abridge(pair.arguments, 500),
                        "result": f"{len(pair.result_text)} chars; preview={_abridge(pair.result_text, RESULT_PREVIEW_CHARS)}",
                    }
                )
                visible.add(pair.call_id)
            if calls:
                entry["tool_calls"] = calls
            if entry["text"] or calls:
                history.append(entry)
        state: Dict[str, Any] = {
            "context": "A coding agent is selecting tool evidence that remains useful for its current task.",
            "goal": focus_topic or "Continue the current task correctly.",
            "history": history,
        }
        while len(_json(state)) > STATE_CHAR_BUDGET and history:
            removed = history.pop(0)
            for call in removed.get("tool_calls", []):
                visible.discard(call.get("id", ""))
        return state, visible

    @staticmethod
    def _batches(candidates: List[_Pair]) -> Iterable[List[_Pair]]:
        for start in range(0, len(candidates), 20):
            yield candidates[start : start + 20]

    def _ask(self, state: Dict[str, Any], batch: List[_Pair]) -> Dict[str, float]:
        api_key = _openrouter_api_key()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")
        questions: Dict[str, Dict[str, str]] = {}
        for pair in batch:
            questions[f"call_{pair.call_id}"] = {
                "type": "noul",
                "instructions": f"Tool call {pair.call_id} ({pair.name}) carries information the task still depends on.",
            }
            questions[f"result_{pair.call_id}"] = {
                "type": "noul",
                "instructions": f"Tool result {pair.call_id} ({pair.name}) contains information needed to continue correctly.",
            }
        answer_properties = {
            name: {
                "type": "object",
                "properties": {"type": {"const": "noul"}, "noul": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["type", "noul"],
                "additionalProperties": False,
            }
            for name in questions
        }
        body = {
            "model": os.getenv("FAST_JEV_MODEL", DEFAULT_MODEL),
            "messages": [
                {
                    "role": "system",
                    "content": "You are Jev, a structured relevance classifier. Return only the strict JSON requested.",
                },
                {"role": "user", "content": _json({"state": state, "questions": questions})},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "jev_decisions",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answers": {
                                "type": "object",
                                "properties": answer_properties,
                                "required": list(questions),
                                "additionalProperties": False,
                            }
                        },
                        "required": ["answers"],
                        "additionalProperties": False,
                    },
                },
            },
            "temperature": 0,
            "max_tokens": max(128, len(questions) * 12),
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=_json(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = float(os.getenv("FAST_JEV_TIMEOUT_SECONDS", "15"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        content = envelope["choices"][0]["message"]["content"]
        parsed = json.loads(_as_text(content))
        answers = parsed.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("answers missing")
        scores: Dict[str, float] = {}
        for name in questions:
            answer = answers.get(name)
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise ValueError("invalid answer")
            score = answer.get("noul")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
                raise ValueError("invalid probability")
            scores[name] = float(score)
        return scores

    @staticmethod
    def _decisions(candidates: List[_Pair], answers: Dict[str, float]) -> Dict[str, str]:
        decisions: Dict[str, str] = {}
        for pair in candidates:
            result_score = answers[f"result_{pair.call_id}"]
            call_score = answers[f"call_{pair.call_id}"]
            if result_score >= 0.25:
                decisions[pair.call_id] = "keep"
            elif call_score >= 0.5:
                decisions[pair.call_id] = "drop_result"
            else:
                decisions[pair.call_id] = "drop_call"
        return decisions

    @staticmethod
    def _apply(messages: List[Dict[str, Any]], pairs: List[_Pair], decisions: Dict[str, str]) -> List[Dict[str, Any]]:
        by_result = {pair.result_index: pair for pair in pairs}
        compacted: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            pair = by_result.get(index)
            if pair is not None:
                action = decisions.get(pair.call_id, "keep")
                if action == "drop_call":
                    continue
                if action == "drop_result":
                    rebuilt = copy.deepcopy(message)
                    rebuilt["content"] = _truncate_result(_as_text(message.get("content")))
                    compacted.append(rebuilt)
                    continue
            if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
                removed = {pair.call_id for pair in pairs if pair.call_index == index and decisions.get(pair.call_id) == "drop_call"}
                if removed:
                    rebuilt = copy.deepcopy(message)
                    rebuilt["tool_calls"] = [
                        tool for tool in rebuilt["tool_calls"] if not isinstance(tool, dict) or tool.get("id") not in removed
                    ]
                    if not rebuilt["tool_calls"] and not _as_text(rebuilt.get("content")).strip():
                        continue
                    compacted.append(rebuilt)
                    continue
            compacted.append(copy.deepcopy(message))
        return compacted

    @staticmethod
    def _valid(messages: List[Dict[str, Any]]) -> bool:
        call_ids = {
            call.get("id")
            for message in messages
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        }
        return all(
            message.get("role") != "tool" or message.get("tool_call_id") in call_ids
            for message in messages
        )


def register(ctx: Any) -> None:
    """Register the one Hermes context engine exposed by this plugin."""
    ctx.register_context_engine(FastJevCompactionEngine())


__all__ = ["FastJevCompactionEngine", "register"]
