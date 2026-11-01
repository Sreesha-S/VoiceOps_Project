"""Talking to Gemini (through its OpenAI-compatible endpoint), with three beginner-friendly safety nets:

  1. RETRIES   - free tiers return "429 too many requests"; we wait and try again automatically.
  2. CACHE     - every answer is saved on disk. Re-running the same evaluation costs 0 requests,
                 which matters a lot when your free quota is small.
  3. TOOL LOOP - the model may answer "please call function X". WE run the function and send the
                 result back. The model never executes anything itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: object


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)


# ------------------------------------------------------------------ helpers
def friendly_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    code = getattr(exc, "status_code", None)
    if code in (401, 403) or "api key" in low or "permission" in low:
        return "Gemini rejected the key. Check GEMINI_API_KEY in your .env file (no quotes, no spaces)."
    if code == 404 or "not found" in low:
        return "Gemini says that model name does not exist for your key. Run `python check_setup.py` to list valid models, then set GEMINI_MODEL."
    if code == 429 or "quota" in low or "resource_exhausted" in low:
        return "Gemini free-tier limit reached. Wait a minute (or until tomorrow if it is the daily limit), or switch GEMINI_MODEL to a lighter 'flash-lite' model. Re-running is cheap: answers are cached."
    return f"{type(exc).__name__}: {text[:300]}"


def with_retries(fn, attempts: int = 6):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            low = str(exc).lower()
            code = getattr(exc, "status_code", None)
            daily = "perday" in low.replace(" ", "").replace("_", "") or "per day" in low
            retryable = code in (429, 500, 502, 503, 504) or any(t in low for t in ("timeout", "timed out", "overloaded", "connection"))
            if not retryable or daily or i == attempts - 1:
                raise
            m = re.search(r"retry in ([\d.]+)\s*s", low)
            if code == 429 and not m and i >= 2:
                raise  # no "retry in N seconds" hint = most likely the DAILY free quota: waiting longer will not help
            wait = min(float(m.group(1)) + 1 if m else 2 ** (i + 1), 60)
            print(f"  (waiting {wait:.0f}s: server busy or rate-limited, retry {i + 1}/{attempts - 1})", file=sys.stderr, flush=True)
            time.sleep(wait)


def normalize_message(msg) -> dict:
    """Keep exactly what the model returned for tool calls (Gemini 3 attaches hidden 'thought signatures'
    that must be sent back unchanged), but drop the null fields."""
    raw = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else dict(msg)
    out = {"role": "assistant", "content": raw.get("content") or ""}
    if raw.get("tool_calls"):
        out["tool_calls"] = raw["tool_calls"]
    return out


class DiskCache:
    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self.data[rec["k"]] = rec["v"]
                except Exception:  # noqa: BLE001
                    continue

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ the client
class LLMClient:
    min_interval = 0.0   # seconds to keep between real API calls (cached answers are free and skip this)
    _last = 0.0

    def __init__(self, settings):
        if not settings.llm_api_key:
            raise LLMError("GEMINI_API_KEY is empty. Copy .env.example to .env and paste your key from aistudio.google.com/apikey.")
        from openai import OpenAI  # the OpenAI library is only the 'phone line'; the brain is Gemini

        self.model, self.effort = settings.llm_model, settings.llm_effort
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=90, max_retries=0)
        self.cache = DiskCache(settings.cache_dir / "llm_cache.jsonl") if settings.llm_cache else None
        # Free tier: ~5 requests/minute for normal Flash models, ~10 for Flash-Lite. Pace ourselves instead of hitting 429s.
        default_gap = "6.5" if "lite" in self.model.lower() else "12.5"
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", default_gap) or 0)

    def _complete(self, messages, tools) -> dict:
        payload = {"m": self.model, "msgs": messages, "tools": tools, "e": self.effort}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        hit = self.cache.get(key) if self.cache is not None else None
        if hit is not None and (hit.get("tool_calls") or str(hit.get("content", "")).strip()):
            return hit  # (an old cached EMPTY answer is ignored, so it gets asked again)
        out = {"role": "assistant", "content": ""}
        for attempt in range(3):  # Gemini sometimes returns an empty message: ask again, with a little variation
            gap = self._last + self.min_interval - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()
            kwargs = dict(model=self.model, messages=messages, temperature=0 if attempt == 0 else 0.4)
            if tools:
                kwargs["tools"] = tools
            if self.effort:
                kwargs["reasoning_effort"] = self.effort
            try:
                resp = with_retries(lambda: self.client.chat.completions.create(**kwargs))
            except Exception as exc:  # noqa: BLE001
                if "reasoning" in str(exc).lower() or "thinking" in str(exc).lower():
                    kwargs.pop("reasoning_effort", None)  # this model has no thinking control: just don't send it
                    self.effort = ""
                    resp = with_retries(lambda: self.client.chat.completions.create(**kwargs))
                else:
                    raise LLMError(friendly_error(exc)) from exc
            out = normalize_message(resp.choices[0].message)
            if out.get("tool_calls") or out["content"].strip():
                break
            print(f"  (Gemini sent an empty answer, finish_reason={getattr(resp.choices[0], 'finish_reason', None)}; asking again {attempt + 1}/2)",
                  file=sys.stderr, flush=True)
        if self.cache is not None and (out.get("tool_calls") or out["content"].strip()):
            self.cache.put(key, out)  # never remember an empty answer
        return out

    def chat(self, messages, *, tools=None, tool_funcs=None, max_rounds: int = 6) -> LLMResponse:
        msgs, records = list(messages), []
        for _ in range(max_rounds):
            try:
                out = self._complete(msgs, tools)
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise LLMError(friendly_error(exc)) from exc
            calls = out.get("tool_calls") or []
            if not calls:
                return LLMResponse(out["content"].strip(), records)
            msgs.append(out)  # the model's own message, unchanged
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = (tool_funcs or {})[name](**args)
                except Exception as exc:  # noqa: BLE001 - tell the model what went wrong instead of crashing
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                records.append(ToolCallRecord(name, args, result))
                msgs.append({"role": "tool", "tool_call_id": c["id"], "name": name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
        final = self._complete(msgs, None)  # too many tool rounds: force a plain answer
        return LLMResponse(final["content"].strip(), records)


class MockLLM:
    """--dry-run only: lets you test the plumbing without any key. Its answers are NOT real."""

    model = "mock"

    def chat(self, messages, *, tools=None, tool_funcs=None, max_rounds: int = 6) -> LLMResponse:
        last = messages[-1]["content"]
        if str(messages[0]["content"]).startswith("You turn a customer message"):
            return LLMResponse(last.splitlines()[-1])
        return LLMResponse("[MOCK ANSWER - not a real model] " + last.split("CONTEXT", 1)[-1][:200].replace("\n", " "))


def make_llm(settings, dry_run: bool = False):
    return MockLLM() if dry_run else LLMClient(settings)
