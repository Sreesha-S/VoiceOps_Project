"""Scoring helpers that mirror how the dataset describes grading.

  must_include      facts separated by ' ; '  -> EVERY fact must appear.
                    one fact = alternatives separated by ' / '  -> ANY one is enough.
  must_not_include  phrases separated by ' ; ' -> NONE may appear.
  gold_evidence     section IDs that support the answer -> retrieval recall.
  expected_tool_calls  e.g. "lookup_order(order_id=123, verification_value=4533)" -> must have been called.
"""
from __future__ import annotations

import re
from datetime import date

_ORD = re.compile(r"(\d+)(?:st|nd|rd|th)(\s+of)?\b")


_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
          "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "thirty": "30"}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b")


def norm(text, number_words: bool = False) -> str:
    t = str(text).lower().replace("\u2019", "'").replace("\u2018", "'").replace(",", "")
    t = _ORD.sub(r"\1", t).replace("₹", "").replace("rs.", "").replace("rs ", "")
    if number_words:
        t = _WORD_RE.sub(lambda m: _WORDS[m.group(1)], t)  # "three attempts" counts as "3"
    return re.sub(r"\s+", " ", t)


def _variants(alt: str) -> list[str]:
    """'2026-10-27' also matches '27 October 2026', 'October 27 2026', '27 Oct 2026'."""
    try:
        d = date.fromisoformat(alt)
    except ValueError:
        return [alt]
    full, short = d.strftime("%B").lower(), d.strftime("%b").lower()
    return [alt, f"{d.day} {full} {d.year}", f"{full} {d.day} {d.year}", f"{d.day} {short} {d.year}", f"{short} {d.day} {d.year}"]


def _split(s) -> list[str]:
    return [x.strip() for x in str(s).split(" ; ") if x.strip()] if isinstance(s, str) else []


def must_include_ok(answer: str, spec) -> bool:
    a = norm(answer, number_words=True)
    for fact in _split(spec):
        alts = [x.strip() for x in fact.split(" / ") if x.strip()]
        if not any(norm(v) in a for alt in alts for v in _variants(alt)):
            return False
    return True


def must_not_ok(answer: str, spec) -> bool:
    a = norm(answer)
    return not any(norm(p) in a for p in _split(spec))


def gold_ids(spec) -> list[str]:
    return _split(spec)


def recall(gold: list[str], got: list[str]):
    return None if not gold else sum(g in set(got) for g in gold) / len(gold)


def parse_expected_calls(spec) -> list[tuple[str, dict]]:
    out = []
    for part in _split(spec):
        m = re.match(r"(\w+)\((.*)\)$", part.strip())
        if not m:
            continue
        args = {}
        for kv in [x for x in m.group(2).split(", ") if "=" in x]:
            k, v = kv.split("=", 1)
            args[k.strip()] = v.strip().strip("'\"")
        out.append((m.group(1), args))
    return out


def tools_ok(expected, events) -> bool:
    """Every expected call must appear among the real calls (extra calls are fine)."""
    for name, args in expected:
        if not any(e.name == name and all(str(e.args.get(k)) == v for k, v in args.items()) for e in events):
            return False
    return True
