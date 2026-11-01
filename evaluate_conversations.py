#!/usr/bin/env python
"""STEP 3: replay the 23 practice conversations (96 customer turns) and check every turn.

    python evaluate_conversations.py                  # all conversations
    python evaluate_conversations.py --only PC002     # one conversation
    python evaluate_conversations.py --limit 3        # first 3 conversations
A turn passes when the reply contains the must_include facts, avoids must_not_include phrases, and the
expected tool calls really happened (e.g. lookup_order with the right verification value, create_ticket
with created=True). Answers are cached, so re-runs are free."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import scoring as sc
from src.agent import SupportAgent
from src.calendar_tools import Calendar
from src.config import ROOT, load_settings, utf8_console
from src.llm import LLMError, make_llm
from src.retriever import build_retriever


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    s = load_settings()
    df = pd.read_csv(s.dataset_dir / "practice" / "conversations.csv", dtype=str).fillna("")
    ids = list(dict.fromkeys(df.conversation_id))
    ids = [args.only] if args.only else (ids[:args.limit] if args.limit else ids)
    agent = SupportAgent(make_llm(s, args.dry_run), build_retriever(s), Calendar(s.dataset_dir, s.today), s, mode="conversation")

    rows = []
    try:
        for cid in ids:
            convo = agent.start()
            for _, r in df[df.conversation_id == cid].sort_values("turn", key=lambda c: c.astype(int)).iterrows():
                res = convo.respond(r.customer_message)
                expected = sc.parse_expected_calls(r.expected_tool_calls)
                ok_in, ok_out = sc.must_include_ok(res.answer, r.must_include), sc.must_not_ok(res.answer, r.must_not_include)
                ok_tools = sc.tools_ok(expected, res.tool_calls)
                if r.expected_decision == "escalate":  # a handover only counts if the ticket REALLY exists
                    ok_tools = ok_tools and any(e.name == "create_ticket" and e.result.get("created") for e in res.tool_calls)
                gold = sc.gold_ids(r.gold_evidence)
                rows.append(dict(conversation=cid, turn=r.turn, scenario=r.scenario, expected=r.expected_decision, customer=r.customer_message,
                                 agent=res.answer, must_include_ok=ok_in, must_not_ok=ok_out, tools_ok=ok_tools,
                                 passed=ok_in and ok_out and ok_tools, recall=sc.recall(gold, res.chunk_ids), chunks=" ; ".join(res.chunk_ids),
                                 tools=json.dumps([e.public() for e in res.tool_calls], ensure_ascii=False)))
                print(f"{cid} T{r.turn} [{r.expected_decision}] {'PASS' if rows[-1]['passed'] else 'fail'}  in={ok_in} out={ok_out} tools={ok_tools}", flush=True)
    except LLMError as exc:
        print(f"\nStopped: {exc}\nEverything answered so far is cached; run the same command again later to continue.")

    if rows:
        res = pd.DataFrame(rows)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "conversation_results.csv", index=False, encoding="utf-8-sig")
        summary = {"turns": int(len(res)), "turns_passed": int(res.passed.sum()), "turn_pass_rate": round(float(res.passed.mean()), 3),
                   "conversations_fully_passed": int(res.groupby("conversation").passed.all().sum()), "conversations": int(res.conversation.nunique()),
                   "by_expected_decision": {k: round(float(v), 3) for k, v in res.groupby("expected").passed.mean().items()},
                   "mean_retrieval_recall": round(float(res.recall.dropna().mean()), 3)}
        (out / "conversation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n" + json.dumps(summary, indent=2))
        if args.dry_run:
            print("\n(dry run: fake model - these numbers mean nothing)")


if __name__ == "__main__":
    main()
