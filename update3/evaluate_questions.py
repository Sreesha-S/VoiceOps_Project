#!/usr/bin/env python
"""STEPS 1 and 2: score the 52 practice questions.

    python evaluate_questions.py --step both           # Step 1 (no docs) vs Step 2 (RAG)  -> numbers for your report
    python evaluate_questions.py --retrieval-only      # search quality only: needs NO LLM key
    python evaluate_questions.py --step 2 --input asr  # use the speech-to-text style text instead
    python evaluate_questions.py --step both --limit 5 # try 5 questions first
Answers are cached on disk, so re-running costs no extra Gemini requests."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import scoring as sc
from src.agent import PromptOnlyAgent, SupportAgent
from src.calendar_tools import Calendar
from src.config import ROOT, load_settings, utf8_console
from src.llm import LLMError, make_llm
from src.retriever import build_retriever


def summarize(df: pd.DataFrame) -> dict:
    out = {"questions": int(len(df)), "passed": int(df.passed.sum()), "pass_rate": round(float(df.passed.mean()), 3) if len(df) else 0}
    if "recall" in df and df.recall.notna().any():
        out["mean_retrieval_recall"] = round(float(df.recall.dropna().mean()), 3)
        out["all_gold_chunks_found"] = round(float((df.recall.dropna() == 1).mean()), 3)
    out["pass_rate_by_language"] = {k: round(float(v), 3) for k, v in df.groupby("language").passed.mean().items()}
    out["pass_rate_unanswerable_vs_answerable"] = {("answerable" if k == "True" else "not in documents"): round(float(v), 3)
                                                   for k, v in df.groupby("answerable").passed.mean().items()}
    return out


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["1", "2", "both"], default="both")
    ap.add_argument("--input", choices=["text", "asr"], default="text")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="only these questions, e.g. PQ0021,PQ0039 (use with --out results_check)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    s = load_settings()
    df = pd.read_csv(s.dataset_dir / "practice" / "questions.csv", dtype=str).fillna("")
    df = df[df.id.isin([i.strip() for i in args.ids.split(",")])] if args.ids else df
    df = df.head(args.limit) if args.limit else df
    col = "asr_text" if args.input == "asr" else "question"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    if args.retrieval_only:
        retr = build_retriever(s)
        rows = []
        for _, r in df.iterrows():
            got = [c.chunk_id for c in retr.search([r[col]], k=s.top_k)]
            gold = sc.gold_ids(r.gold_evidence)
            rows.append(dict(id=r.id, language=r.language, answerable=r.answerable, passed=False, recall=sc.recall(gold, got),
                             recall_with_standing=sc.recall(gold, got + ["KB-00.1", "KB-00.3"]), gold=" ; ".join(gold), found=" ; ".join(got)))
        res = pd.DataFrame(rows)
        res.to_csv(out / "retrieval_only.csv", index=False, encoding="utf-8-sig")
        sub = res[res.recall.notna()]
        summary["retrieval_only"] = {"search": retr.mode, "mean_recall": round(float(sub.recall.mean()), 3),
                                     "mean_recall_with_KB00_always_included": round(float(sub.recall_with_standing.mean()), 3),
                                     "all_gold_found": round(float((sub.recall == 1).mean()), 3)}
    else:
        llm = make_llm(s, args.dry_run)
        cal = Calendar(s.dataset_dir, s.today)
        retr = build_retriever(s) if args.step in ("2", "both") else None
        for step in ([1, 2] if args.step == "both" else [int(args.step)]):
            agent = PromptOnlyAgent(llm) if step == 1 else SupportAgent(llm, retr, cal, s, mode="rag")
            rows = []
            for i, (_, r) in enumerate(df.iterrows(), 1):
                try:
                    res = agent.answer(r[col])
                except LLMError as exc:
                    print(f"\nStopped at question {i}: {exc}\nEverything answered so far is cached; run the same command again later to continue.")
                    break
                gold = sc.gold_ids(r.gold_evidence)
                ok_in, ok_out = sc.must_include_ok(res.answer, r.must_include), sc.must_not_ok(res.answer, r.must_not_include)
                rows.append(dict(id=r.id, language=r.language, intent=r.intent, answerable=r.answerable, input=r[col], answer=res.answer,
                                 search_query=res.rewritten_query, chunks_used=" ; ".join(res.chunk_ids), gold=" ; ".join(gold),
                                 recall=sc.recall(gold, res.chunk_ids) if step == 2 else None, must_include_ok=ok_in, must_not_ok=ok_out,
                                 passed=ok_in and ok_out, tools=json.dumps([e.name for e in res.tool_calls])))
                print(f"[step {step}] {i}/{len(df)} {r.id} {'PASS' if ok_in and ok_out else 'fail'}", flush=True)
            if rows:
                res_df = pd.DataFrame(rows)
                res_df.to_csv(out / f"step{step}_results.csv", index=False, encoding="utf-8-sig")
                summary[f"step{step}"] = summarize(res_df)

    sf = out / "summary.json"
    if sf.exists():  # keep the numbers of steps you did not re-run this time
        try:
            summary = {**json.loads(sf.read_text(encoding="utf-8")), **summary}
        except Exception:  # noqa: BLE001
            pass
    sf.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n(dry run: fake model - these numbers mean nothing)")


if __name__ == "__main__":
    main()
