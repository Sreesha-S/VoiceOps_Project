#!/usr/bin/env python
"""Talk to the agent in the terminal (Steps 1-3).      python chat.py
Try:  "My order hasn't arrived. Can I cancel it?"  ->  give order 8978 2837  ->  last 4 digits 4533
Add --sources to see which document sections it used, and --step 1 for the no-documents baseline."""
import argparse

from src.agent import PromptOnlyAgent, SupportAgent
from src.calendar_tools import Calendar
from src.config import load_settings, utf8_console
from src.llm import make_llm
from src.retriever import build_retriever
from src.orders import ToolEvent  # noqa: F401


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["1", "3"], default="3")
    ap.add_argument("--sources", action="store_true", help="show chunks and tool calls")
    ap.add_argument("--dry-run", action="store_true", help="fake LLM (plumbing test)")
    args = ap.parse_args()
    s = load_settings()
    llm = make_llm(s, args.dry_run)
    if args.step == "1":
        agent = PromptOnlyAgent(llm)
        ask = lambda q: agent.answer(q)  # noqa: E731
    else:
        agent = SupportAgent(llm, build_retriever(s), Calendar(s.dataset_dir, s.today), s, mode="conversation")
        convo = agent.start()
        ask = convo.respond
    print("ZentroMart support (empty line to quit)\n")
    while (q := input("You: ").strip()):
        r = ask(q)
        print(f"\nAgent: {r.answer}\n")
        if args.sources and r.chunk_ids:
            print(f"  [search query: {r.rewritten_query or '-'}]")
            print(f"  [chunks: {', '.join(r.chunk_ids)}]")
            print(f"  [tools: {[(e.public()['name'], e.public()['args']) for e in r.tool_calls]}]\n")


if __name__ == "__main__":
    main()
