#!/usr/bin/env python
"""THE BATCH RUN the organizers use.

    python run.py --input-dir test_audio --output-dir outputs

For every audio file in the folder it: transcribes it (Gnani STT) -> runs the agent -> speaks the reply (Gnani TTS)
and saves, per file, the transcript, the reply text, the reply audio and the document chunks it used.
Everything is also collected in outputs/results.csv and outputs/results.jsonl.

Each file is treated as a separate caller by default. If a folder holds multi-turn calls named
<call>_<turn>.wav (e.g. PC002_1.wav, PC002_2.wav), add --multi-turn so turns of the same call share memory.
One bad file never stops the run: the error is recorded and the next file is processed."""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.agent import SupportAgent
from src.calendar_tools import Calendar
from src.config import load_settings, utf8_console
from src.llm import make_llm
from src.retriever import build_retriever
from src.voice import Voice, voice_turn

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="folder with customer audio files")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--multi-turn", action="store_true", help="files named <call>_<turn>.ext share one conversation")
    ap.add_argument("--mode", choices=["conversation", "rag"], default="conversation")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in AUDIO)
    if not files:
        sys.exit(f"No audio files (.wav .mp3 .flac .ogg .m4a) found in {args.input_dir}")
    out = Path(args.output_dir)
    (out / "reply_audio").mkdir(parents=True, exist_ok=True)
    (out / "reply_text").mkdir(parents=True, exist_ok=True)

    s = load_settings()
    agent = SupportAgent(make_llm(s), build_retriever(s), Calendar(s.dataset_dir, s.today), s, mode=args.mode)
    voice = Voice(s)

    def call_key(p: Path):
        m = re.match(r"^(.*?)[_-](?:t|turn)?(\d+)$", p.stem) if args.multi_turn else None
        return (m.group(1), int(m.group(2))) if m else (p.stem, 0)

    convos, rows = {}, []
    for i, f in enumerate(sorted(files, key=lambda p: call_key(p)), 1):
        key = call_key(f)[0]
        if key not in convos or not args.multi_turn:
            convos[key] = agent.start()
        heard, reply, res, err = voice_turn(voice, convos[key], f, out / "reply_audio" / f"{f.stem}.wav")
        (out / "reply_text" / f"{f.stem}.txt").write_text(reply, encoding="utf-8")
        row = {"file": f.name, "transcript": heard, "reply_text": reply,
               "reply_audio": f"reply_audio/{f.stem}.wav" if (out / "reply_audio" / f"{f.stem}.wav").exists() else "",
               "chunks_used": " ; ".join(res.chunk_ids) if res else "",
               "retrieved_chunks": " ; ".join(res.retrieved_ids) if res else "",
               "search_query": res.rewritten_query if res else "",
               "tool_calls": json.dumps([e.public() for e in res.tool_calls], ensure_ascii=False) if res else "[]",
               "error": err}
        rows.append(row)
        print(f"[{i}/{len(files)}] {f.name}: {heard[:60]!r} -> {reply[:70]!r}" + (f"  !! {err[:80]}" if err else ""), flush=True)

    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(out / "results.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. {len(rows)} files -> {out}/results.csv  (audio in {out}/reply_audio)")


if __name__ == "__main__":
    main()
