#!/usr/bin/env python
"""Talk to the agent by VOICE (Step 4).   python voice_chat.py

Each turn you either
  * give the path of an audio file (record yourself with Windows 'Sound Recorder' / phone; wav, mp3, m4a, flac, ogg; max 60 s), or
  * just press Enter to record from your microphone (needs:  pip install sounddevice).
The agent's spoken reply is saved in the 'voice_replies' folder and opened in your default audio player."""
import argparse
import os
import subprocess
import sys
import threading
import wave
from pathlib import Path

from src.agent import SupportAgent
from src.calendar_tools import Calendar
from src.config import ROOT, load_settings, utf8_console
from src.llm import make_llm
from src.retriever import build_retriever
from src.voice import Voice, voice_turn


def record(path: Path, rate: int = 16000) -> Path:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("To record from the microphone run:  pip install sounddevice   (or give an audio file path instead)")
    frames = []
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", callback=lambda d, *_: frames.append(d.copy())):
        input("  Recording... speak now, press Enter to stop (max 60 s). ")
    import numpy as np

    data = np.concatenate(frames) if frames else np.zeros((1, 1), dtype="int16")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(data.tobytes())
    return path


def play(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["afplay", str(path)], check=False)
        else:
            subprocess.run(["aplay", str(path)], check=False)
    except Exception:  # noqa: BLE001
        print(f"  (open {path} yourself to listen)")


def main():
    utf8_console()
    s = load_settings()
    agent = SupportAgent(make_llm(s), build_retriever(s), Calendar(s.dataset_dir, s.today), s, mode="conversation")
    convo, voice = agent.start(), Voice(s)
    out = ROOT / "voice_replies"
    out.mkdir(exist_ok=True)
    print("Voice demo. Type 'q' to quit.\n")
    n = 0
    while True:
        src = input("Audio file path (Enter = record with microphone): ").strip().strip('"')
        if src.lower() == "q":
            break
        n += 1
        audio = Path(src) if src else record(out / f"customer_{n}.wav")
        if not audio.exists():
            print("  file not found, try again\n")
            n -= 1
            continue
        heard, reply, result, err = voice_turn(voice, convo, audio, out / f"reply_{n}.wav")
        print(f"\n  You said : {heard or '(nothing heard)'}\n  Agent    : {reply}")
        if result is not None:
            print(f"  [chunks: {', '.join(result.chunk_ids[:8])}]")
        if err:
            print(f"  [problem: {err}]")
        if (out / f"reply_{n}.wav").exists():
            play(out / f"reply_{n}.wav")
        print()


if __name__ == "__main__":
    main()
