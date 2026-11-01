#!/usr/bin/env python
"""Compare Gnani voices by ear (uses only Gnani, no Gemini).

    python voice_test.py                      # tries a few example voices
    python voice_test.py Kaveri Asmita Nalini # tries the voices you name
Each voice reads one English and one Hinglish sentence. Files are saved in .cache\\voice_test - open the folder and listen.
Then put the one you like in .env:  GNANI_TTS_VOICE=...  (English)   GNANI_TTS_VOICE_HINDI=...  (Hindi / Hinglish)
Speed is set with GNANI_TTS_SPEED (0.85 slow ... 1.15 fast; default 0.95)."""
import dataclasses
import sys

from src.config import ROOT, load_settings, utf8_console
from src.voice import Voice, VoiceError

EN = "Hello, I am ZentroMart's AI support assistant. We deliver only within India, so we cannot deliver to Dubai."
HINGLISH = "Main ZentroMart ka AI support assistant hoon. Aapka refund 5 se 7 business days mein aa jayega. Madad ke liye main yahan hoon."


def main():
    utf8_console()
    s = load_settings(need_dataset=False)
    names = sys.argv[1:] or ["Kaveri", "Nalini", "Asmita", "Suhana", "Poorvi"]
    out = ROOT / ".cache" / "voice_test"
    for name in names:
        v = Voice(dataclasses.replace(s, tts_voice=name, tts_voice_hindi=name))
        for tag, text, lang in (("english", EN, "en-IN"), ("hinglish", HINGLISH, "hi-en")):
            try:
                path = v.speak(text, out / f"{name}_{tag}.wav", lang)
                print(f"saved {path}")
            except VoiceError as exc:
                print(f"{name} {tag}: FAILED - {exc}")
    print(f"\nOpen this folder and listen:  {out}")


if __name__ == "__main__":
    main()
