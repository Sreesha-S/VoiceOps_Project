#!/usr/bin/env python
"""RUN THIS FIRST:   python check_setup.py
Tests, one by one, everything the project needs, and tells you what to fix:
  1 Python + packages   2 dataset folder   3 Gemini key + model + function calling
  4 Gemini embeddings (optional)   5 Gnani text-to-speech   6 Gnani speech-to-text (re-listens to the audio from step 5)"""
import dataclasses
import sys

from src.config import ROOT, find_dataset, load_settings, utf8_console

OK, BAD = "  [PASS]", "  [FAIL]"


def section(t):
    print(f"\n== {t}")


def main():
    utf8_console()
    section("1. Python and packages")
    print(f"  Python {sys.version.split()[0]}", "(ok)" if sys.version_info >= (3, 9) else "(need 3.9 or newer!)")
    for mod, pipname in [("openai", "openai"), ("numpy", "numpy"), ("pandas", "pandas"), ("dotenv", "python-dotenv"), ("gnani", "gnani-vachana")]:
        try:
            __import__(mod)
            print(f"{OK} {pipname}")
        except ImportError:
            print(f"{BAD} {pipname} missing -> run: pip install -r requirements.txt")

    section("2. Dataset")
    try:
        ds = find_dataset()
        print(f"{OK} found at {ds}")
    except FileNotFoundError as exc:
        print(f"{BAD} {exc}")
        return
    s = load_settings()
    print(f"  today (dataset date) = {s.today}")

    section("3. Gemini (the brain)")
    if not s.llm_api_key:
        print(f"{BAD} GEMINI_API_KEY is empty in your .env file (create one at https://aistudio.google.com/apikey)")
    else:
        try:
            from openai import OpenAI

            from src.calendar_tools import Calendar
            from src.llm import LLMClient, friendly_error

            raw = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url, timeout=60, max_retries=0)
            try:
                names = sorted(m.id.replace("models/", "") for m in raw.models.list() if "gemini" in m.id.lower())
                print(f"{OK} key accepted. Gemini models your key can see (first 25 of {len(names)}):")
                print("     " + ", ".join(names[:25]))
                if s.llm_model not in names:
                    print(f"  [WARN] GEMINI_MODEL={s.llm_model!r} is not in that list. Pick one 'flash' model from it and set GEMINI_MODEL in .env")
            except Exception as exc:  # noqa: BLE001
                print(f"  (could not list models: {friendly_error(exc)})")
            llm = LLMClient(dataclasses.replace(s, llm_cache=False))
            r = llm.chat([{"role": "user", "content": "Reply with exactly: OK"}])
            print(f"{OK} model {s.llm_model} answered: {r.text[:40]!r}")
            cal = Calendar(ds, s.today)
            r = llm.chat([{"role": "user", "content": "Use the date_calc tool: what date is 5 business days after 2026-10-17? Answer with the date."}],
                         tools=cal.schemas(), tool_funcs=cal.functions())
            used = [t.name for t in r.tool_calls]
            print(f"{OK if used else BAD} function calling {'worked' if used else 'did NOT happen'}: tools used={used}, answer={r.text[:60]!r}")
            if used:
                print("     (expected 24 October 2026)")
        except Exception as exc:  # noqa: BLE001
            from src.llm import friendly_error

            print(f"{BAD} {friendly_error(exc)}")

    section("4. Gemini embeddings (optional; the agent falls back to keyword search without them)")
    if s.llm_api_key:
        try:
            from src.retriever import APIEmbedder

            v = APIEmbedder(s).embed(["hello"])
            print(f"{OK} {s.embedding_model} works (vector size {v.shape[1]})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] embeddings not available ({str(exc)[:120]}). Fine: set EMBEDDING_BACKEND=bm25 to skip.")

    section("5. Gnani text-to-speech (the voice)")
    from src.voice import Voice, VoiceError

    voice, wav = Voice(s), ROOT / ".cache" / "setup_check.wav"
    if not s.gnani_api_key:
        print(f"{BAD} GNANI_API_KEY is empty in your .env file (create one at https://app.gnani.ai/voice)")
        return
    try:
        try:
            from gnani.tts import GnaniTTSClient

            v = GnaniTTSClient.supported_voices(model=s.tts_model)
            print(f"  voices for {s.tts_model}: {list(v)[:12] if not isinstance(v, str) else v} ...")
        except Exception:  # noqa: BLE001
            pass
        voice.speak("Hello, this is the ZentroMart support assistant. Your order is on its way.", wav, "en-IN")
        print(f"{OK} spoke a test sentence -> {wav}  (play it to hear it)")
        voice.speak("नमस्ते, मैं ज़ेंट्रोमार्ट का एआई सहायक हूँ।", ROOT / ".cache" / "setup_check_hi.wav", "hi-IN")
        print(f"{OK} Hindi test spoken -> .cache/setup_check_hi.wav  (if this fails or sounds wrong, pick a Hindi voice in GNANI_TTS_VOICE_HINDI)")
    except VoiceError as exc:
        print(f"{BAD} {exc}")

    section("6. Gnani speech-to-text (the ears)")
    try:
        heard = voice.transcribe(wav)
        print(f"{OK} it listened to the sentence above and heard: {heard!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} {exc}")
    print("\nIf everything above says PASS you are ready:  python evaluate_questions.py --step both")


if __name__ == "__main__":
    main()
