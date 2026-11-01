"""STEP 4 - the ears (Gnani speech-to-text) and the voice (Gnani text-to-speech).

Pipeline for one spoken turn:   audio file -> Gnani STT -> text -> our agent -> reply text -> Gnani TTS -> audio file

Things this file handles so the agent keeps working when audio is messy:
  * empty / unclear audio      -> a polite "could you say that again?" instead of crashing
  * numbers                    -> order IDs are spoken in groups of four digits, dates carry the year, rupees are spelled out
  * language                   -> the reply's script/words decide which TTS language code to use
  * Gnani hiccups              -> small retries, and a fallback chain of language codes
Only Gnani is used for speech (the brief's rule). Never add another STT/TTS service, not even as a backup.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from .calendar_tools import pretty

REPROMPT = {
    "en-IN": "Sorry, I didn't catch that. Could you say it again?",
    "hi-IN": "माफ़ कीजिए, मैं सुन नहीं पाया। क्या आप दोबारा बोल सकते हैं?",
    "hi-en": "Sorry, mujhe sunai nahi diya. Kya aap dobara bol sakte hain?",
}
SORRY_BROKEN = "Sorry, something went wrong on my side. Please try again in a moment."

_SCRIPTS = [("hi-IN", 0x0900, 0x097F), ("bn-IN", 0x0980, 0x09FF), ("pa-IN", 0x0A00, 0x0A7F), ("gu-IN", 0x0A80, 0x0AFF),
            ("ta-IN", 0x0B80, 0x0BFF), ("te-IN", 0x0C00, 0x0C7F), ("kn-IN", 0x0C80, 0x0CFF), ("ml-IN", 0x0D00, 0x0D7F)]
_HINGLISH = set("hai hain kya aap aapka aapki aapko nahi nahin karein karna sakte sakta sakti milega hoga honge liye lekin kripya "
                "dhanyavaad shukriya haan theek abhi tak se par ko ka ki ke mujhe mera meri hum humare bataiye batayein".split())


def detect_language(text: str) -> str:
    """Which TTS language code fits this text? Indian scripts by Unicode range; Latin text = English or Hinglish."""
    counts = {code: sum(lo <= ord(c) <= hi for c in text) for code, lo, hi in _SCRIPTS}
    best = max(counts, key=counts.get)
    if counts[best] >= 3:
        return best
    words = re.findall(r"[a-z]+", text.lower())
    return "hi-en" if sum(w in _HINGLISH for w in words) >= 2 else "en-IN"


def speech_normalize(text: str) -> str:
    """Write for the ear (see Gnani's text-normalization guide + brief): no markdown, grouped digits, spoken rupees, dates with year."""
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", t)  # emoji
    t = re.sub(r"\b(\d{4}-\d{2}-\d{2})\b", lambda m: pretty(m.group(1)), t)
    t = re.sub(r"(?:₹|Rs\.?)\s?([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(1).replace(',', '')} rupees", t)
    t = re.sub(r"(?<![\d,.])\d{8,}(?![\d,.])", lambda m: " ".join(m.group(0)[i:i + 4] for i in range(0, len(m.group(0)), 4)), t)
    return re.sub(r"\s+", " ", t).strip()


class VoiceError(RuntimeError):
    pass


def _friendly(exc: Exception, service: str) -> str:
    name, low = type(exc).__name__, str(exc).lower()
    if name == "AuthenticationError" or "401" in low or "unauthor" in low:
        return f"Gnani {service} rejected the key. Check GNANI_API_KEY in your .env file."
    if name == "InvalidAudioError":
        return f"Gnani could not read the audio ({exc}). Use WAV, MP3, FLAC, OGG or M4A, at most 60 seconds."
    return f"Gnani {service} error: {type(exc).__name__}: {str(exc)[:200]}"


def _retry(fn, attempts: int = 3):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "status_code", 0)
            if i == attempts - 1 or code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2 ** i)


class Voice:
    def __init__(self, settings, stt=None, tts=None):
        self.s, self._stt, self._tts = settings, stt, tts

    def _clients(self):
        if self._stt is None or self._tts is None:
            if not self.s.gnani_api_key:
                raise VoiceError("GNANI_API_KEY is empty. Create a key at app.gnani.ai/voice and paste it into your .env file.")
            try:
                from gnani.stt import GnaniSTTClient
                from gnani.tts import GnaniTTSClient
            except ImportError as exc:
                raise VoiceError("The Gnani SDK is not installed. Run: pip install gnani-vachana") from exc
            self._stt = self._stt or GnaniSTTClient(api_key=self.s.gnani_api_key)
            self._tts = self._tts or GnaniTTSClient(api_key=self.s.gnani_api_key)
        return self._stt, self._tts

    # ---- ears
    def transcribe(self, audio_path) -> str:
        stt, _ = self._clients()
        codes = list(dict.fromkeys([self.s.stt_language, "en-IN"]))  # configured language(s) first, plain English as backup
        last = None
        for code in codes:
            try:
                res = _retry(lambda: stt.transcribe(str(audio_path), language_code=code))
                text = res.get("transcript", "") if isinstance(res, dict) else getattr(res, "transcript", "")
                return (text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last = exc
                if type(exc).__name__ == "AuthenticationError":
                    break
        raise VoiceError(_friendly(last, "speech-to-text"))

    # ---- voice
    def speak(self, text: str, out_path, language: str | None = None) -> Path:
        _, tts = self._clients()
        lang = language or detect_language(text)
        chain = {"hi-en": ["hi-en", "hi-IN", "auto", None], "en-IN": ["en-IN", "auto", None]}.get(lang, [lang, "auto", None])
        voice = self.s.tts_voice_hindi if lang in ("hi-IN", "hi-en") else self.s.tts_voice
        spoken, last = speech_normalize(text), None
        v25 = self.s.tts_model != "timbre-v2.0"  # only timbre-v2.5+ accepts `language` and `speed`
        speed = getattr(self.s, "tts_speed", 1.0)
        for use_speed in ([True, False] if (v25 and speed != 1.0) else [False]):
            for code in chain:
                kwargs = dict(voice=voice, model=self.s.tts_model)
                if code and v25:
                    kwargs["language"] = code
                if use_speed:
                    kwargs["speed"] = speed
                try:
                    audio = _retry(lambda: tts.synthesize(spoken, **kwargs))
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if type(exc).__name__ == "AuthenticationError":
                        raise VoiceError(_friendly(exc, "text-to-speech")) from exc
                    continue
                if isinstance(audio, (bytes, bytearray)) and audio:
                    out = Path(out_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(bytes(audio))
                    return out
                last = VoiceError(f"unexpected TTS result type {type(audio).__name__}")
        raise VoiceError(_friendly(last, "text-to-speech"))

    def reprompt_text(self, hint_language: str = "en-IN") -> str:
        return REPROMPT.get(hint_language, REPROMPT["en-IN"])


def voice_turn(voice: Voice, conversation, audio_path, out_path):
    """The brief's ten-line voice layer, hardened. Returns (heard, reply, result_or_None, error_or_empty)."""
    error, result = "", None
    try:
        heard = voice.transcribe(audio_path)
    except VoiceError as exc:
        heard, error = "", str(exc)
    if not heard.strip():
        reply = SORRY_BROKEN if error else voice.reprompt_text()
    else:
        try:
            result = conversation.respond(heard, channel="voice")
            reply = result.answer
        except Exception as exc:  # noqa: BLE001 - the caller should still hear something
            reply, error = SORRY_BROKEN, f"agent error: {str(exc)[:200]}"
    try:
        voice.speak(reply, out_path)
    except VoiceError as exc:
        error = (error + " | " if error else "") + str(exc)
    return heard, reply, result, error
