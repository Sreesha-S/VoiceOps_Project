"""Settings: everything is read from environment variables (or the .env file).

Why: API keys must never live inside code. You change behaviour by editing .env.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # python-dotenv reads the .env file for us
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def utf8_console() -> None:
    """Windows consoles often can't print Hindi or the rupee sign. Force UTF-8 so nothing crashes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def find_dataset() -> Path:
    """Locate the extracted dataset folder (the one that contains kb/, data/, practice/, tools/)."""
    candidates = []
    if _env("DATASET_DIR"):
        candidates.append(Path(_env("DATASET_DIR")).expanduser())
    name = "Capstone_Project_Dataset_FTTT_2026"
    candidates += [ROOT / name, ROOT / "dataset", ROOT.parent / name, Path.cwd() / name, Path.cwd().parent / name]
    for c in candidates:
        for base in ([c] if c.is_absolute() else [ROOT / c, Path.cwd() / c]):
            if (base / "kb").is_dir() and (base / "data").is_dir():
                return base.resolve()
    raise FileNotFoundError(
        "Could not find the dataset folder. Extract Capstone_Project_Dataset_FTTT_2026.zip, then either put the "
        "folder next to this project or set DATASET_DIR in your .env file to its full path."
    )


@dataclass(frozen=True)
class Settings:
    dataset_dir: Path
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_effort: str          # Gemini "thinking" level: low = faster and cheaper
    llm_cache: bool          # remember answers on disk so re-runs don't use up your free quota
    embedding_backend: str   # auto | bm25 | api
    embedding_model: str
    top_k: int
    today: str               # the dataset's fixed "today" - NOT your computer's clock
    gnani_api_key: str
    tts_voice: str
    tts_voice_hindi: str     # voice used for Hindi / Hinglish replies (defaults to tts_voice)
    tts_model: str
    tts_speed: float         # 0.85-1.15; below 1.0 = slower, easier to follow
    stt_language: str
    cache_dir: Path


def load_settings(need_dataset: bool = True) -> Settings:
    return Settings(
        dataset_dir=find_dataset() if need_dataset else ROOT,
        llm_api_key=_env("GEMINI_API_KEY") or _env("LLM_API_KEY"),
        llm_base_url=_env("LLM_BASE_URL", GEMINI_OPENAI_URL),
        llm_model=_env("GEMINI_MODEL") or _env("LLM_MODEL", "gemini-3.8-flash"),
        llm_effort=_env("LLM_REASONING_EFFORT", "low"),
        llm_cache=_env("LLM_CACHE", "1") != "0",
        embedding_backend=_env("EMBEDDING_BACKEND", "auto").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "gemini-embedding-001"),
        top_k=int(_env("TOP_K", "8")),
        today=_env("TODAY", "2026-10-20"),
        gnani_api_key=_env("GNANI_API_KEY"),
        tts_voice=_env("GNANI_TTS_VOICE", "Kaveri"),
        tts_voice_hindi=_env("GNANI_TTS_VOICE_HINDI") or _env("GNANI_TTS_VOICE", "Kaveri"),
        tts_model=_env("GNANI_TTS_MODEL", "timbre-v2.5"),
        tts_speed=min(1.15, max(0.85, float(_env("GNANI_TTS_SPEED", "0.95")))),
        stt_language=_env("GNANI_STT_LANGUAGE", "en-IN,hi-IN"),
        cache_dir=ROOT / ".cache",
    )
