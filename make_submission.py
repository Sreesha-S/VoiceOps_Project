#!/usr/bin/env python
"""Build the submission zip safely:   python make_submission.py

It (1) finds demo.mp4 / report.pdf / sample_outputs if they are not in place yet, (2) checks that every required item exists,
(3) refuses to continue if any of your API keys appears inside a file that would be zipped, and (4) writes sreesha_voiceops.zip
with the layout from the brief. It never includes .env, .venv, .cache, the dataset or __pycache__."""
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEAM = "sreesha"
ZIP = ROOT / f"{TEAM}_voiceops.zip"
FILES = ["README.md", ".env.example", "requirements.txt", "run.py", "report.pdf", "demo.mp4", "chat.py", "voice_chat.py", "voice_test.py",
         "check_setup.py", "evaluate_questions.py", "evaluate_conversations.py", "make_submission.py", ".gitignore"]
DIRS = ["src", "tests", "docs", "sample_outputs"]
SKIP_PARTS = {"__pycache__", ".venv", ".cache", ".git"}
SEARCH = [ROOT, ROOT.parent, Path.home() / "Downloads", Path.home() / "Desktop", Path.home() / "Videos", Path.home() / "Videos" / "Captures",
          Path.home() / "Documents"]


def find(patterns):
    for d in SEARCH:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and any(re.fullmatch(p, f.name, re.I) for p in patterns):
                    return f
    return None


def main():
    problems = []
    if not (ROOT / "demo.mp4").exists():
        f = find([r"demo\.mp4", r"voiceops_demo.*\.mp4", r"voiceops_demo"])
        if f:
            shutil.copy(f, ROOT / "demo.mp4")
            print(f"copied {f} -> demo.mp4")
    if not (ROOT / "report.pdf").exists():
        f = find([r"report\.pdf", r"report \(\d+\)\.pdf"])
        if f:
            shutil.copy(f, ROOT / "report.pdf")
            print(f"copied {f} -> report.pdf")
    so = ROOT / "sample_outputs"
    if (ROOT / "outputs" / "results.csv").exists() and not (so / "results.csv").exists():
        so.mkdir(exist_ok=True)
        shutil.copytree(ROOT / "outputs", so, dirs_exist_ok=True)
        print("copied outputs/ -> sample_outputs/")
    for need in ["README.md", ".env.example", "requirements.txt", "run.py", "src", "report.pdf", "demo.mp4"]:
        if not (ROOT / need).exists():
            problems.append(f"MISSING: {need}")
    if not (so / "results.csv").exists():
        problems.append("MISSING: sample_outputs/results.csv  (run: python run.py --input-dir test_audio --output-dir outputs)")

    # collect files
    items = []
    for name in FILES:
        if (ROOT / name).is_file():
            items.append(ROOT / name)
    for d in DIRS:
        for f in sorted((ROOT / d).rglob("*")) if (ROOT / d).is_dir() else []:
            if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_PARTS):
                items.append(f)

    # secret scan
    try:
        from dotenv import dotenv_values
        env = dotenv_values(ROOT / ".env")
    except Exception:  # noqa: BLE001
        env = {}
    secrets = [v for k, v in {**env, **os.environ}.items() if v and re.search(r"KEY|TOKEN|SECRET", k, re.I) and len(v) >= 12]
    for f in items:
        if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".mp4", ".pdf", ".npy"}:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if re.search(r"AIza[0-9A-Za-z_\-]{30,}", text) or any(s in text for s in secrets):
            problems.append(f"SECRET FOUND inside {f.relative_to(ROOT)} - remove it before submitting")
    for f in items:
        if f.name == ".env.example":
            for line in f.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*([A-Z_]*KEY[A-Z_]*)\s*=\s*(\S+)", line)
                if m:
                    problems.append(f".env.example has a value for {m.group(1)} - it must be empty")

    if problems:
        print("\nCannot build the zip yet:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in items:
            z.write(f, f.relative_to(ROOT).as_posix())
    names = zipfile.ZipFile(ZIP).namelist()
    assert ".env" not in names and not any(n.startswith((".venv", ".cache")) for n in names)
    print(f"\nOK: {ZIP.name}  ({ZIP.stat().st_size / 1e6:.1f} MB, {len(names)} files). Top level:")
    print("  " + ", ".join(sorted({n.split('/')[0] for n in names})))


if __name__ == "__main__":
    main()
