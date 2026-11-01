# VoiceOps 2026 - ZentroMart voice support agent

A customer-support agent for the fictional online store **ZentroMart**. It answers from the company documents (RAG),
remembers the conversation, verifies the caller before sharing order details, hands over to a human through a real
ticket, and talks over audio with **Gnani** speech-to-text and text-to-speech.

- **Author:** Sreesha (individual submission) - **GitHub:** https://github.com/Sreesha-S/VoiceOps_Project
- **Brain:** Google Gemini (`gemini-3.5-flash-lite`, free API key) - **Speech:** Gnani STT + TTS only (no other speech service)

## Results on the practice data
| | Step 1: prompt only | Step 2: with documents (RAG) |
|---|---|---|
| 52 practice questions passed | 18 / 52 (34.6%) | **50 / 52 (96.2%)** |
| Retrieval: gold sections found | - | mean recall 0.954 (all found for 89.4%) |

Step 3 (23 conversations, 96 turns): **92 / 96 turns passed (95.8%)**, 19 of 23 conversations fully passed.
These are practice-set numbers that guided development, so expect lower scores on unseen questions. Details: `report.pdf`.

## Run it (the batch command used for testing)
```bash
python run.py --input-dir <folder-with-customer-audio> --output-dir outputs
```
For every audio file (`.wav .mp3 .flac .ogg .m4a`, max 60 s) it saves: transcript, reply text, reply audio, and the
document chunks used - in `outputs/results.csv`, `outputs/results.jsonl`, `outputs/reply_text/` and `outputs/reply_audio/`.
Each file is treated as a separate caller. For multi-turn calls named `<call>_<turn>.wav` add `--multi-turn`.
One bad file never stops the run: the error is recorded and the next file is processed. `sample_outputs/` shows a real run.

## Setup
1. Python 3.10+. Create and activate a virtual environment, then `pip install -r requirements.txt`.
2. Extract the capstone dataset (`Capstone_Project_Dataset_FTTT_2026`) and either put the folder next to this project
   or set `DATASET_DIR` in `.env` to its full path. (The dataset is not included in this repository.)
3. `cp .env.example .env` (Windows: `copy .env.example .env`) and fill in `GEMINI_API_KEY` (https://aistudio.google.com/apikey)
   and `GNANI_API_KEY` (https://app.gnani.ai/voice). Never commit `.env`.
4. `python check_setup.py` tests every key and service and tells you what to fix.

## Other commands
| Command | Purpose |
|---|---|
| `python -m unittest discover -s tests` | 27 offline tests, no keys needed |
| `python evaluate_questions.py --step both` | Step 1 vs Step 2 on the 52 practice questions |
| `python evaluate_conversations.py` | Step 3: the 23 practice conversations |
| `python chat.py --sources` | text chat; shows chunks and tool calls |
| `python voice_chat.py` | live voice demo (audio file or microphone) |
| `python voice_test.py` | compare Gnani voices by ear |
| `python make_submission.py` | build the submission zip and check no secret is inside |

The free Gemini tier is rate-limited. Answers are cached in `.cache/` and requests are paced, so re-running is cheap;
if you hit a daily limit, run the same command again later and it continues.

## How it works (one turn)
```
audio -> Gnani STT -> text
  code: read order ID / verification digits -> lookup_order      (identity checks are done by code, never by the LLM)
  Gemini: rewrite the message into a clean English search query   (Hindi, Hinglish, speech errors, follow-ups)
  search: BM25 keyword + Gemini embeddings (RRF); policy outranks FAQ; products in their own lane; category-aware
  Gemini: answer ONLY from CALL STATE + ORDER FACTS + retrieved CONTEXT; tools: date_calc, check_pincode, create_ticket
  guards: no claim of a ticket unless create_ticket returned created=true; secrets redacted from replies and tickets
  -> speech-friendly cleanup -> Gnani TTS -> audio
```
| File | Role |
|---|---|
| `src/kb.py` | one chunk per document section (IDs, status, dates kept) and per product row |
| `src/retriever.py` | hybrid search, precedence weights, product lane |
| `src/agent.py` | Step 1 baseline; Step 2/3 agent (retrieval, memory, tools, guards) |
| `src/orders.py` | caller verification (2 attempts, lock-out), order facts, ticket wrapper |
| `src/calendar_tools.py` | business-day maths (Mon-Sat, holidays skipped, fixed date 20 Oct 2026) |
| `src/voice.py` | Gnani STT/TTS layer, language detection, speech-friendly text, fallbacks |
| `src/prompts.py` | all prompts in one place |
| `run.py` | batch run over a folder of audio |

## Safety and rules followed
- Gnani is the only speech service; synthetic data only; own voice recordings only.
- Order details reach the model only after code-verified identity; verification digits never appear in logs, tickets or replies.
- No hardcoded answers: every answer is generated from retrieved document text.
