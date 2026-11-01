# VoiceOps 2026 - Report (max 4 pages; export to report.pdf)

## 1. What I built (one short paragraph per step)
- Step 1 - prompt-only baseline: system prompt, AI disclosure, no documents.
- Step 2 - RAG: one chunk per document section (IDs kept), keyword + embeddings search with precedence priors, product lane, query rewriting, chunk logging, date tool.
- Step 3 - conversation: memory, caller verification in code (one retry, lockout), real lookup_order / create_ticket, handover, refusals.
- Step 4 - voice: Gnani STT -> agent -> Gnani TTS; speech-friendly text; unclear-audio recovery; batch run.py.

## 2. Results (copy from results/summary.json and results/conversation_summary.json)
| | Step 1 (no docs) | Step 2 (RAG) |
|---|---|---|
| Practice questions passed | __ / 52 | __ / 52 |
| Mean retrieval recall | n/a | __ |
Conversation turns passed: __ / 96. Conversations fully passed: __ / 23.

## 3. What went wrong and what I would fix
(Look at the failing rows in results/*.csv. Group them: wrong chunk retrieved? wrong date? ignored precedence? language? Be specific.)

## 4. What I learned / bonus work
(Why Step 2 beat Step 1; what the rewrite step and embeddings changed; any extras you built.)
