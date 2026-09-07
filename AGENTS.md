# AGENTS.md

## Project
Autonomous B2B RFP Bidder — a hackathon MVP built for the Anakin Forge Hackathon.

One-line goal: an agent reads an RFP, drafts grounded answers from our capability docs, stages a bid on a form, and a human clicks approve to submit it.

Core loop: **Read (Anakin Crawl) → Reason (FAISS + Groq) → Act (Anakin Browser Sessions) → HITL approval → Submit**

## Tech stack
- Python 3.11, FastAPI backend
- FAISS (IndexFlatIP) + sentence-transformers (ll-MiniLM-L6-v2) for retrieval — in-memory only, no external DB
- Groq API (Llama-3) via iohttp + syncio.gather, esponse_format: json_object for structured output
- Anakin Crawl API (ingestion) + Anakin Browser Sessions API (actuation)
- Next.js + Tailwind frontend, polling for state updates (no SSE)
- syncio.Event() for the human-in-the-loop approval lock

## Hard rules (do not violate)
- **Mock portal only.** All Browser Sessions actuation targets our own self-hosted FastAPI + HTML mock portal, read from an environment variable. Never write code that targets a real third-party procurement site or a hardcoded external domain.
- **No fabricated metrics.** Never hardcode or invent performance numbers, accuracy percentages, or SLA claims in code, comments, or docs (e.g., no "0% hallucination," no invented latency figures). If something isn't actually measured, it isn't stated as a number.
- **Keys via environment variables only.** Groq and Anakin API keys are read from .env / environment variables, never hardcoded in source. .env is gitignored.
- **Submission is human-gated, always.** The final submit action must never fire without an explicit approval call to /api/bids/{id}/approve. No autonomous submission path, ever, even in test code.
- **No unbounded retries.** If the crawler or browser session hits a blocking condition, abort and flag for manual intervention — do not loop retries.
- **One file or function per task.** Keep generated changes scoped and small enough to review in one pass. No multi-file "build the whole system" generations.

## File layout
/reasoner/vector_store.py       - FAISS retrieval over capability docs
/reasoner/groq_reasoner.py      - concurrent Groq calls, structured JSON generation
/actuation/anakin_client.py     - crawl_rfp(), stage_bid() via Anakin APIs
/actuation/hitl_manager.py      - asyncio.Event() approval lock, approve endpoint
/mock-portal/                   - FastAPI + HTML multi-page form (Security, Tech Specs, Pricing, Submit)
/dashboard/                     - Next.js + Tailwind review UI (polling)
/sample-data/                   - capability docs + one seeded RFP

## Code style
- Small, testable functions. No god-files.
- Every external API call (Groq, Anakin) has a timeout and try/except.
- After writing any async code, add a short comment explaining the concurrency model (what runs concurrently, what the lock blocks).
- Prefer explicit, debuggable code over clever abstractions — this needs to be fixable fast during a hackathon week.
