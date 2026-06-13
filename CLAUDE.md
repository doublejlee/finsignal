# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Setup (first time) — full local pipeline
pip install -r requirements-pipeline.txt
python -m spacy download en_core_web_sm   # spaCy model is not on PyPI
# (root requirements.txt is the minimal serving set used by Streamlit Cloud)

# Stage 1 — ingestion + sentiment (run with VPN ON; YouTube blocks bare IPs)
python main.py

# Stage 2 — reason extraction (run with VPN OFF; Groq blocks VPN IPs)
python -m nlp.backfill_reasons

# Stage 3 — creator accuracy backtest (yfinance prices vs each call, 30-day horizon)
# Also auto-promotes candidate creators to 'tracked' once they clear the screening bar.
python backtest.py

# Out-of-sample validation — split each creator's calls in half by date and check whether
# in-sample beat-SPY skill persists out-of-sample (guards against overfit leaderboard ranks)
python validate.py

# Or run all three stages in order with VPN-toggle prompts:
python refresh.py

# API — serve stored analytics as JSON (run from repo root; interactive docs at /docs)
uvicorn api.main:app --reload

# RAG — embed stored sentences (local, uses torch), then ask grounded questions
python -m nlp.backfill_embeddings
python -m nlp.rag "what is the bull case for NVDA?"

# Dashboard
streamlit run dashboard/app.py

# Initialize or verify DB schema
python -m db.init

# Test individual modules (each has a __main__ block)
python -m ingestion.youtube_fetcher      # fetches one hardcoded video transcript
python -m ingestion.channel_fetcher      # lists recent videos from a channel
python -m nlp.ticker_extractor           # extracts tickers from a transcript
python -m nlp.chunker                    # shows per-ticker sentence context windows
python -m nlp.sentiment                  # scores ticker sentiment for a transcript
python -m nlp.topic_model                # extracts reasons for GOOGL sentiment from DB
```

**Why two stages:** YouTube transcript fetching needs a VPN (bare residential/cloud
IPs get `IpBlocked`), but Groq returns `403 Access denied` from VPN exit IPs. The two
have opposite network requirements, so reason extraction is decoupled from ingestion:
`main.py` stores raw sentences in `transcript_segments`, and `backfill_reasons.py`
reads those later (VPN off) to populate `ticker_reasons`. The backfill is resumable —
it skips `(video_id, ticker)` pairs already in `ticker_reasons`.

## Environment

`.env` at root requires:
- `YOUTUBE_API_KEY` — used by `ingestion/channel_fetcher.py` to fetch channel video lists via YouTube Data API v3
- `GROQ_API_KEY` — used by `nlp/topic_model.py` for Llama 3.3 reason extraction via Groq's OpenAI-compatible endpoint (raw HTTP, not SDK)

Optional:
- `HF_TOKEN` — HuggingFace API token. Used by `nlp/rag.py` to embed queries via the HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2`). Works without a token (anonymous) but authenticated requests get higher rate limits. Set this on Render for the `/ask` endpoint.
- `WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD` — Webshare proxy credentials. When both are set, `youtube_fetcher.py` routes transcript requests through Webshare's rotating gateway (`p.webshare.io:80`) via `WebshareProxyConfig`, bypassing YouTube IP blocks. Note: the free Webshare tier is datacenter proxies, which YouTube may still block; residential proxies are more reliable.

`ingestion/youtube_fetcher.py` resolves network access in this order: Webshare proxy (if `WEBSHARE_PROXY_*` set) → `cookies.txt` (Netscape format at repo root) → Firefox/Chrome browser cookies → unauthenticated. `cookies.txt` is gitignored (contains session tokens).

## Architecture

Four-layer pipeline:

```
ingestion/ → nlp/ → db/ → (future) api/dashboard
```

**ingestion/**
- `channel_fetcher.py` — YouTube Data API v3 to list recent videos from a channel
- `youtube_fetcher.py` — `youtube_transcript_api` to pull full transcript text for a video ID

**nlp/**
- `ticker_extractor.py` — two-pass extraction: regex `$TICKER` format, then company name dictionary lookup. Ambiguous names (e.g. "uber", "intel") are in `AMBIGUOUS_TICKERS` and require explicit `$TICKER` to match. Also exports `sentence_mentions_ticker(sentence, ticker)` (same word-boundary logic), used by `store_transcript_segments` to flag `is_context` and by RAG to cite only genuine mentions.
- `chunker.py` — spaCy `en_core_web_sm` splits transcript into sentences; `get_sentences_for_ticker()` returns sentences mentioning the ticker plus a ±1 context window.
- `sentiment.py` — FinBERT (`ProsusAI/finbert`) scores each sentence; filters out sentences <8 words or confidence <0.6; applies recency weighting (`weight = 1 + 2*(i/n)`) so later sentences carry more influence.
- `topic_model.py` — sends ticker-relevant sentences to Groq (Llama 3.3 70B) to extract 3 short reason phrases explaining the sentiment. Sentences with <8 words are filtered; fewer than 3 remaining returns `["insufficient data"]` without calling the API.
- `backfill_reasons.py` — offline batch runner for `topic_model`. Reads sentences from `transcript_segments`, populates `ticker_reasons`, skips already-done pairs, sleeps 2s between calls for rate limits.
- `backfill_embeddings.py` — embeds every `transcript_segments` sentence with `all-MiniLM-L6-v2` (normalized) into `sentence_embeddings`. Resumable. Local-only (needs torch).
- `rag.py` — retrieval-augmented Q&A. `retrieve()` is hybrid: it reads any ticker (`extract_tickers`) and bull/bear intent (`_query_polarity` → sentiment label) named in the question and pre-filters to that subset (relaxing progressively so it never returns nothing), then ranks by cosine × creator credibility (`_creator_credibility`: weight = 0.5 + Wilson-bound beat-SPY, so proven creators surface first), deduped by sentence and filtered to `is_context = FALSE`. Each hit carries the creator's beat-SPY track record and date, which `answer()` puts in the context so Groq can weight proven creators and recent takes; returns top-k; `answer()` feeds them to Groq for a grounded answer with inline citations. Query embedding goes through the HF Inference API (torch-free, so the cloud serving layer stays light), falling back to local sentence-transformers. Vectors live in the separate `db/embeddings.duckdb`, which `retrieve()` ATTACHes; absent on the cloud deploy, so retrieval there returns nothing and Ask degrades gracefully.

**db/**
- `init.py` — DuckDB schema (7 tables + indexes). DB file lives at `db/finsignal.duckdb`. Each function opens a new connection via `get_connection()`.
- `storage.py` — upsert helpers and canned queries: top bullish/bearish tickers, ticker trend over time, creators mentioning a ticker, cross-creator consensus scores, reasons by ticker, per-creator backtest accuracy, the creator-screening leaderboard (`get_screening_leaderboard`: beat-SPY rate ranked by Wilson lower bound, with a min-calls eligibility gate), `promote_creators` (flips `candidate`→`tracked` when a creator's Wilson lower bound clears a threshold, default 0.5, with the same min-calls gate; called at the end of `backtest.py`), `get_out_of_sample_validation` (per-creator temporal hold-out: oldest-half in-sample vs newest-half out-of-sample beat-SPY rates, to check ranking durability), `delete_creator` (destructive roster maintenance: removes a creator and all rows hanging off their videos, children-first in FK order; raises on a non-unique name), `get_smart_money_view` (per ticker, crowd consensus vs "smart money" = consensus among `_proven_creators` (≥30 calls and beat SPY >50%), surfacing where proven creators diverge from the crowd), and `get_strategy_performance` (backtest equity curve of following the proven creators' bullish calls — monthly cohorts, long-only — vs holding SPY). Consensus/top-ticker scores are recency-decayed via `_recency_weight_sql` (half-life default 30d); the backtest/screening helpers never decay (a past call's correctness is a fact).

**main.py** — orchestrates the pipeline: for each channel, fetch recent videos, run `analyse_video()` per video which calls ingestion → NLP → storage in sequence. 5-second sleep between videos to avoid rate limits.

**api/**
- `main.py` — FastAPI app, a thin read layer over `db/storage.py`. Endpoints: `/consensus`, `/tickers/top`, `/tickers/{ticker}/trend`, `/tickers/{ticker}/creators`, `/reasons`, `/creators/accuracy`, `/ask`, `/screen`, `/validate`. DataFrame results are round-tripped through `df.to_json` to native JSON types; tuple results are mapped to named dicts. `/ask` lazy-imports `nlp.rag` so the lightweight cloud deploy (no torch) still imports and serves the other endpoints; it degrades to an "unavailable" message there. Holds the DuckDB file, so don't run it alongside the pipeline.

**backtest.py** (repo root) — creator accuracy backtest. For each non-neutral `ticker_sentiments` call, fetches the stock's price at the video date vs `HORIZON_DAYS` (30) later via yfinance, scores raw-direction correctness (bullish=price rose, bearish=price fell), and stores per-call rows in `backtest_results`. Also fetches SPY once and records each call's benchmark return and whether it beat the benchmark (`beat_benchmark`) — the metric the Phase 5 screening leaderboard ranks on. Skips calls whose horizon hasn't elapsed yet (not enough future data) and groups price fetches by ticker (one yfinance call per ticker, not per call). Idempotent via upsert. Requires `yfinance`. After scoring, calls `promote_creators()` so any candidate that has earned its `tracked` status flips automatically.

**validate.py** (repo root) — out-of-sample validation runner over `get_out_of_sample_validation`. Read-only; splits each creator's benchmark-scored calls in half by date and prints whether in-sample beat-SPY skill persisted out-of-sample. Run after `backtest.py`.

## DuckDB schema

```
creators (id, channel_id UNIQUE, name, subscriber_count, status['candidate'|'tracked'], created_at)
  └── videos (id, creator_id FK, video_id UNIQUE, title, published_at, created_at)
        └── ticker_sentiments (id, video_id FK, ticker, label, directional_score, sentence_count)
              UNIQUE(video_id, ticker) — upserts on reprocess
        └── transcript_segments (id, video_id FK, ticker, sentence, label, score, is_context)
              is_context=TRUE marks ±1 context-window neighbors that don't themselves mention
              the ticker; RAG retrieves only is_context=FALSE so it never mis-cites them
        └── ticker_reasons (id, video_id FK, ticker, reason)
              populated offline by backfill_reasons.py; delete-then-insert per (video, ticker)
        └── backtest_results (id, video_id FK, ticker, horizon_days, call, return_pct, correct, benchmark_return_pct, beat_benchmark)
              UNIQUE(video_id, ticker, horizon_days) — populated by backtest.py, upserts on rerun

# Separate file: db/embeddings.duckdb (gitignored, local-only — see db/init.py)
sentence_embeddings (id, segment_id → transcript_segments.id, embedding FLOAT[384])
  UNIQUE(segment_id) — populated by backfill_embeddings.py; rag.py ATTACHes this DB and
  joins segment_id back to transcript_segments. Kept out of the main DB because the vectors
  are ~50MB (pushed the committed DB past GitHub's 100MB limit) and the cloud deploy never
  reads them. Regenerable; cross-file so there's no enforced FK.
```

`directional_score` is signed: positive = bullish, negative = bearish (range roughly −1 to +1). `ticker_sentiments.label` and `score` reflect the per-video aggregate; `transcript_segments` stores one row per raw sentence for drill-down.

## Key design decisions

- **Sentence-level attribution**: avoids sentiment bleed between tickers; each ticker only scores sentences that mention it (plus context window).
- **Recency weighting**: later sentences in a video carry more weight to capture the creator's current position rather than historical views mentioned early.
- **0.6 confidence threshold**: FinBERT predictions below this are discarded as noise.
- **Ambiguous tickers**: single-word company names that appear in everyday English are excluded from dictionary lookup and only matched via explicit `$TICKER` format.
- **Known limitation**: FinBERT has no tense awareness — "I was bearish but now I'm bullish" treats both clauses equally, pulling scores toward neutral. Phase 2 plans temporal marker detection to address this.
