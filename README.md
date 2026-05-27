# FinSignal

AI-powered financial sentiment intelligence platform. FinSignal aggregates
finance-YouTube creator opinions, performs **sentence-level** sentiment
attribution using FinBERT, explains *why* each stock is bullish or bearish via
LLM-extracted reasons, and **backtests each creator's track record** against real
price movement.

## Live demo
- **API:** https://finsignal-api.onrender.com ([interactive docs](https://finsignal-api.onrender.com/docs)) — Render free tier, first request may cold-start (~30s)
- **Dashboard:** _Streamlit Community Cloud (add link)_

---

## Architecture

A four-layer pipeline:

```
ingestion/  →  nlp/  →  db/  →  dashboard/
 (YouTube)    (FinBERT,   (DuckDB)   (Streamlit)
              Groq)
```

- **Ingestion** — `youtube-transcript-api` pulls transcripts; YouTube Data API v3
  lists recent videos per channel.
- **Ticker extraction** — regex `$TICKER` + company-name dictionary with
  word-boundary matching and an ambiguity filter.
- **Sentiment** — FinBERT (`ProsusAI/finbert`) run at sentence level with recency
  weighting.
- **Chunking** — spaCy sentence splitting with a ±1 sentence context window per
  ticker mention.
- **Reasons** — ticker-relevant sentences sent to Groq (Llama 3.3 70B) to extract
  3 short phrases explaining the sentiment.
- **Backtesting** — `yfinance` prices compared against each call over a 30-day
  horizon to compute per-creator hit rate.
- **Ask (RAG)** — embeds stored sentences (`all-MiniLM-L6-v2`), retrieves those most
  relevant to a plain-English question, and asks Groq for an answer grounded in them
  with citations back to the source creator/ticker.
- **Screening** — scores each call against SPY (beat/lag), then ranks creators by the
  Wilson lower-bound of their beat-the-benchmark rate (conservative — penalizes small
  samples), with a minimum-calls eligibility gate.
- **Storage** — DuckDB (chosen over SQLite for analytical query performance).
- **API** — FastAPI read layer exposing the stored analytics as JSON endpoints.
- **Dashboard** — Streamlit leaderboards: consensus, bullish/bearish, reasons,
  creator accuracy.

---

## Stack

Python · FinBERT · HuggingFace Transformers · spaCy · sentence-transformers · DuckDB ·
FastAPI · Streamlit · Groq (Llama 3.3 70B) · yfinance · youtube-transcript-api ·
YouTube Data API v3

---

## Setup

Full local pipeline (ingestion, NLP, backtest, API, dashboard):

```bash
pip install -r requirements-pipeline.txt
python -m spacy download en_core_web_sm   # spaCy model isn't on PyPI
```

`.env` at the repo root needs `YOUTUBE_API_KEY` and `GROQ_API_KEY` (and optionally
`WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD`).

Three requirements files, one per deploy target (each minimal — the serving layers
only read the committed DuckDB and need none of the ML/ingestion stack):
- `requirements.txt` — Streamlit dashboard (Streamlit Community Cloud)
- `requirements-api.txt` — FastAPI backend (Render; see `render.yaml`)
- `requirements-pipeline.txt` — full local pipeline (ingestion, NLP, backtest)

## Running the pipeline

The pipeline runs in **three network-isolated stages** (see
[VPN vs Groq conflict](#problem-the-vpn-that-fixes-youtube-breaks-groq) for why):

```bash
# Stage 1 — ingestion + sentiment        (run with VPN ON; YouTube blocks bare IPs)
python main.py

# Stage 2 — reason extraction            (run with VPN OFF; Groq blocks VPN IPs)
python -m nlp.backfill_reasons

# Stage 3 — creator accuracy backtest     (yfinance prices vs each call)
python backtest.py

# API — serve stored analytics as JSON (interactive docs at http://localhost:8000/docs)
uvicorn api.main:app --reload

# Dashboard
streamlit run dashboard/app.py
```

`.env` requires `YOUTUBE_API_KEY` and `GROQ_API_KEY`. A `cookies.txt`
(Netscape format, gitignored) at the repo root authenticates YouTube requests.

---

## Key design decisions

- **Sentence-level attribution** over full-transcript scoring, to avoid sentiment
  bleed between tickers — each ticker only scores sentences that mention it (plus a
  ±1 context window).
- **Recency-weighted scoring** (`weight = 1 + 2·i/n`) so later sentences carry more
  influence, capturing the creator's *current* position rather than views expressed
  earlier in the video.
- **Ambiguous company names** (e.g. "uber", "intel") require explicit `$TICKER`
  format to avoid false positives.
- **Word-boundary ticker matching** instead of substring matching (see the NPHS bug
  below).
- **Decoupled pipeline stages** because ingestion and reason extraction have
  opposite network requirements.
- **DuckDB** for fast analytical aggregation queries.

---

## Progress

- [x] **Phase 0** — working CLI pipeline for a single video
- [x] **Phase 1** — full NLP pipeline, spaCy chunking, DuckDB storage, multi-creator
      ingestion, Streamlit dashboard
- [x] **Phase 2** — intelligence layer: reason extraction, consensus scoring,
      creator accuracy backtesting
- [x] **Phase 3** — FastAPI backend, split architecture (local ingestion + cloud
      serving), deployed dashboard (Streamlit Cloud) and API (Render)
- [~] **Phase 4** — "Ask FinSignal" RAG: semantic retrieval over creator sentences +
      grounded Groq answers with citations. Local MVP done; cloud serving (torch-free
      query embedding) pending.
- [~] **Phase 5** — Creator screening: benchmark-relative backtest + Wilson-bound
      ranking with an eligibility gate. Scoring/surfacing done; promotion and a deeper
      candidate pool (out-of-sample validation) pending.

---

## What Phase 2 delivered

**1. Reason extraction** — for each ticker, 3 short phrases explaining the
sentiment (e.g. NVDA → "AI datacenter demand", "Blackwell chip cycle"). Stored in a
new `ticker_reasons` table; surfaced in the dashboard via a ticker dropdown.

**2. Consensus scoring** — aggregates sentiment across all creators per ticker
(average score, creator count, bullish/bearish/neutral split) so multi-creator
agreement is visible at a glance.

**3. Creator accuracy backtesting** — the headline feature. For every non-neutral
call, FinSignal compares the stock's price on the video date against its price 30
days later and scores whether the creator was directionally right. Aggregated into
a per-creator hit rate.
*First result: Joseph Carlson — **62.9%** over 35 evaluated calls.* (Other creators'
videos are still younger than 30 days, so they populate as the data ages.)

New `backtest_results` table; results shown in a "Creator Accuracy" dashboard
section.

---

## Challenges & solutions

### Phase 0–1

**Problem: Full-transcript sentiment gave every ticker the same score.**
Running FinBERT on the whole transcript ignored which sentences were about which
ticker.
*Solution:* sentence-level attribution with a ±1 sentence context window per ticker
mention.

**Problem: Common words caused false-positive ticker matches.**
Words like "uber" and "intel" appear in everyday English, not just as company
references.
*Solution:* ambiguous tickers moved to a separate list requiring explicit `$TICKER`
format.

**Problem: Mixed-sentiment videos scored incorrectly.**
Creators reference past bearish views before explaining a current bullish position;
simple label counting weighted historical negativity equally.
*Solution:* recency-weighted scoring — later sentences carry more weight.

### Phase 2

**Problem: Reason extraction quality.**
- *Tried — BERTopic keyword extraction:* topic labels were low-quality on small data
  (few videos per ticker), producing noisy, hard-to-read keyword clusters.
- *Tried — Claude Haiku via raw HTTP:* the request was missing its auth headers
  (`x-api-key`, `anthropic-version`), so every call returned 401.
- *Solution — Groq (Llama 3.3 70B):* free tier, OpenAI-compatible endpoint, returns
  clean structured JSON. Produces concise, readable reasons. Works well.

<a name="problem-the-vpn-that-fixes-youtube-breaks-groq"></a>
**Problem: YouTube IP blocking.**
`youtube-transcript-api` started failing with `IpBlocked` — YouTube rate-limits
repeated requests from a single IP.
- *Tried — Chrome cookies via `browser_cookie3`:* failed on Windows with
  "Unable to get key for cookie decryption" — Chrome 127+ uses App-Bound Encryption
  that the library can't decrypt.
- *Tried — `cookies.txt` export (browser extension):* cookies loaded correctly, but
  requests still hit `IpBlocked`. This proved the block was **IP-level, not
  auth-level** — cookies alone could not fix it.
- *Solution — ProtonVPN:* switching exit IP cleared the block immediately.
  (Phase 3 will replace this manual step with rotating Webshare proxies.)

**Problem: The VPN that fixes YouTube breaks Groq.**
With the VPN on, Groq returned `403 Access denied` — it blocks VPN/datacenter exit
IPs. So YouTube *needs* a VPN and Groq *forbids* one: they can't run in the same
pass.
*Solution:* decoupled the pipeline into separate stages. `main.py` stores raw
sentences (VPN on); `nlp/backfill_reasons.py` reads them later and calls Groq (VPN
off). The backfill is resumable — it skips `(video, ticker)` pairs that already have
reasons, so a rate-limit or crash just means rerunning it.

**Problem: `youtube-transcript-api` changed its constructor across versions.**
The `cookies=` argument was removed; passing it raised `TypeError`.
*Solution:* the installed version takes an `http_client` (a `requests.Session`), so
cookies are loaded into a session that's passed in instead.

**Problem: Phantom ticker "NPHS" sent to the backtest.**
- *Detection:* `yfinance` returned `404 — Quote not found for symbol: NPHS`.
- *Root cause:* the ticker dictionary had a typo'd entry `"nphase": "NPHS"` (the real
  ticker is `ENPH`), and ticker matching used **substring** matching — so `"nphase"`
  matched *inside* the word "e‑nphase", emitting the phantom every time Enphase was
  discussed.
- *Solution:* removed the bogus entry and switched to **word-boundary matching**
  (`\b…\b`). This also eliminated a class of silent false positives (e.g.
  "metaverse" → META). Stale NPHS rows were purged from the database.

**Problem: `python -m db.init` crashed on Windows.**
The success message used a Unicode `✓` that the cp1252 console couldn't encode,
raising `UnicodeEncodeError` *after* the schema had already been created.
*Solution:* replaced it with plain ASCII text.

**Note — sparse backtest data (not a bug):** with a 30-day horizon, only videos
older than ~30 days can be evaluated. Most current data is recent, so early backtest
results are concentrated on the creator with the oldest videos. Coverage grows
naturally as videos age.

---

## Known limitations

- **Temporal sentiment conflation.** FinBERT has no sense of tense. "I *was* bearish
  but now I'm bullish" scores both clauses equally, pulling the result toward
  neutral. *Planned fix:* temporal-marker detection ("I used to", "previously") to
  separate historical sentiment from current stance.
- **Competitor mention false positives.** "Walmart competing with Amazon" can
  attribute sentiment to AMZN incorrectly.
- **YouTube IP blocking** is currently mitigated manually with a VPN +
  `time.sleep(5)` between videos. *Planned fix:* rotating Webshare proxies in
  Phase 3.
- **Small sample sizes** make per-ticker and per-creator metrics noisy until more
  videos are ingested.
