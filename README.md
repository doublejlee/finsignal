# FinSignal
AI-powered financial sentiment intelligence platform that aggregates creator 
opinions from YouTube, performs sentence-level sentiment attribution using 
FinBERT, and ranks stocks by weighted bullish/bearish consensus.

## Architecture
- **Ingestion**: youtube-transcript-api pulls transcripts from finance channels
- **Ticker extraction**: regex + company name dictionary with ambiguity filtering
- **Sentiment**: FinBERT (ProsusAI) run at sentence level with recency weighting
- **Chunking**: spaCy sentence splitting with ±2 sentence context window per ticker

## Key design decisions
- Sentence-level attribution chosen over full-transcript scoring to avoid 
  sentiment bleed between tickers
- Recency-weighted scoring to capture a creator's current position rather than 
  historical views expressed earlier in the video
- Ambiguous company names (e.g "uber", "intel") require explicit $TICKER format 
  to avoid false positives

## Stack
Python, FinBERT, HuggingFace Transformers, spaCy, pandas, youtube-transcript-api



## Progress
- [x] Phase 0 — working CLI pipeline for single video
- [ ] Phase 1 — full NLP pipeline with database
- [ ] Phase 2 — intelligence layer
- [ ] Phase 3 — API and dashboard

## Challenges & solutions

**Problem: Full-transcript sentiment gave every ticker the same score**  
Cause: Running FinBERT on the entire transcript ignored which sentences 
were about which ticker.  
Solution: Sentence-level attribution with a ±2 sentence context window 
per ticker mention.

**Problem: Common words causing false positive ticker matches**  
Cause: Words like "uber" and "intel" appear in everyday English, not just 
as company references.  
Solution: Ambiguous tickers moved to a separate list requiring explicit 
$TICKER format to match.

**Problem: Mixed-sentiment videos scoring incorrectly**  
Cause: Creators often reference past bearish views before explaining a 
current bullish position. Simple label counting treated historical 
negativity equally to current sentiment.  
Solution: Recency-weighted scoring where later sentences carry more weight, 
capturing the creator's current position rather than their historical view.

**Known limitation: Temporal sentiment conflation**  
Cause: FinBERT has no understanding of tense or time. A creator saying 
"I was bearish 2 years ago but now I'm bullish" scores the historical 
bearish statements equally to current ones, dragging down the final score.  
Impact: Mixed-history videos like opinion reversals score closer to neutral 
than their current stance warrants.  
Planned fix: Phase 2 topic clustering will detect temporal markers 
("I used to", "previously", "back then") to separate historical sentiment 
from current position.