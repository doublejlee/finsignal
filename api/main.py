"""FastAPI backend for FinSignal.

Thin read layer over db/storage.py — exposes the existing analytical queries as
JSON endpoints. Run from the repo root:

    uvicorn api.main:app --reload

Note: like the Streamlit dashboard, this opens the DuckDB file, so don't run it at
the same time as the ingestion/backtest pipeline (DuckDB allows a single writer).
"""
import json

from fastapi import FastAPI, Query

from db import storage

app = FastAPI(title="FinSignal API", version="0.1.0")

def _records(df):
    """DataFrame -> JSON-safe list of dicts (round-trips numpy types to native)."""
    return json.loads(df.to_json(orient="records"))

@app.get("/")
def root():
    return {"service": "FinSignal API", "version": "0.1.0"}

@app.get("/consensus")
def consensus(min_creators: int = Query(1, ge=1)):
    return _records(storage.get_consensus_scores(min_creators=min_creators))

@app.get("/tickers/top")
def top_tickers(
    direction: str = Query("bullish", pattern="^(bullish|bearish)$"),
    limit: int = Query(10, ge=1, le=100),
):
    rows = storage.get_top_tickers(limit=limit, direction=direction)
    return [{"ticker": t, "total_score": s, "video_count": v} for t, s, v in rows]

@app.get("/tickers/{ticker}/trend")
def ticker_trend(ticker: str):
    rows = storage.get_ticker_sentiment_over_time(ticker.upper())
    return [
        {"published_at": p, "label": l, "directional_score": d, "sentence_count": n}
        for p, l, d, n in rows
    ]

@app.get("/tickers/{ticker}/creators")
def ticker_creators(ticker: str):
    rows = storage.get_creators_mentioning_ticker(ticker.upper())
    return [
        {"creator": name, "channel_id": ch, "label": l, "directional_score": d}
        for name, ch, l, d in rows
    ]

@app.get("/reasons")
def reasons():
    return _records(storage.get_reasons_by_ticker())

@app.get("/creators/accuracy")
def creator_accuracy(horizon_days: int = Query(30, ge=1)):
    return _records(storage.get_creator_accuracy(horizon_days=horizon_days))

@app.get("/screen")
def screen(min_calls: int = Query(30, ge=1)):
    return _records(storage.get_screening_leaderboard(min_calls=min_calls))

@app.get("/validate")
def validate(min_per_split: int = Query(10, ge=1)):
    return _records(storage.get_out_of_sample_validation(min_per_split=min_per_split))

@app.get("/ask")
def ask(q: str = Query(..., min_length=3)):
    # Lazy import: rag pulls in sentence-transformers/torch, which the lightweight
    # cloud deploy doesn't install. Keeps the rest of the API working there.
    try:
        from nlp.rag import answer
    except ImportError:
        return {"answer": "Ask is unavailable in this deployment (no embedding model installed).", "citations": []}
    return answer(q)
