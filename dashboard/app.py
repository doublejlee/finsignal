import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from db.init import get_connection
from db.storage import (get_consensus_scores, get_top_tickers, get_reasons_by_ticker,
                        get_creator_accuracy, get_screening_leaderboard)

st.set_page_config(page_title="FinSignal", layout="wide")
st.title("FinSignal — Financial Sentiment Dashboard")

conn = get_connection()

# Recency control for the "current view" surfaces (consensus + top tickers). Opinions decay:
# a take loses half its weight every HALF_LIFE days, so the board reflects the current
# stance, not months-old calls. (Backtest/screening are deliberately NOT decayed.)
HALF_LIFE = st.sidebar.slider("Opinion recency half-life (days)", 7, 180, 30, step=1,
                              help="Older takes are down-weighted by 0.5 every N days. "
                                   "Backtest & screening are unaffected — they measure historical skill.")
_latest = conn.execute("SELECT MAX(published_at) FROM videos").fetchone()[0]
if _latest:
    st.sidebar.caption(f"Freshest take as of **{str(_latest)[:10]}**")

# Ask FinSignal (RAG)
st.subheader("Ask FinSignal")
st.caption("Ask in plain English — retrieval is filtered by ticker/sentiment and weighted "
           "by each creator's measured track record")
question = st.text_input("e.g. Why is Netflix a good investment?", key="ask")
if question:
    with st.spinner("Thinking..."):
        try:
            from nlp.rag import answer  # lazy: needs the embedding model (local only for now)
            result = answer(question)
            st.markdown(result["answer"])
            with st.expander(f"Sources ({len(result['citations'])})"):
                for h in result["citations"]:
                    track = (f" · _{h['creator_beat_spy']}% beat-SPY_"
                             if h.get("creator_beat_spy") is not None else "")
                    date = f" · {h['date']}" if h.get("date") else ""
                    st.markdown(f"**[{h['number']}]** {h['creator']} on **{h['ticker']}**"
                                f"{track}{date} — {h['sentence']}")
        except ImportError:
            st.info("Ask FinSignal runs locally only for now (the cloud deploy omits the embedding model).")
        except Exception as e:
            st.error(f"Couldn't answer: {e}")

st.divider()

# Creator accuracy
st.subheader("Creator Accuracy (30-day horizon)")
st.caption("Hit rate of each creator's directional calls vs actual price movement 30 days later")
accuracy = get_creator_accuracy(horizon_days=30)
if accuracy.empty:
    st.info("No backtest data yet — run `python backtest.py`")
else:
    st.dataframe(accuracy, use_container_width=True)

st.divider()

# Creator screening
st.subheader("Creator Screening")
st.caption("Candidates ranked by the Wilson lower-bound of their beat-SPY rate "
           "(conservative — penalizes small samples; needs ≥30 calls to be eligible)")
screening = get_screening_leaderboard()
if screening.empty:
    st.info("No screening data yet — run `python backtest.py`")
else:
    st.dataframe(screening, use_container_width=True)

st.divider()

# Consensus scores
st.subheader("Consensus Scores")
st.caption(f"Aggregated across all creators, recency-weighted ({HALF_LIFE}-day half-life) — "
           "ordered by current signal strength")
consensus = get_consensus_scores(min_creators=1, half_life_days=HALF_LIFE)
if consensus.empty:
    st.info("No consensus data yet — run the pipeline on more videos")
else:
    st.dataframe(consensus, use_container_width=True)

st.divider()

# Reasons behind sentiment — grouped by stance so bullish/bearish drivers don't blur together
st.subheader("Why? — Reasons Behind Sentiment")
reasons_df = get_reasons_by_ticker()
if reasons_df.empty:
    st.info("No reasons yet — run `python -m nlp.backfill_reasons` (VPN off)")
else:
    reason_ticker = st.selectbox("Select a ticker", sorted(reasons_df["ticker"].unique()))
    sub = reasons_df[reasons_df["ticker"] == reason_ticker]
    for col, (stance, heading) in zip(st.columns(3),
                                      [("bullish", "🟢 Bullish"), ("bearish", "🔴 Bearish"), ("neutral", "⚪ Neutral")]):
        items = sub[sub["stance"] == stance]["reason"].tolist()
        with col:
            st.markdown(f"**{heading}**")
            if items:
                for r in items:
                    st.markdown(f"- {r}")
            else:
                st.caption("—")

st.divider()

_top_cols = ["ticker", "total_score", "video_count"]

# Top bullish tickers (recency-weighted, same half-life as consensus)
st.subheader("Top Bullish Tickers")
st.caption(f"Recency-weighted ({HALF_LIFE}-day half-life)")
bullish = pd.DataFrame(get_top_tickers(limit=10, direction="bullish", half_life_days=HALF_LIFE),
                       columns=_top_cols)
if bullish.empty:
    st.info("No bullish data yet — run the pipeline on more videos")
else:
    st.dataframe(bullish)

# Top bearish tickers
st.subheader("Top Bearish Tickers")
st.caption(f"Recency-weighted ({HALF_LIFE}-day half-life)")
bearish = pd.DataFrame(get_top_tickers(limit=10, direction="bearish", half_life_days=HALF_LIFE),
                       columns=_top_cols)
if bearish.empty:
    st.info("No bearish data yet — run the pipeline on more videos")
else:
    st.dataframe(bearish)

# All sentiment data
st.subheader("All Ticker Sentiment")
all_data = conn.execute("""
    SELECT ts.ticker, ts.label, ts.directional_score, ts.sentence_count,
           c.name as creator, v.video_id
    FROM ticker_sentiments ts
    JOIN videos v ON ts.video_id = v.id
    JOIN creators c ON v.creator_id = c.id
    ORDER BY ts.directional_score DESC
""").fetchdf()

st.dataframe(all_data)