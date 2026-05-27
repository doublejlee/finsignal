import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.init import get_connection
from db.storage import get_consensus_scores, get_reasons_by_ticker, get_creator_accuracy, get_screening_leaderboard

st.set_page_config(page_title="FinSignal", layout="wide")
st.title("FinSignal — Financial Sentiment Dashboard")

conn = get_connection()

# Ask FinSignal (RAG)
st.subheader("Ask FinSignal")
st.caption("Ask in plain English — answered from creator transcripts, with sources")
question = st.text_input("e.g. Why is Netflix a good investment?", key="ask")
if question:
    with st.spinner("Thinking..."):
        try:
            from nlp.rag import answer  # lazy: needs the embedding model (local only for now)
            result = answer(question)
            st.markdown(result["answer"])
            with st.expander(f"Sources ({len(result['citations'])})"):
                for h in result["citations"]:
                    st.markdown(f"**[{h['number']}]** {h['creator']} on **{h['ticker']}** — {h['sentence']}")
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
st.caption("Aggregated across all creators — ordered by signal strength")
consensus = get_consensus_scores(min_creators=1)
if consensus.empty:
    st.info("No consensus data yet — run the pipeline on more videos")
else:
    st.dataframe(consensus, use_container_width=True)

st.divider()

# Reasons behind sentiment
st.subheader("Why? — Reasons Behind Sentiment")
reasons_df = get_reasons_by_ticker()
if reasons_df.empty:
    st.info("No reasons yet — run `python -m nlp.backfill_reasons` (VPN off)")
else:
    ticker = st.selectbox("Select a ticker", reasons_df["ticker"].tolist())
    row = reasons_df[reasons_df["ticker"] == ticker].iloc[0]
    for reason in row["reasons"].split(", "):
        st.markdown(f"- {reason}")

st.divider()

# Top bullish tickers
st.subheader("Top Bullish Tickers")
bullish = conn.execute("""
    SELECT ticker, SUM(directional_score) as total_score, COUNT(*) as video_count
    FROM ticker_sentiments
    WHERE directional_score > 0
    GROUP BY ticker
    ORDER BY total_score DESC
    LIMIT 10
""").fetchdf()

if bullish.empty:
    st.info("No bullish data yet — run the pipeline on more videos")
else:
    st.dataframe(bullish)

# Top bearish tickers
st.subheader("Top Bearish Tickers")
bearish = conn.execute("""
    SELECT ticker, SUM(directional_score) as total_score, COUNT(*) as video_count
    FROM ticker_sentiments
    WHERE directional_score < 0
    GROUP BY ticker
    ORDER BY total_score ASC
    LIMIT 10
""").fetchdf()

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