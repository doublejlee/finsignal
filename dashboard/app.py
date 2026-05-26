import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.init import get_connection

st.set_page_config(page_title="FinSignal", layout="wide")
st.title("FinSignal — Financial Sentiment Dashboard")

conn = get_connection()

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