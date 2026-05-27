from db.init import get_connection
from datetime import datetime

def get_or_create_creator(channel_id: str, name: str, subscriber_count: int = None) -> int:
    """Get or create creator, returns creator_id."""
    conn = get_connection()

    # Check if exists
    result = conn.execute(
        "SELECT id FROM creators WHERE channel_id = ?",
        [channel_id]
    ).fetchall()

    if result:
        return result[0][0]

    # Create new
    conn.execute(
        "INSERT INTO creators (channel_id, name, subscriber_count) VALUES (?, ?, ?)",
        [channel_id, name, subscriber_count]
    )
    conn.commit()

    result = conn.execute(
        "SELECT id FROM creators WHERE channel_id = ?",
        [channel_id]
    ).fetchall()

    return result[0][0]

def get_or_create_video(creator_id: int, video_id: str, title: str, published_at: datetime = None) -> int:
    """Get or create video, returns video_id (database id)."""
    conn = get_connection()

    result = conn.execute(
        "SELECT id FROM videos WHERE video_id = ?",
        [video_id]
    ).fetchall()

    if result:
        return result[0][0]

    conn.execute(
        "INSERT INTO videos (creator_id, video_id, title, published_at) VALUES (?, ?, ?, ?)",
        [creator_id, video_id, title, published_at]
    )
    conn.commit()

    result = conn.execute(
        "SELECT id FROM videos WHERE video_id = ?",
        [video_id]
    ).fetchall()

    return result[0][0]

def store_ticker_sentiment(video_id: int, ticker: str, label: str, directional_score: float, sentence_count: int):
    """Store aggregated ticker sentiment for a video."""
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO ticker_sentiments (video_id, ticker, label, directional_score, sentence_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(video_id, ticker) DO UPDATE SET
            label = excluded.label,
            directional_score = excluded.directional_score,
            sentence_count = excluded.sentence_count
        """,
        [video_id, ticker, label, directional_score, sentence_count]
    )
    conn.commit()

def store_transcript_segments(video_id: int, ticker: str, sentences: list, label: str, score: float):
    """
    Store raw transcript segments.
    sentences: list of sentence strings that mention the ticker
    """
    conn = get_connection()

    for sentence in sentences:
        conn.execute(
            """
            INSERT INTO transcript_segments (video_id, ticker, sentence, label, score)
            VALUES (?, ?, ?, ?, ?)
            """,
            [video_id, ticker, sentence, label, score]
        )

    conn.commit()

def store_ticker_reasons(video_id: int, ticker: str, reasons: list):
    """Store reason phrases for a ticker in a video. Replaces any existing reasons."""
    conn = get_connection()
    conn.execute(
        "DELETE FROM ticker_reasons WHERE video_id = ? AND ticker = ?",
        [video_id, ticker]
    )
    for reason in reasons:
        conn.execute(
            "INSERT INTO ticker_reasons (video_id, ticker, reason) VALUES (?, ?, ?)",
            [video_id, ticker, reason]
        )
    conn.commit()

def get_consensus_scores(min_creators: int = 1):
    """Aggregate sentiment across all creators per ticker."""
    conn = get_connection()
    return conn.execute("""
        SELECT
            ts.ticker,
            ROUND(AVG(ts.directional_score), 3)          AS avg_score,
            COUNT(DISTINCT c.id)                          AS creator_count,
            SUM(CASE WHEN ts.directional_score >  0.05 THEN 1 ELSE 0 END) AS bullish,
            SUM(CASE WHEN ts.directional_score < -0.05 THEN 1 ELSE 0 END) AS bearish,
            SUM(CASE WHEN ts.directional_score BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END) AS neutral
        FROM ticker_sentiments ts
        JOIN videos v ON ts.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
        GROUP BY ts.ticker
        HAVING COUNT(DISTINCT c.id) >= ?
        ORDER BY ABS(AVG(ts.directional_score)) DESC
    """, [min_creators]).fetchdf()

def store_backtest_result(video_id: int, ticker: str, horizon_days: int, call: str, return_pct: float, correct: bool):
    """Store one backtested call. Upserts on (video_id, ticker, horizon_days)."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO backtest_results (video_id, ticker, horizon_days, call, return_pct, correct)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, ticker, horizon_days) DO UPDATE SET
            call = excluded.call,
            return_pct = excluded.return_pct,
            correct = excluded.correct
        """,
        [video_id, ticker, horizon_days, call, return_pct, correct]
    )
    conn.commit()

def get_creator_accuracy(horizon_days: int = 30):
    """Per-creator hit rate across all backtested calls."""
    conn = get_connection()
    return conn.execute("""
        SELECT
            c.name AS creator,
            COUNT(*) AS calls_evaluated,
            SUM(CASE WHEN b.correct THEN 1 ELSE 0 END) AS correct,
            ROUND(100.0 * SUM(CASE WHEN b.correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS hit_rate_pct
        FROM backtest_results b
        JOIN videos v ON b.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
        WHERE b.horizon_days = ?
        GROUP BY c.name
        ORDER BY hit_rate_pct DESC
    """, [horizon_days]).fetchdf()

def get_reasons_by_ticker():
    """Distinct reason phrases per ticker, aggregated across all videos."""
    conn = get_connection()
    return conn.execute("""
        SELECT ticker, string_agg(reason, ', ') AS reasons
        FROM (
            SELECT DISTINCT ticker, reason
            FROM ticker_reasons
            WHERE reason <> 'insufficient data'
        )
        GROUP BY ticker
        ORDER BY ticker
    """).fetchdf()

def get_top_tickers(limit: int = 10, direction: str = "bullish") -> list:
    """Get top bullish or bearish tickers across all videos."""
    conn = get_connection()

    if direction == "bullish":
        query = """
            SELECT ticker, SUM(directional_score) as total_score, COUNT(*) as video_count
            FROM ticker_sentiments
            WHERE directional_score > 0
            GROUP BY ticker
            ORDER BY total_score DESC
            LIMIT ?
        """
    else:  # bearish
        query = """
            SELECT ticker, SUM(directional_score) as total_score, COUNT(*) as video_count
            FROM ticker_sentiments
            WHERE directional_score < 0
            GROUP BY ticker
            ORDER BY total_score ASC
            LIMIT ?
        """

    return conn.execute(query, [limit]).fetchall()

def get_ticker_sentiment_over_time(ticker: str):
    """Get sentiment trend for a ticker over time."""
    conn = get_connection()

    return conn.execute("""
        SELECT v.published_at, ts.label, ts.directional_score, ts.sentence_count
        FROM ticker_sentiments ts
        JOIN videos v ON ts.video_id = v.id
        WHERE ts.ticker = ?
        ORDER BY v.published_at ASC
    """, [ticker]).fetchall()

def get_creators_mentioning_ticker(ticker: str):
    """Get creators who mentioned a ticker and their sentiment."""
    conn = get_connection()

    return conn.execute("""
        SELECT DISTINCT c.name, c.channel_id, ts.label, ts.directional_score
        FROM ticker_sentiments ts
        JOIN videos v ON ts.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
        WHERE ts.ticker = ?
        ORDER BY ts.directional_score DESC
    """, [ticker]).fetchall()
