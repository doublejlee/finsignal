from db.init import get_connection
from datetime import datetime
import math
import pandas as pd

MIN_ELIGIBLE_CALLS = 30

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

def delete_creator(name: str) -> dict:
    """Delete a creator and all rows that hang off them, children-first (DuckDB enforces FKs).

    Cascade: sentence_embeddings -> transcript_segments / ticker_sentiments / ticker_reasons /
    backtest_results (all keyed on the creator's videos) -> videos -> the creator row.
    Matches on exact creator name; raises if it isn't unique so a typo can't wipe the wrong
    creator. Returns per-table delete counts. Destructive — the DB is in git, so a bad call
    is recoverable from history.
    """
    conn = get_connection()
    ids = conn.execute("SELECT id FROM creators WHERE name = ?", [name]).fetchall()
    if not ids:
        raise ValueError(f"No creator named {name!r}")
    if len(ids) > 1:
        raise ValueError(f"{len(ids)} creators named {name!r}; refusing to guess which to delete")
    creator_id = ids[0][0]

    video_ids = [r[0] for r in conn.execute(
        "SELECT id FROM videos WHERE creator_id = ?", [creator_id]).fetchall()]

    deleted = {}
    if video_ids:
        placeholders = ",".join("?" * len(video_ids))
        deleted["sentence_embeddings"] = conn.execute(
            f"DELETE FROM sentence_embeddings WHERE segment_id IN "
            f"(SELECT id FROM transcript_segments WHERE video_id IN ({placeholders}))",
            video_ids).fetchall()
        for tbl in ["transcript_segments", "ticker_sentiments", "ticker_reasons", "backtest_results"]:
            conn.execute(f"DELETE FROM {tbl} WHERE video_id IN ({placeholders})", video_ids)
    conn.execute("DELETE FROM videos WHERE creator_id = ?", [creator_id])
    conn.execute("DELETE FROM creators WHERE id = ?", [creator_id])
    conn.commit()

    return {"creator": name, "creator_id": creator_id, "videos_deleted": len(video_ids)}

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

def _recency_weight_sql(half_life_days: float) -> str:
    """SQL expression for an exponential recency weight on a video: 0.5 ** (age_days/half).

    A take loses half its weight every `half_life_days`, so a stale opinion counts less
    toward the *current* view than a fresh one. Undated videos (NULL published_at) are
    treated as very old (≈0 weight). half_life_days is cast to float by the caller, so
    inlining it is injection-safe. NB: this is for opinion/consensus aggregation only —
    the backtest must never recency-weight a creator's calls (a past call's correctness is
    a fact, and decaying it would discard out-of-sample evidence and shrink the sample).
    """
    hl = float(half_life_days)
    age = "COALESCE(date_diff('day', v.published_at, CURRENT_TIMESTAMP), 9999)"
    return f"pow(0.5, {age} / {hl})"

def get_consensus_scores(min_creators: int = 1, half_life_days: float = 30.0):
    """Aggregate sentiment across all creators per ticker.

    When `half_life_days` is set (default 30), `avg_score` is a recency-weighted mean so a
    months-old take doesn't read the same as last week's. Pass `half_life_days=None` for a
    plain average. Counts (creator_count, bullish/bearish/neutral) stay raw — they describe
    total evidence; only the score is time-decayed.
    """
    conn = get_connection()
    if half_life_days:
        w = _recency_weight_sql(half_life_days)
        avg_expr = f"ROUND(SUM(({w}) * ts.directional_score) / NULLIF(SUM({w}), 0), 3)"
    else:
        avg_expr = "ROUND(AVG(ts.directional_score), 3)"

    return conn.execute(f"""
        SELECT
            ts.ticker,
            {avg_expr}                                    AS avg_score,
            COUNT(DISTINCT c.id)                          AS creator_count,
            SUM(CASE WHEN ts.directional_score >  0.05 THEN 1 ELSE 0 END) AS bullish,
            SUM(CASE WHEN ts.directional_score < -0.05 THEN 1 ELSE 0 END) AS bearish,
            SUM(CASE WHEN ts.directional_score BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END) AS neutral
        FROM ticker_sentiments ts
        JOIN videos v ON ts.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
        GROUP BY ts.ticker
        HAVING COUNT(DISTINCT c.id) >= ?
        ORDER BY ABS(avg_score) DESC
    """, [min_creators]).fetchdf()

def store_backtest_result(video_id: int, ticker: str, horizon_days: int, call: str, return_pct: float, correct: bool,
                          benchmark_return_pct: float = None, beat_benchmark: bool = None):
    """Store one backtested call. Upserts on (video_id, ticker, horizon_days)."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO backtest_results
            (video_id, ticker, horizon_days, call, return_pct, correct, benchmark_return_pct, beat_benchmark)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, ticker, horizon_days) DO UPDATE SET
            call = excluded.call,
            return_pct = excluded.return_pct,
            correct = excluded.correct,
            benchmark_return_pct = excluded.benchmark_return_pct,
            beat_benchmark = excluded.beat_benchmark
        """,
        [video_id, ticker, horizon_days, call, return_pct, correct, benchmark_return_pct, beat_benchmark]
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

def _wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval — a conservative estimate of the
    true success rate that penalizes small samples (so 5/7 doesn't outrank 120/200)."""
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / (1 + z * z / n))

def get_screening_leaderboard(min_calls: int = MIN_ELIGIBLE_CALLS):
    """Rank creators by the Wilson lower-bound of their beat-the-benchmark rate.

    Only calls scored against the benchmark (beat_benchmark NOT NULL) count. Creators
    with fewer than min_calls eligible calls are shown but flagged not-eligible.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT cr.name, cr.status,
               COUNT(*) AS n,
               SUM(CASE WHEN b.beat_benchmark THEN 1 ELSE 0 END) AS beats
        FROM backtest_results b
        JOIN videos v ON b.video_id = v.id
        JOIN creators cr ON v.creator_id = cr.id
        WHERE b.beat_benchmark IS NOT NULL
        GROUP BY cr.name, cr.status
    """).fetchall()

    records = []
    for name, status, n, beats in rows:
        beats = int(beats or 0)
        records.append({
            "creator": name,
            "status": status,
            "calls": n,
            "beat_spy_pct": round(100.0 * beats / n, 1) if n else 0.0,
            "wilson_lower_pct": round(100.0 * _wilson_lower_bound(beats, n), 1),
            "eligible": n >= min_calls,
        })

    df = pd.DataFrame(records, columns=["creator", "status", "calls", "beat_spy_pct", "wilson_lower_pct", "eligible"])
    if not df.empty:
        df = df.sort_values(["eligible", "wilson_lower_pct"], ascending=[False, False]).reset_index(drop=True)
    return df

def promote_creators(threshold: float = 0.5, min_calls: int = MIN_ELIGIBLE_CALLS) -> list:
    """Flip 'candidate' creators to 'tracked' once they prove out on the backtest.

    A candidate is promoted when it has at least `min_calls` benchmark-scored calls and
    its Wilson lower-bound beat-SPY rate clears `threshold` — the same conservative metric
    get_screening_leaderboard ranks on. The threshold of 0.5 means we promote only when we
    can say, with 95% confidence, the creator beats SPY more often than not. Promotion is
    one-way here; demotion is left to manual review.

    Returns a list of dicts (one per creator promoted in this run); empty if none qualified.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT cr.id, cr.name,
               COUNT(*) AS n,
               SUM(CASE WHEN b.beat_benchmark THEN 1 ELSE 0 END) AS beats
        FROM backtest_results b
        JOIN videos v ON b.video_id = v.id
        JOIN creators cr ON v.creator_id = cr.id
        WHERE b.beat_benchmark IS NOT NULL AND cr.status = 'candidate'
        GROUP BY cr.id, cr.name
    """).fetchall()

    promoted = []
    for creator_id, name, n, beats in rows:
        beats = int(beats or 0)
        if n < min_calls:
            continue
        wilson = _wilson_lower_bound(beats, n)
        if wilson >= threshold:
            conn.execute("UPDATE creators SET status = 'tracked' WHERE id = ?", [creator_id])
            promoted.append({
                "creator": name,
                "calls": n,
                "beat_spy_pct": round(100.0 * beats / n, 1),
                "wilson_lower_pct": round(100.0 * wilson, 1),
            })

    if promoted:
        conn.commit()
    return promoted

def get_out_of_sample_validation(split_fraction: float = 0.5, min_per_split: int = 10):
    """Temporal hold-out check: does in-sample beat-SPY skill persist out-of-sample?

    For each creator, benchmark-scored calls are ordered by video date and split at
    `split_fraction` (default: oldest 50% = in-sample 'train', newest 50% = out-of-sample
    'holdout'). We report the beat-SPY rate and Wilson lower bound for each half. A creator
    who screens well in-sample but collapses out-of-sample is a sign the leaderboard ranking
    reflects luck rather than durable skill — the whole point of screening on a hold-out.

    Splitting per creator (rather than on one global cutoff date) keeps both halves balanced
    despite uneven histories. Creators with fewer than `min_per_split` calls in either half
    are excluded — the split can't say anything reliable about them. Returns a DataFrame,
    one row per evaluable creator.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT cr.name,
               CASE WHEN b.beat_benchmark THEN 1 ELSE 0 END AS beat
        FROM backtest_results b
        JOIN videos v ON b.video_id = v.id
        JOIN creators cr ON v.creator_id = cr.id
        WHERE b.beat_benchmark IS NOT NULL AND v.published_at IS NOT NULL
        ORDER BY cr.name, v.published_at
    """).fetchall()

    by_creator = {}
    for name, beat in rows:
        by_creator.setdefault(name, []).append(int(beat))

    def _summary(arr):
        k, m = sum(arr), len(arr)
        return m, round(100.0 * k / m, 1), round(100.0 * _wilson_lower_bound(k, m), 1)

    records = []
    for name, beats in by_creator.items():
        cut = int(len(beats) * split_fraction)
        in_sample, oos = beats[:cut], beats[cut:]
        if len(in_sample) < min_per_split or len(oos) < min_per_split:
            continue
        is_m, is_pct, is_wlb = _summary(in_sample)
        oos_m, oos_pct, oos_wlb = _summary(oos)
        records.append({
            "creator": name,
            "in_sample_calls": is_m,
            "in_sample_beat_pct": is_pct,
            "in_sample_wilson_pct": is_wlb,
            "oos_calls": oos_m,
            "oos_beat_pct": oos_pct,
            "oos_wilson_pct": oos_wlb,
            "delta_pct": round(oos_pct - is_pct, 1),
            "persisted": oos_pct >= 50.0,
        })

    df = pd.DataFrame(records, columns=[
        "creator", "in_sample_calls", "in_sample_beat_pct", "in_sample_wilson_pct",
        "oos_calls", "oos_beat_pct", "oos_wilson_pct", "delta_pct", "persisted",
    ])
    if not df.empty:
        df = df.sort_values("in_sample_wilson_pct", ascending=False).reset_index(drop=True)
    return df

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

def get_top_tickers(limit: int = 10, direction: str = "bullish", half_life_days: float = 30.0) -> list:
    """Get top bullish or bearish tickers across all videos.

    A mention is classed bullish/bearish by its raw sentiment; its contribution to the
    ranking score is recency-weighted when `half_life_days` is set (default 30), so fresh
    takes dominate the board. `video_count` stays a raw count. `half_life_days=None` sums
    raw scores.
    """
    conn = get_connection()
    score = f"({_recency_weight_sql(half_life_days)}) * ts.directional_score" if half_life_days \
        else "ts.directional_score"
    where, order = ("ts.directional_score > 0", "DESC") if direction == "bullish" \
        else ("ts.directional_score < 0", "ASC")

    return conn.execute(f"""
        SELECT ts.ticker, SUM({score}) AS total_score, COUNT(*) AS video_count
        FROM ticker_sentiments ts
        JOIN videos v ON ts.video_id = v.id
        WHERE {where}
        GROUP BY ts.ticker
        ORDER BY total_score {order}
        LIMIT ?
    """, [limit]).fetchall()

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
