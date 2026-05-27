"""Offline reason extraction. Run with VPN OFF (Groq blocks VPN IPs).

Reads stored sentences from transcript_segments, calls Groq per (video, ticker),
and writes results to ticker_reasons. Skips pairs that already have reasons, so
reruns are cheap and resumable after a rate-limit or crash.
"""
import time
from db.init import get_connection
from db.storage import store_ticker_reasons
from nlp.topic_model import extract_reasons

def backfill():
    conn = get_connection()

    pairs = conn.execute("""
        SELECT DISTINCT video_id, ticker FROM transcript_segments
    """).fetchall()

    done = set(conn.execute("""
        SELECT DISTINCT video_id, ticker FROM ticker_reasons
    """).fetchall())

    todo = [p for p in pairs if p not in done]
    print(f"{len(pairs)} ticker mentions, {len(done)} already done, {len(todo)} to process\n")

    for video_id, ticker in todo:
        rows = conn.execute(
            "SELECT sentence FROM transcript_segments WHERE video_id = ? AND ticker = ?",
            [video_id, ticker]
        ).fetchall()
        sentences = [r[0] for r in rows]

        try:
            reasons = extract_reasons(sentences, ticker=ticker)
            store_ticker_reasons(video_id, ticker, reasons)
            print(f"  {ticker} (video {video_id}): {', '.join(reasons)}")
        except Exception as e:
            print(f"  {ticker} (video {video_id}) FAILED: {e}")

        time.sleep(2)  # stay under Groq free-tier rate limit (~30/min)

if __name__ == "__main__":
    backfill()
