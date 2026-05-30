"""Creator accuracy backtest.

For each non-neutral sentiment call, compare the stock's price at the video date
against its price HORIZON_DAYS later (raw direction). A bullish call is correct if
the price rose; a bearish call is correct if it fell. Calls whose horizon hasn't
elapsed yet (not enough future price data) are skipped. Each call is also scored
against SPY over the same window (benchmark_return_pct / beat_benchmark) for the
screening metric. Results land in backtest_results; rerun is idempotent.
"""
from collections import defaultdict
from datetime import date, timedelta
import yfinance as yf

from db.init import get_connection
from db.storage import store_backtest_result, promote_creators

HORIZON_DAYS = 30
NEUTRAL_THRESHOLD = 0.05

def _to_date(value):
    return value.date() if hasattr(value, "date") else value

def price_on_or_after(closes, target):
    """First available close on or after target date, or None if none exists."""
    for ts, price in closes.items():
        if _to_date(ts) >= target:
            return float(price)
    return None

def get_calls(conn):
    return conn.execute("""
        SELECT ts.video_id, ts.ticker, ts.directional_score, v.published_at
        FROM ticker_sentiments ts
        JOIN videos v ON ts.video_id = v.id
        WHERE ABS(ts.directional_score) > ?
          AND v.published_at IS NOT NULL
        ORDER BY ts.ticker, v.published_at
    """, [NEUTRAL_THRESHOLD]).fetchall()

def run():
    conn = get_connection()
    calls = get_calls(conn)

    by_ticker = defaultdict(list)
    for video_id, ticker, score, published_at in calls:
        by_ticker[ticker].append((video_id, score, _to_date(published_at)))

    today = date.today()
    evaluated = skipped = 0

    # Benchmark (SPY) prices over the full window, fetched once.
    spy_closes = None
    if by_ticker:
        earliest_all = min(c[2] for calls in by_ticker.values() for c in calls)
        try:
            spy_data = yf.download("SPY", start=earliest_all - timedelta(days=5),
                                   end=today + timedelta(days=1), progress=False, auto_adjust=True)
            if not spy_data.empty:
                spy_closes = spy_data["Close"]
                if hasattr(spy_closes, "columns"):
                    spy_closes = spy_closes["SPY"]
        except Exception as e:
            print(f"  SPY benchmark fetch failed ({e}); benchmark metrics will be null.")

    for ticker, ticker_calls in by_ticker.items():
        earliest = min(c[2] for c in ticker_calls)
        start = earliest - timedelta(days=5)
        end = today + timedelta(days=1)

        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  {ticker}: price fetch failed ({e}), skipping {len(ticker_calls)} calls")
            skipped += len(ticker_calls)
            continue

        if data.empty:
            print(f"  {ticker}: no price data, skipping {len(ticker_calls)} calls")
            skipped += len(ticker_calls)
            continue

        closes = data["Close"]
        if hasattr(closes, "columns"):  # multiindex when yfinance returns a frame
            closes = closes[ticker]

        for video_id, score, pub_date in ticker_calls:
            entry = price_on_or_after(closes, pub_date)
            exit_price = price_on_or_after(closes, pub_date + timedelta(days=HORIZON_DAYS))

            if entry is None or exit_price is None:
                skipped += 1
                continue

            call = "bullish" if score > 0 else "bearish"
            return_pct = (exit_price - entry) / entry * 100
            correct = (call == "bullish" and return_pct > 0) or (call == "bearish" and return_pct < 0)

            benchmark_return_pct = beat_benchmark = None
            if spy_closes is not None:
                spy_entry = price_on_or_after(spy_closes, pub_date)
                spy_exit = price_on_or_after(spy_closes, pub_date + timedelta(days=HORIZON_DAYS))
                if spy_entry is not None and spy_exit is not None:
                    benchmark_return_pct = (spy_exit - spy_entry) / spy_entry * 100
                    beat_benchmark = (
                        (call == "bullish" and return_pct > benchmark_return_pct)
                        or (call == "bearish" and return_pct < benchmark_return_pct)
                    )

            store_backtest_result(video_id, ticker, HORIZON_DAYS, call, return_pct, correct,
                                  benchmark_return_pct, beat_benchmark)
            evaluated += 1
            mark = "OK " if correct else "MISS"
            bench = f" vs SPY {benchmark_return_pct:+.1f}% {'beat' if beat_benchmark else 'lag'}" if benchmark_return_pct is not None else ""
            print(f"  [{mark}] {ticker:<6} {call:<8} {return_pct:+6.1f}%{bench} (video {video_id})")

    print(f"\nDone. {evaluated} calls evaluated, {skipped} skipped (horizon not elapsed or no data).")

    # Now that beat_benchmark is fresh, see if any candidate creator has earned tracking.
    promoted = promote_creators()
    if promoted:
        for p in promoted:
            print(f"  PROMOTED {p['creator']} -> tracked "
                  f"({p['calls']} calls, Wilson LB {p['wilson_lower_pct']}% beat SPY)")
    else:
        print("  No candidates cleared the promotion threshold.")

if __name__ == "__main__":
    run()
