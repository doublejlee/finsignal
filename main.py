import sys
sys.stdout.reconfigure(encoding="utf-8")  # titles can contain emoji; avoid cp1252 console crashes

from ingestion.youtube_fetcher import get_transcript
from nlp.ticker_extractor import extract_tickers
from nlp.sentiment import analyse_sentiment_for_ticker
from db.storage import get_or_create_creator, get_or_create_video, store_ticker_sentiment, store_transcript_segments
from db.init import init_schema
from datetime import datetime, timedelta, timezone
import time

def directional_score(label, score):
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0

def analyse_video(video_id, channel_id="unknown", creator_name="unknown", title="unknown", published_at=None):
    if isinstance(published_at, str):
        published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    date_str = published_at.strftime("%Y-%m-%d") if published_at else "unknown date"
    print(f"\n{title}")
    print(f"{date_str} | {video_id}")
    print("Fetching transcript...")
    text = get_transcript(video_id)

    print("Extracting tickers...")
    tickers = extract_tickers(text)
    print(f"Found {len(tickers)} tickers: {tickers}\n")

    creator_id = get_or_create_creator(channel_id, creator_name)
    db_video_id = get_or_create_video(creator_id, video_id, title=title, published_at=published_at)

    results = []
    for ticker in tickers:
        label, score, sentences = analyse_sentiment_for_ticker(text, ticker)
        d_score = directional_score(label, score)

        store_ticker_sentiment(db_video_id, ticker, label, d_score, len(sentences))
        store_transcript_segments(db_video_id, ticker, sentences, label, score)

        results.append({
            "ticker": ticker,
            "sentiment": label,
            "score": score,
            "directional_score": d_score,
            "sentences_analysed": len(sentences)
        })

    results.sort(key=lambda x: x["directional_score"], reverse=True)

    print(f"{'TICKER':<8} {'SENTIMENT':<12} {'SCORE':<10} {'SENTENCES'}")
    print("-" * 50)
    for r in results:
        print(f"{r['ticker']:<8} {r['sentiment']:<12} {r['directional_score']:<10.3f} {r['sentences_analysed']}")

    return results


if __name__ == "__main__":
    from ingestion.channel_fetcher import get_recent_videos, resolve_handle

    init_schema()

    # Creators to track, by YouTube @handle (resolved to channel IDs at runtime).
    HANDLES = [
        "@MeetKevin",
        "@AndreiJikh",
        "@JosephCarlsonShow",
        "@everythingmoney",
        "@ChipStockInvestor",
        "@FinancialEducation",  # Jeremy Lefebvre
        "@tomnashtv",           # Tom Nash (668K, Stock-MVP); @TomNash is a dormant 2010 channel, not him
    ]

    # Videos pulled per channel.
    MAX_VIDEOS_PER_CHANNEL = 15

    # Only pull videos already past the backtest's 30-day horizon (+1-day buffer), so every
    # ingested video is immediately evaluable. Frequent posters (e.g. Meet Kevin) otherwise
    # yield only un-scorable last-week uploads. Newer videos get picked up on later runs as
    # they age past the horizon. Keep this in sync with backtest.HORIZON_DAYS.
    BACKTEST_HORIZON_DAYS = 30
    cutoff = datetime.now(timezone.utc) - timedelta(days=BACKTEST_HORIZON_DAYS + 1)
    published_before = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Pulling up to {MAX_VIDEOS_PER_CHANNEL} videos per channel published before "
          f"{published_before} (evaluable history).")

    for handle in HANDLES:
        print(f"\n{'='*50}")
        print(f"Processing channel: {handle}")
        print(f"{'='*50}")

        try:
            channel_id = resolve_handle(handle)
        except Exception as e:
            print(f"Skipping {handle}: {e}")
            continue

        videos = get_recent_videos(channel_id, max_results=MAX_VIDEOS_PER_CHANNEL,
                                   published_before=published_before)
        if not videos:
            print(f"No videos found for {handle}")
            continue

        creator_name = videos[0]["channel_name"]

        for video in videos:
            try:
                analyse_video(video["video_id"],
                              channel_id=channel_id,
                              creator_name=creator_name,
                              title=video["title"],
                              published_at=video["published_at"])
                time.sleep(5)  # wait 5 seconds between videos
            except Exception as e:
                print(f"Skipping {video['video_id']}: {e}")
                continue