from ingestion.youtube_fetcher import get_transcript
from nlp.ticker_extractor import extract_tickers
from nlp.sentiment import analyse_sentiment_for_ticker
from db.storage import get_or_create_creator, get_or_create_video, store_ticker_sentiment, store_transcript_segments
from db.init import init_schema
from datetime import datetime
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
    from ingestion.channel_fetcher import get_recent_videos
    
    init_schema()
    
    # Channels to track
    channels = [
    {"channel_id": "UCUvvj5lwue7PspotMDjk5UA", "name": "Meet Kevin"},
    {"channel_id": "UCGy7SkBjcIAgTiwkXEtPnYg", "name": "Andrei Jikh"},
    {"channel_id": "UCbta0n8i6Rljh0obO7HzG9A", "name": "Joseph Carlson"},
]
    
    for channel in channels:
        print(f"\n{'='*50}")
        print(f"Processing channel: {channel['name']}")
        print(f"{'='*50}")
        
        videos = get_recent_videos(channel["channel_id"], max_results=2)
        
        
        for video in videos:
            try:
             analyse_video(video["video_id"],
                           channel_id=channel["channel_id"],
                           creator_name=channel["name"],
                           title=video["title"],
                           published_at=video["published_at"]
                           )
             time.sleep(5)  # wait 5 seconds between videos
            except Exception as e:
             print(f"Skipping {video['video_id']}: {e}")
             continue