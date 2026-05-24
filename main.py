from ingestion.youtube_fetcher import get_transcript
from nlp.ticker_extractor import extract_tickers
from nlp.sentiment import analyse_sentiment_for_ticker
from db.storage import get_or_create_creator, get_or_create_video, store_ticker_sentiment, store_transcript_segments
from db.init import init_schema

def directional_score(label, score):
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0

def analyse_video(video_id, channel_id="unknown", creator_name="unknown"):
    print(f"\nFetching transcript for video: {video_id}")
    text = get_transcript(video_id)

    print("Extracting tickers...")
    tickers = extract_tickers(text)
    print(f"Found {len(tickers)} tickers: {tickers}\n")

    # Store creator and video in database
    creator_id = get_or_create_creator(channel_id, creator_name)
    db_video_id = get_or_create_video(creator_id, video_id, title="unknown")

    results = []
    for ticker in tickers:
        label, score, sentences = analyse_sentiment_for_ticker(text, ticker)
        d_score = directional_score(label, score)

        # Store in database
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

    print(f"{'TICKER':<8} {'SENTIMENT':<12} {'DIRECTIONAL SCORE':<20} {'SENTENCES'}")
    print("-" * 50)
    for r in results:
        print(f"{r['ticker']:<8} {r['sentiment']:<12} {r['directional_score']:<20} {r['sentences_analysed']}")

    return results

if __name__ == "__main__":
    init_schema()
    analyse_video("Hl8sgbmBF98", channel_id="UCGy7SkBjcIAgTiwkXEtPnYg", creator_name="Meet Kevin")