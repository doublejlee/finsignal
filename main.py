from ingestion.youtube_fetcher import get_transcript
from nlp.ticker_extractor import extract_tickers
from nlp.sentiment import analyse_sentiment_for_ticker

def directional_score(label, score):
    if label == "positive":
        return score        # 0 to 1, bullish
    elif label == "negative":
        return -score       # -1 to 0, bearish
    else:
        return 0            # neutral counts as 0
    
def analyse_video(video_id):
    print(f"\nFetching transcript for video: {video_id}")
    text = get_transcript(video_id)
    
    print("Extracting tickers...")
    tickers = extract_tickers(text)
    
    print(f"Found {len(tickers)} tickers: {tickers}\n")
    print("-" * 40)
    
    results = []
    for ticker in tickers:
        label, score, sentences = analyse_sentiment_for_ticker(text, ticker)
        results.append({
            "ticker": ticker,
            "sentiment": label,
            "score": score,
            "directional_score": directional_score(label, score),
            "sentences_analysed": len(sentences)
        })
    
    # Sort by directional score descending
    results.sort(key=lambda x: x["directional_score"], reverse=True)
    
    print(f"{'TICKER':<8} {'SENTIMENT':<12} {'DIRECTIONAL SCORE':^18} {'SENTENCES':^10}")
    print("-" * 55)
    for r in results:
        print(f"{r['ticker']:<8} {r['sentiment']:<12} {r['directional_score']:^18.2f} {r['sentences_analysed']:^10}")
    
    return results

if __name__ == "__main__":
    analyse_video("Hl8sgbmBF98")    