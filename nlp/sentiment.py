from transformers import pipeline
from nlp.chunker import get_sentences_for_ticker
from collections import Counter

classifier = pipeline("text-classification", model="ProsusAI/finbert")


def analyse_sentiment_for_ticker(text, ticker):
    sentences = get_sentences_for_ticker(text, ticker)
    
    if not sentences:
        return "neutral", 0.0, []
    
    results = []
    for sentence in sentences:
        if len(sentence.split()) < 8:
            continue
        trimmed = sentence[:512]
        result = classifier(trimmed)
        # Only keep high confidence results - low confidence is usually noise
        if result[0]["score"] > 0.6:
            results.append(result[0])
    
    if not results:
        return "neutral", 0.0, sentences

    # Calculate weighted directional score instead of just counting labels
    # Later sentences get higher weight (recency bias)
    total_weight = 0
    weighted_score = 0
    
    for i, result in enumerate(results):
        # Weight increases for later sentences
        weight = 1 + (2 * (i / len(results)))
        label = result["label"]
        score = result["score"]
        
        if label == "positive":
            direction = score
        elif label == "negative":
            direction = -score
        else:
            direction = 0
            
        weighted_score += direction * weight
        total_weight += weight
    
    final_score = weighted_score / total_weight
    
    if final_score > 0.05:
        label = "positive"
    elif final_score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return label, round(abs(final_score), 2), sentences

if __name__ == "__main__":
    from ingestion.youtube_fetcher import get_transcript
    from nlp.ticker_extractor import extract_tickers

    text = get_transcript("Hl8sgbmBF98")
    tickers = extract_tickers(text)

    print(f"Analysing sentiment for {len(tickers)} tickers...\n")
    for ticker in tickers:
        label, score, sentences = analyse_sentiment_for_ticker(text, ticker)
        print(f"{ticker}: {label} ({score}) — based on {len(sentences)} sentences")
    
    from nlp.chunker import get_sentences_for_ticker
    sentences = get_sentences_for_ticker(text, "NPHS")
    print("\nNPHS weighted calculation:")
    filtered = []
    for sentence in sentences:
        if len(sentence.split()) < 8:
            continue
        result = classifier(sentence[:512])
        if result[0]["score"] > 0.6:
            filtered.append(result[0])
            
    total_weight = 0
    weighted_score = 0
    for i, result in enumerate(filtered):
        weight = 1 + (2 * (i / len(filtered)))
        label = result["label"]
        score = result["score"]
        direction = score if label == "positive" else -score if label == "negative" else 0
        weighted_score += direction * weight
        total_weight += weight
        print(f"  i={i} weight={weight:.2f} label={label} score={score:.2f} direction={direction:.2f}")
    
    print(f"\n  final_score = {weighted_score/total_weight:.4f}")
    