from transformers import pipeline
import re
from nlp.ticker_extractor import TICKER_DICT

# Load FinBERT once when the module is imported
# This takes a few seconds the first time as it downloads the model
classifier = pipeline("text-classification", model="ProsusAI/finbert")

def get_sentences_for_ticker(text, ticker):
    """Extract sentences that mention a specific ticker from the transcript"""
    # Find all company names that map to this ticker
    company_names = [name for name, tick in TICKER_DICT.items() if tick == ticker]
    
    # Split into sentences on . ! ?
    sentences = re.split(r'[.!?]+', text)
    
    relevant_sentences = []
    for sentence in sentences:
        sentence_stripped = sentence.strip()
        if not sentence_stripped:
            continue
        
        # Check if sentence mentions ticker in $TICKER format
        if f"${ticker}" in sentence:
            relevant_sentences.append(sentence_stripped)
        else:
            # Check if sentence mentions any company name (case-insensitive)
            sentence_lower = sentence.lower()
            for company in company_names:
                if company in sentence_lower:
                    relevant_sentences.append(sentence_stripped)
                    break
    
    return relevant_sentences

def analyse_sentiment_for_ticker(text, ticker):
    sentences = get_sentences_for_ticker(text, ticker)
    
    if not sentences:
        return "neutral", 0.0, []
    
    # Run FinBERT on each relevant sentence
    results = []
    for sentence in sentences:
        trimmed = sentence[:512]
        result = classifier(trimmed)
        results.append(result[0])
    
    # Average the scores grouped by label
    from collections import Counter
    labels = [r["label"] for r in results]
    most_common_label = Counter(labels).most_common(1)[0][0]
    avg_score = sum(r["score"] for r in results) / len(results)
    
    return most_common_label, round(avg_score, 2), sentences

if __name__ == "__main__":
    from ingestion.youtube_fetcher import get_transcript
    from nlp.ticker_extractor import extract_tickers

    text = get_transcript("Hl8sgbmBF98")
    tickers = extract_tickers(text)

    print(f"Analysing sentiment for {len(tickers)} tickers...\n")
    for ticker in tickers:
        label, score, sentences = analyse_sentiment_for_ticker(text, ticker)
        print(f"{ticker}: {label} ({score}) — based on {len(sentences)} sentences")