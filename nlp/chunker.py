import spacy

# Load the spaCy English model once when the module is imported
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    # spaCy reads the text and splits it into proper sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def get_sentences_for_ticker(text, ticker, context_window=1):
    from nlp.ticker_extractor import TICKER_DICT
    
    # Find all company names that map to this ticker
    company_names = [name for name, tick in TICKER_DICT.items() if tick == ticker]
    
    sentences = split_sentences(text)
    relevant_indices = set()
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Check for $TICKER format or company name mention
        mentioned = f"${ticker}" in sentence or any(name in sentence_lower for name in company_names)
        
        if mentioned:
            # Add the sentence itself plus surrounding context window
            for j in range(max(0, i - context_window), min(len(sentences), i + context_window + 1)):
                relevant_indices.add(j)
    
    relevant_sentences = [sentences[i] for i in sorted(relevant_indices)]
    return relevant_sentences

if __name__ == "__main__":
    from ingestion.youtube_fetcher import get_transcript
    from nlp.ticker_extractor import extract_tickers

    text = get_transcript("Hl8sgbmBF98")
    tickers = extract_tickers(text)

    for ticker in tickers:
        sentences = get_sentences_for_ticker(text, ticker)
        print(f"\n{ticker} — {len(sentences)} sentences:")
        for s in sentences:
            print(f"  - {s}")