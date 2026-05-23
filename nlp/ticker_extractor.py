import re

TICKER_DICT = {
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
    "meta": "META", "tesla": "TSLA", "amd": "AMD",
    "netflix": "NFLX", "palantir": "PLTR", "nphase": "NPHS",
    "enphase": "ENPH", "qualcomm": "QCOM",
    "broadcom": "AVGO", "snowflake": "SNOW", "coinbase": "COIN",
    "shopify": "SHOP", "airbnb": "ABNB"
}

# These words are too ambiguous as plain text — only match via $TICKER format
AMBIGUOUS_TICKERS = {"INTC", "UBER", "AMD"}

def extract_tickers(text):
    found = set()

    regex_matches = re.findall(r'\$([A-Z]{1,5})', text)
    for match in regex_matches:
        found.add(match)

    text_lower = text.lower()
    for name, ticker in TICKER_DICT.items():
        if ticker in AMBIGUOUS_TICKERS:
            continue
        if name in text_lower:
            found.add(ticker)

    return list(found)

if __name__ == "__main__":
    from ingestion.youtube_fetcher import get_transcript
    text = get_transcript("Hl8sgbmBF98")
    tickers = extract_tickers(text)
    print("Tickers found:", tickers)