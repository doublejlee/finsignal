import re

TICKER_DICT = {
    # Tech
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
    "meta": "META", "tesla": "TSLA", "amd": "AMD",
    "netflix": "NFLX", "palantir": "PLTR", "snowflake": "SNOW",
    "coinbase": "COIN", "shopify": "SHOP", "airbnb": "ABNB",
    "qualcomm": "QCOM", "broadcom": "AVGO", "salesforce": "CRM",
    "oracle": "ORCL", "adobe": "ADBE", "paypal": "PYPL",
    "spotify": "SPOT", "twitter": "X", "lyft": "LYFT",
    "pinterest": "PINS", "snap": "SNAP", "roblox": "RBLX",
    "crowdstrike": "CRWD", "datadog": "DDOG", "cloudflare": "NET",
    "fortinet": "FTNT", "palo alto": "PANW", "servicenow": "NOW",
    "workday": "WDAY", "mongodb": "MDB", "gitlab": "GTLB",
    "arm holdings": "ARM", "asml": "ASML", "taiwan semiconductor": "TSM",
    "micron": "MU", "applied materials": "AMAT",

    # Finance
    "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS",
    "bank of america": "BAC", "wells fargo": "WFC", "citigroup": "C",
    "blackrock": "BLK", "visa": "V", "mastercard": "MA",
    "american express": "AXP", "berkshire": "BRK",

    # Energy
    "exxon": "XOM", "chevron": "CVX", "conocophillips": "COP",
    "enphase": "ENPH", "nphase": "NPHS", "first solar": "FSLR",
    "nextera": "NEE", "plug power": "PLUG",

    # Healthcare
    "johnson and johnson": "JNJ", "unitedhealth": "UNH",
    "pfizer": "PFE", "moderna": "MRNA", "abbvie": "ABBV",
    "eli lilly": "LLY", "novo nordisk": "NVO",

    # Consumer
    "walmart": "WMT", "target": "TGT", "costco": "COST",
    "nike": "NKE", "starbucks": "SBUX", "mcdonald": "MCD",
    "disney": "DIS", "warner bros": "WBD", "comcast": "CMCSA",

    # ETFs commonly mentioned
    "spy": "SPY", "qqq": "QQQ", "voo": "VOO",
}

AMBIGUOUS_TICKERS = {"INTC", "UBER", "AMD", "NET", "NOW", "V", "C", "X"}

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