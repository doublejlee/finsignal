import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# 8b-instant over 70b-versatile: extracting 3 short reason phrases is an easy task, and on
# Groq's free tier 8b-instant has a much larger daily token budget (~500k TPD vs 70b's 100k)
# in its own separate bucket — so the full backfill fits in a day. Swap back to
# "llama-3.3-70b-versatile" if reason quality ever needs it.
GROQ_MODEL = "llama-3.1-8b-instant"

def summarise_reasons_with_llm(sentences: list, ticker: str) -> list:
    """Use Groq (Llama 3) to extract key reasons from sentences."""
    text = "\n".join(sentences[:12])

    prompt = f"""These sentences are from finance YouTube videos discussing {ticker}.
Extract exactly 3 short reasons (3-5 words each) that explain the sentiment.
Return only a JSON array of 3 strings, nothing else.

Sentences:
{text}

Example output: ["AI datacenter demand", "strong margin expansion", "Blackwell chip cycle"]"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
        },
        json={
            "model": GROQ_MODEL,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    data = response.json()
    if "choices" not in data:
        raise RuntimeError(f"Groq error {response.status_code}: {data.get('error', data)}")

    content = data["choices"][0]["message"]["content"]
    reasons = json.loads(content)
    return reasons[:3]

def extract_reasons(sentences: list, ticker: str = "unknown") -> list:
    """Extract reasons behind sentiment for a ticker."""
    filtered = [s for s in sentences if len(s.split()) >= 8]

    if len(filtered) < 3:
        return ["insufficient data"]

    return summarise_reasons_with_llm(filtered, ticker)

if __name__ == "__main__":
    from db.init import get_connection

    conn = get_connection()

    rows = conn.execute("""
        SELECT sentence FROM transcript_segments
        WHERE ticker = 'GOOGL'
    """).fetchall()

    sentences = [row[0] for row in rows]
    print(f"Extracting reasons from {len(sentences)} sentences for GOOGL...\n")

    reasons = extract_reasons(sentences, ticker="GOOGL")
    print("Reasons:")
    for r in reasons:
        print(f"  - {r}")
