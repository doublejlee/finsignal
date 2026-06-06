"""RAG over stored creator sentences: retrieve relevant sentences, answer with Groq.

Run with VPN OFF (Groq blocks VPN IPs).

    python -m nlp.rag "what is the bull case for NVDA?"
"""
import os
import re
import numpy as np
import requests
from dotenv import load_dotenv
from db.init import get_connection, EMBEDDINGS_DB_PATH
from db.storage import _wilson_lower_bound
from nlp.ticker_extractor import extract_tickers

load_dotenv()

# Lexical cues for reading bull/bear intent from a question, used to pre-filter retrieval
# by stored sentiment label before semantic ranking.
_BULL_WORDS = ("bull", "buy", "long", "upside", "optimistic", "undervalued", "good investment")
_BEAR_WORDS = ("bear", "sell", "short", "downside", "overvalued", "avoid", "concern", "risk")

def _query_polarity(query: str):
    """positive / negative if the question clearly asks for one side, else None."""
    q = query.lower()
    bull, bear = any(w in q for w in _BULL_WORDS), any(w in q for w in _BEAR_WORDS)
    if bull and not bear:
        return "positive"
    if bear and not bull:
        return "negative"
    return None

def _creator_credibility(conn) -> dict:
    """Per-creator retrieval weight from their measured track record: weight = 0.5 + Wilson
    lower-bound of their beat-SPY rate (so a proven creator ~1.5x, an unproven/poor one down
    to 0.5x). Creators with no benchmark-scored calls get a neutral 1.0 — unknown, not bad."""
    rows = conn.execute("""
        SELECT cr.name,
               COUNT(b.id) FILTER (WHERE b.beat_benchmark IS NOT NULL) AS n,
               COALESCE(SUM(CASE WHEN b.beat_benchmark THEN 1 ELSE 0 END), 0) AS beats
        FROM creators cr
        LEFT JOIN videos v ON v.creator_id = cr.id
        LEFT JOIN backtest_results b ON b.video_id = v.id
        GROUP BY cr.name
    """).fetchall()
    cred = {}
    for name, n, beats in rows:
        n, beats = int(n or 0), int(beats or 0)
        if n > 0:
            cred[name] = {"weight": 0.5 + _wilson_lower_bound(beats, n),
                          "beat_pct": round(100 * beats / n, 1), "calls": n}
        else:
            cred[name] = {"weight": 1.0, "beat_pct": None, "calls": 0}
    return cred

_HF_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
_local_model = None

def _embed_query(query: str) -> np.ndarray:
    """Embed query via HF Inference API; falls back to local sentence-transformers if unreachable."""
    # Cloud path: torch-free, required on Render
    try:
        headers = {"Content-Type": "application/json"}
        token = os.getenv("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(
            _HF_URL,
            headers=headers,
            json={"inputs": query, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data[0] if isinstance(data[0], list) else data, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception:
        pass

    # Local fallback: sentence-transformers (pipeline env only, requires torch)
    global _local_model
    try:
        from sentence_transformers import SentenceTransformer
        if _local_model is None:
            _local_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _local_model.encode([query], normalize_embeddings=True)[0]
    except ImportError:
        raise RuntimeError(
            "Query embedding failed: HF Inference API unreachable and sentence-transformers not installed."
        )

def retrieve(query: str, top_k: int = 8, weight_by_credibility: bool = True) -> list:
    """Return the top_k stored sentences most relevant to the query.

    Hybrid retrieval: if the question names a ticker (via extract_tickers) and/or a side
    (bull/bear -> sentiment label), pre-filter to that subset, then rank semantically within
    it — far more precise than dense search over everything. Filters relax progressively so a
    too-narrow query never returns nothing. The final rank also multiplies cosine similarity
    by each creator's credibility (Wilson-bound beat-SPY weight), so proven creators surface
    first. Only is_context=FALSE rows (genuine mentions) are eligible.

    Embeddings live in a separate, local-only DB (EMBEDDINGS_DB_PATH); it's absent on the
    cloud deploy, so retrieval there returns nothing and Ask degrades gracefully.
    """
    if not EMBEDDINGS_DB_PATH.exists():
        return []

    conn = get_connection()
    # IF NOT EXISTS: a long-lived process (dashboard/API) shares one DB instance, so a second
    # call would otherwise fail with "database emb already exists".
    conn.execute(f"ATTACH IF NOT EXISTS '{EMBEDDINGS_DB_PATH.as_posix()}' AS emb (READ_ONLY)")

    base = """
        SELECT se.embedding, ts.sentence, ts.ticker, c.name, v.video_id, v.published_at
        FROM emb.sentence_embeddings se
        JOIN transcript_segments ts ON se.segment_id = ts.id
        JOIN videos v ON ts.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
        WHERE {where}
    """
    tickers = extract_tickers(query)
    polarity = _query_polarity(query)
    tfilter = (f"ts.ticker IN ({','.join(['?'] * len(tickers))})", list(tickers)) if tickers else None
    pfilter = ("ts.label = ?", [polarity]) if polarity else None

    # strictest first, then relax — always end on the unfiltered semantic search (deduped)
    attempts = []
    for combo in ([tfilter, pfilter], [tfilter], [pfilter], []):
        cleaned = [f for f in combo if f]
        if cleaned not in attempts:
            attempts.append(cleaned)
    rows = []
    for combo in attempts:
        where = " AND ".join(["ts.is_context = FALSE"] + [f[0] for f in combo])
        params = [p for f in combo for p in f[1]]
        rows = conn.execute(base.format(where=where), params).fetchall()
        if rows:
            break
    if not rows:
        return []

    vectors = np.array([r[0] for r in rows], dtype=np.float32)
    q = _embed_query(query)
    cosine = vectors @ q  # both normalized -> dot product is cosine similarity

    cred = _creator_credibility(conn)  # always shown in results; only applied to rank if enabled
    if weight_by_credibility:
        weights = np.array([cred.get(r[3], {}).get("weight", 1.0) for r in rows], dtype=np.float32)
        ranked = cosine * weights
    else:
        ranked = cosine

    # Walk best-first, keeping each distinct sentence once (the same sentence is
    # stored per-ticker, so it can otherwise fill top_k with duplicates).
    results = []
    seen = set()
    for i in np.argsort(ranked)[::-1]:
        sentence = rows[i][1]
        if sentence in seen:
            continue
        seen.add(sentence)
        c = cred.get(rows[i][3], {})
        results.append({
            "sentence": sentence,
            "ticker": rows[i][2],
            "creator": rows[i][3],
            "video_id": rows[i][4],
            "date": str(rows[i][5])[:10] if rows[i][5] else None,
            "creator_beat_spy": c.get("beat_pct"),
            "creator_calls": c.get("calls", 0),
            "score": float(cosine[i]),
        })
        if len(results) >= top_k:
            break

    return results

def answer(query: str, top_k: int = 8) -> dict:
    """Retrieve context, then ask Groq to answer grounded in it with inline citations."""
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return {"answer": "No data available to answer that yet.", "citations": []}

    def _label(h):
        track = (f", track record {h['creator_beat_spy']}% beat-SPY over {h['creator_calls']} calls"
                 if h.get("creator_beat_spy") is not None else "")
        date = f", {h['date']}" if h.get("date") else ""
        return f"{h['creator']}{track} on {h['ticker']}{date}"

    context = "\n".join(f"[{i + 1}] ({_label(h)}) {h['sentence']}" for i, h in enumerate(hits))

    prompt = f"""You are a financial research assistant. Answer the question using ONLY the numbered context below — sentences from finance YouTube creators.

Each source notes the creator's track record: how often their past stock calls beat the S&P 500, and the date of the take.

Rules:
- Use only sentences that state a concrete, substantive point. Ignore vague, rhetorical, or filler sentences (e.g. "where does that leave us?").
- Give more weight to creators with a stronger track record and to more recent takes; you may note when a key point comes from a proven creator.
- Cite sources inline like [1], [2], and only cite the sentences you actually rely on.
- If the context does not contain a real answer, say so plainly.

Question: {query}

Context:
{context}

Answer:"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    data = response.json()
    if "choices" not in data:
        raise RuntimeError(f"Groq error {response.status_code}: {data.get('error', data)}")

    answer_text = data["choices"][0]["message"]["content"]

    # Show only the sources the answer actually cited (keeping their original numbers).
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    citations = [
        {"number": i + 1, **h}
        for i, h in enumerate(hits)
        if (i + 1) in used
    ]

    return {"answer": answer_text, "citations": citations}

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the bull case for NVDA?"
    result = answer(q)
    print("Q:", q)
    print("\nA:", result["answer"])
    print("\nSources:")
    for h in result["citations"]:
        print(f"  [{h['number']}] {h['creator']} on {h['ticker']} (score {h['score']:.2f}): {h['sentence'][:80]}")
