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

load_dotenv()

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

def retrieve(query: str, top_k: int = 8) -> list:
    """Return the top_k stored sentences most similar to the query (cosine).

    Embeddings live in a separate, local-only DB (EMBEDDINGS_DB_PATH); it's absent on the
    cloud deploy, so retrieval there returns nothing and Ask degrades gracefully.
    """
    if not EMBEDDINGS_DB_PATH.exists():
        return []

    conn = get_connection()
    conn.execute(f"ATTACH '{EMBEDDINGS_DB_PATH.as_posix()}' AS emb (READ_ONLY)")
    rows = conn.execute("""
        SELECT se.embedding, ts.sentence, ts.ticker, c.name, v.video_id
        FROM emb.sentence_embeddings se
        JOIN transcript_segments ts ON se.segment_id = ts.id
        JOIN videos v ON ts.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
    """).fetchall()

    if not rows:
        return []

    vectors = np.array([r[0] for r in rows], dtype=np.float32)
    q = _embed_query(query)
    scores = vectors @ q  # both normalized -> dot product is cosine similarity

    # Walk best-first, keeping each distinct sentence once (the same sentence is
    # stored per-ticker, so it can otherwise fill top_k with duplicates).
    results = []
    seen = set()
    for i in np.argsort(scores)[::-1]:
        sentence = rows[i][1]
        if sentence in seen:
            continue
        seen.add(sentence)
        results.append({
            "sentence": sentence,
            "ticker": rows[i][2],
            "creator": rows[i][3],
            "video_id": rows[i][4],
            "score": float(scores[i]),
        })
        if len(results) >= top_k:
            break

    return results

def answer(query: str, top_k: int = 8) -> dict:
    """Retrieve context, then ask Groq to answer grounded in it with inline citations."""
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return {"answer": "No data available to answer that yet.", "citations": []}

    context = "\n".join(
        f"[{i + 1}] ({h['creator']} on {h['ticker']}) {h['sentence']}"
        for i, h in enumerate(hits)
    )

    prompt = f"""You are a financial research assistant. Answer the question using ONLY the numbered context below — sentences from finance YouTube creators.

Rules:
- Use only sentences that state a concrete, substantive point. Ignore vague, rhetorical, or filler sentences (e.g. "where does that leave us?").
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
