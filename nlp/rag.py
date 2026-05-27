"""RAG over stored creator sentences: retrieve relevant sentences, answer with Groq.

Local MVP — query embedding uses sentence-transformers (torch). Phase 4b will swap
this for a hosted embedding API so the cloud serving layer can stay torch-free.
Run with VPN OFF (Groq blocks VPN IPs).

    python -m nlp.rag "what is the bull case for NVDA?"
"""
import os
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from db.init import get_connection

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def retrieve(query: str, top_k: int = 8) -> list:
    """Return the top_k stored sentences most similar to the query (cosine)."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT se.embedding, ts.sentence, ts.ticker, c.name, v.video_id
        FROM sentence_embeddings se
        JOIN transcript_segments ts ON se.segment_id = ts.id
        JOIN videos v ON ts.video_id = v.id
        JOIN creators c ON v.creator_id = c.id
    """).fetchall()

    if not rows:
        return []

    vectors = np.array([r[0] for r in rows], dtype=np.float32)
    q = _get_model().encode([query], normalize_embeddings=True)[0]
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
