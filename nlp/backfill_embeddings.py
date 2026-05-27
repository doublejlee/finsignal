"""Embed stored transcript sentences for RAG retrieval. Run locally (uses torch).

Resumable: only embeds transcript_segments rows not yet in sentence_embeddings.
Vectors are L2-normalized so cosine similarity reduces to a dot product at query time.

    python -m nlp.backfill_embeddings
"""
from sentence_transformers import SentenceTransformer
from db.init import get_connection

MODEL_NAME = "all-MiniLM-L6-v2"

def backfill():
    conn = get_connection()
    rows = conn.execute("""
        SELECT ts.id, ts.sentence
        FROM transcript_segments ts
        LEFT JOIN sentence_embeddings se ON ts.id = se.segment_id
        WHERE se.segment_id IS NULL
    """).fetchall()

    if not rows:
        print("Nothing to embed — all sentences already have embeddings.")
        return

    print(f"Embedding {len(rows)} sentences with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    ids = [r[0] for r in rows]
    sentences = [r[1] for r in rows]
    vectors = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

    for seg_id, vec in zip(ids, vectors):
        conn.execute(
            "INSERT INTO sentence_embeddings (segment_id, embedding) VALUES (?, ?)",
            [seg_id, vec.tolist()]
        )
    conn.commit()
    print(f"Stored {len(ids)} embeddings.")

if __name__ == "__main__":
    backfill()
