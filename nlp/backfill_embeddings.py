"""Embed stored transcript sentences for RAG retrieval. Run locally (uses torch).

Resumable: only embeds transcript_segments rows not yet in sentence_embeddings.
Vectors are L2-normalized so cosine similarity reduces to a dot product at query time.

    python -m nlp.backfill_embeddings
"""
from sentence_transformers import SentenceTransformer
from db.init import get_connection, get_embeddings_connection, init_embeddings_schema

MODEL_NAME = "all-MiniLM-L6-v2"

def backfill():
    init_embeddings_schema()
    main = get_connection()
    emb = get_embeddings_connection()

    # Sentences live in the main DB, vectors in the embeddings DB — diff across the two.
    all_segments = main.execute("SELECT id, sentence FROM transcript_segments").fetchall()
    done = {r[0] for r in emb.execute("SELECT segment_id FROM sentence_embeddings").fetchall()}
    rows = [(sid, sent) for sid, sent in all_segments if sid not in done]

    if not rows:
        print("Nothing to embed — all sentences already have embeddings.")
        return

    print(f"Embedding {len(rows)} sentences with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    ids = [r[0] for r in rows]
    sentences = [r[1] for r in rows]
    vectors = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

    for seg_id, vec in zip(ids, vectors):
        emb.execute(
            "INSERT INTO sentence_embeddings (segment_id, embedding) VALUES (?, ?)",
            [seg_id, vec.tolist()]
        )
    emb.commit()
    print(f"Stored {len(ids)} embeddings.")

if __name__ == "__main__":
    backfill()
