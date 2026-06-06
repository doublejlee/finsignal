import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent / "finsignal.duckdb"
# RAG vectors live in their own file, gitignored and not deployed: they're a large
# (~50MB+), regenerable local index that the cloud serving layer never reads, and bundling
# them pushed the committed DB past GitHub's 100MB limit. Rebuild with backfill_embeddings.
EMBEDDINGS_DB_PATH = Path(__file__).parent / "embeddings.duckdb"

def get_connection():
    """Get or create the main DuckDB connection (serving data)."""
    return duckdb.connect(str(DB_PATH))

def get_embeddings_connection():
    """Connection to the local-only RAG embeddings DB (see EMBEDDINGS_DB_PATH)."""
    return duckdb.connect(str(EMBEDDINGS_DB_PATH))

def init_embeddings_schema():
    """Create the sentence_embeddings table in the separate embeddings DB (idempotent)."""
    conn = get_embeddings_connection()
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_sentence_embeddings START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentence_embeddings (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_sentence_embeddings'),
            segment_id INTEGER NOT NULL,
            embedding FLOAT[384] NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(segment_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentence_embeddings_segment_id ON sentence_embeddings(segment_id)")
    conn.commit()

def init_schema():
    """Initialize database schema."""
    conn = get_connection()
    
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_creators START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_videos START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_ticker_sentiments START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_segments START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_ticker_reasons START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_backtest_results START 1")


    conn.execute("""
        CREATE TABLE IF NOT EXISTS creators (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_creators'),
            channel_id VARCHAR UNIQUE NOT NULL,
            name VARCHAR NOT NULL,
            subscriber_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_videos'),
            creator_id INTEGER NOT NULL,
            video_id VARCHAR UNIQUE NOT NULL,
            title VARCHAR,
            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (creator_id) REFERENCES creators(id)
        )
    """)



    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_sentiments (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_ticker_sentiments'),
            video_id INTEGER NOT NULL,
            ticker VARCHAR NOT NULL,
            label VARCHAR NOT NULL,
            directional_score FLOAT NOT NULL,
            sentence_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id),
            UNIQUE(video_id, ticker)
        )
    """)

  

    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcript_segments (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_segments'),
            video_id INTEGER NOT NULL,
            ticker VARCHAR NOT NULL,
            sentence VARCHAR,
            label VARCHAR NOT NULL,
            score FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)



    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_reasons (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_ticker_reasons'),
            video_id INTEGER NOT NULL,
            ticker VARCHAR NOT NULL,
            reason VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_backtest_results'),
            video_id INTEGER NOT NULL,
            ticker VARCHAR NOT NULL,
            horizon_days INTEGER NOT NULL,
            call VARCHAR NOT NULL,
            return_pct FLOAT NOT NULL,
            correct BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id),
            UNIQUE(video_id, ticker, horizon_days)
        )
    """)

    # sentence_embeddings now lives in a separate, gitignored DB — see init_embeddings_schema().

    # Migrations for columns added after initial release (idempotent)
    def _columns(table):
        return [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]
        ).fetchall()]

    if "status" not in _columns("creators"):
        conn.execute("ALTER TABLE creators ADD COLUMN status VARCHAR DEFAULT 'candidate'")

    # is_context marks ±1 context-window neighbors (sentences that don't themselves mention
    # the ticker) so RAG can cite only genuine mentions — see store_transcript_segments.
    if "is_context" not in _columns("transcript_segments"):
        conn.execute("ALTER TABLE transcript_segments ADD COLUMN is_context BOOLEAN DEFAULT FALSE")

    # stance labels each reason bullish/bearish/neutral (topic_model). No default: existing
    # rows stay NULL so backfill_reasons re-extracts them with a stance.
    if "stance" not in _columns("ticker_reasons"):
        conn.execute("ALTER TABLE ticker_reasons ADD COLUMN stance VARCHAR")

    backtest_cols = _columns("backtest_results")
    if "benchmark_return_pct" not in backtest_cols:
        conn.execute("ALTER TABLE backtest_results ADD COLUMN benchmark_return_pct FLOAT")
    if "beat_benchmark" not in backtest_cols:
        conn.execute("ALTER TABLE backtest_results ADD COLUMN beat_benchmark BOOLEAN")

    # Indexes for query performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_sentiments_ticker ON ticker_sentiments(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_sentiments_video_id ON ticker_sentiments(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transcript_ticker ON transcript_segments(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transcript_video_id ON transcript_segments(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_creator_id ON videos(creator_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_reasons_ticker ON ticker_reasons(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_reasons_video_id ON ticker_reasons(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_video_id ON backtest_results(video_id)")

    conn.commit()
    print("Database schema initialized")
    
def check_data():
    conn = get_connection()
    
    print("\n--- Creators ---")
    print(conn.execute("SELECT * FROM creators").fetchall())
    
    print("\n--- Videos ---")
    print(conn.execute("SELECT * FROM videos").fetchall())
    
    print("\n--- Ticker Sentiments ---")
    print(conn.execute("SELECT * FROM ticker_sentiments").fetchall())

if __name__ == "__main__":
    init_schema()
    check_data()

