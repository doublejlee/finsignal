import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent / "finsignal.duckdb"

def get_connection():
    """Get or create DuckDB connection."""
    return duckdb.connect(str(DB_PATH))

def init_schema():
    """Initialize database schema."""
    conn = get_connection()
    
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_creators START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_videos START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_ticker_sentiments START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_segments START 1")
    

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



    # Indexes for query performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_sentiments_ticker ON ticker_sentiments(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_sentiments_video_id ON ticker_sentiments(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transcript_ticker ON transcript_segments(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transcript_video_id ON transcript_segments(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_creator_id ON videos(creator_id)")

    conn.commit()
    print("✓ Database schema initialized")
    
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

