import datetime as dt

from db import storage


def test_wilson_lower_bound():
    wlb = storage._wilson_lower_bound
    assert wlb(0, 0) == 0.0
    assert 0.0 < wlb(5, 5) < 1.0                 # 100% but penalized for tiny sample
    assert wlb(50, 100) > wlb(3, 6)              # same p=0.5, larger n -> higher bound
    assert 0.0 <= wlb(7, 10) <= 1.0


def test_promote_creators_flips_only_the_proven(add_calls, db):
    add_calls("Pro", "cpro", n=30, beat=True)    # 30/30 beat SPY -> Wilson LB well above 0.5
    add_calls("Weak", "cweak", n=5, beat=True)   # great rate but too few calls -> not eligible
    promoted = {p["creator"] for p in db.promote_creators()}
    assert "Pro" in promoted and "Weak" not in promoted

    status = dict(db.get_connection().execute("SELECT name, status FROM creators").fetchall())
    assert status["Pro"] == "tracked"
    assert status["Weak"] == "candidate"


def test_delete_creator_cascades(add_calls, db):
    add_calls("Gone", "cgone", n=3)
    add_calls("Keep", "ckeep", n=2)

    result = db.delete_creator("Gone")
    assert result["videos_deleted"] == 3

    conn = db.get_connection()
    assert conn.execute("SELECT COUNT(*) FROM creators WHERE name = 'Gone'").fetchone()[0] == 0
    # only "Keep"'s rows survive in the child tables
    assert conn.execute("SELECT COUNT(*) FROM ticker_sentiments").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0] == 2


def test_strategy_beats_spy_when_calls_win(add_calls, db):
    # proven creator, every bullish call returns +5% while SPY only +1%
    add_calls("Pro", "cpro", n=30, beat=True, ret=5.0, bench=1.0)
    df, metrics = db.get_strategy_performance()
    assert not df.empty
    assert metrics["trades"] == 30
    assert metrics["strategy_return_pct"] > metrics["spy_return_pct"]


def test_consensus_recency_decay(db):
    creator_id = db.get_or_create_creator("c1", "C1")
    now = dt.datetime.now()
    fresh = db.get_or_create_video(creator_id, "v_fresh", "t", now - dt.timedelta(days=2))
    old = db.get_or_create_video(creator_id, "v_old", "t", now - dt.timedelta(days=300))
    db.store_ticker_sentiment(fresh, "NVDA", "positive", 0.9, 5)   # recent bullish
    db.store_ticker_sentiment(old, "NVDA", "negative", -0.9, 5)    # stale bearish

    plain = db.get_consensus_scores(min_creators=1, half_life_days=None)
    decayed = db.get_consensus_scores(min_creators=1, half_life_days=30)
    p = float(plain[plain.ticker == "NVDA"]["avg_score"].iloc[0])
    d = float(decayed[decayed.ticker == "NVDA"]["avg_score"].iloc[0])

    assert abs(p) < 0.2     # plain mean of +0.9 and -0.9 ≈ 0
    assert d > 0.3          # recency makes the current view clearly bullish
