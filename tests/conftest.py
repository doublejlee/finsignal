"""Shared fixtures: each test gets a fresh temp DuckDB.

get_connection() resolves db.init.DB_PATH at call time, so monkeypatching it points the whole
storage layer at a throwaway database — no mocks, the real SQL runs against real tables.
"""
import datetime as dt
import pytest

import db.init as dbinit
from db import storage


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(dbinit, "DB_PATH", tmp_path / "test.duckdb")
    dbinit.init_schema()
    return storage


@pytest.fixture
def add_calls(db):
    """Insert n calls for a creator — each a video + ticker sentiment + backtest row."""
    def _add(name, channel, n, beat=True, ticker="NVDA", score=0.5, label="positive",
             call="bullish", ret=5.0, bench=1.0, published=dt.datetime(2025, 6, 1)):
        creator_id = db.get_or_create_creator(channel, name)
        for i in range(n):
            vid = db.get_or_create_video(creator_id, f"{channel}_v{i}", "title", published)
            db.store_ticker_sentiment(vid, ticker, label, score, 5)
            db.store_backtest_result(vid, ticker, 30, call, ret, ret > 0, bench, beat)
        return creator_id
    return _add
