from nlp.ticker_extractor import extract_tickers, sentence_mentions_ticker


def test_dollar_ticker_matches():
    assert "NVDA" in extract_tickers("loading up on $NVDA here")


def test_company_name_matches():
    assert "AAPL" in extract_tickers("Apple is a strong buy")


def test_ambiguous_name_requires_dollar():
    # "uber" is an everyday word -> must be explicit $UBER to count
    assert "UBER" not in extract_tickers("I uber to the office")
    assert "UBER" in extract_tickers("$UBER looks cheap")


def test_word_boundary_no_substring_false_positive():
    # the NPHS/metaverse class of bug: a name must be a whole word, not a substring
    assert "META" not in extract_tickers("the metaverse is overhyped")
    assert "META" in extract_tickers("meta keeps grinding higher")


def test_sentence_mentions_ticker():
    assert sentence_mentions_ticker("Nvidia keeps winning", "NVDA")
    assert sentence_mentions_ticker("$NVDA to the moon", "NVDA")
    # a context-window neighbor that doesn't name the ticker -> not a mention
    assert not sentence_mentions_ticker("the market dropped today", "NVDA")
    # ambiguous name without $ is not a mention
    assert not sentence_mentions_ticker("I uber everywhere", "UBER")
