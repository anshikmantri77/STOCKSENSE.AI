import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stocksense"))

from modules.sentiment import get_sentiment


def test_get_sentiment_empty():
    result = get_sentiment([])
    assert result["positive"] == 0
    assert result["negative"] == 0
    assert result["neutral"] == 1
    assert sum(result.values()) == 1
