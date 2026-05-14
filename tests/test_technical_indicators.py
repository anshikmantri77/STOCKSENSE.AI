import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stocksense"))

import pandas as pd
import numpy as np
from modules.technical_indicators import add_technical_indicators


def test_add_technical_indicators():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {"Open": np.random.randn(200) + 100,
         "High": np.random.randn(200) + 102,
         "Low": np.random.randn(200) + 98,
         "Close": np.random.randn(200) + 100,
         "Volume": np.random.randint(1000, 10000, 200)},
        index=dates,
    )
    result = add_technical_indicators(df)
    assert "MA20" in result.columns
    assert "MA50" in result.columns
    assert "RSI" in result.columns
    assert "BB_upper" in result.columns
    assert "BB_lower" in result.columns
    assert "MACD" in result.columns
    assert "MACD_signal" in result.columns
    assert result["RSI"].max() <= 100
    assert result["RSI"].min() >= 0


def test_rsi_values():
    close = list(range(100, 150)) + list(range(150, 100, -1))
    dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
    df = pd.DataFrame(
        {"Open": close, "High": [c + 2 for c in close],
         "Low": [c - 2 for c in close], "Close": close,
         "Volume": 10000},
        index=dates,
    )
    result = add_technical_indicators(df)
    assert not result["RSI"].isna().all()
