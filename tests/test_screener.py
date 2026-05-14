import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stocksense"))

from modules.stock_data import format_large_number
from utils.helpers import export_portfolio_csv
import pandas as pd


def test_export_portfolio_csv():
    df = pd.DataFrame({"Ticker": ["RELIANCE.NS"], "Price": [2500]})
    result = export_portfolio_csv(df)
    assert isinstance(result, bytes)
    assert b"Ticker" in result
