import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stocksense"))

import pandas as pd
from utils.helpers import export_portfolio_csv, color_change_val, format_large_number


def test_export_portfolio_csv_content():
    df = pd.DataFrame({"Ticker": ["RELIANCE.NS", "TCS.NS"], "Price": [2500, 3800]})
    result = export_portfolio_csv(df)
    assert isinstance(result, bytes)
    assert b"Ticker" in result
    assert b"RELIANCE.NS" in result
    assert b"TCS.NS" in result


def test_color_change_val_positive():
    assert "color: green" in color_change_val(5)


def test_color_change_val_negative():
    assert "color: red" in color_change_val(-3)


def test_color_change_val_nan():
    assert "color: black" in color_change_val(float("nan"))


def test_format_large_number_crores():
    assert format_large_number(150_000_000) == "15.00 Cr"


def test_format_large_number_lakhs():
    assert format_large_number(500_000) == "5.00 L"


def test_format_large_number_none():
    assert format_large_number(None) == "N/A"


def test_format_large_number_small():
    assert format_large_number(1234.56) == "1,234.56"
