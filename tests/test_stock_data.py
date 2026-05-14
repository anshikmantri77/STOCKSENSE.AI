import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stocksense"))

from modules.stock_data import format_large_number


def test_format_large_number_crores():
    assert format_large_number(150_000_000) == "15.00 Cr"


def test_format_large_number_lakhs():
    assert format_large_number(500_000) == "5.00 L"


def test_format_large_number_none():
    assert format_large_number(None) == "N/A"


def test_format_large_number_small():
    assert format_large_number(1234.56) == "1,234.56"
