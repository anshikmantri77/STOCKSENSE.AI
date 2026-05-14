import io
from datetime import datetime

import pandas as pd


def export_portfolio_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def get_download_button(df: pd.DataFrame, filename_prefix: str = "portfolio"):
    return dict(
        data=export_portfolio_csv(df),
        file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


def color_change_val(val):
    if pd.isna(val):
        return "color: black"
    if val > 0:
        return "color: green"
    elif val < 0:
        return "color: red"
    return "color: black"


def format_large_number(num):
    if num is None:
        return "N/A"
    if num >= 1e7:
        return f"{num / 1e7:,.2f} Cr"
    elif num >= 1e5:
        return f"{num / 1e5:,.2f} L"
    return f"{num:,.2f}"
