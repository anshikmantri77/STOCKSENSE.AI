import streamlit as st
import pandas as pd
import numpy as np
from config import stock_analyzer
from modules.stock_data import get_current_stock_info


@st.cache_data(ttl=600)
def run_screener(tickers: list, filters: dict) -> pd.DataFrame:
    results = []
    for ticker in tickers:
        try:
            info = get_current_stock_info(ticker)
            if not info:
                continue

            pe = info.get("trailingPE")
            roe = info.get("returnOnEquity")
            debt_eq = info.get("debtToEquity")
            market_cap = info.get("marketCap")
            revenue_growth = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")
            company_name = info.get(
                "shortName", info.get("longName", ticker.replace(".NS", ""))
            )

            if filters.get("max_pe") and pe and pe > filters["max_pe"]:
                continue
            if filters.get("min_roe") and roe and roe < filters["min_roe"]:
                continue
            if filters.get("max_de") and debt_eq and debt_eq > filters["max_de"]:
                continue
            if (
                filters.get("min_market_cap")
                and market_cap
                and market_cap < filters["min_market_cap"]
            ):
                continue
            if (
                filters.get("min_rev_growth")
                and revenue_growth
                and revenue_growth < filters["min_rev_growth"]
            ):
                continue
            if (
                filters.get("min_earn_growth")
                and earnings_growth
                and earnings_growth < filters["min_earn_growth"]
            ):
                continue

            results.append(
                {
                    "Company": company_name,
                    "Ticker": ticker,
                    "P/E": round(pe, 2) if pe else "N/A",
                    "ROE (%)": f"{round(roe * 100, 1)}%" if roe else "N/A",
                    "Debt/Equity": round(debt_eq, 2) if debt_eq else "N/A",
                    "Market Cap": (
                        f"{market_cap / 1e7:.2f} Cr" if market_cap else "N/A"
                    ),
                    "Rev Growth (%)": (
                        f"{round(revenue_growth * 100, 1)}%"
                        if revenue_growth
                        else "N/A"
                    ),
                    "Earn Growth (%)": (
                        f"{round(earnings_growth * 100, 1)}%"
                        if earnings_growth
                        else "N/A"
                    ),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(results)
