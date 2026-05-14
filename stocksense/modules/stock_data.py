import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


@st.cache_data(ttl=300)
def get_stock_data(ticker: str, period: str = "1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker}: {e}")
        return None


@st.cache_data(ttl=300)
def get_current_stock_info(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "regularMarketPrice" not in info:
            return None
        return info
    except Exception as e:
        st.error(f"Error fetching current info for {ticker}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_index_data(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m", prepost=False)

        if not data.empty:
            latest_price = data["Close"].iloc[-1]
            current_open = data["Open"].iloc[0]
            change = latest_price - current_open
            percent_change = (change / current_open) * 100 if current_open else 0
            return latest_price, change, percent_change
        else:
            data_daily = ticker.history(period="2d")
            if not data_daily.empty and len(data_daily) >= 2:
                latest_price = data_daily["Close"].iloc[-1]
                prev_close = data_daily["Close"].iloc[-2]
                change = latest_price - prev_close
                percent_change = (change / prev_close) * 100 if prev_close else 0
                return latest_price, change, percent_change
            return None, None, None
    except Exception as e:
        st.error(f"Could not fetch data for {symbol}: {e}")
        return None, None, None


@st.cache_data(ttl=300)
def get_news_headlines(ticker: str, limit: int = 10):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return []
        return [item.get("title", "") for item in news[:limit] if item.get("title")]
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_earnings_calendar(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if not calendar:
            return None
        return {
            "earnings_date": (
                calendar.get("Earnings Date", ["N/A"])[0]
                if isinstance(calendar.get("Earnings Date"), list)
                else calendar.get("Earnings Date", "N/A")
            ),
            "revenue_estimate": calendar.get("Revenue Estimate", "N/A"),
            "eps_estimate": calendar.get("EPS Estimate", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_advanced_financials(ticker: str):
    info = get_current_stock_info(ticker)
    if not info:
        return pd.DataFrame()

    metrics = {
        "ROE (%)": info.get("returnOnEquity", "N/A"),
        "Debt/Equity": info.get("debtToEquity", "N/A"),
        "Revenue Growth (YoY)": info.get("revenueGrowth", "N/A"),
        "Earnings Growth (YoY)": info.get("earningsGrowth", "N/A"),
        "Operating Cash Flow": info.get("operatingCashFlow", "N/A"),
        "Free Cash Flow": info.get("freeCashFlow", "N/A"),
    }

    formatted = {}
    for k, v in metrics.items():
        if v == "N/A":
            formatted[k] = "N/A"
        elif k in ("ROE (%)", "Revenue Growth (YoY)", "Earnings Growth (YoY)"):
            formatted[k] = f"{round(float(v) * 100, 2)}%"
        elif k in ("Operating Cash Flow", "Free Cash Flow"):
            formatted[k] = format_large_number(float(v))
        elif k == "Debt/Equity":
            formatted[k] = round(float(v), 2)
        else:
            formatted[k] = v

    df = pd.DataFrame(list(formatted.items()), columns=["Metric", "Value"])
    return df


def calculate_portfolio_metrics(tickers: list, weights: list, period: str = "1y"):
    returns = []
    for ticker in tickers:
        df = get_stock_data(ticker, period)
        if df is not None:
            daily_return = df["Close"].pct_change().dropna()
            returns.append(daily_return)

    if len(returns) < 2:
        return {"sharpe_ratio": 0, "annual_return": 0, "volatility": 0}

    portfolio_return = sum(w * r for w, r in zip(weights, returns))
    sharpe = (portfolio_return.mean() * 252) / (portfolio_return.std() * np.sqrt(252))
    annual_return = portfolio_return.mean() * 252
    volatility = portfolio_return.std() * np.sqrt(252)

    return {
        "sharpe_ratio": round(float(sharpe), 3),
        "annual_return": round(float(annual_return) * 100, 2),
        "volatility": round(float(volatility) * 100, 2),
    }


def calculate_correlation(tickers: list, period: str = "1y"):
    price_data = {}
    for ticker in tickers:
        df = get_stock_data(ticker, period)
        if df is not None and not df.empty:
            price_data[ticker] = df["Close"].pct_change().dropna()
    if not price_data:
        return pd.DataFrame()
    return pd.DataFrame(price_data).corr().round(3)


def calculate_beta(ticker: str, benchmark: str = "^NSEI", period: str = "1y"):
    stock_ret = get_stock_data(ticker, period)
    bench_ret = get_stock_data(benchmark, period)
    if stock_ret is None or bench_ret is None:
        return None
    stock_ret = stock_ret["Close"].pct_change().dropna()
    bench_ret = bench_ret["Close"].pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if aligned.empty:
        return None
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return None
    return round(cov / var, 3)


def format_large_number(num):
    if num is None:
        return "N/A"
    if abs(num) >= 1e7:
        return f"{num / 1e7:,.2f} Cr"
    elif abs(num) >= 1e5:
        return f"{num / 1e5:,.2f} L"
    return f"{num:,.2f}"
