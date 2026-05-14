import streamlit as st
from datetime import datetime
from modules.stock_data import get_current_stock_info


def init_watchlist():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}


def add_to_watchlist(ticker: str, alert_price: float, direction: str = "above"):
    init_watchlist()
    st.session_state.watchlist[ticker] = {
        "alert_price": alert_price,
        "direction": direction,
        "added_at": datetime.now().isoformat(),
    }


def remove_from_watchlist(ticker: str):
    init_watchlist()
    st.session_state.watchlist.pop(ticker, None)


def check_alerts():
    init_watchlist()
    triggered = []
    to_remove = []
    for ticker, config in st.session_state.watchlist.items():
        info = get_current_stock_info(ticker)
        if not info:
            continue
        current = info.get("regularMarketPrice", 0)
        if current == 0:
            continue
        alert = config["alert_price"]
        direction = config["direction"]
        if direction == "above" and current >= alert:
            triggered.append((ticker, current, alert, "above"))
            to_remove.append(ticker)
        elif direction == "below" and current <= alert:
            triggered.append((ticker, current, alert, "below"))
            to_remove.append(ticker)

    for t in to_remove:
        remove_from_watchlist(t)
    return triggered


def get_watchlist_prices():
    init_watchlist()
    results = []
    for ticker in st.session_state.watchlist:
        info = get_current_stock_info(ticker)
        price = info.get("regularMarketPrice", "N/A") if info else "N/A"
        results.append({
            "Ticker": ticker,
            "Current Price": price,
            "Alert": st.session_state.watchlist[ticker]["alert_price"],
            "Direction": st.session_state.watchlist[ticker]["direction"],
        })
    return results
