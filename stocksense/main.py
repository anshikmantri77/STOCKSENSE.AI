import streamlit as st
import pandas as pd
import warnings
from datetime import datetime, timedelta

from config import stock_analyzer, all_stock_symbols, INDIAN_INDICES, TOP_MOVERS_SAMPLE
from modules.stock_data import (
    get_stock_data,
    get_current_stock_info,
    fetch_index_data,
    get_news_headlines,
    get_advanced_financials,
    get_earnings_calendar,
    calculate_correlation,
    calculate_beta,
    calculate_portfolio_metrics,
    format_large_number,
)
from modules.sentiment import get_sentiment
from modules.ai_agent import Chatbot
from modules.portfolio import PortfolioBuilder
from modules.screener import run_screener
from modules.charts import candlestick_chart, portfolio_pie_chart
from modules.technical_indicators import add_technical_indicators, create_technical_chart
from modules.watchlist import init_watchlist, add_to_watchlist, remove_from_watchlist, check_alerts, get_watchlist_prices
from utils.helpers import color_change_val, export_portfolio_csv

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DARK THEME CSS ---
st.markdown(
    """
<style>
    .stApp { background: #0A0E1A; color: #F9FAFB; }
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1F2937;
    }
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #00D4AA, #3498DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center; color: #6B7280; font-size: 1rem; margin-bottom: 2rem;
    }
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1rem;
    }
    .positive { color: #00D4AA !important; }
    .negative { color: #FF4757 !important; }
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #6B7280; border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: #1F2937; color: #F9FAFB;
    }
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background: #1F2937;
        border: 1px solid #374151;
        color: #F9FAFB;
        border-radius: 8px;
    }
    .stButton button {
        background: #00D4AA; color: #0A0E1A; border: none;
        border-radius: 8px; font-weight: 600;
    }
    .stButton button:hover { opacity: 0.9; }
    .stDataFrame, .stTable { background: transparent; }
    .js-plotly-plot .plotly { background: transparent !important; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #F9FAFB;
        border-left: 3px solid #00D4AA; padding-left: 12px;
        margin: 1.5rem 0 1rem;
    }
    .badge-positive { background: #064E3B; color: #00D4AA; padding: 4px 12px; border-radius: 20px; font-size: 12px; }
    .badge-negative { background: #450A0A; color: #FF4757; padding: 4px 12px; border-radius: 20px; font-size: 12px; }
    .badge-neutral  { background: #1F2937; color: #9CA3AF; padding: 4px 12px; border-radius: 20px; font-size: 12px; }
    .sidebar-chatbot { margin-top: auto; padding-top: 20px; border-top: 1px solid #1F2937; }
    .sidebar-footer { text-align: center; font-size: 0.85rem; color: #6B7280; }
    .sidebar-footer a { color: #00D4AA; text-decoration: none; }
    h1, h2, h3, h4, h5, h6 { color: #F9FAFB; }
    .stAlert { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-header'>StockSense AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Real-time analytics · FinBERT sentiment · AI-powered insights</p>",
    unsafe_allow_html=True,
)

init_watchlist()

# Check for price alerts
triggered = check_alerts()
for ticker, current, alert, direction in triggered:
    st.toast(
        f"🔔 {ticker} hit ₹{current:,.2f} ({direction} ₹{alert:,.2f})",
        icon="🚨",
    )

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("### 📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Stock Screener", "Portfolio Builder", "Earnings Calendar", "Stock Comparison", "AI Reports"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Select a Stock")

selected_stock_symbol_from_list = st.sidebar.selectbox(
    "Choose a stock:",
    all_stock_symbols,
    index=all_stock_symbols.index("RELIANCE.NS") if "RELIANCE.NS" in all_stock_symbols else 0,
    label_visibility="collapsed",
)

custom_symbol = st.sidebar.text_input("Or enter NSE Symbol:", value="").upper()
selected_stock_symbol = selected_stock_symbol_from_list
if custom_symbol:
    selected_stock_symbol = custom_symbol if custom_symbol.endswith(".NS") else f"{custom_symbol}.NS"

# --- WATCHLIST IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Watchlist")
wl = get_watchlist_prices()
if wl:
    for item in wl:
        col_a, col_b = st.sidebar.columns([3, 1])
        col_a.write(f"**{item['Ticker']}** ₹{item['Current Price']}")
        if col_b.button("✕", key=f"remove_{item['Ticker']}", help="Remove"):
            remove_from_watchlist(item["Ticker"])
            st.rerun()
else:
    st.sidebar.info("Your watchlist is empty.", icon="👁️")

st.sidebar.markdown("---")

# --- CHATBOT IN SIDEBAR ---
st.sidebar.markdown("<div class='sidebar-chatbot'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🤖 AI Assistant")

if "chatbot_instance" not in st.session_state:
    st.session_state.chatbot_instance = Chatbot()

for message in st.session_state.get("chat_history", []):
    with st.sidebar.chat_message(message["role"]):
        st.sidebar.markdown(message["content"])

user_query = st.sidebar.chat_input("Ask about stocks, finance...", key="chat_input")

if user_query:
    st.session_state.setdefault("chat_history", []).append(
        {"role": "user", "content": user_query}
    )
    with st.sidebar.chat_message("user"):
        st.sidebar.markdown(user_query)
    with st.sidebar.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 Thinking...")
        response = st.session_state.chatbot_instance.get_response(user_query)
        placeholder.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# PAGE: DASHBOARD
# ================================================================
if page == "Dashboard":
    temp_info = get_current_stock_info(selected_stock_symbol)
    display_name = selected_stock_symbol
    if temp_info:
        display_name = temp_info.get("shortName") or temp_info.get("longName") or selected_stock_symbol

    st.header(f"📈 {display_name} ({selected_stock_symbol})")

    stock_data = get_stock_data(selected_stock_symbol)
    stock_info = get_current_stock_info(selected_stock_symbol)

    if stock_data is not None and stock_info is not None:
        curr_price = stock_info.get("regularMarketPrice", "N/A")
        prev_close = stock_info.get("previousClose", curr_price)

        if curr_price != "N/A" and prev_close != "N/A":
            change = curr_price - prev_close
            pct = (change / prev_close) * 100 if prev_close else 0
            cls = "positive" if change > 0 else "negative"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"₹{curr_price:,.2f}", f"{change:,.2f} ({pct:,.2f}%)")
            m2.metric("Market Cap", format_large_number(stock_info.get("marketCap")))
            pe_val = stock_info.get("trailingPE", "N/A")
            m3.metric("P/E Ratio", f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else "N/A")
            sentiment_val = "N/A"
            headlines = get_news_headlines(selected_stock_symbol, limit=3)
            if headlines:
                sent = get_sentiment(headlines)
                sentiment_val = f"{sent['positive']*100:.0f}% Pos"
            m4.metric("Sentiment", sentiment_val)

        st.markdown("<div class='section-header'>Price Chart & Technical Indicators</div>", unsafe_allow_html=True)
        show_tech = st.checkbox("Show Technical Indicators (MA, RSI, MACD, Bollinger)", value=False)

        if show_tech:
            ta_df = add_technical_indicators(stock_data)
            tech_fig = create_technical_chart(ta_df, selected_stock_symbol)
            st.plotly_chart(tech_fig, width='stretch')
        else:
            fig = candlestick_chart(stock_data, selected_stock_symbol)
            st.plotly_chart(fig, width='stretch')

        st.markdown("<div class='section-header'>Key Financial Metrics</div>", unsafe_allow_html=True)
        pe_val = stock_info.get("trailingPE", "N/A")
        fpe_val = stock_info.get("forwardPE", "N/A")
        roe_val = stock_info.get("returnOnEquity")
        de_val = stock_info.get("debtToEquity")
        div_val = stock_info.get("dividendYield")
        beta_val = stock_info.get("beta")
        vol_val = stock_info.get("regularMarketVolume")
        metrics_data = {
            "P/E Ratio": f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else "N/A",
            "Forward P/E": f"{fpe_val:.2f}" if isinstance(fpe_val, (int, float)) else "N/A",
            "Market Cap": format_large_number(stock_info.get("marketCap")),
            "ROE": f"{round(roe_val * 100, 2)}%" if roe_val else "N/A",
            "Debt/Equity": f"{de_val:.2f}" if de_val else "N/A",
            "Dividend Yield": f"{round(div_val * 100, 2)}%" if div_val else "N/A",
            "52W High": f"₹{stock_info.get('fiftyTwoWeekHigh', 0):,.2f}",
            "52W Low": f"₹{stock_info.get('fiftyTwoWeekLow', 0):,.2f}",
            "Volume": f"{vol_val:,.0f}" if vol_val else "N/A",
            "Beta": f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A",
        }
        metrics_df = pd.DataFrame(metrics_data.items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df.set_index("Metric"), width='stretch')

        st.markdown("<div class='section-header'>News & FinBERT Sentiment</div>", unsafe_allow_html=True)
        with st.spinner("Analyzing sentiment..."):
            headlines = get_news_headlines(selected_stock_symbol)
        if headlines:
            sentiment = get_sentiment(headlines)
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", f"{sentiment['positive']*100:.1f}%")
            c2.metric("Negative", f"{sentiment['negative']*100:.1f}%")
            c3.metric("Neutral", f"{sentiment['neutral']*100:.1f}%")
            news_df = pd.DataFrame(headlines, columns=["Headline"])
            st.dataframe(news_df, width='stretch')
        else:
            st.info("No news headlines available.")

        st.markdown("<div class='section-header'>Advanced Financials</div>", unsafe_allow_html=True)
        adv_df = get_advanced_financials(selected_stock_symbol)
        if not adv_df.empty:
            st.dataframe(adv_df, width='stretch')
        else:
            st.info("No advanced financial data available.")
    else:
        st.error("Could not load data. Please check the symbol or try again later.")

    # --- WATCHLIST CONTROLS ---
    st.markdown("---")
    with st.expander("Add to Watchlist"):
        wl_ticker = st.text_input("Ticker", value=selected_stock_symbol, key="wl_ticker")
        wl_price = st.number_input("Alert Price (₹)", min_value=0.0, value=1000.0, step=50.0)
        wl_dir = st.selectbox("Direction", ["above", "below"])
        if st.button("Add to Watchlist"):
            add_to_watchlist(
                wl_ticker if wl_ticker.endswith(".NS") else f"{wl_ticker}.NS",
                wl_price, wl_dir,
            )
            st.success(f"Added {wl_ticker} to watchlist!")
            st.rerun()

    # --- MARKET OVERVIEW ---
    st.markdown("<div class='section-header'>Market Overview</div>", unsafe_allow_html=True)
    mkt_data = []
    for name, symbol in INDIAN_INDICES.items():
        price, change, pct = fetch_index_data(symbol)
        if price is not None:
            mkt_data.append({"Index": name, "Price": f"{price:,.2f}", "Change": change, "% Change": f"{pct:,.2f}%"})
    if mkt_data:
        df_mkt = pd.DataFrame(mkt_data)
        df_mkt["Change"] = pd.to_numeric(df_mkt["Change"], errors="coerce")
        st.dataframe(df_mkt.style.applymap(color_change_val, subset=["Change"]), hide_index=True, width='stretch')

    st.markdown("<div class='section-header'>Top Market Movers</div>", unsafe_allow_html=True)
    movers = []
    for sym in TOP_MOVERS_SAMPLE:
        info = get_current_stock_info(sym)
        if info:
            price = info.get("regularMarketPrice")
            pc = info.get("previousClose")
            if price and pc:
                chg = price - pc
                movers.append({"Symbol": sym, "Price": f"{price:,.2f}", "Change": chg, "% Change": f"{(chg/pc)*100:,.2f}%"})
    if movers:
        df_mv = pd.DataFrame(movers)
        df_mv["Cn"] = pd.to_numeric(df_mv["Change"], errors="coerce")
        cg, cl = st.columns(2)
        with cg:
            st.markdown("##### Top Gainers")
            st.dataframe(df_mv.nlargest(5, "Cn").drop(columns=["Cn"]), hide_index=True, width='stretch')
        with cl:
            st.markdown("##### Top Losers")
            st.dataframe(df_mv.nsmallest(5, "Cn").drop(columns=["Cn"]), hide_index=True, width='stretch')

# ================================================================
# PAGE: STOCK SCREENER
# ================================================================
elif page == "Stock Screener":
    st.header("🔍 Custom Stock Screener")
    st.markdown("Filter stocks based on real financial criteria:")

    with st.spinner("Loading screener..."):
        cf1, cf2 = st.columns(2)
        with cf1:
            max_pe = st.slider("Max P/E Ratio", 0.0, 300.0, 150.0, 0.1)
            min_roe = st.slider("Min ROE (%)", 0.0, 100.0, 10.0, 0.5)
            max_de = st.slider("Max Debt/Equity", 0.0, 10.0, 2.0, 0.1)
        with cf2:
            min_mcap = st.slider("Min Market Cap (Cr)", 0.0, 500000.0, 1000.0, 100.0)
            min_rev = st.slider("Min Revenue Growth (%)", -50.0, 100.0, 0.0, 1.0)
            min_earn = st.slider("Min Earnings Growth (%)", -50.0, 100.0, 0.0, 1.0)

        cap_filter = st.selectbox("Market Cap Category", ["All", "Large Cap", "Mid Cap", "Small Cap"])
        if cap_filter == "All":
            tickers = stock_analyzer.get_all_stock_symbols()[:80]
        else:
            tickers = stock_analyzer.get_stock_by_category(cap_filter)[:80]

        s_filters = {
            "max_pe": max_pe, "min_roe": min_roe / 100, "max_de": max_de,
            "min_market_cap": min_mcap * 1e7, "min_rev_growth": min_rev / 100,
            "min_earn_growth": min_earn / 100,
        }
        screener_df = run_screener(tickers, s_filters)

    if not screener_df.empty:
        st.subheader(f"Matching Stocks ({len(screener_df)} found)")
        st.dataframe(screener_df, width='stretch')

        csv_data = export_portfolio_csv(screener_df)
        st.download_button(
            "📥 Export Results as CSV",
            data=csv_data,
            file_name=f"screener_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No stocks match your criteria. Try adjusting the filters.")

# ================================================================
# PAGE: PORTFOLIO BUILDER
# ================================================================
elif page == "Portfolio Builder":
    st.header("💰 Portfolio Builder")
    pb = PortfolioBuilder(stock_analyzer)

    st.markdown("<div class='section-header'>1. Risk-Based Asset Allocation</div>", unsafe_allow_html=True)
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
    allocation = pb.get_asset_allocation(risk_profile)
    allowed = {"Large Cap", "Mid Cap", "Small Cap"}
    allocation = {k: v for k, v in allocation.items() if k in allowed}

    if allocation:
        alloc_df = pd.DataFrame(allocation.items(), columns=["Asset Class", "Allocation (%)"])
        alloc_df["Allocation (%)"] *= 100
        st.dataframe(alloc_df.set_index("Asset Class"), width='stretch')
        st.plotly_chart(portfolio_pie_chart(alloc_df), width='stretch')

    st.markdown("<div class='section-header'>2. Investment Projection</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        initial = st.number_input("One-time (₹)", min_value=0, value=100000, step=10000)
    with c2:
        sip = st.number_input("Monthly SIP (₹)", min_value=0, value=5000, step=1000)
    years = st.slider("Duration (Years)", 1, 30, 10)

    if st.button("Project Investment"):
        projected = pb.project_investment(initial, sip, years, risk_profile)
        st.metric(f"Estimated Value in {years} Years", f"₹{projected:,.2f}")

        total_invested = initial + (sip * 12 * years)
        returns = projected - total_invested
        st.info(
            f"Total invested: ₹{total_invested:,.2f} | "
            f"Estimated returns: ₹{returns:,.2f} "
            f"({(returns/total_invested)*100:.1f}% gain)"
        )

    st.markdown("<div class='section-header'>3. Stock Suggestions</div>", unsafe_allow_html=True)
    suggestions = pb.get_stock_suggestions(risk_profile)
    rows = []
    for asset, stocks in suggestions.items():
        for s in stocks:
            info = get_current_stock_info(s)
            price = info.get("regularMarketPrice", "N/A") if info else "N/A"
            rows.append({"Category": asset, "Ticker": s, "Current Price": f"₹{price:,.2f}" if isinstance(price, (int, float)) else price})
    if rows:
        sugg_df = pd.DataFrame(rows)
        st.dataframe(sugg_df, width='stretch')

        csv_data = export_portfolio_csv(sugg_df)
        st.download_button(
            "📥 Export Suggestions as CSV",
            data=csv_data,
            file_name=f"portfolio_suggestions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ================================================================
# PAGE: EARNINGS CALENDAR
# ================================================================
elif page == "Earnings Calendar":
    st.header("🗓️ Earnings Calendar")
    st.markdown("---")

    with st.spinner("Fetching earnings data..."):
        sample_symbols = all_stock_symbols[:30]
        rows = []
        for sym in sample_symbols:
            info = get_current_stock_info(sym)
            company = (info.get("shortName") or info.get("longName") or sym.replace(".NS", "")) if info else sym.replace(".NS", "")
            cal = get_earnings_calendar(sym)
            if cal and "earnings_date" in cal and cal["earnings_date"] != "N/A" and "error" not in cal:
                try:
                    ed = cal["earnings_date"]
                    if hasattr(ed, "strftime"):
                        ed_str = ed.strftime("%Y-%m-%d")
                    else:
                        ed_str = str(ed)[:10]
                    rows.append({
                        "Company": company, "Symbol": sym,
                        "Date": ed_str,
                        "Revenue Est.": cal.get("revenue_estimate", "N/A"),
                        "EPS Est.": cal.get("eps_estimate", "N/A"),
                    })
                except Exception:
                    continue

        if rows:
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date").reset_index(drop=True)
            st.subheader("Upcoming Earnings")
            st.dataframe(df, width='stretch')
        else:
            st.info("No upcoming earnings data available from yfinance for the selected stocks.")

# ================================================================
# PAGE: STOCK COMPARISON
# ================================================================
elif page == "Stock Comparison":
    st.header("⚖️ Stock Comparison Tool")
    st.markdown("Compare up to 3 stocks side-by-side with correlation and beta.")

    opts = all_stock_symbols
    defaults = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    idxs = [opts.index(s) for s in defaults if s in opts]

    c1, c2, c3 = st.columns(3)
    with c1:
        s1 = st.selectbox("Stock 1", opts, index=idxs[0] if len(idxs) > 0 else 0, key="cmp1")
    with c2:
        s2 = st.selectbox("Stock 2", opts, index=idxs[1] if len(idxs) > 1 else min(1, len(opts) - 1), key="cmp2")
    with c3:
        s3 = st.selectbox("Stock 3", opts, index=idxs[2] if len(idxs) > 2 else min(2, len(opts) - 1), key="cmp3")

    symbols = [s1, s2, s3]
    metrics = ["regularMarketPrice", "trailingPE", "marketCap", "dividendYield",
               "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta", "sector", "industry"]
    labels = {"regularMarketPrice": "Price", "trailingPE": "P/E", "marketCap": "Market Cap",
              "dividendYield": "Div. Yield", "fiftyTwoWeekHigh": "52W High",
              "fiftyTwoWeekLow": "52W Low", "beta": "Beta", "sector": "Sector", "industry": "Industry"}

    rows = []
    for sym in symbols:
        info = get_current_stock_info(sym)
        if info:
            name = info.get("shortName") or info.get("longName") or sym
            row = {"Symbol": name}
            for m in metrics:
                val = info.get(m, "N/A")
                if m == "marketCap" and val != "N/A":
                    val = format_large_number(val)
                elif m == "dividendYield" and val != "N/A":
                    val = f"{val * 100:.2f}%"
                elif isinstance(val, (int, float)):
                    val = f"{val:,.2f}"
                row[labels[m]] = val
            rows.append(row)

    if rows:
        comp_df = pd.DataFrame(rows)
        st.subheader("Comparison")
        st.dataframe(comp_df.set_index("Symbol"), width='stretch')

        st.subheader("Correlation Matrix (1Y Returns)")
        with st.spinner("Calculating..."):
            corr = calculate_correlation(symbols)
            if not corr.empty:
                st.dataframe(corr, width='stretch')

        st.subheader("Beta (vs NIFTY 50)")
        for sym in symbols:
            beta = calculate_beta(sym)
            st.write(f"**{sym}**: {beta:.3f}" if beta else f"**{sym}**: N/A")

# ================================================================
# PAGE: AI REPORTS
# ================================================================
elif page == "AI Reports":
    st.header("📝 AI-Generated Stock Reports")
    st.markdown("Generate comprehensive investment analysis with real data + Gemini.")

    report_sym = st.selectbox("Select a stock:", all_stock_symbols, key="rpt_sym")
    rpt_info = get_current_stock_info(report_sym)
    rpt_name = report_sym
    if rpt_info:
        rpt_name = rpt_info.get("shortName") or rpt_info.get("longName") or report_sym

    use_gemini = st.checkbox("Use Gemini AI for report generation (requires API key)", value=False)

    if st.button(f"Generate Report for {rpt_name}"):
        with st.spinner("Gathering data and generating report..."):
            info = get_current_stock_info(report_sym)
            headlines = get_news_headlines(report_sym, limit=5)

            if not info:
                st.error("Could not fetch data.")
                st.stop()

            if use_gemini:
                sentiment = get_sentiment(headlines) if headlines else {"positive": 0, "negative": 0, "neutral": 1}
                prompt = f"""
Generate a concise investment analysis report for {rpt_name} ({report_sym}).

Financial Data:
- Current Price: ₹{info.get('regularMarketPrice', 'N/A')}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Forward P/E: {info.get('forwardPE', 'N/A')}
- Market Cap: {info.get('marketCap', 'N/A')}
- 52W High/Low: ₹{info.get('fiftyTwoWeekHigh', 'N/A')} / ₹{info.get('fiftyTwoWeekLow', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 'N/A')}
- Profit Margin: {info.get('profitMargins', 'N/A')}
- ROE: {info.get('returnOnEquity', 'N/A')}
- Beta: {info.get('beta', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}

Recent News Sentiment (FinBERT):
- Positive: {sentiment['positive']:.1%}
- Negative: {sentiment['negative']:.1%}
- Neutral: {sentiment['neutral']:.1%}

Recent Headlines:
{chr(10).join(f'- {h}' for h in headlines[:5])}

Provide: 1) Executive Summary 2) Key Strengths 3) Key Risks 4) Sentiment Assessment.
Keep under 300 words. Do not give investment advice.
"""
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(google_api_key=st.secrets.get("GOOGLE_API_KEY"), temperature=0.3)
                    report = llm.predict(prompt)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.info("Falling back to template-based report...")
                    use_gemini = False

            if not use_gemini:
                report = f"**Executive Summary for {rpt_name} ({report_sym}):**\n\n"
                price = info.get("regularMarketPrice", "N/A")
                pe = info.get("trailingPE", "N/A")
                roe = info.get("returnOnEquity", "N/A")
                rev = info.get("revenueGrowth", "N/A")
                mcap = format_large_number(info.get("marketCap"))
                div_y = f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "N/A"

                report += f"- Current Price: ₹{price:,.2f}\n" if isinstance(price, (int, float)) else f"- Current Price: {price}\n"
                report += f"- P/E Ratio: {pe}\n"
                report += f"- Market Cap: {mcap}\n"
                report += f"- Dividend Yield: {div_y}\n"
                report += f"- Sector: {info.get('sector', 'N/A')}\n"
                report += f"- ROE: {round(roe*100, 2)}%\n" if isinstance(roe, float) else f"- ROE: {roe}\n"
                report += f"- Revenue Growth: {round(rev*100, 2)}%\n" if isinstance(rev, float) else f"- Revenue Growth: {rev}\n"

                report += "\n**Key Highlights:**\n"
                if isinstance(pe, (int, float)):
                    report += f"- {'Reasonable' if pe < 25 else 'Elevated'} P/E ratio ({pe:.1f}).\n"
                if isinstance(roe, float):
                    report += f"- {'Strong' if roe > 0.15 else 'Moderate'} ROE ({roe*100:.1f}%).\n"
                if isinstance(rev, float) and rev > 0:
                    report += "- Positive revenue growth signals business expansion.\n"

                if headlines:
                    sentiment = get_sentiment(headlines)
                    report += f"\n**FinBERT Sentiment:** Positive {sentiment['positive']*100:.1f}% | Negative {sentiment['negative']*100:.1f}% | Neutral {sentiment['neutral']*100:.1f}%\n"
                    for h in headlines[:3]:
                        report += f"- {h}\n"

                report += "\n\n---\n**Disclaimer:** Not financial advice. For educational purposes only."
                st.markdown(report)

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ Not financial advice. For educational purposes only."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <p><b>StockSense AI</b> by Anshik Mantri</p>
        <p><a href="mailto:anshikmantri26@gmail.com">Email</a> · <a href="http://www.linkedin.com/in/anshikmantri" target="_blank">LinkedIn</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
