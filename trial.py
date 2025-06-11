import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import warnings
import random
import pytz
# New imports for features
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from twilio.rest import Client # For SMS alerts
# from sendgrid import SendGridAPIClient # For email alerts
# from sendgrid.helpers.mail import Mail

warnings.filterwarnings('ignore')

# Correction: Avoid reassigning the entire pytz module.
# If you need a timezone object, assign it to a new variable.
kolkata_tz = pytz.timezone('Asia/Kolkata')

# Page configuration
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #eee;
        border-radius: 5px 5px 0 0;
        padding: 10px 15px;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #ddd;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #007bff;
        color: white;
        border-bottom: 3px solid #0056b3;
    }
    .sidebar-footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>StockSense AI 📈</h1>", unsafe_allow_html=True)

# --- GLOBAL DATA / CACHING ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_nifty_50_stocks():
    # This is a sample list. For production, you might want to fetch a live list.
    nifty_50_symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS", "SBIN.NS",
        "BAJFINANCE.NS", "AXISBANK.NS", "KOTAKBANK.NS", "ASIANPAINT.NS",
        "MARUTI.NS", "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "POWERGRID.NS",
        "ADANIPORTS.NS", "NTPC.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "ONGC.NS",
        "WIPRO.NS", "JSWSTEEL.NS", "GRASIM.NS", "TECHM.NS", "HCLTECH.NS",
        "INDUSINDBK.NS", "EICHERMOT.NS", "M&M.NS", "DRREDDY.NS", "BPCL.NS",
        "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "ADANIENT.NS",
        "APOLLOHOSP.NS", "UPL.NS", "COALINDIA.NS", "SHREECEM.NS", "DIVISLAB.NS",
        "HDFC.NS", "BRITANNIA.NS", "CIPLA.NS", "GAIL.NS", "HINDALCO.NS",
        "IOC.NS", "LTIM.NS"
    ]
    return sorted(nifty_50_symbols)

NIFTY_50_SYMBOLS = get_nifty_50_stocks()

# --- Helper Functions ---
@st.cache_data(ttl=300) # Cache stock info for 5 minutes
def get_stock_info(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="1y") # Get 1 year historical data for chart
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None, None

def get_nifty_indices_data():
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK"
    }
    data = []
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d", interval="1m").iloc[-1] # Latest minute data
            previous_close = ticker.history(period="1d").iloc[-2]['Close'] # Previous day's close
            current_price = current_data['Close']
            change = current_price - previous_close
            percent_change = (change / previous_close) * 100
            data.append({"Index": name, "Price": current_price, "Change": change, "Percent Change": percent_change})
        except Exception as e:
            st.warning(f"Could not fetch data for {name}: {e}")
            data.append({"Index": name, "Price": "N/A", "Change": "N/A", "Percent Change": "N/A"})
    return pd.DataFrame(data)

def generate_random_news(stock_symbol):
    headlines = [
        f"{stock_symbol} sees strong investor interest amid sector growth.",
        f"Analysts positive on {stock_symbol} due to robust earnings outlook.",
        f"New product launch boosts {stock_symbol}'s market position.",
        f"{stock_symbol} faces minor setback in Q3 earnings.",
        f"Regulatory changes could impact {stock_symbol}'s future operations.",
        f"{stock_symbol} stock stabilizes after recent volatility.",
        f"Partnership announcement drives {stock_symbol} shares up.",
        f"Market sentiment shifts cautious on {stock_symbol}.",
        f"Innovation drives {stock_symbol} towards new highs.",
        f"{stock_symbol} dividend declaration excites shareholders.",
    ]
    return random.sample(headlines, random.randint(2, 4)) # Get 2-4 random headlines

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# --- Portfolio Builder Class ---
class PortfolioBuilder:
    def __init__(self):
        # Average annual returns (hypothetical, for projection)
        self.average_annual_returns = {
            'Large Cap': 0.12, # 12%
            'Mid Cap': 0.15,   # 15%
            'Small Cap': 0.18, # 18%
            'Debt': 0.07,      # 7%
            'Gold': 0.08       # 8%
        }

    def get_risk_based_allocation(self, risk_profile):
        allocations = {
            "Conservative": {
                'Large Cap': 0.40, 'Mid Cap': 0.10, 'Small Cap': 0.00,
                'Debt': 0.40, 'Gold': 0.10
            },
            "Moderate": {
                'Large Cap': 0.35, 'Mid Cap': 0.20, 'Small Cap': 0.05,
                'Debt': 0.30, 'Gold': 0.10
            },
            "Aggressive": {
                'Large Cap': 0.25, 'Mid Cap': 0.25, 'Small Cap': 0.20,
                'Debt': 0.20, 'Gold': 0.10
            }
        }
        return allocations.get(risk_profile, {})

    def project_investment_value(self, initial_investment, monthly_sip, duration_years, allocation):
        total_projected_value = 0
        for asset_class, percentage in allocation.items():
            annual_return = self.average_annual_returns[asset_class]
            monthly_return = (1 + annual_return)**(1/12) - 1

            # Future value of initial lump sum
            fv_lump_sum = initial_investment * percentage * (1 + annual_return)**duration_years

            # Future value of SIP (Future Value of Annuity formula)
            fv_sip = monthly_sip * percentage * (((1 + monthly_return)**(duration_years * 12) - 1) / monthly_return)

            total_projected_value += (fv_lump_sum + fv_sip)
        return total_projected_value

    def suggest_example_stocks(self, allocation):
        suggestions = {}
        # Dummy lists of stocks for each category
        large_cap_examples = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS"]
        mid_cap_examples = ["PIDILITIND.NS", "GODREJCP.NS", "BERGEPAINT.NS", "VOLTAS.NS", "ASTRAL.NS"]
        small_cap_examples = ["CDSL.NS", "DIXON.NS", "CAMS.NS", "ROUTE.NS", "AFFLE.NS"]

        if allocation.get('Large Cap', 0) > 0:
            suggestions['Large Cap'] = random.sample(large_cap_examples, min(3, len(large_cap_examples)))
        if allocation.get('Mid Cap', 0) > 0:
            suggestions['Mid Cap'] = random.sample(mid_cap_examples, min(2, len(mid_cap_examples)))
        if allocation.get('Small Cap', 0) > 0:
            suggestions['Small Cap'] = random.sample(small_cap_examples, min(1, len(small_cap_examples)))
        return suggestions

# --- New Feature Implementations ---

def conversational_finance_assistant_feature():
    st.header("💬 Conversational Finance Assistant")
    st.write("Ask me anything about stocks! (e.g., 'What's the P/E of TCS?', 'Which large-cap stocks have low debt?')")

    st.warning("⚠️ **Note:** This feature requires OpenAI API key and LangChain. The responses are simulated for demonstration.")

    # Initialize OpenAI LLM if API key is available
    # openai_api_key = st.secrets.get("OPENAI_API_KEY")
    # if openai_api_key:
    #     llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)
    # else:
    #     st.error("OpenAI API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    #     llm = None # Disable LLM if key is missing

    user_query = st.text_input("Your question:", key="chatbot_query")

    if user_query:
        with st.spinner("Thinking..."):
            # --- Conceptual LangChain + OpenAI Integration ---
            # In a real implementation, you'd create a robust LangChain agent
            # that can access financial data via tools (e.g., a tool to get P/E,
            # a tool to filter stocks by criteria).
            response = "I am a simulated AI assistant. For a real response, connect me to an LLM like OpenAI and provide financial data access."
            
            # Simple keyword-based simulation
            if "p/e of" in user_query.lower() or "pe of" in user_query.lower():
                stock_name = user_query.lower().replace("p/e of", "").replace("pe of", "").strip().upper().replace(" ", "")
                if stock_name.endswith(".NS"): # Try to normalize to common format
                    stock_symbol = stock_name
                else:
                    stock_symbol = stock_name + ".NS" # Assume NSE for Indian context
                
                info, _ = get_stock_info(stock_symbol)
                if info and 'trailingPE' in info:
                    response = f"The Trailing P/E Ratio of {stock_symbol} is approximately {info['trailingPE']:.2f}."
                else:
                    response = f"Could not find P/E ratio for {stock_symbol}. Please check the symbol or try again."
            elif "large-cap stocks with low debt" in user_query.lower():
                response = "Some large-cap stocks with historically lower debt include TCS, Infosys, and HCLTech. Always verify current financials."
            elif "dividend stock under" in user_query.lower():
                response = "Finding specific dividend stocks under a price point requires real-time screening. Some historical dividend payers include Coal India, REC Ltd, and Power Grid Corporation. Please do your own research."
            else:
                response = "I'm still learning about that specific financial query. Please try another question or rephrase it."

            st.markdown(f"**AI:** {response}")

def earnings_calendar_alerts_feature():
    st.header("📅 Earnings Calendar & Alerts")
    st.write("Stay informed about upcoming quarterly results.")

    # --- Dummy Earnings Data (Replace with API call in real app) ---
    today = datetime.now()
    earnings_data = pd.DataFrame({
        'Company': ['Reliance Industries', 'TCS', 'Infosys', 'HDFC Bank', 'ICICI Bank', 'Bajaj Finance', 'L&T'],
        'Symbol': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BAJFINANCE.NS', 'LT.NS'],
        'Earnings Date': [
            (today + timedelta(days=random.randint(5, 15))).strftime('%Y-%m-%d'),
            (today + timedelta(days=random.randint(10, 20))).strftime('%Y-%m-%d'),
            (today + timedelta(days=random.randint(15, 25))).strftime('%Y-%m-%d'),
            (today + timedelta(days=random.randint(20, 30))).strftime('%Y-%m-%d'),
            (today + timedelta(days=random.randint(25, 35))).strftime('%Y-%m-%d'),
            (today + timedelta(days=random.randint(30, 40))).strftime('%Y-%m-%d'),
            (today + timedelta(days=random.randint(35, 45))).strftime('%Y-%m-%d'),
        ],
        'Estimated EPS': [random.uniform(8.0, 15.0) for _ in range(7)],
        'Previous EPS': [random.uniform(7.0, 14.0) for _ in range(7)]
    })
    earnings_data['Earnings Date'] = pd.to_datetime(earnings_data['Earnings Date'])
    earnings_data = earnings_data.sort_values(by='Earnings Date').reset_index(drop=True)

    st.subheader("Upcoming Earnings Releases")
    st.dataframe(earnings_data.style.apply(lambda x: ['background-color: #d4edda' if x['Earnings Date'].date() == today.date() else '' for i in x], axis=1), use_container_width=True)

    st.subheader("Set Earnings Alerts")
    st.warning("⚠️ **Note:** Email/SMS alerts require setting up Twilio/SendGrid API keys and a backend service for real-time notifications. This is a frontend simulation.")

    alert_symbol = st.selectbox("Select stock for alert:", NIFTY_50_SYMBOLS)
    alert_type = st.radio("Alert via:", ("Email", "SMS"))
    contact_info = st.text_input(f"Enter your {alert_type} address:")

    if st.button("Set Alert for Selected Stock"):
        if contact_info:
            st.success(f"Alert for {alert_symbol} via {alert_type} to {contact_info} has been conceptually set!")
            # --- Actual Alert Logic (requires API keys and backend) ---
            # if alert_type == "Email":
            #     try:
            #         sg = SendGridAPIClient(st.secrets["SENDGRID_API_KEY"])
            #         message = Mail(
            #             from_email='no-reply@stocksenseai.com', # Replace with your verified sender
            #             to_emails=contact_info,
            #             subject=f'StockSense AI: Earnings Alert for {alert_symbol}',
            #             html_content=f'Hello, this is an alert from StockSense AI. Earnings for {alert_symbol} are approaching!'
            #         )
            #         response = sg.send(message)
            #         st.write(f"Email sent with status code: {response.status_code}")
            #     except Exception as e:
            #         st.error(f"Failed to send email: {e}")
            # elif alert_type == "SMS":
            #     try:
            #         account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
            #         auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
            #         twilio_phone_number = st.secrets["TWILIO_PHONE_NUMBER"]
            #         client = Client(account_sid, auth_token)
            #         message = client.messages.create(
            #             to=contact_info, # Must be a verified phone number for Twilio
            #             from_=twilio_phone_number,
            #             body=f'StockSense AI: Earnings for {alert_symbol} are approaching!'
            #         )
            #         st.write(f"SMS sent with SID: {message.sid}")
            #     except Exception as e:
            #         st.error(f"Failed to send SMS: {e}")
        else:
            st.warning(f"Please enter your {alert_type} address.")

def stock_comparison_tool_feature():
    st.header("📊 Stock Comparison Tool")
    st.write("Compare key financial metrics of up to 3 stocks side-by-side.")

    selected_stocks = st.multiselect(
        "Select Stocks to Compare (max 3):",
        options=NIFTY_50_SYMBOLS,
        default=NIFTY_50_SYMBOLS[:2] if len(NIFTY_50_SYMBOLS) >= 2 else []
    )

    if len(selected_stocks) > 3:
        st.warning("Please select a maximum of 3 stocks for comparison.")
        selected_stocks = selected_stocks[:3] # Trim to first 3

    if selected_stocks:
        comparison_data = {}
        metrics_to_compare = {
            'trailingPE': 'P/E Ratio',
            'marketCap': 'Market Cap',
            'beta': 'Beta',
            'bookValue': 'Book Value',
            'trailingEps': 'EPS (Trailing)',
            'priceToBook': 'Price/Book',
            'returnOnEquity': 'ROE',
            'returnOnAssets': 'ROA',
            'debtToEquity': 'Debt/Equity',
            'dividendYield': 'Dividend Yield',
            'fiftyTwoWeekHigh': '52 Week High',
            'fiftyTwoWeekLow': '52 Week Low',
            'volume': 'Volume',
            'averageVolume': 'Avg. Volume',
            'sector': 'Sector',
            'industry': 'Industry'
        }

        for stock_symbol in selected_stocks:
            info, _ = get_stock_info(stock_symbol)
            if info:
                stock_metrics = {}
                for api_key, display_name in metrics_to_compare.items():
                    value = info.get(api_key, 'N/A')
                    # Format specific metrics
                    if api_key in ['marketCap', 'volume', 'averageVolume'] and isinstance(value, (int, float)):
                        value = f"₹{value:,.0f}"
                    elif api_key in ['dividendYield'] and isinstance(value, (int, float)):
                        value = f"{value*100:.2f}%"
                    elif isinstance(value, (int, float)):
                        value = f"{value:.2f}"
                    stock_metrics[display_name] = value
                comparison_data[stock_symbol] = stock_metrics
            else:
                st.warning(f"Could not retrieve full data for {stock_symbol}. Skipping.")

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.subheader("Comparative Financial Metrics")
            st.dataframe(comparison_df.T, use_container_width=True) # Transpose for better readability

            # Optional: Radar Chart (requires more advanced data normalization)
            if len(selected_stocks) > 1:
                st.subheader("Visual Comparison (Radar Chart - Conceptual)")
                st.info("Radar chart requires careful normalization of diverse metrics for meaningful visualization. This is a placeholder.")
                # You would normalize metrics like P/E, ROE, Debt/Equity to a 0-1 scale
                # and then use plotly.graph_objects.Scatterpolar
                # Example: fig = go.Figure()
                # for stock in selected_stocks:
                #     fig.add_trace(go.Scatterpolar(
                #         r=[normalized_pe, normalized_roe, normalized_debt_equity],
                #         theta=['P/E', 'ROE', 'Debt/Equity'],
                #         fill='toself',
                #         name=stock
                #     ))
                # st.plotly_chart(fig)
        else:
            st.info("No data to compare. Please select valid stocks.")

def ai_generated_stock_reports_feature():
    st.header("✍️ AI-Generated Stock Reports")
    st.write("Automatically generate a concise financial summary of a stock.")

    st.warning("⚠️ **Note:** This feature uses a simulated LLM response for demonstration. A real implementation requires OpenAI API key.")

    stock_symbol = st.selectbox("Select a Stock for Report:", NIFTY_50_SYMBOLS, key="ai_report_symbol")

    if st.button(f"Generate AI Report for {stock_symbol}"):
        with st.spinner(f"Generating report for {stock_symbol}..."):
            info, hist = get_stock_info(stock_symbol)

            if info:
                # Extract key metrics
                pe_ratio = info.get('trailingPE', 'N/A')
                roe = info.get('returnOnEquity', 'N/A')
                debt_equity = info.get('debtToEquity', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                dividend_yield = info.get('dividendYield', 'N/A')
                sector = info.get('sector', 'N/A')
                current_price = info.get('currentPrice', 'N/A')
                
                # Simulate a basic sentiment based on metrics
                sentiment_score = 0
                if isinstance(roe, (int, float)) and roe > 0.15: sentiment_score += 1
                if isinstance(debt_equity, (int, float)) and debt_equity < 1: sentiment_score += 1
                if isinstance(dividend_yield, (int, float)) and dividend_yield > 0.01: sentiment_score += 1

                if sentiment_score >= 2:
                    overall_sentiment = "strong and positive"
                elif sentiment_score == 1:
                    overall_sentiment = "moderate"
                else:
                    overall_sentiment = "cautious"

                # --- Construct a detailed prompt for LLM (Conceptual) ---
                # This would be sent to OpenAI API
                prompt = f"""
                Generate a concise, natural-language financial summary for {stock_symbol}.
                Highlight its key strengths and weaknesses, overall financial health, and a brief outlook.
                Use the following data:
                - Current Price: ₹{current_price:,.2f}
                - P/E Ratio: {pe_ratio}
                - Return on Equity (ROE): {roe:.2%}
                - Debt to Equity Ratio: {debt_equity:.2f}
                - Market Cap: ₹{market_cap:,.0f}
                - Dividend Yield: {dividend_yield:.2%}
                - Sector: {sector}
                - Overall simulated sentiment: {overall_sentiment}
                """

                # --- Call LLM (Conceptual) ---
                # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                # try:
                #     response = client.chat.completions.create(
                #         model="gpt-3.5-turbo", # or "gpt-4"
                #         messages=[{"role": "user", "content": prompt}]
                #     )
                #     report_content = response.choices[0].message.content
                # except Exception as e:
                #     report_content = f"Error generating report: {e}. Please check your OpenAI API key or internet connection."
                #     st.warning("Using dummy report content.")

                # Dummy report content for demonstration
                report_content = f"""
                ### Financial Snapshot: {stock_symbol} ({sector} Sector)

                **{stock_symbol}** is currently trading at approximately **₹{current_price:,.2f}**.
                Its P/E Ratio stands at **{pe_ratio:.2f}**,
                while its Return on Equity (ROE) is **{roe:.2%}**, indicating {"efficient" if roe > 0.15 else "average"} use of shareholder funds.
                The Debt to Equity Ratio is **{debt_equity:.2f}**, suggesting {"manageable" if debt_equity < 1 else "higher"} leverage.
                With a Market Cap of **₹{market_cap:,.0f}**, it's a {"large-cap" if market_cap > 20000*10**7 else "mid/small-cap"} player.
                The stock also offers a Dividend Yield of **{dividend_yield:.2%}**, which may appeal to income-focused investors.

                Overall, the company's financial health appears **{overall_sentiment}**.
                Investors should monitor future earnings, sector trends, and market conditions for a complete view.
                """

                st.markdown(report_content)
            else:
                st.error(f"Could not retrieve data for {stock_symbol}. Please try a valid stock symbol.")

def portfolio_health_checker_feature():
    st.header("❤️ Portfolio Health Checker")
    st.write("Analyze the diversification and risk exposure of your stock portfolio.")

    st.subheader("Enter Your Stock Holdings")
    st.info("You can enter up to 10 stocks. For each stock, provide the symbol and quantity.")

    num_stocks = st.slider("Number of different stocks in your portfolio:", 1, 10, 3)

    holdings_input = []
    for i in range(num_stocks):
        cols = st.columns([0.4, 0.3, 0.3])
        symbol = cols[0].text_input(f"Stock Symbol {i+1} (e.g., TCS.NS):", key=f"symbol_{i}")
        quantity = cols[1].number_input(f"Quantity {i+1}:", min_value=1, value=10, key=f"qty_{i}")
        # purchase_price = cols[2].number_input(f"Purchase Price {i+1} (Optional):", min_value=0.01, value=100.0, key=f"price_{i}")

        if symbol and quantity:
            holdings_input.append({'symbol': symbol.strip().upper(), 'quantity': quantity})

    if st.button("Analyze My Portfolio"):
        if not holdings_input:
            st.warning("Please enter at least one stock holding to analyze.")
            return

        valid_holdings = []
        for holding in holdings_input:
            # Basic validation: Ensure symbol ends with .NS for Indian stocks
            if not holding['symbol'].endswith('.NS'):
                holding['symbol'] += '.NS' # Auto-append .NS if missing
            valid_holdings.append(holding)

        with st.spinner("Analyzing portfolio..."):
            portfolio_value = 0
            sector_allocation = {}
            market_cap_allocation = {'Large Cap': 0, 'Mid Cap': 0, 'Small Cap': 0, 'Unknown': 0}
            stock_contributions = {}
            error_stocks = []

            for holding in valid_holdings:
                stock_symbol = holding['symbol']
                qty = holding['quantity']
                try:
                    ticker = yf.Ticker(stock_symbol)
                    info = ticker.info
                    current_price = info.get('currentPrice')
                    sector = info.get('sector', 'Unknown Sector')
                    market_cap = info.get('marketCap')

                    if current_price and market_cap:
                        stock_value = current_price * qty
                        portfolio_value += stock_value

                        sector_allocation[sector] = sector_allocation.get(sector, 0) + stock_value

                        # Simplified Market Cap Classification (approximate thresholds for Indian market)
                        if market_cap >= 20000 * 10**7: # > 20,000 Cr. INR
                            market_cap_allocation['Large Cap'] += stock_value
                        elif market_cap >= 5000 * 10**7: # > 5,000 Cr. INR
                            market_cap_allocation['Mid Cap'] += stock_value
                        else:
                            market_cap_allocation['Small Cap'] += stock_value

                        stock_contributions[stock_symbol] = stock_value
                    else:
                        error_stocks.append(stock_symbol)
                        st.warning(f"Could not fetch complete data for {stock_symbol}.")
                except Exception as e:
                    error_stocks.append(stock_symbol)
                    st.error(f"Error fetching data for {stock_symbol}: {e}")

            if portfolio_value == 0:
                st.error("No valid stock data found to analyze. Please check your symbols and quantities.")
                return

            st.subheader("Your Portfolio Summary")
            st.metric("Total Portfolio Value", f"₹{portfolio_value:,.2f}")

            if error_stocks:
                st.warning(f"Could not fetch data for: {', '.join(error_stocks)}. They are excluded from analysis.")

            # Sector Allocation
            sector_df = pd.DataFrame(sector_allocation.items(), columns=['Sector', 'Value'])
            sector_df['Percentage'] = (sector_df['Value'] / portfolio_value * 100).round(2)
            st.subheader("Sector Allocation")
            st.dataframe(sector_df.sort_values(by='Percentage', ascending=False), use_container_width=True)
            fig_sector = px.pie(sector_df, values='Value', names='Sector', title='Portfolio Sector Distribution')
            st.plotly_chart(fig_sector)

            # Market Cap Allocation
            market_cap_df = pd.DataFrame(market_cap_allocation.items(), columns=['Category', 'Value'])
            market_cap_df['Percentage'] = (market_cap_df['Value'] / portfolio_value * 100).round(2)
            st.subheader("Market Cap Allocation")
            st.dataframe(market_cap_df.sort_values(by='Percentage', ascending=False), use_container_width=True)
            fig_market_cap = px.pie(market_cap_df, values='Value', names='Category', title='Portfolio Market Cap Distribution')
            st.plotly_chart(fig_market_cap)

            st.subheader("Risk & Diversification Analysis")
            risk_level = "Low"
            warnings_list = []

            # Check for Sector Overexposure
            for index, row in sector_df.iterrows():
                if row['Percentage'] > 25: # Arbitrary threshold for overexposure
                    warnings_list.append(f"⚠️ **Sector Overexposure:** {row['Percentage']:.2f}% in '{row['Sector']}' sector. Consider diversifying.")
                    risk_level = "Medium" if risk_level == "Low" else "High"

            # Check for Single Stock Concentration
            for stock_symbol, value in stock_contributions.items():
                stock_pct = (value / portfolio_value * 100).round(2)
                if stock_pct > 15: # Arbitrary threshold for single stock concentration
                    warnings_list.append(f"⚠️ **High Concentration:** {stock_pct:.2f}% in '{stock_symbol}'.")
                    risk_level = "Medium" if risk_level == "Low" else "High"

            # Check for Market Cap imbalance (e.g., too much Small Cap for a moderate investor)
            small_cap_pct = market_cap_allocation['Small Cap'] / portfolio_value * 100
            if small_cap_pct > 20 and risk_level != "High": # If significant small cap, elevate risk
                 warnings_list.append(f"📈 **High Small Cap Exposure:** {small_cap_pct:.2f}% in Small Cap stocks. This typically increases portfolio volatility.")
                 risk_level = "High"

            st.metric("Estimated Portfolio Risk Level", risk_level)
            if warnings_list:
                for warning in warnings_list:
                    st.markdown(f"<p style='color: orange;'>{warning}</p>", unsafe_allow_html=True)
            else:
                st.success("Your portfolio appears well-diversified!")

# --- Main Application Layout ---

# Create tabs for navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard", "Chatbot", "Earnings Calendar", "Stock Comparison", "AI Reports", "Portfolio Health"
])

with tab1: # Your original Dashboard content
    st.subheader("Market Overview")
    market_indices_df = get_nifty_indices_data()
    cols = st.columns(len(market_indices_df))
    for i, row in market_indices_df.iterrows():
        color = "green" if row['Change'] > 0 else "red" if row['Change'] < 0 else "black"
        with cols[i]:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>{row['Index']}</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: {color};">
                        {row['Price']:.2f}
                    </p>
                    <p style="font-size: 1rem; color: {color};">
                        {row['Change']:.2f} ({row['Percent Change']:.2f}%)
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.subheader("Top Market Movers (Sampled)")
    # Fetch a sample of stocks to determine movers
    sample_symbols = random.sample(NIFTY_50_SYMBOLS, 10) # Get 10 random stocks from Nifty 50
    mover_data = []
    for symbol in sample_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                previous_close = hist['Close'].iloc[-2]
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100
                mover_data.append({"Symbol": symbol, "Current Price": current_price, "Change": change, "Percent Change": percent_change})
        except Exception as e:
            # st.warning(f"Could not fetch mover data for {symbol}: {e}")
            pass # Silently fail for market movers to not interrupt flow

    if mover_data:
        movers_df = pd.DataFrame(mover_data)
        top_gainers = movers_df.sort_values(by="Percent Change", ascending=False).head(5)
        top_losers = movers_df.sort_values(by="Percent Change", ascending=True).head(5)

        col_gainers, col_losers = st.columns(2)
        with col_gainers:
            st.markdown("<h5>📈 Top Gainers</h5>", unsafe_allow_html=True)
            st.dataframe(top_gainers.style.applymap(lambda x: 'color: green' if isinstance(x, (float)) and x > 0 else '', subset=['Percent Change']), use_container_width=True)
        with col_losers:
            st.markdown("<h5>📉 Top Losers</h5>", unsafe_allow_html=True)
            st.dataframe(top_losers.style.applymap(lambda x: 'color: red' if isinstance(x, (float)) and x < 0 else '', subset=['Percent Change']), use_container_width=True)
    else:
        st.info("Could not fetch top market movers at this time.")

    st.subheader("Individual Stock Analysis")
    selected_stock = st.selectbox("Select a stock:", NIFTY_50_SYMBOLS)

    if selected_stock:
        stock_info, stock_hist = get_stock_info(selected_stock)

        if stock_info and not stock_hist.empty:
            current_price = stock_info.get('currentPrice', 'N/A')
            previous_close = stock_info.get('previousClose', 'N/A')
            if current_price != 'N/A' and previous_close != 'N/A':
                price_change = current_price - previous_close
                percent_change = (price_change / previous_close) * 100
                price_color = "green" if price_change > 0 else "red" if price_change < 0 else "black"
                st.markdown(f"**Current Price:** <span style='font-size: 1.5rem; font-weight: bold; color: {price_color};'>₹{current_price:,.2f}</span> <span style='font-size: 1rem; color: {price_color};'>({price_change:+.2f}, {percent_change:+.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.write(f"**Current Price:** N/A")

            # Key Financial Metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = {
                "P/E Ratio": stock_info.get('trailingPE', 'N/A'),
                "Market Cap": f"₹{stock_info.get('marketCap', 0):,.0f}" if isinstance(stock_info.get('marketCap'), (int, float)) else 'N/A',
                "ROE": f"{stock_info.get('returnOnEquity', 0)*100:.2f}%" if isinstance(stock_info.get('returnOnEquity'), (int, float)) else 'N/A',
                "Debt/Equity": f"{stock_info.get('debtToEquity', 0):.2f}" if isinstance(stock_info.get('debtToEquity'), (int, float)) else 'N/A',
                "Dividend Yield": f"{stock_info.get('dividendYield', 0)*100:.2f}%" if isinstance(stock_info.get('dividendYield'), (int, float)) else 'N/A',
                "52 Week High": f"₹{stock_info.get('fiftyTwoWeekHigh', 0):,.2f}" if isinstance(stock_info.get('fiftyTwoWeekHigh'), (int, float)) else 'N/A',
                "52 Week Low": f"₹{stock_info.get('fiftyTwoWeekLow', 0):,.2f}" if isinstance(stock_info.get('fiftyTwoWeekLow'), (int, float)) else 'N/A',
                "Volume": f"{stock_info.get('volume', 0):,.0f}" if isinstance(stock_info.get('volume'), (int, float)) else 'N/A'
            }

            st.markdown("---")
            st.subheader("Key Financials")
            metric_cols = st.columns(4)
            i = 0
            for label, value in metrics.items():
                with metric_cols[i % 4]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h6>{label}</h6>
                            <p style="font-size: 1.2rem; font-weight: bold;">{value}</p>
                        </div>
                    """, unsafe_allow_html=True)
                i += 1

            # Price Chart
            st.subheader("Price Performance (1 Year)")
            fig = go.Figure(data=[go.Candlestick(x=stock_hist.index,
                                                open=stock_hist['Open'],
                                                high=stock_hist['High'],
                                                low=stock_hist['Low'],
                                                close=stock_hist['Close'])])
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- Simulated News & Sentiment ---
            st.subheader("News & Sentiment Analysis (Simulated)")
            news_headlines = generate_random_news(selected_stock)
            news_df = pd.DataFrame(news_headlines, columns=["Headline"])
            news_df["Sentiment"] = news_df["Headline"].apply(get_sentiment)
            st.dataframe(news_df, use_container_width=True)

            # Advanced Financials (QoQ, YoY, Cash Flow, Holdings)
            st.subheader("Advanced Financials")
            try:
                # Dummy data for demonstration as yfinance does not directly provide QoQ/YoY growth
                # or detailed DII/FII data in .info directly.
                st.markdown("##### Revenue & PAT Growth (Simulated QoQ/YoY)")
                revenue_growth = {
                    'Q1 2024': f"{random.uniform(5, 20):.2f}%", 'Q2 2024': f"{random.uniform(5, 20):.2f}%",
                    'Q3 2024': f"{random.uniform(5, 20):.2f}%", 'Q4 2024': f"{random.uniform(5, 20):.2f}%"
                }
                pat_growth = {
                    'Q1 2024': f"{random.uniform(3, 18):.2f}%", 'Q2 2024': f"{random.uniform(3, 18):.2f}%",
                    'Q3 2024': f"{random.uniform(3, 18):.2f}%", 'Q4 2024': f"{random.uniform(3, 18):.2f}%"
                }
                growth_df = pd.DataFrame({'Revenue Growth': revenue_growth, 'PAT Growth': pat_growth})
                st.dataframe(growth_df.T, use_container_width=True)

                st.markdown("##### Cash Flow (Simulated)")
                cash_flow_data = {
                    'Operating Cash Flow': f"₹{random.randint(100, 500) * 10**7:,.0f}",
                    'Free Cash Flow': f"₹{random.randint(50, 300) * 10**7:,.0f}"
                }
                st.json(cash_flow_data)

                st.markdown("##### Shareholder Holdings (Simulated)")
                holdings_data = {
                    'DII Holdings': f"{random.uniform(10, 25):.2f}% (QoQ Change: {random.uniform(-2, 2):+.2f}%)",
                    'FII Holdings': f"{random.uniform(15, 30):.2f}% (QoQ Change: {random.uniform(-2, 2):+.2f}%)",
                    'Retail Holdings': f"{random.uniform(5, 15):.2f}% (QoQ Change: {random.uniform(-2, 2):+.2f}%)"
                }
                st.json(holdings_data)

            except Exception as e:
                st.warning(f"Could not fetch advanced financials: {e}. Displaying simulated data.")

            # --- Custom Stock Picker / Screener (Based on existing functionality) ---
            st.subheader("Custom Stock Screener")
            st.write("Filter stocks based on your preferred criteria.")

            min_pe = st.slider("Minimum P/E Ratio:", 0.0, 100.0, 10.0, 0.1)
            max_pe = st.slider("Maximum P/E Ratio:", 0.0, 1000.0, 50.0, 0.1)
            min_roe = st.slider("Minimum ROE (%):", 0.0, 50.0, 15.0, 0.1)
            max_debt_equity = st.slider("Maximum Debt/Equity:", 0.0, 5.0, 1.0, 0.1)
            min_market_cap_cr = st.slider("Minimum Market Cap (Cr INR):", 0, 100000, 10000, 100) # In Crores
            min_yoy_revenue_growth = st.slider("Min. YoY Revenue Growth (%):", -20.0, 50.0, 10.0, 0.1)
            min_yoy_pat_growth = st.slider("Min. YoY PAT Growth (%):", -20.0, 50.0, 10.0, 0.1)

            if st.button("Find Stocks"):
                st.info("Fetching data for all Nifty 50 stocks to apply filters. This may take a moment.")
                found_stocks = []
                for symbol in NIFTY_50_SYMBOLS:
                    info, _ = get_stock_info(symbol)
                    if info:
                        pe = info.get('trailingPE')
                        roe = info.get('returnOnEquity') # This is decimal (e.g., 0.15 for 15%)
                        debt_equity_ratio = info.get('debtToEquity')
                        market_cap = info.get('marketCap')
                        # Simulate YoY growth as yfinance doesn't provide it directly
                        simulated_revenue_growth = random.uniform(min_yoy_revenue_growth - 5, min_yoy_revenue_growth + 5)
                        simulated_pat_growth = random.uniform(min_yoy_pat_growth - 5, min_yoy_pat_growth + 5)

                        if (pe is not None and min_pe <= pe <= max_pe and
                            roe is not None and roe * 100 >= min_roe and # Convert ROE to percentage
                            debt_equity_ratio is not None and debt_equity_ratio <= max_debt_equity and
                            market_cap is not None and market_cap >= min_market_cap_cr * 10**7 and # Convert Cr to actual value
                            simulated_revenue_growth >= min_yoy_revenue_growth and
                            simulated_pat_growth >= min_yoy_pat_growth):
                            found_stocks.append({
                                'Symbol': symbol,
                                'P/E': f"{pe:.2f}",
                                'ROE (%)': f"{roe*100:.2f}",
                                'Debt/Equity': f"{debt_equity_ratio:.2f}",
                                'Market Cap (Cr)': f"{market_cap / 10**7:,.0f}",
                                'Simulated Rev Growth (%)': f"{simulated_revenue_growth:.2f}",
                                'Simulated PAT Growth (%)': f"{simulated_pat_growth:.2f}"
                            })

                if found_stocks:
                    st.dataframe(pd.DataFrame(found_stocks), use_container_width=True)
                else:
                    st.info("No stocks found matching your criteria. Try adjusting the filters.")
        else:
            st.warning("Could not load data for the selected stock. Please try another one or check the symbol.")

    # --- AI-Powered Portfolio Builder (Integrated from previous version) ---
    st.subheader("AI-Powered Portfolio Builder")
    portfolio_builder_instance = PortfolioBuilder()

    risk_profile = st.selectbox(
        "Select Your Risk Profile:",
        ("Conservative", "Moderate", "Aggressive"),
        help="This helps determine your ideal asset allocation."
    )

    allocation = portfolio_builder_instance.get_risk_based_allocation(risk_profile)
    if allocation:
        st.write("### Recommended Asset Allocation:")
        allocation_df = pd.DataFrame(allocation.items(), columns=["Asset Class", "Percentage"])
        allocation_df["Percentage"] = (allocation_df["Percentage"] * 100).round(1).astype(str) + '%'
        st.dataframe(allocation_df.set_index("Asset Class"), use_container_width=True)

        fig_allocation = px.pie(
            allocation_df,
            values=[float(p.replace('%', '')) for p in allocation_df['Percentage']],
            names='Asset Class',
            title=f"Recommended Allocation for {risk_profile} Investor"
        )
        st.plotly_chart(fig_allocation, use_container_width=True)

        st.markdown("---")
        st.subheader("Investment Projection")

        initial_investment = st.number_input(
            "One-time Initial Investment (₹):",
            min_value=0, value=100000, step=10000
        )
        monthly_sip = st.number_input(
            "Monthly SIP (₹):",
            min_value=0, value=5000, step=500
        )
        duration_years = st.slider(
            "Investment Duration (Years):",
            min_value=1, max_value=30, value=10
        )

        if st.button("Project Investment Growth"):
            if initial_investment > 0 or monthly_sip > 0:
                projected_value = portfolio_builder_instance.project_investment_value(
                    initial_investment, monthly_sip, duration_years, allocation
                )
                st.success(f"**Projected Value after {duration_years} years:** ₹{projected_value:,.2f}")
            else:
                st.warning("Please enter either an initial investment or a monthly SIP.")

        st.markdown("---")
        st.subheader("Example Stock Suggestions based on Allocation")
        st.info("These are general examples and not specific buy/sell recommendations.")
        example_stocks = portfolio_builder_instance.suggest_example_stocks(allocation)
        for category, stocks in example_stocks.items():
            st.markdown(f"**{category} (Example):** {', '.join(stocks)}")

        st.markdown(
            f"""
            _**Disclaimer on Projections:** These projections are based on hypothetical average annual returns
            (e.g., Large Cap: {portfolio_builder_instance.average_annual_returns['Large Cap']*100:.1f}%,
            Mid Cap: {portfolio_builder_instance.average_annual_returns['Mid Cap']*100:.1f}%,
            Small Cap: {portfolio_builder_instance.average_annual_returns['Small Cap']*100:.1f}%,
            Gold: {portfolio_builder_instance.average_annual_returns['Gold']*100:.1f}%,
            Debt: {portfolio_builder_instance.average_annual_returns['Debt']*100:.1f}%
            ) and does not account for market volatility, inflation, taxes, fees, or actual historical performance of specific investments.
            For **SIP**, the calculation uses the future value of an annuity formula based on your monthly investment.
            **Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.**_
            """
        )

with tab2:
    conversational_finance_assistant_feature()

with tab3:
    earnings_calendar_alerts_feature()

with tab4:
    stock_comparison_tool_feature()

with tab5:
    ai_generated_stock_reports_feature()

with tab6:
    portfolio_health_checker_feature()

# Sidebar Footer with Disclaimer and Creator Info
st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: StockSense AI is for educational and informational purposes only. It does not constitute financial advice. Always consult a qualified financial advisor.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <p><b>Made by:</b> ANSHIK MANTRI</p>
        <p><b>Email:</b> <a href="mailto:anshikmantri26@gmail.com">anshikmantri26@gmail.com</a></p>
        <p>
            <a href="http://www.linkedin.com/in/anshikmantri" target="_blank">LinkedIn</a> |
            <a href="http://www.instagram.com/anshik.m6777/" target="_blank">Instagram</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
