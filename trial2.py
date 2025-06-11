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

# --- NEW IMPORTS FOR OPENAI CHATBOT ---
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# --- END NEW IMPORTS ---

warnings.filterwarnings('ignore') # Suppress warnings, typically from yfinance

# Page configuration
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈", # Emoji for page icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TOP-LEVEL SESSION STATE INITIALIZATION ---
# This is crucial for avoiding KeyError and ensuring persistence across reruns.
# All session state variables must be initialized BEFORE they are read.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot_instance" not in st.session_state:
    st.session_state.chatbot_instance = None # Initialize to None first
# --- END TOP-LEVEL SESSION STATE INITIALIZATION ---

# Custom CSS for better styling, especially for icon visibility on white background
st.markdown("""
<style>
    /* General body and text color for better contrast */
    body {
        color: #333333; /* Dark gray text for readability */
        background-color: #F8F8F8; /* Slightly off-white background */
    }
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
        background: linear-gradient(135deg, #e0f2f7, #ffffff);
        border-left: 5px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #555555;
        margin-top: 0.5rem;
    }
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    /* Style for the chatbot in sidebar */
    .sidebar-chatbot {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .sidebar-footer {
        font-size: 0.8rem;
        color: #777777;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eeeeee;
    }
    .sidebar-footer a {
        color: #1f77b4;
        text-decoration: none;
    }
    .sidebar-footer a:hover {
        text-decoration: underline;
    }

    /* Streamlit specific adjustments for better UI */
    .stSelectbox, .stTextInput, .stDateInput, .stButton, .stNumberInput {
        margin-bottom: 1rem;
    }
    /* Ensure icons in expanders and other elements are visible */
    /* Targeting SVG icons inside Streamlit components to make them dark */
    .stExpander svg,
    .stSelectbox svg,
    .stMultiSelect svg,
    .stRadio svg {
        fill: #333333 !important; /* Forces a dark fill color for all these icons */
        color: #333333 !important; /* Fallback for other icon types */
    }
    /* Specifically for the chevron (arrow) in expanders */
    .stExpander [data-testid="stExpanderToggleIcon"] svg {
        fill: #333333 !important;
    }

    /* Text within elements for better contrast */
    .stButton > button p,
    .stSelectbox > div > label p,
    .stMultiSelect > div > label p,
    .stRadio > label p,
    .stTextInput > label p,
    .stNumberInput > label p,
    .stDateInput > label p {
        color: #333333 !important; /* Dark text for labels */
    }

    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #1a6396;
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 10px;
        padding-left: 20px;
        padding-right: 20px;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"] label p {
        color: #555555;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e6e9ee;
        border-bottom: 2px solid #1f77b4; /* Highlight on hover */
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1f77b4;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05); /* Subtle shadow for active tab */
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] label p {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions (Caching for performance) ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker_symbol, start_date, end_date):
    """Fetches historical stock data for a given ticker, start, and end date."""
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No historical data found for {ticker_symbol} in the specified date range.")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker_symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300) # Cache data for 5 minutes
def get_current_stock_info(ticker_symbol):
    """Fetches current stock information (e.g., price, market cap, sector)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        if not info: # Check if info dictionary is empty
            st.warning(f"No current information found for {ticker_symbol}.")
            return {}
        return info
    except Exception as e:
        st.error(f"Error fetching current info for {ticker_symbol}: {e}")
        return {}

def analyze_sentiment(text):
    """Performs basic sentiment analysis on text using TextBlob."""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# --- Chatbot Class (Leveraging OpenAI & LangChain) ---
class Chatbot:
    """A chatbot integrated with OpenAI's GPT models via LangChain for conversational AI."""
    def __init__(self):
        # Access the API key securely using st.secrets.get()
        # This line is where the API key is retrieved. It expects OPENAI_API_KEY
        # to be set in your .streamlit/secrets.toml file or as an environment variable.
        openai_api_key = st.secrets.get("OPENAI_API_KEY")

        if not openai_api_key:
            st.error("OpenAI API key not found. Please set it securely in Streamlit Cloud secrets or .streamlit/secrets.toml.")
            self.conversation = None
            self.is_initialized = False # Flag to indicate if chatbot is ready
        else:
            try:
                # Initialize OpenAI LLM for text generation
                # Using "gpt-3.5-turbo-instruct" for conversational completions.
                # Temperature (0.0-1.0): controls randomness/creativity. 0.7 is a good balance.
                llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo-instruct")
                
                # Initialize ConversationChain with memory to maintain chat context
                self.conversation = ConversationChain(
                    llm=llm,
                    memory=ConversationBufferMemory() # Stores previous messages for context
                )
                self.is_initialized = True # Chatbot successfully initialized
            except Exception as e:
                st.error(f"Error initializing OpenAI LLM: {e}. Please check your API key, model access, or network connection.")
                self.conversation = None
                self.is_initialized = False # Initialization failed

    def get_response(self, user_input):
        """Generates a response from the chatbot."""
        if not self.is_initialized:
            # If chatbot wasn't initialized, return a clear error message.
            return "AI Assistant is not available. Please ensure the OpenAI API key is correctly configured."

        if self.conversation:
            try:
                # Use the conversation chain to predict the next response based on input and memory
                response = self.conversation.run(input=user_input)
                return response
            except Exception as e:
                # Catch and display errors during API call (e.g., rate limits, invalid requests)
                return f"An error occurred while generating a response: {e}. Please try again."
        else:
            # Fallback for unexpected states where conversation object is None
            return "AI Assistant internal error. Please try restarting the app."

# --- Stock Data Management Class ---
class StockDataLists:
    """Manages categorized lists of Indian stocks (Large, Mid, Small Cap) for selection."""
    def __init__(self):
        # Extensive lists of Indian stocks for broader selection
        self.large_cap_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
            'INFY.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
            'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS', 'COALINDIA.NS',
            'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'GRASIM.NS', 'JSWSTEEL.NS',
            'TATASTEEL.NS', 'HINDALCO.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS', 'SHREECEM.NS',
            'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'BPCL.NS', 'IOC.NS', 'INDUSINDBK.NS', 'APOLLOHOSP.NS',
            'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'ADANIENT.NS', 'M&M.NS'
        ]
        
        self.mid_cap_stocks = [
            'DMART.NS', 'PIDILITIND.NS', 'BERGEPAINT.NS', 'GODREJCP.NS', 'MARICO.NS',
            'DABUR.NS', 'COLPAL.NS', 'MCDOWELL-N.NS', 'PGHH.NS', 'HAVELLS.NS',
            'VOLTAS.NS', 'PAGEIND.NS', 'MPHASIS.NS', 'LTIM.NS', 'LTTS.NS',
            'PERSISTENT.NS', 'COFORGE.NS', 'BIOCON.NS', 'LUPIN.NS', 'TORNTPHARM.NS',
            'AUBANK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'MOTHERSON.NS',
            'ASHOKLEY.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'TVSMOTOR.NS',
            'BALKRISIND.NS', 'APOLLOTYRE.NS', 'MRF.NS', 'CUMMINSIND.NS', 'BATAINDIA.NS',
            'RELAXO.NS', 'VBL.NS', 'TATACONSUM.NS', 'JUBLFOOD.NS', 'CROMPTON.NS',
            'WHIRLPOOL.NS', 'SIEMENS.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'DLF.NS'
        ]
        
        self.small_cap_stocks = [
            'AFFLE.NS', 'ROUTE.NS', 'INDIAMART.NS', 'ZOMATO.NS', 'PAYTM.NS',
            'POLICYBZR.NS', 'FSL.NS', 'CARBORUNIV.NS', 'PGHL.NS', 'VINATIORGA.NS',
            'SYMPHONY.NS', 'RAJESHEXPO.NS', 'ASTRAL.NS', 'NILKAMAL.NS', 'CERA.NS',
            'JKCEMENT.NS', 'RAMCOCEM.NS', 'HEIDELBERG.NS', 'PRISMCEM.NS', 'SUPRAJIT.NS',
            'SCHAEFFLER.NS', 'TIMKEN.NS', 'SKFINDIA.NS', 'NRBBEARING.NS', 'FINEORG.NS',
            'AAVAS.NS', 'HOMEFIRST.NS', 'UJJIVANSFB.NS', 'SPANDANA.NS', 'CREDITACC.NS'
        ]

    def get_all_stock_symbols(self):
        """Returns a combined, unique, and sorted list of all defined stock symbols."""
        all_symbols = (
            self.large_cap_stocks +
            self.mid_cap_stocks +
            self.small_cap_stocks
        )
        return sorted(list(set(all_symbols))) # Ensure uniqueness and sort alphabetically

# Initialize StockDataLists instance globally
stock_data_manager = StockDataLists()
all_selectable_symbols = stock_data_manager.get_all_stock_symbols()

# --- Portfolio Builder Class ---
class PortfolioBuilder:
    def __init__(self, stock_data_manager_instance):
        self.stock_data_manager = stock_data_manager_instance
        self.average_annual_returns = {
            'Stocks': 0.10,
            'Gold': 0.07,
            'Debt': 0.05
        }
        self.risk_profiles = {
            "Conservative": {"Large Cap": 0.40, "Mid Cap": 0.10, "Small Cap": 0.00, "Debt": 0.40, "Gold": 0.10},
            "Moderate": {"Large Cap": 0.35, "Mid Cap": 0.20, "Small Cap": 0.10, "Debt": 0.25, "Gold": 0.10},
            "Aggressive": {"Large Cap": 0.25, "Mid Cap": 0.25, "Small Cap": 0.20, "Debt": 0.20, "Gold": 0.10}
        }

    def get_asset_allocation(self, risk_profile):
        return self.risk_profiles.get(risk_profile, {})

    def project_investment(self, principal, duration_years, asset_class):
        rate = self.average_annual_returns.get(asset_class, 0.0)
        future_value = principal * ((1 + rate) ** duration_years)
        return future_value

    def project_sip_investment(self, monthly_investment, duration_years, asset_class):
        rate = self.average_annual_returns.get(asset_class, 0.0)
        monthly_rate = rate / 12
        num_payments = duration_years * 12
        # Annuity due formula (payments at the beginning of period)
        if monthly_rate != 0:
            future_value = monthly_investment * (((1 + monthly_rate)**num_payments - 1) / monthly_rate) * (1 + monthly_rate)
        else: # Handle zero interest rate
            future_value = monthly_investment * num_payments
        return future_value

    def get_stock_suggestions(self, risk_profile):
        allocation = self.get_asset_allocation(risk_profile)
        suggestions = {}
        for asset_class, percentage in allocation.items():
            if percentage > 0:
                if asset_class == "Large Cap":
                    available_stocks = self.stock_data_manager.large_cap_stocks
                elif asset_class == "Mid Cap":
                    available_stocks = self.stock_data_manager.mid_cap_stocks
                elif asset_class == "Small Cap":
                    available_stocks = self.stock_data_manager.small_cap_stocks
                elif asset_class == "Debt":
                    available_stocks = ["GOVTSEC.NS (Debt Fund Placeholder)"] # Placeholder for debt
                elif asset_class == "Gold":
                    available_stocks = ["GOLDBEES.NS (Gold ETF Placeholder)"] # Placeholder for gold
                else:
                    available_stocks = []

                if available_stocks:
                    num_suggestions = min(3, len(available_stocks)) # Suggest up to 3 stocks
                    suggestions[asset_class] = random.sample(available_stocks, num_suggestions)
        return suggestions

# --- Main App Title ---
st.markdown("<h1 class='main-header'>StockSense AI 📈</h1>", unsafe_allow_html=True)
st.write("Your intelligent companion for stock market insights and portfolio management.")

# --- Sidebar Content ---
st.sidebar.markdown("<h2 class='sidebar-header'>Select Stock for Dashboard</h2>", unsafe_allow_html=True)

# Main stock selection for the dashboard tab
selected_ticker_from_list = st.sidebar.selectbox(
    "Choose a stock from our curated list:",
    all_selectable_symbols,
    index=all_selectable_symbols.index("RELIANCE.NS") if "RELIANCE.NS" in all_selectable_symbols else 0 # Default selection
)

custom_ticker_input = st.sidebar.text_input("Or enter any custom ticker (e.g., AAPL, GOOGL, RELIANCE.NS):", value="").upper()

# Determine the final ticker to be used across the app's main dashboard
current_dashboard_ticker = selected_ticker_from_list
if custom_ticker_input:
    # Append .NS for Indian stocks if not already present and it seems like an Indian stock without suffix
    if len(custom_ticker_input) <= 10 and not custom_ticker_input.endswith(".NS") and not custom_ticker_input.endswith(".BO"):
         # Simple heuristic: if it's not a common US ticker and no suffix, try .NS
        try_symbol = custom_ticker_input + ".NS"
        info_check = get_current_stock_info(try_symbol)
        if info_check and info_check.get('longName'):
            current_dashboard_ticker = try_symbol
        else:
            current_dashboard_ticker = custom_ticker_input # Fallback to original if .NS doesn't work
    else:
        current_dashboard_ticker = custom_ticker_input

st.sidebar.markdown("---")

# --- Chatbot in Sidebar (Persistent across tabs) ---
st.sidebar.markdown("<div class='sidebar-chatbot'>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 class='sidebar-header'>🤖 AI Assistant</h2>", unsafe_allow_html=True)
st.sidebar.write("Ask me anything about finance, stocks, or general knowledge!")

# Initialize chatbot instance only once (on first run or if it was explicitly set to None)
if st.session_state.chatbot_instance is None:
    st.session_state.chatbot_instance = Chatbot()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.sidebar.chat_message(message["role"]):
        st.sidebar.markdown(message["content"])

# Accept user input for chatbot
user_query_chatbot = st.sidebar.chat_input("Type your question here...", key="chat_input_sidebar")

if user_query_chatbot:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query_chatbot})
    # Display user message in chat message container
    with st.sidebar.chat_message("user"):
        st.sidebar.markdown(user_query_chatbot)

    # Get response from the actual OpenAI chatbot
    with st.sidebar.chat_message("assistant"):
        # Ensure chatbot is initialized and ready before calling its methods
        if st.session_state.chatbot_instance and st.session_state.chatbot_instance.is_initialized:
            with st.sidebar.spinner("Thinking..."):
                response_chatbot = st.session_state.chatbot_instance.get_response(user_query_chatbot)
                st.sidebar.markdown(response_chatbot)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_chatbot})
        else:
            # If chatbot is not initialized, display an error directly
            error_msg = "AI Assistant not available. Please ensure OpenAI API key is configured in Streamlit Cloud secrets."
            st.sidebar.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

st.sidebar.markdown("</div>", unsafe_allow_html=True) # Close sidebar-chatbot div

# --- Sidebar Footer ---
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

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "💰 Portfolio Builder", "💡 AI Report"])

with tab1: # Stock Analysis Tab
    st.markdown(f"<h2>📈 Stock Analysis for {current_dashboard_ticker}</h2>", unsafe_allow_html=True)

    current_info = get_current_stock_info(current_dashboard_ticker)
    if current_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Current Price</div><div class='metric-value'>${current_info.get('regularMarketPrice', 'N/A'):.2f}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Day High</div><div class='metric-value'>${current_info.get('dayHigh', 'N/A'):.2f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Day Low</div><div class='metric-value'>${current_info.get('dayLow', 'N/A'):.2f}</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Volume</div><div class='metric-value'>{current_info.get('regularMarketVolume', 'N/A'):,}</div></div>", unsafe_allow_html=True)

        st.subheader("Company Profile")
        st.write(f"**Name:** {current_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {current_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {current_info.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** ${current_info.get('marketCap', 'N/A'):,} USD")
        st.write(f"**Description:** {current_info.get('longBusinessSummary', 'N/A')}")
    else:
        st.warning(f"Could not retrieve current information for {current_dashboard_ticker}. Please ensure it's a valid ticker symbol and try again.")

    st.subheader("Historical Data & Charts")
    data_source = st.radio("Select Data Range", ["Past Year", "Past 5 Years", "Max Available", "Custom Date Range"], horizontal=True)

    end_date = datetime.now().date() # Use .date() for st.date_input compatibility
    start_date = end_date - timedelta(days=365) # Default to 1 year ago

    if data_source == "Past Year":
        start_date = end_date - timedelta(days=365)
    elif data_source == "Past 5 Years":
        start_date = end_date - timedelta(days=365 * 5)
    elif data_source == "Max Available":
        # yfinance will fetch max history if start_date is very old or None
        start_date = datetime(1900, 1, 1).date() # Effectively max
    else: # Custom Date Range
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start date", value=end_date - timedelta(days=365*2), key='start_date_input')
        with col_date2:
            end_date = st.date_input("End date", value=end_date, key='end_date_input')
        
        if start_date > end_date:
            st.error("Error: End date cannot be before start date.")
            # Set a default valid range to prevent further errors if user input is invalid
            start_date = end_date - timedelta(days=365)

    stock_data = get_stock_data(current_dashboard_ticker, start_date, end_date)

    if not stock_data.empty:
        # Create a simple analyzer for chart generation within this scope
        class ChartGenerator:
            def __init__(self, ticker_sym):
                self.ticker = ticker_sym
            
            def get_price_chart(self, data_df):
                fig = go.Figure(data=[go.Candlestick(x=data_df.index,
                                                     open=data_df['Open'],
                                                     high=data_df['High'],
                                                     low=data_df['Low'],
                                                     close=data_df['Close'])])
                fig.update_layout(xaxis_rangeslider_visible=False, title=f'{self.ticker} Price Chart')
                return fig

            def get_volume_chart(self, data_df):
                fig = px.bar(data_df, x=data_df.index, y='Volume', title=f'{self.ticker} Volume')
                return fig

            def get_moving_averages(self, data_df, window1=20, window2=50):
                data_df[f'MA{window1}'] = data_df['Close'].rolling(window=window1).mean()
                data_df[f'MA{window2}'] = data_df['Close'].rolling(window=window2).mean()
                fig = px.line(data_df, x=data_df.index, y=['Close', f'MA{window1}', f'MA{window2}'],
                              title=f'{self.ticker} Close Price with Moving Averages')
                return fig

        chart_analyzer = ChartGenerator(current_dashboard_ticker)

        st.subheader("Price Chart (Candlestick)")
        st.plotly_chart(chart_analyzer.get_price_chart(stock_data), use_container_width=True)

        st.subheader("Volume Chart")
        st.plotly_chart(chart_analyzer.get_volume_chart(stock_data), use_container_width=True)

        st.subheader("Moving Averages (MA20 & MA50)")
        st.plotly_chart(chart_analyzer.get_moving_averages(stock_data), use_container_width=True)

        st.subheader("News Sentiment Analysis (Simulated Data for Demo)")
        # Simulate news as fetching real-time news with yfinance is limited/inconsistent
        simulated_news_headlines = [
            f"{current_dashboard_ticker} announces strong quarterly earnings.",
            f"Market expresses caution over {current_dashboard_ticker}'s future outlook.",
            f"{current_dashboard_ticker} stock gains after new product launch.",
            f"Regulatory concerns weigh on {current_dashboard_ticker} shares.",
            f"{current_dashboard_ticker} signs major new partnership deal.",
            f"Supply chain issues impact {current_dashboard_ticker}'s production.",
            f"Analysts upgrade rating for {current_dashboard_ticker}.",
            f"{current_dashboard_ticker} faces increased competition."
        ]
        
        # Pick 3-5 random headlines
        num_headlines = random.randint(3, 5)
        selected_news_for_sentiment = random.sample(simulated_news_headlines, k=num_headlines)

        news_data = []
        for headline in selected_news_for_sentiment:
            sentiment_label = analyze_sentiment(headline)
            news_data.append({"Headline": headline, "Sentiment": sentiment_label})
        
        if news_data:
            news_df = pd.DataFrame(news_data)
            st.dataframe(news_df, use_container_width=True)

            sentiment_counts = news_df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_sentiment = px.pie(sentiment_counts, values='Count', names='Sentiment',
                                    title=f'{current_dashboard_ticker} News Sentiment Distribution',
                                    color='Sentiment',
                                    color_discrete_map={'Positive':'#2ca02c', 'Negative':'#d62728', 'Neutral':'#ff7f0e'})
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("No simulated news available for sentiment analysis.")

    else:
        st.info("Please select a valid stock and date range to view historical data and charts.")

with tab2: # Portfolio Builder Tab
    st.markdown("<h2>💰 AI Portfolio Builder</h2>", unsafe_allow_html=True)
    portfolio_builder_instance = PortfolioBuilder(stock_data_manager) # Pass stock_data_manager

    st.subheader("Investment Projection Calculator")
    investment_type = st.radio("Select Investment Type", ["Lump Sum", "Systematic Investment Plan (SIP)"], horizontal=True)

    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        principal_investment = st.number_input("Initial Investment Amount ($)", min_value=100.0, value=10000.0, step=100.0)
    with col_inv2:
        duration_years = st.slider("Investment Duration (Years)", min_value=1, max_value=40, value=10)
    
    monthly_sip = 0.0
    if investment_type == "SIP":
        monthly_sip = st.number_input("Monthly SIP Amount ($)", min_value=10.0, value=500.0, step=10.0)
    
    asset_class = st.selectbox("Select Asset Class", ["Stocks", "Gold", "Debt"])

    if st.button("Project Future Value"):
        st.subheader("Projected Investment Value")
        if investment_type == "Lump Sum":
            projected_value = portfolio_builder_instance.project_investment(
                principal_investment, duration_years, asset_class
            )
            st.success(f"Your lump sum investment of **${principal_investment:,.2f}** in **{asset_class}** over **{duration_years} years** could grow to approximately **${projected_value:,.2f}**.")
        else: # SIP
            projected_value_sip = portfolio_builder_instance.project_sip_investment(
                monthly_sip, duration_years, asset_class
            )
            total_invested = monthly_sip * duration_years * 12
            st.success(f"Your SIP of **${monthly_sip:,.2f} per month** in **{asset_class}** over **{duration_years} years** (total invested: **${total_invested:,.2f}**) could grow to approximately **${projected_value_sip:,.2f}**.")

        # --- Projection Assumptions Display ---
        st.markdown(f"""
            <details>
            <summary>Assumptions for Projection</summary>
            _This projection uses **simulated average annual returns** for {asset_class}
            (Stocks: {portfolio_builder_instance.average_annual_returns['Stocks']*100:.1f}%,
            Gold: {portfolio_builder_instance.average_annual_returns['Gold']*100:.1f}%,
            Debt: {portfolio_builder_instance.average_annual_returns['Debt']*100:.1f}%)._
            _It's a simplified calculation and does not account for market volatility, inflation, taxes, fees, or actual historical performance of specific investments.
            For **SIP**, the calculation uses the future value of an annuity formula based on your monthly investment.
            **Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.**_
            </details>
        """)
        # --- END Projection Assumptions Display ---

    st.subheader("Asset Allocation & Stock Suggestions")
    risk_profile = st.selectbox(
        "Select your risk profile for asset allocation:",
        ["Conservative", "Moderate", "Aggressive"],
        key="risk_profile_select"
    )
    allocation = portfolio_builder_instance.get_asset_allocation(risk_profile)
    if allocation:
        st.write(f"Based on a **{risk_profile}** risk profile, here's a suggested asset allocation:")
        allocation_df = pd.DataFrame(allocation.items(), columns=["Asset Class", "Allocation (%)"])
        allocation_df['Allocation (%)'] = allocation_df['Allocation (%)'] * 100
        st.dataframe(allocation_df.set_index("Asset Class"), use_container_width=True)

        fig_pie = px.pie(allocation_df, values='Allocation (%)', names='Asset Class',
                         title='Suggested Portfolio Allocation',
                         hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.write("Here are some **example stock suggestions** based on your selected risk profile and asset allocation. These are for illustrative purposes only and require thorough research before investing.")
        suggestions = portfolio_builder_instance.get_stock_suggestions(risk_profile)
        for asset, stock_list in suggestions.items():
            if stock_list:
                st.markdown(f"**{asset} Stocks:** {', '.join(stock_list)}")
            else:
                st.markdown(f"**{asset} Stocks:** No specific suggestions available for this category in our demo list.")
    else:
        st.warning("Could not retrieve asset allocation for the selected risk profile.")

with tab3: # AI Report Tab
    st.markdown("<h2>💡 AI Report Generator</h2>", unsafe_allow_html=True)
    st.write("Generate an AI-powered report on a specific stock or a broader financial topic.")

    report_type = st.radio("Select Report Type:", ["Stock Report", "General Financial Topic"], horizontal=True)

    report_query = ""
    if report_type == "Stock Report":
        report_stock_symbol_select = st.selectbox(
            "Select stock for AI Report:",
            all_selectable_symbols,
            index=all_selectable_symbols.index("TCS.NS") if "TCS.NS" in all_selectable_symbols else 0,
            key="ai_report_stock_select"
        )
        report_query = f"Provide a detailed financial analysis and future outlook for {report_stock_symbol_select}. Include key metrics, recent news, and potential risks/opportunities. Structure it clearly."
    else: # General Financial Topic
        report_query = st.text_input("Enter general financial topic for AI Report (e.g., 'Impact of inflation on tech stocks', 'Beginner's guide to mutual funds'):",
                                     placeholder="e.g., 'Future of renewable energy investments'")

    if st.button("Generate AI Report"):
        if report_query:
            if st.session_state.chatbot_instance and st.session_state.chatbot_instance.is_initialized:
                st.subheader("Generated AI Report")
                with st.spinner("Generating AI Report... This may take a moment."):
                    # Leverage the existing chatbot instance to generate the report
                    ai_report_content = st.session_state.chatbot_instance.get_response(report_query)
                    st.markdown(ai_report_content)
                st.success("AI Report Generated!")
                st.markdown("---")
                st.info("Disclaimer: This report is generated by an AI and is based on its training data and current capabilities. It may not reflect real-time market conditions or provide exhaustive analysis. Always verify information and consult a qualified financial advisor before making investment decisions.")
            else:
                st.error("AI Assistant is not available to generate reports. Please ensure your OpenAI API key is correctly configured in Streamlit Cloud secrets.")
        else:
            st.warning("Please enter a stock or topic to generate an AI Report.")
