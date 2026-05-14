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

# --- NEW IMPORTS FOR GEMINI CHATBOT ---
from langchain_google_genai import ChatGoogleGenerativeAI # Updated import for Gemini LLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# --- END NEW IMPORTS ---

warnings.filterwarnings('ignore')

kolkata_tz = pytz.timezone('Asia/Kolkata')

# Page configuration
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (UPDATED FOR DARK THEME VISIBILITY)
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
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-card h3 {
        color: #1f77b4;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    /* --- UPDATED TAB STYLES FOR DARK THEME --- */
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #333;
        color: white;
        border-radius: 8px 8px 0 0;
        margin: 0 5px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #444;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: black !important; /* Black for selected tab */
        color: white !important;
        border: 1px solid #ff7f0e;
    }
    /* --- UPDATED BUTTON STYLES FOR DARK THEME --- */
    .stButton>button {
        background-color: black; /* Black background for buttons */
        color: white;
        border: 1px solid #ff7f0e; /* Orange border for visibility */
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1rem;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff7f0e; /* Orange on hover */
        color: black; /* Black text on hover for contrast */
        border: 1px solid black;
    }
    .css-1d391kg {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .sidebar-footer {
        text-align: center;
        font-size: 0.9rem;
        color: #555;
    }
    .sidebar-footer a {
        color: #1f77b4;
        text-decoration: none;
    }
    .sidebar-footer a:hover {
        text-decoration: underline;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stException {
        background-color: #ffe0e0;
        border-left: 5px solid #ff4d4d;
        padding: 10px;
        border-radius: 5px;
    }
    /* Specific styling for the chatbot in sidebar to make it look distinct */
    .st-emotion-cache-1pxy10k.e1arcw5v0 { /* Targeting the sidebar wrapper */
        display: flex;
        flex-direction: column;
    }
    #root > div:nth-child(1) > div.with-sidebar > div > div > div > section.main.st-emotion-cache-nahz7x.e1arcw5v1 > div.block-container.st-emotion-cache-1y4y3u2.ea3g5fm5 > div:nth-child(1) > div > div.st-emotion-cache-c3wz9u.e1arcw5v4 > div > div > div.st-emotion-cache-pkc1r8.e1arcw5v2 {
        flex-grow: 1; /* Allows main content to fill space */
    }
    .st-emotion-cache-1d9w3d1 { /* Targeting sidebar content container for bottom alignment logic */
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .sidebar-top-content {
        flex-grow: 1; /* Pushes the chatbot and footer to the bottom */
    }
    .sidebar-chatbot {
        margin-top: auto; /* Pushes it to the bottom of the flexible container */
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<h1 class='main-header'>StockSense AI 📈</h1>", unsafe_allow_html=True)
st.write("Your intelligent companion for stock market insights and portfolio management.")

# --- Data Fetching Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(symbol, period="1y"):
    """Fetches historical stock data for a given symbol and period."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {e}")
        return None

@st.cache_data(ttl=300) # Cache data for 5 minutes
def get_current_stock_info(symbol):
    """Fetches current stock information (price, metrics) for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # Basic check for essential info
        if not info or 'regularMarketPrice' not in info:
            return None
        return info
    except Exception as e:
        st.error(f"Error fetching current info for {symbol}: {e}")
        return None

# Function to fetch index data (updated for robustness)
@st.cache_data(ttl=300) # Cache data for 5 minutes
def fetch_index_data(symbol):
    """Fetches current price, daily change, and percent change for market indices."""
    try:
        ticker = yf.Ticker(symbol)
        # Try fetching 1-minute interval data for the most recent price.
        data = ticker.history(period="1d", interval="1m", prepost=False)

        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            current_open = data['Open'].iloc[0] # Open of the current day
            change = latest_price - current_open
            percent_change = (change / current_open) * 100 if current_open else 0
            return latest_price, change, percent_change
        else:
            # If 1-minute data is empty (e.g., outside market hours), try 2-day daily data
            # to get previous close for a robust daily change calculation.
            data_daily = ticker.history(period="2d")
            if not data_daily.empty and len(data_daily) >= 2:
                latest_price = data_daily['Close'].iloc[-1]
                prev_close = data_daily['Close'].iloc[-2]
                change = latest_price - prev_close
                percent_change = (change / prev_close) * 100 if prev_close else 0
                return latest_price, change, percent_change
            else:
                st.warning(f"No sufficient data fetched for {symbol} even after trying 2-day daily period.")
                return None, None, None
    except Exception as e:
        st.error(f"Could not fetch data for {symbol}: {e}")
        return None, None, None

# Function to generate simulated news and sentiment
def get_simulated_news_and_sentiment(stock_symbol):
    """Generates simulated news headlines and performs basic sentiment analysis."""
    sample_headlines = [
        f"{stock_symbol} shares rally on strong Q1 earnings.",
        f"Analysts cautious about {stock_symbol} amid market slowdown.",
        f"{stock_symbol} announces new product line, stock remains stable.",
        f"Supply chain disruptions hit {stock_symbol}'s production targets.",
        f"Positive outlook for {stock_symbol} as economy recovers.",
        f"Regulators investigate {stock_symbol} for data privacy concerns.",
        f"{stock_symbol} partners with tech giant for innovation.",
        f"Dividend increase expected from {stock_symbol} this quarter.",
        f"{stock_symbol} faces stiff competition in key market segment.",
        f"CEO of {stock_symbol} addresses climate change initiatives."
    ]
    selected_headlines = random.sample(sample_headlines, k=random.randint(2, 4))
    news_data = []
    for headline in selected_headlines:
        analysis = TextBlob(headline)
        sentiment = analysis.sentiment.polarity
        if sentiment > 0.1:
            sentiment_label = "Positive"
        elif sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        news_data.append({"Headline": headline, "Sentiment": sentiment_label})
    return pd.DataFrame(news_data)

# Simulated Financial Data - For demo purposes
def get_simulated_advanced_financials(symbol):
    """Provides simulated advanced financial metrics like growth, cash flow, and holdings."""
    data = {
        'Metric': ['QoQ Revenue Growth', 'YoY Revenue Growth', 'QoQ PAT Growth', 'YoY PAT Growth',
                   'Operating Cash Flow (Cr)', 'Free Cash Flow (Cr)',
                   'DII Holding (%)', 'FII Holding (%)', 'Retail Holding (%)'],
        'Value': [f"{random.uniform(-5, 25):.2f}%", f"{random.uniform(0, 30):.2f}%",
                  f"{random.uniform(-10, 30):.2f}%", f"{random.uniform(5, 40):.2f}%",
                  f"{random.randint(100, 5000):,.0f}", f"{random.randint(50, 4000):,.0f}",
                  f"{random.uniform(5, 15):.2f}", f"{random.uniform(10, 30):.2f}", f"{random.uniform(20, 50):.2f}"],
        'QoQ Change': [f"{random.uniform(-2, 2):.2f}%" for _ in range(4)] + ['N/A','N/A'] + [f"{random.uniform(-0.5, 0.5):.2f}" for _ in range(3)]
    }
    df = pd.DataFrame(data)
    return df

# Placeholder for stock screener data - for demo purposes, using a small sample
@st.cache_data(ttl=3600)
def get_nifty50_sample_data_for_screener():
    """Generates simulated financial data for a sample of Nifty 50 stocks for the screener.
    Uses stock_analyzer.large_cap_stocks for a wider sample.
    """
    # Use a subset of large_cap_stocks to avoid too many API calls for the screener's initial load
    sample_symbols = random.sample(stock_analyzer.large_cap_stocks, min(20, len(stock_analyzer.large_cap_stocks)))

    data_list = []
    for symbol in sample_symbols:
        info = get_current_stock_info(symbol)
        if info:
            try:
                # Get company name from info, or use symbol if not available
                company_name = info.get('shortName', info.get('longName', symbol.replace(".NS", "")))

                pe_ratio = info.get('trailingPE', np.nan)
                roe = random.uniform(10, 30) # Simulated ROE
                debt_equity = random.uniform(0.1, 1.5) # Simulated Debt/Equity
                market_cap = info.get('marketCap', np.nan)
                yoy_revenue_growth = random.uniform(5, 25) # Simulated YoY Revenue Growth
                yoy_pat_growth = random.uniform(10, 35) # Simulated YoY PAT Growth
                data_list.append({
                    'Company': company_name,
                    'Symbol': symbol,
                    'P/E Ratio': pe_ratio,
                    'ROE (%)': roe,
                    'Debt/Equity': debt_equity,
                    'Market Cap (Cr)': market_cap / 10**7 if market_cap else np.nan, # Convert to Crores
                    'YoY Revenue Growth (%)': yoy_revenue_growth,
                    'YoY PAT Growth (%)': yoy_pat_growth
                })
            except Exception as e:
                st.warning(f"Could not process data for {symbol}: {e}")
    df = pd.DataFrame(data_list)
    df['P/E Ratio'].fillna(0, inplace=True) # Replace NaN with 0 for P/E
    df['Market Cap (Cr)'].fillna(0, inplace=True) # Replace NaN with 0 for Market Cap
    return df

# --- Classes ---

class StockAnalyzer:
    """Manages categorized lists of Indian stocks (Large, Mid, Small Cap) - 500 stocks total."""

    def __init__(self):
        # Large Cap Stocks - 180 stocks
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
            'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'ADANIENT.NS', 'M&M.NS',
            'TATACONSUM.NS', 'GODREJCP.NS', 'DABUR.NS', 'MARICO.NS', 'COLPAL.NS',
            'PIDILITIND.NS', 'BERGEPAINT.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'LTIM.NS',
            'MINDTREE.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'LTTS.NS',
            'BIOCON.NS', 'LUPIN.NS', 'TORNTPHARM.NS', 'GLAND.NS', 'LALPATHLAB.NS',
            'DRREDDYS.NS', 'AUBANK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS',
            'RBLBANK.NS', 'YESBANK.NS', 'PNB.NS', 'CANBK.NS', 'UNIONBANK.NS',
            'BANKBARODA.NS', 'INDIANB.NS', 'CENTRALBK.NS', 'IOB.NS', 'PFC.NS',
            'RECLTD.NS', 'IRFC.NS', 'SAIL.NS', 'NMDC.NS', 'VEDL.NS',
            'JINDALSTEL.NS', 'JSPL.NS', 'WELCORP.NS', 'NATIONALUM.NS', 'RATNAMANI.NS',
            'APOLLOTYRE.NS', 'MRF.NS', 'BALKRISIND.NS', 'CEAT.NS', 'JK.NS',
            'MOTHERSON.NS', 'BOSCHLTD.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'TVSMOTOR.NS',
            'BAJAJHLDNG.NS', 'ESCORTS.NS', 'FORCEMOT.NS', 'ASHOKLEY.NS', 'MAHINDCIE.NS',
            'CUMMINSIND.NS', 'BHARATFORG.NS', 'RAMCOCEM.NS', 'JKCEMENT.NS', 'HEIDELBERG.NS',
            'AMBUJCEM.NS', 'ACC.NS', 'INDIACEM.NS', 'DALMIA.NS', 'JKLAKSHMI.NS',
            'ORIENTCEM.NS', 'PRISMCEM.NS', 'CENTURYPLY.NS', 'GREENPLY.NS', 'WABCOINDIA.NS',
            'BATAINDIA.NS', 'RELAXO.NS', 'LIBERTY.NS', 'MIRCHUTE.NS', 'VBL.NS',
            'JUBLFOOD.NS', 'WESTLIFE.NS', 'DEVYANI.NS', 'SPECIALITY.NS', 'ZOMATO.NS',
            'NAUKRI.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'AFFLE.NS', 'ROUTE.NS',
            'INDIAMART.NS', 'JUSTDIAL.NS', 'REDINGTON.NS', 'RATEGAIN.NS', 'TATAELXSI.NS',
            'CYIENT.NS', 'KPITTECH.NS', 'ZENSAR.NS', 'SONATSOFTW.NS', 'NIITTECH.NS',
            'L&TFH.NS', 'CHOLAFIN.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS', 'MMTC.NS',
            'MOIL.NS', 'GMRINFRA.NS', 'GVK.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS',
            'TORNTPOWER.NS', 'CESC.NS', 'JINDALSAW.NS', 'WELSPUNIND.NS', 'TRIDENT.NS',
            'VARDHMAN.NS', 'ALOKTEXT.NS', 'PAGEIND.NS', 'HAVELLS.NS', 'VOLTAS.NS',
            'BLUESTARCO.NS', 'WHIRLPOOL.NS', 'CROMPTON.NS', 'VGUARD.NS', 'ORIENTELEC.NS',
            'KEI.NS', 'POLYCAB.NS', 'FINOLEX.NS', 'SIEMENS.NS', 'ABB.NS',
            'SCHNEIDER.NS', 'HONAUT.NS', 'THERMAX.NS', 'BHEL.NS', 'BEML.NS',
            'BEL.NS', 'HAL.NS', 'COCHINSHIP.NS', 'GRINDWELL.NS', 'CRISIL.NS',
            'CREDITACC.NS', 'EQUITAS.NS', 'CDSL.NS', 'NSDL.NS', 'BSE.NS',
            'MCX.NS', 'MSEI.NS', 'NAZARA.NS', 'DELTACORP.NS', 'ONMOBILE.NS',
            'NETWORK18.NS', 'TV18BRDCST.NS', 'DISHTV.NS', 'SUNTV.NS', 'BALRAMCHIN.NS',
            'DHANUKA.NS', 'RALLIS.NS', 'GHCL.NS', 'AAVAS.NS', 'HOMEFIRST.NS',
            'UJJIVANSFB.NS', 'SPANDANA.NS', 'AROHAN.NS', 'DMART.NS', 'MCDOWELL-N.NS',
            'PGHH.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'DLF.NS', 'PRESTIGE.NS',
            'BRIGADE.NS', 'SOBHA.NS', 'PHOENIXLTD.NS', 'PVRINOX.NS', 'CONCOR.NS',
            'FORTIS.NS', 'MAXHEALTH.NS', 'NHPC.NS', 'SJVN.NS', 'RAILTEL.NS',
            'RITES.NS', 'IRCON.NS', 'NBCC.NS', 'HUDCO.NS', 'NIACL.NS',
            'GICRE.NS', 'ORIENTREF.NS', 'HINDPETRO.NS', 'MRPL.NS', 'GAIL.NS',
            'PETRONET.NS', 'IGL.NS', 'MGL.NS', 'GSPL.NS', 'AEGISCHEM.NS',
            'DEEPAKNI.NS', 'ALKYLAMINE.NS', 'CLEAN.NS', 'NOCIL.NS', 'VINDHYATEL.NS'
        ]

        # Mid Cap Stocks - 160 stocks
        self.mid_cap_stocks = [
            'JSWENERGY.NS', 'RENUKA.NS', 'DHAMPUR.NS', 'BAJAJCON.NS', 'EMAMILTD.NS',
            'GODREJIND.NS', 'JYOTHYLAB.NS', 'CHOLAHLDNG.NS', 'TIMKEN.NS', 'SKFINDIA.NS',
            'SCHAEFFLER.NS', 'NRB.NS', 'FINEORG.NS', 'SUPRAJIT.NS', 'ENDURANCE.NS',
            'SUNDRMFAST.NS', 'MINDAIND.NS', 'SWARAJENG.NS', 'KIOCL.NS', 'SHRIRAMFIN.NS',
            'SRTRANSFIN.NS', 'CAPLIPOINT.NS', 'ESAFSFB.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'CAMS.NS', 'CARERATING.NS', 'ICRA.NS', 'BRICKWORK.NS', 'SMIFS.NS',
            'MOTILALOF.NS', 'ANGELONE.NS', 'IIFL.NS', 'GEOJITFSL.NS', 'VENKEYS.NS',
            'SUGANDHA.NS', 'KRBL.NS', 'KOHINOOR.NS', 'LAXMIMACH.NS', 'TEXRAIL.NS',
            'KNRCON.NS', 'IRB.NS', 'SADBHAV.NS', 'GPPL.NS', 'ZFSTEERING.NS',
            'KANSAINER.NS', 'AIAENG.NS', 'KIRLOSENG.NS', 'CARYSIL.NS', 'HINDWAREAP.NS',
            'DIXON.NS', 'AMBER.NS', 'MINDA.NS', 'SUNDARAM.NS', 'GLENMARK.NS',
            'CADILAHC.NS', 'ALKEM.NS', 'AJANTPHARM.NS', 'ABBOTINDIA.NS', 'PFIZER.NS',
            'GSK.NS', 'NOVARTIS.NS', 'SANOFI.NS', 'MERCK.NS', 'JBCHEPHARM.NS',
            'STRIDES.NS', 'CAPLIN.NS', 'LAURUSLABS.NS', 'SUVEN.NS', 'PIRAMAL.NS',
            'WOCKPHARMA.NS', 'ZYDUSWEL.NS', 'AUROPHARMA.NS', 'ZYDUSLIFE.NS', 'DIVIS.NS',
            'SEQUENT.NS', 'GRANULES.NS', 'METROPOLIS.NS', 'THYROCARE.NS', 'SRL.NS',
            'KIMS.NS', 'RAINBOW.NS', 'GLOBALHLT.NS', 'TEJASNET.NS', 'RCOM.NS',
            'IDEA.NS', 'GTLINFRA.NS', 'HINDCOPPER.NS', 'MAZAGON.NS', 'WESTLIFE.NS',
            'DEVYANI.NS', 'SPECIALITY.NS', 'HAPPSTMNDS.NS', 'INTELLECT.NS', 'RAMKY.NS',
            'VAIBHAVGBL.NS', 'NYKAA.NS', 'CARTRADE.NS', 'EASEMYTRIP.NS', 'RVNL.NS',
            'RAILVIKAS.NS', 'IREDA.NS', 'POWERINDIA.NS', 'TORPOWER.NS', 'RELINFRA.NS',
            'ADANIGAS.NS', 'MAHINDRACO.NS', 'LINDEINDIA.NS', 'PRAXAIR.NS', 'INOXAIR.NS',
            'BASF.NS', 'AKZOINDIA.NS', 'KANSAI.NS', 'BERGER.NS', 'SHALBY.NS',
            'ASTER.NS', 'NARAYANANHL.NS', 'CIGNITI.NS', 'INDIGO.NS', 'SPICEJET.NS',
            'VIP.NS', 'SAFARI.NS', 'SKUMAR.NS', 'CCL.NS', 'RADICO.NS',
            'GLOBUSSPR.NS', 'RAYMOND.NS', 'SIYARAM.NS', 'WELSPUN.NS', 'DONEAR.NS',
            'UJJIVAN.NS', 'SYMPHONY.NS', 'RAJESHEXPO.NS', 'ASTRAL.NS', 'NILKAMAL.NS',
            'CERA.NS', 'VINATIORGA.NS', 'FSL.NS', 'CARBORUNIV.NS', 'PGHL.NS',
            'NRBBEARING.NS', 'RPOWER.NS', 'BATA.NS', 'RELAXO.NS', 'LIBERTY.NS',
            'MIRCHUTE.NS', 'BLUESTARCO.NS', 'ORIENTELEC.NS', 'KEI.NS', 'POLYCAB.NS',
            'FINOLEX.NS', 'ABB.NS', 'SCHNEIDER.NS', 'HONAUT.NS', 'BHEL.NS',
            'BEML.NS', 'BEL.NS', 'HAL.NS', 'COCHINSHIP.NS', 'MAZAGON.NS'
        ]

        # Small Cap Stocks - 160 stocks
        self.small_cap_stocks = [
            'CRISIL.NS', 'EQUITAS.NS', 'CDSL.NS', 'NSDL.NS', 'BSE.NS',
            'MCX.NS', 'MSEI.NS', 'NAZARA.NS', 'ONMOBILE.NS', 'NETWORK18.NS',
            'TV18BRDCST.NS', 'DISHTV.NS', 'SUNTV.NS', 'DHANUKA.NS', 'RALLIS.NS',
            'GHCL.NS', 'AROHAN.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS', 'CAPLIPOINT.NS',
            'SRTRANSFIN.NS', 'SHRIRAMFIN.NS', 'ESAFSFB.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'CAMS.NS', 'CARERATING.NS', 'ICRA.NS', 'BRICKWORK.NS', 'SMIFS.NS',
            'MOTILALOF.NS', 'ANGELONE.NS', 'IIFL.NS', 'GEOJITFSL.NS', 'VENKEYS.NS',
            'SUGANDHA.NS', 'KRBL.NS', 'KOHINOOR.NS', 'AMBER.NS', 'MINDA.NS',
            'SUNDARAM.NS', 'GLENMARK.NS', 'CADILAHC.NS', 'ALKEM.NS', 'AJANTPHARM.NS',
            'ABBOTINDIA.NS', 'PFIZER.NS', 'GSK.NS', 'NOVARTIS.NS', 'SANOFI.NS',
            'MERCK.NS', 'JBCHEPHARM.NS', 'STRIDES.NS', 'CAPLIN.NS', 'LAURUSLABS.NS',
            'SUVEN.NS', 'PIRAMAL.NS', 'WOCKPHARMA.NS', 'ZYDUSWEL.NS', 'AUROPHARMA.NS',
            'ZYDUSLIFE.NS', 'DIVIS.NS', 'SEQUENT.NS', 'GRANULES.NS', 'METROPOLIS.NS',
            'THYROCARE.NS', 'SRL.NS', 'KIMS.NS', 'RAINBOW.NS', 'GLOBALHLT.NS',
            'TEJASNET.NS', 'RCOM.NS', 'IDEA.NS', 'GTLINFRA.NS', 'RAILTEL.NS',
            'RITES.NS', 'IRCON.NS', 'NBCC.NS', 'HUDCO.NS', 'NIACL.NS',
            'GICRE.NS', 'ORIENTREF.NS', 'HINDPETRO.NS', 'MRPL.NS', 'GAIL.NS',
            'PETRONET.NS', 'IGL.NS', 'MGL.NS', 'GSPL.NS', 'AEGISCHEM.NS',
            'DEEPAKNI.NS', 'ALKYLAMINE.NS', 'CLEAN.NS', 'NOCIL.NS', 'VINDHYATEL.NS',
            'JSWENERGY.NS', 'RENUKA.NS', 'DHAMPUR.NS', 'BALRAMCHIN.NS', 'BAJAJCON.NS',
            'EMAMILTD.NS', 'GODREJIND.NS', 'JYOTHYLAB.NS', 'CHOLAHLDNG.NS', 'ENDURANCE.NS',
            'SUNDRMFAST.NS', 'MINDAIND.NS', 'SWARAJENG.NS', 'KIOCL.NS', 'HINDCOPPER.NS',
            'NATIONALUM.NS', 'RATNAMANI.NS', 'APOLLOTYRE.NS', 'CEAT.NS', 'JK.NS',
            'BOSCHLTD.NS', 'BAJAJHLDNG.NS', 'FORCEMOT.NS', 'MAHINDCIE.NS', 'BHARATFORG.NS',
            'AMBUJCEM.NS', 'ACC.NS', 'INDIACEM.NS', 'DALMIA.NS', 'JKLAKSHMI.NS',
            'WABCOINDIA.NS', 'LIBERTY.NS', 'MIRCHUTE.NS', 'WESTLIFE.NS', 'DEVYANI.NS',
            'SPECIALITY.NS', 'JUSTDIAL.NS', 'RATEGAIN.NS', 'TATAELXSI.NS', 'CYIENT.NS',
            'KPITTECH.NS', 'ZENSAR.NS', 'SONATSOFTW.NS', 'NIITTECH.NS', 'HAPPSTMNDS.NS',
            'INTELLECT.NS', 'RAMKY.NS', 'VAIBHAVGBL.NS', 'NYKAA.NS', 'CARTRADE.NS',
            'EASEMYTRIP.NS', 'RVNL.NS', 'RAILVIKAS.NS', 'IREDA.NS', 'SJVN.NS',
            'POWERINDIA.NS', 'TORPOWER.NS', 'RELINFRA.NS', 'ADANIGAS.NS', 'MAHINDRACO.NS',
            'LINDEINDIA.NS', 'PRAXAIR.NS', 'INOXAIR.NS', 'BASF.NS', 'AKZOINDIA.NS'
        ]
    def get_all_stock_symbols(self):
        """Returns a combined, unique list of all large, mid, and small cap stock symbols."""
        all_symbols = (
            self.large_cap_stocks +
            self.mid_cap_stocks +
            self.small_cap_stocks
        )
        return sorted(list(set(all_symbols))) # Return unique and sorted list

    def get_stock_by_category(self, category):
        """Returns stock symbols for a given category."""
        if category == "Large Cap":
            return self.large_cap_stocks
        elif category == "Mid Cap":
            return self.mid_cap_stocks
        elif category == "Small Cap":
            return self.small_cap_stocks
        else:
            return []

# Initialize StockAnalyzer
stock_analyzer = StockAnalyzer()
all_stock_symbols = stock_analyzer.get_all_stock_symbols()

class PortfolioBuilder:
    """Manages asset allocation, investment projections, and stock suggestions."""
    def __init__(self, stock_analyzer_instance):
        self.stock_analyzer = stock_analyzer_instance
        # Average annual returns for different asset classes (hypothetical, for projection)
        self.average_annual_returns = {
            "Large Cap": 0.12, "Mid Cap": 0.15, "Small Cap": 0.18
            
        }
        # Risk-based asset allocation percentages
        self.risk_profiles = {
            "Conservative": {"Large Cap": 0.50, "Mid Cap": 0.30, "Small Cap": 0.20},
            "Moderate": {"Large Cap": 0.30, "Mid Cap": 0.50, "Small Cap": 0.20},
            "Aggressive": {"Large Cap": 0.20, "Mid Cap": 0.40, "Small Cap": 0.40}
        }

    def get_asset_allocation(self, risk_profile):
        return self.risk_profiles.get(risk_profile, {})

    def project_investment(self, initial_investment, monthly_sip, duration_years, risk_profile):
        allocation = self.get_asset_allocation(risk_profile)
        total_projected_value = 0

        for asset, percentage in allocation.items():
            rate = self.average_annual_returns.get(asset, 0)
            # Future value of lump sum
            fv_lump_sum = initial_investment * percentage * ((1 + rate)**duration_years)

            # Future value of SIP (Annuity Future Value Formula)
            monthly_rate = rate / 12
            duration_months = duration_years * 12
            fv_sip = monthly_sip * percentage * (((1 + monthly_rate)**duration_months - 1) / monthly_rate) if monthly_rate != 0 else monthly_sip * percentage * duration_months

            total_projected_value += fv_lump_sum + fv_sip
        return total_projected_value

    def get_stock_suggestions(self, risk_profile):
        allocation = self.get_asset_allocation(risk_profile)
        suggestions = {}
        for asset, percentage in allocation.items():
            if percentage > 0:
                # Get stocks from StockAnalyzer based on asset class
                if asset == "Large Cap":
                    available_stocks = self.stock_analyzer.large_cap_stocks
                elif asset == "Mid Cap":
                    available_stocks = self.stock_analyzer.mid_cap_stocks
                elif asset == "Small Cap":
                    available_stocks = self.stock_analyzer.small_cap_stocks
        
                else:
                    available_stocks = []

                if available_stocks:
                    num_suggestions = min(3, len(available_stocks)) # Suggest up to 3 stocks
                    suggestions[asset] = random.sample(available_stocks, num_suggestions)
        return suggestions

# --- UPDATED CHATBOT CLASS USING GEMINI AND LANGCHAIN ---
class Chatbot:
    """A chatbot integrated with Google's Gemini models via LangChain for conversational AI."""
    def __init__(self):
        # --- LINE TO PUT YOUR GEMINI API KEY ---
        # It's highly recommended to use Streamlit Secrets for production deployment:
        # Create a file named .streamlit/secrets.toml in your project root,
        # and add: GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
        # Then access it via st.secrets["GOOGLE_API_KEY"]
        gemini_api_key = st.secrets.get("GOOGLE_API_KEY") # Use .get() to avoid KeyError if not found

        if not gemini_api_key:
            st.error("Google Gemini API key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
            self.conversation = None
            # Optionally st.stop() here if chatbot functionality is critical
        else:
            # Initialize Gemini LLM
            # Using "models/gemini-pro" which is the more explicit and robust way to call the model
            # Temperature controls creativity (0.0 for factual, higher for more creative)
            llm = ChatGoogleGenerativeAI(google_api_key=gemini_api_key, temperature=0.7, model="models/gemini-2.5-flash-preview-05-20")
            # Initialize ConversationChain with memory to maintain chat history
            self.conversation = ConversationChain(
                llm=llm,
                memory=ConversationBufferMemory()
            )
            # Initialize chat history in session state if not already present
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

    # --- CORRECTED get_response METHOD ---
    def get_response(self, query):
        """
        Gets a response from the chatbot.
        This method is now safe to call from within st.spinner because it no longer calls st.error itself.
        Instead, it returns an error message as a string, which can be safely displayed.
        """
        if self.conversation:
            try:
                # Predict the response based on the query and current conversation history
                response = self.conversation.predict(input=query)
                return response
            except Exception as e:
                # Return the error message as a string instead of calling st.error
                return f"I'm sorry, I encountered an error while processing your request: {e}. Please try again."
        else:
            return "Chatbot is not initialized. Please ensure the Google Gemini API key is set correctly."

# --- END UPDATED CHATBOT CLASS ---


# --- Sidebar Content ---
st.sidebar.header("Select a Stock")

# Use the combined list from StockAnalyzer for the primary selectbox
selected_stock_symbol_from_list = st.sidebar.selectbox(
    "Choose a stock from our curated list:",
    all_stock_symbols,
    index=all_stock_symbols.index("RELIANCE.NS") if "RELIANCE.NS" in all_stock_symbols else 0 # Default to Reliance or first stock
)

custom_stock_symbol_input = st.sidebar.text_input("Or enter any NSE Symbol (e.g., NESTLEIND.NS):", value="").upper()

# Determine the final selected stock symbol for the dashboard
selected_stock_symbol = selected_stock_symbol_from_list
if custom_stock_symbol_input:
    if not custom_stock_symbol_input.endswith(".NS"):
        custom_stock_symbol_input += ".NS"
    selected_stock_symbol = custom_stock_symbol_input


# --- Chatbot in Sidebar (Persistent across tabs) ---
st.sidebar.markdown("<div class='sidebar-chatbot'>", unsafe_allow_html=True)
st.sidebar.header("🤖 AI Assistant")
st.sidebar.write("Ask me anything about finance, stocks, or general knowledge!")

# Initialize chatbot only once
if "chatbot_instance" not in st.session_state:
    st.session_state.chatbot_instance = Chatbot()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.sidebar.chat_message(message["role"]):
        st.sidebar.markdown(message["content"])

# Accept user input
user_query_sidebar = st.sidebar.chat_input("Type your question here...", key="chat_input_sidebar")

# --- ****NEW ROBUST CHATBOT LOGIC USING ST.EMPTY()**** ---
if user_query_sidebar:
    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_query_sidebar})
    with st.sidebar.chat_message("user"):
        st.sidebar.markdown(user_query_sidebar)

    # Display assistant response using the st.empty() pattern
    with st.sidebar.chat_message("assistant"):
        # 1. Create a placeholder
        placeholder = st.empty()
        # 2. Show a thinking message in the placeholder
        placeholder.markdown("🤔 Thinking...")

        # 3. Get the response from the chatbot
        response_sidebar = st.session_state.chatbot_instance.get_response(user_query_sidebar)

        # 4. Overwrite the placeholder with the actual response
        placeholder.markdown(response_sidebar)

    # 5. Add the actual response to the chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response_sidebar})


st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.markdown("---") # Separator before footer


# --- Tabs for different features ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "🔍 Stock Screener", "💰 AI Portfolio Builder",
    "🗓️ Earnings Calendar", "⚖️ Stock Comparison", "📝 AI Reports"
])

with tab1: # Dashboard Tab
    # Dynamically fetch the full name for display if available, else use symbol
    display_stock_name = selected_stock_symbol
    temp_info = get_current_stock_info(selected_stock_symbol)
    if temp_info and temp_info.get('shortName'):
        display_stock_name = temp_info['shortName']
    elif temp_info and temp_info.get('longName'):
        display_stock_name = temp_info['longName']

    st.header(f"📈 {display_stock_name} ({selected_stock_symbol})")
    st.markdown("---")

    stock_data = get_stock_data(selected_stock_symbol)
    stock_info = get_current_stock_info(selected_stock_symbol)

    if stock_data is not None and stock_info is not None:
        col1, col2, col3 = st.columns(3)
        current_price = stock_info.get('regularMarketPrice', 'N/A')
        previous_close = stock_info.get('previousClose', current_price)

        if current_price != 'N/A' and previous_close != 'N/A':
            change = current_price - previous_close
            percent_change = (change / previous_close) * 100 if previous_close else 0
            change_color = "green" if change > 0 else "red" if change < 0 else "black"
            change_icon = "▲" if change > 0 else "▼" if change < 0 else ""

            with col1:
                st.markdown(f"<div class='metric-card'><h3>Current Price</h3><p style='color:{change_color}'>{current_price:,.2f} INR</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><h3>Daily Change</h3><p style='color:{change_color}'>{change:,.2f} {change_icon}</p></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><h3>% Change</h3><p style='color:{change_color}'>{percent_change:,.2f}%</p></div>", unsafe_allow_html=True)
        else:
            st.warning("Could not fetch current price or previous close data.")

        st.subheader("Key Financial Metrics")
        metrics_data = {
            "P/E Ratio": stock_info.get('trailingPE', 'N/A'),
            "Market Cap (Cr)": f"{stock_info.get('marketCap', 0) / 10**7:,.2f}" if stock_info.get('marketCap') else 'N/A', # Convert to Crores
            "ROE (Simulated)": f"{random.uniform(10, 30):.2f}%", # Simulated
            "Debt/Equity (Simulated)": f"{random.uniform(0.1, 1.5):.2f}", # Simulated
            "Dividend Yield": f"{stock_info.get('dividendYield', 0) * 100:.2f}%" if stock_info.get('dividendYield') else 'N/A',
            "52-Week High": f"{stock_info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
            "52-Week Low": f"{stock_info.get('fiftyTwoWeekLow', 'N/A'):,.2f}",
            "Volume": f"{stock_info.get('regularMarketVolume', 'N/A'):,.0f}"
        }
        metrics_df = pd.DataFrame(metrics_data.items(), columns=["Metric", "Value"])
        st.table(metrics_df.set_index("Metric"))

        st.subheader("Historical Price Performance (1 Year)")
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                             open=stock_data['Open'],
                                             high=stock_data['High'],
                                             low=stock_data['Low'],
                                             close=stock_data['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Simulated News & Sentiment Analysis")
        news_df = get_simulated_news_and_sentiment(selected_stock_symbol)
        st.dataframe(news_df, use_container_width=True)

        st.subheader("Simulated Advanced Financials")
        advanced_financials_df = get_simulated_advanced_financials(selected_stock_symbol)
        st.dataframe(advanced_financials_df, use_container_width=True)

    else:
        st.error("Could not load data for the selected stock. Please check the symbol or try again later.")

    # --- Market Overview Section (Updated) ---
    st.header("📈 Market Overview")
    st.markdown("---")

    market_overview_data = []

    # Define Indian index symbols
    indian_indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK"
    }

    for name, symbol in indian_indices.items():
        price, change, percent_change = fetch_index_data(symbol)
        if price is not None:
            market_overview_data.append({
                "Index": name,
                "Price": f"{price:,.2f}",
                "Change": change, # Keep as float for comparison
                "% Change": f"{percent_change:,.2f}%"
            })

    if market_overview_data:
        df_market_overview = pd.DataFrame(market_overview_data)

        # Convert 'Change' column to numeric, handling potential errors
        df_market_overview['Change'] = pd.to_numeric(df_market_overview['Change'], errors='coerce')

        # Apply color based on change
        def color_change_val(val):
            if pd.isna(val):
                return 'color: black' # Default color for NaN
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            else:
                return 'color: black'

        # Apply styling directly to the DataFrame for Streamlit
        st.dataframe(df_market_overview.style.applymap(color_change_val, subset=['Change']),
                     hide_index=True)
    else:
        st.info("No market overview data available at this time.")

    st.subheader("Top Market Movers (Sample Nifty 50 Stocks)")
    # Using a smaller, fixed sample for performance
    nifty50_sample_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                              "SBIN.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS"]
    movers_data = []
    for symbol in nifty50_sample_symbols:
        info = get_current_stock_info(symbol)
        if info:
            current_price = info.get('regularMarketPrice')
            previous_close = info.get('previousClose')
            if current_price and previous_close:
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100 if previous_close else 0
                movers_data.append({
                    "Symbol": symbol,
                    "Price": f"{current_price:,.2f}",
                    "Change": change,
                    "% Change": f"{percent_change:,.2f}%"
                })
    if movers_data:
        df_movers = pd.DataFrame(movers_data)
        df_movers['Change_Num'] = pd.to_numeric(df_movers['Change'], errors='coerce') # For sorting
        top_gainers = df_movers.nlargest(5, 'Change_Num').drop(columns=['Change_Num'])
        top_losers = df_movers.nsmallest(5, 'Change_Num').drop(columns=['Change_Num'])

        col_g, col_l = st.columns(2)
        with col_g:
            st.markdown("#### Top Gainers")
            st.dataframe(top_gainers.style.applymap(lambda x: 'color: green' if isinstance(x, str) and '+' in x else '', subset=['% Change']),
                         hide_index=True)
        with col_l:
            st.markdown("#### Top Losers")
            st.dataframe(top_losers.style.applymap(lambda x: 'color: red' if isinstance(x, str) and '-' in x else '', subset=['% Change']),
                         hide_index=True)
    else:
        st.info("Could not fetch data for market movers.")

with tab2: # Stock Screener Tab
    st.header("🔍 Custom Stock Screener")
    st.markdown("---")

    st.write("Filter Nifty 50 sample stocks based on your criteria:")

    # Use the specific screener data function
    screener_df = get_nifty50_sample_data_for_screener()

    if not screener_df.empty:
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            # Safely determine min/max for sliders, handling potential empty data or NaNs
            # P/E Ratio
            pe_min_val_for_slider = 0.0 # Slider starts from 0 for P/E
            pe_max_val_for_data =  pe_max_val_for_data = float(screener_df['P/E Ratio'].max()) if not screener_df['P/E Ratio'].empty else 150.0 # Max from data or default
            # CHANGED: Label to "Max P/E Ratio" and default value to a higher number
            pe_max = st.slider("Max P/E Ratio", 0.0, 300.0, 150.0, 0.1) # Or any other fixed max value you prefer, like 200.0 or 500.0

            # ROE (%)
            roe_min_val_for_slider = 0.0
            roe_max_val_for_data = float(screener_df['ROE (%)'].max()) if not screener_df['ROE (%)'].empty else 50.0
            roe_min = st.slider("Min ROE (%)", roe_min_val_for_slider, roe_max_val_for_data + 0.1, 15.0)

            # Debt/Equity
            debt_equity_min_val_for_slider = 0.0
            debt_equity_max_val_for_data = float(screener_df['Debt/Equity'].max()) if not screener_df['Debt/Equity'].empty else 5.0
            debt_equity_max = st.slider("Max Debt/Equity", debt_equity_min_val_for_slider, debt_equity_max_val_for_data + 0.1, 1.0)
        with col_s2:
            # Market Cap (Cr)
            market_cap_min_val_for_slider = 0.0 # Slider starts from 0 for Market Cap
            market_cap_max_val_for_data = float(screener_df['Market Cap (Cr)'].max()) if not screener_df['Market Cap (Cr)'].empty else 500000.0 # Max from data or default
            # CHANGED: Label to "Max Market Cap (Cr)" and default value to a higher number
            market_cap_max = st.slider("Max Market Cap (Cr)", market_cap_min_val_for_slider, market_cap_max_val_for_data + 0.1, market_cap_max_val_for_data)

            # YoY Revenue Growth (%)
            yoy_rev_min_val_for_slider = 0.0
            yoy_rev_max_val_for_data = float(screener_df['YoY Revenue Growth (%)'].max()) if not screener_df['YoY Revenue Growth (%)'].empty else 50.0
            yoy_revenue_growth_min = st.slider("Min YoY Revenue Growth (%)", yoy_rev_min_val_for_slider, yoy_rev_max_val_for_data + 0.1, 10.0)

            # YoY PAT Growth (%)
            yoy_pat_min_val_for_slider = 0.0
            yoy_pat_max_val_for_data = float(screener_df['YoY PAT Growth (%)'].max()) if not screener_df['YoY PAT Growth (%)'].empty else 50.0
            yoy_pat_growth_min = st.slider("Min YoY PAT Growth (%)", yoy_pat_min_val_for_slider, yoy_pat_max_val_for_data + 0.1, 15.0)


        filtered_df = screener_df[
            # CHANGED: P/E Ratio filter from >= to <=
            (screener_df['P/E Ratio'] <= pe_max) &
            (screener_df['ROE (%)'] >= roe_min) &
            (screener_df['Debt/Equity'] <= debt_equity_max) &
            # CHANGED: Market Cap filter from >= to <=
            (screener_df['Market Cap (Cr)'] <= market_cap_max) &
            (screener_df['YoY Revenue Growth (%)'] >= yoy_revenue_growth_min) &
            (screener_df['YoY PAT Growth (%)'] >= yoy_pat_growth_min)
        ]

        if not filtered_df.empty:
            st.subheader("Matching Stocks")
            st.dataframe(filtered_df.round(2), use_container_width=True)
        else:
            st.info("No stocks match your criteria. Try adjusting the filters.")
    else:
        st.warning("Could not load Nifty 50 sample data for the screener. Please try again later.")
with tab3:  # AI Portfolio Builder Tab
    st.header("💰 AI Portfolio Builder")
    st.markdown("---")

    # Pass the stock_analyzer instance to PortfolioBuilder
    portfolio_builder_instance = PortfolioBuilder(stock_analyzer)

    st.subheader("1. Risk-Based Asset Allocation")
    risk_profile = st.selectbox(
        "Select your risk profile:",
        ["Conservative", "Moderate", "Aggressive"]
    )
    allocation = portfolio_builder_instance.get_asset_allocation(risk_profile)

    # Filter only Large, Mid, and Small Cap allocations
    allowed_classes = {"Large Cap", "Mid Cap", "Small Cap"}
    allocation = {k: v for k, v in allocation.items() if k in allowed_classes}

    if allocation:
        st.write(f"Based on a **{risk_profile}** risk profile, here's a suggested asset allocation:")
        allocation_df = pd.DataFrame(allocation.items(), columns=["Asset Class", "Allocation (%)"])
        allocation_df['Allocation (%)'] = allocation_df['Allocation (%)'] * 100
        st.dataframe(allocation_df.set_index("Asset Class"), use_container_width=True)

        fig_pie = px.pie(allocation_df, values='Allocation (%)', names='Asset Class',
                         title='Suggested Portfolio Allocation',
                         hole=0.4)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Could not retrieve asset allocation for the selected risk profile.")

    st.subheader("2. Investment Projection")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        initial_investment = st.number_input("One-time Investment (INR)", min_value=0, value=100000, step=10000)
    with col_p2:
        monthly_sip = st.number_input("Monthly SIP (INR)", min_value=0, value=5000, step=1000)
    duration_years = st.slider("Investment Duration (Years)", min_value=1, max_value=30, value=10)

    if st.button("Project Investment"):
        projected_value = portfolio_builder_instance.project_investment(
            initial_investment, monthly_sip, duration_years, risk_profile
        )
        st.subheader("Projected Investment Value:")
        st.metric(label=f"Estimated Value in {duration_years} Years", value=f"INR {projected_value:,.2f}")

        st.markdown(f"""
            <details>
            <summary>Assumptions for Projection</summary>
            _This projection uses **simulated average annual returns** based on your risk profile
            (e.g., Large Cap: {portfolio_builder_instance.average_annual_returns['Large Cap']*100:.1f}%,
            Mid Cap: {portfolio_builder_instance.average_annual_returns['Mid Cap']*100:.1f}%,
            Small Cap: {portfolio_builder_instance.average_annual_returns['Small Cap']*100:.1f}%)._

            _It's a simplified calculation and does not account for market volatility, inflation, taxes, fees, or actual historical performance of specific investments.
            For **SIP**, the calculation uses the future value of an annuity formula based on your monthly investment.
            **Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.**_
            </details>
        """)

    st.subheader("3.Stock Suggestions")
    st.write("Here are some sample stock suggestions based on your selected risk profile and asset allocation. These are for illustrative purposes only and require thorough research before investing.")
    suggestions = portfolio_builder_instance.get_stock_suggestions(risk_profile)

    for asset, stock_list in suggestions.items():
        if asset in allowed_classes:
            if stock_list:
                st.markdown(f"**{asset} Stocks:** {', '.join(stock_list)}")
            else:
                st.markdown(f"**{asset} Stocks:** No specific suggestions available for this category.")


with tab4: # Earnings Calendar Tab
    st.header("🗓️ Earnings Calendar & Alerts")
    st.markdown("---")

    st.write("This section would display upcoming earnings announcements and allow you to set alerts.")

    # --- Simulated Earnings Calendar ---
    st.subheader("Simulated Upcoming Earnings (Next 30 Days)")
    today = datetime.now().date()
    # Use a small random sample of stocks from the analyzer for the calendar
    sample_symbols_for_earnings = random.sample(all_stock_symbols, min(10, len(all_stock_symbols)))

    earnings_data = []
    for symbol in sample_symbols_for_earnings:
        # FIXED: Handle cases where get_current_stock_info returns None
        info = get_current_stock_info(symbol)
        if info:
            company_name = info.get('shortName', info.get('longName', symbol.replace(".NS", "")))
        else:
            company_name = symbol.replace(".NS", "") # Fallback to symbol

        earnings_data.append({
            "Company": company_name,
            "Symbol": symbol,
            "Date": (today + timedelta(days=random.randint(2, 30))).strftime("%Y-%m-%d"),
            "Time": random.choice(["Before Market Open", "After Market Close", "During Market Hours"])
        })

    earnings_df = pd.DataFrame(earnings_data)
    earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
    earnings_df = earnings_df.sort_values(by='Date').reset_index(drop=True)
    st.dataframe(earnings_df, use_container_width=True)

    st.subheader("Set Earnings Alert (Simulated)")
    alert_company_options = earnings_df['Company'].tolist()
    if alert_company_options:
        alert_company = st.selectbox("Select company for alert:", alert_company_options)
        alert_type = st.radio("Alert me via:", ["Email", "SMS"])

        # NEW: User input for email/phone number
        user_contact = ""
        if alert_type == "Email":
            user_contact = st.text_input("Enter your Email Address:")
        elif alert_type == "SMS":
            user_contact = st.text_input("Enter your Phone Number (with country code, e.g., +91XXXXXXXXXX):")

        if st.button("Set Alert (Simulated)"):
            if user_contact:
                st.success(f"Simulated alert set for {alert_company} via {alert_type} to {user_contact}. In a real app, this would use services like Twilio (SMS) or SendGrid (Email) and a backend.")
            else:
                st.warning(f"Please enter your {alert_type.lower()} address/number to set the alert.")
    else:
        st.info("No companies available for setting earnings alerts.")


with tab5: # Stock Comparison Tab
    st.header("⚖️ Stock Comparison Tool")
    st.markdown("---")

    st.write("Compare key financial metrics of up to 3 stocks side-by-side.")

    # Use all available symbols from StockAnalyzer for comparison selection
    comparison_options = all_stock_symbols

    # Set default values to common large caps if available, otherwise first few in the list
    default_indices = []
    default_symbols_to_try = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    for sym in default_symbols_to_try:
        try:
            default_indices.append(comparison_options.index(sym))
        except ValueError:
            # If default symbol not in list, find next available index
            if len(default_indices) < len(comparison_options):
                default_indices.append(len(default_indices)) # Use next available index

    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        comp_stock1 = st.selectbox("Stock 1", comparison_options, index=default_indices[0] if len(default_indices) > 0 else 0, key="comp_stock1")
    with col_comp2:
        comp_stock2 = st.selectbox("Stock 2", comparison_options, index=default_indices[1] if len(default_indices) > 1 else (1 if len(comparison_options) > 1 else 0), key="comp_stock2")
    with col_comp3:
        comp_stock3 = st.selectbox("Stock 3", comparison_options, index=default_indices[2] if len(default_indices) > 2 else (2 if len(comparison_options) > 2 else 0), key="comp_stock3")

    comparison_symbols = [comp_stock1, comp_stock2, comp_stock3]
    comparison_data = []

    metrics_to_compare = [
        'regularMarketPrice', 'trailingPE', 'marketCap', 'dividendYield',
        'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'sector', 'industry'
    ]

    for stock_sym in comparison_symbols:
        info = get_current_stock_info(stock_sym)
        # FIXED: Handle cases where get_current_stock_info returns None
        if info:
            # Get display name for the comparison table
            display_name = info.get('shortName', info.get('longName', stock_sym))
            row_data = {"Symbol": display_name} # Use the display name here
            for metric in metrics_to_compare:
                val = info.get(metric, 'N/A')
                if metric == 'marketCap' and val != 'N/A':
                    val = f"{val / 10**7:,.2f} Cr" # Convert to Crores
                elif metric == 'dividendYield' and val != 'N/A':
                    val = f"{val * 100:.2f}%"
                elif metric == 'regularMarketPrice' and val != 'N/A':
                    val = f"{val:,.2f}"
                elif metric in ['trailingPE', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'] and val != 'N/A':
                    val = f"{val:,.2f}"
                row_data[metric.replace('regularMarketPrice', 'Price').replace('trailingPE', 'P/E Ratio').replace('marketCap', 'Market Cap').replace('dividendYield', 'Dividend Yield').replace('fiftyTwoWeekHigh', '52W High').replace('fiftyTwoWeekLow', '52W Low').replace('sector', 'Sector').replace('industry', 'Industry')] = val
            comparison_data.append(row_data)
        else:
            # Create a row with N/A values if info is None
            row_data = {"Symbol": stock_sym.replace(".NS", "")}
            for metric in metrics_to_compare:
                row_data[metric.replace('regularMarketPrice', 'Price').replace('trailingPE', 'P/E Ratio').replace('marketCap', 'Market Cap').replace('dividendYield', 'Dividend Yield').replace('fiftyTwoWeekHigh', '52W High').replace('fiftyTwoWeekLow', '52W Low').replace('sector', 'Sector').replace('industry', 'Industry')] = 'N/A'
            comparison_data.append(row_data)


    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        st.subheader("Comparison Table")
        # Set the 'Symbol' column as index after creation if desired for better display
        st.dataframe(comp_df.set_index("Symbol"), use_container_width=True)
    else:
        st.info("Select stocks above to see their comparison.")

with tab6: # AI-Generated Stock Reports Tab
    st.header("📝 AI-Generated Stock Reports")
    st.markdown("---")

    st.write("Get a quick, AI-generated summary report for a selected stock based on its simulated financials and news.")

    # Use all available symbols from StockAnalyzer for report selection
    report_stock_symbol = st.selectbox("Select a stock for AI Report:", all_stock_symbols, key="report_stock_select")

    # Get the display name for the button
    report_display_name = report_stock_symbol
    temp_info_report = get_current_stock_info(report_stock_symbol)
    if temp_info_report:
        if temp_info_report.get('shortName'):
            report_display_name = temp_info_report['shortName']
        elif temp_info_report.get('longName'):
            report_display_name = temp_info_report['longName']
    else:
        report_display_name = report_stock_symbol.replace(".NS","")


    if st.button(f"Generate Report for {report_display_name} (Simulated)"):
        st.subheader(f"AI Report for {report_display_name} ({report_stock_symbol})")

        stock_info_r = get_current_stock_info(report_stock_symbol)
        advanced_financials_r = get_simulated_advanced_financials(report_stock_symbol)
        news_r = get_simulated_news_and_sentiment(report_stock_symbol)

        if stock_info_r and not advanced_financials_r.empty and not news_r.empty:
            report_text = f"**Executive Summary for {report_display_name} ({report_stock_symbol}):**\n\n"

            # Overview from basic info
            current_price = stock_info_r.get('regularMarketPrice', 'N/A')
            pe_ratio = stock_info_r.get('trailingPE', 'N/A')
            market_cap_cr = f"{stock_info_r.get('marketCap', 0) / 10**7:,.2f}" if stock_info_r.get('marketCap') else 'N/A'
            dividend_yield = f"{stock_info_r.get('dividendYield', 0) * 100:.2f}%" if stock_info_r.get('dividendYield') else 'N/A'
            sector = stock_info_r.get('sector', 'N/A')

            report_text += f"- Current Price: INR {current_price:,.2f}\n"
            report_text += f"- P/E Ratio: {pe_ratio}\n"
            report_text += f"- Market Cap: {market_cap_cr} Crores\n"
            report_text += f"- Dividend Yield: {dividend_yield}\n"
            report_text += f"- Sector: {sector}\n\n"

            # Insights from simulated advanced financials
            yoy_rev_growth_val = advanced_financials_r[advanced_financials_r['Metric'] == 'YoY Revenue Growth']['Value'].iloc[0]
            yoy_pat_growth_val = advanced_financials_r[advanced_financials_r['Metric'] == 'YoY PAT Growth']['Value'].iloc[0]
            dii_holding_val = advanced_financials_r[advanced_financials_r['Metric'] == 'DII Holding (%)']['Value'].iloc[0]
            fii_holding_val = advanced_financials_r[advanced_financials_r['Metric'] == 'FII Holding (%)']['Value'].iloc[0]

            report_text += "**Financial Performance (Simulated):**\n"
            report_text += f"- YoY Revenue Growth: {yoy_rev_growth_val}\n"
            report_text += f"- YoY PAT Growth: {yoy_pat_growth_val}\n"
            report_text += f"- DII Holding: {dii_holding_val}%, FII Holding: {fii_holding_val}%\n\n"

            try:
                if isinstance(yoy_pat_growth_val, str) and '%' in yoy_pat_growth_val and float(yoy_pat_growth_val.replace('%','')) > 15:
                    report_text += "The company shows strong simulated year-on-year PAT growth, indicating healthy profitability expansion. "
                if isinstance(dii_holding_val, str) and float(dii_holding_val) > 10 or (isinstance(fii_holding_val, str) and float(fii_holding_val) > 20):
                    report_text += "Significant institutional holding (DII/FII) suggests confidence from major investors.\n\n"
            except (ValueError, TypeError):
                report_text += "Growth and holding percentages could not be parsed for detailed analysis. "


            # Insights from simulated news sentiment
            positive_news = news_r[news_r['Sentiment'] == 'Positive'].shape[0]
            negative_news = news_r[news_r['Sentiment'] == 'Negative'].shape[0]
            neutral_news = news_r[news_r['Sentiment'] == 'Neutral'].shape[0]

            report_text += "**Recent News Sentiment (Simulated):**\n"
            report_text += f"- Out of {len(news_r)} recent news headlines, {positive_news} were Positive, {negative_news} were Negative, and {neutral_news} were Neutral.\n"
            if positive_news > negative_news:
                report_text += "Overall sentiment from recent simulated news appears predominantly positive, which could indicate favorable market perception. "
            elif negative_news > positive_news:
                report_text += "Recent simulated news suggests a leaning towards negative sentiment, which warrants further investigation. "
            else:
                report_text += "News sentiment is mixed/neutral, suggesting no strong directional catalyst from recent headlines.\n\n"


            report_text += "**Disclaimer:** *This report is based on simulated and publicly available data and is for illustrative purposes only. It is not financial advice. Consult a qualified financial advisor for investment decisions.*"
            st.markdown(report_text)
        else:
            st.error("Could not generate report due to missing data for the selected stock.")
        st.markdown("*(Note: A full AI-generated report would require integration with a powerful Large Language Model (LLM) like Google's Gemini or similar, which would analyze real-time data and generate comprehensive insights.)*")


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
