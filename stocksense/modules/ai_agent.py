import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from modules.stock_data import get_current_stock_info, get_news_headlines
from modules.sentiment import get_sentiment

system_prompt = (
    "You are a helpful financial AI assistant. You have tools for checking "
    "stock prices and sentiment. Use them when the user asks about stocks. "
    "If a question is not related to stocks or finance, answer conversationally. "
    "Keep responses concise and informative."
)


@tool
def get_price(ticker: str) -> str:
    """Get current stock price, P/E ratio, and 52-week range. Input: ticker symbol (e.g., RELIANCE.NS)."""
    info = get_current_stock_info(ticker)
    if not info:
        return f"Could not fetch data for {ticker}."
    price = info.get("regularMarketPrice", "N/A")
    pe = info.get("trailingPE", "N/A")
    high = info.get("fiftyTwoWeekHigh", "N/A")
    low = info.get("fiftyTwoWeekLow", "N/A")
    return (
        f"{ticker}: Current Price ₹{price} | P/E: {pe} | "
        f"52W High: ₹{high} | 52W Low: ₹{low}"
    )


@tool
def get_sentiment_for_stock(ticker: str) -> str:
    """Get FinBERT sentiment score for a stock based on recent news. Input: ticker symbol (e.g., TCS.NS)."""
    headlines = get_news_headlines(ticker, limit=5)
    if not headlines:
        return f"No news headlines available for {ticker}."
    scores = get_sentiment(headlines)
    return (
        f"FinBERT Sentiment for {ticker}: "
        f"Positive {scores['positive']*100:.1f}%, "
        f"Negative {scores['negative']*100:.1f}%, "
        f"Neutral {scores['neutral']*100:.1f}%"
    )


tools = [get_price, get_sentiment_for_stock]


class Chatbot:
    def __init__(self):
        gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
        if not gemini_api_key:
            st.error("Google Gemini API key not found. Please set it in .streamlit/secrets.toml.")
            self.agent = None
            self.llm = None
            return

        self.llm = ChatGoogleGenerativeAI(
            google_api_key=gemini_api_key,
            temperature=0.7,
            model="models/gemini-2.5-flash-preview-05-20",
        )

        try:
            self.agent = create_react_agent(self.llm, tools, state_modifier=system_prompt)
        except Exception:
            self.agent = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def get_response(self, query: str) -> str:
        stock_keywords = [
            "stock", "price", "share", "market", "pe", "sentiment",
            "ticker", "nse", "bse", "reliance", "tcs", "infy",
            "hdfc", "icici", "sbi", "invest", "trade", "finance",
        ]
        is_stock_query = any(kw in query.lower() for kw in stock_keywords)

        if self.agent and is_stock_query:
            try:
                result = self.agent.invoke({"messages": [("human", query)]})
                return result["messages"][-1].content
            except Exception as e:
                return f"I encountered an error: {e}. Let me answer directly."

        if self.llm:
            try:
                response = self.llm.invoke(
                    [HumanMessage(content=query)]
                )
                return response.content
            except Exception as e:
                return f"I'm sorry, I encountered an error: {e}. Please try again."

        return "Chatbot is not initialized. Please ensure the Google Gemini API key is set correctly."
