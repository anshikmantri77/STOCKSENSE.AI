# 📈 StockSense AI – Smart Stock Market Intelligence Dashboard

**StockSense AI** is a powerful Streamlit-based web application that provides real-time stock analysis, FinBERT-powered sentiment analysis, portfolio building, AI-generated reports, and interactive financial dashboards.

Built for individual investors, financial analysts, and learners exploring stock market data and AI integration.

---

## 🚀 Features

- 📊 **Stock Dashboard** – Real-time price data, key financial metrics, candlestick charts, and market overview
- 🧠 **AI Assistant (Gemini LLM)** – Ask anything about finance or stocks via sidebar chatbot
- 🔍 **Custom Screener** – Filter stocks by P/E, ROE, Debt/Equity, growth rates, and market cap (real data)
- 💰 **Portfolio Builder** – Build portfolios based on risk profile with investment projections
- 🗓️ **Earnings Calendar** – Real earnings dates via yfinance
- ⚖️ **Stock Comparison** – Compare up to 3 stocks side-by-side with correlation + beta
- 📝 **AI Reports** – Generate summary reports using real financial data + sentiment
- 🏷️ **FinBERT Sentiment** – Real NLP sentiment classification on live news headlines

---

## 🌐 Live Demo

🚀 Check out the deployed application here:  
👉 **[StockSense AI Live App](https://anshikmantri-stocksense-ai.streamlit.app/)**

---

## 🛠️ Tech Stack

| Component             | Technology                            |
|-----------------------|---------------------------------------|
| Frontend              | Streamlit                             |
| Data Sources          | Yahoo Finance (via `yfinance`)        |
| Charts                | Plotly                                |
| AI Assistant          | LangChain + Gemini LLM                |
| Sentiment Engine      | FinBERT (ProsusAI) via HuggingFace    |
| Caching               | `st.cache_data` TTL-based             |
| Language              | Python 3.11+                          |
| Containerization      | Docker                                |

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/stocksense-ai.git
cd stocksense-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
Create a file called `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your_gemini_api_key_here"
```

Alternatively, copy the env example and set the environment variable:
```bash
cp .env.example .env
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### 4. Run the App
```bash
streamlit run stocksense/main.py
```

---

## 🧠 AI Capabilities

- Built-in chatbot (Gemini) answers finance & market queries
- FinBERT transformer model for real-time news sentiment classification
- Generates comprehensive reports based on real financial data

---

## 📋 Data Sources

This app uses **real data** from Yahoo Finance via the `yfinance` library:
- Real-time and historical stock prices
- Actual financial metrics (P/E, ROE, Market Cap, etc.)
- Live news headlines with FinBERT sentiment analysis
- Real earnings calendar data

---

## 🧩 Project Structure

```
stocksense/
├── main.py                  # Streamlit entry point
├── config.py                # Constants, stock lists, settings
├── modules/
│   ├── stock_data.py        # yfinance fetching + caching
│   ├── sentiment.py         # FinBERT sentiment pipeline
│   ├── ai_agent.py          # LangChain chatbot
│   ├── portfolio.py         # Portfolio builder logic
│   ├── screener.py          # Stock screener filters
│   └── charts.py            # Plotly chart functions
├── utils/
│   ├── helpers.py           # Formatting utilities
├── tests/                   # Test files
├── .streamlit/
│   └── secrets.toml
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8501`.

---

## 👨‍💻 Developed By

**Anshik Mantri**  
📧 [anshikmantri26@gmail.com](mailto:anshikmantri26@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/anshikmantri)

---

> ⚠️ **Disclaimer:** StockSense AI is built for education and learning. It does not offer financial advice or trading recommendations. Always consult a certified financial advisor.
