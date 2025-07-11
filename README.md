# 📈 StockSense AI – Smart Stock Market Intelligence Dashboard

**StockSense AI** is a powerful Streamlit-based web application that provides real-time stock analysis, personalized portfolio building, simulated AI-generated reports, and interactive financial dashboards — all in one place.

This tool is ideal for individual investors, financial analysts, or learners looking to explore stock market data and AI integration.

---

## 🚀 Features at a Glance

- 📊 **Stock Dashboard** – View price trends, key metrics, candlestick charts  
- 🧠 **AI Assistant (Gemini LLM)** – Ask anything about finance or stocks  
- 🔍 **Custom Screener** – Filter stocks by P/E, ROE, Debt/Equity, Growth, etc.  
- 💰 **Portfolio Builder** – Build simulated portfolios based on risk profile  
- 🗓️ **Earnings Calendar** – See upcoming earnings (simulated) + set alerts  
- ⚖️ **Stock Comparison** – Compare up to 3 stocks side-by-side  
- 📝 **AI Reports** – Generate simulated summary reports for stocks  

---

## 🌐 Live Demo

🚀 Check out the deployed application here:  
👉 **[StockSense AI Live App](https://anshik-stocksenseai.streamlit.app/)**

---

## 🛠️ Tech Stack

| Component         | Technology                     |
|------------------|--------------------------------|
| Frontend         | Streamlit                      |
| Data Sources     | Yahoo Finance (via `yfinance`) |
| Charts           | Plotly, Altair                 |
| AI Assistant     | LangChain + Gemini LLM         |
| Sentiment Engine | TextBlob                       |
| Language         | Python 3.8+                    |

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

Alternatively, set an environment variable:
```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### 4. Run the App
```bash
streamlit run trial2.py
```

---

## 🧠 AI Capabilities

- Built-in chatbot (Gemini) answers finance & market queries  
- Generates basic summary reports based on simulated financials  
- Maintains chat history across sessions using Streamlit session state  

---

## 📋 Simulated Data Notice

This app uses:
- Simulated news & financial metrics  
- Simulated earnings calendar  
- Randomized values for demo portfolio projections  

🔔 *It is for **educational/demo purposes only. Not for trading or investment advice.*

---

## 🧩 Future Additions

- ✅ Real earnings calendar integration  
- ✅ Live alerts via email or SMS (Twilio/SendGrid)  
- ✅ Full AI reports using real-time data  
- ✅ Deploy on Streamlit Cloud or Hugging Face Spaces  

---

## 👨‍💻 Developed By

**Anshik Mantri**  
📧 [anshikmantri26@gmail.com](mailto:anshikmantri26@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/anshikmantri)  
📷 [Instagram](https://www.instagram.com/anshik.m6777/)

---


> ⚠️ **Disclaimer:** StockSense AI is a simulated app built for education and learning. It does not offer financial advice or trading recommendations. Always consult a certified financial advisor.
