import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, json
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob # Lightweight sentiment analyzer

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY) # New News Client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
st.set_page_config(page_title="AI Pro Terminal + Sentiment", layout="wide")

# --- 2. INITIALIZATION ---
SETTINGS_FILE = "settings.json"

def init_session_state():
    defaults = {
        "tickers": ["SPY", "AMZN", "NVDA", "GOOGL"], 
        "run_bot": False, 
        "order_mode": "USD", 
        "order_val": 100.0,
        "logs": [],
        "trailing_pct": 0.02,
        "profit_target": 0.05,
        "ai_threshold": 0.70,
        "sentiment_weight": 0.30
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()

def save_settings():
    keys = ["tickers", "run_bot", "order_mode", "order_val", "trailing_pct", "profit_target", "ai_threshold"]
    with open(SETTINGS_FILE, "w") as f:
        json.dump({k: st.session_state[k] for k in keys}, f)

# --- 3. ANALYSIS ENGINES ---
def get_news_sentiment(symbol):
    """Fetches recent news and calculates average sentiment score (-1 to 1)"""
    try:
        req = NewsRequest(symbols=symbol, limit=5)
        news = news_client.get_news(req)
        scores = [TextBlob(n.headline).sentiment.polarity for n in news.news]
        return sum(scores) / len(scores) if scores else 0.0
    except: return 0.0

def get_ai_prediction(df):
    try:
        df = df.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU'])]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df[features][:-1], df['target'][:-1])
        prob = model.predict_proba(df[features].tail(1))
        return float(prob[0][1])
    except: return 0.50

# --- 4. DASHBOARD ---
st.title("🚀 AI Terminal: Technical + Sentiment")

@st.fragment(run_every=60)
def dashboard():
    # Performance Chart
    try:
        hist = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period="1D", timeframe="15Min"))
        st.area_chart(pd.DataFrame(hist.equity, index=pd.to_datetime(hist.timestamp, unit='s')), height=150)
    except: pass

    # Live Signal Feed
    st.subheader("⚡ Integrated Signal Feed")
    h1, h2, h3, h4, h5 = st.columns([1, 1, 1.5, 1.5, 1])
    h1.caption("SYMBOL"); h2.caption("PRICE"); h3.caption("AI CONFIDENCE"); h4.caption("NEWS MOOD"); h5.caption("ACTION")

    for s in st.session_state.tickers:
        try:
            # Data Fetching
            req = StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=365), feed=DataFeed.IEX)
            df = data_client.get_stock_bars(req).df.reset_index()

            price = float(df['close'].iloc[-1])
            ai_conf = get_ai_prediction(df)
            sentiment = get_news_sentiment(s) # -1.0 to 1.0

            # Normalize sentiment for progress bar (0.0 to 1.0)
            norm_sentiment = (sentiment + 1) / 2

            c1, c2, c3, c4, c5 = st.columns([1, 1, 1.5, 1.5, 1])
            c1.write(f"**{s}**")
            c2.write(f"${price:.2f}")
            c3.progress(ai_conf, text=f"{ai_conf*100:.0f}% Confidence")

            sentiment_label = "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral"
            c4.progress(norm_sentiment, text=f"{sentiment_label} ({sentiment:.2f})")

            if c5.button("Buy", key=f"b_{s}"):
                qty = round(st.session_state.order_val / price, 2) if st.session_state.order_mode == "USD" else st.session_state.order_val
                trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                st.toast(f"Order Sent for {s}")

            # Combined AI + Sentiment Bot Entry
            if st.session_state.run_bot:
                if ai_conf > st.session_state.ai_threshold and sentiment > 0.05:
                    qty = round(st.session_state.order_val / price, 2) if st.session_state.order_mode == "USD" else st.session_state.order_val
                    trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                    st.session_state.logs.append(f"🤖 Bot Buy: {s} (AI:{ai_conf:.2f}, Sent:{sentiment:.2f})")
        except: continue

dashboard()
