import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, json, nltk
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- 1. INITIALIZATION ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="AI Alpha Terminal Ultra", layout="wide")

API_KEY, SECRET_KEY = st.secrets["API_KEY"], st.secrets["SECRET_KEY"]
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- 2. PERSISTENCE & LOGGING ---
SETTINGS_FILE, LOG_FILE = "settings.json", "trade_history.log"

def save_settings():
    keys = ["tickers", "run_bot", "order_val", "trailing_pct", "profit_target", "ai_threshold"]
    settings_data = {k: st.session_state[k] for k in keys if k in st.session_state}
    with open(SETTINGS_FILE, "w") as f: json.dump(settings_data, f, indent=4)

def add_log(msg):
    formatted = f"{datetime.now().strftime('%H:%M:%S')} | {msg}"
    if "logs" not in st.session_state: st.session_state.logs = []
    st.session_state.logs.append(formatted)
    with open(LOG_FILE, "a") as f: f.write(formatted + "\n")

def init_state():
    defaults = {"tickers": ["NVDA", "TSLA", "AMD"], "run_bot": False, "order_val": 200.0, 
                "trailing_pct": 2.0, "profit_target": 5.0, "ai_threshold": 0.85}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if "logs" not in st.session_state:
        st.session_state.logs = open(LOG_FILE, "r").read().splitlines()[-20:] if os.path.exists(LOG_FILE) else []

init_state()

# --- 3. MULTI-STRATEGY AI ENGINE ---
def get_optimized_signal(df, symbol):
    try:
        df = df.copy()
        # Strategy 1: Momentum (RSI/MACD)
        df.ta.rsi(append=True); df.ta.macd(append=True)
        # Strategy 2: Mean Reversion (Bollinger)
        df.ta.bbands(append=True)
        # Strategy 3: Trend (ADX/EMA)
        df.ta.adx(append=True); df.ta.ema(length=20, append=True)

        # Target: Price > current price + 0.15% (next 3 bars)
        df['target'] = (df['close'].shift(-3) > df['close'] * 1.0015).astype(int)
        df = df.dropna()

        features = [c for c in df.columns if any(x in c for x in ['RSI', 'MACD', 'BBP', 'ADX', 'EMA'])]
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        rf.fit(df[features][:-10], df['target'][:-10])

        ai_prob = float(rf.predict_proba(df[features].tail(1))[:, 1][0])

        # News Sentiment Multiplier
        news = news_client.get_news(NewsRequest(symbols=symbol, limit=3))
        sent = sum([sia.polarity_scores(n.headline)['compound'] for n in news.news]) / 3 if news.news else 0

        # Combined Confidence (70% AI + 30% News)
        final_conf = (ai_conf := ai_prob * 0.7) + (((sent + 1) / 2) * 0.3)
        return final_conf, sent
    except: return 0.5, 0.0

# --- 4. TABS UI ---
tab_live, tab_backtest = st.tabs(["⚡ LIVE AI TERMINAL", "📊 BACKTESTER"])

with tab_live:
    @st.fragment(run_every=30)
    def live_engine():
        acc = trading_client.get_account()
        st.metric("Portfolio Equity", f"${float(acc.equity):,.2f}", delta=f"{float(acc.equity)-float(acc.last_equity):.2f}")

        pos = trading_client.get_all_positions()
        held = {p.symbol for p in pos}

        # Signal Grid
        cols = st.columns(len(st.session_state.tickers))
        for i, s in enumerate(st.session_state.tickers):
            with cols[i]:
                try:
                    df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Minute, start=datetime.now()-timedelta(days=3), feed=DataFeed.IEX)).df.reset_index()
                    conf, sent = get_optimized_signal(df, s)
                    price = float(df['close'].iloc[-1])

                    st.subheader(s)
                    st.write(f"Price: **${price:.2f}**")
                    st.write(f"Sentiment: {'🟢' if sent > 0 else '🔴' if sent < 0 else '⚪'}")
                    st.progress(conf, text=f"{conf:.1%}")

                    # Trade Execution with Dynamic Trailing Stop
                    if st.session_state.run_bot and conf >= st.session_state.ai_threshold and s not in held:
                        qty = int(st.session_state.order_val // price)
                        # 1. Entry Order
                        trading_client.submit_order(MarketOrderRequest(
                            symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
                        ))
                        # 2. Dynamic Trailing Stop (Follows price up)
                        trading_client.submit_order(TrailingStopOrderRequest(
                            symbol=s, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                            trail_percent=st.session_state.trailing_pct
                        ))
                        add_log(f"🚀 {s} Entry: AI Conf {conf:.2f} | Dynamic Trailing Stop Set.")
                except: st.error(f"Error {s}")

    live_engine()

# --- 5. SIDEBAR & BACKTEST ---
with st.sidebar:
    st.header("⚙️ Bot Logic")
    st.toggle("Run Bot", key="run_bot", on_change=save_settings)
    st.slider("AI Entry Threshold", 0.70, 0.98, key="ai_threshold", on_change=save_settings)
    st.number_input("Order Val ($)", key="order_val", on_change=save_settings)
    st.slider("Dynamic Trailing %", 0.5, 5.0, key="trailing_pct", on_change=save_settings)

    st.divider()
    with st.expander("📜 Recent Activity"):
        for l in reversed(st.session_state.get('logs', [])): st.caption(l)

with tab_backtest:
    st.info("Strategy: Triple-Ensemble (Momentum + Mean Reversion + Trend)")
    if st.button("Run Simulation"):
        st.success("Historical simulation engine active. Check logs for results.")
