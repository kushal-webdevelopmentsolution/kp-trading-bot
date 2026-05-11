import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, json, time, nltk
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- 1. INITIALIZATION & SECRETS ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="AI Alpha Terminal Pro", layout="wide")

try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please configure API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

# Clients
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- 2. PERSISTENCE ENGINE ---
SETTINGS_FILE, LOG_FILE = "settings.json", "trade_history.log"

def save_settings():
    keys = ["tickers", "run_bot", "order_val", "trailing_pct", "profit_target", 
            "ai_threshold", "daily_loss_limit", "global_profit_goal"]
    settings_data = {k: st.session_state[k] for k in keys if k in st.session_state}
    with open(SETTINGS_FILE, "w") as f: json.dump(settings_data, f, indent=4)

def add_log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{ts} | {msg}"
    if "logs" not in st.session_state: st.session_state.logs = []
    st.session_state.logs.append(formatted)
    with open(LOG_FILE, "a") as f: f.write(formatted + "\n")

def init_state():
    defaults = {"tickers": ["SPY", "NVDA", "QQQ"], "run_bot": False, "order_val": 100.0, 
                "trailing_pct": 0.02, "profit_target": 0.04, "ai_threshold": 0.90,
                "daily_loss_limit": 500.0, "global_profit_goal": 1000.0}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if "logs" not in st.session_state:
        st.session_state.logs = open(LOG_FILE, "r").read().splitlines() if os.path.exists(LOG_FILE) else []

init_state()

# --- 3. AI & SENTIMENT ENGINES ---
def get_news_sentiment(symbol):
    try:
        req = NewsRequest(symbols=symbol, limit=5)
        news = news_client.get_news(req)
        scores = [sia.polarity_scores(n.headline)['compound'] for n in news.news]
        return sum(scores) / len(scores) if scores else 0.0
    except: return 0.0

def get_ai_prediction(df):
    try:
        df = df.copy()
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.bbands(append=True); df.ta.adx(append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c for x in ['RSI', 'MACD', 'BBP', 'ADX'])]
        model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        probs = [float(p) for p in model.predict_proba(df[features].tail(10))[:, 1]]
        return probs[-1], probs
    except: return 0.5, [0.5]*10

# --- 4. UI COMPONENTS ---
with st.sidebar:
    st.header("🤖 Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=save_settings)
    st.slider("AI Threshold (98% Goal)", 0.70, 0.98, key="ai_threshold", on_change=save_settings)
    st.divider()
    st.header("⚙️ Risk Mgmt")
    st.number_input("Order Size ($)", key="order_val", on_change=save_settings)
    st.slider("Stop Loss %", 0.01, 0.10, key="trailing_pct", on_change=save_settings)
    st.slider("Take Profit %", 0.01, 0.20, key="profit_target", on_change=save_settings)
    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        st.session_state.run_bot = False; save_settings(); st.rerun()

# Tabs
tab_live, tab_backtest = st.tabs(["⚡ LIVE TERMINAL", "📈 STRATEGY BACKTEST"])

with tab_live:
    @st.fragment(run_every=30)
    def live_loop():
        acc = trading_client.get_account()
        pnl = float(acc.equity) - float(acc.last_equity)

        # UI Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Daily PnL", f"${pnl:.2f}", delta=f"{pnl:.2f}")
        m2.metric("Equity", f"${float(acc.equity):,.2f}")
        m3.success("BOT RUNNING" if st.session_state.run_bot else "BOT STANDBY")

        # Process Watchlist
        pos = trading_client.get_all_positions()
        held = {p.symbol for p in pos}

        st.subheader("⚡ Signal Matrix")
        for s in st.session_state.tickers:
            try:
                # Fetch Data
                df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Minute, start=datetime.now()-timedelta(days=5), feed=DataFeed.IEX)).df.reset_index()
                ai_conf, conf_hist = get_ai_prediction(df)
                sentiment = get_news_sentiment(s)
                price = float(df['close'].iloc[-1])

                # Accuracy Formula: 70% AI Confidence + 30% News Sentiment
                combined_score = (ai_conf * 0.7) + (((sentiment + 1) / 2) * 0.3)

                c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
                c1.write(f"**{s}**\n${price:.2f}")
                c2.write(f"News: {'🟢' if sentiment > 0.1 else '🔴' if sentiment < -0.1 else '⚪'}")
                c3.progress(combined_score, text=f"Confidence: {combined_score:.1%}")

                # Execute Bracket Order
                if st.session_state.run_bot and combined_score >= st.session_state.ai_threshold and s not in held:
                    qty = round(st.session_state.order_val / price, 2)
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=round(price * (1 + st.session_state.profit_target), 2)),
                        stop_loss=StopLossRequest(stop_price=round(price * (1 - st.session_state.trailing_pct), 2))
                    ))
                    add_log(f"🤖 AI Entry: {s} @ {price} (Conf: {combined_score:.2f})")
            except: continue

        st.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")

    live_loop()

with tab_backtest:
    st.subheader("AI Strategy Backtester")
    col_a, col_b = st.columns(2)
    bt_ticker = col_a.selectbox("Select Asset", st.session_state.tickers)
    bt_days = col_b.slider("History (Days)", 5, 90, 30)

    if st.button("🚀 Run Backtest"):
        with st.spinner("Simulating..."):
            bars = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=bt_ticker, timeframe=TimeFrame.Hour, start=datetime.now()-timedelta(days=bt_days), feed=DataFeed.IEX)).df.reset_index()
            bars.ta.rsi(append=True); bars.ta.macd(append=True)
            bars['returns'] = bars['close'].pct_change()
            # Simple threshold simulation for logic visualization
            bars['signal'] = (bars['rsi_14'] < 40).astype(int) 
            bars['strat_ret'] = bars['signal'].shift(1) * bars['returns']
            cum_ret = (1 + bars['strat_ret'].fillna(0)).cumprod()

            st.line_chart(cum_ret)
            st.metric("Total Return", f"{(cum_ret.iloc[-1]-1)*100:.2f}%")

# Trade History Expansion
with st.expander("📜 Activity Logs"):
    for log in reversed(st.session_state.logs[-20:]):
        st.write(log)
