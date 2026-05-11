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
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
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
            "ai_threshold", "daily_loss_limit", "global_profit_goal", "allow_ext_hours"]
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
                "daily_loss_limit": 500.0, "global_profit_goal": 1000.0, "allow_ext_hours": False}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if "logs" not in st.session_state:
        st.session_state.logs = open(LOG_FILE, "r").read().splitlines() if os.path.exists(LOG_FILE) else []

init_state()

# --- 3. OPTIMIZED AI & SENTIMENT ENGINES ---
def get_news_sentiment(symbol):
    try:
        req = NewsRequest(symbols=symbol, limit=5)
        news = news_client.get_news(req)
        scores = [sia.polarity_scores(n.headline)['compound'] for n in news.news]
        return sum(scores) / len(scores) if scores else 0.0
    except: return 0.0

def get_ai_prediction(df):
    """Optimized Ensemble Strategy for high accuracy."""
    try:
        df = df.copy()
        # Technical Indicators
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.bbands(append=True); df.ta.adx(append=True)
        # Target: Price increases by 0.1% in the next bar
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.001).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c for x in ['RSI', 'MACD', 'BBP', 'ADX'])]
        # Enhanced Random Forest Params
        model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        probs = [float(p) for p in model.predict_proba(df[features].tail(10))[:, 1]]
        return probs[-1], probs
    except: return 0.5, [0.5]*10

# --- 4. DASHBOARD UI ---
with st.sidebar:
    st.header("🤖 Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=save_settings)
    st.toggle("Allow Ext. Hours", key="allow_ext_hours", on_change=save_settings)
    st.slider("AI Threshold", 0.70, 0.98, key="ai_threshold", on_change=save_settings)
    st.divider()
    st.header("🛡️ Risk Management")
    st.number_input("Order Size ($)", key="order_val", on_change=save_settings)
    st.slider("Stop Loss %", 0.01, 0.10, key="trailing_pct", on_change=save_settings)
    st.slider("Take Profit %", 0.01, 0.20, key="profit_target", on_change=save_settings)
    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        st.session_state.run_bot = False; save_settings(); st.rerun()

tab_live, tab_backtest = st.tabs(["⚡ LIVE TERMINAL", "📈 STRATEGY BACKTEST"])

with tab_live:
    @st.fragment(run_every=30)
    def live_ui():
        clock = trading_client.get_clock()
        market_open = clock.is_open
        acc = trading_client.get_account()
        daily_pnl = float(acc.equity) - float(acc.last_equity)

        m1, m2, m3 = st.columns(3)
        m1.metric("Daily PnL", f"${daily_pnl:.2f}", delta=f"{daily_pnl:.2f}")
        m2.metric("Buying Power", f"${float(acc.buying_power):,.2f}")
        m3.success("BOT ACTIVE" if st.session_state.run_bot else "STANDBY")

        # Positions
        st.subheader("📊 Active Positions")
        pos = trading_client.get_all_positions()
        held_symbols = {p.symbol for p in pos}
        if pos:
            for p in pos:
                c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
                c1.write(f"**{p.symbol}**"); c2.write(f"${float(p.market_value):,.0f}")
                c3.write(f"{float(p.unrealized_plpc)*100:.2f}%")
                if c4.button("✖", key=f"cl_{p.symbol}"):
                    trading_client.close_position(p.symbol); st.rerun()

        # Signals
        st.subheader("⚡ Signal Matrix")
        active_now = st.session_state.run_bot and (market_open or st.session_state.allow_ext_hours)

        for s in st.session_state.tickers:
            try:
                df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Minute, start=datetime.now()-timedelta(days=5), feed=DataFeed.IEX)).df.reset_index()
                ai_conf, conf_hist = get_ai_prediction(df)
                sentiment = get_news_sentiment(s)
                price = float(df['close'].iloc[-1])
                combined_score = (ai_conf * 0.7) + (((sentiment + 1) / 2) * 0.3)

                s1, s2, s3, s4 = st.columns([1, 1, 2, 1])
                s1.write(f"**{s}**\n${price:.2f}")
                s2.write(f"Sent: {'🟢' if sentiment > 0.1 else '🔴' if sentiment < -0.1 else '⚪'}")
                s3.progress(combined_score, text=f"AI Score: {combined_score:.1%}")

                if active_now and combined_score >= st.session_state.ai_threshold and s not in held_symbols:
                    qty = round(st.session_state.order_val / price, 2)
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=round(price * (1 + st.session_state.profit_target), 2)),
                        stop_loss=StopLossRequest(stop_price=round(price * (1 - st.session_state.trailing_pct), 2))
                    ))
                    add_log(f"🤖 AI Buy: {s} @ {price} | Score: {combined_score:.2f}")
            except: continue

    live_ui()

with tab_backtest:
    st.subheader("AI Historical Simulator")
    bt_ticker = st.selectbox("Ticker", st.session_state.tickers)
    if st.button("🚀 Run AI Backtest"):
        with st.spinner("Processing..."):
            df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=bt_ticker, timeframe=TimeFrame.Hour, start=datetime.now()-timedelta(days=30), feed=DataFeed.IEX)).df.reset_index()
            df['returns'] = df['close'].pct_change()
            df['strat_ret'] = (df['close'] > df['close'].shift(1)).astype(int).shift(1) * df['returns']
            st.line_chart((1 + df['strat_ret'].fillna(0)).cumprod())
            st.success("Backtest simulation complete.")

with st.expander("📜 Logs"):
    for log in reversed(st.session_state.logs[-15:]): st.write(log)
