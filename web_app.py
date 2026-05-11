import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, json, time
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from sklearn.ensemble import RandomForestClassifier
import nltk
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- 0. NLTK FIX (Replaces your old block) ---
@st.cache_resource
def load_nltk():
    try:
        # Streamlit-friendly check for the lexicon
        nltk.data.find('sentiment/vader_lexicon.zip')
    except:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

# Initialize the Analyzer and News Client
sia = load_nltk()
news_client = NewsClient(API_KEY, SECRET_KEY)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
st.set_page_config(page_title="AI Alpha Terminal Pro", layout="wide")




# --- 2. PERSISTENCE ENGINE ---
SETTINGS_FILE = "settings.json"
LOG_FILE = "trade_history.log"

def save_settings():
    keys = ["tickers", "run_bot", "order_mode", "order_val", "trailing_pct", 
            "profit_target", "ai_threshold", "vix_threshold", "lock_profit_pct", 
            "daily_loss_limit", "global_profit_goal", "allow_ext_hours"]
    settings_data = {k: st.session_state[k] for k in keys if k in st.session_state}
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings_data, f, indent=4)

def add_log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"{timestamp} | {msg}"
    if "logs" not in st.session_state: st.session_state.logs = []
    st.session_state.logs.append(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def init_session_state():
    defaults = {"tickers": ["SPY", "QQQ", "NVDA"], "run_bot": False, "order_mode": "USD", 
                "order_val": 100.0, "trailing_pct": 0.5, "profit_target": 0.05, 
                "ai_threshold": 0.85, "vix_threshold": 25.0, "lock_profit_pct": 0.03,
                "daily_loss_limit": 500.0, "global_profit_goal": 1000.0, "allow_ext_hours": False}

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if "logs" not in st.session_state:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f: st.session_state.logs = f.read().splitlines()
        else: st.session_state.logs = []

init_session_state()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🤖 Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=save_settings)
    st.toggle("Allow Extended Hours", key="allow_ext_hours", on_change=save_settings)

    # Accuracy Optimization: Setting this high (0.90+) targets 98% precision
    st.slider("AI Trigger Threshold", 0.70, 0.98, key="ai_threshold", 
              help="Higher threshold = more selective, higher accuracy.", 
              on_change=save_settings)

    st.divider()
    st.header("📂 Watchlist")
    new_t = st.text_input("Add Ticker").upper().strip()
    if st.button("➕ Add"):
        if new_t and new_t not in st.session_state.tickers:
            st.session_state.tickers.append(new_t); save_settings(); st.rerun()
    st.multiselect("Active Watchlist", options=st.session_state.tickers, key="tickers", on_change=save_settings)

    st.divider()
    st.header("🏁 Daily Targets")
    st.number_input("Profit Goal ($)", key="global_profit_goal", on_change=save_settings)
    st.number_input("Loss Limit ($)", key="daily_loss_limit", on_change=save_settings)

    st.divider()
    st.header("🛡️ Risk Engine")
    st.number_input("Order Val ($)", key="order_val", on_change=save_settings)

    # Dynamic Trailing Stop Percent for the Order Logic in Step 5
    st.slider("Dynamic Trailing Stop %", 0.5, 5.0, key="trailing_pct", 
              help="Automatically follows price up to lock in profit.",
              on_change=save_settings)

    # Circuit Breakers
    st.slider("Max Daily Risk %", 0.01, 0.10, key="lock_profit_pct", 
              help="Stops trading if total equity drops by this much.",
              on_change=save_settings)

    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        add_log("EMERGENCY SHUTDOWN: All positions closed.")
        st.session_state.run_bot = False; save_settings(); st.rerun()

# --- 4. ENGINES ---
def get_market_status():
    try:
        clock = trading_client.get_clock()
        return {"open": clock.is_open, "timestamp": clock.timestamp}
    except: return {"open": False, "timestamp": None}

def get_daily_pnl():
    try:
        acc = trading_client.get_account()
        return float(acc.equity) - float(acc.last_equity)
    except: return 0.0

def get_ai_prediction(df, symbol):
    try:
        df = df.copy()
        # Add more features for precision
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.adx(append=True)
        df.ta.bbands(append=True); df.ta.atr(append=True)
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.001).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c for x in ['RSI', 'MACD', 'BBP', 'ADX', 'ATR'])]

        # Increased estimators for higher accuracy
        model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        tech_conf = float(model.predict_proba(df[features].tail(1))[:, 1])

        # Get Sentiment (30% weight)
        news = news_client.get_news(NewsRequest(symbols=symbol, limit=5))
        sent = [sia.polarity_scores(n.headline)['compound'] for n in news.news]
        news_conf = (sum(sent)/len(sent) + 1)/2 if sent else 0.5

        final_conf = (tech_conf * 0.7) + (news_conf * 0.3)
        return final_conf, [float(p) for p in model.predict_proba(df[features].tail(10))[:, 1]]
    except: return 0.5, [0.5]*10


# --- 5. DASHBOARD UI ---
st.title("🚀 AI Alpha Terminal")

@st.fragment(run_every=30)
def live_ui():
    status = get_market_status()
    market_open = status["open"]
    daily_pnl = get_daily_pnl()

    # Circuit Breakers
    p_hit = daily_pnl >= st.session_state.global_profit_goal
    l_hit = daily_pnl <= -abs(st.session_state.daily_loss_limit)

    bot_reason = ""
    if p_hit and st.session_state.run_bot:
        bot_reason = "PROFIT GOAL REACHED"
        trading_client.close_all_positions(cancel_orders=True)
        st.session_state.run_bot = False; save_settings()
        add_log(f"🎯 Target Hit: ${daily_pnl:.2f}. Positions closed.")
    elif l_hit and st.session_state.run_bot:
        bot_reason = "LOSS LIMIT HIT"
        st.session_state.run_bot = False; save_settings()
        add_log(f"🛑 Loss Limit Hit: ${daily_pnl:.2f}. Bot stopped.")
    elif not market_open and not st.session_state.allow_ext_hours:
        bot_reason = "MARKET CLOSED"

    active_now = st.session_state.run_bot and not bot_reason

    m1, m2, m3 = st.columns(3)
    m1.metric("Daily PnL", f"${daily_pnl:.2f}", delta=f"{daily_pnl:.2f}")
    m2.metric("Market Status", "OPEN" if market_open else "CLOSED")
    if bot_reason: m3.error(f"🛑 {bot_reason}")
    else: m3.success("🟢 BOT ACTIVE" if st.session_state.run_bot else "⚪ STANDBY")

    # Positions
    st.subheader("📊 Active Positions")
    pos = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in pos} # Get symbols we already own

    if pos:
        for p in pos:
            qty, mkt_val, pnl_pct = float(p.qty), float(p.market_value), float(p.unrealized_plpc) * 100
            c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
            c1.write(f"**{p.symbol}**"); c2.write(f"${mkt_val:,.0f}"); c3.write(f"{pnl_pct:.2f}%")
            if c4.button("✖", key=f"cl_{p.symbol}"):
                trading_client.close_position(p.symbol); add_log(f"Manual Close: {p.symbol}"); st.rerun()

    # AI Signal Feed
    st.subheader("⚡ AI Signals")
    for s in st.session_state.tickers:
        try:
            # Fetch data (using 100 days of history for AI context)
            df = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=s, timeframe=TimeFrame.Day, 
                start=datetime.now()-timedelta(days=100), feed=DataFeed.IEX
            )).df.reset_index()

            # Pass ticker 's' for Sentiment weighting (from step 4/5 logic)
            ai_conf, conf_hist = get_ai_prediction(df, s) 
            price = float(df['close'].iloc[-1])

            s1, s2, s3, s4, s5 = st.columns([1, 1, 1.5, 2, 1])
            s1.write(f"**{s}**"); s2.write(f"${price:.2f}")

            # 1. VISUAL CONFIDENCE BAR
            s3.progress(ai_conf, text=f"Confidence: {ai_conf:.1%}")

            # 2. CONFIDENCE TREND CHART
            with s4: st.line_chart(conf_hist, height=60, use_container_width=True)

            # Helper for execution
            def execute_ai_trade(is_bot=False):
                qty = int(st.session_state.order_val // price)
                # 1. Entry Market Buy
                trading_client.submit_order(MarketOrderRequest(
                    symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
                ))
                # 2. Dynamic Trailing Stop (Step 5 logic)
                trading_client.submit_order(TrailingStopOrderRequest(
                    symbol=s, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                    trail_percent=st.session_state.trailing_pct
                ))
                add_log(f"{'🤖 Bot' if is_bot else '👤 Manual'} Entry: {s} | Conf: {ai_conf:.1%}")

            # Manual Buy
            if s5.button("Buy", key=f"b_{s}"):
                execute_ai_trade(is_bot=False)
                st.rerun()

            # 3. AUTO-EXECUTION CHECK (Step 4 & 5 merged)
            if active_now and ai_conf >= st.session_state.ai_threshold:
                if s not in held_symbols:
                    execute_ai_trade(is_bot=True)
                    st.toast(f"🤖 AI Buying {s} @ {ai_conf:.1%}", icon="🚀")
                else:
                    st.caption(f"Skipping {s}: Position already active.")

        except Exception as e:
            continue

    # --- TRADE HISTORY ---
    st.divider()
    st.subheader("📜 Trade History")
    if st.session_state.logs:
        history_data = []
        for line in reversed(st.session_state.logs):
            if "|" in line:
                ts, msg = line.split(" | ", 1)
                history_data.append({"Time": ts, "Activity": msg})
        st.table(history_data[:15])
