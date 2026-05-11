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

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

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
                "order_val": 100.0, "trailing_pct": 0.02, "profit_target": 0.05, 
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
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

init_session_state()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🤖 Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=save_settings)
    st.toggle("Allow Extended Hours", key="allow_ext_hours", on_change=save_settings)
    st.slider("AI Trigger Threshold", 0.70, 0.98, key="ai_threshold", on_change=save_settings)

    st.divider()
    st.header("📂 Watchlist")
    new_t = st.text_input("Add Ticker").upper().strip()
    if st.button("➕ Add"):
        if new_t and new_t not in st.session_state.tickers:
            st.session_state.tickers.append(new_t); save_settings(); st.rerun()
    st.multiselect("Active Watchlist", options=st.session_state.tickers, key="tickers", on_change=save_settings)

    st.divider()
    st.header("🛡️ Strategy Settings")
    st.number_input("Profit Goal ($)", key="global_profit_goal", on_change=save_settings)
    st.number_input("Loss Limit ($)", key="daily_loss_limit", on_change=save_settings)

    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        add_log("EMERGENCY SHUTDOWN"); st.session_state.run_bot = False; save_settings(); st.rerun()

# --- 4. ENGINES ---
def get_market_status():
    try:
        clock = trading_client.get_clock()
        if clock.is_open: return "OPEN", "#28a745"
        now = datetime.now()
        if (now.hour < 9 or (now.hour == 9 and now.minute < 30)) or (now.hour >= 16):
            return "EXTENDED", "#fd7e14"
        return "CLOSED", "#dc3545"
    except: return "UNKNOWN", "gray"

def get_account_details():
    try:
        acc = trading_client.get_account()
        return float(acc.cash), float(acc.equity), float(acc.last_equity)
    except: return 0.0, 0.0, 0.0

def get_ai_prediction(df):
    try:
        df = df.copy()
        df.ta.rsi(append=True); df.ta.macd(append=True)
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.002).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'MACD'])]
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        prob = float(model.predict_proba(df[features].tail(1))[:, 1])
        return prob, df['RSI_14'].iloc[-1], df['MACDh_12_26_9'].iloc[-1]
    except: return 0.5, 50.0, 0.0

# --- 5. DASHBOARD UI ---
st.title("🚀 AI Alpha Terminal")

@st.fragment(run_every=1)
def dashboard_fragment():
    # Countdown Logic
    refresh_interval = 30
    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, int(refresh_interval - elapsed))

    if remaining == 0:
        st.session_state.last_refresh = time.time()
        st.rerun()

    cash, equity, last_equity = get_account_details()
    m_text, m_color = get_market_status()
    daily_pnl = equity - last_equity

    # --- Top Row ---
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("AVAILABLE CASH", f"${cash:,.2f}")
    b2.metric("PORTFOLIO EQUITY", f"${equity:,.2f}")

    # Market Status & Countdown
    b3.markdown(f"""
        <div style="background-color:{m_color}; padding:10px; border-radius:10px; text-align:center;">
            <h3 style="color:white; margin:0;">MARKET {m_text}</h3>
            <p style="color:white; margin:0; font-size: 0.8em;">Refreshing in {remaining}s</p>
        </div>
    """, unsafe_allow_html=True)

    b4.metric("DAILY PnL", f"${daily_pnl:.2f}", delta=f"{daily_pnl:.2f}")

    # --- Signals ---
    st.subheader("⚡ AI Signal Feed & Technical Factors")
    h1, h2, h3, h4, h5, h6 = st.columns([1,1,1,1,1,1])
    h1.caption("SYMBOL"); h2.caption("PRICE"); h3.caption("AI CONF."); h4.caption("RSI"); h5.caption("MACD HIST"); h6.caption("ACTION")

    for s in st.session_state.tickers:
        try:
            df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=60), feed=DataFeed.IEX)).df.reset_index()
            ai_conf, rsi, macd_h = get_ai_prediction(df)
            price = float(df['close'].iloc[-1])

            c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
            c1.write(f"**{s}**")
            c2.write(f"${price:.2f}")
            c3.write(f"**{ai_conf*100:.0f}%**")
            c4.write(f"{rsi:.1f}")
            c5.write(f"{'📈' if macd_h > 0 else '📉'} {macd_h:.2f}")
            if c6.button("Buy", key=f"b_{s}"):
                q = round(st.session_state.order_val/price, 2)
                trading_client.submit_order(MarketOrderRequest(symbol=s, qty=q, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                add_log(f"Manual Buy: {s}")
        except: continue

    st.divider()
    st.subheader("📜 Activity Log")
    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-10:]):
            st.text(log)

dashboard_fragment()
