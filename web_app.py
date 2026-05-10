import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, time, concurrent.futures, json
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

st.set_page_config(page_title="AI Pro: Automated Trading", layout="wide")

SETTINGS_FILE = "settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except: return None
    return None

def save_settings():
    # Sync internal state to file
    settings = {
        "run_bot": st.session_state.bot_toggle,
        "profit_goal": st.session_state.goal_input,
        "loss_limit": st.session_state.loss_input,
        "tickers": st.session_state.tickers,
        "logs": st.session_state.logs[-50:]
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# INITIAL BOOTSTRAP
if "init_done" not in st.session_state:
    saved = load_settings()
    # Use 'bot_toggle' as the key to match the widget
    st.session_state.bot_toggle = saved.get("run_bot", False) if saved else False
    st.session_state.goal_input = saved.get("profit_goal", 200.0) if saved else 200.0
    st.session_state.loss_input = saved.get("loss_limit", 100.0) if saved else 100.0
    st.session_state.tickers = saved.get("tickers", ["SPY", "AMZN"]) if saved else ["SPY", "AMZN"]
    st.session_state.logs = saved.get("logs", []) if saved else []
    st.session_state.init_done = True

# --- SIDEBAR UI ---
st.sidebar.header("🛡️ Bot Control")

# FIXED: We use the Session State value as the 'value' but do NOT update the key manually later
st.sidebar.toggle("Activate AI Bot", key="bot_toggle", on_change=save_settings)
st.sidebar.number_input("Profit Goal ($)", key="goal_input", on_change=save_settings)
st.sidebar.number_input("Max Loss ($)", key="loss_input", on_change=save_settings)

# (Other sidebar components like Watchlist Add/Remove here...)

# --- WORKER ---
def scan_ticker(symbol, run_bot_active):
    try:
        end = datetime.now() - timedelta(minutes=20)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=end-timedelta(days=730), end=end, feed=DataFeed.IEX)
        df = data_client.get_stock_bars(req).df.reset_index()
        df.ta.rsi(length=14, append=True); df.ta.bbands(length=20, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU'])]
        model = RandomForestClassifier(n_estimators=100).fit(df[cols][:-1], df['target'][:-1])
        prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        if run_bot_active and prob >= 0.90:
             # Trade Logic here...
             st.session_state.logs.append(f"🤖 AI BUY: {symbol} (AI: {prob*100:.1f}%)")
             save_settings()
        return {"symbol": symbol, "prob": prob, "price": df['close'].iloc[-1]}
    except: return None

# --- MAIN DASHBOARD ---
@st.fragment(run_every=30)
def trading_dashboard():
    # --- SAFE GLOBAL SYNC ---
    # We load the file but DO NOT update the widget keys directly
    # Instead, we check the file value and use it for logic ONLY
    saved_file_data = load_settings()

    # Logic Choice: File value OR Widget value? 
    # To avoid the error, we use the File value as the "Master Truth"
    current_run_bot = saved_file_data.get("run_bot", st.session_state.bot_toggle) if saved_file_data else st.session_state.bot_toggle

    try:
        acc = trading_client.get_account()
        st.columns(2)[0].metric("PORTFOLIO", f"${float(acc.equity):,.2f}")
        st.columns(2)[1].metric("PnL", f"${float(acc.equity)-float(acc.last_equity):.2f}")
    except: pass

    st.subheader("⚡ Signal Feed")
    # Parallel Scan using 'current_run_bot' (The master truth from file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = [f.result() for f in concurrent.futures.as_completed([ex.submit(scan_ticker, s, current_run_bot) for s in st.session_state.tickers])]

    # Table Rendering...
    for res in [r for r in results if r]:
        st.write(f"**{res['symbol']}** | Conf: {res['prob']*100:.1f}% | Bot Status: {'ACTIVE' if current_run_bot else 'OFF'}")

    st.subheader("📜 Log")
    st.code("\n".join(st.session_state.logs[-10:]))

trading_dashboard()
