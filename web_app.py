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

# --- CLOUD CONFIG ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

st.set_page_config(page_title="AI Trader Cloud Pro", layout="wide")

# --- CORE TRADING FUNCTION ---
def execute_trade(symbol, qty, price, is_bot=False):
    try:
        order_data = MarketOrderRequest(
            symbol=symbol, 
            qty=qty, 
            side=OrderSide.BUY, 
            time_in_force=TimeInForce.GTC, 
            order_class=OrderClass.BRACKET, 
            take_profit=TakeProfitRequest(limit_price=round(price * 1.04, 2)), 
            stop_loss=StopLossRequest(stop_price=round(price * 0.98, 2))
        )
        trading_client.submit_order(order_data)
        prefix = "🤖 AI" if is_bot else "👤 MANUAL"
        st.toast(f"{prefix} Order Placed: {symbol}", icon="✅")
        return f"{prefix} BUY: {symbol} @ {price:.2f} ({qty} units)"
    except Exception as e:
        st.error(f"Trade Failed: {str(e)}")
        return f"❌ ERROR {symbol}: {str(e)}"

# --- PERSISTENCE ---
SETTINGS_FILE = "settings.json"

def load_settings():
    defaults = {"tickers": ["SPY", "AMZN", "NVDA", "GOOGL"], "run_bot": False, 
                "profit_goal": 200.0, "loss_limit": 100.0, "order_mode": "USD", "order_val": 100.0}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return {**defaults, **json.load(f)}
        except: return defaults
    return defaults

def save_settings():
    settings = {k: st.session_state[k] for k in ["tickers", "run_bot", "profit_goal", "loss_limit", "order_mode", "order_val"]}
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

if "init_loaded" not in st.session_state:
    saved = load_settings()
    for k, v in saved.items(): st.session_state[k] = v
    st.session_state.init_loaded = True
    st.session_state.logs = []

# --- SIDEBAR ---
st.sidebar.header("🛡️ Bot Control")
st.session_state.run_bot = st.sidebar.toggle("Activate AI Bot", value=st.session_state.run_bot, on_change=save_settings)
st.session_state.order_mode = st.sidebar.radio("Sizing:", ["Shares", "USD"], index=0 if st.session_state.order_mode == "Shares" else 1, on_change=save_settings)
st.session_state.order_val = st.sidebar.number_input("Value", value=st.session_state.order_val, on_change=save_settings)

if st.sidebar.button("🚨 EMERGENCY SELL ALL", type="primary"):
    trading_client.close_all_positions(cancel_orders=True)
    st.rerun()

# --- ANALYSIS ENGINE ---
def scan_ticker(symbol, mode, val):
    try:
        end = datetime.now() - timedelta(minutes=20)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=end-timedelta(days=365), end=end, feed=DataFeed.IEX)
        df = data_client.get_stock_bars(req).df.reset_index()
        if df.empty: return {"symbol": symbol, "error": "No Data"}

        df.ta.rsi(length=14, append=True); df.ta.bbands(length=20, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU'])]
        model = RandomForestClassifier(n_estimators=50).fit(df[cols][:-1], df['target'][:-1])
        prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(val if mode == "Shares" else round(val / price, 2))

        return {"symbol": symbol, "price": price, "prob": prob, "df": df, "qty": qty}
    except Exception as e: return {"symbol": symbol, "error": str(e)}

# --- UI DASHBOARD ---
st.title("🚀 AI Multi-Threaded Cloud Terminal")

@st.fragment(run_every=60)
def trading_dashboard():
    try:
        acc = trading_client.get_account()
        st.metric("PORTFOLIO EQUITY", f"${float(acc.equity):,.2f}")
    except: pass

    st.subheader("⚡ Signal Feed")
    if st.session_state.tickers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(lambda s: scan_ticker(s, st.session_state.order_mode, st.session_state.order_val), st.session_state.tickers))

        for res in results:
            if "error" in res: continue

            col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
            col1.write(f"**{res['symbol']}**")
            col2.write(f"${res['price']:.2f}")
            col3.progress(res['prob'], text=f"AI: {res['prob']*100:.0f}%")

            # --- EXECUTION LOGIC ---
            # 1. Bot Execution (Automated)
            if st.session_state.run_bot and res['prob'] >= 0.90:
                log_msg = execute_trade(res['symbol'], res['qty'], res['price'], is_bot=True)
                st.session_state.logs.append(log_msg)

            # 2. Manual Execution (Button)
            if col4.button(f"Buy {res['qty']}", key=f"btn_{res['symbol']}"):
                log_msg = execute_trade(res['symbol'], res['qty'], res['price'], is_bot=False)
                st.session_state.logs.append(log_msg)

    st.subheader("📜 Recent Activity")
    for log in reversed(st.session_state.logs[-5:]):
        st.write(log)

trading_dashboard()
