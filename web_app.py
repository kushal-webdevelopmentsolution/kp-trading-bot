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
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

st.set_page_config(page_title="AI Trader Global Sync", layout="wide")

# --- PERSISTENCE ENGINE ---
SETTINGS_FILE = "settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except: return None
    return None

def save_settings():
    """Captures the current key-based state and writes to disk."""
    settings = {
        "tickers": st.session_state.get("tickers", ["SPY", "AMZN"]),
        "run_bot": st.session_state.get("bot_toggle", False),
        "profit_goal": st.session_state.get("goal_input", 200.0),
        "loss_limit": st.session_state.get("loss_input", 100.0),
        "order_mode": st.session_state.get("mode_radio", "USD"),
        "order_val": st.session_state.get("val_input", 100.0),
        "logs": st.session_state.get("logs", [])[-50:]
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# INITIAL BOOTSTRAP (Runs once per session start)
if "init_done" not in st.session_state:
    saved = load_settings()
    if saved:
        # Restore widget keys directly
        st.session_state.tickers = saved.get("tickers", ["SPY", "AMZN"])
        st.session_state.logs = saved.get("logs", [])
        st.session_state.bot_toggle = saved.get("run_bot", False)
        st.session_state.goal_input = saved.get("profit_goal", 200.0)
        st.session_state.loss_input = saved.get("loss_limit", 100.0)
        st.session_state.mode_radio = saved.get("order_mode", "USD")
        st.session_state.val_input = saved.get("order_val", 100.0)
    else:
        st.session_state.tickers = ["SPY", "AMZN", "NVDA", "GOOGL"]
        st.session_state.logs = []
    st.session_state.init_done = True

# --- SIDEBAR UI ---
st.sidebar.header("📂 Watchlist")
new_ticker = st.sidebar.text_input("Add Symbol").upper().strip()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker)
        save_settings()
        st.rerun()

st.session_state.tickers = st.sidebar.multiselect(
    "Active Watchlist", 
    options=st.session_state.tickers, 
    default=st.session_state.tickers,
    on_change=save_settings
)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Bot Control")
# Widgets use 'key' only. Value is managed by session_state sync.
st.sidebar.toggle("Activate AI Bot", key="bot_toggle", on_change=save_settings)
st.sidebar.number_input("Profit Goal ($)", key="goal_input", on_change=save_settings)
st.sidebar.number_input("Max Loss ($)", key="loss_input", on_change=save_settings)

st.sidebar.markdown("---")
st.sidebar.radio("Sizing:", ["Shares", "USD"], key="mode_radio", on_change=save_settings)
st.sidebar.number_input("Value", key="val_input", on_change=save_settings)

if st.sidebar.button("🚨 EMERGENCY SELL ALL", type="primary", use_container_width=True):
    trading_client.close_all_positions(cancel_orders=True)
    st.session_state.logs.append(f"🚨 LIQUIDATED at {datetime.now().strftime('%H:%M:%S')}")
    save_settings()
    st.rerun()

# --- AI SCANNER WORKER ---
def scan_ticker(symbol, run_bot, mode, val):
    try:
        end = datetime.now() - timedelta(minutes=20)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=end-timedelta(days=730), end=end, feed=DataFeed.IEX)
        df = data_client.get_stock_bars(req).df.reset_index()
        if df.empty: return {"symbol": symbol, "error": "No Data"}

        df.ta.rsi(length=14, append=True); df.ta.bbands(length=20, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU'])]
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[cols][:-1], df['target'][:-1])
        prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(val if mode == "Shares" else round(val / price, 2))

        if run_bot and prob >= 0.90:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET, 
                take_profit=TakeProfitRequest(limit_price=round(price*1.04, 2)), 
                stop_loss=StopLossRequest(stop_price=round(price*0.98, 2))
            ))
            st.session_state.logs.append(f"🤖 {symbol} BUY (AI: {prob*100:.0f}%)")
            save_settings()
        return {"symbol": symbol, "price": price, "prob": prob, "df": df, "qty": qty}
    except Exception as e: return {"symbol": symbol, "error": str(e)}

# --- MAIN DASHBOARD ---
st.title("🚀 AI Pro: Automate Trading")

@st.fragment(run_every=30)
def trading_dashboard():
    # --- GLOBAL SYNC STEP ---
    # Refresh settings from file every 30 seconds to catch changes from other devices
    saved = load_settings()
    if saved:
        st.session_state.bot_toggle = saved.get("run_bot", st.session_state.bot_toggle)
        st.session_state.goal_input = saved.get("profit_goal", st.session_state.goal_input)
        st.session_state.loss_input = saved.get("loss_limit", st.session_state.loss_input)
        st.session_state.tickers = saved.get("tickers", st.session_state.tickers)
        st.session_state.logs = saved.get("logs", st.session_state.logs)

    try:
        acc = trading_client.get_account()
        pnl = float(acc.equity) - float(acc.last_equity)
        st.columns(2).metric("PORTFOLIO", f"${float(acc.equity):,.2f}")
        st.columns(2).metric("DAILY PnL", f"${pnl:.2f}", delta=f"{pnl:.2f}")

        # Risk guard
        bot_allowed = st.session_state.bot_toggle
        if pnl >= st.session_state.goal_input or pnl <= -abs(st.session_state.loss_input):
            bot_allowed = False
    except: bot_allowed = False

    st.subheader("⚡ Signal Feed")
    # Aligned UI columns
    h1, h2, h3, h4 = st.columns([1, 1, 2, 1])
    h1.caption("SYMBOL"); h2.caption("PRICE"); h3.caption("AI CONFIDENCE"); h4.caption("ACTION")

    if st.session_state.tickers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            results = [f.result() for f in concurrent.futures.as_completed([ex.submit(scan_ticker, s, bot_allowed, st.session_state.mode_radio, st.session_state.val_input) for s in st.session_state.tickers])]

        best_ticker = None; max_conf = 0
        for res in results:
            if "error" in res: continue
            if res['prob'] > max_conf: max_conf, best_ticker = res['prob'], (res['symbol'], res['df'])

            r1, r2, r3, r4 = st.columns([1, 1, 2, 1])
            r1.write(f"**{res['symbol']}**")
            r2.write(f"${res['price']:.2f}")
            r3.progress(res['prob'], text=f"{res['prob']*100:.0f}%")
            if r4.button(f"Buy {res['qty']}", key=f"b_{res['symbol']}"):
                st.session_state.logs.append(f"👤 MAN BUY: {res['symbol']}")
                save_settings()

        if best_ticker:
            st.markdown("---")
            st.subheader(f"📈 Chart: {best_ticker[0]}")
            st.line_chart(best_ticker[1][['close']].tail(50))

    st.subheader("📜 Persistent Log")
    st.code("\n".join(st.session_state.logs[-15:]))
    st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")

trading_dashboard()
