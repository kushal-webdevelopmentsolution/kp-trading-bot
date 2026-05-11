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
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce
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
st.set_page_config(page_title="AI Alpha Terminal: Sector Elite", layout="wide")

# --- 2. PERSISTENCE ENGINE ---
SETTINGS_FILE = "settings.json"
LOG_FILE = "trade_history.log"

def save_settings():
    keys = ["tickers", "run_bot", "order_mode", "order_val", "trailing_pct", 
            "profit_target", "ai_threshold", "vix_threshold", "lock_profit_pct", "daily_loss_limit"]
    settings_data = {k: st.session_state[k] for k in keys if k in st.session_state}
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings_data, f, indent=4)

def add_log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    if "logs" not in st.session_state: st.session_state.logs = []
    st.session_state.logs.append(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def update_val(key):
    save_settings()

def init_session_state():
    defaults = {"tickers": ["SPY", "QQQ", "NVDA", "AAPL"], "run_bot": False, "order_mode": "USD", 
                "order_val": 100.0, "trailing_pct": 0.02, "profit_target": 0.05, 
                "ai_threshold": 0.85, "vix_threshold": 25.0, "lock_profit_pct": 0.03,
                "daily_loss_limit": 500.0}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if "logs" not in st.session_state:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f: st.session_state.logs = f.read().splitlines()[-50:]
        else: st.session_state.logs = []

init_session_state()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🤖 AI Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=update_val, args=("run_bot",))
    st.slider("AI Confidence Threshold", 0.70, 0.98, key="ai_threshold", on_change=update_val, args=("ai_threshold",))

    st.divider()
    st.header("🛑 Circuit Breaker")
    st.number_input("Daily Loss Limit ($)", key="daily_loss_limit", on_change=update_val, args=("daily_loss_limit",))

    st.divider()
    st.header("🛡️ Strategy & Risk")
    st.slider("Profit Trailing Start %", 0.01, 0.10, key="lock_profit_pct", on_change=update_val, args=("lock_profit_pct",))
    st.slider("Stop Loss %", 0.01, 0.10, key="trailing_pct", on_change=update_val, args=("trailing_pct",))

    st.divider()
    st.header("📂 Config")
    st.multiselect("Watchlist", options=["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "GOOGL", "META", "AMZN"], key="tickers", on_change=update_val, args=("tickers",))
    st.radio("Size Mode", ["USD", "Shares"], key="order_mode", on_change=update_val, args=("order_mode",))
    st.number_input("Amount", key="order_val", on_change=update_val, args=("order_val",))

    if st.button("🚨 LIQUIDATE ALL", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        add_log("EMERGENCY LIQUIDATION TRIGGERED")
        st.session_state.run_bot = False
        save_settings()
        st.rerun()

# --- 4. ENGINES ---
def get_daily_pnl():
    try:
        acc = trading_client.get_account()
        return float(acc.equity) - float(acc.last_equity)
    except: return 0.0

def get_vix_proxy():
    try:
        req = StockBarsRequest(symbol_or_symbols="VXX", timeframe=TimeFrame.Minute, start=datetime.now()-timedelta(days=1), feed=DataFeed.IEX)
        return float(data_client.get_stock_bars(req).df['close'].iloc[-1])
    except: return 20.0

def get_ai_prediction(df):
    try:
        df = df.copy()
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.adx(append=True)
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.002).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'MACD', 'ADX'])]
        model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        probs = [float(p) for p in model.predict_proba(df[features].tail(10))[:, 1]]
        return probs[-1], probs
    except: return 0.5, [0.5]*10

# --- 5. DASHBOARD UI ---
st.title("🚀 AI Alpha Terminal")

@st.fragment(run_every=60)
def live_ui():
    daily_pnl = get_daily_pnl()
    vix = get_vix_proxy()
    panic = vix > st.session_state.vix_threshold
    breaker_tripped = daily_pnl <= -abs(st.session_state.daily_loss_limit)

    if breaker_tripped and st.session_state.run_bot:
        st.session_state.run_bot = False
        save_settings()
        add_log(f"🛑 BREAKER TRIPPED: Daily PnL ${daily_pnl:.2f}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Daily PnL", f"${daily_pnl:.2f}")
    m2.metric("VIX Proxy", f"{vix:.2f}", delta="PANIC" if panic else "OK", delta_color="inverse")
    if breaker_tripped: m3.error("🚨 BREAKER TRIPPED")
    elif panic: m3.warning("⚠️ VOLATILITY HALT")
    else: m3.success("🟢 BOT ACTIVE" if st.session_state.run_bot else "⚪ BOT STANDBY")

    st.subheader("📊 Positions")
    pos = trading_client.get_all_positions()
    if pos:
        cols = st.columns([1, 1, 1, 1, 1, 1, 0.5])
        for col, head in zip(cols, ["SYMBOL", "AMOUNT", "P/L %", "STOP TYPE", "STOP PRICE", "DIST", "EXIT"]): col.caption(head)
        for p in pos:
            qty, mkt_val, curr_price = float(p.qty), float(p.market_value), float(p.current_price)
            avg_entry, pnl_pct = float(p.avg_entry_price), float(p.unrealized_plpc) * 100

            is_trailing = pnl_pct >= (st.session_state.lock_profit_pct * 100)
            stop_price = curr_price * (1-st.session_state.trailing_pct) if is_trailing else avg_entry * (1-st.session_state.trailing_pct)
            dist_pct = ((curr_price - stop_price) / curr_price) * 100

            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 0.5])
            c1.write(f"**{p.symbol}**"); c2.write(f"${mkt_val:,.0f}"); c3.write(f"{pnl_pct:.2f}%")
            c4.write("🔥 Trail" if is_trailing else "🧊 Base"); c5.write(f"${stop_price:.2f}"); c6.write(f"{dist_pct:.1f}%")
            if c7.button("✖", key=f"cl_{p.symbol}"):
                trading_client.close_position(p.symbol); add_log(f"Manual Close: {p.symbol}"); st.rerun()

    st.subheader("⚡ AI Confidence & Signal Feed")
    prices_for_corr = {}
    for s in st.session_state.tickers:
        try:
            df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=100), feed=DataFeed.IEX)).df.reset_index()
            ai_conf, conf_hist = get_ai_prediction(df)
            price = float(df['close'].iloc[-1])
            prices_for_corr[s] = df.set_index('timestamp')['close']

            s1, s2, s3, s4, s5 = st.columns([1, 1, 1.5, 2, 1])
            s1.write(f"**{s}**"); s2.write(f"${price:.2f}")
            s3.progress(ai_conf, text=f"AI: {ai_conf*100:.0f}%")
            with s4: st.line_chart(conf_hist, height=50, use_container_width=True)

            if s5.button("Buy", key=f"b_{s}"):
                qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                add_log(f"Manual Buy: {s} @ {price}")

            if st.session_state.run_bot and not panic and not breaker_tripped and ai_conf >= st.session_state.ai_threshold:
                qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                add_log(f"🤖 AI Buy Triggered: {s} (Confidence: {ai_conf:.2f})")
        except: continue

    if len(prices_for_corr) > 1:
        st.divider()
        st.subheader("🕸️ Sector Correlation Matrix")
        corr_df = pd.concat(prices_for_corr.values(), axis=1, keys=prices_for_corr.keys()).corr()
        st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)
        st.caption("Lower correlation (red) suggests better diversification.")

    st.divider()
    st.subheader("📜 Trade History")
    st.code("\n".join(st.session_state.logs[-15:]))

live_ui()
