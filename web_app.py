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
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from sklearn.ensemble import RandomForestClassifier

# --- 1. CONFIG & CLIENTS ---
st.set_page_config(page_title="AI Alpha Terminal Pro", layout="wide")

try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- 2. PERSISTENCE & STATE ---
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

init_session_state()

# --- 3. LOGIC ENGINES ---
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

def get_ai_prediction(df):
    try:
        df = df.copy()
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.adx(append=True)
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.002).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'MACD', 'ADX'])]
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        probs = [float(p) for p in model.predict_proba(df[features].tail(10))[:, 1]]
        return probs[-1], probs
    except: return 0.5, [0.5]*10

# --- 4. SIDEBAR CONTROLS ---
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
    st.header("🏁 Daily Targets")
    st.number_input("Profit Goal ($)", key="global_profit_goal", on_change=save_settings)
    st.number_input("Loss Limit ($)", key="daily_loss_limit", on_change=save_settings)

    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        add_log("EMERGENCY SHUTDOWN: All positions closed.")
        st.session_state.run_bot = False; save_settings(); st.rerun()

# --- 5. MAIN DASHBOARD ---
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
    held_symbols = {p.symbol for p in pos}

    if pos:
        for p in pos:
            mkt_val, pnl_pct = float(p.market_value), float(p.unrealized_plpc) * 100
            c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
            c1.write(f"**{p.symbol}**")
            c2.write(f"${mkt_val:,.0f}")
            c3.write(f"{pnl_pct:.2f}%")
            if c4.button("✖", key=f"cl_{p.symbol}"):
                trading_client.close_position(p.symbol)
                add_log(f"Manual Close: {p.symbol}"); st.rerun()
    else:
        st.info("No active positions.")

    # AI Signal Feed
    st.subheader("⚡ AI Signals")
    for s in st.session_state.tickers:
        try:
            # Using 5-minute bars for better intraday resolution
            df = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=s, 
                timeframe=TimeFrame.Minute, 
                start=datetime.now()-timedelta(days=5), 
                feed=DataFeed.IEX
            )).df.reset_index()

            ai_conf, conf_hist = get_ai_prediction(df)
            price = float(df['close'].iloc[-1])

            s1, s2, s3, s4, s5 = st.columns([1, 1, 1.5, 2, 1])
            s1.write(f"**{s}**")
            s2.write(f"${price:.2f}")
            s3.progress(ai_conf)
            with s4: st.line_chart(conf_hist, height=60, use_container_width=True)

            def submit_order(is_bot=False):
                qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                if not market_open and st.session_state.allow_ext_hours:
                    req = LimitOrderRequest(symbol=s, qty=qty, limit_price=price, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, extended_hours=True)
                else:
                    req = MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)

                trading_client.submit_order(req)
                add_log(f"{'🤖 Bot' if is_bot else '👤 Manual'} Buy: {s} @ {price}")

            if s5.button("Buy", key=f"b_{s}"): 
                submit_order()

            # Bot Auto-execution logic
            if active_now and ai_conf >= st.session_state.ai_threshold and s not in held_symbols:
                submit_order(is_bot=True)
                held_symbols.add(s) # Prevent duplicate orders in the same fragment cycle

        except Exception as e:
            continue

    # Trade History
    st.divider()
    with st.expander("📜 Activity Logs & Export", expanded=False):
        if st.session_state.logs:
            history_data = [{"Time": l.split(" | ")[0], "Activity": l.split(" | ")[1]} for l in reversed(st.session_state.logs) if " | " in l]
            st.table(history_data[:15])
            csv = pd.DataFrame(history_data).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv, file_name="trade_history.csv", mime="text/csv")

    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Data: IEX (Real-time)")

live_ui()
