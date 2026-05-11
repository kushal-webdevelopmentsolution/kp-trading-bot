import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, json, time
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
st.set_page_config(page_title="AI Alpha Terminal: Elite Pro", layout="wide")

# --- 2. INITIALIZATION ---
SETTINGS_FILE = "settings.json"

def init_session_state():
    defaults = {
        "tickers": ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA"], 
        "run_bot": False, 
        "order_mode": "USD", 
        "order_val": 100.0, 
        "logs": [], 
        "trailing_pct": 0.02, 
        "profit_target": 0.05, 
        "ai_threshold": 0.85,
        "vix_threshold": 25.0,
        "lock_profit_pct": 0.03 # New: Start trailing once 3% profit is reached
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()

def save_settings():
    keys = ["tickers", "run_bot", "order_mode", "order_val", "trailing_pct", "profit_target", "ai_threshold", "vix_threshold", "lock_profit_pct"]
    with open(SETTINGS_FILE, "w") as f: json.dump({k: st.session_state[k] for k in keys}, f)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🤖 AI Bot Control")
    st.session_state.run_bot = st.toggle("Activate AI Bot", value=st.session_state.run_bot, on_change=save_settings)
    st.session_state.ai_threshold = st.slider("Min Confidence Trigger", 0.70, 0.98, st.session_state.ai_threshold, on_change=save_settings)

    st.divider()
    st.header("🛡️ Profit-Trailing Stop")
    st.session_state.lock_profit_pct = st.slider("Start Trailing after % Gain", 0.01, 0.10, st.session_state.lock_profit_pct, on_change=save_settings)
    st.caption("Locks in profit by following price upward once target is met.")

    st.divider()
    st.header("⚠️ Panic Filter (VIX)")
    st.session_state.vix_threshold = st.number_input("Halt Trading if VIX >", value=st.session_state.vix_threshold, on_change=save_settings)

    st.divider()
    st.header("📂 Portfolio Settings")
    st.session_state.tickers = st.multiselect("Symbols", options=st.session_state.tickers, default=st.session_state.tickers, on_change=save_settings)
    st.session_state.trailing_pct = st.slider("Max Stop Loss %", 0.01, 0.10, st.session_state.trailing_pct, on_change=save_settings)

    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        st.session_state.run_bot = False
        save_settings(); st.rerun()

# --- 4. OPTIMIZED AI ENGINE ---
def get_vix_proxy():
    try:
        req = StockBarsRequest(symbol_or_symbols="VXX", timeframe=TimeFrame.Minute, start=datetime.now()-timedelta(days=1), feed=DataFeed.IEX)
        bars = data_client.get_stock_bars(req).df
        return float(bars['close'].iloc[-1])
    except: return 20.0

def get_optimized_ai_prediction(df):
    try:
        df = df.copy()
        df.ta.rsi(length=14, append=True); df.ta.macd(append=True); df.ta.adx(append=True); df.ta.atr(append=True)
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.002).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'MACD', 'ADX', 'ATR'])]
        model = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])
        probs = [float(p[1]) for p in model.predict_proba(df[features].tail(10))]
        return probs[-1], probs
    except: return 0.5, [0.5]*10

# --- 5. DASHBOARD ---
st.title("🚀 AI Alpha Terminal: Profit-Trailing Elite")
tab1, tab2 = st.tabs(["⚡ Live Dashboard", "📊 Analytics"])

with tab1:
    @st.fragment(run_every=60)
    def live_ui():
        # Market Panic Filter
        vix_current = get_vix_proxy()
        panic_mode = vix_current > st.session_state.vix_threshold
        v_col1, v_col2 = st.columns([1, 3])
        v_col1.metric("VIX (VXX)", f"{vix_current:.2f}", delta="PANIC" if panic_mode else "CALM", delta_color="inverse")
        if panic_mode: v_col2.error("🚫 PANIC DETECTED: Bot Entries Suspended.")

        # POSITIONS & TRAILING STOP LOGIC
        st.subheader("📊 Open Positions")
        pos = trading_client.get_all_positions()
        if pos:
            p_cols = st.columns([1, 0.8, 1, 1, 1, 1, 1, 0.5])
            heads = ["SYMBOL", "SHARES", "AMOUNT", "P/L %", "STOP TYPE", "STOP PRICE", "DIST", "EXIT"]
            for col, head in zip(p_cols, heads): col.caption(head)

            for p in pos:
                qty, mkt_val, curr_price = float(p.qty), float(p.market_value), float(p.current_price)
                avg_entry, pnl_pct = float(p.avg_entry_price), float(p.unrealized_plpc) * 100

                # Logic: Dynamic Profit-Trailing Stop
                # If current profit > lock_profit_pct, trail from current price. Else, trail from entry.
                if pnl_pct >= (st.session_state.lock_profit_pct * 100):
                    stop_price = curr_price * (1 - st.session_state.trailing_pct)
                    stop_type = "🔥 Trailing"
                else:
                    stop_price = avg_entry * (1 - st.session_state.trailing_pct)
                    stop_type = "🧊 Initial"

                dist_pct = ((curr_price - stop_price) / curr_price) * 100

                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 0.8, 1, 1, 1, 1, 1, 0.5])
                c1.write(f"**{p.symbol}**"); c2.write(f"{qty}"); c3.write(f"${mkt_val:,.0f}")
                c4.write(f":{'green' if pnl_pct >= 0 else 'red'}[{pnl_pct:.2f}%]")
                c5.write(stop_type)
                c6.write(f"${stop_price:.2f}")
                c7.write(f":{'orange' if dist_pct < 1.0 else 'gray'}[{dist_pct:.1f}%]")

                if c8.button("✖", key=f"cl_{p.symbol}"): trading_client.close_position(p.symbol); st.rerun()

                # Bot Exit (Triggers if price falls below the current dynamic stop)
                if st.session_state.run_bot and curr_price <= stop_price:
                    trading_client.close_position(p.symbol)
                    st.session_state.logs.append(f"🤖 Bot {stop_type} Exit: {p.symbol}")
        else: st.info("No active positions.")

        # SIGNALS
        st.subheader("⚡ AI Intelligence Feed")
        all_data = {}
        for s in st.session_state.tickers:
            try:
                df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=120), feed=DataFeed.IEX)).df.reset_index()
                ai_conf, conf_hist = get_optimized_ai_prediction(df)
                price = float(df['close'].iloc[-1])
                all_data[s] = df.set_index('timestamp')['close']

                s1, s2, s3, s4, s5 = st.columns([1, 1, 1.5, 2, 1])
                s1.write(f"**{s}**"); s2.write(f"${price:.2f}")
                s3.progress(ai_conf, text=f"AI: {ai_conf*100:.0f}%")
                with s4: st.line_chart(conf_hist, height=50, use_container_width=True)

                if s5.button("Buy", key=f"buy_{s}"):
                    qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                    trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))

                if st.session_state.run_bot and not panic_mode and ai_conf >= st.session_state.ai_threshold:
                    qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                    trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                    st.session_state.logs.append(f"🤖 AI Entry: {s}")
            except: continue

        # CORRELATION
        if len(all_data) > 1:
            st.divider(); st.subheader("🕸️ Sector Exposure Matrix")
            st.dataframe(pd.concat(all_data.values(), axis=1, keys=all_data.keys()).corr().style.background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

    live_ui()

st.subheader("📜 Recent Activity")
st.code("\n".join(st.session_state.logs[-5:]))
