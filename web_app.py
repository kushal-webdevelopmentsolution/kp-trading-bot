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
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
st.set_page_config(page_title="AI Trader Pro + Backtest", layout="wide")

# --- 2. INITIALIZATION ---
SETTINGS_FILE = "settings.json"

def init_session_state():
    defaults = {
        "tickers": ["SPY", "AMZN", "NVDA", "GOOGL"], 
        "run_bot": False, 
        "order_mode": "USD", 
        "order_val": 100.0,
        "logs": [],
        "trailing_pct": 0.02,
        "profit_target": 0.05
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()

def save_settings():
    keys = ["tickers", "run_bot", "order_mode", "order_val", "trailing_pct", "profit_target"]
    with open(SETTINGS_FILE, "w") as f:
        json.dump({k: st.session_state[k] for k in keys}, f)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Bot Control")
    st.session_state.run_bot = st.toggle("Activate AI Bot", value=st.session_state.run_bot, on_change=save_settings)
    st.session_state.tickers = st.multiselect("Watchlist", st.session_state.tickers, default=st.session_state.tickers, on_change=save_settings)
    st.divider()
    st.session_state.order_mode = st.radio("Sizing", ["USD", "Shares"], on_change=save_settings)
    st.session_state.order_val = st.number_input("Value", value=st.session_state.order_val, on_change=save_settings)
    st.divider()
    st.session_state.profit_target = st.slider("Take Profit %", 0.01, 0.15, st.session_state.profit_target, on_change=save_settings)
    st.session_state.trailing_pct = st.slider("Stop Loss %", 0.01, 0.10, st.session_state.trailing_pct, on_change=save_settings)

# --- 4. BACKTESTING ENGINE ---
def run_backtest(symbol, rsi_buy=35, target=0.05, stop=0.02):
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=365), feed=DataFeed.IEX)
    df = data_client.get_stock_bars(req).df.reset_index()
    df['RSI'] = ta.rsi(df['close'], length=14)

    cash = 10000
    shares = 0
    history = []

    for i in range(len(df)):
        price = df['close'].iloc[i]
        rsi = df['RSI'].iloc[i]

        # Sell Logic
        if shares > 0:
            pnl = (price - entry_price) / entry_price
            if pnl >= target or pnl <= -stop:
                cash = shares * price
                shares = 0

        # Buy Logic
        elif rsi < rsi_buy:
            shares = cash / price
            cash = 0
            entry_price = price

        history.append(cash + (shares * price))

    df['Strategy_Equity'] = history
    df['Buy_Hold'] = (df['close'] / df['close'].iloc[0]) * 10000
    return df

# --- 5. UI TABS ---
tab1, tab2 = st.tabs(["🚀 Live Terminal", "📊 Backtest Strategy"])

with tab1:
    @st.fragment(run_every=60)
    def live_dashboard():
        try:
            hist = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period="1W", timeframe="1H"))
            st.area_chart(pd.DataFrame(hist.equity, index=pd.to_datetime(hist.timestamp, unit='s')), height=150)
        except: pass

        st.subheader("📊 Positions")
        pos = trading_client.get_all_positions()
        if pos:
            cols = st.columns(4)
            for i, p in enumerate(pos):
                with cols[i%4]:
                    st.metric(p.symbol, f"{p.qty}", f"{float(p.unrealized_plpc)*100:.2f}%")
                    if st.button(f"Close {p.symbol}"): trading_client.close_position(p.symbol); st.rerun()

        st.subheader("⚡ Signals")
        for s in st.session_state.tickers:
            try:
                req = StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=30), feed=DataFeed.IEX)
                df = data_client.get_stock_bars(req).df.reset_index()
                df['RSI'] = ta.rsi(df['close'], length=14)
                price, rsi = df['close'].iloc[-1], df['RSI'].iloc[-1]

                c1, c2, c3, c4 = st.columns([1,1,1,1])
                c1.write(f"**{s}**")
                c2.write(f"${price:.2f}")
                c3.write(f"RSI: {rsi:.1f}")
                if c4.button(f"Buy", key=f"b_{s}"):
                    qty = round(st.session_state.order_val / price, 2) if st.session_state.order_mode == "USD" else st.session_state.order_val
                    trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                    st.toast(f"Bought {s}")
            except: continue
    live_dashboard()

with tab2:
    st.header("Strategy Backtester (1-Year)")
    bt_symbol = st.selectbox("Select Symbol", st.session_state.tickers)
    if st.button("Run Backtest"):
        res = run_backtest(bt_symbol, target=st.session_state.profit_target, stop=st.session_state.trailing_pct)
        st.line_chart(res.set_index('timestamp')[['Strategy_Equity', 'Buy_Hold']])

        final_return = ((res['Strategy_Equity'].iloc[-1] - 10000) / 10000) * 100
        bh_return = ((res['Buy_Hold'].iloc[-1] - 10000) / 10000) * 100

        m1, m2 = st.columns(2)
        m1.metric("Strategy Return", f"{final_return:.2f}%")
        m2.metric("Buy & Hold Return", f"{bh_return:.2f}%")

st.subheader("📜 Activity Log")
st.code("\n".join(st.session_state.logs[-5:]))
