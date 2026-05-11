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
st.set_page_config(page_title="AI Alpha Terminal Pro", layout="wide")

# --- 2. INITIALIZATION ---
SETTINGS_FILE = "settings.json"

def init_session_state():
    defaults = {"tickers": ["SPY", "AMZN", "NVDA", "GOOGL"], "run_bot": False, "order_mode": "USD", 
                "order_val": 100.0, "logs": [], "trailing_pct": 0.02, "profit_target": 0.05, "ai_threshold": 0.70}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: defaults.update(json.load(f))
        except: pass
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()

def save_settings():
    keys = ["tickers", "run_bot", "order_mode", "order_val", "trailing_pct", "profit_target", "ai_threshold"]
    with open(SETTINGS_FILE, "w") as f: json.dump({k: st.session_state[k] for k in keys}, f)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🤖 AI Bot Control")
    st.session_state.run_bot = st.toggle("Activate AI Bot", value=st.session_state.run_bot, on_change=save_settings)
    st.session_state.ai_threshold = st.slider("AI Confidence Trigger", 0.50, 0.95, st.session_state.ai_threshold, on_change=save_settings)
    st.divider()
    st.header("📂 Watchlist")
    st.session_state.tickers = st.multiselect("Symbols", options=st.session_state.tickers, default=st.session_state.tickers, on_change=save_settings)
    st.divider()
    st.header("⚙️ Order Settings")
    st.session_state.order_mode = st.radio("Sizing", ["USD", "Shares"], index=0 if st.session_state.order_mode=="USD" else 1, on_change=save_settings)
    st.session_state.order_val = st.number_input("Value", value=st.session_state.order_val, on_change=save_settings)
    st.divider()
    st.header("🛡️ Risk Management")
    st.session_state.profit_target = st.slider("Take Profit %", 0.01, 0.20, st.session_state.profit_target, on_change=save_settings)
    st.session_state.trailing_pct = st.slider("Stop Loss %", 0.01, 0.10, st.session_state.trailing_pct, on_change=save_settings)
    if st.button("🚨 SELL ALL & HALT", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        st.session_state.run_bot = False
        save_settings(); st.rerun()

# --- 4. ENGINES ---
def get_news_sentiment(symbol):
    try:
        news = news_client.get_news(NewsRequest(symbols=symbol, limit=5))
        scores = [TextBlob(n.headline).sentiment.polarity for n in news.news]
        return sum(scores) / len(scores) if scores else 0.0
    except: return 0.0

def get_ai_prediction_history(df):
    """Returns the latest confidence and the last 10 confidence scores for a sparkline"""
    try:
        df = df.copy()
        df.ta.rsi(length=14, append=True); df.ta.bbands(length=20, append=True)
        df.ta.vwap(append=True); df.ta.adx(length=14, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        features = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU', 'VWAP', 'ADX'])]

        # Train on data up to 10 days ago to simulate walk-forward confidence
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(df[features][:-10], df['target'][:-10])

        # Get probabilities for the last 10 days
        probs = [float(p[1]) for p in model.predict_proba(df[features].tail(10))]
        return probs[-1], probs
    except: return 0.5, [0.5]*10

# --- 5. DASHBOARD ---
st.title("🚀 AI Multi-Threaded Cloud Terminal")
tab1, tab2 = st.tabs(["⚡ Live Dashboard", "📊 Strategy Backtest"])

with tab1:
    @st.fragment(run_every=60)
    def live_ui():
        # A. Portfolio Chart
        try:
            hist = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period="1D", timeframe="15Min"))
            st.area_chart(pd.DataFrame(hist.equity, index=pd.to_datetime(hist.timestamp, unit='s')), height=150)
        except: pass

        # B. POSITIONS
        st.subheader("📊 Open Positions")
        pos = trading_client.get_all_positions()
        if pos:
            cols = st.columns([1, 0.8, 1.2, 1, 1, 1.2, 0.5])
            for col, head in zip(cols, ["SYMBOL", "SHARES", "AMOUNT", "P/L %", "STOP PRICE", "DIST TO STOP", "EXIT"]): col.caption(head)
            for p in pos:
                qty, mkt_val, curr_price = float(p.qty), float(p.market_value), float(p.current_price)
                avg_entry, pnl_pct = float(p.avg_entry_price), float(p.unrealized_plpc) * 100
                stop_price = avg_entry * (1 - st.session_state.trailing_pct)
                dist_pct = ((curr_price - stop_price) / curr_price) * 100
                c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 0.8, 1.2, 1, 1, 1.2, 0.5])
                c1.write(f"**{p.symbol}**"); c2.write(f"{qty}"); c3.write(f"${mkt_val:,.2f}")
                c4.write(f":{'green' if pnl_pct >= 0 else 'red'}[{pnl_pct:.2f}%]"); c5.write(f"${stop_price:.2f}")
                c6.write(f":{'orange' if dist_pct < 1.0 else 'gray'}[{dist_pct:.1f}%]"); 
                if c7.button("✖", key=f"cl_{p.symbol}"): trading_client.close_position(p.symbol); st.rerun()
                if st.session_state.run_bot and (pnl_pct >= st.session_state.profit_target*100 or pnl_pct <= -st.session_state.trailing_pct*100):
                    trading_client.close_position(p.symbol); st.session_state.logs.append(f"🤖 Bot Exit: {p.symbol}")
        else: st.info("No active positions.")

        # C. REFINED SIGNAL FEED WITH CONFIDENCE HISTORY
        st.subheader("⚡ Integrated Signals")
        for s in st.session_state.tickers:
            try:
                df = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime.now()-timedelta(days=120), feed=DataFeed.IEX)).df.reset_index()
                ai_conf, conf_hist = get_ai_prediction_history(df)
                price, sent = float(df['close'].iloc[-1]), get_news_sentiment(s)

                s1, s2, s3, s4, s5, s6 = st.columns([1, 1, 1.5, 1.5, 1.5, 1])
                s1.write(f"**{s}**"); s2.write(f"${price:.2f}")
                s3.progress(ai_conf, text=f"AI: {ai_conf*100:.0f}%")
                with s4: st.caption("Conf. Trend"); st.line_chart(conf_hist, height=60, use_container_width=True)
                s5.progress((sent+1)/2, text=f"Mood: {sent:.2f}")

                if s6.button("Buy", key=f"buy_{s}"):
                    qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                    trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                    st.toast(f"Manual Buy: {s}")

                if st.session_state.run_bot and ai_conf >= st.session_state.ai_threshold and sent > 0:
                    qty = round(st.session_state.order_val/price, 2) if st.session_state.order_mode=="USD" else st.session_state.order_val
                    trading_client.submit_order(MarketOrderRequest(symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                    st.session_state.logs.append(f"🤖 Bot Entry: {s} @ {price}")
            except: continue

        st.subheader("📜 Activity Log")
        st.code("\n".join(st.session_state.logs[-5:]))
    live_ui()

with tab2:
    st.info("Backtest results available here.")
