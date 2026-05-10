import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, time, concurrent.futures
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed # Added for IEX fix
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except KeyError:
    st.error("API Keys missing! Add them to Streamlit Secrets dashboard.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

st.set_page_config(page_title="AI Multi-Threaded Trader", layout="wide")

# Default ticker list
DEFAULT_TICKERS = ["SPY","AMZN"]

# Initialize Session States
if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- MULTI-THREADED SCANNER FUNCTIONS ---
def scan_ticker(symbol, run_bot_active, order_mode, order_val):
    """Worker function optimized for Cloud resource limits and IEX feed."""
    try:
        # Buffer to avoid SIP restriction (15-minute delay)
        end_time = datetime.now() - timedelta(minutes=15)

        # 1. Strategy Optimization (Lightweight) - Explicitly use IEX feed
        req_opt = StockBarsRequest(
            symbol_or_symbols=symbol, 
            timeframe=TimeFrame.Day,
            start=datetime.now()-timedelta(days=365), 
            end=end_time,
            feed=DataFeed.IEX 
        )
        df_opt = data_client.get_stock_bars(req_opt).df.reset_index()
        best_rsi = 14
        max_strength = 0
        for r in [10, 14, 20]:
            rsi = ta.rsi(df_opt['close'], length=r)
            if rsi is not None:
                strength = rsi.diff().abs().mean()
                if strength > max_strength:
                    max_strength, best_rsi = strength, r

        # 2. Fetch Data & AI Prediction - Explicitly use IEX feed
        req_data = StockBarsRequest(
            symbol_or_symbols=symbol, 
            timeframe=TimeFrame.Day,
            start=datetime.now()-timedelta(days=1000), 
            end=end_time,
            feed=DataFeed.IEX
        )
        df = data_client.get_stock_bars(req_data).df.reset_index()
        df.ta.rsi(length=best_rsi, append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.mfi(length=14, append=True)
        df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()

        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU', 'MFI', 'RVOL'])]

        clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2)], voting='soft')
        model.fit(df[cols][:-1], df['target'][:-1])

        prob_array = model.predict_proba(df[cols].tail(1))
        up_prob = float(prob_array[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(order_val if order_mode == "Shares" else round(order_val / price, 2))

        if run_bot_active and up_prob >= 0.90:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(price*1.04, 2)),
                stop_loss=StopLossRequest(stop_price=round(price*0.98, 2))
            ))
            st.session_state.logs.append(f"🤖 AUTO BUY: {symbol} @ {price:.2f}")

        return {"symbol": symbol, "price": price, "prob": up_prob, "rsi": best_rsi, "df": df, "qty": qty}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# --- SIDEBAR COMMANDS ---
st.sidebar.header("🛡️ Risk Control")
run_bot = st.sidebar.toggle("Activate 90% Auto-Bot")
profit_goal = st.sidebar.number_input("Profit Goal ($)", value=200.0)
max_loss = st.sidebar.number_input("Max Loss ($)", value=100.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 Order Settings")
order_mode = st.sidebar.radio("Sizing By:", ["Shares", "USD Value"])
order_val = st.sidebar.number_input("Value", min_value=0.01, value=1.0 if order_mode=="Shares" else 100.0)

st.sidebar.markdown("---")
st.sidebar.header("📂 Watchlist Manager")
new_ticker = st.sidebar.text_input("Add Ticker", key="ticker_input").upper().strip()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker)
        st.rerun()

updated_list = st.sidebar.multiselect("Current Wishlist", 
                                     options=st.session_state.tickers, 
                                     default=st.session_state.tickers)
if updated_list != st.session_state.tickers:
    st.session_state.tickers = updated_list
    st.rerun()

if st.sidebar.button("🔴 MANUAL SELL ALL"):
    trading_client.close_all_positions(cancel_orders=True)
    st.sidebar.warning("Liquidated!")

# --- MAIN DASHBOARD ---
clock = trading_client.get_clock()
status_color = "#00ff00" if clock.is_open else "#ff0000"
st.title("🚀 AI Multi-Threaded Command Center")
st.markdown(f"<div style='border-left: 5px solid {status_color}; padding-left:10px;'><b>Market Status:</b> {'OPEN' if clock.is_open else 'CLOSED'} (15min Delayed Feed)</div>", unsafe_allow_html=True)

@st.fragment(run_every=30)
def trading_dashboard():
    col1, col2 = st.columns([2.5, 1])
    with col1:
        try:
            acc = trading_client.get_account()
            m1, m2 = st.columns(2)
            m1.metric("PORTFOLIO VALUE", f"${float(acc.equity):,.2f}")
            m2.metric("DAILY PnL", f"${float(acc.equity)-float(acc.last_equity):.2f}")

            st.subheader("⚡ Signal Feed & AI Confidence")
            h1, h2, h3, h4 = st.columns([1, 1, 2, 1])
            h1.caption("SYMBOL"); h2.caption("PRICE"); h3.caption("AI CONFIDENCE GAUGE"); h4.caption("ACTION")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(scan_ticker, s, run_bot, order_mode, order_val) for s in st.session_state.tickers]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            best_ticker = None
            max_conf = 0
            for res in results:
                if "error" in res:
                    st.error(f"Error {res['symbol']}: {res['error']}")
                    continue

                s, price, prob, qty = res['symbol'], res['price'], res['prob'], res['qty']
                if prob > max_conf: 
                    max_conf, best_ticker = prob, (s, res['df'])

                r1, r2, r3, r4 = st.columns([1, 1, 2, 1])
                r1.write(f"**{s}**")
                r2.write(f"${price:.2f}")
                r3.progress(min(max(prob, 0.0), 1.0), text=f"{prob*100:.1f}%")
                if r4.button(f"Buy {qty}", key=f"buy_{s}"):
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=s, qty=qty, side=OrderSide.BUY, 
                        time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET, 
                        take_profit=TakeProfitRequest(limit_price=round(price*1.04,2)), 
                        stop_loss=StopLossRequest(stop_price=round(price*0.98,2))))
                    st.session_state.logs.append(f"👤 MAN BUY: {s} @ {price:.2f}")

            if best_ticker:
                st.markdown("---")
                st.subheader(f"📊 Analysis: {best_ticker[0]}")
                st.line_chart(best_ticker[1][['close']].tail(50))
        except Exception as e: st.error(f"UI Error: {e}")

    with col2:
        st.subheader("📜 Activity Log")
        log_text = "\n".join(st.session_state.logs[-15:])
        st.text_area("Log", value=log_text, height=600, label_visibility="collapsed")
        st.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")

trading_dashboard()
