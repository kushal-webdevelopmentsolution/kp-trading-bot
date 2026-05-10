import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, time, concurrent.futures
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except KeyError:
    st.error("API Keys missing! Add them to Streamlit Secrets dashboard.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

st.set_page_config(page_title="AI Multi-Threaded Trader", layout="wide")

# Default ticker list
if "tickers" not in st.session_state:
    st.session_state.tickers = ["SPY","AMZN", "NVDA", "TSLA"]
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- CORE AI ENGINES ---
def scan_ticker(symbol, run_bot_active, order_mode, order_val):
    try:
        end_time = datetime.now() - timedelta(minutes=15)
        req_data = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=datetime.now()-timedelta(days=1000), end=end_time, feed=DataFeed.IEX
        )
        df = data_client.get_stock_bars(req_data).df.reset_index()
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.mfi(length=14, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()

        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU', 'MFI'])]
        clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2)], voting='soft')
        model.fit(df[cols][:-1], df['target'][:-1])

        up_prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(order_val if order_mode == "Shares" else round(order_val / price, 2))

        if run_bot_active and up_prob >= 0.90:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(price*1.04, 2)),
                stop_loss=StopLossRequest(stop_price=round(price*0.98, 2))
            ))
            st.session_state.logs.append(f"{datetime.now().strftime('%H:%M')} | 🤖 AUTO BUY: {symbol} @ {price:.2f}")

        return {"symbol": symbol, "price": price, "prob": up_prob, "df": df, "qty": qty}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# --- SIDEBAR ---
st.sidebar.header("🛡️ Risk Control")
run_bot = st.sidebar.toggle("Activate 90% Auto-Bot")
profit_goal = st.sidebar.number_input("Profit Goal ($)", value=200.0)
max_loss = st.sidebar.number_input("Max Loss ($)", value=100.0)

st.sidebar.markdown("---")
order_mode = st.sidebar.radio("Sizing:", ["Shares", "USD Value"])
order_val = st.sidebar.number_input("Amount:", min_value=0.01, value=1.0 if order_mode=="Shares" else 100.0)

st.sidebar.markdown("---")
st.sidebar.header("📂 Watchlist")
new_ticker = st.sidebar.text_input("Add Ticker").upper().strip()
if st.sidebar.button("➕ Add") and new_ticker:
    if new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker)
        st.rerun()

updated_list = st.sidebar.multiselect("Current Watchlist", options=st.session_state.tickers, default=st.session_state.tickers)
if updated_list != st.session_state.tickers:
    st.session_state.tickers = updated_list
    st.rerun()

# Feature 3: Download Logs
if st.sidebar.button("📊 Export Trade Log"):
    df_logs = pd.DataFrame(st.session_state.logs, columns=["Action"])
    st.sidebar.download_button("Download CSV", df_logs.to_csv(index=False), "trades.csv", "text/csv")

# --- MAIN DASHBOARD ---
clock = trading_client.get_clock()
status_color = "#00ff00" if clock.is_open else "#ff0000"
st.title("🚀 AI Multi-Threaded Command Center")

@st.fragment(run_every=30)
def trading_dashboard():
    col1, col2 = st.columns([2.5, 1])
    with col1:
        try:
            acc = trading_client.get_account()
            equity = float(acc.equity)
            pnl = float(acc.equity) - float(acc.last_equity)

            # Feature 2: Color Coding Metrics
            m1, m2 = st.columns(2)
            m1.metric("PORTFOLIO VALUE", f"${equity:,.2f}")
            m2.metric("DAILY PnL", f"${pnl:.2f}", delta=f"{pnl:.2f}", delta_color="normal")

            st.subheader("⚡ Signal Feed & AI Confidence")
            h1, h2, h3, h4 = st.columns([1, 1, 2, 1])
            h1.caption("SYMBOL"); h2.caption("PRICE"); h3.caption("AI CONFIDENCE GAUGE"); h4.caption("ACTION")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(scan_ticker, s, run_bot, order_mode, order_val) for s in st.session_state.tickers]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            best_ticker = None
            max_conf = 0
            for res in results:
                if "error" in res: continue
                s, price, prob, qty = res['symbol'], res['price'], res['prob'], res['qty']
                if prob > max_conf: max_conf, best_ticker = prob, (s, res['df'])

                r1, r2, r3, r4 = st.columns([1, 1, 2, 1])
                r1.write(f"**{s}**")
                r2.write(f"${price:.2f}")

                # Feature 4: Signal Strength Label
                strength = "ULTRA" if prob >= 0.95 else "HIGH" if prob >= 0.90 else "NEUTRAL"
                r3.progress(min(max(prob, 0.0), 1.0), text=f"{strength} ({prob*100:.1f}%)")

                if r4.button(f"Buy {qty}", key=f"buy_{s}"):
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=s, qty=qty, side=OrderSide.BUY, 
                        time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET, 
                        take_profit=TakeProfitRequest(limit_price=round(price*1.04,2)), 
                        stop_loss=StopLossRequest(stop_price=round(price*0.98,2))))
                    st.session_state.logs.append(f"{datetime.now().strftime('%H:%M')} | 👤 MAN BUY: {s}")

            if best_ticker:
                st.markdown("---")
                st.subheader(f"📊 Top Analysis: {best_ticker[0]}")
                st.line_chart(best_ticker[1][['close']].tail(50))
        except Exception as e: st.error(f"UI Error: {e}")

    with col2:
        st.subheader("📜 Activity Log")
        log_text = "\n".join(st.session_state.logs[-15:])
        st.text_area("Log", value=log_text, height=600, label_visibility="collapsed")
        st.caption(f"Status: {'OPEN' if clock.is_open else 'CLOSED'} | Refreshed: {datetime.now().strftime('%H:%M:%S')}")

trading_dashboard()
