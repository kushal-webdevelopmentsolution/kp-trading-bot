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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- CLOUD CONFIG ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

st.set_page_config(page_title="AI Trader Cloud", layout="wide")

# Persistent State
if "tickers" not in st.session_state:
    st.session_state.tickers = ["SPY", "AMZN", "NVDA", "TSLA"]
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- CLOUD-SAFE RISK ENGINE (WITH CACHING) ---
@st.cache_data(ttl=300) # Only refresh data every 5 minutes to save CPU
def get_cloud_risk_matrix(tickers):
    try:
        combined_data = {}
        end_time = datetime.now() - timedelta(minutes=20)
        start_date = end_time - timedelta(days=60)

        for symbol in tickers:
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                   start=start_date, end=end_time, feed=DataFeed.IEX)
            df = data_client.get_stock_bars(req).df.reset_index()
            if not df.empty:
                df['timestamp'] = df['timestamp'].dt.date
                combined_data[symbol] = df.set_index('timestamp')['close']

        if len(combined_data) > 1:
            return pd.concat(combined_data.values(), axis=1, keys=combined_data.keys(), join='inner').corr()
    except:
        pass
    return None

# --- WORKER FUNCTION ---
def scan_ticker(symbol, run_bot, order_mode, order_val):
    try:
        end_time = datetime.now() - timedelta(minutes=20)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=datetime.now()-timedelta(days=730), end=end_time, feed=DataFeed.IEX)
        df = data_client.get_stock_bars(req).df.reset_index()
        df.ta.rsi(length=14, append=True); df.ta.bbands(length=20, append=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU'])]
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[cols][:-1], df['target'][:-1])
        prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(order_val if order_mode == "Shares" else round(order_val / price, 2))

        if run_bot and prob >= 0.90:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=round(price*1.04, 2)),
                stop_loss=StopLossRequest(stop_price=round(price*0.98, 2))
            ))
            st.session_state.logs.append(f"🤖 {symbol} BUY @ {price:.2f}")

        return {"symbol": symbol, "price": price, "prob": prob, "df": df, "qty": qty}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# --- UI ---
st.title("🚀 AI Multi-Threaded Cloud Terminal")

st.sidebar.header("🛡️ Control")
run_bot = st.sidebar.toggle("Activate AI Bot")
order_mode = st.sidebar.radio("Sizing:", ["Shares", "USD"])
order_val = st.sidebar.number_input("Value:", min_value=0.01, value=1.0 if order_mode=="Shares" else 100.0)

@st.fragment(run_every=30)
def trading_dashboard():
    # 1. Metrics
    acc = trading_client.get_account()
    m1, m2 = st.columns(2)
    m1.metric("PORTFOLIO", f"${float(acc.equity):,.2f}")
    m2.metric("PnL", f"${float(acc.equity)-float(acc.last_equity):.2f}")

    # 2. Scanning
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(scan_ticker, s, run_bot, order_mode, order_val) for s in st.session_state.tickers]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # 3. Render Table
    st.subheader("⚡ Signal Feed")
    best_ticker = None
    max_conf = 0
    for res in results:
        if "error" in res: continue
        if res['prob'] > max_conf: max_conf, best_ticker = res['prob'], (res['symbol'], res['df'])

        r = st.columns([1, 1, 2, 1])
        r[0].write(f"**{res['symbol']}**")
        r[1].write(f"${res['price']:.2f}")
        r[2].progress(res['prob'], text=f"AI: {res['prob']*100:.0f}%")
        if r[3].button(f"Buy", key=f"b_{res['symbol']}"):
            st.session_state.logs.append(f"👤 Manual {res['symbol']}")

    # 4. Visuals (Explicitly separated for Cloud stability)
    st.markdown("---")
    left, right = st.columns([1.5, 1])

    with left:
        if best_ticker:
            st.subheader(f"📈 Chart: {best_ticker[0]}")
            st.line_chart(best_ticker[1][['close']].tail(50))

    with right:
        st.subheader("🔗 Risk Matrix")
        risk_df = get_cloud_risk_matrix(st.session_state.tickers)
        if risk_df is not None:
            # Use axis=None and simple styling for Cloud compatibility
            st.dataframe(risk_df.style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)
        else:
            st.info("Gathering historical correlation data...")

    st.subheader("📜 Log")
    st.code("\n".join(st.session_state.logs[-10:]))

trading_dashboard()
