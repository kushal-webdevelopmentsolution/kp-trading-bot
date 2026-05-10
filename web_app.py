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

# --- PERSISTENCE ENGINE (All Settings + Logs) ---
SETTINGS_FILE = "settings.json"

def load_settings():
    defaults = {
        "tickers": ["SPY", "AMZN", "NVDA", "GOOGL"], 
        "run_bot": False, 
        "profit_goal": 200.0, 
        "loss_limit": 100.0, 
        "order_mode": "USD", 
        "order_val": 100.0,
        "logs": []
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return {**defaults, **json.load(f)}
        except: return defaults
    return defaults

def save_settings():
    # Capture current session state and write to file
    settings = {
        "tickers": st.session_state.tickers,
        "run_bot": st.session_state.run_bot,
        "profit_goal": st.session_state.profit_goal,
        "loss_limit": st.session_state.loss_limit,
        "order_mode": st.session_state.order_mode,
        "order_val": st.session_state.order_val,
        "logs": st.session_state.logs[-50:] # Keep last 50 entries
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# Initialize Session State from JSON on first load
if "init_loaded" not in st.session_state:
    saved = load_settings()
    for k, v in saved.items():
        st.session_state[k] = v
    st.session_state.init_loaded = True

# --- SIDEBAR ---
st.sidebar.header("📂 Watchlist")
new_ticker = st.sidebar.text_input("Add Symbol").upper().strip()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker)
        save_settings()
        st.rerun()

# Removal logic via multiselect
current_list = st.sidebar.multiselect("Active Watchlist", 
                                     options=st.session_state.tickers, 
                                     default=st.session_state.tickers)
if current_list != st.session_state.tickers:
    st.session_state.tickers = current_list
    save_settings()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Bot Control")

# All inputs trigger save_settings on change
st.session_state.run_bot = st.sidebar.toggle("Activate AI Bot", value=st.session_state.run_bot, on_change=save_settings)
st.session_state.profit_goal = st.sidebar.number_input("Profit Goal ($)", value=st.session_state.profit_goal, on_change=save_settings)
st.session_state.loss_limit = st.sidebar.number_input("Max Loss ($)", value=st.session_state.loss_limit, on_change=save_settings)

st.sidebar.markdown("---")
st.session_state.order_mode = st.sidebar.radio("Sizing:", ["Shares", "USD"], index=0 if st.session_state.order_mode == "Shares" else 1, on_change=save_settings)
st.session_state.order_val = st.sidebar.number_input("Value", value=st.session_state.order_val, on_change=save_settings)

if st.sidebar.button("🚨 EMERGENCY SELL ALL", type="primary", use_container_width=True):
    trading_client.close_all_positions(cancel_orders=True)
    st.session_state.logs.append(f"🚨 EMERGENCY: Liquidation triggered at {datetime.now().strftime('%H:%M:%S')}")
    save_settings()
    st.rerun()

# --- AI WORKER ---
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
        model = RandomForestClassifier(n_estimators=100).fit(df[cols][:-1], df['target'][:-1])
        prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(val if mode == "Shares" else round(val / price, 2))

        if run_bot and prob >= 0.90:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET, 
                take_profit=TakeProfitRequest(limit_price=round(price*1.04, 2)), 
                stop_loss=StopLossRequest(stop_price=round(price*0.98, 2))
            ))
            st.session_state.logs.append(f"🤖 AUTO BUY: {symbol} @ {price:.2f} (AI: {prob*100:.0f}%)")
            save_settings() # Save log entry instantly
        return {"symbol": symbol, "price": price, "prob": prob, "df": df, "qty": qty}
    except Exception as e: return {"symbol": symbol, "error": str(e)}

# --- UI DASHBOARD ---
st.title("🚀 AI Multi-Threaded Cloud Terminal")

@st.fragment(run_every=30)
def trading_dashboard():
    try:
        acc = trading_client.get_account()
        pnl = float(acc.equity) - float(acc.last_equity)
        m1, m2 = st.columns(2)
        m1.metric("PORTFOLIO", f"${float(acc.equity):,.2f}")
        m2.metric("DAILY PnL", f"${pnl:.2f}", delta=f"{pnl:.2f}")
        run_bot_active = False if (pnl >= st.session_state.profit_goal or pnl <= -abs(st.session_state.loss_limit)) else st.session_state.run_bot
    except: run_bot_active = False

    st.subheader("⚡ Signal Feed")
    h1, h2, h3, h4 = st.columns([1, 1, 2, 1])
    h1.caption("SYMBOL"); h2.caption("PRICE"); h3.caption("AI CONFIDENCE GAUGE"); h4.caption("ACTION")

    if st.session_state.tickers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            results = [f.result() for f in concurrent.futures.as_completed([ex.submit(scan_ticker, s, run_bot_active, st.session_state.order_mode, st.session_state.order_val) for s in st.session_state.tickers])]

        best_ticker = None; max_conf = 0
        for res in results:
            if "error" in res: continue
            if res['prob'] > max_conf: max_conf, best_ticker = res['prob'], (res['symbol'], res['df'])

            r1, r2, r3, r4 = st.columns([1, 1, 2, 1])
            r1.write(f"**{res['symbol']}**")
            r2.write(f"${res['price']:.2f}")
            r3.progress(res['prob'], text=f"{res['prob']*100:.1f}%")
            if r4.button(f"Buy {res['qty']}", key=f"b_{res['symbol']}"):
                st.session_state.logs.append(f"👤 MAN BUY: {res['symbol']} @ {res['price']:.2f}")
                save_settings()

        st.markdown("---")
        l, r = st.columns([1.5, 1])
        if best_ticker:
            with l: st.subheader(f"📈 Chart: {best_ticker[0]}"); st.line_chart(best_ticker[1][['close']].tail(50))
            with r:
                st.subheader("🔗 Risk Matrix")
                data = {res['symbol']: res['df'].set_index('timestamp')['close'] for res in results if "error" not in res}
                if len(data) > 1: st.dataframe(pd.concat(data.values(), axis=1, keys=data.keys(), join='inner').corr().style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)

    st.subheader("📜 Persistent Activity Log")
    st.code("\n".join(st.session_state.logs[-15:]))

trading_dashboard()
