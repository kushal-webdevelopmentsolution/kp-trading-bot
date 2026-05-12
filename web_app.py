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
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
st.set_page_config(page_title="AI Alpha Terminal Pro", layout="wide")

# --- 2. PERSISTENCE ENGINE ---
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
                "order_val": 1000.0, "trailing_pct": 0.02, "profit_target": 0.05, 
                "ai_threshold": 0.85, "vix_threshold": 25.0, "lock_profit_pct": 0.05,
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

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🛒 Order Configuration")
    # Toggle between USD (Dollar) and Shares (Stock)
    # This updates st.session_state["order_mode"] automatically
    st.selectbox("Order Mode", options=["USD", "Shares"], key="order_mode", on_change=save_settings)

    # Dynamic Label and Value based on selection
    if st.session_state.order_mode == "USD":
        st.number_input("Order Amount ($)", min_value=1.0, step=10.0, key="order_val", on_change=save_settings)
    else:
        st.number_input("Number of Shares", min_value=1.0, step=1.0, key="order_val", on_change=save_settings)
    st.divider()    
    st.header("🤖 Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=save_settings)
    st.toggle("Allow Extended Hours", key="allow_ext_hours", on_change=save_settings)
    st.slider("AI Trigger Threshold", 0.50, 0.98, key="ai_threshold", on_change=save_settings)

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

    st.divider()
    st.header("🛡️ Strategy")
    st.slider("Trailing Start %", 0.01, 0.50, key="lock_profit_pct", on_change=save_settings)
    st.slider("Stop Loss %", 0.01, 0.50, key="trailing_pct", on_change=save_settings)

    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True):
        trading_client.close_all_positions(cancel_orders=True)
        add_log("EMERGENCY SHUTDOWN: All positions closed.")
        st.session_state.run_bot = False; save_settings(); st.rerun()

# --- 4. ENGINES ---
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

if "live_brains" not in st.session_state:
    st.session_state.live_brains = {}

def get_ai_prediction(df, symbol):
    # Standard Fallbacks
    neutral_conf = 0.5
    neutral_hist = [0.5] * 10
    neutral_map = {}

    try:
        # DATA VALIDATION
        if df is None or len(df) < 60:
            return neutral_conf, neutral_hist, neutral_map

        df = df.copy()
        # Clean and calculate technical suite
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.adx(append=True)
        df.ta.bbands(append=True); df.ta.vwap(append=True); df.ta.atr(append=True)

        # Normalized Volume-Price Trend (Z-Score)
        vpt_raw = ta.vpt(df['close'], df['volume'])
        df['vpt'] = (vpt_raw - vpt_raw.rolling(30).mean()) / (vpt_raw.rolling(30).std() + 1e-9)
        df['vpt_ema'] = ta.ema(df['vpt'], length=10)

        # Momentum vs Trend Filter
        df['self_rs'] = df['close'] / (ta.ema(df['close'], length=50) + 1e-9)

        # Precision Target (0.15% gain in next bar)
        df['target'] = (df['close'].shift(-1) > df['close'] * 1.0015).astype(int)
        df = df.dropna()

        features = [c for c in df.columns if any(x in c for x in ['RSI', 'MACD', 'ADX', 'BBP', 'VWAP', 'vpt', 'self_rs', 'ATR'])]

        # --- SMART ENSEMBLE TRAINING ---
        now = time.time()
        brain_data = st.session_state.live_brains.get(symbol, {"time": 0})

        # Training Logic (Triggered every 1 hour)
        if (now - brain_data["time"]) > 3600:
            # We use st.status here so you see the activity in the UI
            with st.status(f"🧠 AI STUDYING: {symbol} (30D Patterns)", expanded=False) as status:
                st.write("Fetching historical deep-data...")
                # Training on 30 Days of Minute Data
                train_bars = data_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, 
                    start=datetime.now()-timedelta(days=30), feed=DataFeed.IEX
                )).df.reset_index()

                st.write("Calculating multi-strategy confluence...")
                train_bars.ta.rsi(append=True); train_bars.ta.macd(append=True)
                train_bars.ta.adx(append=True); train_bars.ta.bbands(append=True)
                train_bars.ta.vwap(append=True); train_bars.ta.atr(append=True)
                v_raw = ta.vpt(train_bars['close'], train_bars['volume'])
                train_bars['vpt'] = (v_raw - v_raw.rolling(30).mean()) / (v_raw.rolling(30).std() + 1e-9)
                train_bars['self_rs'] = train_bars['close'] / (ta.ema(train_bars['close'], length=50) + 1e-9)
                train_bars['target'] = (train_df['close'].shift(-1) > train_df['close'] * 1.0015).astype(int)
                train_df = train_bars.dropna()

                st.write("Optimizing Ensemble (RF + XGBoost)...")
                m_rf = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
                m_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=6, tree_method='hist', n_jobs=-1)

                X, y = train_df[features], train_df['target']
                m_rf.fit(X, y); m_xgb.fit(X, y)

                st.session_state.live_brains[symbol] = {
                    "models": (m_rf, m_xgb), 
                    "time": now,
                    "features": features,
                    "importance": dict(zip(features, m_rf.feature_importances_))
                }
                status.update(label=f"✅ {symbol} Brain Optimized!", state="complete")

        # --- INSTANT INFERENCE ---
        stored = st.session_state.live_brains[symbol]
        m_rf, m_xgb = stored["models"]

        # Predict on fresh data
        last_data = df[features].tail(10)
        p_rf = m_rf.predict_proba(last_data)[:, 1]
        p_xgb = m_xgb.predict_proba(last_data)[:, 1]
        tech_probs = (p_rf + p_xgb) / 2

        # Precision Gates
        vpt_gate = 1.0 if df['vpt'].iloc[-1] > df['vpt_ema'].iloc[-1] else 0.5
        vwap_gate = 1.0 if df['close'].iloc[-1] > df['VWAP_D'].iloc[-1] else 0.0

        # 80% Technical Model + 20% Hard Gates
        final_conf = (tech_probs[-1] * 0.8) + (vpt_gate * 0.1) + (vwap_gate * 0.1)

        return float(final_conf), [float(p) for p in tech_probs], stored["importance"]

    except Exception as e:
        return neutral_conf, neutral_hist, neutral_map

# Helper for execution (Supports USD/Shares toggle, Extended Hours, and Duplicate Protection)
def execute_trade(s, price, ai_conf, is_bot=False):
    try:
        # --- 1. DUPLICATE PROTECTION: Check Positions and Pending Orders ---
        # Check if we already have a position
        pos = trading_client.get_all_positions()
        if any(p.symbol == s for p in pos):
            return # Exit silently if position exists

        # Check if an order is already waiting to be filled
        order_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[s])
        open_orders = trading_client.get_orders(filter=order_filter)
        if open_orders:
            return # Exit silently if order is pending

        # --- 2. MODE LOGIC: Calculate QTY based on USD or Shares ---
        if st.session_state.order_mode == "USD":
            qty = int(st.session_state.order_val // price)
        else:
            qty = int(st.session_state.order_val)

        if qty < 1:
            st.error(f"Order value too low for {s}")
            return

        clock = trading_client.get_clock()

        # --- 3. MARKET HOURS EXECUTION ---
        if clock.is_open:
            # REGULAR MARKET HOURS
            trading_client.submit_order(MarketOrderRequest(
                symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            ))

            # Trailing Stop (Lifts automatically with price)
            trading_client.submit_order(TrailingStopOrderRequest(
                symbol=s, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                trail_percent=round(float(st.session_state.trailing_pct * 100), 2)
            ))
        else:
            # EXTENDED HOURS (Limit order required)
            trading_client.submit_order(LimitOrderRequest(
                symbol=s, qty=qty, limit_price=price, side=OrderSide.BUY, 
                time_in_force=TimeInForce.DAY, extended_hours=True
            ))

        # --- 4. LOGGING & NOTIFICATION ---
        msg = f"{'🤖 Bot' if is_bot else '👤 Manual'} Entry: {s} @ {price} | Conf: {ai_conf:.1%} | Mode: {st.session_state.order_mode}"
        add_log(msg)
        st.toast(msg, icon="🚀")

    except Exception as e:
        st.error(f"Trade Failed: {e}")



# --- 5. DASHBOARD UI ---
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
    clock = trading_client.get_clock()
    held_symbols = {p.symbol for p in pos} # Essential for auto-execution check
    if pos:
        # Loop through positions to handle UI and Virtual Monitoring
        for p in pos:
            # --- START VIRTUAL MONITORING LOGIC ---
            # If market is CLOSED, we manually monitor the trailing stop-loss
            if not clock.is_open:
                current_price = float(p.current_price)
                avg_entry = float(p.avg_entry_price)

                # Calculate PnL percentage from entry
                current_pnl_pct = ((current_price - avg_entry) / avg_entry) * 100

                # If price drops below your trailing threshold (e.g., -2.0%)
                if current_pnl_pct <= -st.session_state.trailing_pct:
                    trading_client.submit_order(LimitOrderRequest(
                        symbol=p.symbol, 
                        qty=p.qty, 
                        limit_price=round(float(current_price), 2), 
                        side=OrderSide.SELL, 
                        time_in_force=TimeInForce.DAY, 
                        extended_hours=True
                    ))
                    add_log(f"🌙 After-Hours Stop Triggered: Sold {p.symbol} at {current_price}")
            # --- END VIRTUAL MONITORING LOGIC ---

            # --- START EXISTING UI CODE ---
            qty, mkt_val, pnl_pct = float(p.qty), float(p.market_value), float(p.unrealized_plpc) * 100
            c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
            c1.write(f"**{p.symbol}**")
            c2.write(f"${mkt_val:,.0f}")
            c3.write(f"{pnl_pct:.2f}%")

            if c4.button("✖", key=f"cl_{p.symbol}"):
                trading_client.close_position(p.symbol)
                add_log(f"Manual Close: {p.symbol}")
                st.rerun()
            # --- END EXISTING UI CODE ---
    else:
        st.info("No active positions.")

    try:
        order_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        all_open_orders = trading_client.get_orders(filter=order_filter)
        # Create a dictionary: { "SYMBOL": count }
        pending_counts = {}
        for o in all_open_orders:
            pending_counts[o.symbol] = pending_counts.get(o.symbol, 0) + 1
    except:
        pending_counts = {}

    # --- TOP OF live_ui() ---
    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        c1.markdown("#### 🤖 Bot Pulse")
        if st.session_state.run_bot:
            c2.caption(f"Last Full System Scan: {datetime.now().strftime('%H:%M:%S')}")
            # Optional: Show which models are currently "Warm" (in memory)
            warm_models = ", ".join(st.session_state.live_brains.keys())
            c2.write(f"**Warm Brains:** {warm_models if warm_models else 'None (Initializing...)'}")
        # AI Signal Feed
    st.subheader("⚡ AI Signals")
    for s in st.session_state.tickers:
        try:
            # Fetch data (ensure IEX feed for free tier or SIP for paid)
            df = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=s, 
                timeframe=TimeFrame.Day, 
                start=datetime.now()-timedelta(days=100), 
                feed=DataFeed.IEX
            )).df.reset_index()

            ai_conf, conf_hist, feat_map = get_ai_prediction(df,s)
            # Inside your ticker loop in live_ui()
            #ai_conf, conf_hist, feat_map = get_ai_prediction(df, s)
            price = float(df['close'].iloc[-1])

            # Layout columns
            s1, s2, s3, s4, s5 = st.columns([1, 1, 1.5, 2, 1])

            # s1.write(f"**{s}**")
            p_count = pending_counts.get(s, 0)
            if p_count > 0:
                s1.write(f"**{s}** :orange[({p_count} Pending)]")
            else:
                s1.write(f"**{s}**")
            s2.write(f"${price:.2f}")

            # 1. VISUAL CONFIDENCE BAR
            # Colors progress based on confidence level
            conf_label = f"Confidence: {ai_conf:.1%}"
            s3.progress(ai_conf, text=conf_label)

            # 2. CONFIDENCE TREND CHART
            # Shows if the AI is becoming more or less certain over the last 10 bars
            with s4:
                st.line_chart(conf_hist, height=60, use_container_width=True)




            # Order Logic
            # --- PRE-CHECK FOR MESSAGING ---
            is_held = s in held_symbols
            # Get pending count from the dictionary we created in the previous step
            is_pending = pending_counts.get(s, 0) > 0

            # Manual Buy Button
            if s5.button("Buy", key=f"b_{s}"):
                if is_held:
                    st.error(f"Cannot Buy: You already have a position in {s}")
                elif is_pending:
                    st.warning(f"Cannot Buy: An order for {s} is already pending.")
                else:
                    execute_trade(s, round(float(price), 2), ai_conf, is_bot=False)
                    st.toast(f"👤 Manual Order Sent: {s}", icon="📥")
                    time.sleep(1)
                    st.rerun()

            # --- 3. AUTO-EXECUTION CHECK (The 98% Accuracy Gate) ---
            if active_now and ai_conf >= st.session_state.ai_threshold:
                if not is_held and not is_pending:
                    execute_trade(s, round(float(price), 2), ai_conf, is_bot=True)
                    st.success(f"🤖 AI TRIGGERED: Buying {s} at {ai_conf:.1%} confidence")
                elif is_pending:
                    st.caption(f"⏳ Bot skipping {s}: Order already in flight.")
                else:
                    # Subtle indicator that the bot is watching but already owns it
                    st.caption(f"🛡️ Bot Watching {s} (Position Active)")

            # --- MANUAL BUY BUTTON ---
            # if s5.button("Buy", key=f"b_{s}"):
                # Pass the local variables 's', 'price', and 'ai_conf' into the function
                # execute_trade(s, price, ai_conf, is_bot=False)
                # time.sleep(1) # Brief pause for Alpaca sync
                # Insert your submit_order() call here
                # st.toast(f"Manual Buy Order Sent for {s}")
                # st.rerun()


            # --- 3. AUTO-EXECUTION CHECK (The 98% Accuracy Gate) ---
            # if active_now and ai_conf >= st.session_state.ai_threshold:
                # if s not in held_symbols:
                    # Pass the local variables 's', 'price', and 'ai_conf' into the function
                    # execute_trade(s, price, ai_conf, is_bot=True)
                    # st.success(f"🤖 AI TRIGGERED: Buying {s} at {ai_conf:.1%} confidence")
                # else:
                    # Subtle indicator that the bot is watching but already owns it
                    # st.caption(f"Bot Watching {s} (Position Active)")

        except Exception as e:
            st.error(f"Error loading {s}: {e}")
            continue

    # --- TRADE HISTORY TABLE ---
    st.divider()
    st.subheader("📜 Trade History")
    if st.session_state.logs:
        # Parsing log strings into a dataframe for visual table
        history_data = []
        for line in reversed(st.session_state.logs):
            if "|" in line:
                ts, msg = line.split(" | ", 1)
                history_data.append({"Time": ts, "Activity": msg})

        st.table(history_data[:15]) # Display last 15 actions

        # CSV Export
        csv = pd.DataFrame(history_data).to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Export History (CSV)", data=csv, file_name="trade_history.csv", mime="text/csv")
    st.divider()
    with st.expander(f"🔍 AI Logic Breakdown: {s}"):
        if feat_map:
            # Convert to DataFrame for easy charting
            feat_df = pd.DataFrame(list(feat_map.items()), columns=['Factor', 'Weight'])
            feat_df = feat_df.sort_values(by='Weight', ascending=True)

            # Display as a horizontal bar chart
            st.bar_chart(feat_df.set_index('Factor'), horizontal=True, height=200)

            # Quick Insight Text
            top_factor = feat_df.iloc[-1]['Factor']
            st.caption(f"Bot is currently prioritizing **{top_factor}** for {s} entries.")
        else:
            st.caption("Training data pending...")
live_ui()
