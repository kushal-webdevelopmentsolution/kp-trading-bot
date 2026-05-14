import streamlit as st
import pandas as pd
import pandas_ta as ta
import os, json, time
import datetime
from pytz import UTC
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
# Add Sort to your imports at the top of your file
from alpaca.trading.enums import OrderClass, QueryOrderStatus


# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY =  st.secrets["API_KEY"]
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

if "model_vault" not in st.session_state:
    st.session_state.model_vault = {}


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
                "order_val": 1000.0, "trailing_pct": 2.0, "profit_target": 5.0, 
                "ai_threshold": 0.85, "vix_threshold": 25.0, "lock_profit_pct": 5.0,
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
    st.slider("Trailing Start %", 1.0, 50.0, key="lock_profit_pct", on_change=save_settings)
    st.slider("Stop Loss %", 1.0, 50.0, key="trailing_pct", on_change=save_settings)

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

def get_ai_prediction(df, symbol):
    # --- 1. INITIALIZE DEFAULTS & SESSION STATE ---
    direction = "NEUTRAL"
    neutral_conf = 0.5
    neutral_hist = [0.5] * 10
    neutral_map = {}

    if "model_vault" not in st.session_state:
        st.session_state.model_vault = {}

    try:
        # --- 2. DATA VALIDATION ---
        if df is None or len(df) < 50:
            return direction, neutral_conf, neutral_hist, neutral_map

        df = df.copy()

        # Indicator Calculations
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.adx(append=True)
        df.ta.ema(length=20, append=True); df.ta.atr(append=True)
        df.ta.bbands(append=True); df.ta.mfi(append=True)
        df['VOL_SMA'] = ta.sma(df['volume'], length=20) 

        # Multi-Class Target (1=Long, 2=Short, 0=Neutral)
        df['target'] = 0
        df.loc[df['close'].shift(-1) > df['close'] * 1.0015, 'target'] = 1
        df.loc[df['close'].shift(-1) < df['close'] * 0.9985, 'target'] = 2

        df = df.ffill().dropna()

        feature_keywords = ['RSI', 'MACD', 'ADX', 'EMA', 'ATR', 'BBL', 'BBU', 'MFI', 'volume', 'VOL_SMA']
        features = [c for c in df.columns if any(x in c for x in feature_keywords)]

        # --- 3. TRAINING LOGIC ---
        now = time.time()
        brain = st.session_state.model_vault.get(symbol, {"time": 0})

        if (now - brain["time"]) > 3600:
            m_rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            m_xgb = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=6, 
                objective='multi:softprob', 
                num_class=3, 
                tree_method='hist', 
                n_jobs=-1
            )

            train_data = df.iloc[-500:] if len(df) > 500 else df
            # Ensure target classes 0, 1, 2 are present
            X, y = train_data[features].iloc[:-2], train_data['target'].iloc[:-2]

            m_rf.fit(X, y); m_xgb.fit(X, y)

            st.session_state.model_vault[symbol] = {
                "models": (m_rf, m_xgb), 
                "time": now,
                "importance": dict(zip(features, m_rf.feature_importances_))
            }

        # --- 4. INFERENCE ---
        active_brain = st.session_state.model_vault[symbol]
        rf_m, xgb_m = active_brain["models"]

        last_rows = df[features].tail(10)
        p_rf = rf_m.predict_proba(last_rows)
        p_xgb = xgb_m.predict_proba(last_rows)

        # Average probability (Class indices: 0=Neutral, 1=Long, 2=Short)
        avg_probs = (p_rf + p_xgb) / 2
        long_probs = avg_probs[:, 1]
        short_probs = avg_probs[:, 2]

        # --- 5. LOGIC GATES ---
        mfi_col = [c for c in df.columns if 'MFI' in c][0]
        bbu_col = [c for c in df.columns if 'BBU' in c][0]
        bbl_col = [c for c in df.columns if 'BBL' in c][0]

        l_price, l_mfi = df['close'].iloc[-1], df[mfi_col].iloc[-1]
        l_ema = df['EMA_20'].iloc[-1]
        l_bbu, l_bbl = df[bbu_col].iloc[-1], df[bbl_col].iloc[-1]

        final_conf = 0.0

        # Long Logic
        if long_probs[-1] > short_probs[-1] and long_probs[-1] > 0.4:
            trend_gate = 1.0 if (l_price > l_ema and l_mfi > 45) else 0.5
            final_conf = (long_probs[-1] * 0.7) + (trend_gate * 0.3)
            if l_price <= l_bbl: final_conf = min(1.0, final_conf + 0.1)
            if l_price >= l_bbu: final_conf = 0.0
            if final_conf > 0.5: direction = "LONG"

        # Short Logic
        elif short_probs[-1] > long_probs[-1] and short_probs[-1] > 0.4:
            trend_gate = 1.0 if (l_price < l_ema and l_mfi < 55) else 0.5
            final_conf = (short_probs[-1] * 0.7) + (trend_gate * 0.3)
            if l_price <= l_bbl: final_conf = 0.0
            if final_conf > 0.5: direction = "SHORT"

        return direction, float(final_conf), [float(p) for p in long_probs], active_brain["importance"]

    except Exception as e:
        st.error(f"AI Error: {e}")
        return "NEUTRAL", 0.5, [0.5]*10, {}

# Helper for execution (Supports USD/Shares toggle, Extended Hours, and Duplicate Protection)
def execute_trade(s, price, ai_conf, side=OrderSide.BUY, is_bot=False):
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

        # Determine exit side for the Trailing Stop
        # If entry is BUY, stop is SELL. If entry is SELL (Short), stop is BUY.
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # --- 3. MARKET HOURS EXECUTION ---
        if clock.is_open:
            # REGULAR MARKET HOURS
            trading_client.submit_order(MarketOrderRequest(
                symbol=s, qty=qty, side=side.value, time_in_force=TimeInForce.GTC
            ))

            # Trailing Stop (Lifts/Drops automatically with price)
            trading_client.submit_order(TrailingStopOrderRequest(
                symbol=s, qty=qty, side=exit_side.value, time_in_force=TimeInForce.GTC,
                trail_percent=st.session_state.trailing_pct
            ))
        else:
            # EXTENDED HOURS (Limit order required)
            trading_client.submit_order(LimitOrderRequest(
                symbol=s, qty=qty, limit_price=price, side=side.value, 
                time_in_force=TimeInForce.DAY, extended_hours=True
            ))

        # --- 4. LOGGING & NOTIFICATION ---
        action_type = "Long" if side == OrderSide.BUY else "Short"
        msg = f"{'🤖 Bot' if is_bot else '👤 Manual'} {action_type} Entry: {s} @ {price} | Conf: {ai_conf:.1%} | Mode: {st.session_state.order_mode}"
        add_log(msg)
        st.toast(msg, icon="🚀")

    except Exception as e:
        st.error(f"Trade Failed: {e}")

def get_pending_orders_df(trading_client):
    """Fetches open/pending orders and converts them into a formatted Pandas DataFrame."""
    try:
        # Request only 'open' (pending) orders from Alpaca
        filter_request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(filter=filter_request)

        if not open_orders:
            return pd.DataFrame() # Return empty dataframe if no orders pending

        # Parse relevant attributes into rows
        order_data = []
        for order in open_orders:
            order_data.append({
                "Symbol": order.symbol.upper(),
                "Side": order.side.value.upper(),
                "Qty": float(order.qty) if order.qty else 0,
                "Type": order.type.value,
                "Limit Price": float(order.limit_price) if order.limit_price else None,
                "Stop Price": float(order.stop_price) if order.stop_price else None,
                "Status": order.status.value,
                "Created At": order.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })

        return pd.DataFrame(order_data)
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return pd.DataFrame()

def get_trade_history_df(trading_client, limit=50, start_date=None, end_date=None):
    """Fetches closed/filled orders and converts them into a Pandas DataFrame."""
    try:
        # Convert date objects directly to ISO string formats accepted by Alpaca to avoid namespace bugs
        api_start = f"{start_date}T00:00:00Z" if start_date else None
        api_end = f"{end_date}T23:59:59Z" if end_date else None

        # Request closed orders with date boundary parameters
        filter_request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, 
            limit=limit,
            after=api_start,
            until=api_end
        )
        closed_orders = trading_client.get_orders(filter=filter_request)

        if not closed_orders:
            return pd.DataFrame() # Return empty if history is blank

        history_data = []
        for order in closed_orders:
            # We filter for 'filled' status to only show successful historical trades
            if order.status.value == "filled":
                history_data.append({
                    "Symbol": order.symbol.upper(),
                    "Side": order.side.value.upper(),
                    "Filled Qty": float(order.filled_qty) if order.filled_qty else 0,
                    "Avg Price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    "Total Value": round(float(order.filled_qty) * float(order.filled_avg_price), 2) if order.filled_qty and order.filled_avg_price else 0.0,
                    "Type": order.type.value,
                    "Execution Time": order.filled_at.strftime("%Y-%m-%d %H:%M:%S") if order.filled_at else "N/A"
                })

        return pd.DataFrame(history_data)
    except Exception as e:
        st.error(f"Error fetching trade history: {e}")
        return pd.DataFrame()




# --- 5. DASHBOARD UI ---
st.title("🚀 AI Alpha Terminal")

@st.fragment(run_every=60)
def live_ui():
    now_dt = datetime.now()
    last_refresh = now_dt.strftime("%I:%M:%S %p")

    # 2. Display Header
    t1, t2 = st.columns([1, 1])
    t1.caption(f"🕒 Last Refresh: **{last_refresh}**")
    prog_placeholder = t2.empty() # Placeholder for the countdown

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
                # Detect side - handle different API response formats (attribute vs dict)
                p_side = getattr(p, 'side', 'long').lower()

                # Calculate PnL percentage based on side
                if p_side == 'short':
                    # SHORT: Lose money if price rises (Entry - Current)
                    current_pnl_pct = ((avg_entry - current_price) / avg_entry) * 100
                else:
                    # LONG: Lose money if price falls (Current - Entry)
                    current_pnl_pct = ((current_price - avg_entry) / avg_entry) * 100

                # --- SELL ORDER PLACE CHECK & BYPASS ---
                is_sell_placed = False

                try:
                    # Check 1: Inspect Open Orders (Waiting to execute)
                    open_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[p.symbol])
                    open_orders = trading_client.get_orders(filter=open_filter)

                    for order in open_orders:
                        # Normalize side to string for comparison
                        ord_side = (order.side.value if hasattr(order.side, 'value') else str(order.side)).lower()
                        if ord_side == "sell":
                            is_sell_placed = True
                            add_log(f"⏭️ Bypass: An open SELL order already exists for {p.symbol}")
                            break

                    # Check 2: If no open sell order, inspect Closed/Filled Orders from today
                    if not is_sell_placed:
                        closed_filter = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[p.symbol], limit=10)
                        closed_orders = trading_client.get_orders(filter=closed_filter)

                        for order in closed_orders:
                            ord_side = (order.side.value if hasattr(order.side, 'value') else str(order.side)).lower()
                            # If a sell order was filled or is processing, mark as placed to avoid double execution
                            if ord_side == "sell" and order.status in ["new","filled", "partially_filled", "calculated"]:
                                is_sell_placed = True
                                add_log(f"⏭️ Bypass: A SELL order was already executed today for {p.symbol}")
                                break

                except Exception as check_err:
                    add_log(f"⚠️ Order history check failed for {p.symbol}: {check_err}")


                # If PnL drops below your trailing threshold (e.g., -2.0%)
                if current_pnl_pct <= -st.session_state.trailing_pct and not is_sell_placed:
                    # Determine Order Side: Shorts must BUY to cover; Longs must SELL
                    # Extracts the string value and normalizes it to lowercase for safety
                    p_side_str = (p_side.value if hasattr(p_side, 'value') else str(p_side)).strip().lower()

                    # Alpaca expects raw lowercase strings ('buy' or 'sell') for the side parameter
                    order_side = "buy" if p_side_str == 'short' else "sell"

                    trading_client.submit_order(LimitOrderRequest(
                        symbol=p.symbol, 
                        qty=abs(float(p.qty)), # abs() ensures positive qty for short covers
                        limit_price=round(float(current_price), 2), 
                        side=order_side, 
                        time_in_force=TimeInForce.DAY, 
                        extended_hours=True
                    ))
                    add_log(f"🌙 After-Hours Stop Triggered ({p_side_str.upper()}): {p.symbol} at {current_price}")
                # Optional: Log a message if the PnL breached but it was bypassed due to an existing order
                elif current_pnl_pct <= -st.session_state.trailing_pct and is_sell_placed:
                    add_log(f"⏭️ Bypassed execution for {p.symbol}: PnL threshold breached but a SELL order is already active.")
            # --- END VIRTUAL MONITORING LOGIC ---

            # --- START EXISTING UI CODE ---
            qty, mkt_val, pnl_pct = float(p.qty), float(p.market_value), float(p.unrealized_plpc) * 100

            # Dynamic UI based on Side
            current_side = getattr(p, 'side', 'long').lower()
            side_icon = "🔴" if current_side == 'short' else "🟢"
            pnl_color = "red" if pnl_pct < 0 else "green"

            c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
            c1.write(f"{side_icon} **{p.symbol}**")
            c2.write(f"${mkt_val:,.0f}")
            # Styled PnL for better visibility
            c3.markdown(f":{pnl_color}[{pnl_pct:.2f}%]")

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

    # AI Signal Feed
    st.subheader("⚡ AI Signals")
    for s in st.session_state.tickers:
        try:
            # Fetch data (ensure IEX feed for free tier or SIP for paid)
            df = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=s, 
                timeframe=TimeFrame.Day, 
                start=datetime.now()-timedelta(days=365), 
                feed=DataFeed.IEX
            )).df.reset_index()

            #ai_conf, conf_hist, feat_map = get_ai_prediction(df, s)
            ai_dir, ai_conf, conf_hist, feat_map = get_ai_prediction(df, s)
            price = float(df['close'].iloc[-1])

            # --- START 24H MARKET STATUS CHECK ---
            try:
                # Query Alpaca assets framework to extract trading attributes
                asset_info = data_client.get_asset(s)
                # Append green indicator badge if overnight_tradable is listed in assets features
                if asset_info and hasattr(asset_info, "attributes") and "overnight_tradable" in asset_info.attributes:
                    market_badge = " :green[[24H]] "
                else:
                    market_badge = " :gray[[Reg]] "
            except Exception:
                market_badge = " "  # Fallback gracefully if asset mapping fails
            # --- END 24H MARKET STATUS CHECK ---

            # Layout columns
            s1, s2, s3, s4, s5 = st.columns([1, 1, 1.5, 2, 1])

            # s1.write(f"**{s}**")
            p_count = pending_counts.get(s, 0)
            if p_count > 0:
                s1.write(f"**{s}** {market_badge}:orange[({p_count} Pending)]")
            else:
                s1.write(f"**{s}** {market_badge}")

            # 1. VISUAL CONFIDENCE BAR
            # Colors progress based on confidence level and trade direction
            conf_label = f"{ai_dir} Conf: {ai_conf:.1%}"

            # Route colors dynamically: LONG -> Green, SHORT -> Red, NEUTRAL -> Default/Gray
            if ai_dir == "LONG":
                # Starpf/Progress color syntax variation using custom markdown wrapper
                s3.markdown(f"**Long Certainty**")
                s3.progress(ai_conf, text=f"🍏 {conf_label}")
            elif ai_dir == "SHORT":
                s3.markdown(f"**Short Certainty**")
                s3.progress(ai_conf, text=f"🍎 {conf_label}")
            else:
                s3.markdown(f"**Neutral/Flat**")
                s3.progress(ai_conf, text=f"⚪ {conf_label}")

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

            # --- 3. AUTO-EXECUTION CHECK (The Directional Accuracy Gate) ---
            if active_now and ai_conf >= st.session_state.ai_threshold:

                # --- COOL-DOWN GATE FOR CONSECUTIVE LOSSES (15 MINUTE LOCK) ---
                is_cooled_down = False
                try:
                    # Request the last 2 closed/filled orders for this specific symbol
                    order_filter = GetOrdersRequest(
                        status=QueryOrderStatus.CLOSED,
                        symbols=[s],
                        limit=2,
                        direction="desc"  # FIX: Replaced Sort.DESC with raw string value
                    )
                    recent_orders = trading_client.get_orders(order_filter)

                    # Check if we have at least 2 closed orders to evaluate
                    if recent_orders and len(recent_orders) >= 2:
                        loss_count = 0
                        # FIXED: Calls datetime.now() directly assuming 'from datetime import datetime' is used
                        time_threshold = datetime.now(UTC) - timedelta(minutes=15)

                        for o in recent_orders:
                            # Confirm the order actually filled and happened within the 15-minute window
                            if o.filled_at and o.filled_at >= time_threshold:
                                f_price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
                                legs = o.legs if hasattr(o, 'legs') and o.legs else []

                                # Method 1: Evaluate bracket structures
                                if hasattr(o, 'legs') and legs:
                                    for leg in legs:
                                        if leg.status.value == "filled":
                                            lf_price = float(leg.filled_avg_price) if leg.filled_avg_price else 0.0
                                            if (o.side == OrderSide.BUY and lf_price < f_price) or (o.side == OrderSide.SELL and lf_price > f_price):
                                                loss_count += 1
                                else:
                                    # Method 2: Standard tracking evaluation 
                                    opp_side_loss = (o.side == OrderSide.SELL and f_price < float(price)) or (o.side == OrderSide.BUY and f_price > float(price))
                                    if opp_side_loss:
                                        loss_count += 1

                        if loss_count >= 2:
                            is_cooled_down = True
                            st.error(f"🛑 {s} auto-execution locked! 2 consecutive losses in the last 15 mins.")
                except Exception as e:
                    st.caption(f"⚠️ Error checking cool-down status for {s}: {e}")

                # Process trading signals only if the cooldown gate is not locked
                if not is_cooled_down:
                    # Check if we are currently neutral (not holding anything)
                    if not is_held and not is_pending:

                        # --- LONG TRIGGER (Buy) ---
                        if ai_dir == "LONG":
                            execute_trade(s, round(float(price), 2), ai_conf, side=OrderSide.BUY, is_bot=True)
                            st.success(f"🤖 AI LONG: Buying {s} at {ai_conf:.1%} confidence")

                        # --- SHORT TRIGGER (Sell Short) ---
                        elif ai_dir == "SHORT":
                            # Criteria: Ensure we aren't shorting at the statistical floor (BBL)
                            bbl_cols = [c for c in df.columns if 'BBL' in c]
                            l_bbl = df[bbl_cols[0]].iloc[-1] if bbl_cols else price

                            # Additional Short Safety: Price must be at least 0.2% above the floor
                            if price > (l_bbl * 1.0015):
                                execute_trade(s, round(float(price), 2), ai_conf, side=OrderSide.BUY, is_bot=True)
                                st.warning(f"📉 AI SHORT: Selling {s} at {ai_conf:.1%} confidence")
                            else:
                                st.caption(f"⚠️ Short skipped: {s} is too close to support floor (BBL).")
                    elif is_pending:
                        st.caption(f"⏳ Bot skipping {s}: Order already in flight.")
                    else:
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

    # --- PENDING ORDERS DASHBOARD TAB/ROW ---
    st.markdown("---")
    st.subheader("📋 Active Pending Orders")

    # Fetch current pending data
    pending_df = get_pending_orders_df(trading_client) # Pass your active Alpaca trading client instance

    if not pending_df.empty:
        # Display an interactive, sortable UI table
        st.dataframe(
            pending_df, 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("No active pending orders found.")




    # --- TRADE HISTORY DASHBOARD SECTION WITH FILTERS ---
    st.divider()
    st.subheader("📜 Trade History (Filled Orders Only)")

    # Create UI layout rows for the data filters
    f_col1, f_col2, f_col3 = st.columns([1, 1.5, 1])

    with f_col1:
        # Text input search box for symbols
        search_symbol = st.text_input("🔍 Search Symbol", value="").strip().upper()

    with f_col2:
        # Decouple native references by pulling fresh datetime boundaries locally
        import datetime as ui_dt
        current_now = ui_dt.datetime.now()
        thirty_days_ago = current_now - ui_dt.timedelta(days=30)

        # Date range selector window
        date_range = st.date_input(
            "📅 Execution Date Range", 
            value=[thirty_days_ago.date(), current_now.date()]
        )

    # Safely split dates if a full range selection exists
    start_dt, end_dt = None, None
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt, end_dt = date_range[0], date_range[1]

    # Fetch the raw historical dataset matching the date range boundaries
    history_df = get_trade_history_df(trading_client, limit=100, start_date=start_dt, end_date=end_dt)

    if not history_df.empty:
        # Client-side processing: Filter the resulting dataframe matches for symbols dynamically
        if search_symbol:
            history_df = history_df[history_df["Symbol"] == search_symbol]

        # Re-verify that data items remain after applying user text match filters
        if not history_df.empty:
            with f_col3:
                st.write("") # Layout spacer alignment padding 
                st.write("") 
                # Convert the currently active filtered dataset state into export files
                csv_bytes = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Filtered (CSV)", 
                    data=csv_bytes, 
                    file_name="filtered_trade_history.csv", 
                    mime="text/csv",
                    key="btn_export_filtered_history"
                )

            # Render the interactive UI table matching your specifications
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No completed records found matching the symbol '{search_symbol}'.")
    else:
        st.info("No completed filled orders found in the selected date window.")
    st.divider()



    # Add this right after the progress bar in your loop
    with st.expander(f"🔍 Why {s}?"):
        if feat_map:
            f_df = pd.DataFrame(list(feat_map.items()), columns=['Factor', 'Weight']).sort_values('Weight')
            st.bar_chart(f_df.set_index('Factor'), horizontal=True, height=200)

    for percent_complete in range(100):
        time.sleep(0.6) # 0.6s * 100 = 60 seconds
        prog_placeholder.progress(percent_complete + 1, text=f"Next update in {60 - int(percent_complete*0.6)}s")
live_ui()
