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
from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from alpaca.trading.requests import GetOrdersRequest
# Add Sort to your imports at the top of your file
from alpaca.trading.enums import OrderClass, QueryOrderStatus
from zoneinfo import ZoneInfo

# --- 1. CONFIG & CLIENTS ---
try:
    API_KEY =  st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
except:
    st.error("Please set API_KEY and SECRET_KEY in Streamlit Secrets.")
    st.stop()

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
st.set_page_config(page_title="KP-ALPHAFORGE", layout="wide")

# --- 2. PERSISTENCE ENGINE ---
SETTINGS_FILE = "settings.json"
LOG_FILE = "trade_history.log"

if "model_vault" not in st.session_state:
    st.session_state.model_vault = {}


def save_settings():
    keys = ["tickers", "run_bot", "order_mode","sizing_mode", "order_val", "trailing_pct", 
            "profit_target", "ai_threshold", "vix_threshold", "lock_profit_pct", 
            "daily_loss_limit", "global_profit_goal", "allow_ext_hours","profit_target","bot_active"]
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
    defaults = {"tickers": ["SPY", "QQQ", "NVDA","GOOGL", "IWM","FDVV"], "run_bot": False, "order_mode": "USD","sizing_mode": "USD", 
                "order_val": 500.0, "trailing_pct": .2, "profit_target": 0.5, 
                "ai_threshold": 0.85, "vix_threshold": 25.0, "lock_profit_pct": 0.5,
                "daily_loss_limit": 100.0, "global_profit_goal": 1000.0, "allow_ext_hours": False}
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

def calculate_win_loss_metrics(history_df):
    """Calculates win/loss statistics and gross USD PnL metrics by matching buy/sell transactions with built-in memory caching."""
    if history_df.empty:
        return {
            "win_rate": "0.0%", "win_loss_ratio": "0.00", "total_trades": 0, "wins": 0, "losses": 0,
            "gross_profit_usd": 0.0, "gross_loss_usd": 0.0, "net_profit_usd": 0.0, "profit_factor": "0.00"
        }

    # ====================================================================================
    # HYBRID MEMORY CACHE LAYER: DETECTS DATASET MODIFICATIONS USING STRUCTURAL HASHING
    # ====================================================================================
    try:
        # Create a completely unique string signature based on row count and data characteristics
        data_signature = f"metrics_hash_{len(history_df)}_{history_df.iloc[0]['Execution Time'] if not history_df.empty else 'empty'}_{history_df.iloc[-1]['Execution Time'] if not history_df.empty else 'empty'}"

        # Serve results instantly from fast local memory if this exact dataset signature was analyzed already
        if "cached_win_loss_metrics" in st.session_state and st.session_state.get("metrics_cache_sig") == data_signature:
            return st.session_state["cached_win_loss_metrics"]
    except Exception:
        data_signature = None
    # ====================================================================================

    # Sort history from oldest to newest to track progression accurately
    df = history_df.copy()
    df['Execution Time'] = pd.to_datetime(df['Execution Time'])
    df = df.sort_values(by='Execution Time', ascending=True)

    # Storage trackers for matching pairs
    open_trades = {} # Maps symbol -> (entry_price, entry_qty, side)
    wins = 0
    losses = 0
    gross_profit_usd = 0.0
    gross_loss_usd = 0.0

    for _, row in df.iterrows():
        symbol = row['Symbol']
        side = row['Side'] # BUY or SELL
        price = row['Avg Price']
        qty = row['Filled Qty']

        # If we don't have an active tracking position for this ticker, open one
        if symbol not in open_trades:
            open_trades[symbol] = (price, qty, side)
        else:
            entry_price, entry_qty, entry_side = open_trades[symbol]

            # Verify that this transaction closes out our original entry direction
            if side != entry_side:
                # Use the smaller quantity of the two matches to prevent scaling bugs
                trade_qty = min(qty, entry_qty)

                if entry_side == "BUY":
                    # LONG Trade: Profit = (Exit Price - Entry Price) * Quantity
                    trade_pnl = (price - entry_price) * trade_qty
                    is_win = price > entry_price
                else:
                    # SHORT Trade: Profit = (Entry Price - Exit Price) * Quantity
                    trade_pnl = (entry_price - price) * trade_qty
                    is_win = price < entry_price

                if is_win:
                    wins += 1
                    gross_profit_usd += abs(trade_pnl)
                else:
                    losses += 1
                    gross_loss_usd += abs(trade_pnl)

                # Close the tracking loop for this matched trade pair
                del open_trades[symbol]

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    win_loss_ratio = (wins / losses) if losses > 0 else float(wins)

    # Advanced Financial Calculations
    net_profit_usd = gross_profit_usd - gross_loss_usd
    profit_factor = (gross_profit_usd / gross_loss_usd) if gross_loss_usd > 0 else float(gross_profit_usd)

    final_metrics_output = {
        "win_rate": f"{win_rate:.1f}%",
        "win_loss_ratio": f"{win_loss_ratio:.2f}",
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "gross_profit_usd": gross_profit_usd,
        "gross_loss_usd": gross_loss_usd,
        "net_profit_usd": net_profit_usd,
        "profit_factor": f"{profit_factor:.2f}" if gross_loss_usd > 0 else "Max (No Losses)"
    }

    # ====================================================================================
    # COMMIT RESULTS TO CACHE LAYER BEFORE EXITING THE CORE ROUTINE
    # ====================================================================================
    if data_signature:
        st.session_state["cached_win_loss_metrics"] = final_metrics_output
        st.session_state["metrics_cache_sig"] = data_signature
    # ====================================================================================

    return final_metrics_output


if "is_admin_unlocked" not in st.session_state:
        st.session_state["is_admin_unlocked"] = False

# --- 3. SIDEBAR ---
with st.sidebar:

    # ====================================================================================
    # 🔒 SINGLE-FORM ADMIN AUTHENTICATION GATE (STREAMLIT SECRETS)
    # ====================================================================================
    # Initialize the persistent lock boolean in memory if missing

    st.sidebar.markdown("### 🔑 System Authorization")

    # Render lock status indicator badges
    if st.session_state["is_admin_unlocked"]:
        st.sidebar.success("🔓 Admin Mode Active")
        if st.sidebar.button("🔒 Lock Admin Settings", use_container_width=True):
            st.session_state["is_admin_unlocked"] = False
            st.rerun()
    else:
        st.sidebar.warning("🔒 Viewer Mode (Controls Locked)")

        # Isolated authentication container form wrapper
        with st.sidebar.form("admin_auth_form", clear_on_submit=True):
            input_pass = st.text_input("Enter Admin Password", type="password", placeholder="Password")
            submit_auth = st.form_submit_button("⚡ Unlock Admin Access", use_container_width=True)

            if submit_auth:
                try:
                    # Compare entered token string against the hidden Streamlit secret string
                    if input_pass == st.secrets["trading_credentials"]["admin_password"]:
                        st.session_state["is_admin_unlocked"] = True
                        st.toast("Admin clearance granted! Controls unlocked.", icon="🔓")
                        st.rerun()
                    else:
                        st.error("Incorrect key token.")
                except KeyError:
                    st.error("Secrets file '[trading_credentials][admin_password]' is missing.")

    st.sidebar.divider()
    # ====================================================================================

    # --- 3. DYNAMIC CONTROL SYSTEM REGIME ---
    # Assign the master true/false disabled flag parameter based on the form status
    admin_disabled = not st.session_state["is_admin_unlocked"]

    # 1. Interactive Sidebar Risk Controls
    st.sidebar.markdown("### 🛡️ System Risk Settings")
    # Calibrated specifically to track VIXY daily percentage spike metrics
    cfg_vix_max = st.sidebar.slider("Max VIXY Daily Spike %", 5.0, 30.0, 10.0, 0.5, disabled=admin_disabled)
    cfg_index_drop = st.sidebar.slider("Max Index Daily Drop %", -10.0, -1.0, -5.0, 0.1, disabled=admin_disabled)


    # Manual Override Checkbox to completely force-ignore circuit breaker shutdowns
    breaker_bypass = st.sidebar.checkbox("🔓 Bypass Crash Protection", value=False, help="Forces bot to trade regardless of crashes", disabled=admin_disabled)


    st.header("🛒 Order Configuration")

    # Ensure required tracking states exist to prevent index out of bounds flags
    if "sizing_mode" not in st.session_state:
        st.session_state["sizing_mode"] = "USD"
    if "order_mode" not in st.session_state:
        st.session_state["order_mode"] = "USD"

    # ====================================================================================
    # 1. STATE CONTROLLER: SYNCHRONIZE INPUT VARIATION MATRIX AUTOMATICALLY
    # ====================================================================================
    # Advanced models (Kelly / Volatility) process capital allocations natively in USD.
    # We force-sync the order mode to USD behind the scenes if an advanced matrix is chosen.
    if st.session_state["sizing_mode"] in ["Kelly", "Volatility"]:
        st.session_state["order_mode"] = "USD"
        inputs_disabled = True  # Locks manual adjustments because the engine takes control
    else:
        inputs_disabled = admin_disabled  # Defaults back to standard admin toggles

    # Toggle between USD (Dollar) and Shares (Stock)
    st.selectbox(
        "Order Mode", 
        options=["USD", "Shares"], 
        key="order_mode", 
        on_change=save_settings, 
        disabled=inputs_disabled # Dynamic lock maintains framework sync
    )

    # Dynamic Label and Value based on selection
    if st.session_state.order_mode == "USD":
        st.number_input(
            "Order Amount ($)", 
            min_value=1.0, 
            step=10.0, 
            key="order_val", 
            on_change=save_settings, 
            disabled=inputs_disabled
        )
    else:
        st.number_input(
            "Number of Shares", 
            min_value=1.0, 
            step=1.0, 
            key="order_val", 
            on_change=save_settings, 
            disabled=inputs_disabled
        )
    st.divider()

    # ====================================================================================
    # 2. INTEGRATED CAPITAL SIZING ENGINE UI MATRIX (SYNCHRONIZED PATHWAY)
    # ====================================================================================
    st.markdown("### 🧮 Capital Sizing Engine")

    # Primary Sizing Mechanism Selector
    selected_mode = st.radio(
        "Position Sizing Mode",
        options=["USD", "Shares", "Kelly", "Volatility"],
        index=["USD", "Shares", "Kelly", "Volatility"].index(st.session_state["sizing_mode"]),
        key="sizing_mode",
        on_change=save_settings,
        disabled=admin_disabled,
        help="USD/Shares: Static entry values. Kelly: Risk-adjusted win percentage weight. Volatility: Automatically down-sizes cash exposure when VIXY triggers a market spike."
    )

    # 3. DYNAMIC CONTEXTUAL METRICS DISPLAY LAYER
    if selected_mode == "Kelly":
        cached_stats = st.session_state.get("cached_win_loss_metrics", None)

        if cached_stats and cached_stats.get("total_trades", 0) >= 5:
            try:
                win_rate = float(cached_stats["win_rate"].replace("%", "")) / 100.0
                win_loss_ratio = float(cached_stats["win_loss_ratio"])

                # Re-verify the formula matrix: K% = W - [(1 - W) / R]
                kelly_pct = win_rate - ((1.0 - win_rate) / win_loss_ratio)
                fractional_kelly = kelly_pct * 0.20 
                safe_kelly_pct = max(0.0, min(fractional_kelly, 0.10))

                st.success(f"Advance Allocation Rule Active:\n🤖 System is overriding inputs to risk exactly **{safe_kelly_pct:.1%}** of portfolio equity.")
            except Exception:
                st.caption("🔄 Computing model parameters...")
        else:
            st.warning("⏳ Kelly Pipeline Paused: Requires at least **5 closed trades** in your Performance History to compute a baseline. Currently using manual input entries.")

    elif selected_mode == "Volatility":
        current_vixy_shift = 0.0
        if "global_market_risk_matrix" in st.session_state:
            for row in st.session_state["global_market_risk_matrix"].get("matrix_rows", []):
                if "VIXY" in row.get("Technical Factor", ""):
                    try: current_vixy_shift = float(row["Daily Chg %"].replace("%", ""))
                    except Exception: pass

        max_vix_limit = float(st.session_state.get("cfg_vix_max", 10.0))

        if current_vixy_shift > 0:
            vol_multiplier = max(0.20, 1.0 - (current_vixy_shift / max_vix_limit))
        else:
            vol_multiplier = 1.0

        if vol_multiplier < 1.0:
            st.warning(f"⚠️ Volatility Scaling Active:\n🤖 Scaling manual input value **${st.session_state.order_val:,.2f}** down to **{vol_multiplier:.0%}** (${(st.session_state.order_val * vol_multiplier):,.2f}) due to market stress.")
        else:
            st.info(f"🍏 Volatility Safe\nMarket stable. Using full manual allocation boundary (**${st.session_state.order_val:,.2f}**).")

    else:
        st.caption(f"Using fixed execution bounds configured by your manual order settings.")

    st.divider()


    st.header("🤖 Bot Control")
    st.toggle("Activate AI Bot", key="run_bot", on_change=save_settings, disabled=admin_disabled)
    st.toggle("Allow Extended Hours", key="allow_ext_hours", on_change=save_settings, disabled=admin_disabled)
    st.slider("AI Trigger Threshold", 0.50, 0.98, key="ai_threshold", on_change=save_settings, disabled=admin_disabled)

    st.divider()
    st.header("📂 Watchlist")
    new_t = st.text_input("Add Ticker").upper().strip()
    if st.button("➕ Add", disabled=admin_disabled):
        if new_t and new_t not in st.session_state.tickers:
            st.session_state.tickers.append(new_t); save_settings(); st.rerun()
    st.multiselect("Active Watchlist", options=st.session_state.tickers, key="tickers", on_change=save_settings, disabled=admin_disabled)

    st.divider()
    st.header("🏁 Daily Targets")
    st.number_input("Profit Goal ($)", key="global_profit_goal", on_change=save_settings, disabled=admin_disabled)
    st.number_input("Loss Limit ($)", key="daily_loss_limit", on_change=save_settings, disabled=admin_disabled)

    st.divider()
    st.header("🛡️ Strategy")
    st.slider("Trailing Start %", 0.0, 50.0, key="lock_profit_pct", on_change=save_settings, disabled=admin_disabled)
    # Take Profit threshold slider (e.g., automatically close position at +5.0% profit)
    st.slider(
        "Take Profit Target %", 
        min_value=0.0, 
        max_value=50.0, 
        step=0.5,
        key="profit_target",
        on_change=save_settings,
        help="Automatically liquidates an active position if its profit matches or exceeds this percentage.", disabled=admin_disabled
    )
    st.slider("Stop Loss %", 0.0, 50.0, key="trailing_pct", on_change=save_settings, disabled=admin_disabled)

    if st.button("🚨 EMERGENCY LIQUIDATE", type="primary", use_container_width=True, disabled=admin_disabled):
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

        # --- UPGRADED: MULTI-BAR LOOKAHEAD & DYNAMIC ATR THRESHOLDS ---
        LOOKAHEAD_BARS = 10  # Expanded from 5 to 10 to capture structural micro-trends
        ATR_MULTIPLIER = 1.5 # Requires the target move to exceed 1.5x the current asset volatility

        # Find the ATR column name dynamically
        atr_col = [c for c in df.columns if 'ATR' in c][0]

        # Calculate individual dynamic percent thresholds for every single row
        # (ATR / Close) gives the volatility percentage, which we multiply by our multiplier
        df['dynamic_threshold'] = (df[atr_col] / df['close']) * ATR_MULTIPLIER

        # Prevent the threshold from dropping too low in compressed squeeze regimes (minimum 0.08%)
        df['dynamic_threshold'] = df['dynamic_threshold'].clip(lower=0.0008)

        # Isolate the close column first, then apply the reverse slicing operations
        close_series = df['close']
        future_highest_close = close_series[::-1].rolling(window=LOOKAHEAD_BARS, min_periods=1).max()[::-1]
        future_lowest_close = close_series[::-1].rolling(window=LOOKAHEAD_BARS, min_periods=1).min()[::-1]

        # Shift ahead by -1 so the current bar doesn't look at itself
        future_max = future_highest_close.shift(-1)
        future_min = future_lowest_close.shift(-1)

        # Multi-Class Target (1=Long, 2=Short, 0=Neutral)
        df['target'] = 0

        # UPGRADED TARGETING: Replaced hardcoded 1.0015/0.9985 with the dynamic volatility band
        df.loc[future_max > df['close'] * (1.0 + df['dynamic_threshold']), 'target'] = 1
        df.loc[future_min < df['close'] * (1.0 - df['dynamic_threshold']), 'target'] = 2

        # Conflict Reconciliation if both thresholds are crossed within the lookahead window
        both_hit = (future_max > df['close'] * (1.0 + df['dynamic_threshold'])) & \
                   (future_min < df['close'] * (1.0 - df['dynamic_threshold']))
        long_distance = future_max - df['close']
        short_distance = df['close'] - future_min

        df.loc[both_hit & (long_distance >= short_distance), 'target'] = 1
        df.loc[both_hit & (short_distance > long_distance), 'target'] = 2

        df = df.ffill().dropna()

        # Drop incomplete edge records at the end of the dataframe to prevent training lookahead leakage
        if len(df) > LOOKAHEAD_BARS:
            df = df.iloc[:-LOOKAHEAD_BARS]

        feature_keywords = ['RSI', 'MACD', 'ADX', 'EMA', 'ATR', 'BBL', 'BBU', 'MFI', 'volume', 'VOL_SMA']
        features = [c for c in df.columns if any(x in c for x in feature_keywords)]

        # --- 3. TRAINING LOGIC ---
        now = time.time()
        brain = st.session_state.model_vault.get(symbol, {"time": 0})

        if (now - brain["time"]) > 3600:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()

            train_data = df.iloc[-500:] if len(df) > 500 else df
            X, y_raw = train_data[features].iloc[:-2].copy(), train_data['target'].iloc[:-2].copy()

            # Use LabelEncoder to compress classes into consecutive integers starting at 0
            y = le.fit_transform(y_raw)
            detected_classes = len(le.classes_)

            m_rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            m_xgb = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=6, 
                objective='multi:softprob' if detected_classes > 2 else 'binary:logistic', 
                num_class=detected_classes if detected_classes > 2 else None, 
                tree_method='hist', 
                n_jobs=-1
            )

            m_rf.fit(X, y); m_xgb.fit(X, y)

            st.session_state.model_vault[symbol] = {
                "models": (m_rf, m_xgb), 
                "time": now,
                "importance": dict(zip(features, m_rf.feature_importances_)),
                "encoder": le
            }

        # --- 4. INFERENCE ---
        active_brain = st.session_state.model_vault[symbol]
        rf_m, xgb_m = active_brain["models"]
        le = active_brain["encoder"]

        last_rows = df[features].tail(10)
        p_rf = rf_m.predict_proba(last_rows)
        p_xgb = xgb_m.predict_proba(last_rows)

        # Calculate raw average probabilities across columns
        avg_probs = (p_rf + p_xgb) / 2

        # Reconstruct full 3-class projection maps (0, 1, 2) matching your downstream logic gates
        long_probs = [0.0] * len(last_rows)
        short_probs = [0.0] * len(last_rows)

        # Map transformed column positions back to their true target labels
        for position_idx, true_label in enumerate(le.classes_):
            if true_label == 1:
                long_probs = avg_probs[:, position_idx]
            elif true_label == 2:
                short_probs = avg_probs[:, position_idx]

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


def calculate_kelly_qty(portfolio_value, price, stats, risk_fraction=0.20):
    """
    Uses the Kelly Criterion to find the optimal cash allocation percentage.
    risk_fraction (Fractional Kelly) tones down the math to prevent aggressive over-betting.
    """
    try:
        # Extract win rate and win/loss ratio from your cached stats dictionary
        win_rate = float(stats["win_rate"].replace("%", "")) / 100.0
        win_loss_ratio = float(stats["win_loss_ratio"])

        if win_rate <= 0 or win_loss_ratio <= 0:
            return None # Return fallback to your original static configurations

        # Kelly Formula: K% = W - [(1 - W) / R]
        # W = Win Probability, R = Win/Loss Ratio
        kelly_percentage = win_rate - ((1.0 - win_rate) / win_loss_ratio)

        # Apply the fractional risk padding scaling multiplier and boundary limits (0% to 10% max portfolio cash per trade)
        safe_kelly_pct = max(0.0, min(kelly_percentage * risk_fraction, 0.10))

        # Translate optimal cash percentage to an exact share allocation quantity
        target_usd_allocation = portfolio_value * safe_kelly_pct
        qty = int(target_usd_allocation // price)
        return qty if qty >= 1 else 1
    except Exception:
        return None # Return fallback if division-by-zero or value errors map out

def calculate_volatility_adjusted_qty(base_usd_val, price, current_vixy_pct):
    """
    Volatility-Based Sizing (ATR alternative using dynamic VIXY percentage metrics).
    Scales down position size linearly if current VIXY shifts past your sidebar target ceiling.
    """
    try:
        # Read the active sidebar crash parameter directly from memory
        max_vix_limit = float(st.session_state.get("cfg_vix_max", 10.0))

        # Determine current multiplier: if volatility is 0% or negative, size stays at 100%
        if current_vixy_pct <= 0:
            vol_multiplier = 1.0
        else:
            # Linear decay: as VIXY approaches or breaches limits, cash allocation drops smoothly down to a minimum of 20%
            vol_multiplier = max(0.20, 1.0 - (current_vixy_pct / max_vix_limit))

        adjusted_usd_val = base_usd_val * vol_multiplier
        qty = int(adjusted_usd_val // price)
        return qty if qty >= 1 else 1
    except Exception:
        return None



# Helper for execution (Supports USD/Shares/Kelly/Volatility Sizing, Extended Hours, and Duplicate Protection)
def execute_trade(s, price, ai_conf, side=OrderSide.BUY, is_bot=False):
    try:
        # --- 1. DUPLICATE PROTECTION: Check Positions and Pending Orders ---
        pos = trading_client.get_all_positions()
        if any(p.symbol == s for p in pos):
            return # Exit silently if position exists

        order_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[s])
        open_orders = trading_client.get_orders(filter=order_filter)
        if open_orders:
            return # Exit silently if order is pending

        # --- 2. ADVANCED HYBRID SIZING ENGAGEMENT (INTEGRATED KELLY & VOLATILITY MATRIX) ---
        base_usd_value = float(st.session_state.order_val)
        computed_qty = None

        # Pull account equity from optimized hybrid registers or fall back safely to live REST
        portfolio_value = float(st.session_state.get("ws_equity", 0.0))
        if portfolio_value <= 0:
            try: portfolio_value = float(trading_client.get_account().equity)
            except: portfolio_value = 10000.0 # Standard defensive seed fallback

        sizing_mode = st.session_state.get("sizing_mode", "USD") # Managed by your sidebar radio widget

        if sizing_mode == "Kelly":
            # Extract live round-trip metric parameters from your dynamic cache signature dictionary
            cached_stats = st.session_state.get("cached_win_loss_metrics", None)
            if cached_stats and cached_stats.get("total_trades", 0) >= 5:
                try:
                    win_rate = float(cached_stats["win_rate"].replace("%", "")) / 100.0
                    win_loss_ratio = float(cached_stats["win_loss_ratio"])

                    if win_rate > 0 and win_loss_ratio > 0:
                        # Kelly Formula: K% = W - [(1 - W) / R] with a fractional multiplier constraint of 0.20
                        kelly_percentage = win_rate - ((1.0 - win_rate) / win_loss_ratio)
                        safe_kelly_pct = max(0.0, min(kelly_percentage * 0.20, 0.10)) # Cap risk to max 10% cash

                        target_usd_allocation = portfolio_value * safe_kelly_pct
                        computed_qty = int(target_usd_allocation // price)
                except Exception:
                    pass

        elif sizing_mode == "Volatility":
            # Read real-time VIXY dynamic percent shifts computed by your top-level layout block
            current_vixy_shift = 0.0
            if "global_market_risk_matrix" in st.session_state:
                for row in st.session_state["global_market_risk_matrix"].get("matrix_rows", []):
                    if "VIXY" in row.get("Technical Factor", ""):
                        try: current_vixy_shift = float(row["Daily Chg %"].replace("%", ""))
                        except: pass
            try:
                max_vix_limit = float(st.session_state.get("cfg_vix_max", 10.0))
                # Linear decay fallback: scales position size lower if VIXY triggers a spike (min size 20%)
                vol_multiplier = max(0.20, 1.0 - (current_vixy_shift / max_vix_limit)) if current_vixy_shift > 0 else 1.0
                computed_qty = int((base_usd_value * vol_multiplier) // price)
            except Exception:
                pass

        # --- FALLBACK SELECTION STRUCTURE ---
        if computed_qty is not None:
            qty = computed_qty
        elif st.session_state.order_mode == "USD":
            qty = int(st.session_state.order_val // price)
        else:
            qty = int(st.session_state.order_val)

        if qty < 1:
            st.error(f"Order value too low for {s}")
            return

        clock = trading_client.get_clock()

        # Determine exit side for the advanced legs
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # --- 3. ADVANCED EXIT TARGET PRICE CALCULATIONS ---
        # Derive precise price values based on your saved sidebar slider states
        # Handles Short Sell math automatically if side == OrderSide.SELL
        tp_multiplier = 1 + (st.session_state.profit_target / 100.0) if side == OrderSide.BUY else 1 - (st.session_state.profit_target / 100.0)
        sl_multiplier = 1 - (st.session_state.trailing_pct / 100.0) if side == OrderSide.BUY else 1 + (st.session_state.trailing_pct / 100.0)

        target_take_profit_price = round(price * tp_multiplier, 2)
        target_stop_loss_price = round(price * sl_multiplier, 2)

        # --- 4. EXECUTION MATRIX ---
        if clock.is_open:
            # REGULAR MARKET HOURS: Advanced Bracket Order (Entry via LIMIT)
            # The exchange holds the exit legs until your entry price is filled
            trading_client.submit_order(LimitOrderRequest(
                symbol=s,
                qty=qty,
                limit_price=price,
                side=side.value,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET, # Bundles the entire sequence
                take_profit=TakeProfitRequest(limit_price=target_take_profit_price),
                stop_loss=StopLossRequest(stop_price=target_stop_loss_price)
            ))
        else:
            # ====================================================================================
            # STEP 1: SUBMIT ENTRY LIMIT ORDER (COMPLIANT WITH AFTER-HOURS RULES)
            # ====================================================================================
            entry_order = trading_client.submit_order(LimitOrderRequest(
                symbol=s,
                qty=qty,
                limit_price=price,
                side=side.value,
                time_in_force=TimeInForce.DAY,  # Mandatory for extended hours
                extended_hours=True            # Bypasses regular hours validation
            ))

            print(f"Entry order {entry_order.id} submitted. Awaiting execution...")

            # ====================================================================================
            # STEP 2: MONITOR ENTRY ORDER FILL STATE (HYBRID WEBSOCKET TIMEOUT STRUCTURE)
            # ====================================================================================
            is_filled = False
            while not is_filled:
                # Primary Path: Pull real-time order update states from WebSocket streams
                ws_update_key = f"ws_order_status_{entry_order.id}"
                if ws_update_key in st.session_state and st.session_state[ws_update_key] is not None:
                    current_status = st.session_state[ws_update_key]
                else:
                    # Backup Path: Silently fetch from REST API if streaming updates have an expected network delay
                    try:
                        check_order = trading_client.get_order_by_id(entry_order.id)
                        current_status = check_order.status
                    except Exception:
                        current_status = OrderStatus.HELD # Maintain placeholder while network reconnects

                if current_status == OrderStatus.FILLED:
                    print("Entry order filled! Deploying synthetic brackets...")
                    is_filled = True
                elif current_status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    raise RuntimeError(f"Entry order terminated without filling: {current_status}")

                time.sleep(1) # Frequency buffer to prevent API rate limiting

            # ====================================================================================
            # STEP 3: SUBMIT INDEPENDENT AFTER-HOURS EXIT LEGS
            # ====================================================================================
            # Profit Booking Leg (Standard Limit)
            tp_order = trading_client.submit_order(LimitOrderRequest(
                symbol=s,
                qty=qty,
                limit_price=target_take_profit_price,
                side=exit_side.value,
                time_in_force=TimeInForce.DAY,
                extended_hours=True
            ))

            # Synthetic Stop Loss Leg (Must be submitted as a LIMIT order for after-hours)
            sl_order = trading_client.submit_order(LimitOrderRequest(
                symbol=s,
                qty=qty,
                limit_price=target_stop_loss_price,
                side=exit_side.value,
                time_in_force=TimeInForce.DAY,
                extended_hours=True
            ))

            # ====================================================================================
            # STEP 4: OCO REPLICATION LOOP (MUTUAL CANCELLATION ENGINE)
            # ====================================================================================
            while True:
                # Read Take Profit tracking index status with dynamic hybrid streaming fallbacks
                tp_update_key = f"ws_order_status_{tp_order.id}"
                if tp_update_key in st.session_state and st.session_state[tp_update_key] is not None:
                    tp_status = st.session_state[tp_update_key]
                else:
                    try: tp_status = trading_client.get_order_by_id(tp_order.id).status
                    except: tp_status = OrderStatus.HELD

                # Read Stop Loss tracking index status with dynamic hybrid streaming fallbacks
                sl_update_key = f"ws_order_status_{sl_order.id}"
                if sl_update_key in st.session_state and st.session_state[sl_update_key] is not None:
                    sl_status = st.session_state[sl_update_key]
                else:
                    try: sl_status = trading_client.get_order_by_id(sl_order.id).status
                    except: sl_status = OrderStatus.HELD

                # If profit target hit, kill the stop loss
                if tp_status == OrderStatus.FILLED:
                    print("Take profit filled. Cancelling synthetic stop loss.")
                    trading_client.cancel_order_by_id(sl_order.id)
                    break

                # If stop target hit, kill the profit target
                if sl_status == OrderStatus.FILLED:
                    print("Stop loss filled. Cancelling take profit order.")
                    trading_client.cancel_order_by_id(tp_order.id)
                    break

                # Safeguard against manual or daily closing cancellations
                if tp_status == OrderStatus.CANCELED or sl_status == OrderStatus.CANCELED:
                    print("One of the legs was cancelled externally. Cleaning up remaining positions.")
                    try: trading_client.cancel_order_by_id(tp_order.id)
                    except: pass
                    try: trading_client.cancel_order_by_id(sl_order.id)
                    except: pass
                    break

                time.sleep(1)

        # --- 5. LOGGING & NOTIFICATION ---
        action_type = "Long" if side == OrderSide.BUY else "Short"
        msg = f"{'🤖 Bot' if is_bot else '👤 Manual'} {action_type} Advanced Limit Entry Set: {s} @ ${price} (Qty: {qty}) | TP Target: ${target_take_profit_price} | SL Target: ${target_stop_loss_price}"
        add_log(msg)
        st.toast(msg, icon="🚀")

    except Exception as e:
        st.error(f"Trade Failed: {e}")


def get_pending_orders_df(trading_client):
    """Fetches open/pending orders and converts them into a formatted Pandas DataFrame."""
    import time as sys_time

    # ====================================================================================
    # HYBRID MEMORY CACHE LAYER: ELIMINATES REST API LAG ON MANUAL UI INTERACTIONS
    # ====================================================================================
    cache_key = "cached_pending_orders_df"
    cache_time_key = "pending_orders_last_update"

    # If the cache exists and was updated less than 60 seconds ago, return it instantly
    if cache_key in st.session_state and sys_time.time() - st.session_state.get(cache_time_key, 0) < 60:
        return st.session_state[cache_key]
    # ====================================================================================

    try:
        # Request only 'open' (pending) orders from Alpaca
        filter_request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(filter=filter_request)

        if not open_orders:
            empty_df = pd.DataFrame()
            # Cache the empty state to remain consistent
            st.session_state[cache_key] = empty_df
            st.session_state[cache_time_key] = sys_time.time()
            return empty_df # Return empty dataframe if no orders pending

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

        final_df = pd.DataFrame(order_data)

        # Save the structured dataframe to the local session state cache
        st.session_state[cache_key] = final_df
        st.session_state[cache_time_key] = sys_time.time()

        return final_df
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return pd.DataFrame()

def get_trade_history_df(trading_client, limit=50, start_date=None, end_date=None):
    """Fetches closed/filled orders and converts them into a Pandas DataFrame."""
    import time as sys_time

    # ====================================================================================
    # HYBRID MEMORY CACHE LAYER: PARAMETER-AWARE FILTER ELIMINATES UNNECESSARY HISTORY RE-FETCHING
    # ====================================================================================
    # Create a unique cache identity key based completely on the current search filter parameters
    param_cache_identity = f"history_cache_{limit}_{start_date}_{end_date}"
    time_cache_identity = f"history_last_update_{limit}_{start_date}_{end_date}"

    # If this exact date/limit query was executed within the last 60 seconds, serve it from memory instantly
    if param_cache_identity in st.session_state and sys_time.time() - st.session_state.get(time_cache_identity, 0) < 60:
        return st.session_state[param_cache_identity]
    # ====================================================================================

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
            empty_df = pd.DataFrame()
            # Cache the empty state payload for consistency
            st.session_state[param_cache_identity] = empty_df
            st.session_state[time_cache_identity] = sys_time.time()
            return empty_df # Return empty if history is blank

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

        final_df = pd.DataFrame(history_data)

        # Save the finalized DataFrame and current timestamp into the local parameters cache slot
        st.session_state[param_cache_identity] = final_df
        st.session_state[time_cache_identity] = sys_time.time()

        return final_df
    except Exception as e:
        st.error(f"Error fetching trade history: {e}")
        return pd.DataFrame()


if "cached_asset_names" not in st.session_state:
    st.session_state["cached_asset_names"] = {}


# --- 5. DASHBOARD UI ---
#st.title("🚀 AI Alpha Terminal")

st.markdown(
    """
    <div style="background-color:#0f172a; padding:24px; border-radius:12px; border: 1px solid #1e293b; border-top: 5px solid #d97706; margin-bottom:25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color:#ffffff; margin:0; font-family:sans-serif; font-size:26px; font-weight:700; letter-spacing: 1px;">
                    🪙 KP-ALPHAFORGE <span style="color:#d97706; font-size:13px; font-weight:600; vertical-align:super;">PRO v2.5</span>
                </h1>
                <p style="color:#94a3b8; margin:6px 0 0 0; font-family:sans-serif; font-size:13px; font-style:normal; letter-spacing: 0.5px;">
                    High-Yield Automated Predictive Equity System
                </p>
            </div>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

@st.fragment(run_every=10)
def live_ui():

    # Fetch current time localized explicitly to the Central Time Zone
    now_dt = datetime.now(ZoneInfo("America/Chicago"))
    last_refresh = now_dt.strftime("%I:%M:%S %p CST")

    # 2. Display Header
    t1, t2 = st.columns([1, 1])
    t1.caption(f"🕒 Last Refresh: **{last_refresh}**")
    prog_placeholder = t2.empty() # Placeholder for the countdown

    status = get_market_status()
    market_open = status["open"]
    daily_pnl = get_daily_pnl()
#========================================================================================
    import datetime as ui_dt

    # 1. INITIALIZE DEFAULT FALLBACK DATE VALUES IN SESSION STATE
    current_now = ui_dt.datetime.now(ZoneInfo("America/Chicago"))
    thirty_days_ago = current_now - ui_dt.timedelta(days=30)

    # Store dates in session state to handle the data fetch step before drawing widgets
    if "perf_start_val" not in st.session_state:
        st.session_state["perf_start_val"] = thirty_days_ago.date()
    if "perf_end_val" not in st.session_state:
        st.session_state["perf_end_val"] = current_now.date()

    current_start = st.session_state["perf_start_val"]
    current_end = st.session_state["perf_end_val"]

    # 2. HYBRID FETCH: Run analysis using the active dates to build our dynamic title
    history_df = get_trade_history_df(trading_client, limit=100, start_date=current_start, end_date=current_end)

    if not history_df.empty:
        stats = calculate_win_loss_metrics(history_df)
        current_rate_str = stats["win_rate"]
    else:
        stats = {"win_rate": "0.0%", "wins": 0, "losses": 0, "win_loss_ratio": "0.00", "total_trades": 0}
        current_rate_str = "No Data"

    # ====================================================================================
    # DYNAMIC TEXT HEADER EXPANDER GRID (HOUSES BOTH INPUTS AND METRICS CARDS)
    # ====================================================================================
    expander_title = f"🏆 View Detailed AI Performance Metrics (Current Win Rate: {current_rate_str})"

    with st.expander(expander_title, expanded=False):

        # MOVE INPUTS HERE: Draw calendar pickers inside the top of the expander box
        st.markdown("#### 📅 Performance Date Range Filter")
        date_col1, date_col2 = st.columns(2)

        with date_col1:
            perf_start = st.date_input("Start Date", value=current_start, key="perf_start_input")
        with date_col2:
            perf_end = st.date_input("End Date", value=current_end, key="perf_end_input")

        # If dates change, immediately update backing values and trigger a rerun to calculate fresh stats
        if perf_start != current_start or perf_end != current_end:
            if perf_start > perf_end:
                st.error("Error: Start Date cannot be further in the future than End Date.")
            else:
                st.session_state["perf_start_val"] = perf_start
                st.session_state["perf_end_val"] = perf_end
                st.rerun()

        st.markdown("---")

        # Display performance scorecard summary rows inside the expander
        st.markdown("### 🏆 AI Performance Metrics")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)

        s_col1.metric(
            label="🎯 Model Win Rate", 
            value=stats["win_rate"],
            delta=f"{stats['wins']}W - {stats['losses']}L"
        )

        ratio_val = float(stats["win_loss_ratio"])
        s_col2.metric(
            label="📊 Win/Loss Ratio", 
            value=stats["win_loss_ratio"],
            delta="Profitable Regime" if ratio_val >= 1.0 else "Sub-Optimal Regime",
            delta_color="normal" if ratio_val >= 1.0 else "inverse"
        )

        s_col3.metric(
            label="🔄 Total Closed Round-Trips", 
            value=str(stats["total_trades"])
        )

        # Swapped static placeholder out for dynamic Profit Factor metric integration
        s_col4.metric(
            label="📈 Profit Factor", 
            value=stats.get("profit_factor", "0.00")
        )

        st.markdown("---")

        # NEW INTEGRATION BLOCK: Financial performance metrics matrix breakdown (USD)
        st.markdown("### 💵 Financial Performance (USD)")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)

        # Formatted values extraction logic for clean conditional layout styling 
        net_val = stats.get("net_profit_usd", 0.0)
        net_color = "normal" if net_val >= 0 else "inverse"

        p_col1.metric(
            label="💰 Net Profit/Loss", 
            value=f"${net_val:,.2f}", 
            delta="Profitable" if net_val >= 0 else "Negative", 
            delta_color=net_color
        )
        p_col2.metric(
            label="🟢 Gross Profit", 
            value=f"${stats.get('gross_profit_usd', 0.0):,.2f}"
        )
        p_col3.metric(
            label="🔴 Gross Loss", 
            value=f"${stats.get('gross_loss_usd', 0.0):,.2f}"
        )

        # Use the fourth column slot for your clear original range window tracking notice
        with p_col4:
            st.caption("%📆 **Active Filter Window**")
            st.write(f"From: `{perf_start}`")
            st.write(f"To: `{perf_end}`")

        st.markdown("---")

        # Show inline info alert if the selected inner window returns blank data rows
        if history_df.empty:
            st.info(f"No trade records found between {perf_start} and {perf_end} to analyze performance.")
    # ====================================================================================


    # ====================================================================================
    # STEP 1: MARKET CRASH TECHNICAL FACTOR FETCH (RUNS ONCE ON TOP OF STREAMLIT LAYOUT)
    # ====================================================================================
    try:
        import datetime as crash_dt
        import time as sys_time

        # 1. Fetch current Volatility Technical Factor (VIXY) with explicit start boundaries
        # A lookback window of 5 days guarantees we fetch yesterday and today's daily bars across weekends
        vix_start_time = crash_dt.datetime.now() - crash_dt.timedelta(days=5)

        vix_req = StockBarsRequest(
            symbol_or_symbols="VIXY",
            timeframe=TimeFrame.Day,
            start=vix_start_time,
            feed=DataFeed.IEX
        )

        try:
            # Fetch raw structural bars and normalize the index payload
            vix_data = data_client.get_stock_bars(vix_req).df.reset_index()
        except Exception as api_err:
            # Fallback initialization mapping if network requests experience data delivery drops
            vix_data = pd.DataFrame()

        # --- START DYNAMIC DAILY PERCENTAGE CHANGE CALCULATIONS ---
        if not vix_data.empty and len(vix_data) >= 2:
            prev_vixy_close = float(vix_data['close'].iloc[-2])
            current_vix = float(vix_data['close'].iloc[-1])

            # Calculate daily shift percentage to assign to the status fields
            vixy_daily_change_pct = ((current_vix - prev_vixy_close) / prev_vixy_close) * 100
            vix_chg_str = f"{vixy_daily_change_pct:+.2f}%"

            # Set active target metric variable to point to the percentage shift instead of raw price
            vix_target_metric = vixy_daily_change_pct
        else:
            current_vix = float(vix_data['close'].iloc[-1]) if not vix_data.empty else 12.50
            vixy_daily_change_pct = 0.0
            vix_chg_str = "0.00%"
            vix_target_metric = 0.0 # Fallback to standard 0.0% change metric if bars look missing
        # --- END DYNAMIC DAILY PERCENTAGE CHANGE CALCULATIONS ---


        # 2. Fetch Multi-Index Benchmark Profiles (SPY, QQQ, IWM)
        benchmarks = ["SPY", "QQQ", "IWM"]
        bench_req = StockBarsRequest(
            symbol_or_symbols=benchmarks,
            timeframe=TimeFrame.Day,
            start=crash_dt.datetime.now() - crash_dt.timedelta(days=4),
            feed=DataFeed.IEX
        )
        bench_data = data_client.get_stock_bars(bench_req).df.reset_index()

        crash_reasons = []
        matrix_rows = []

        # Add VIX status data to row log collections using the updated percentage evaluation logic
        vix_status = "⚠️ CRASH" if vix_target_metric > cfg_vix_max else "🍏 SAFE"
        matrix_rows.append({
            "Technical Factor": "Volatility (VIXY)", 
            "Live Value": f"${current_vix:.2f}", 
            "Daily Chg %": vix_chg_str, 
            "Status": vix_status
        })

        # Verify high volatility regime against interactive slider value using the percentage logic
        if vix_target_metric > cfg_vix_max:
            crash_reasons.append(f"VIXY Spike ({vix_chg_str} > {cfg_vix_max:.1f}%)")

        # Global compilation dictionary so individual tickers in the loop below can read the status instantly
        st.session_state["global_market_risk_matrix"] = {
            "is_market_crashing": len(crash_reasons) > 0,
            "crash_reasons": crash_reasons,
            "matrix_rows": matrix_rows
        }
    except Exception as top_level_err:
        st.caption(f"⚠️ Critical Market-wide Matrix Extraction Failed: {top_level_err}")
        # Populate safe fallbacks to ensure rendering layout grid below doesn't experience crash errors
        vix_status = "🍏 SAFE"
        vix_chg_str = "0.00%"
        bench_data = pd.DataFrame()


    # ====================================================================================
    # STEP 2: HIGH-DENSITY PROGRESSIVE METRIC GRID (SHOWS EXACTLY ONCE AT THE TOP)
    # ====================================================================================
    with st.expander("📊 Market Risk Factor Metrics", expanded=False):

        # Create 4 columns for VIXY, SPY, QQQ, and IWM
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        # 1. Volatility Metric Card (VIXY Daily Percent Change vs Max Limit)
        with m_col1:
            # FIXED: vix_status is now guaranteed to exist
            vix_status_icon = "🟢" if "SAFE" in vix_status else "🔴"
            st.metric(
                label=f"{vix_status_icon} Volatility (VIXY)", 
                value=vix_chg_str, # Displays the live daily return string (e.g. +4.25%)
                delta=f"Max Limit: +{cfg_vix_max:.1f}%",
                delta_color="normal" if "SAFE" in vix_status else "inverse"
            )

        # 2. Extract and parse index daily performances dynamically from your data
        for ticker, col_target in zip(["SPY", "QQQ", "IWM"], [m_col2, m_col3, m_col4]):
            if not bench_data.empty and ticker in bench_data['symbol'].values:
                ticker_df = bench_data[bench_data['symbol'] == ticker]
                if len(ticker_df) >= 2:
                    p_close = float(ticker_df['close'].iloc[-2])
                    c_price = float(ticker_df['close'].iloc[-1])
                    ret_pct = ((c_price - p_close) / p_close) * 100

                    idx_icon = "🍏" if ret_pct > cfg_index_drop else "🚨"
                    with col_target:
                        st.metric(
                            label=f"{idx_icon} {ticker} Benchmark",
                            value=f"${c_price:,.2f}",
                            delta=f"{ret_pct:+.2f}% (Limit: {cfg_index_drop:+.1f}%)",
                            delta_color="normal" if ret_pct > cfg_index_drop else "inverse"
                        )
                else:
                    with col_target:
                        st.metric(label=f"🔄 {ticker}", value="Loading...")
            else:
                with col_target:
                    st.metric(label=f"🔄 {ticker}", value="Loading...")

    # ====================================================================================    

    # Circuit Breakers
    p_hit = daily_pnl >= st.session_state.global_profit_goal
    l_hit = daily_pnl <= -abs(st.session_state.daily_loss_limit)

    bot_reason = ""
    if p_hit and st.session_state.run_bot:
        bot_reason = "PROFIT GOAL REACHED"
        trading_client.close_all_positions(cancel_orders=True)
        # 1. Clear the active widget key from memory to break the Streamlit lock
        if "run_bot" in st.session_state:
            del st.session_state["run_bot"]

        # 2. Reinitialize the key back to a safe, default false state
        st.session_state["run_bot"] = False; save_settings(); st.rerun()
        add_log(f"🎯 Target Hit: ${daily_pnl:.2f}. Positions closed.")
    elif l_hit and st.session_state.run_bot:
        bot_reason = "LOSS LIMIT HIT"
        # 1. Clear the active widget key from memory to break the Streamlit lock
        if "run_bot" in st.session_state:
            del st.session_state["run_bot"]

        # 2. Reinitialize the key back to a safe, default false state
        st.session_state["run_bot"] = False; save_settings(); st.rerun()
        add_log(f"🛑 Loss Limit Hit: ${daily_pnl:.2f}. Bot stopped.")
    elif not market_open and not st.session_state.allow_ext_hours:
        bot_reason = "MARKET CLOSED"

    active_now = st.session_state.run_bot and not bot_reason

    # --- FETCH ACCOUNT VALUE AND BUYING POWER METRICS ---
    try:
        if "ws_cash" in st.session_state and "ws_equity" in st.session_state:
            avail_balance = float(st.session_state["ws_cash"])
            portfolio_value = float(st.session_state["ws_equity"])
        else:
            # Pull real-time account data from Alpaca REST API as a secure fallback gate
            acct_profile = trading_client.get_account()

            # 'cash' or 'buying_power' represents what is available to deploy immediately
            avail_balance = float(acct_profile.cash)

            # 'equity' represents the sum of your cash plus current market values of positions
            portfolio_value = float(acct_profile.equity)

            # Seed the local state cache for future rapid layout draws
            st.session_state["ws_cash"] = avail_balance
            st.session_state["ws_equity"] = portfolio_value
    except Exception:
        # Fallback values preserve stability if both connections face a temporary failure
        avail_balance = st.session_state.get("ws_cash", 0.0)
        portfolio_value = st.session_state.get("ws_equity", 0.0)

    # Expand column template boundaries to a 5-column layout matrix
    m1, m_bal, m_port, m2, m3 = st.columns([1, 1, 1, 1, 1.2])

    # 1. Core performance tracking metric
    m1.metric("Daily PnL", f"${daily_pnl:.2f}", delta=f"{daily_pnl:.2f}")

    # 2. Display available cash spending balance card
    m_bal.metric("Available Balance", f"${avail_balance:,.2f}")

    # 3. Display total net equity value card
    m_port.metric("Portfolio Value", f"${portfolio_value:,.2f}")

    # 4. Exchange operations schedule monitor
    m2.metric("Market Status", "OPEN" if market_open else "CLOSED")

    # 5. Strategic status gate checks
    if bot_reason: 
        m3.error(f"🛑 {bot_reason}")
    else: 
        m3.success("🟢 BOT ACTIVE" if st.session_state.run_bot else "⚪ STANDBY")

    # Positions
    st.subheader("📊 Active Positions")
    pos = trading_client.get_all_positions()
    clock = trading_client.get_clock()
    held_symbols = {p.symbol for p in pos} # Essential for auto-execution check
    if pos:
        # --- START TABLE COLUMNS HEADER ---
        # Define header row outside or at the start of your positions iteration block
        # FIXED: Expanded layouts matrix from 6 to 7 columns to allocate room for Current Price
        #h1, h_price, h2, h3, h_tot, h_day, h4 = st.columns([1, 1, 1, 1, 1.2, 1.2, 0.5])
        h1, h_name, h_price, h2, h3, h_tot, h_day, h4 = st.columns([1, 1.8, 1, 1, 1, 1.2, 1.2, 0.5])
        h1.markdown("**Symbol**")
        h_name.markdown("**Asset Name**") # New Column Header
        h_price.markdown("**Current Price**") # New Column Header
        h2.markdown("**Market Value**")
        h3.markdown("**PnL %**")
        h_tot.markdown("**Total Gain**")
        h_day.markdown("**Daily Gain**")
        h4.markdown("**Close**")
        st.divider() # Creates a clean separation line under the headers
        # --- END TABLE COLUMNS HEADER ---
        # Loop through positions to handle UI and Virtual Monitoring
        for p in pos:
            # --- START VIRTUAL MONITORING LOGIC ---
            # If market is CLOSED, we manually monitor the trailing stop-loss
            if not clock.is_open:
                # ====================================================================================
                # HYBRID SYSTEM: WEBSOCKET REAL-TIME PRICE ROUTING WITH POSITION SNAPSHOT FALLBACK
                # ====================================================================================
                websocket_key = f"ws_latest_bar_{p.symbol}"

                # Primary Path: Pull the absolute freshest real-time price from the WebSocket memory pool
                if websocket_key in st.session_state and st.session_state[websocket_key] is not None:
                    current_price = float(st.session_state[websocket_key].get('close', p.current_price))
                # Secondary Backup Path: Fall back securely to the REST API position object property
                else:
                    current_price = float(p.current_price)
                # ====================================================================================

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
                            if ord_side == "sell" and order.status in ["filled", "partially_filled", "calculated"]:
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

                    # ====================================================================================
                    # AFTER-HOURS SESSION & 24-HOUR ELIGIBILITY FILTER
                    # ====================================================================================
                    try:
                        # Query Alpaca's master registry for asset details to verify attributes
                        asset_details = trading_client.get_asset(p.symbol)

                        # Check if the asset explicitly supports 24-hour trading attributes or fractional/overnight setups
                        # If it does not support extended execution setups or specific asset tracking, bypass ordering
                        is_24h_eligible = getattr(asset_details, 'fractionable', False) # Fallback heuristic or custom attribute if available

                        # Exclude ordering if the asset does not meet your specific 24-hour target tags
                        if not asset_details.tradable:
                            add_log(f"⏭️ Bypass After-Hour Order: {p.symbol} is marked as non-tradable.")
                            continue

                    except Exception as asset_err:
                        add_log(f"⚠️ Could not verify 24h/After-Hours properties for {p.symbol}: {asset_err}")
                    # ====================================================================================

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

            # --- START CURRENT PRICE FETCH ---
            # Hybrid Approach applied to UI: reads stream if available, falls back to raw position attribute
            websocket_ui_key = f"ws_latest_bar_{p.symbol}"
            if websocket_ui_key in st.session_state and st.session_state[websocket_ui_key] is not None:
                live_ticker_price = float(st.session_state[websocket_ui_key].get('close', p.current_price))
            else:
                live_ticker_price = float(p.current_price) if hasattr(p, 'current_price') else 0.0
            # --- END CURRENT PRICE FETCH ---


            # ====================================================================================
            # OPTIMIZED LOCAL MEMORY ASSET LOOKUP
            # ====================================================================================
            # Check if this ticker's company name is already stored in our cache
            if p.symbol not in st.session_state["cached_asset_names"]:
                try:
                    # Query Alpaca's master registry for the data structure
                    asset_details = trading_client.get_asset(p.symbol)
                    if asset_details and hasattr(asset_details, 'name') and asset_details.name:
                        st.session_state["cached_asset_names"][p.symbol] = asset_details.name
                    else:
                        st.session_state["cached_asset_names"][p.symbol] = "Asset Profile"
                except Exception:
                    # Temporary fallback if API rate limits hit; will retry on next UI update
                    st.session_state["cached_asset_names"][p.symbol] = f"{p.symbol} Stock"

            # Read name directly from local super-fast session state dictionary
            full_asset_name = st.session_state["cached_asset_names"][p.symbol]
            # ====================================================================================

            # Dynamic UI based on Side
            current_side = getattr(p, 'side', 'long').lower()
            side_icon = "🔴" if current_side == 'short' else "🟢"
            pnl_color = "red" if pnl_pct < 0 else "green"

            # --- START GAIN CALCULATIONS ---
            # Fetch total and daily dollar profit/loss from Alpaca properties
            total_gain = float(getattr(p, 'unrealized_pl', 0.0))
            daily_gain = float(getattr(p, 'unrealized_intraday_pl', 0.0))

            # Dynamically assign styling colors based on profit values
            tot_color = "red" if total_gain < 0 else "green"
            day_color = "red" if daily_gain < 0 else "green"

            # Create formatting prefix strings (+ or -)
            tot_prefix = "+" if total_gain >= 0 else ""
            day_prefix = "+" if daily_gain >= 0 else ""
            # --- END GAIN CALCULATIONS ---

            # Layout columns adjusted to safely fit the new performance columns
            # FIXED: Aligned structure columns definition matching the header configuration above
            #c1, c_price, c2, c3, c_tot, c_day, c4 = st.columns([1, 1, 1, 1, 1.2, 1.2, 0.5])
            c1, c_name, c_price, c2, c3, c_tot, c_day, c4 = st.columns([1, 1.8, 1, 1, 1, 1.2, 1.2, 0.5])

            c1.write(f"{side_icon} **{p.symbol}**")

            # Output the company description name string from cache
            c_name.write(f"{full_asset_name}")

            # 1. Print the Live Ticker Price value token
            c_price.write(f"${live_ticker_price:,.2f}")

            c2.write(f"${mkt_val:,.0f}")

            # Styled PnL for better visibility
            c3.markdown(f":{pnl_color}[{pnl_pct:.2f}%]")

            # 1. Display total overall gain column
            c_tot.markdown(f":{tot_color}[{tot_prefix}${total_gain:,.2f}]")

            # 2. Display daily dynamic intraday gain column
            c_day.markdown(f":{day_color}[{day_prefix}${daily_gain:,.2f}]")

            if c4.button("✖", key=f"cl_{p.symbol}", disabled=admin_disabled):
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
            # ====================================================================================
            # HYBRID DATA PIPELINE: WEBSOCKET DRIVEN WITH AUTOMATIC REST API FALLBACK
            # ====================================================================================
            state_key = f"historical_df_{s}"
            websocket_key = f"ws_latest_bar_{s}" # Expected key where your background WebSocket thread saves incoming live bars

            # 1. INITIALIZATION BASELINE: Fetch 30 days of history exactly once via REST API
            if state_key not in st.session_state:
                start_time = datetime.now() - timedelta(days=30)
                try:
                    init_df = data_client.get_stock_bars(StockBarsRequest(
                        symbol_or_symbols=s, 
                        timeframe=TimeFrame.Minute, 
                        start=start_time, 
                        feed=DataFeed.IEX
                    )).df.reset_index()

                    if not init_df.empty:
                        st.session_state[state_key] = init_df
                    else:
                        add_log(f"⚠️ Initial REST fetch returned no data for {s}. Market might be closed.")
                        continue
                except Exception as init_err:
                    add_log(f"❌ Failed to load initial baseline for {s}: {init_err}")
                    continue

            # 2. HYBRID LIVE UPDATE: Check WebSocket stream first, fallback to REST API if necessary
            else:
                # Primary Path: Pull the latest real-time bar directly from the background WebSocket stream
                if websocket_key in st.session_state and st.session_state[websocket_key] is not None:
                    ws_bar_dict = st.session_state[websocket_key]
                    latest_bars_df = pd.DataFrame([ws_bar_dict])

                    # Clear the WebSocket memory buffer slot so we don't process the exact same bar twice
                    st.session_state[websocket_key] = None 

                # Secondary Backup Path: If WebSocket has no bar, poll a small REST snapshot as a fail-safe
                else:
                    recent_start = datetime.now() - timedelta(minutes=10)
                    try:
                        latest_bars_df = data_client.get_stock_bars(StockBarsRequest(
                            symbol_or_symbols=s, 
                            timeframe=TimeFrame.Minute, 
                            start=recent_start, 
                            feed=DataFeed.IEX
                        )).df.reset_index()
                    except Exception as api_err:
                        latest_bars_df = pd.DataFrame() # Create empty dataframe to safely bypass concat step
                        st.caption(f"⚠️ Both streaming and REST backup failed for {s}: {api_err}")

                # Merge any found live update (WebSocket or REST backup) into our master training matrix
                if not latest_bars_df.empty:
                    combined_df = pd.concat([st.session_state[state_key], latest_bars_df])
                    combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

                    # Prevent memory bloating over long periods by keeping exactly the latest 30 days of data (approx 12,000 bars)
                    st.session_state[state_key] = combined_df.iloc[-12000:]

            # FINAL VALIDATION CHECK: Block empty arrays from entering RandomForestClassifier
            if state_key not in st.session_state or st.session_state[state_key].empty:
                st.caption(f"🔄 Syncing live structural matrix streams for {s}...")
                continue 

            df = st.session_state[state_key]
            # ====================================================================================


            #ai_conf, conf_hist, feat_map = get_ai_prediction(df, s)
            ai_dir, ai_conf, conf_hist, feat_map = get_ai_prediction(df, s)
            price = float(df['close'].iloc[-1])

            # --- START CACHED NAME LOOKUP FOR WATCHLIST SYMBOL ---
            if "cached_asset_names" not in st.session_state:
                st.session_state["cached_asset_names"] = {}

            if s not in st.session_state["cached_asset_names"]:
                try:
                    asset_details = trading_client.get_asset(s)
                    if asset_details and hasattr(asset_details, 'name') and asset_details.name:
                        st.session_state["cached_asset_names"][s] = asset_details.name
                    else:
                        st.session_state["cached_asset_names"][s] = f"{s} Asset"
                except Exception:
                    st.session_state["cached_asset_names"][s] = f"{s} Stock"

            full_asset_name = st.session_state["cached_asset_names"][s]
            # --- END CACHED NAME LOOKUP FOR WATCHLIST SYMBOL ---


            # Layout columns
            s1, s_name, s2, s3, s4, s5 = st.columns([1, 1.8, 1, 1.5, 2, 1])

            # s1.write(f"**{s}**")
            p_count = pending_counts.get(s, 0)
            if p_count > 0:
                s1.write(f"**{s}** :orange[({p_count} Pending)]")
            else:
                s1.write(f"**{s}** ")

            # Render the cached company description asset name string
            s_name.write(f"**{full_asset_name}**")

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
            if s5.button("Buy", key=f"b_{s}", disabled=admin_disabled):
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

                # ====================================================================================
                # INSERT POINT A: MULTI-INDEX TECHNICAL CRASH PROTECTION WITH UI CONTROLS & BYPASS
                # ====================================================================================
                is_market_crashing = False
                try:
                    st.session_state["risk_matrix_last_update"] = sys_time.time()

                    # Read fast-cached structural values directly to pass execution barriers
                    cached_risk = st.session_state["global_market_risk_matrix"]
                    crash_reasons = cached_risk["crash_reasons"]
                    # ================================================================================

                    # Render live structural metric table directly inside the main UI container view 
                    #with st.expander("📊 Live Market Risk Factor Matrix", expanded=False):
                     #   st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

                    # 3. Evaluate Global Circuit Breaker State & Recovery Timer Tracking
                    if crash_reasons:
                        if not breaker_bypass:
                            is_market_crashing = True
                            # Store timestamp of the breach event to track recovery cooling periods
                            st.session_state["last_crash_timestamp"] = sys_time.time()

                            st.error(f"🚨 GLOBAL CIRCUIT BREAKER TRIPPED: {', '.join(crash_reasons)}")

                            # Emergency Capital Safeguard: Purge exposures immediately
                            trading_client.cancel_orders()
                            trading_client.close_all_positions(cancel_orders=True)
                        else:
                            st.info("ℹ️ Circuit Breaker Tripped, but Bypass is active.")
                    else:
                        # If conditions are healthy, check if we are still cooling down from a recent crash
                        if "last_crash_timestamp" in st.session_state:
                            elapsed_seconds = sys_time.time() - st.session_state["last_crash_timestamp"]
                            remaining_minutes = int((1800 - elapsed_seconds) / 60)

                            if elapsed_seconds < 1800 and not breaker_bypass:  # 1800 seconds = 30 minutes rest period
                                is_market_crashing = True
                                st.warning(f"⏳ Post-Crash Safety Cooling: Resuming in {remaining_minutes} mins.")
                            else:
                                # Clean up state indicator once the 30 minute lock cleanly expires or bypass trips it
                                del st.session_state["last_crash_timestamp"]
                                st.success("🍏 Recovery Completed: Bot re-armed.")

                except Exception as crash_err:
                    st.caption(f"⚠️ Risk Framework temporary bypass: {crash_err}")

                # --- COOL-DOWN GATE FOR CONSECUTIVE LOSSES (15 MINUTE LOCK) ---
                is_cooled_down = False
                try:
                    # Request the last 2 closed/filled orders for this specific symbol
                    order_filter = GetOrdersRequest(
                        status=QueryOrderStatus.CLOSED,
                        symbols=[s],
                        limit=2,
                        direction='desc'  # FIX: Replaced Sort.DESC with raw string value
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

                # ====================================================================================
                # INSERT POINT B: MODIFIED OPERATIONAL GATE WAY
                # ====================================================================================
                if not is_cooled_down and not is_market_crashing:
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
                elif is_market_crashing:
                    st.caption(f"🛑 Bot Blocked: Order routing blocked by multi-index crash constraints.")

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

    # Fetch current pending data (Instantly pulls from fast local memory under the hybrid approach)
    pending_df = get_pending_orders_df(trading_client) # Pass your active Alpaca trading client instance

    if not pending_df.empty:
        # ====================================================================================
        # HYBRID SYSTEM: FAST-MEMORY VISUAL SYNC TIMESTAMPS
        # ====================================================================================
        # Reads the exact moment your hybrid background cache layer last hit the REST API
        if "pending_orders_last_update" in st.session_state:
            import datetime as status_dt
            last_sync_ts = status_dt.datetime.fromtimestamp(st.session_state["pending_orders_last_update"])
            last_sync_str = last_sync_ts.strftime("%I:%M:%S %p")
            st.caption(f"⚡ *Displaying cached orders matrix (Last verified live with Alpaca at {last_sync_str})*")
        # ====================================================================================

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
        # Decouple native references by forcing the Central Timezone (CST/CDT)
        import datetime as ui_dt

        # Pull current moment localized explicitly to Chicago/Central time
        current_now = ui_dt.datetime.now(ZoneInfo("America/Chicago"))
        thirty_days_ago = current_now - ui_dt.timedelta(days=30)

        # Date range selector window using localized date objects
        date_range = st.date_input(
            "📅 Execution Date Range (CST)", 
            value=[thirty_days_ago.date(), current_now.date()]
        )

        # 🟢 FIXED: Safety clamp protects your historical parsing block from crashing 
        # while the user is actively clicking or changing dates on the calendar
        if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
            st.stop() # Silently pauses layout draw until the user selects the second date


    # --- FIXED TRANSITIONAL DATE PARSING ---
    start_dt, end_dt = None, None
    if isinstance(date_range, (list, tuple)) and len(date_range) > 0:
        start_dt = date_range[0]
        end_dt = date_range[1] if len(date_range) == 2 else date_range[0]
    # --- END FIXED TRANSITIONAL DATE PARSING ---

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
                    key="btn_export_filtered_history", disabled=admin_disabled
                )

            # ====================================================================================
            # HYBRID SYSTEM: FAST-MEMORY VISUAL CACHE MONITOR STATUS OVERLAY
            # ====================================================================================
            # Dynamically checks if this specific search configuration key matches the cached memory stamp
            param_cache_identity = f"history_cache_100_{start_dt}_{end_dt}"
            if param_cache_identity in st.session_state:
                st.caption("⚡ *Displaying cached historical ledger data (Loads in 0ms without hitting API network lag)*")
            else:
                st.caption("📡 *Displaying direct live REST API history payload data*")
            # ====================================================================================

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
        time.sleep(0.1) # 0.6s * 100 = 60 seconds
        prog_placeholder.progress(percent_complete + 1, text=f"Next update in {10 - int(percent_complete*0.1)}s")
live_ui()
