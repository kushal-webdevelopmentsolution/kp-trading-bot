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
from sklearn.model_selection import cross_val_score

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

st.set_page_config(page_title="AI Pro Trader", layout="wide")

if "tickers" not in st.session_state:
    st.session_state.tickers = ["SPY","AMZN", "NVDA", "TSLA"]
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- RISK ENGINE (FIXED FOR MARKET CLOSED) ---
def get_correlation_matrix(tickers):
    try:
        series_list = []
        # Use a longer window (90 days) to ensure we get data even if market is closed
        start_date = datetime.now() - timedelta(days=90)
        end_time = datetime.now() - timedelta(minutes=15)

        for symbol in tickers:
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                   start=start_date, end=end_time, feed=DataFeed.IEX)
            df = data_client.get_stock_bars(req).df.reset_index()
            # Align by date
            s = df.set_index('timestamp')['close'].rename(symbol)
            series_list.append(s)

        # Inner Join ensures we only correlate days where ALL stocks have data
        combined_df = pd.concat(series_list, axis=1, join='inner')
        return combined_df.corr()
    except Exception as e:
        return pd.DataFrame()

# --- SENTIMENT ENGINE ---
def get_sentiment_score(symbol):
    try:
        req = NewsRequest(symbols=symbol, limit=5)
        news = news_client.get_news(req).news
        if not news: return 0.0
        pos = ['up', 'upgrade', 'buy', 'growth', 'bullish', 'profit', 'beat', 'high']
        neg = ['down', 'downgrade', 'sell', 'risk', 'bearish', 'loss', 'miss', 'low']
        score = 0
        for article in news:
            text = article.headline.lower()
            score += sum(1 for w in pos if word in text)
            score -= sum(1 for w in neg if word in text)
        return max(min(score / 5, 1.0), -1.0)
    except: return 0.0

# --- CORE AI ENGINE ---
def scan_ticker(symbol, run_bot_active, order_mode, order_val):
    try:
        end_time = datetime.now() - timedelta(minutes=15)
        req_data = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                    start=datetime.now()-timedelta(days=1000), end=end_time, feed=DataFeed.IEX)
        df = data_client.get_stock_bars(req_data).df.reset_index()

        df.ta.rsi(length=14, append=True); df.ta.bbands(length=20, append=True); df.ta.mfi(length=14, append=True)
        for lag in [1, 2, 3]:
            df[f'rsi_lag_{lag}'] = df['RSI_14'].shift(lag)
            df[f'price_lag_{lag}'] = df['close'].shift(lag)

        sentiment = get_sentiment_score(symbol)
        df['sentiment'] = sentiment
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()

        cols = [c for c in df.columns if any(x in c.upper() for x in ['RSI', 'BBL', 'BBU', 'MFI', 'LAG', 'SENTIMENT'])]
        X, y = df[cols][:-1], df['target'][:-1]

        model = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ], voting='soft')

        scores = cross_val_score(model, X, y, cv=3)
        win_rate = scores.mean()
        model.fit(X, y)

        up_prob = float(model.predict_proba(df[cols].tail(1))[0][1])
        price = float(df['close'].iloc[-1])
        qty = float(order_val if order_mode == "Shares" else round(order_val / price, 2))

        if run_bot_active and up_prob >= 0.90:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=round(price*1.04, 2)),
                stop_loss=StopLossRequest(stop_price=round(price*0.98, 2))
            ))
            st.session_state.logs.append(f"{datetime.now().strftime('%H:%M')} | 🤖 AUTO BUY: {symbol} @ {price:.2f}")

        return {"symbol": symbol, "price": price, "prob": up_prob, "win_rate": win_rate, "sent": sentiment, "df": df, "qty": qty}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# --- UI ---
st.title("🚀 AI Terminal: Technical + Sentiment Dashboard")

st.sidebar.header("🛡️ Control Panel")
run_bot = st.sidebar.toggle("Activate 90% Bot")
order_mode = st.sidebar.radio("Sizing:", ["Shares", "USD Value"])
order_val = st.sidebar.number_input("Amount:", min_value=0.01, value=1.0 if order_mode=="Shares" else 100.0)

@st.fragment(run_every=30)
def trading_dashboard():
    col1, col2 = st.columns([3.5, 1])
    with col1:
        try:
            acc = trading_client.get_account()
            st.columns(2)[0].metric("PORTFOLIO", f"${float(acc.equity):,.2f}")
            st.columns(2)[1].metric("DAILY PnL", f"${float(acc.equity)-float(acc.last_equity):.2f}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(scan_ticker, s, run_bot, order_mode, order_val) for s in st.session_state.tickers]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            st.subheader("⚡ Signal Matrix")
            h = st.columns([1, 1, 1.2, 1, 1, 1])
            col_names = ["SYMBOL", "PRICE", "AI CONFIDENCE", "SENTIMENT", "HIST. WIN", "ACTION"]
            for i, name in enumerate(col_names): h[i].caption(name)

            best_ticker = None
            max_conf = 0
            for res in results:
                if "error" in res: continue
                s, price, prob, win_r, sent, qty = res['symbol'], res['price'], res['prob'], res['win_rate'], res['sent'], res['qty']
                if prob > max_conf: max_conf, best_ticker = prob, (s, res['df'])

                r = st.columns([1, 1, 1.2, 1, 1, 1])
                r[0].write(f"**{s}**")
                r[1].write(f"${price:.2f}")
                r[2].progress(prob, text=f"{prob*100:.0f}%")

                sc = "green" if sent > 0 else "red" if sent < 0 else "gray"
                sl = "Bullish" if sent > 0 else "Bearish" if sent < 0 else "Neutral"
                r[3].markdown(f":{sc}[{sl}]")

                r[4].write(f"{win_r*100:.1f}%")
                if r[5].button(f"Buy {qty}", key=f"b_{s}"):
                    st.session_state.logs.append(f"{datetime.now().strftime('%H:%M')} | 👤 MAN BUY: {s}")

            # --- VISUAL ANALYSIS (Risk Matrix and Chart) ---
            st.markdown("---")
            chart_col, risk_col = st.columns([1.5, 1])

            with chart_col:
                if best_ticker:
                    st.subheader(f"📊 Trajectory: {best_ticker[0]}")
                    st.line_chart(best_ticker[1][['close']].tail(50))

            with risk_col:
                st.subheader("🔗 Risk Matrix")
                corr = get_correlation_matrix(st.session_state.tickers)
                if not corr.empty:
                    # Clean display for Cloud
                    st.dataframe(corr.style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)
                else:
                    st.info("Market Closed: Waiting for historical sync...")

        except Exception as e: st.error(f"UI Error: {e}")

    with col2:
        st.subheader("📜 Activity Log")
        log_text = "\n".join(st.session_state.logs[-15:])
        st.text_area("Log", value=log_text, height=650, label_visibility="collapsed")

trading_dashboard()
