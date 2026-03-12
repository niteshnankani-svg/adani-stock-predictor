import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

st.set_page_config(page_title="Adani Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Adani Enterprises — AI Stock Analysis Dashboard")
st.markdown("**Powered by: XGBoost ML Model + Vectorless RAG (Annual Reports) + News Sentiment + LLM Analysis**")
st.markdown("---")

@st.cache_resource
def load_model():
    m = joblib.load("xgboost_model.pkl")
    with open("feature_cols.json", "r") as f:
        fc = json.load(f)
    return m, fc

@st.cache_resource
def load_annual_insights():
    try:
        with open("annual_insights.json", "r") as f:
            return json.load(f)
    except:
        return {}

model, feature_cols = load_model()
annual_insights = load_annual_insights()
analyzer = SentimentIntensityAnalyzer()

st.sidebar.header("⚙️ Settings")
openai_key = st.sidebar.text_input("OpenAI API Key (for AI Analyst)", type="password")
st.sidebar.markdown("---")
st.sidebar.header("📊 About This Dashboard")
st.sidebar.markdown("""
**Data Sources Used:**
- 📈 Live stock prices (yfinance)
- 📰 Live news + sentiment (GDELT + VADER)
- 📄 3 Annual Reports FY22-FY24 (Vectorless RAG)
- 🤖 XGBoost ML Model (56.9% accuracy)
- 🧠 GPT-4o-mini for analysis

**Built by Nitesh Nankani**
""")

@st.cache_data(ttl=3600)
def fetch_stock():
    t = yf.Ticker("ADANIENT.NS")
    df = t.history(period="6mo")[["Open","Close","High","Low","Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

@st.cache_data(ttl=3600)
def fetch_news():
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    try:
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params={
            "query": "Adani Enterprises", "mode": "artlist", "format": "json",
            "startdatetime": week_ago.strftime("%Y%m%d") + "000000",
            "enddatetime": today.strftime("%Y%m%d") + "235959",
            "maxrecords": 50, "sort": "DateDesc"
        }, timeout=15)
        if r.text.strip():
            return r.json().get("articles", [])
    except:
        pass
    return []

def compute_features(df, news):
    df = df.copy()
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    for lag in [1,2,3,5,7]:
        df[f"Return_Lag_{lag}"] = df["Daily_Return"].shift(lag)
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)
    df["High_Low_Range"] = (df["High"] - df["Low"]) / (df["Low"] + 0.001)
    df["Close_Position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 0.001)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA5_above_MA20"] = (df["MA_5"] > df["MA_20"]).astype(int)
    df["Price_vs_MA5"] = df["Close"] / df["MA_5"]
    df["Price_vs_MA20"] = df["Close"] / df["MA_20"]
    df["Volatility_5"] = df["Daily_Return"].rolling(5).std()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Volume_MA5"] = df["Volume"].rolling(5).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA5"]
    df["Momentum_3"] = df["Close"].pct_change(3)
    df["Momentum_5"] = df["Close"].pct_change(5)
    df["Momentum_10"] = df["Close"].pct_change(10)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 0.001)))
    df["Up_Day"] = (df["Close"] > df["Open"]).astype(int)
    df["Consecutive_Up"] = df["Up_Day"].rolling(5).sum()
    sents = [analyzer.polarity_scores(a.get("title",""))["compound"] for a in news]
    avg_s = np.mean(sents) if sents else 0
    df["news_count"] = len(news)
    df["avg_sentiment"] = avg_s
    df["negative_count"] = sum(1 for s in sents if s < -0.05)
    df["positive_count"] = sum(1 for s in sents if s > 0.05)
    for lag in [1,2,3]:
        df[f"sentiment_lag_{lag}"] = avg_s
        df[f"news_count_lag_{lag}"] = len(news)
    df["sentiment_MA3"] = avg_s
    df["sentiment_MA5"] = avg_s
    return df

with st.spinner("Fetching live market data..."):
    df_stock = fetch_stock()
    news_articles = fetch_news()
    df_feat = compute_features(df_stock, news_articles)

current = df_stock.iloc[-1]
prev = df_stock.iloc[-2]
change_pct = (current["Close"] - prev["Close"]) / prev["Close"] * 100
recent30 = df_stock.tail(30)
avg_move = abs(recent30["Close"] - recent30["Open"]).mean()
daily_rets = (recent30["Close"] / recent30["Close"].shift(1) - 1).dropna()
vol_1sd = daily_rets.std() * current["Close"]

pivot = (current["High"] + current["Low"] + current["Close"]) / 3
r1 = (2 * pivot) - current["Low"]
s1 = (2 * pivot) - current["High"]
r2 = pivot + (current["High"] - current["Low"])
s2 = pivot - (current["High"] - current["Low"])
ma5 = df_stock["Close"].rolling(5).mean().iloc[-1]
ma20 = df_stock["Close"].rolling(20).mean().iloc[-1]
d = df_stock["Close"].diff()
g = d.clip(lower=0).rolling(14).mean()
l = (-d.clip(upper=0)).rolling(14).mean()
rsi = (100 - (100 / (1 + g / (l + 0.001)))).iloc[-1]

# ===== CURRENT PRICE =====
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Current Price", f"₹{current['Close']:,.2f}", f"{change_pct:+.2f}%")
col2.metric("📈 Day High", f"₹{current['High']:,.2f}")
col3.metric("📉 Day Low", f"₹{current['Low']:,.2f}")
col4.metric("📊 Volume", f"{current['Volume']:,.0f}")

st.markdown("---")

# ===== PREDICTION =====
st.header("🔮 Tomorrow's Prediction & Price Levels")

last_row = df_feat.iloc[-1:]
for c in feature_cols:
    if c not in last_row.columns:
        last_row[c] = 0
X_pred = last_row[feature_cols]
pred = model.predict(X_pred)[0]
proba = model.predict_proba(X_pred)[0]

if pred == 1:
    target = current["Close"] + avg_move
    pred_high = current["Close"] + avg_move * 1.5
    pred_low = current["Close"] - avg_move * 0.5
    conf = proba[1] * 100
    st.success(f"📈 **MODEL PREDICTION: STOCK GOES UP** — Confidence: {conf:.1f}%")
else:
    target = current["Close"] - avg_move
    pred_high = current["Close"] + avg_move * 0.5
    pred_low = current["Close"] - avg_move * 1.5
    conf = proba[0] * 100
    st.error(f"📉 **MODEL PREDICTION: STOCK GOES DOWN** — Confidence: {conf:.1f}%")

p1, p2, p3 = st.columns(3)
p1.metric("🟢 Predicted High", f"₹{pred_high:,.2f}")
p2.metric("🎯 Target Price", f"₹{target:,.2f}")
p3.metric("🔴 Predicted Low", f"₹{pred_low:,.2f}")

st.markdown("---")

# ===== PRICE LEVELS =====
st.header("📐 Key Price Levels for Tomorrow")

lc1, lc2, lc3 = st.columns(3)

with lc1:
    st.subheader("Pivot Points")
    st.markdown(f"""
    | Level | Price |
    |-------|-------|
    | **R2** | ₹{r2:,.2f} |
    | **R1** | ₹{r1:,.2f} |
    | **Pivot** | ₹{pivot:,.2f} |
    | **S1** | ₹{s1:,.2f} |
    | **S2** | ₹{s2:,.2f} |
    """)

with lc2:
    st.subheader("Support & Resistance")
    st.markdown(f"""
    | Level | Price |
    |-------|-------|
    | **Resistance (50d)** | ₹{df_stock.tail(50)['High'].max():,.2f} |
    | **Resistance (20d)** | ₹{df_stock.tail(20)['High'].max():,.2f} |
    | **Support (20d)** | ₹{df_stock.tail(20)['Low'].min():,.2f} |
    | **Support (50d)** | ₹{df_stock.tail(50)['Low'].min():,.2f} |
    """)

with lc3:
    st.subheader("Technical Indicators")
    rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
    st.markdown(f"""
    | Indicator | Value |
    |-----------|-------|
    | **5-Day MA** | ₹{ma5:,.2f} |
    | **20-Day MA** | ₹{ma20:,.2f} |
    | **RSI (14)** | {rsi:.1f} {rsi_status} |
    | **Volatility** | ±₹{vol_1sd:,.2f} |
    | **Avg Daily Move** | ₹{avg_move:,.2f} |
    """)

st.markdown("---")

# ===== CHART =====
st.header("📊 Candlestick Chart with Levels")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df_stock.index, open=df_stock["Open"],
    high=df_stock["High"], low=df_stock["Low"], close=df_stock["Close"], name="ADANIENT"))
fig.add_hline(y=pivot, line_dash="dash", line_color="yellow", annotation_text="Pivot")
fig.add_hline(y=r1, line_dash="dot", line_color="green", annotation_text="R1")
fig.add_hline(y=s1, line_dash="dot", line_color="red", annotation_text="S1")
fig.add_hline(y=ma5, line_dash="solid", line_color="cyan", annotation_text="MA5")
fig.add_hline(y=ma20, line_dash="solid", line_color="orange", annotation_text="MA20")
fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark",
    title="Adani Enterprises (NSE) — 6 Month Chart with Key Levels")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ===== ANNUAL REPORTS =====
st.header("📄 Annual Report Insights (Extracted via Vectorless RAG)")
st.caption("These insights were extracted using our custom tree-index reasoning system — same concept as PageIndex")

if annual_insights:
    tabs = st.tabs(["📑 FY2022", "📑 FY2023 (Hindenburg Year)", "📑 FY2024"])
    for tab, fy in zip(tabs, ["FY2022", "FY2023", "FY2024"]):
        with tab:
            if fy in annual_insights:
                st.markdown(annual_insights[fy])
            else:
                st.info(f"No data for {fy}")
else:
    st.info("Annual report insights not loaded")

st.markdown("---")

# ===== LLM ANALYST =====
st.header("🧠 Ask the AI Analyst")
st.caption("Uses ALL data sources: Stock prices + News + Annual Reports + ML Model")

if openai_key:
    question = st.text_input("Ask anything about Adani Enterprises:",
        placeholder="e.g., Should I be worried about debt levels? How did Hindenburg affect financials?")

    if question:
        with st.spinner("AI Analyst is thinking..."):
            client = OpenAI(api_key=openai_key)
            news_ctx = "\n".join([f"- {a.get('title','')[:100]} (sentiment: {analyzer.polarity_scores(a.get('title',''))['compound']:.2f})" for a in news_articles[:10]])
            ar_ctx = json.dumps(annual_insights, indent=2)[:4000] if annual_insights else "N/A"

            full_context = f"""LIVE DATA (as of {df_stock.index[-1].strftime('%d %b %Y')}):
Price: ₹{current['Close']:,.2f} | Change: {change_pct:+.2f}%
RSI: {rsi:.1f} | MA5: ₹{ma5:,.2f} | MA20: ₹{ma20:,.2f}
ML Prediction: {"UP" if pred==1 else "DOWN"} ({conf:.1f}% confidence)
Target: ₹{target:,.2f} | Range: ₹{pred_low:,.2f} - ₹{pred_high:,.2f}

RECENT NEWS:
{news_ctx}

ANNUAL REPORT DATA (FY2022-FY2024):
{ar_ctx}"""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst at Goldman Sachs specializing in Adani Group. "
                     "You have access to: live stock data, ML model predictions, 3 years of annual reports, and recent news with sentiment. "
                     "Give specific, data-backed analysis. Use exact numbers. Be honest about risks. "
                     "Always end with a clear, balanced assessment. This is NOT financial advice."},
                    {"role": "user", "content": f"Data:\n{full_context}\n\nAnalyze: {question}"}
                ],
                temperature=0.3
            )
            st.markdown("### 📊 Analysis")
            st.markdown(resp.choices[0].message.content)
else:
    st.warning("👈 Enter your OpenAI API key in the sidebar to enable the AI Analyst")

st.markdown("---")

# ===== NEWS =====
st.header("📰 This Week's News & Sentiment")
if news_articles:
    for a in news_articles[:15]:
        title = a.get("title", "")
        url = a.get("url", "")
        score = analyzer.polarity_scores(title)["compound"]
        emoji = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
        st.markdown(f"{emoji} **{title[:120]}** ([link]({url})) — `{score:.2f}`")
else:
    st.info("No news found for this week")

st.markdown("---")
st.header("🔧 Feature Importance — What Drives the Model?")
imp = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
imp = imp.sort_values("Importance", ascending=False).head(10)
st.bar_chart(imp.set_index("Feature"))

st.markdown("---")
st.caption("⚠️ This is a research project and NOT financial advice. Stock markets are unpredictable. Never invest based solely on model predictions.")
st.caption("Built by Nitesh Nankani | Tech: XGBoost + Vectorless RAG + ChromaDB + GDELT + VADER + OpenAI + Streamlit")
