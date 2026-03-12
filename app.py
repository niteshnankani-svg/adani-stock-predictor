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

st.set_page_config(page_title="Adani Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Adani Enterprises — Stock Prediction Dashboard")
st.markdown("**XGBoost + News Sentiment + Technical Indicators**")
st.markdown("---")

@st.cache_resource
def load_model():
    m = joblib.load("xgboost_model.pkl")
    with open("feature_cols.json", "r") as f:
        fc = json.load(f)
    return m, fc

model, feature_cols = load_model()
analyzer = SentimentIntensityAnalyzer()

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
            "query": "Adani Enterprises",
            "mode": "artlist", "format": "json",
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

with st.spinner("Fetching live data..."):
    df_stock = fetch_stock()
    news_articles = fetch_news()
    df_feat = compute_features(df_stock, news_articles)

st.sidebar.header("🔧 Model Info")
st.sidebar.write("**Model:** XGBoost")
st.sidebar.write(f"**Features:** {len(feature_cols)}")
st.sidebar.write("**Training Accuracy:** 56.9%")

col1, col2, col3, col4 = st.columns(4)
latest = df_stock.iloc[-1]
prev = df_stock.iloc[-2]
change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
col1.metric("Current Price", f"₹{latest['Close']:,.2f}", f"{change_pct:+.2f}%")
col2.metric("Day High", f"₹{latest['High']:,.2f}")
col3.metric("Day Low", f"₹{latest['Low']:,.2f}")
col4.metric("Volume", f"{latest['Volume']:,.0f}")

st.markdown("---")
st.header("🤖 Tomorrow\'s Prediction")

last_row = df_feat.iloc[-1:]
for c in feature_cols:
    if c not in last_row.columns:
        last_row[c] = 0
X_pred = last_row[feature_cols]
pred = model.predict(X_pred)[0]
proba = model.predict_proba(X_pred)[0]

if pred == 1:
    st.success(f"📈 **PREDICTION: STOCK GOES UP** — Confidence: {proba[1]*100:.1f}%")
else:
    st.error(f"📉 **PREDICTION: STOCK GOES DOWN** — Confidence: {proba[0]*100:.1f}%")

st.caption(f"Based on {len(news_articles)} recent news articles & technical indicators as of {df_stock.index[-1].strftime('%d %b %Y')}")

st.markdown("---")
st.header("📊 Stock Price — Last 6 Months")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df_stock.index, open=df_stock["Open"],
    high=df_stock["High"], low=df_stock["Low"], close=df_stock["Close"], name="ADANIENT"))
fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("📰 Recent News & Sentiment")
if news_articles:
    for a in news_articles[:10]:
        title = a.get("title", "")
        url = a.get("url", "")
        score = analyzer.polarity_scores(title)["compound"]
        emoji = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
        st.markdown(f"{emoji} **{title[:120]}** ([link]({url})) — Sentiment: `{score:.2f}`")
else:
    st.info("No recent news found")

st.markdown("---")
st.header("🔧 Top Features Driving Prediction")
imp = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
imp = imp.sort_values("Importance", ascending=False).head(10)
st.bar_chart(imp.set_index("Feature"))

st.markdown("---")
st.caption("⚠️ Research project only — NOT financial advice. Markets are unpredictable.")
st.caption("Built by Nitesh | XGBoost + ChromaDB + Vectorless RAG + GDELT + yfinance")
