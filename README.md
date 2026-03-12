# 📊 Adani Stock Predictor — Multi-Source AI Financial Analysis System

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://adani-stock-predictor-igjwntvgevqqn7wjq2et4y.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/niteshnankani-svg/adani-stock-predictor)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai)](https://openai.com)

> **An end-to-end AI system that combines document reasoning (annual reports), semantic news search (vector DB), and ML price prediction — all in one interactive dashboard.**

🔗 **Live App**: https://adani-stock-predictor-igjwntvgevqqn7wjq2et4y.streamlit.app

---

## 🧠 What This Project Does

Most stock analysis tools answer either *"What do the financials say?"* or *"What is the price doing?"* — never both together.

This system answers **both, simultaneously**, by connecting three independent data sources through an LLM agent:

| Source | Data | Method |
|--------|------|--------|
| 📄 Annual Reports (FY22/23/24) | Business strategy, risk, financials | Tree-Index Reasoning (custom PageIndex) |
| 📰 2,499 News Articles | Sentiment, market events | ChromaDB Vector Search |
| 📈 740 Days of NSE Stock Data | OHLCV, volume, technicals | XGBoost ML Classifier |

**Ask it anything:**
- *"Why did Adani stock fall in 2023?"*
- *"What did management say about debt in the FY24 annual report?"*
- *"What is the predicted direction for tomorrow?"*

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────┐
│                    USER QUERY                           │
└──────────────────────┬──────────────────────────────────┘
│
▼
┌────────────────────────┐
│   OpenAI LLM Agent     │  ← Decides which sources to query
│   (GPT-4o-mini)        │
└──────┬──────┬──────┬───┘
│      │      │
┌────────┘  ┌───┘  └───────────┐
▼           ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌───────────────┐
│  Tree Index  │ │  ChromaDB    │ │  XGBoost ML   │
│  Reasoning   │ │  Vector DB   │ │  Classifier   │
│              │ │              │ │               │
│ Annual Rpts  │ │ 2,499 News   │ │ 740 Days OHLC │
│ FY22+23+24   │ │ Articles     │ │ + 27 Features │
└──────┬───────┘ └──────┬───────┘ └───────┬───────┘
│                │                 │
└────────────────┴─────────────────┘
│
▼
┌─────────────────────────┐
│   Unified AI Response   │
│   + 7-Day Forecast      │
│   + Support/Resistance  │
└─────────────────────────┘

---

## ⚙️ Key Technical Components

### 1. Tree-Index Reasoning (Custom PageIndex)
Instead of traditional vector RAG (chunk → embed → cosine similarity), this system uses **hierarchical document reasoning**:
Step 1: Build Tree
└── Every 10 pages → GPT-4o-mini writes a semantic description (node)
└── Nodes stored as a navigation map
Step 2: Query
└── LLM reads the tree map → selects relevant node IDs
└── LLM reads actual pages from selected nodes
└── Returns answer with page number citations

**Why this is better than standard RAG for long documents:**
- No information loss from chunking at arbitrary positions
- LLM understands document structure (sections, chapters)
- Returns exact page citations, not just similarity scores
- ## 2. ChromaDB Vector Database (News Semantic Search)
- **2,499 news articles** from GDELT (FY2022–FY2024)
- Each article embedded with `all-MiniLM-L6-v2` (sentence-transformers)
- VADER sentiment scores stored as metadata
- Query by semantic similarity: *"find articles about Adani debt crisis"*

### 3. XGBoost Stock Direction Classifier
- **56.9% accuracy** (baseline: 50% random flip)
- Trained on 740 trading days of NSE data (ADANIENT.NS)
- **27 engineered features** across 4 categories:
- Technical:   RSI, MA5/MA20 crossover, price vs MA, volatility (5d/10d)
Momentum:    3d/5d/10d momentum, consecutive up-day streaks
Returns:     Lag 1/2/3/5/7 day returns, high-low range, close position
Sentiment:   Avg daily sentiment, +ve/-ve article count,
sentiment lag 1/2/3, sentiment MA3/MA5, news volume

### 4. 7-Day Recursive Price Forecast
Each day's prediction feeds into the next day's features — outputs Direction, Confidence %, Target Price, High/Low estimates per day.

### 5. Price Levels (Pivot Points)
Classic pivot formula: Pivot = (High + Low + Close) / 3, with R1/R2/S1/S2 resistance and support levels.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | OpenAI GPT-4o-mini |
| **ML Model** | XGBoost Classifier |
| **Vector DB** | ChromaDB (persistent) |
| **Embeddings** | sentence-transformers all-MiniLM-L6-v2 |
| **Document Parsing** | PyMuPDF (fitz) |
| **Stock Data** | yfinance (Yahoo Finance — NSE) |
| **News Data** | GDELT API (free, no key required) |
| **Sentiment** | VADER |
| **UI / Dashboard** | Streamlit |
| **Deployment** | Streamlit Community Cloud + GitHub |
| **Language** | Python 3.10 |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| XGBoost Accuracy | **56.9%** |
| Random Baseline | 50.0% |
| Training Data | 740 days (FY2022–FY2024) |
| News Articles Indexed | 2,499 |
| Annual Report Pages Indexed | ~300 pages across 3 FY reports |
| Features Used | 27 |
| Prediction Horizon | 1-day + 7-day recursive forecast |

---

## 💡 What Makes This Different From Tutorials

| Most Tutorial Projects | This Project |
|------------------------|-------------|
| Only use price data | 3 heterogeneous data sources fused by LLM agent |
| Single data source | Annual reports + news + stock data combined |
| No document understanding | Custom tree-index reasoning with page citations |
| No deployment | Live on Streamlit Cloud with GitHub CI |
| Toy S&P 500 data | Real NSE stock (ADANIENT.NS) |

---

## 🚀 Run Locally
```bash
git clone https://github.com/niteshnankani-svg/adani-stock-predictor
cd adani-stock-predictor
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```
---

## 👨‍💻 Author

**Nitesh Nankani**
📧 niteshnankani@gmail.com
🔗 https://github.com/niteshnankani-svg/adani-stock-predictor

*Built as part of an intensive self-directed AI engineering study programme covering: LLM agents, RAG pipelines, vector databases, ML deployment, and production Streamlit applications.*

---

⭐ If this project helped you learn something, give it a star!

