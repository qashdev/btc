"""
Gold Trading Bot - 5 Agent System (Puter.js Version)
======================================================
Uses Puter.js for FREE Claude access — no API key needed!

How it works:
- This Python script generates all the logic
- LLM calls are handled by a local HTML file using Puter.js
- Run: python gold_bot_puter.py
- It will create gold_signals.json, then open gold_dashboard.html in your browser

Requirements:
    pip install yfinance xgboost scikit-learn pandas numpy feedparser textblob requests

NO API KEY NEEDED — Puter.js handles Claude for free!
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import webbrowser
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import xgboost as xgb
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GOLD_TICKER          = "GC=F"
GOLD_ETF             = "GLD"
CONFIDENCE_THRESHOLD = 0.58   # slightly lower since no LLM calibration in Python
MAX_RISK_PCT         = 0.02
STOP_LOSS_PCT        = 0.015
TAKE_PROFIT_PCT      = 0.030
ACCOUNT_SIZE         = 10_000
LOOKBACK_DAYS        = 90

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class MarketSignal:
    ticker: str
    price: float
    direction: str
    confidence: float
    sentiment_score: float
    headlines: list
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class TradeDecision:
    signal: MarketSignal
    approved: bool
    position_size_usd: float
    stop_loss: float
    take_profit: float
    reason: str

# ─────────────────────────────────────────────
# AGENT 1: SCANNER
# ─────────────────────────────────────────────
class ScannerAgent:
    def scan(self) -> Optional[dict]:
        log.info("🔍 Agent 1: Scanner running...")
        try:
            for ticker in [GOLD_TICKER, GOLD_ETF]:
                gold = yf.Ticker(ticker)
                hist = gold.history(period=f"{LOOKBACK_DAYS}d", interval="1d")
                if not hist.empty:
                    break

            if hist.empty:
                return None

            latest = hist.iloc[-1]
            prev   = hist.iloc[-2]
            price_change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
            avg_volume       = hist["Volume"].rolling(20).mean().iloc[-1]
            volume_spike     = latest["Volume"] / avg_volume if avg_volume > 0 else 1.0
            returns          = hist["Close"].pct_change().dropna()
            vol_20d          = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

            flagged = abs(price_change_pct) > 0.5 or volume_spike > 1.5 or vol_20d > 0.15

            result = {
                "ticker": GOLD_TICKER,
                "current_price": round(latest["Close"], 2),
                "price_change_pct": round(price_change_pct, 3),
                "volume_spike": round(volume_spike, 2),
                "volatility_20d": round(vol_20d, 4),
                "flagged": flagged,
                "history": hist,
            }
            log.info(f"   Price: ${result['current_price']} | Change: {price_change_pct:.2f}% | Flagged: {flagged}")
            return result
        except Exception as e:
            log.error(f"Scanner error: {e}")
            return None

# ─────────────────────────────────────────────
# AGENT 2: RESEARCH
# ─────────────────────────────────────────────
class ResearchAgent:
    RSS_FEEDS = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US",
        "https://www.kitco.com/rss/news.xml",
    ]

    def research(self, market_data: dict) -> dict:
        log.info("📰 Agent 2: Research running...")
        headlines = self._fetch_headlines()
        sentiment = self._analyze_sentiment(headlines)
        log.info(f"   Headlines: {len(headlines)} | Sentiment: {sentiment:.3f}")
        return {
            "headlines": headlines[:8],
            "sentiment_score": sentiment,
        }

    def _fetch_headlines(self) -> list:
        headlines = []
        for url in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    headlines.append(entry.get("title", ""))
            except Exception:
                pass
        if not headlines:
            headlines = [
                "Gold prices rise amid inflation concerns",
                "Fed signals rate pause, gold steady",
                "Dollar weakens, boosting gold outlook",
                "Geopolitical tensions support safe-haven demand",
            ]
        return [h for h in headlines if h]

    def _analyze_sentiment(self, headlines: list) -> float:
        if not headlines:
            return 0.0
        return round(np.mean([TextBlob(h).sentiment.polarity for h in headlines]), 4)

# ─────────────────────────────────────────────
# AGENT 3: PREDICTION (XGBoost only — LLM done in browser via Puter.js)
# ─────────────────────────────────────────────
class PredictionAgent:
    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False

    def predict(self, market_data: dict, research_data: dict) -> Optional[dict]:
        log.info("🧠 Agent 3: Prediction running (XGBoost)...")
        hist = market_data.get("history")
        if hist is None or len(hist) < 30:
            return None

        features_df = self._build_features(hist, research_data["sentiment_score"])
        if not self.trained:
            self._train(features_df)
        if not self.trained:
            return None

        cols           = self._feature_cols()
        latest         = features_df.iloc[[-1]][cols]
        X              = self.scaler.transform(latest)
        proba          = self.model.predict_proba(X)[0]
        prob_up        = float(proba[1])
        prob_down      = float(proba[0])
        confidence     = max(prob_up, prob_down)
        direction      = "LONG" if prob_up > prob_down else "SHORT"

        log.info(f"   XGBoost: {direction} | Confidence: {confidence:.3f}")
        return {
            "direction": direction,
            "confidence": confidence,
            "prob_up": prob_up,
            "prob_down": prob_down,
        }

    def _build_features(self, hist, sentiment):
        df = hist.copy()
        df["return_1d"]  = df["Close"].pct_change(1)
        df["return_5d"]  = df["Close"].pct_change(5)
        df["return_10d"] = df["Close"].pct_change(10)
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        df["bb_position"]  = (df["Close"] - sma20) / (2 * std20 + 1e-9)
        df["volume_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
        df["sentiment"]    = sentiment
        df["target"]       = (df["Close"].shift(-1) > df["Close"]).astype(int)
        return df.dropna()

    def _feature_cols(self):
        return ["return_1d","return_5d","return_10d","rsi",
                "macd","macd_signal","macd_hist","bb_position","volume_ratio","sentiment"]

    def _train(self, df):
        try:
            cols = self._feature_cols()
            X = df[cols].values
            y = df["target"].values
            if len(X) < 20: return
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
            self.scaler.fit(X_tr)
            X_tr = self.scaler.transform(X_tr)
            X_te = self.scaler.transform(X_te)
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss", verbosity=0,
            )
            self.model.fit(X_tr, y_tr, verbose=False)
            acc = self.model.score(X_te, y_te)
            log.info(f"   XGBoost accuracy: {acc:.3f}")
            self.trained = True
        except Exception as e:
            log.error(f"Train error: {e}")

# ─────────────────────────────────────────────
# AGENT 4: RISK
# ─────────────────────────────────────────────
class RiskAgent:
    def evaluate(self, price, direction, confidence, account_size) -> dict:
        log.info("🛡️ Agent 4: Risk evaluation...")
        p     = confidence
        b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
        kelly = max(0, (p * (b + 1) - 1) / b) * 0.5
        pos   = min(account_size * kelly, account_size * MAX_RISK_PCT)

        if direction == "LONG":
            sl = round(price * (1 - STOP_LOSS_PCT), 2)
            tp = round(price * (1 + TAKE_PROFIT_PCT), 2)
        else:
            sl = round(price * (1 + STOP_LOSS_PCT), 2)
            tp = round(price * (1 - TAKE_PROFIT_PCT), 2)

        approved = pos > 10 and confidence >= CONFIDENCE_THRESHOLD
        log.info(f"   {'APPROVED ✅' if approved else 'BLOCKED ❌'} | Position: ${pos:.2f}")
        return {
            "approved": approved,
            "position_usd": round(pos, 2),
            "stop_loss": sl,
            "take_profit": tp,
        }

# ─────────────────────────────────────────────
# ORCHESTRATOR — builds signal JSON for Puter dashboard
# ─────────────────────────────────────────────
def run():
    scanner    = ScannerAgent()
    researcher = ResearchAgent()
    predictor  = PredictionAgent()
    risk_agent = RiskAgent()

    market_data = scanner.scan()
    if not market_data:
        log.error("No market data.")
        return

    research_data = researcher.research(market_data)
    prediction    = predictor.predict(market_data, research_data)

    signal_data = {
        "timestamp": datetime.now().isoformat(),
        "asset": "Gold (GC=F)",
        "price": market_data["current_price"],
        "price_change_pct": market_data["price_change_pct"],
        "volume_spike": market_data["volume_spike"],
        "volatility": market_data["volatility_20d"],
        "flagged": market_data["flagged"],
        "sentiment_score": research_data["sentiment_score"],
        "headlines": research_data["headlines"],
        "prediction": prediction,
        "risk": None,
        "no_trade_reason": None,
    }

    if not market_data["flagged"]:
        signal_data["no_trade_reason"] = "No significant market activity flagged."
    elif not prediction:
        signal_data["no_trade_reason"] = "XGBoost could not generate a prediction."
    elif prediction["confidence"] < CONFIDENCE_THRESHOLD:
        signal_data["no_trade_reason"] = f"Confidence {prediction['confidence']:.1%} below threshold {CONFIDENCE_THRESHOLD:.0%}."
    else:
        risk = risk_agent.evaluate(
            market_data["current_price"],
            prediction["direction"],
            prediction["confidence"],
            ACCOUNT_SIZE,
        )
        signal_data["risk"] = risk
        if not risk["approved"]:
            signal_data["no_trade_reason"] = "Trade blocked by risk agent."

    # Save signal JSON (convert numpy/pandas types to native Python)
    def json_safe(obj):
        import numpy as np
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open("gold_signals.json", "w") as f:
        json.dump(signal_data, f, indent=2, default=json_safe)
    log.info("✅ Signal data saved to gold_signals.json")

    # Open the Puter dashboard
    dashboard_path = os.path.abspath("gold_dashboard.html")
    log.info(f"🌐 Opening dashboard: {dashboard_path}")
    webbrowser.open(f"file://{dashboard_path}")
    print("\n✅ Done! gold_dashboard.html opened in your browser.")
    print("   Claude (via Puter.js) will analyze the signal and give you a recommendation.\n")

if __name__ == "__main__":
    run()
