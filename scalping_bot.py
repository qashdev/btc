"""
Professional Scalping Bot — Gold & Bitcoin
============================================
Strategy combines 5 best signals from 2025/2026 research:
  1. EMA 9/21 crossover (momentum)
  2. VWAP bounce (institutional mean reversion)
  3. RSI on 5-min chart (overbought/oversold)
  4. Bollinger Band squeeze breakout (volatility)
  5. Volume spike confirmation (order flow)
  6. London/NY session filter (peak liquidity only)

Timeframe: 5-minute candles
Targets:   0.15% TP | 0.08% SL → 1.87:1 R:R
Runs every 5 minutes, saves signals to scalp_signals.json
Opens scalp_dashboard.html in browser for Claude AI analysis

Run: py scalping_bot.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import webbrowser
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import feedparser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ASSETS = {
    "GOLD": {"ticker": "GC=F",    "fallback": "GLD",  "pip": 0.10,  "name": "Gold"},
    "BTC":  {"ticker": "BTC-USD", "fallback": None,   "pip": 1.00,  "name": "Bitcoin"},
}

TIMEFRAME        = "5m"
LOOKBACK         = "2d"        # fetch last 2 days of 5-min data
EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20          # rolling VWAP window
VOLUME_SPIKE_MIN = 1.5         # volume must be 1.5x average

# Trade params
TAKE_PROFIT_PCT  = 0.0015      # 0.15%
STOP_LOSS_PCT    = 0.0008      # 0.08%
MIN_SCORE        = 3           # need at least 3/5 signals to fire

# Session filter (UTC hours) — London open to NY close
SESSION_START_UTC = 7    # 7 AM UTC = London open
SESSION_END_UTC   = 21   # 9 PM UTC = NY close

# ─────────────────────────────────────────────
# JSON SERIALIZER
# ─────────────────────────────────────────────
def json_safe(obj):
    if isinstance(obj, (np.bool_,)):      return bool(obj)
    if isinstance(obj, np.integer):       return int(obj)
    if isinstance(obj, np.floating):      return float(obj)
    if isinstance(obj, np.ndarray):       return obj.tolist()
    if isinstance(obj, pd.Series):        return obj.tolist()
    if isinstance(obj, pd.Timestamp):     return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")

# ─────────────────────────────────────────────
# SESSION FILTER
# ─────────────────────────────────────────────
def is_trading_session() -> bool:
    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour
    weekday = now_utc.weekday()  # 0=Mon, 6=Sun
    if weekday >= 5:  # weekend
        return False
    return SESSION_START_UTC <= hour < SESSION_END_UTC

# ─────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────
def fetch_ohlcv(ticker: str, fallback: str = None) -> pd.DataFrame:
    for t in ([ticker] + ([fallback] if fallback else [])):
        try:
            df = yf.download(t, period=LOOKBACK, interval=TIMEFRAME,
                             progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                return df.dropna()
        except Exception:
            pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # EMAs
    d["ema_fast"] = d["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    d["ema_slow"] = d["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    d["ema_cross"] = np.sign(d["ema_fast"] - d["ema_slow"])
    d["ema_cross_prev"] = d["ema_cross"].shift(1)

    # RSI
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Bollinger Bands
    sma   = d["Close"].rolling(BB_PERIOD).mean()
    std   = d["Close"].rolling(BB_PERIOD).std()
    d["bb_upper"] = sma + BB_STD * std
    d["bb_lower"] = sma - BB_STD * std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (sma + 1e-9)
    d["bb_pct"]   = (d["Close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    # VWAP (rolling)
    tp         = (d["High"] + d["Low"] + d["Close"]) / 3
    vol        = d["Volume"].replace(0, 1)
    d["vwap"]  = (tp * vol).rolling(VWAP_PERIOD).sum() / vol.rolling(VWAP_PERIOD).sum()
    d["vwap_dist"] = (d["Close"] - d["vwap"]) / d["vwap"]

    # Volume
    d["vol_avg"]   = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] / (d["vol_avg"] + 1e-9)

    # ATR
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"]  - d["Close"].shift()).abs()
    d["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    d["atr_pct"] = d["atr"] / d["Close"]

    # Momentum
    d["mom3"]  = d["Close"].pct_change(3)
    d["mom10"] = d["Close"].pct_change(10)

    return d.dropna()

# ─────────────────────────────────────────────
# SIGNAL SCORER — returns LONG/SHORT/NONE + score
# ─────────────────────────────────────────────
def score_signals(df: pd.DataFrame) -> dict:
    row      = df.iloc[-1]
    prev     = df.iloc[-2]
    signals  = {}
    score_l  = 0
    score_s  = 0

    # ── Signal 1: EMA Crossover ──
    cross_now  = float(row["ema_cross"])
    cross_prev = float(prev["ema_cross"])
    if cross_prev <= 0 and cross_now > 0:
        signals["ema_cross"] = "LONG ✅"
        score_l += 1
    elif cross_prev >= 0 and cross_now < 0:
        signals["ema_cross"] = "SHORT ✅"
        score_s += 1
    elif cross_now > 0:
        signals["ema_cross"] = "LONG (trend) ✅"
        score_l += 0.5
    elif cross_now < 0:
        signals["ema_cross"] = "SHORT (trend) ✅"
        score_s += 0.5
    else:
        signals["ema_cross"] = "NEUTRAL ❌"

    # ── Signal 2: RSI ──
    rsi = float(row["rsi"])
    if rsi < 35:
        signals["rsi"] = f"OVERSOLD {rsi:.1f} → LONG ✅"
        score_l += 1
    elif rsi > 65:
        signals["rsi"] = f"OVERBOUGHT {rsi:.1f} → SHORT ✅"
        score_s += 1
    elif 40 <= rsi <= 60:
        signals["rsi"] = f"NEUTRAL {rsi:.1f} ❌"
    else:
        signals["rsi"] = f"MILD {rsi:.1f} ❌"

    # ── Signal 3: VWAP Bounce ──
    vwap_dist = float(row["vwap_dist"])
    if vwap_dist < -0.001:   # price below VWAP → bounce long
        signals["vwap"] = f"BELOW VWAP {vwap_dist:.3%} → LONG ✅"
        score_l += 1
    elif vwap_dist > 0.001:  # price above VWAP → short reversion
        signals["vwap"] = f"ABOVE VWAP {vwap_dist:.3%} → SHORT ✅"
        score_s += 1
    else:
        signals["vwap"] = f"AT VWAP {vwap_dist:.3%} ❌"

    # ── Signal 4: Bollinger Band ──
    bb_pct   = float(row["bb_pct"])
    bb_width = float(row["bb_width"])
    squeeze  = bb_width < float(df["bb_width"].rolling(20).mean().iloc[-1]) * 0.8
    if bb_pct < 0.2:
        signals["bollinger"] = f"NEAR LOWER BAND {bb_pct:.2f} → LONG ✅"
        score_l += 1
    elif bb_pct > 0.8:
        signals["bollinger"] = f"NEAR UPPER BAND {bb_pct:.2f} → SHORT ✅"
        score_s += 1
    elif squeeze:
        signals["bollinger"] = f"SQUEEZE {bb_width:.4f} → BREAKOUT SOON ⚡"
    else:
        signals["bollinger"] = f"MID BAND {bb_pct:.2f} ❌"

    # ── Signal 5: Volume Spike ──
    vol_spike = float(row["vol_spike"])
    mom3      = float(row["mom3"])
    if vol_spike >= VOLUME_SPIKE_MIN:
        if mom3 > 0:``
            signals["volume"] = f"SPIKE {vol_spike:.1f}x + UP MOM → LONG ✅"
            score_l += 1
        else:
            signals["volume"] = f"SPIKE {vol_spike:.1f}x + DOWN MOM → SHORT ✅"
            score_s += 1
    else:
        signals["volume"] = f"NORMAL {vol_spike:.1f}x ❌"

    # ── Final Decision ──
    if score_l >= MIN_SCORE and score_l > score_s:
        direction  = "LONG"
        confidence = min(score_l / 5.0, 1.0)
    elif score_s >= MIN_SCORE and score_s > score_l:
        direction  = "SHORT"
        confidence = min(score_s / 5.0, 1.0)
    else:
        direction  = "NONE"
        confidence = 0.0

    return {
        "direction":  direction,
        "confidence": round(confidence, 3),
        "score_long": score_l,
        "score_short": score_s,
        "signals":    signals,
        "rsi":        round(rsi, 2),
        "vwap_dist":  round(vwap_dist * 100, 4),
        "vol_spike":  round(vol_spike, 2),
        "bb_pct":     round(bb_pct, 3),
        "atr_pct":    round(float(row["atr_pct"]) * 100, 4),
    }

# ─────────────────────────────────────────────
# NEWS SENTIMENT
# ─────────────────────────────────────────────
def get_sentiment(asset_name: str) -> dict:
    feeds = {
        "Gold":    ["https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US"],
        "Bitcoin": ["https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US"],
    }
    headlines = []
    for url in feeds.get(asset_name, []):
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:5]:
                headlines.append(e.get("title", ""))
        except Exception:
            pass
    if not headlines:
        return {"score": 0.0, "headlines": []}
    scores = [TextBlob(h).sentiment.polarity for h in headlines if h]
    return {
        "score":     round(float(np.mean(scores)), 4) if scores else 0.0,
        "headlines": headlines[:5],
    }

# ─────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────
def scan_asset(asset_key: str) -> dict:
    cfg  = ASSETS[asset_key]
    log.info(f"📊 Scanning {cfg['name']} ({cfg['ticker']})...")

    df = fetch_ohlcv(cfg["ticker"], cfg.get("fallback"))
    if df.empty or len(df) < 50:
        return {"asset": cfg["name"], "error": "Insufficient data"}

    df   = compute_indicators(df)
    last = df.iloc[-1]

    price      = float(last["Close"])
    price_prev = float(df.iloc[-2]["Close"])
    change_pct = (price - price_prev) / price_prev * 100

    scoring    = score_signals(df)
    sentiment  = get_sentiment(cfg["name"])
    session_ok = is_trading_session()

    # Trade levels
    if scoring["direction"] == "LONG":
        tp = round(price * (1 + TAKE_PROFIT_PCT), 4)
        sl = round(price * (1 - STOP_LOSS_PCT), 4)
    elif scoring["direction"] == "SHORT":
        tp = round(price * (1 - TAKE_PROFIT_PCT), 4)
        sl = round(price * (1 + STOP_LOSS_PCT), 4)
    else:
        tp = sl = None

    result = {
        "asset":        cfg["name"],
        "ticker":       cfg["ticker"],
        "timestamp":    datetime.now().isoformat(),
        "price":        round(price, 4),
        "change_pct":   round(change_pct, 4),
        "session_ok":   bool(session_ok),
        "direction":    scoring["direction"],
        "confidence":   scoring["confidence"],
        "score_long":   scoring["score_long"],
        "score_short":  scoring["score_short"],
        "signals":      scoring["signals"],
        "rsi":          scoring["rsi"],
        "vwap_dist_pct":scoring["vwap_dist"],
        "vol_spike":    scoring["vol_spike"],
        "bb_pct":       scoring["bb_pct"],
        "atr_pct":      scoring["atr_pct"],
        "take_profit":  tp,
        "stop_loss":    sl,
        "tp_pct":       TAKE_PROFIT_PCT * 100,
        "sl_pct":       STOP_LOSS_PCT * 100,
        "rr_ratio":     round(TAKE_PROFIT_PCT / STOP_LOSS_PCT, 2),
        "sentiment":    sentiment,
        "trade_valid":  bool(scoring["direction"] != "NONE" and session_ok),
    }

    arrow = "🟢 LONG" if scoring["direction"] == "LONG" else ("🔴 SHORT" if scoring["direction"] == "SHORT" else "⚪ NONE")
    log.info(f"   {cfg['name']}: ${price:,.2f} | {arrow} | Score L:{scoring['score_long']} S:{scoring['score_short']} | Session:{'✅' if session_ok else '❌'}")
    return result

# ─────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────
DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scalping Bot Dashboard</title>
<script src="https://js.puter.com/v2/"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a14; color: #e0e0e0; font-family: 'Segoe UI', monospace; }
  .header { background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 20px 30px; border-bottom: 2px solid #FFD700; }
  .header h1 { color: #FFD700; font-size: 1.6em; }
  .header p  { color: #888; font-size: 0.85em; margin-top: 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
  .card { background: #12122a; border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 1em; color: #aaa; margin-bottom: 14px; letter-spacing: 1px; text-transform: uppercase; }
  .price { font-size: 2.2em; font-weight: bold; color: #FFD700; }
  .change.pos { color: #00e676; } .change.neg { color: #ff1744; }
  .direction { font-size: 1.6em; font-weight: bold; padding: 10px 20px;
    border-radius: 8px; display: inline-block; margin: 10px 0; }
  .LONG  { background: #00c85322; color: #00e676; border: 1px solid #00e676; }
  .SHORT { background: #ff174422; color: #ff1744; border: 1px solid #ff1744; }
  .NONE  { background: #ffffff11; color: #888; border: 1px solid #444; }
  .signal-row { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #1a1a3a; font-size: 0.85em; }
  .signal-key { color: #aaa; } .signal-val { color: #e0e0e0; text-align: right; max-width: 60%; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold; }
  .badge.green { background: #00e67622; color: #00e676; border: 1px solid #00e676; }
  .badge.red   { background: #ff174422; color: #ff1744; border: 1px solid #ff1744; }
  .badge.gold  { background: #FFD70022; color: #FFD700; border: 1px solid #FFD700; }
  .levels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }
  .level-box { background: #1a1a2e; border-radius: 8px; padding: 10px; text-align: center; }
  .level-box .label { font-size: 0.7em; color: #888; text-transform: uppercase; }
  .level-box .val { font-size: 1em; font-weight: bold; margin-top: 4px; }
  .tp  { color: #00e676; } .sl { color: #ff1744; } .rr { color: #FFD700; }
  .ai-box { background: #0d1117; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 16px; min-height: 120px; font-size: 0.9em; line-height: 1.6; color: #c9d1d9; }
  .ai-thinking { color: #FFD700; font-style: italic; }
  .btn { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;
    border: none; padding: 12px 28px; border-radius: 8px; font-weight: bold;
    cursor: pointer; font-size: 1em; margin-top: 14px; width: 100%; }
  .btn:hover { opacity: 0.9; } .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .full { grid-column: 1 / -1; }
  .session { padding: 4px 12px; border-radius: 12px; font-size: 0.8em; display: inline-block; margin-top: 8px; }
  .session.ok  { background: #00e67622; color: #00e676; border: 1px solid #00e676; }
  .session.off { background: #ff174422; color: #ff1744; border: 1px solid #ff1744; }
  .score-bar { height: 8px; border-radius: 4px; margin: 6px 0; }
  .score-bar-long  { background: linear-gradient(90deg, #00e676, #00c853); }
  .score-bar-short { background: linear-gradient(90deg, #ff1744, #c62828); }
  .tab-bar { display: flex; gap: 10px; padding: 0 20px; border-bottom: 1px solid #2a2a4a; }
  .tab { padding: 12px 20px; cursor: pointer; color: #888; border-bottom: 3px solid transparent; }
  .tab.active { color: #FFD700; border-bottom-color: #FFD700; }
  .tab-content { display: none; } .tab-content.active { display: block; }
  .headline { padding: 6px 0; border-bottom: 1px solid #1a1a3a; font-size: 0.82em; color: #aaa; }
</style>
</head>
<body>
<div class="header">
  <h1>⚡ Professional Scalping Bot</h1>
  <p>5-Signal System • EMA Crossover • VWAP • RSI • Bollinger Bands • Volume Spike • Via Puter.js AI</p>
</div>

<div class="tab-bar">
  <div class="tab active" onclick="showTab('gold')">🥇 Gold</div>
  <div class="tab" onclick="showTab('btc')">₿ Bitcoin</div>
</div>

<div id="tab-gold" class="tab-content active">
  <div class="grid" id="gold-grid">
    <div class="card" style="text-align:center; grid-column:1/-1">
      <p style="color:#888">Loading Gold data...</p>
    </div>
  </div>
</div>

<div id="tab-btc" class="tab-content">
  <div class="grid" id="btc-grid">
    <div class="card" style="text-align:center; grid-column:1/-1">
      <p style="color:#888">Loading BTC data...</p>
    </div>
  </div>
</div>

<script>
function showTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['gold','btc'][i]===name));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
}

function fmt(v, d=2) { return v == null ? 'N/A' : Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}); }

function renderAsset(data, gridId) {
  const grid = document.getElementById(gridId);
  if (!data || data.error) {
    grid.innerHTML = `<div class="card full"><p style="color:#ff1744">Error: ${data?.error||'No data'}</p></div>`;
    return;
  }

  const dirClass = data.direction;
  const dirIcon  = data.direction==='LONG' ? '🟢 LONG — BUY' : data.direction==='SHORT' ? '🔴 SHORT — SELL' : '⚪ NO SIGNAL';
  const chgClass = data.change_pct >= 0 ? 'pos' : 'neg';
  const chgSign  = data.change_pct >= 0 ? '+' : '';
  const sessionHtml = data.session_ok
    ? '<span class="session ok">✅ London/NY Session ACTIVE</span>'
    : '<span class="session off">❌ Outside Trading Hours</span>';

  const signalRows = Object.entries(data.signals).map(([k,v]) =>
    `<div class="signal-row"><span class="signal-key">${k.toUpperCase()}</span><span class="signal-val">${v}</span></div>`
  ).join('');

  const sentimentRows = (data.sentiment?.headlines||[]).map(h =>
    `<div class="headline">• ${h}</div>`
  ).join('');

  const scoreL = Math.min((data.score_long / 5) * 100, 100);
  const scoreS = Math.min((data.score_short / 5) * 100, 100);

  grid.innerHTML = `
    <div class="card">
      <h2>📊 ${data.asset} (${data.ticker})</h2>
      <div class="price">$${fmt(data.price)}</div>
      <div class="change ${chgClass}">${chgSign}${fmt(data.change_pct,3)}% (5min)</div>
      <div class="direction ${dirClass}">${dirIcon}</div>
      ${sessionHtml}
      <div style="margin-top:12px">
        <div style="display:flex;justify-content:space-between;font-size:0.8em;color:#888">
          <span>LONG score: ${data.score_long}/5</span><span>SHORT score: ${data.score_short}/5</span>
        </div>
        <div class="score-bar score-bar-long"  style="width:${scoreL}%"></div>
        <div class="score-bar score-bar-short" style="width:${scoreS}%"></div>
      </div>
    </div>

    <div class="card">
      <h2>🎯 Trade Levels</h2>
      <div class="levels">
        <div class="level-box"><div class="label">Entry</div><div class="val">$${fmt(data.price)}</div></div>
        <div class="level-box"><div class="label">Take Profit</div><div class="val tp">$${fmt(data.take_profit)||'—'}</div></div>
        <div class="level-box"><div class="label">Stop Loss</div><div class="val sl">$${fmt(data.stop_loss)||'—'}</div></div>
      </div>
      <div class="levels" style="margin-top:10px">
        <div class="level-box"><div class="label">TP %</div><div class="val tp">+${data.tp_pct}%</div></div>
        <div class="level-box"><div class="label">SL %</div><div class="val sl">-${data.sl_pct}%</div></div>
        <div class="level-box"><div class="label">R:R</div><div class="val rr">1:${data.rr_ratio}</div></div>
      </div>
      <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.85em">
        <div class="level-box"><div class="label">RSI</div><div class="val">${data.rsi}</div></div>
        <div class="level-box"><div class="label">Vol Spike</div><div class="val">${data.vol_spike}x</div></div>
        <div class="level-box"><div class="label">VWAP Dist</div><div class="val">${data.vwap_dist_pct}%</div></div>
        <div class="level-box"><div class="label">ATR %</div><div class="val">${data.atr_pct}%</div></div>
      </div>
    </div>

    <div class="card">
      <h2>📡 5-Signal Breakdown</h2>
      ${signalRows}
    </div>

    <div class="card">
      <h2>📰 News Sentiment: <span style="color:${data.sentiment.score>0?'#00e676':data.sentiment.score<0?'#ff1744':'#888'}">${data.sentiment.score > 0 ? 'BULLISH' : data.sentiment.score < 0 ? 'BEARISH' : 'NEUTRAL'} (${data.sentiment.score})</span></h2>
      ${sentimentRows || '<p style="color:#888;font-size:0.85em">No headlines available</p>'}
    </div>

    <div class="card full">
      <h2>🤖 Claude AI Scalp Analysis <span class="badge gold">VIA PUTER.JS — FREE</span></h2>
      <div class="ai-box" id="ai-box-${gridId}">
        <span class="ai-thinking">Click "Run AI Analysis" to get Claude's scalping recommendation...</span>
      </div>
      <button class="btn" id="btn-${gridId}" onclick="runAI('${gridId}', ${JSON.stringify(data).replace(/'/g,"&#39;")})">
        ⚡ Run AI Analysis
      </button>
    </div>
  `;
}

async function runAI(gridId, data) {
  const box = document.getElementById('ai-box-' + gridId);
  const btn = document.getElementById('btn-' + gridId);
  btn.disabled = true;
  box.innerHTML = '<span class="ai-thinking">🤖 Claude is analyzing the scalp signal...</span>';

  const prompt = `You are a professional scalping trader. Analyze this real-time signal and give a clear recommendation.

Asset: ${data.asset} (${data.ticker})
Price: $${data.price} | 5min Change: ${data.change_pct}%
Direction: ${data.direction} | Confidence: ${(data.confidence*100).toFixed(0)}%
Score: LONG ${data.score_long}/5 | SHORT ${data.score_short}/5
Trading Session: ${data.session_ok ? 'ACTIVE (London/NY)' : 'OUTSIDE HOURS'}

5-Signal Breakdown:
${Object.entries(data.signals).map(([k,v]) => `- ${k}: ${v}`).join('\\n')}

Key Metrics:
- RSI: ${data.rsi}
- VWAP Distance: ${data.vwap_dist_pct}%
- Volume Spike: ${data.vol_spike}x average
- ATR: ${data.atr_pct}% of price
- Bollinger Band Position: ${data.bb_pct}

Trade Levels:
- Entry: $${data.price}
- Take Profit: $${data.take_profit} (+${data.tp_pct}%)
- Stop Loss: $${data.stop_loss} (-${data.sl_pct}%)
- R:R: 1:${data.rr_ratio}

News Sentiment: ${data.sentiment.score} (${data.sentiment.headlines.slice(0,3).join(' | ')})

Give your analysis in this format:
1. TRADE DECISION: ENTER / SKIP / WAIT
2. REASON: (2-3 sentences)
3. KEY RISK: (1 sentence)
4. ENTRY TIMING: (immediate / wait for X)`;

  try {
    const response = await puter.ai.chat(prompt);
    const text = typeof response === 'string' ? response : response?.message?.content?.[0]?.text || JSON.stringify(response);
    box.innerHTML = text.replace(/\\n/g,'<br>').replace(/\*\*(.*?)\*\*/g,'<strong style="color:#FFD700">$1</strong>');
  } catch(e) {
    box.innerHTML = `<span style="color:#ff1744">Error: ${e.message}. Make sure you are logged into Puter.js</span>`;
  }
  btn.disabled = false;
}

// Load data from JSON files
async function loadData() {
  const files = [
    { file: 'gold_scalp_signals.json', grid: 'gold-grid' },
    { file: 'btc_scalp_signals.json',  grid: 'btc-grid'  },
  ];
  for (const {file, grid} of files) {
    try {
      const r = await fetch(file + '?t=' + Date.now());
      if (r.ok) {
        const data = await r.json();
        renderAsset(data, grid);
      }
    } catch(e) {
      console.log('Could not load', file, e);
    }
  }
}

loadData();
setInterval(loadData, 60000); // auto-refresh every 60 seconds
</script>
</body>
</html>'''

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    log.info("⚡ Scalping Bot starting...")
    log.info(f"   Session: {'✅ ACTIVE' if is_trading_session() else '❌ Outside hours (still scanning)'}")

    # Write dashboard HTML
    with open("scalp_dashboard.html", "w", encoding="utf-8") as f:
        f.write(DASHBOARD_HTML)

    # Scan both assets
    for asset_key in ASSETS:
        result = scan_asset(asset_key)
        fname  = f"{asset_key.lower()}_scalp_signals.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2, default=json_safe)
        log.info(f"   ✅ Saved → {fname}")

    # Open dashboard
    path = os.path.abspath("scalp_dashboard.html")
    webbrowser.open(f"file://{path}")
    log.info("🌐 Dashboard opened in browser")
    print("\n✅ Done! scalp_dashboard.html opened.")
    print("   Click 'Run AI Analysis' on Gold or BTC for Claude's recommendation.\n")

if __name__ == "__main__":
    run()
